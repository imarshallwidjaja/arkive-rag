from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import stat
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anyio
import httpx2
import numpy as np
import pytest
from mcp import Client
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError
from typer.testing import CliRunner

from arkive_rag import compiler, induction
from arkive_rag.llm import BudgetExceededError, Tool, ToolReply, ToolResult, UsageMeter
from arkive_rag.models import (
    DocumentInfo,
    ProviderError,
    UnsafePathError,
    corpus_fingerprint,
    fold_for_quote,
)
from arkive_rag.server import INSTRUCTIONS, TopicCatalog, create_mcp_server
from arkive_rag.snapshot import TopicSnapshot
from evals import drafting
from evals import run as eval_run
from evals.ablation import first_satisfying_rank, run_layer1, summarize_ranks
from evals.agents import (
    QA_INSTRUCTIONS,
    AgentRun,
    CountingEmbedder,
    JudgedClaim,
    QAJudgeScore,
    RawCorpus,
    arkive_mcp_tools,
    qa_evidence,
    run_agent,
    run_layer2_qa,
    with_server_instructions,
)
from evals.drafting import (
    EXEMPLAR_DRAFTING_RULES,
    CandidateAssessment,
    DraftBlindComparison,
    LexicalOverlap,
    OverlapSources,
    assert_holdout_clean,
    blind_assignment,
    check_reference_held_out,
    compile_holdout_snapshot,
    exemplars_prompt,
    rubric_diagnostics,
    run_layer3_holdout,
    score_rubric,
    select_holdout_sources,
    without_visual_transcripts,
)
from evals.opencode_check.scenarios import ARKIVE_TOOLS
from evals.run import _layer2_report, _layer3_report, make_private_directory, write_private_text
from evals.run import app as eval_app
from evals.schema import (
    EvalPack,
    Holdout,
    QACase,
    RetrievalCase,
    RubricCheck,
    normalize_for_match,
    satisfies_groups,
)
from tests.support import (
    BATHYMETRY_CORPUS,
    FakeGateway,
    ScriptedOpenAI,
    compile_snapshot,
    function_call,
    message,
    refusal_body,
    response_body,
    structured_body,
    usage,
    write_pdf,
)

EXAMPLE_PACK_PATH = Path(__file__).resolve().parents[1] / "evals" / "example_pack.eval.json"
EXAMPLE_HOLDOUT = json.loads(EXAMPLE_PACK_PATH.read_text(encoding="utf-8"))["holdouts"][0]


@pytest.fixture(scope="module")
def example_pack() -> EvalPack:
    return EvalPack.load(EXAMPLE_PACK_PATH)


def holdout_task(pack: EvalPack) -> Holdout:
    (holdout,) = pack.holdouts
    return holdout


@dataclass(frozen=True)
class RankedChunk:
    doc_id: str
    content: str


def chunk(doc_id: str, content: str) -> RankedChunk:
    return RankedChunk(doc_id, content)


def retrieval_case(groups: list[list[str]], docs: tuple[str, ...] = ()) -> RetrievalCase:
    return RetrievalCase.model_validate(
        {"case_id": "c", "category": "visual_only", "query": "q", "required_groups": groups}
        | {"expected_docs": docs}
    )


def test_example_pack_defines_every_layer(example_pack: EvalPack) -> None:
    task = holdout_task(example_pack)
    assert example_pack.retrieval_cases and example_pack.qa_cases and task.rubric
    assert {c.category for c in task.rubric} == {"house_structure", "anti_contamination"}
    assert {c.kind for c in example_pack.qa_cases} == {"lookup", "abstention"}
    assert task.exemplar_paths == ("p101-harbour-lidar/harbour-memo.md",)


def test_example_collection_packs_match_their_collections_and_hold_out_their_references() -> None:
    evals_dir = EXAMPLE_PACK_PATH.parent
    collections = sorted(p.name for p in (evals_dir / "example_corpora").iterdir() if p.is_dir())
    pack_paths = sorted((evals_dir / "example_packs").glob("*.eval.json"))
    assert [p.name.removesuffix(".eval.json") for p in pack_paths] == collections
    for pack_path in pack_paths:
        pack = EvalPack.load(pack_path)
        corpus_dir = evals_dir / "example_corpora" / pack_path.name.removesuffix(".eval.json")
        sources = compiler.discover_corpus(corpus_dir)
        digests = {s.rel_path: hashlib.sha256(compiler.read_source(s)).hexdigest() for s in sources}
        assert pack.corpus_fingerprint == corpus_fingerprint(digests.items()), pack_path.name
        # Document ids follow the build: doc_001 is the first discovered file in path order.
        by_id = {f"doc_{i:03d}": s for i, s in enumerate(sources, start=1)}
        named = {a.doc_id for case in pack.qa_cases for a in case.source_anchors}
        named |= {d for case in pack.retrieval_cases for d in case.expected_docs}
        assert named <= by_id.keys(), pack_path.name
        for holdout in pack.holdouts:
            source = by_id[holdout.reference_doc_id]
            # The guards read doc_id, rel_path, sha256 and is_curator_note; the rest is neutral.
            reference = DocumentInfo(
                doc_id=holdout.reference_doc_id,
                rel_path=source.rel_path,
                sha256=digests[source.rel_path],
                format=source.format,
                title=source.rel_path,
                doc_date=None,
                deliverable_type="unknown",
                summary="",
                page_count=0,
                is_curator_note=source.is_curator_note,
                facets={},
            )
            check_reference_held_out(holdout, reference)
            select_holdout_sources(corpus_dir, holdout, reference)


def pack_with(**overrides: Any) -> dict[str, Any]:
    pack = json.loads(EXAMPLE_PACK_PATH.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        target = pack
        *parents, leaf = dotted.split("__")
        for key in parents:
            target = target[int(key)] if isinstance(target, list) else target[key]
        target[leaf] = value
    return cast(dict[str, Any], pack)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"format": "arkive-eval-pack/1"}, "format"),
        ({"retrieval_cases__3__project": "p102-estuary-multibeam"}, "Extra inputs"),
        ({"retrieval_cases__3__filters": {"project": " "}}, "pattern"),
        ({"corpus_fingerprint": "abc"}, "corpus_fingerprint"),
        ({"surprise": 1}, "Extra inputs"),
        ({"retrieval_cases__1__case_id": "exact_identifiers_1"}, "duplicate retrieval case_id"),
        ({"retrieval_cases__0__required_groups": [[]]}, "at least 1 item"),
        ({"qa_cases__0__required_groups": [["  "]]}, "pattern"),
        ({"holdouts__0__included_paths": ["../outside"]}, "relative to the corpus root"),
        ({"holdouts__0__included_paths": ["/abs"]}, "relative to the corpus root"),
        ({"holdouts__0__included_paths": ["P102-notes"]}, "held-out marker"),
        ({"holdouts__0__included_paths": ["p101-harbour-lidar/"]}, "plain corpus-relative"),
        ({"holdouts__0__included_paths": ["./p101-harbour-lidar"]}, "plain corpus-relative"),
        ({"holdouts__0__included_paths": ["p101//memo.md"]}, "plain corpus-relative"),
        ({"holdouts__0__included_paths": ["."]}, "plain corpus-relative"),
        ({"holdouts__0__rubric__0__require_groups": []}, "no require_groups or forbid_any"),
        ({"holdouts__0__rubric__1__check_id": "scope_and_budget"}, "duplicate rubric check_id"),
        ({"qa_cases__0__kind": "trivia"}, "'lookup', 'multi_document', 'aggregation'"),
        ({"qa_cases__0__source_anchors": []}, "of kind 'lookup' has no source_anchors"),
        ({"qa_cases__0__source_anchors": ["harbour-memo.md (doc_001)"]}, "valid dictionary"),
        ({"qa_cases__0__source_anchors": [{"doc_id": "doc_001"}]}, "pages\n  Field required"),
        ({"qa_cases__0__source_anchors__0__pages": [3, 2]}, r"pages \[3, 2\] run backwards"),
        ({"qa_cases__0__source_anchors__0__pages": [0, 1]}, "greater than 0"),
        ({"qa_cases__0__source_anchors__0__pages": [1, 2, 3]}, "at most 2 items"),
        ({"qa_cases__0__source_anchors__0__pages": [True, 2]}, "valid integer"),
        ({"qa_cases__0__source_anchors__0__pages": ["2", 3]}, "valid integer"),
        ({"qa_cases__0__source_anchors__0__pages": [1, 3.0]}, "valid integer"),
        ({"holdouts": [EXAMPLE_HOLDOUT, EXAMPLE_HOLDOUT]}, "duplicate holdout_id"),
        ({"holdouts__0__holdout_id": "../estuary"}, "pattern"),
        ({"holdouts__0__memo_instructions": "Write it."}, "Extra inputs"),
        (
            {"holdouts__0__exemplar_paths": ["p102-estuary-multibeam/estuary-proposal.md"]},
            "exemplar path 'p102-estuary-multibeam/estuary-proposal.md' is not under any",
        ),
        ({"holdouts__0__exemplar_paths": ["p101-harbour-lidar/"]}, "plain corpus-relative"),
    ],
)
def test_eval_pack_rejects_malformed_packs(overrides: dict[str, Any], message: str) -> None:
    EvalPack.model_validate(pack_with())
    with pytest.raises(ValidationError, match=message):
        EvalPack.model_validate(pack_with(**overrides))


def test_eval_pack_refuses_an_older_format_by_name_with_one_error() -> None:
    older = pack_with(format="arkive-eval-pack/2", holdout_memo={"memo_instructions": "x"})
    with pytest.raises(ValidationError) as refused:
        EvalPack.model_validate(older)
    (error,) = refused.value.errors()
    assert "format 'arkive-eval-pack/2' is not supported" in error["msg"]
    assert "reads 'arkive-eval-pack/3' packs only" in error["msg"]


def test_eval_pack_accepts_an_abstention_case_without_anchors_and_a_holdout_without_rubric() -> (
    None
):
    pack = EvalPack.model_validate(
        pack_with(holdouts__0__rubric=[], holdouts__0__exemplar_paths=[])
    )
    assert holdout_task(pack).rubric == () and holdout_task(pack).exemplar_paths == ()
    (abstention,) = [c for c in pack.qa_cases if c.kind == "abstention"]
    assert abstention.source_anchors == ()


def test_group_matching_folds_case_dashes_whitespace_and_thousands_separators() -> None:
    assert satisfies_groups("Area 59494 KM\u00b2  and post-process  =  False", (("59,494 km2",),))
    assert satisfies_groups("Sentinel\u20112 cloud", (("sentinel-2",), ("rain", "cloud")))
    assert not satisfies_groups("Sentinel-2 only", (("sentinel-2",), ("rain", "cloud")))


@pytest.mark.parametrize(
    "written", ["1800x1800", "1800 x 1800", "1800\u00d71800", "1,800 \u00d7 1,800"]
)
def test_group_matching_folds_dimension_signs_between_digits(written: str) -> None:
    assert normalize_for_match(f"tiles of {written} px") == "tiles of 1800 x 1800 px"
    assert satisfies_groups(f"a 3\u00d73 window, {written} tiles", (("3 x 3",), ("1800 x 1800",)))
    assert normalize_for_match("box x 3") == "box x 3"


def test_first_satisfying_rank_needs_every_group_and_expected_doc_within_the_prefix() -> None:
    case = retrieval_case([["211.4"], ["215.8"]], ("doc_014",))
    ranking = [
        chunk("doc_009", "unrelated"),
        chunk("doc_014", "Central median 211.4 mm"),
        chunk("doc_002", "noise"),
        chunk("doc_014", "Northern median 215.8 mm"),
    ]
    assert first_satisfying_rank(case, ranking) == 4
    wrong_doc = [chunk("doc_009", "211.4 and 215.8")]
    assert first_satisfying_rank(case, wrong_doc) is None
    beyond_top_k = [chunk("doc_001", "x")] * 5 + [chunk("doc_014", "211.4 215.8")]
    assert first_satisfying_rank(case, beyond_top_k) is None


def test_rank_summary_computes_completion_rates_and_mrr() -> None:
    summary = summarize_ranks([1, 2, None, 5])
    assert (summary.completion_at_1, summary.completion_at_3, summary.completion_at_5) == (
        0.25,
        0.5,
        0.75,
    )
    assert summary.completion_mrr_at_5 == pytest.approx((1 + 0.5 + 0 + 0.2) / 4)
    with pytest.raises(ValueError):
        summarize_ranks([])


@dataclass
class RecordingEmbedder:
    calls: list[str] = field(default_factory=list)

    def __call__(self, query: str) -> NDArray[np.float32]:
        self.calls.append(query)
        return FakeGateway().embed_query(query)


def test_layer1_runs_the_example_pack_with_one_embedding_per_case(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    cases = example_pack.retrieval_cases
    embed = RecordingEmbedder()
    result = run_layer1(snapshot_path, embed, cases)
    assert embed.calls == [case.query for case in cases]
    rows = cast(list[Any], result["cases"])
    assert all(r.ranks["bm25_only"] is not None for r in rows), [r.ranks for r in rows]
    assert all(r.ranks["hybrid_rrf"] is not None for r in rows), [r.ranks for r in rows]
    by_category = cast(dict[str, Any], result["by_category"])
    assert set(by_category) == {c.category for c in cases}


FAITHFUL_DRAFT = """\
# Estuary Multibeam Bathymetry Proposal
## Scope
A multibeam survey of 812 km2 delivered in 2000 x 2000 pixel tiles. Unlike the harbour
project's 1 m lidar grid, the multibeam surface is not resampled.
| File Name | Masking |
| estuary_depth_grid.tif | No masking applied. |
| estuary_depth_uncertainty.tif | No masking applied. |
## Budget
Total fee 120,000 AUD.
## Limitations
Ground truth is needed over moorings and dredged channels.
"""


def test_rubric_rewards_faithful_drafts_and_catches_targeted_contamination(
    example_pack: EvalPack,
) -> None:
    rubric = holdout_task(example_pack).rubric
    assert all(score_rubric(FAITHFUL_DRAFT, rubric).values())
    borrowed_tiles = FAITHFUL_DRAFT.replace("2000 x 2000 pixel", "500 x 500 pixel")
    borrowed_mask = FAITHFUL_DRAFT + "\nShallows were masked using the harbour coastline land mask."
    assert score_rubric(borrowed_tiles, rubric) == {
        **score_rubric(FAITHFUL_DRAFT, rubric),
        "tile_size_not_contaminated": False,
    }
    assert not score_rubric(borrowed_mask, rubric)["no_borrowed_masking_claim"]
    dropped = FAITHFUL_DRAFT.replace("estuary_depth_uncertainty.tif", "")
    assert not score_rubric(dropped, rubric)["deliverables_and_area_kept"]
    assert not any(score_rubric("", rubric).values())


# A reference paragraph that a holdout section also holds: house text.
HOUSE = (
    "Northwind Marine reduces every sounding to the stated vertical datum before it grids"
    " the surface."
)
# Held by the holdout too, but restated in the run notes: authorised input.
AUTHORISED = "The client requires every deliverable as a GeoTIFF file on the projected grid."
# Held by the holdout too, but under STANDING_MIN_WORDS words.
SHORT = "Crews log each tide gauge reading."
REFERENCE_ONLY = "The estuary campaign runs over six weeks of calm weather windows in early autumn."
# Plain text on one side, a visual transcript on the other.
CHART = "The sounding density chart compares the inner and outer approaches by month."
MAP = "The survey boundary map marks every mooring buoy inside the estuary channel."
HOLDOUT_ONLY = "The harbour lidar grid was masked with the coastline land layer before export."
HOLDOUT_TABLE = (
    "| Layer | Treatment |\n| --- | --- |\n| Coastline land layer | Masked before every export |"
)


def visual(label: str, text: str) -> str:
    return f"> **[Visual \u2014 Page 1: {label}]**\n> {text}"


def overlap_sources() -> OverlapSources:
    return OverlapSources.from_texts(
        reference_sections=[
            f"# Estuary proposal\n\n{HOUSE}\n\n{SHORT}\n\n{AUTHORISED}",
            f"## Figures\n\n{CHART}\n\n{visual('Map', MAP)}\n\n{REFERENCE_ONLY}",
        ],
        holdout_sections=[
            f"# About us\n\n{HOUSE}\n\n{SHORT}\n\n{AUTHORISED}",
            f"{visual('Chart', CHART)}\n\n{MAP}\n\n{HOLDOUT_ONLY}\n\n{HOLDOUT_TABLE}",
        ],
        run_notes=f"Client: Contoso Ports. Area: 812 km2.\n{AUTHORISED}",
        instructions="Write the estuary proposal in Markdown.",
    )


def grams(text: str) -> frozenset[tuple[str, ...]]:
    return induction.shingles(fold_for_quote(text))


def measure(sources: OverlapSources, draft: str) -> LexicalOverlap:
    """`sources.measure(draft)`, checking that both measures lie in 0..1 whenever defined."""
    measured = sources.measure(draft)
    for value in (measured.house_text_recall, measured.copied_text_contamination):
        assert value is None or 0.0 <= value <= 1.0, measured
    return measured


def test_visual_transcripts_are_block_quotes_whose_first_line_opens_with_the_prefix() -> None:
    ordinary = f"> {HOUSE}\n> **[Visual \u2014 Page 2: Chart]** is how the compiler labels figures."
    markdown = f"{ordinary}\n\n{visual('Map', MAP)}\n\nAfter."

    assert without_visual_transcripts(markdown) == f"{ordinary}\n\n\nAfter."
    # `standing_units` reads the same block as the one transcript.
    assert [u.text for u in induction.standing_units(markdown)] == [ordinary, "After."]


def test_house_text_is_reference_prose_the_holdout_holds_outside_authorised_input() -> None:
    sources = overlap_sources()
    measured = measure(sources, f"# Estuary\n\n{HOUSE}")
    # REFERENCE_ONLY is not held; SHORT is too short; CHART is held only as a visual transcript
    # and MAP is a transcript in the reference, and transcripts never count. AUTHORISED is a
    # house unit whose shingles are all authorised input, so the 12 house shingles are HOUSE's.
    assert [u.text_head for u in measured.house_units] == [HOUSE, AUTHORISED]
    assert len(sources.house_shingles) == len(grams(HOUSE)) == 12
    assert measured.house_text_recall == 1.0


def test_house_text_recall_grades_how_much_of_the_house_shingles_a_draft_holds() -> None:
    sources = overlap_sources()
    full = measure(sources, f"# Estuary\n\n{HOUSE}\n\nThe estuary grid keeps every sounding.")
    assert full.house_text_recall == 1.0
    assert [(u.text_head, u.share) for u in full.house_units] == [(HOUSE, 1.0), (AUTHORISED, 0.0)]
    partial = "Northwind Marine reduces every sounding to the stated vertical datum."
    assert measure(sources, partial).house_text_recall == pytest.approx(7 / 12)
    assert measure(sources, "The estuary grid keeps every sounding.").house_text_recall == 0.0
    # Table text is not prose: a house paragraph placed only in a table cell earns nothing.
    in_table = measure(sources, f"| Note |\n| --- |\n| {HOUSE} |\n\n{partial}")
    assert in_table.house_text_recall == pytest.approx(7 / 12)
    without_house = OverlapSources.from_texts(
        reference_sections=[REFERENCE_ONLY],
        holdout_sections=[HOUSE],
        run_notes="notes",
        instructions="Write it.",
    )
    assert measure(without_house, HOUSE).house_text_recall is None


def test_copied_text_contamination_counts_copied_holdout_text_absent_from_the_reference() -> None:
    sources = overlap_sources()
    own = "The estuary grid keeps every sounding."
    stray_draft = f"# Estuary\n\n{HOLDOUT_ONLY}\n\n{own}"
    stray = measure(sources, stray_draft)
    # The denominator is the draft's prose shingles: the heading adds none.
    assert stray.draft_shingles == len(grams(HOLDOUT_ONLY) | grams(own)) == 13
    assert stray.stray_shingles == len(grams(HOLDOUT_ONLY)) == 10
    assert stray.copied_text_contamination == pytest.approx(10 / 13)
    assert [(u.text_head, u.stray_shingles) for u in stray.copied_units] == [(HOLDOUT_ONLY, 10)]
    # A table copied from the holdout neither counts as stray nor dilutes the denominator.
    with_table = measure(sources, f"{stray_draft}\n\n{HOLDOUT_TABLE}")
    assert (with_table.stray_shingles, with_table.draft_shingles) == (10, 13)
    # Words the drafter added to a copied passage are not in the holdout, so they are not stray.
    adapted = measure(sources, f"{HOLDOUT_ONLY[:-1]} to Contoso.")
    assert [(u.text_head, u.stray_shingles) for u in adapted.copied_units] == [
        (f"{HOLDOUT_ONLY[:-1]} to Contoso.", 10)
    ]
    assert adapted.copied_text_contamination == pytest.approx(10 / 12)
    # A copied paragraph the reference also carries is house text, not contamination.
    house = measure(sources, f"# Estuary\n\n{HOUSE}")
    assert house.copied_text_contamination == 0.0
    assert [(u.text_head, u.stray_shingles) for u in house.copied_units] == [(HOUSE, 0)]
    no_copy = measure(sources, f"{own} It keeps them at full resolution for the client.")
    assert no_copy.copied_units == () and no_copy.copied_text_contamination == 0.0
    assert measure(sources, "Draft.").copied_text_contamination is None
    assert measure(sources, f"# Estuary\n\n{HOLDOUT_TABLE}").copied_text_contamination is None


def test_a_draft_without_holdout_access_that_restates_the_run_notes_is_not_contaminated() -> None:
    # A zero-shot draft restates a run-notes sentence that a holdout section also holds, and
    # adds its own words; nothing in it comes from the holdout alone.
    draft = measure(overlap_sources(), f"{AUTHORISED[:-1]} at Contoso.")
    assert [u.stray_shingles for u in draft.copied_units] == [0]
    assert draft.copied_text_contamination == 0.0


def test_copied_text_contamination_never_exceeds_one() -> None:
    sources = OverlapSources.from_texts(
        reference_sections=["Rainfall totals were checked against the regional gauge network."],
        holdout_sections=["1. a1 a1 a1 a1 a1 a1 a1 a-"],
        run_notes="notes",
        instructions="Write it.",
    )
    # Folded as one string, the line-break hyphen joined "a-" to the next line and the draft
    # had fewer shingles than its copied unit; a draft's units are folded each on its own.
    measured = measure(sources, "1. a1 a1 a1 a1 a1 a1 a1 a-\n1. a1")
    assert measured.copied_text_contamination == 1.0


def test_identical_notes_reference_and_draft_across_a_dash_before_a_numbered_item_match() -> None:
    # Folding the whole text joins the dash with "1": the per-unit shingles of the same text
    # differ from its whole-text shingles, so text shingles hold both forms.
    notes_and_reference = (
        "Northwind Marine applies two quality checks to every bathymetric surface\u2014\n"
        "1. Every sounding line is reduced to the stated chart datum before gridding."
    )
    sources = OverlapSources.from_texts(
        reference_sections=[notes_and_reference],
        holdout_sections=[notes_and_reference.replace("\u2014", ":")],
        run_notes=notes_and_reference,
        instructions="Write it.",
    )
    measured = measure(sources, notes_and_reference)
    assert len(measured.copied_units) == 2
    assert measured.house_text_recall is None
    assert measured.copied_text_contamination == 0.0


def test_a_numbered_and_a_dashed_list_item_measure_the_same() -> None:
    item = "Northwind Marine reduces every sounding to datum."
    sources = OverlapSources.from_texts(
        reference_sections=[f"{REFERENCE_ONLY}\n\n- {item}"],
        holdout_sections=[f"- {item}", HOLDOUT_ONLY],
        run_notes="notes",
        instructions="Write it.",
    )
    # Seven words without the marker: neither form is long enough to be a copied unit.
    assert measure(sources, f"1. {item}") == measure(sources, f"- {item}")
    assert measure(sources, f"1. {HOLDOUT_ONLY}").copied_text_contamination == 1.0


def test_a_marker_alone_on_its_line_is_a_word_of_the_paragraph_it_opens() -> None:
    # "1." with nothing after it on its line is no list marker, so the paragraph keeps the "1"
    # and has the eight words of a house unit.
    text = "1.\nEach measurement report records current site baseline"
    sources = OverlapSources.from_texts(
        reference_sections=[text], holdout_sections=[text], run_notes="notes", instructions="Go."
    )
    assert len(sources.house_units) == 1
    assert measure(sources, text).house_text_recall == 1.0


def test_a_paragraph_joining_sentences_of_two_holdout_sections_is_a_copied_unit() -> None:
    harbour = "Harbour lidar soundings required coastline land masking before final export."
    reef = "Reef sonar mosaics exhibited acoustic shadowing behind coral pinnacles."
    sources = OverlapSources.from_texts(
        reference_sections=[REFERENCE_ONLY],
        holdout_sections=[harbour, reef],
        run_notes="notes",
        instructions="Write it.",
    )
    measured = measure(sources, f"{harbour} {reef}")
    # Neither section holds 80% of the joined paragraph's shingles; the holdout text does.
    assert [u.stray_shingles for u in measured.copied_units] == [len(grams(harbour) | grams(reef))]
    assert measured.copied_text_contamination == pytest.approx(13 / 16)


def test_a_fenced_block_is_kept_as_written_and_never_starts_a_visual_transcript() -> None:
    fenced = f"```\n{visual('Chart', HOLDOUT_ONLY)}\n```"
    sources = OverlapSources.from_texts(
        reference_sections=[f"# Estuary\n\n{fenced}"],
        holdout_sections=[HOLDOUT_ONLY],
        run_notes="notes",
        instructions="Write it.",
    )
    # The reference carries the sentence inside its fenced block, so a draft copying it from
    # the holdout copies nothing stray.
    copied = measure(sources, HOLDOUT_ONLY)
    assert [u.stray_shingles for u in copied.copied_units] == [0]
    prompt = exemplars_prompt([("p101/memo.md", f"{fenced}\n\n{visual('Map', MAP)}")], "notes")
    assert f"{fenced}\n\nRUN NOTES:" in prompt and MAP not in prompt


def test_rubric_diagnostics_label_where_each_required_group_occurs_in_the_inputs() -> None:
    check = RubricCheck.model_validate(
        {
            "check_id": "kept",
            "category": "c",
            "description": "d",
            "require_groups": [["812 km2"], ["Markdown"], ["ground truth", "check line"], ["S-44"]],
        }
    )
    (diagnostic,) = rubric_diagnostics(
        [check],
        run_notes="Area: 812 km2.",
        instructions="Write it in Markdown.",
        holdout_sections=["# Memo", "Ground truthing confirmed the depths."],
    )
    assert diagnostic.check_id == "kept"
    assert [g.found_in for g in diagnostic.groups] == [
        "run_notes",
        "instructions",
        "holdout",
        "none",
    ]


def test_exemplars_prompt_labels_documents_by_path_without_visuals_and_ends_with_run_notes() -> (
    None
):
    prompt = exemplars_prompt(
        [
            (
                "p101/memo.md",
                f"# Memo\n\nBefore.\n\n{visual('Chart', 'Median 330.9 mm')}\n\nAfter.",
            ),
            ("p103/review.md", "# Review\n\nCut the jargon."),
        ],
        "- area: 812 km2",
    )
    assert prompt.index("===== DOCUMENT p101/memo.md =====\n# Memo") < prompt.index(
        "===== DOCUMENT p103/review.md =====\n# Review"
    )
    assert "After." in prompt and "330.9" not in prompt and "[Visual" not in prompt
    assert prompt.endswith("RUN NOTES:\n- area: 812 km2")


# Each drafting obligation of the MCP server's instructions: the phrase that carries it there
# and in the exemplars-in-context rules.
DRAFTING_OBLIGATIONS = {
    "follow_shared_structure": (
        "follow its outline, conventions and reviewer pitfalls",
        "Follow the structure and conventions the documents share",
    ),
    "mirror_layout": (
        "mirror its heading levels and generic table column headers",
        "Mirror the documents' heading levels and generic table column headers",
    ),
    "no_project_specific_headings": (
        "Do not reuse project-specific headings or facts",
        "Do not reuse their project-specific headings or facts",
    ),
    "project_facts_from_inputs_only": (
        "comes only from the user's inputs",
        "comes only from the run notes",
    ),
    "project_facts_take_precedence": ("This rule takes precedence", "This rule takes precedence"),
    "house_wording_as_written": (
        "keep their organisation names, method references and attributions as written",
        "keep its organisation names, method references and attributions as written",
    ),
    "corpus_wording_only_as_house_text": (
        "Reuse other wording only from standing passages",
        "Reuse other wording, names, references and attributions from the documents only within",
    ),
    "enumerations_in_full": ("list every item in full", "list every item in full"),
    "standalone": ("Write a standalone deliverable", "Write a standalone deliverable"),
}


# The server's drafting rules as `EXEMPLAR_DRAFTING_RULES` was last reconciled with them.
SERVER_DRAFTING_RULES = """\
2. Drafting a new deliverable from the user's own inputs:
   - Call get_blueprint for the deliverable type and follow its outline, conventions
     and reviewer pitfalls.
   - For each section whose layout matters, read one of its cited exemplar sections
     with read_section and mirror its heading levels and generic table column headers.
     Do not reuse project-specific headings or facts. Reuse other wording only from
     standing passages.
   - Every fact about the user's project (its client and other parties, people, places,
     dates, quantities, budgets, targets, findings, identifiers, file names) comes only
     from the user's inputs. This rule takes precedence over the next one.
   - The blueprint's standing passages are the collection's own house text. Reproduce
     them with their wording, without commentary about where they come from, and keep
     their organisation names, method references and attributions as written. Adapt or
     leave out only the part of a passage that belongs to one particular project (its
     client, site, dates, quantities or results), using the user's inputs. Names,
     references and attributions from the corpus may be reused only when they appear in
     a standing passage.
   - Where the user's inputs enumerate items (files, deliverables, parameters,
     results), list every item in full, never as patterns or "etc.".
   - Write a standalone deliverable: no Arkive doc_id or section_id, and no mention of
     the exemplar or feedback documents.
"""


# The drafting guidance of the server's tool descriptions, as the server publishes them.
SERVER_DRAFTING_TOOL_DESCRIPTIONS = {
    "get_blueprint": (
        "The full writing blueprint for one deliverable type: section outline with cited\n"
        "exemplar sections, conventions, reviewer pitfalls, exemplar and feedback documents,\n"
        "and standing passages: the collection's house text, which recurs in another exemplar\n"
        "document, exactly as it appears in its first cited section. When drafting, keep a\n"
        "passage's organisation names, method references and attributions as written, and\n"
        "adapt only the part that belongs to one particular project. The type is matched\n"
        "ignoring case and surrounding spaces."
    ),
    "read_section": (
        "Full Markdown of one section, with the IDs of its neighbouring sections.\n\n"
        "When drafting a new deliverable, mirror an exemplar section's heading levels and\n"
        "generic table column headers, but not its project-specific headings or facts. Reuse\n"
        "other wording, names, references and attributions only from the blueprint's standing\n"
        "passages, which are house text kept as written apart from\n"
        "their project-specific parts; take every fact about the user's project from the\n"
        "user's inputs, and never cite doc_id or section_id in the deliverable.\n"
    ),
}

# `EXEMPLAR_DRAFTING_RULES` as last reconciled with the server's drafting surfaces above.
PINNED_EXEMPLAR_DRAFTING_RULES = """\
The prompt holds prior deliverables of the same collection, each labelled with its file path,
followed by the run notes of the new project.
- Follow the structure and conventions the documents share, and avoid what any reviewer
  feedback among them criticises.
- Every fact about the new project (its client and other parties, people, places, dates,
  quantities, budgets, targets, findings, identifiers, file names) comes only from the run
  notes. This rule takes precedence over the next ones and over anything in the documents.
- Mirror the documents' heading levels and generic table column headers. Do not reuse their
  project-specific headings or facts.
- The documents' house wording (organisation or tool background, method references,
  acknowledgements, disclaimers, standing caveats) is the collection's own text. Reproduce it
  where it applies, with its wording, and keep its organisation names, method references and
  attributions as written. Adapt or leave out only the part that belongs to one particular
  project (its client, site, dates, quantities or results), using the run notes. Reuse other
  wording, names, references and attributions from the documents only within that house
  wording.
- Where the run notes enumerate items (files, deliverables, parameters, results), list every
  item in full, never as patterns or "etc.".
- Write a standalone deliverable: no file paths of the documents, and no mention of the
  documents in the prompt."""


def test_exemplars_in_context_rules_carry_the_pinned_drafting_rules_of_the_server() -> None:
    async def published_descriptions() -> dict[str, str]:
        with TopicCatalog([]) as catalog:
            async with Client(create_mcp_server(catalog)) as session:
                listed = await session.list_tools()
        return {t.name: t.description or "" for t in listed.tools}

    descriptions = anyio.run(published_descriptions)
    # Every drafting surface is pinned word for word: an edit to any of them fails here until
    # the pins, the obligation map and, where an obligation changed, the rules move together.
    assert INSTRUCTIONS[INSTRUCTIONS.index("2. Drafting") :] == SERVER_DRAFTING_RULES
    for name, pinned in SERVER_DRAFTING_TOOL_DESCRIPTIONS.items():
        assert descriptions[name] == pinned, name
    assert EXEMPLAR_DRAFTING_RULES == PINNED_EXEMPLAR_DRAFTING_RULES
    server_text = " ".join(SERVER_DRAFTING_RULES.split())
    rules_text = " ".join(EXEMPLAR_DRAFTING_RULES.split())
    for name, (server, rules) in DRAFTING_OBLIGATIONS.items():
        assert server in server_text, name
        assert rules in rules_text, name


def write_corpus(root: Path, files: dict[str, str | bytes]) -> Path:
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        (root / rel).write_bytes(data)
    return root


REFERENCE_PATH = "p102-estuary-multibeam/Estuary Proposal FINAL.md"
REFERENCE_TEXT = "# Estuary\n\nMultibeam 2000 x 2000 tiles."
HOLDOUT_FILES: dict[str, str | bytes] = {
    "p101-harbour-lidar/harbour-memo.md": "# Harbour\n\nLidar grid 1 m.",
    REFERENCE_PATH: REFERENCE_TEXT,
    "p103-reef-sonar/memo.md": "# Reef\n\nSidescan notes.",
    "AGENTS.md": "# Notes\n\nCorpus guidance.",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def reference_doc(**overrides: Any) -> DocumentInfo:
    fields: dict[str, Any] = {
        "doc_id": "doc_003",
        "rel_path": REFERENCE_PATH,
        "sha256": hashlib.sha256(REFERENCE_TEXT.encode()).hexdigest(),
        "format": "md",
        "title": "Estuary Proposal",
        "doc_date": None,
        "deliverable_type": "proposal",
        "summary": "Estuary proposal.",
        "page_count": 1,
        "is_curator_note": False,
        "facets": {},
    }
    return DocumentInfo.model_validate(fields | overrides)


def test_holdout_selection_keeps_included_files_and_refuses_held_out_material(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    task = holdout_task(example_pack)
    reference = reference_doc()
    check_reference_held_out(task, reference)
    with pytest.raises(ValueError, match="no held-out marker"):
        check_reference_held_out(task, reference_doc(rel_path="p101-harbour-lidar/harbour-memo.md"))

    corpus = write_corpus(tmp_path / "corpus", HOLDOUT_FILES)
    selected = select_holdout_sources(corpus, task, reference)
    assert sorted(selected) == ["p101-harbour-lidar/harbour-memo.md", "p103-reef-sonar/memo.md"]

    renamed = write_corpus(
        tmp_path / "renamed",
        HOLDOUT_FILES | {"p103-reef-sonar/old draft.md": HOLDOUT_FILES[reference.rel_path]},
    )
    with pytest.raises(ValueError, match="same bytes as the reference"):
        select_holdout_sources(renamed, task, reference)
    misfiled = write_corpus(
        tmp_path / "misfiled",
        HOLDOUT_FILES | {"p101-harbour-lidar/Estuary\u2010Multibeam notes.md": "x"},
    )
    # Markers match case- and dash-insensitively.
    with pytest.raises(ValueError, match="held-out marker 'estuary-multibeam'"):
        select_holdout_sources(misfiled, task, reference)
    # Folders mean nothing to the guard: a reference at the corpus root leaves the other root
    # files usable when they carry no marker and differ in bytes, and a nested collection can
    # include a sibling of the reference's folder under the same top-level folder.
    at_root = task.model_copy(update={"included_paths": ("AGENTS.md",), "exemplar_paths": ()})
    root_reference = reference_doc(rel_path="P102 Proposal FINAL.md", sha256="0")
    check_reference_held_out(at_root, root_reference)
    assert sorted(select_holdout_sources(corpus, at_root, root_reference)) == ["AGENTS.md"]
    flat = write_corpus(tmp_path / "flat", {"AGENTS.md": "# Notes", "P102 Proposal.md": "x"})
    with pytest.raises(ValueError, match="same bytes as the reference"):
        select_holdout_sources(
            flat, at_root, reference_doc(rel_path="P102.md", sha256=_sha("# Notes"))
        )
    nested = write_corpus(
        tmp_path / "nested",
        {"clients/contoso/p101/memo.md": "# Harbour", "clients/contoso/p102/proposal.md": "x"},
    )
    sibling = task.model_copy(
        update={"included_paths": ("clients/contoso/p101",), "exemplar_paths": ()}
    )
    nested_reference = reference_doc(rel_path="clients/contoso/p102/proposal.md", sha256=_sha("x"))
    assert sorted(select_holdout_sources(nested, sibling, nested_reference)) == [
        "clients/contoso/p101/memo.md"
    ]
    # An exemplar must be one of the included files, not just a path under an included folder.
    absent = task.model_copy(update={"exemplar_paths": ("p101-harbour-lidar/gone.md",)})
    with pytest.raises(
        ValueError, match=r"'p101-harbour-lidar/gone\.md' is not an included corpus"
    ):
        select_holdout_sources(corpus, absent, reference)
    # An exemplar is a delivered document, not a curator note.
    note_exemplar = task.model_copy(
        update={
            "included_paths": ("p101-harbour-lidar", "p103-reef-sonar"),
            "exemplar_paths": ("p103-reef-sonar/AGENTS.md",),
        }
    )
    noted = write_corpus(tmp_path / "noted", HOLDOUT_FILES | {"p103-reef-sonar/AGENTS.md": "x"})
    with pytest.raises(ValueError, match=r"'p103-reef-sonar/AGENTS\.md' is a curator note"):
        select_holdout_sources(noted, note_exemplar, reference)
    with pytest.raises(ValueError, match="is a curator note"):
        check_reference_held_out(
            task, reference_doc(rel_path="p102-estuary-multibeam/AGENTS.md", is_curator_note=True)
        )
    linked = write_corpus(tmp_path / "linked", HOLDOUT_FILES)
    (linked / "p101-harbour-lidar" / "reef.md").symlink_to(linked / reference.rel_path)
    with pytest.raises(UnsafePathError, match="symlinks"):
        select_holdout_sources(linked, task, reference)
    missing = write_corpus(tmp_path / "missing", {"p101-harbour-lidar/harbour-memo.md": "x"})
    with pytest.raises(ValueError, match="'p103-reef-sonar' matches no corpus file"):
        select_holdout_sources(missing, task, reference)

    with pytest.raises(ValueError, match="held-out marker 'p102'"):
        assert_holdout_clean(task, reference, [reference_doc(rel_path="moved/P102.md", sha256="0")])
    with pytest.raises(ValueError, match="same bytes"):
        assert_holdout_clean(task, reference, [reference_doc(rel_path="p101/x.md")])
    # A file beside the reference without a marker in its own path is not held-out material.
    assert_holdout_clean(task, reference, [reference_doc(rel_path="final/other.md", sha256="0")])


def test_holdout_snapshot_is_published_only_after_its_guard_passes(
    tmp_path: Path, example_pack: EvalPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = holdout_task(example_pack)
    corpus = write_corpus(tmp_path / "corpus", HOLDOUT_FILES)
    output = tmp_path / "out" / "holdout.arkive.db"
    output.parent.mkdir()
    compile_holdout_snapshot(corpus, output, FakeGateway(), task, reference_doc())
    with TopicSnapshot.open(output) as holdout:
        assert [d.rel_path for d in holdout.list_documents()] == [
            "p101-harbour-lidar/harbour-memo.md",
            "p103-reef-sonar/memo.md",
        ]
        # Facets are induced from the included files only.
        assert [v.value for v in holdout.facets[0].values] == [
            "p101-harbour-lidar",
            "p103-reef-sonar",
        ]
    output.unlink()

    def leak(*args: object) -> None:
        raise ValueError("holdout snapshot leaks held-out material")

    gateway = FakeGateway()
    renamed = write_corpus(
        tmp_path / "renamed",
        HOLDOUT_FILES | {"p101-harbour-lidar/copy.md": REFERENCE_TEXT},
    )
    with pytest.raises(ValueError, match="same bytes"):
        compile_holdout_snapshot(renamed, output, gateway, task, reference_doc())
    assert gateway.calls == [] and list(output.parent.iterdir()) == []

    monkeypatch.setattr(drafting, "assert_holdout_clean", leak)
    with pytest.raises(ValueError, match="leaks"):
        compile_holdout_snapshot(corpus, output, FakeGateway(), task, reference_doc())
    assert list(output.parent.iterdir()) == []


def test_eval_cli_refuses_mismatched_inputs_before_outputs_or_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    corpus = tmp_path / "bathymetry-corpus"
    with TopicSnapshot.open(snapshot_path) as snapshot:
        fingerprint = snapshot.corpus_fingerprint
    retrieval_only = tmp_path / "retrieval.eval.json"
    retrieval_only.write_text(json.dumps(pack_with(qa_cases=[], holdouts=[])))
    other_corpus = tmp_path / "other.eval.json"
    other_corpus.write_text(json.dumps(pack_with(corpus_fingerprint="0" * 64)))
    out = tmp_path / "o"
    runner = CliRunner()

    def run(pack: Path, *extra: str, out: Path = out, snapshot: Path = snapshot_path) -> str:
        result = runner.invoke(
            eval_app,
            ["--pack", str(pack), "--snapshot", str(snapshot), "--out", str(out), *extra],
        )
        assert result.exit_code == 2, result.output
        # Drop the error panel's borders so wrapped messages read as one line.
        return " ".join(result.output.replace("\u2502", " ").split())

    assert fingerprint in run(other_corpus, "--layer", "1")
    assert "defines no cases for layer 2" in run(retrieval_only, "--layer", "2")
    assert "need --corpus" in run(EXAMPLE_PACK_PATH, "--layer", "3")
    no_out = runner.invoke(
        eval_app, ["--pack", str(EXAMPLE_PACK_PATH), "--snapshot", str(snapshot_path)]
    )
    assert no_out.exit_code == 2 and "Missing option '--out'" in " ".join(no_out.output.split())
    assert "overlaps --snapshot" in run(EXAMPLE_PACK_PATH, "--layer", "1", out=tmp_path)
    layer2 = ("--layer", "2", "--corpus", str(corpus))
    assert "overlaps --corpus" in run(EXAMPLE_PACK_PATH, *layer2, out=corpus / "results")
    cache_error = run(EXAMPLE_PACK_PATH, *layer2, "--cache", str(snapshot_path))
    assert "--cache" in cache_error and "overlaps --snapshot" in cache_error
    assert "overlaps --pack" in run(
        EXAMPLE_PACK_PATH, "--layer", "1", "--cache", str(EXAMPLE_PACK_PATH)
    )
    # The cache and the output directory must not overlap each other either.
    shared = tmp_path / "shared"
    for clash in (shared, shared / "report.md"):
        clash_error = run(EXAMPLE_PACK_PATH, *layer2, "--cache", str(clash), out=shared)
        assert "--cache" in clash_error and "overlaps --out" in clash_error
    assert not shared.exists()
    # Layer 3 needs a fresh --out: one holding an earlier run's holdouts is refused.
    reused = tmp_path / "reused"
    (reused / "holdouts" / "estuary-proposal").mkdir(parents=True)
    layer3 = ("--layer", "3", "--corpus", str(corpus))
    assert "already holds a holdouts directory" in run(EXAMPLE_PACK_PATH, *layer3, out=reused)
    assert sorted(p.name for p in reused.rglob("*")) == ["estuary-proposal", "holdouts"]
    linked_out = tmp_path / "linked-out"
    linked_out.symlink_to(reused, target_is_directory=True)
    assert "is a symlink" in run(EXAMPLE_PACK_PATH, "--layer", "1", out=linked_out)

    def pack_file(name: str, **overrides: Any) -> Path:
        path = tmp_path / f"{name}.eval.json"
        path.write_text(json.dumps(pack_with(**overrides)))
        return path

    # An invalid pack or snapshot is a clean usage error, not a traceback.
    assert "is not a valid eval pack" in run(pack_file("v1", format="arkive-eval-pack/1"))
    assert "reads 'arkive-eval-pack/3' packs only" in run(
        pack_file("v2", format="arkive-eval-pack/2")
    )
    not_utf8 = tmp_path / "latin1.eval.json"
    not_utf8.write_bytes(b'{"name": "caf\xe9"}')
    assert "is not a valid eval pack" in run(not_utf8)
    garbage = tmp_path / "garbage.arkive.db"
    garbage.write_bytes(b"not a database")
    assert "--snapshot" in run(EXAMPLE_PACK_PATH, "--layer", "1", snapshot=garbage)
    older = tmp_path / "older.arkive.db"
    older.write_bytes(snapshot_path.read_bytes())
    with closing(sqlite3.connect(older)) as db, db:
        db.execute("UPDATE corpus_meta SET value = '2.0' WHERE key = 'schema_version'")
    assert "schema version '2.0'" in run(EXAMPLE_PACK_PATH, "--layer", "1", snapshot=older)
    # With --layer all, a later layer's inputs are checked before layer 1 calls any model.
    everything = ("--corpus", str(corpus))
    unknown_filter = pack_file("filter", retrieval_cases__3__filters={"project": "p999"})
    assert "retrieval case 'disambiguation_and_structure_1': unknown project value" in run(
        unknown_filter, *everything
    )
    unknown_expected = pack_file("expected", retrieval_cases__1__expected_docs=["doc_009"])
    assert "retrieval case 'visual_only_1': unknown document 'doc_009'" in run(
        unknown_expected, "--layer", "1"
    )
    note_reference = pack_file("note", holdouts__0__reference_doc_id="doc_003")
    assert "holdout 'estuary-proposal': reference document doc_003" in run(
        note_reference, *everything
    )
    assert "is a curator note" in run(note_reference, *everything)
    missing_part = pack_file("part", holdouts__0__included_paths=["p101-harbour-lidar", "p104"])
    assert "'p104' matches no corpus file" in run(missing_part, *everything)
    assert "--holdout 'nope' is not in the pack: ['estuary-proposal']" in run(
        EXAMPLE_PACK_PATH, *everything, "--holdout", "nope"
    )
    assert "layer 3 is not run" in run(EXAMPLE_PACK_PATH, "--layer", "1", "--holdout", "x")
    assert "--drafts sets layer 3" in run(EXAMPLE_PACK_PATH, "--layer", "1", "--drafts", "2")
    assert "--drafts" in run(EXAMPLE_PACK_PATH, *everything, "--drafts", "0")
    unknown_doc = pack_file(
        "doc", qa_cases__0__source_anchors=[{"doc_id": "doc_009", "pages": None}]
    )
    assert "qa case 'harbour_median_rainfall': unknown document 'doc_009'" in run(
        unknown_doc, *everything
    )
    past_end = pack_file("pages", qa_cases__0__source_anchors__0__pages=[1, 4])
    assert (
        "source anchor doc_001 pages [1, 4]: p101-harbour-lidar/harbour-memo.md has no pages"
        in (run(past_end, *everything))
    )

    (corpus / "p102-estuary-multibeam" / "estuary-proposal.md").write_text("# Edited\n")
    drift = run(EXAMPLE_PACK_PATH, *layer2)
    assert "rebuild the snapshot" in drift and fingerprint in drift
    (corpus / "link.md").symlink_to(corpus / "p102-estuary-multibeam" / "estuary-proposal.md")
    assert "symlinks are not allowed" in run(EXAMPLE_PACK_PATH, *layer2)
    assert not out.exists() and not (corpus / "results").exists()


def test_eval_outputs_are_private_files(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    target.write_text("old")
    target.chmod(0o644)
    write_private_text(target, "new")
    assert target.read_text() == "new" and stat.S_IMODE(target.stat().st_mode) == 0o600
    # The mode does not depend on the umask: a restrictive one still yields a readable 0600 file.
    previous_umask = os.umask(0o477)
    try:
        write_private_text(target, "under umask")
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600 and target.read_text() == "under umask"
    # A failure while writing leaves the earlier file byte for byte and no temporary file.
    earlier = tmp_path / "results.json"
    earlier.write_bytes(b'{"earlier": true}\n')
    with pytest.raises(UnicodeEncodeError):
        write_private_text(earlier, "unencodable \ud800")
    assert earlier.read_bytes() == b'{"earlier": true}\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.md", "results.json"]
    # A link at the destination is replaced, not followed and truncated.
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    link = tmp_path / "linked.json"
    link.symlink_to(victim)
    write_private_text(link, "new")
    assert not link.is_symlink() and link.read_text() == "new"
    assert victim.read_text() == "keep"
    # Output directories must be real directories, not links to one.
    make_private_directory(tmp_path / "events")
    assert stat.S_IMODE((tmp_path / "events").stat().st_mode) == 0o700
    make_private_directory(tmp_path / "events")
    (tmp_path / "linked").symlink_to(tmp_path / "events", target_is_directory=True)
    for not_a_directory in (tmp_path / "linked", victim):
        with pytest.raises(ValueError, match="not a real directory"):
            make_private_directory(not_a_directory)


def test_blind_assignment_is_a_seeded_permutation() -> None:
    arms = ["zero_shot", "chunk_search", "arkive_mcp", "exemplars_in_context"]
    key = blind_assignment(arms)
    assert key == blind_assignment(arms)
    assert list(key) == ["Candidate_1", "Candidate_2", "Candidate_3", "Candidate_4"]
    assert sorted(key.values()) == sorted(arms)


def test_a_blind_comparison_must_assess_and_rank_exactly_the_labels_sent_once_each() -> None:
    labels = ["Candidate_1", "Candidate_2"]
    valid = DraftBlindComparison(
        candidates=[candidate_scores(label, 5) for label in labels],
        ranking=["Candidate_2", "Candidate_1"],
    )
    assert valid.label_problems(labels) == []
    mislabelled = DraftBlindComparison(
        candidates=[
            candidate_scores("Candidate 1", 5),
            candidate_scores("Candidate_2", 5),
            candidate_scores("Candidate_2", 4),
        ],
        ranking=["Candidate_2", "Candidate_2", "Candidate_3"],
    )
    assert mislabelled.label_problems(labels) == [
        "candidates missing ['Candidate_1']",
        "candidates unknown ['Candidate 1']",
        "candidates repeated ['Candidate_2']",
        "ranking missing ['Candidate_1']",
        "ranking unknown ['Candidate_3']",
        "ranking repeated ['Candidate_2']",
    ]


def invoke(tools: list[Tool], name: str, **args: Any) -> ToolResult:
    """Call a tool the way the gateway does: awaiting it when it is a coroutine function."""
    (tool,) = [t for t in tools if t.name == name]

    async def call() -> ToolResult:
        result = tool.invoke(args)
        return await result if inspect.isawaitable(result) else result

    return anyio.run(call)


def test_raw_corpus_tools_read_text_layers_and_page_images(tmp_path: Path) -> None:
    (tmp_path / "p1").mkdir()
    (tmp_path / "AGENTS.md").write_text("# Notes\nStill closing out.", encoding="utf-8")
    write_pdf(tmp_path / "p1" / "memo.pdf", [["Rainfall 330.9 mm"], ["Outputs depth_grid.tif"]])
    tools = RawCorpus(tmp_path).tools

    assert json.loads(cast(str, invoke(tools, "list_corpus_files"))) == [
        {"rel_path": "AGENTS.md", "pages": 1},
        {"rel_path": "p1/memo.pdf", "pages": 2},
    ]
    assert json.loads(cast(str, invoke(tools, "grep_raw_text", pattern="330.9"))) == [
        {"rel_path": "p1/memo.pdf", "page": 1, "line": "Rainfall 330.9 mm"}
    ]
    pages = invoke(tools, "read_raw_pages", rel_path="p1/memo.pdf", start_page=1, end_page=9)
    assert "--- page 2 ---\nOutputs depth_grid.tif" in cast(str, pages)
    image = invoke(tools, "view_raw_page_image", rel_path="p1/memo.pdf", page_number=2)
    assert isinstance(image, ToolReply) and image.images[0].mime_type == "image/webp"
    assert image.images[0].data[8:12] == b"WEBP"
    no_image = invoke(tools, "view_raw_page_image", rel_path="AGENTS.md", page_number=1)
    assert cast(str, no_image).startswith("ERROR")
    # Malformed arguments are invalid input, which the gateway returns to the model.
    with pytest.raises(ValueError, match="start_page must be an integer page number, got None"):
        invoke(tools, "read_raw_pages", rel_path="p1/memo.pdf", start_page=None, end_page=1)
    with pytest.raises(ValueError, match="page_number"):
        invoke(tools, "view_raw_page_image", rel_path="p1/memo.pdf", page_number="2")


def test_arkive_mcp_adapter_exposes_every_server_tool_and_forwards_images(tmp_path: Path) -> None:
    snapshot_path = compile_snapshot(
        tmp_path, BATHYMETRY_CORPUS, pdfs={"p101-harbour-lidar/chart.pdf": [["Figure 330.9"]]}
    )

    async def run() -> tuple[list[str], ToolResult, ToolResult, ToolResult, str]:
        async with arkive_mcp_tools(snapshot_path, FakeGateway().embed_query) as (
            tools,
            server_instructions,
        ):
            by_name = {t.name: t for t in tools}

            async def call(name: str, **args: Any) -> ToolResult:
                result = by_name[name].invoke(args)
                return await result if inspect.isawaitable(result) else result

            grep = await call("grep", topic="bathymetry", pattern="330.9")
            page = await call("view_page", topic="bathymetry", doc_id="doc_001", page_number=1)
            missing = await call("read_section", topic="bathymetry", section_id="nope")
            return sorted(by_name), grep, page, missing, server_instructions

    names, grep, page, missing, server_instructions = anyio.run(run)
    assert server_instructions == INSTRUCTIONS
    assert with_server_instructions("Answer.", "  ") == "Answer."
    # The OpenCode check keeps its own closed list of Arkive tools; it must name exactly the
    # tools the server registers, or the check rejects a valid call as a non-Arkive tool.
    assert names == sorted(ARKIVE_TOOLS)
    assert len(names) == 10
    assert isinstance(grep, str) and "330.9" in grep
    assert isinstance(page, ToolReply) and page.images[0].mime_type == "image/webp"
    assert isinstance(missing, str) and missing.startswith("ERROR:")


# Agent conversations: the run-status contract.


def echo_tool(embedder: CountingEmbedder) -> Tool:
    async def echo(args: dict[str, Any]) -> str:
        if args.get("fail") == "provider":
            raise ProviderError("embedding service down")
        if args.get("fail") == "bug":
            raise RuntimeError("harness bug")
        embedder(str(args["value"]))
        return f"echo={args['value']}"

    return Tool("echo", "Echo a value.", {"type": "object", "properties": {}}, echo)


def converse(api: ScriptedOpenAI, max_turns: int = 4, **gateway: Any) -> AgentRun:
    embedder = CountingEmbedder(FakeGateway().embed_query)
    return anyio.run(
        lambda: run_agent(
            api.gateway(**gateway),
            operation="qa_probe",
            instructions="i",
            prompt="q",
            tools=[echo_tool(embedder)],
            max_turns=max_turns,
            embedder=embedder,
        )
    )


def scripted(*bodies: dict[str, Any]) -> ScriptedOpenAI:
    api = ScriptedOpenAI()
    api.script("/responses", *bodies)
    return api


def test_conversation_feeds_outputs_back_and_accounts_tokens_turns_and_tools() -> None:
    api = scripted(
        response_body(
            [function_call("c1", "echo", '{"value": 7}'), function_call("c2", "echo", "not json")]
        ),
        response_body([message("final answer")]),
    )

    run = converse(api)

    assert (run.answer, run.status, run.turns, len(run.tool_calls), run.error) == (
        "final answer",
        "completed",
        2,
        2,
        None,
    )
    assert (run.input_tokens, run.output_tokens, run.search_embed_calls) == (200, 20, 1)
    first, second = api.requests("/responses")
    assert (first["instructions"], first["reasoning"]) == ("i", {"effort": "medium"})
    outputs = [i["output"] for i in second["input"] if i.get("type") == "function_call_output"]
    assert outputs[0] == "echo=7" and outputs[1].startswith("ERROR: JSONDecodeError")

    looping = [response_body([function_call(f"c{i}", "echo", '{"value": 1}')]) for i in range(2)]
    capped = converse(scripted(*looping), max_turns=2)
    assert (capped.answer, capped.status, capped.turns, capped.search_embed_calls) == (
        "",
        "max_turns",
        2,
        2,
    )


def test_conversation_reports_unfinished_runs_and_surfaces_non_input_failures() -> None:
    partial = response_body([message("half a memo")], "incomplete", "max_output_tokens")
    run = converse(scripted(partial))
    assert (run.status, run.answer, len(run.tool_calls), run.input_tokens) == (
        "incomplete",
        "half a memo",
        0,
        100,
    )
    assert converse(scripted(response_body([message("  ")]))).status == "empty_answer"
    failed = converse(
        scripted(response_body([function_call("c1", "echo", '{"fail": "provider"}')]))
    )
    assert failed.status == "provider_error" and "embedding service down" in (failed.error or "")
    rejected = ScriptedOpenAI()
    rejected.script("/responses", httpx2.Response(400, json={"error": {"message": "too long"}}))
    turn_failed = converse(rejected)
    assert (turn_failed.status, turn_failed.turns) == ("provider_error", 1)
    assert "too long" in (turn_failed.error or "")
    with pytest.raises(RuntimeError, match="harness bug"):
        converse(scripted(response_body([function_call("c1", "echo", '{"fail": "bug"}')])))
    # The spend limit is not a run outcome: it ends the evaluation.
    costly = response_body(
        [function_call("c1", "echo", '{"value": 1}')], tokens=usage(100_000, 10_000)
    )
    with pytest.raises(BudgetExceededError):
        converse(scripted(costly), meter=UsageMeter(max_cost_usd=0.30))


def mcp_search_run(
    tmp_path: Path, embed: Any, queries: tuple[str, ...] = ("rainfall",)
) -> tuple[AgentRun, ScriptedOpenAI]:
    """One Arkive agent run whose model calls `search` once per query and then answers."""
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    embedder = CountingEmbedder(embed)
    searches = [
        function_call(f"c{i}", "search", json.dumps({"topic": "bathymetry", "query": query}))
        for i, query in enumerate(queries)
    ]
    api = scripted(response_body(searches), response_body([message("The median is 330.9 mm.")]))

    async def run() -> AgentRun:
        async with arkive_mcp_tools(snapshot_path, embedder) as (tools, _):
            return await run_agent(
                api.gateway(),
                operation="qa_arkive_mcp",
                instructions="Answer.",
                prompt="What is the median rainfall?",
                tools=tools,
                max_turns=4,
                embedder=embedder,
            )

    return anyio.run(run), api


def test_query_embedding_failures_behind_mcp_search_end_the_run_or_the_evaluation(
    tmp_path: Path,
) -> None:
    def unavailable(query: str) -> NDArray[np.float32]:
        raise ProviderError("embedding service down")

    run, api = mcp_search_run(tmp_path / "down", unavailable)
    # The model saw a tool error and answered anyway; the run is still not usable.
    assert len(api.requests("/responses")) == 2
    assert (run.status, run.search_embed_calls) == ("provider_error", 0)
    assert "embedding service down" in (run.error or "")

    working, _ = mcp_search_run(tmp_path / "up", FakeGateway().embed_query)
    assert (working.status, working.search_embed_calls) == ("completed", 1)

    def over_budget(query: str) -> NDArray[np.float32]:
        raise BudgetExceededError("estimated spend reached the limit")

    with pytest.raises(BudgetExceededError):
        mcp_search_run(tmp_path / "budget", over_budget)

    # A later failure that stops the evaluation outranks an earlier ordinary provider error.
    for name, stop in (
        ("down-then-budget", BudgetExceededError("estimated spend reached the limit")),
        ("down-then-bug", RuntimeError("harness bug")),
    ):
        failures = iter([ProviderError("embedding service down"), stop])

        def down_then_stop(query: str, failures: Any = failures) -> NDArray[np.float32]:
            raise next(failures)

        with pytest.raises(type(stop), match=str(stop)):
            mcp_search_run(tmp_path / name, down_then_stop, ("rainfall", "lidar"))


def test_a_query_embedding_failure_marks_only_the_run_it_happened_in() -> None:
    down = {"now": True}

    def flaky(query: str) -> NDArray[np.float32]:
        if down["now"]:
            raise ProviderError("embedding service down")
        return FakeGateway().embed_query(query)

    embedder = CountingEmbedder(flaky)

    def search(args: dict[str, Any]) -> str:
        # Like the MCP server, the tool hands the model an error instead of raising.
        try:
            embedder(str(args["query"]))
        except ProviderError as exc:
            return f"ERROR: {exc}"
        return "hits"

    def bug(args: dict[str, Any]) -> str:
        raise RuntimeError("harness bug")

    schema = {"type": "object", "properties": {}}
    tools = [Tool("search", "Search.", schema, search), Tool("bug", "Break.", schema, bug)]

    def run(*bodies: dict[str, Any]) -> AgentRun:
        api = scripted(*bodies)
        return anyio.run(
            lambda: run_agent(
                api.gateway(),
                operation="qa_probe",
                instructions="i",
                prompt="q",
                tools=tools,
                max_turns=4,
                embedder=embedder,
            )
        )

    # Outside any run, as in layer 1, a failure reaches its caller and is not kept.
    with pytest.raises(ProviderError):
        embedder("query outside a run")
    # A run that a harness failure stops never reads back its embedding failure.
    with pytest.raises(RuntimeError, match="harness bug"):
        run(
            response_body([function_call("c1", "search", '{"query": "rainfall"}')]),
            response_body([function_call("c2", "bug", "{}")]),
        )
    down["now"] = False
    later = run(
        response_body([function_call("c1", "search", '{"query": "rainfall"}')]),
        response_body([message("done")]),
    )
    assert (later.status, later.error, later.search_embed_calls) == ("completed", None, 1)


def test_layer2_does_not_judge_or_credit_unfinished_runs(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    cases = example_pack.qa_cases
    runs = 2 * len(cases)
    api = ScriptedOpenAI()
    api.script(
        "/responses",
        *[response_body([message("330.9 median")], "incomplete") for _ in range(runs)],
    )
    embedder = CountingEmbedder(FakeGateway().embed_query)
    result = anyio.run(
        run_layer2_qa, api.gateway(), snapshot_path, tmp_path / "bathymetry-corpus", embedder, cases
    )
    # One request per run: no judge request was made for an unfinished run.
    requests = api.requests("/responses")
    assert len(requests) == runs and api.request_count == runs
    # Like an MCP host, only the Arkive arm sees the server's initialize instructions.
    mcp_instructions = f"{QA_INSTRUCTIONS}\n\nMCP SERVER INSTRUCTIONS:\n{INSTRUCTIONS.strip()}"
    assert [r["instructions"] for r in requests] == [QA_INSTRUCTIONS, mcp_instructions] * len(cases)
    rows = cast(list[dict[str, Any]], result["rows"])
    assert [r["status"] for r in rows] == ["incomplete"] * runs
    assert not any(r["phrase_coverage"] or r["groups_matched"] or r["judge"] for r in rows)
    report = "\n".join(_layer2_report(result))
    n = len(cases)
    assert f"| raw_files | 0/{n} | 0/{n} | 0/{n} | n/a |" in report
    assert "harbour_median_rainfall / arkive_mcp: incomplete" in report


# A three-page report whose second section spans pages 2 and 3.
DREDGE_REPORT: dict[str, Sequence[Sequence[str]]] = {
    "p104-dredge/report.pdf": [["Intro text one"], ["# Results", "Volume 42 m3"], ["Appendix"]]
}


def qa_case(case_id: str, anchors: list[dict[str, Any]], kind: str = "lookup") -> QACase:
    return QACase.model_validate(
        {"case_id": case_id, "kind": kind, "question": "q", "ground_truth_facts": ["f"]}
        | {"required_groups": [["f"]], "source_anchors": anchors}
    )


def test_qa_evidence_labels_the_sections_each_anchor_covers_and_refuses_oversized_evidence(
    tmp_path: Path,
) -> None:
    long_log = "# Log\n\n" + "Sounding checked. " * 3000
    # Page 2 of the gap report is blank, so no section overlaps it.
    gap_report = {"p106-gap/report.pdf": [["# Intro", "Intro text"], [], ["# End", "Closing"]]}
    snapshot_path = compile_snapshot(
        tmp_path,
        BATHYMETRY_CORPUS | {"p105-archive/log.md": long_log},
        pdfs=DREDGE_REPORT | gap_report,
    )
    page = {"doc_id": "doc_004", "pages": [3, 3]}
    whole = {"doc_id": "doc_003", "pages": None}
    with TopicSnapshot.open(snapshot_path) as snapshot:
        assert [d.rel_path for d in snapshot.list_documents()][3:] == [
            "p104-dredge/report.pdf",
            "p105-archive/log.md",
            "p106-gap/report.pdf",
        ]
        evidence = qa_evidence(snapshot, qa_case("c", [page, whole, page]))
        too_long = qa_case("too_long", [{"doc_id": "doc_005", "pages": None}])
        with pytest.raises(ValueError, match=r"'too_long': the evidence .* over the 48000"):
            qa_evidence(snapshot, too_long)
        assert qa_evidence(snapshot, qa_case("none", [], kind="abstention")) == (
            "(no source anchors listed)"
        )
        blank = {"doc_id": "doc_006", "pages": [2, 2]}
        with pytest.raises(
            ValueError,
            match=r"'gap': source anchor doc_006 pages \[2, 2\]: no section of"
            r" p106-gap/report.pdf overlaps it",
        ):
            qa_evidence(snapshot, qa_case("gap", [whole, blank]))
        assert qa_evidence(snapshot, qa_case("absent", [blank], kind="abstention")) == (
            "(no section overlaps the source anchors)"
        )
    # Only the report section that overlaps page 3 is evidence, and once although anchored twice.
    assert evidence == (
        "--- p104-dredge/report.pdf (doc_004), section doc_004_sec_002, pages 2-3 ---\n"
        "# Results\n\nVolume 42 m3\n\nAppendix"
        "\n\n--- p103-reef-sonar/AGENTS.md (doc_003), section doc_003_sec_001, pages 1-1 ---\n"
        "# Status\n\nThe reef Sentinel-2 case study has not been written yet."
    )


def test_eval_cli_refuses_a_qa_case_whose_evidence_is_over_the_judge_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    long_log = "# Log\n\n" + "Sounding checked. " * 3000
    snapshot_path = compile_snapshot(
        tmp_path, BATHYMETRY_CORPUS | {"p105-archive/log.md": long_log}
    )
    pack = tmp_path / "long.eval.json"
    anchors = [{"doc_id": "doc_004", "pages": None}]
    pack.write_text(json.dumps(pack_with(qa_cases__1__source_anchors=anchors)))
    out = tmp_path / "out"
    arguments = ["--pack", str(pack), "--snapshot", str(snapshot_path), "--out", str(out)]
    arguments += ["--corpus", str(tmp_path / "bathymetry-corpus"), "--layer", "2"]

    result = CliRunner().invoke(eval_app, arguments)

    assert result.exit_code == 2, result.output
    message = " ".join(result.output.replace("\u2502", " ").split())
    assert "qa case 'reef_case_study_status': the evidence of its source anchors is" in message
    assert "over the 48000 characters the judge accepts" in message
    assert not out.exists()


def test_layer2_judges_answers_against_labelled_anchor_evidence_and_counts_claims(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS, pdfs=DREDGE_REPORT)
    paged = qa_case("dredge_volume", [{"doc_id": "doc_004", "pages": [2, 2]}], kind="aggregation")
    cases = (*example_pack.qa_cases, paged)
    verdict = QAJudgeScore(
        factual_accuracy_1_to_5=4,
        citation_quality_1_to_5=3,
        completeness_1_to_5=5,
        claims=[
            JudgedClaim(claim="The median is 330.9 mm.", status="supported"),
            JudgedClaim(claim="It rained in 2019.", status="unverifiable"),
            JudgedClaim(claim="The median is 307.7 mm.", status="contradicted"),
        ],
        rationale="r",
    )
    api = ScriptedOpenAI()
    for _ in range(2 * len(cases)):
        api.script("/responses", response_body([message("330.9 median")]), structured_body(verdict))
    embedder = CountingEmbedder(FakeGateway().embed_query)

    result = anyio.run(
        run_layer2_qa, api.gateway(), snapshot_path, tmp_path / "bathymetry-corpus", embedder, cases
    )

    judged = [r["input"][0]["content"][0]["text"] for r in api.requests("/responses")[1::2]]
    whole_document, curator_note, abstention, page_range = judged[::2]
    assert (
        "EVIDENCE:\n--- p101-harbour-lidar/harbour-memo.md (doc_001), section doc_001_sec_001,"
        " pages 1-1 ---\n# Rainfall" in whole_document
    )
    assert "section doc_001_sec_003, pages 1-1 ---\n# Bathymetry" in whole_document
    assert "SOURCE ANCHORS:\n- doc_001, whole document\n" in whole_document
    assert "(doc_003), section doc_003_sec_001" in curator_note
    assert "SOURCE ANCHORS:\n(none listed)\n\nEVIDENCE:\n(no source anchors listed)" in abstention
    assert "- doc_004, pages 2-2\n" in page_range
    assert "--- p104-dredge/report.pdf (doc_004), section doc_004_sec_002, pages 2-3 ---" in (
        page_range
    )
    assert "doc_004_sec_001" not in page_range
    rows = cast(list[dict[str, Any]], result["rows"])
    assert [r["kind"] for r in rows[::2]] == ["lookup", "lookup", "abstention", "aggregation"]
    counts = {"supported": 1, "contradicted": 1, "unverifiable": 1}
    assert all(r["claim_counts"] == counts for r in rows)
    report = "\n".join(_layer2_report(result))
    # Contradicted and unverifiable claims per arm, and per kind and arm.
    assert "| raw_files | 4/4 | 4/4 | 1/4 | 4.00 | 3.00 | 5.00 | 4 | 4 |" in report
    assert "| abstention | arkive_mcp | 1/1 | 1/1 | 0/1 | 4.00 | 3.00 | 5.00 | 1 | 1 |" in report
    assert "| lookup | raw_files | 2/2 | 2/2 | 1/2 | 4.00 | 3.00 | 5.00 | 2 | 2 |" in report


def test_a_failed_qa_judge_call_leaves_its_row_unjudged_with_its_phrase_coverage(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    lookup = example_pack.qa_cases[0]
    verdict = QAJudgeScore(
        factual_accuracy_1_to_5=4,
        citation_quality_1_to_5=3,
        completeness_1_to_5=5,
        claims=[JudgedClaim(claim="The median is 330.9 mm.", status="supported")],
        rationale="r",
    )
    answer = response_body([message("The nine year median is 330.9 mm.")])
    api = ScriptedOpenAI()
    api.script(
        "/responses",
        answer,
        invalid_judge_body(verdict, factual_accuracy_1_to_5=0),
        answer,
        refusal_body(),
        answer,
        structured_body(verdict),
        answer,
        structured_body(verdict),
    )
    embedder = CountingEmbedder(FakeGateway().embed_query)

    result = anyio.run(
        run_layer2_qa,
        api.gateway(),
        snapshot_path,
        tmp_path / "bathymetry-corpus",
        embedder,
        [lookup, lookup.model_copy(update={"case_id": "again"})],
    )

    assert not api.scripts["/responses"]
    rows = cast(list[dict[str, Any]], result["rows"])
    assert [r["judge"] is None for r in rows] == [True, True, False, False]
    assert "returned invalid output" in rows[0]["judge_missing"]
    assert "returned no parsed output" in rows[1]["judge_missing"]
    assert [r["judge_missing"] for r in rows[2:]] == [None, None]
    assert all(r["phrase_coverage"] for r in rows) and rows[0]["claim_counts"] is None
    report = "\n".join(_layer2_report(result))
    # One judged row per arm: the judge averages cover it alone.
    assert "| raw_files | 2/2 | 1/2 | 2/2 | 4.00 | 3.00 | 5.00 | 0 | 0 |" in report
    assert "Completed runs not judged:" in report
    assert f"- {lookup.case_id} / arkive_mcp: the judge call failed:" in report


def test_layer2_report_shows_no_claim_counts_for_an_arm_with_no_judged_row(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    answer = response_body([message("The nine year median is 330.9 mm.")])
    api = ScriptedOpenAI()
    api.script("/responses", answer, refusal_body(), answer, refusal_body())
    embedder = CountingEmbedder(FakeGateway().embed_query)

    result = anyio.run(
        run_layer2_qa,
        api.gateway(),
        snapshot_path,
        tmp_path / "bathymetry-corpus",
        embedder,
        [example_pack.qa_cases[0]],
    )

    report = "\n".join(_layer2_report(result))
    # Zero contradicted claims would claim a judgement no row received.
    for arm in ("raw_files", "arkive_mcp"):
        assert f"| {arm} | 1/1 | 0/1 | 1/1 | n/a | n/a | n/a | n/a | n/a |" in report
        assert f"| lookup | {arm} | 1/1 | 0/1 | 1/1 | n/a | n/a | n/a | n/a | n/a |" in report


# The reference and the holdout share HOUSE; NOTE sits in the holdout only in a curator note.
NOTE = "Every reef sonar report names the tide gauge used for each sounding line of the survey."
DRAFTING_FILES = {
    "p101-harbour-lidar/harbour-memo.md": (
        f"# Harbour memo\n\n## About Northwind Marine\n\n{HOUSE}\n\n## Rainfall\n\n"
        f"{visual('Chart', 'Nine Year Median = 330.9 mm')}\n\nThe harbour grid is one metre.\n"
    ),
    "p103-reef-sonar/AGENTS.md": f"# Reef notes\n\n{NOTE}\n",
    "p102-estuary-multibeam/estuary-proposal.md": (
        f"# Estuary proposal\n\n## About Northwind Marine\n\n{HOUSE}\n\n## Reporting\n\n"
        f"{NOTE}\n\n{visual('Map', 'Survey boundary with mooring buoys')}\n"
    ),
}


def candidate_scores(label: str, score: int) -> CandidateAssessment:
    return CandidateAssessment(
        candidate=label,
        structure_and_house_style=score,
        technical_rigor=score,
        limitations_and_review_readiness=score,
        factual_discipline_no_contamination=score,
        editing_effort_saved=score,
        strengths=[],
        defects=[],
    )


def prompt_text(request: dict[str, Any]) -> str:
    """The text of a request's first input message."""
    first = request["input"] if isinstance(request["input"], str) else request["input"][0]
    if isinstance(first, str):
        return first
    content = first["content"]
    return content if isinstance(content, str) else str(content[0]["text"])


def drafting_holdout(tmp_path: Path, task: Holdout) -> tuple[Path, Path, Holdout]:
    corpus = write_corpus(tmp_path / "corpus", dict(DRAFTING_FILES))
    full = compile_snapshot(tmp_path, DRAFTING_FILES)
    with TopicSnapshot.open(full) as snapshot:
        reference_id = next(
            d.doc_id for d in snapshot.list_documents() if d.rel_path.startswith("p102")
        )
    return corpus, full, task.model_copy(update={"reference_doc_id": reference_id})


def test_layer3_runs_every_arm_in_every_round_and_judges_each_round_once(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    corpus, full, task = drafting_holdout(tmp_path, holdout_task(example_pack))
    out = tmp_path / "out"
    out.mkdir()
    api = ScriptedOpenAI()
    partial = "Northwind Marine reduces every sounding to the stated vertical datum."
    zero_shot_drafts = [f"# Proposal\n\n{HOUSE}", partial, "The estuary grid keeps every sounding."]
    for number, zero_shot in enumerate(zero_shot_drafts, start=1):
        arkive = (
            response_body([message("cut")], "incomplete")
            if number == 1
            else response_body([message("# Arkive proposal for the estuary survey")])
        )
        api.script(
            "/responses",
            response_body([message(zero_shot)]),
            response_body([message("# Chunk proposal for the estuary survey")]),
            arkive,
            response_body([message("# Exemplar proposal for the estuary survey")]),
            # Round 1 has three drafts, so this output names an unknown label there and its
            # ranking leaves labels out: that round is not judged.
            structured_body(
                DraftBlindComparison(
                    candidates=[
                        candidate_scores(f"Candidate_{i}", 2 * number) for i in (1, 2, 3, 4)
                    ],
                    ranking=["Candidate_1"]
                    if number == 1
                    else [f"Candidate_{i}" for i in (4, 3, 2, 1)],
                )
            ),
        )
    holdout_gateway = FakeGateway()

    result = anyio.run(
        lambda: run_layer3_holdout(
            api.gateway(),
            holdout_gateway=holdout_gateway,
            full_snapshot_path=full,
            corpus_dir=corpus,
            embedder=CountingEmbedder(FakeGateway().embed_query),
            output_dir=out,
            holdout=task,
            drafts=3,
        )
    )

    assert holdout_gateway.calls and (out / "holdout.arkive.db").is_file()
    assert not api.scripts["/responses"]
    rounds = cast(list[dict[str, Any]], result["rounds"])
    arms = ["zero_shot", "chunk_search", "arkive_mcp", "exemplars_in_context"]
    assert [list(r["arms"]) for r in rounds] == [arms] * 3
    assert result["skipped_arms"] == {}
    # The curator note's NOTE is not holdout text, so HOUSE is the only house unit.
    assert (result["house_units"], result["house_shingles"]) == (1, 12)
    first = rounds[0]["arms"]
    assert first["arkive_mcp"]["run"].status == "incomplete"
    assert first["arkive_mcp"]["rubric"] is None and first["arkive_mcp"]["overlap"] is None
    assert sorted(rounds[0]["blind_key"].values()) == sorted(
        ["zero_shot", "chunk_search", "exemplars_in_context"]
    )
    recall = [r["arms"]["zero_shot"]["overlap"].house_text_recall for r in rounds]
    assert recall == [1.0, pytest.approx(7 / 12), 0.0]
    assert all(r["arms"]["zero_shot"]["rubric"] is not None for r in rounds)
    # The unjudged round keeps its drafts and measures and says why it has no comparison.
    assert rounds[0]["blind_comparison"] is None and len(rounds[0]["blind_drafts"]) == 3
    assert rounds[0]["blind_comparison_missing"] == (
        "the judge output does not cover the round: candidates unknown ['Candidate_4'];"
        " ranking missing ['Candidate_2', 'Candidate_3']"
    )
    assert [r["blind_comparison_missing"] for r in rounds[1:]] == [None, None]

    requests = api.requests("/responses")
    judged = [r for r in requests if r.get("text", {}).get("format", {}).get("name")]
    assert len(judged) == 3
    assert "figures, charts and maps are out of scope" in judged[0]["instructions"]
    reference_text = prompt_text(judged[0])
    assert "## Reporting" in reference_text and "Survey boundary" not in reference_text
    exemplar_requests = [r for r in requests if EXEMPLAR_DRAFTING_RULES in r["instructions"]]
    assert len(exemplar_requests) == 3
    exemplar_prompt = prompt_text(exemplar_requests[0])
    assert exemplar_prompt.startswith("===== DOCUMENT p101-harbour-lidar/harbour-memo.md =====")
    assert HOUSE in exemplar_prompt and "330.9" not in exemplar_prompt
    assert exemplar_prompt.endswith(f"RUN NOTES:\n{task.run_notes}")
    assert exemplar_requests[0]["instructions"].startswith(task.instructions)

    report = "\n".join(_layer3_report(result))
    # Median [minimum, maximum] (n) over the rounds; arkive_mcp completed rounds 2 and 3 only,
    # and only rounds 2 and 3 were judged.
    assert "| zero_shot | 3/3 | 0.583 [0.000, 1.000] (n=3) |" in report
    assert "| arkive_mcp | 2/3 | 0.000 [0.000, 0.000] (n=2) |" in report
    assert "| zero_shot | 2/3 | 5.0 [4.0, 6.0] (n=2) | 5.0 [4.0, 6.0] (n=2) |" in report
    assert "| arkive_mcp | 2/3 | 5.0 [4.0, 6.0] (n=2) |" in report
    assert "- round 1: not judged: the judge output does not cover the round" in report
    assert "their tokens and tool calls are counted" in report
    assert "| scope_and_budget | 0/3 | 0/3 | 0/2 | 0/3 |" in report
    assert "- round 1 / arkive_mcp: incomplete" in report
    assert (
        "- `no_borrowed_masking_claim`: required phrase found in no supplied input:"
        " ['no masking applied']" in report
    )
    assert "`deliverables_and_area_kept`" not in report


def test_layer3_skips_the_exemplars_arm_and_says_so_when_the_holdout_lists_no_exemplars(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    task = holdout_task(example_pack).model_copy(update={"exemplar_paths": (), "rubric": ()})
    corpus, full, task = drafting_holdout(tmp_path, task)
    out = tmp_path / "out"
    out.mkdir()
    api = ScriptedOpenAI()
    api.script(
        "/responses",
        *[response_body([message(f"# Draft {i} of the estuary proposal")]) for i in range(3)],
        structured_body(DraftBlindComparison(candidates=[], ranking=[])),
    )

    result = anyio.run(
        lambda: run_layer3_holdout(
            api.gateway(),
            holdout_gateway=FakeGateway(),
            full_snapshot_path=full,
            corpus_dir=corpus,
            embedder=CountingEmbedder(FakeGateway().embed_query),
            output_dir=out,
            holdout=task,
            drafts=1,
        )
    )

    assert not api.scripts["/responses"]
    (only_round,) = cast(list[dict[str, Any]], result["rounds"])
    assert list(only_round["arms"]) == ["zero_shot", "chunk_search", "arkive_mcp"]
    assert result["skipped_arms"] == {"exemplars_in_context": "the holdout lists no exemplar_paths"}
    report = "\n".join(_layer3_report(result))
    assert "exemplars_in_context was not run: the holdout lists no exemplar_paths." in report
    assert "The holdout has no rubric checks." in report


def test_layer3_reads_an_empty_curator_note_among_the_included_files_as_no_sections(
    tmp_path: Path,
) -> None:
    files = {
        "AGENTS.md": "",
        "p101/memo.md": f"# Memo\n\n{HOUSE}\n",
        "p102/proposal.md": f"# Proposal\n\n{HOUSE}\n\nThe estuary grid keeps every sounding.\n",
    }
    corpus = write_corpus(tmp_path / "corpus", dict(files))
    full = compile_snapshot(tmp_path, files)
    with TopicSnapshot.open(full) as snapshot:
        documents = {d.rel_path: d for d in snapshot.list_documents()}
    assert documents["AGENTS.md"].is_curator_note
    task = Holdout.model_validate(
        {
            "holdout_id": "proposal",
            "included_paths": ["AGENTS.md", "p101"],
            "held_out_markers": ["p102"],
            "reference_doc_id": documents["p102/proposal.md"].doc_id,
            "instructions": "Write the proposal.",
            "run_notes": "Area: 812 km2.",
            "exemplar_paths": ["p101/memo.md"],
        }
    )
    api = ScriptedOpenAI()
    api.script(
        "/responses",
        *[response_body([message(f"# Draft {i}\n\n{HOUSE}")]) for i in range(4)],
        structured_body(DraftBlindComparison(candidates=[], ranking=[])),
    )
    out = tmp_path / "out"
    out.mkdir()

    result = anyio.run(
        lambda: run_layer3_holdout(
            api.gateway(),
            holdout_gateway=FakeGateway(),
            full_snapshot_path=full,
            corpus_dir=corpus,
            embedder=CountingEmbedder(FakeGateway().embed_query),
            output_dir=out,
            holdout=task,
            drafts=1,
        )
    )

    assert result["holdout_documents"] == ["AGENTS.md", "p101/memo.md"]
    (only_round,) = cast(list[dict[str, Any]], result["rounds"])
    assert only_round["arms"]["zero_shot"]["overlap"].house_text_recall == 1.0


def invalid_judge_body(output: BaseModel, **invalid: Any) -> dict[str, Any]:
    """A completed response whose structured output breaks its schema with `invalid`."""
    payload = output.model_dump()
    for path, value in invalid.items():
        *parents, leaf = path.split("__")
        target = payload
        for key in parents:
            target = target[int(key)] if isinstance(target, list) else target[key]
        target[leaf] = value
    return response_body([message(json.dumps(payload))])


def test_a_failed_blind_judge_call_is_recorded_on_its_round_and_the_spend_limit_still_stops(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    task = holdout_task(example_pack).model_copy(update={"exemplar_paths": (), "rubric": ()})
    corpus, full, task = drafting_holdout(tmp_path, task)
    labels = [f"Candidate_{i}" for i in (1, 2, 3)]
    comparison = DraftBlindComparison(
        candidates=[candidate_scores(label, 5) for label in labels], ranking=labels
    )
    drafts = [response_body([message(f"# Draft {i}\n\n{HOUSE}")]) for i in range(3)]
    api = ScriptedOpenAI()
    api.script(
        "/responses", *drafts, invalid_judge_body(comparison, candidates__0__editing_effort_saved=0)
    )
    api.script("/responses", *drafts, refusal_body())

    def layer3(gateway: Any, out: Path, rounds: int) -> dict[str, object]:
        out.mkdir()
        return anyio.run(
            lambda: run_layer3_holdout(
                gateway,
                holdout_gateway=FakeGateway(),
                full_snapshot_path=full,
                corpus_dir=corpus,
                embedder=CountingEmbedder(FakeGateway().embed_query),
                output_dir=out,
                holdout=task,
                drafts=rounds,
            )
        )

    result = layer3(api.gateway(), tmp_path / "out", 2)

    assert not api.scripts["/responses"]
    rounds = cast(list[dict[str, Any]], result["rounds"])
    # Each failed judge call is that round's outcome: its drafts and measures stay.
    missing = [r["blind_comparison_missing"] for r in rounds]
    assert missing[0].startswith("the judge call failed: judge_drafts on gpt-6.1-sol returned")
    assert "invalid output" in missing[0] and "no parsed output" in missing[1]
    for draft_round in rounds:
        assert draft_round["blind_comparison"] is None and len(draft_round["blind_drafts"]) == 3
        assert draft_round["arms"]["zero_shot"]["overlap"].house_text_recall == 1.0
    report = "\n".join(_layer3_report(result))
    assert "| zero_shot | 0/2 | n/a | n/a | n/a | n/a | n/a |" in report
    assert "- round 2: not judged: the judge call failed:" in report

    over_budget = api.gateway()

    def spend_limit_reached(request: object) -> None:
        raise BudgetExceededError("estimated spend reached the limit")

    over_budget.run = spend_limit_reached  # type: ignore[method-assign, assignment]
    api.script("/responses", *drafts)
    with pytest.raises(BudgetExceededError):
        layer3(over_budget, tmp_path / "over-budget", 1)


def test_eval_runner_writes_tool_calls_claims_and_per_holdout_outputs_for_the_example_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, example_pack: EvalPack
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    verdict = QAJudgeScore(
        factual_accuracy_1_to_5=5,
        citation_quality_1_to_5=4,
        completeness_1_to_5=5,
        claims=[
            JudgedClaim(claim="The median is 330.9 mm.", status="supported"),
            JudgedClaim(claim="The memo is from 2019.", status="unverifiable"),
        ],
        rationale="r",
    )
    api = ScriptedOpenAI()
    grep = json.dumps({"topic": "bathymetry", "pattern": "330.9"})
    for _ in example_pack.qa_cases:
        api.script(
            "/responses",
            response_body([function_call("c1", "list_corpus_files", "{}")]),
            response_body([message("The median is 330.9 mm (harbour-memo.md).")]),
            structured_body(verdict),
            response_body([function_call("c1", "grep", grep)]),
            response_body([message("The median is 330.9 mm (doc_001).")]),
            structured_body(verdict),
        )
    copied = "The chart shows the Nine Year Median = 330.9 mm reference line for 2016"
    labels = [f"Candidate_{i}" for i in (1, 2, 3, 4)]
    for _ in range(2):
        api.script(
            "/responses",
            response_body([message("# Zero-shot proposal for the estuary survey")]),
            response_body([function_call("c1", "search_chunks", '{"query": "scope"}')]),
            response_body([message("# Chunk proposal for the estuary survey")]),
            response_body([message("# Arkive proposal for the estuary survey")]),
            response_body([message(f"# Exemplar proposal\n\n{copied}.")]),
            structured_body(
                DraftBlindComparison(
                    candidates=[candidate_scores(label, 5) for label in labels], ranking=labels
                )
            ),
        )

    def scripted_gateway(**options: Any) -> Any:
        # The holdout build runs on the fake compiler gateway; agents and judges are scripted.
        if "cache" in options:
            return FakeGateway()
        return api.gateway(**options)

    monkeypatch.setattr(eval_run, "OpenAIGateway", scripted_gateway)
    monkeypatch.setattr(
        eval_run, "_embedder", lambda *args: CountingEmbedder(FakeGateway().embed_query)
    )
    out = tmp_path / "out"
    arguments = ["--pack", str(EXAMPLE_PACK_PATH), "--snapshot", str(snapshot_path)]
    arguments += ["--out", str(out), "--corpus", str(tmp_path / "bathymetry-corpus")]

    result = CliRunner().invoke(eval_app, [*arguments, "--drafts", "2"])

    assert result.exit_code == 0, result.output
    assert not api.scripts["/responses"]
    results = json.loads((out / "results.json").read_text())
    qa_rows = results["layer2"]["rows"]
    assert [r["kind"] for r in qa_rows[::2]] == ["lookup", "lookup", "abstention"]
    raw_call, mcp_call = qa_rows[0]["run"]["tool_calls"], qa_rows[1]["run"]["tool_calls"]
    assert [(c["name"], c["arguments"], c["images"]) for c in raw_call + mcp_call] == [
        ("list_corpus_files", "{}", 0),
        ("grep", grep, 0),
    ]
    assert mcp_call[0]["result_chars"] > 200 and len(mcp_call[0]["result_head"]) == 200
    assert mcp_call[0]["result_head"].startswith('{\n  "doc_id": "doc_001"')
    assert qa_rows[0]["judge"]["claims"][1] == {
        "claim": "The memo is from 2019.",
        "status": "unverifiable",
    }
    assert all(
        r["claim_counts"] == {"supported": 1, "contradicted": 0, "unverifiable": 1} for r in qa_rows
    )
    holdout = results["layer3"]["estuary-proposal"]
    holdout_dir = out / "holdouts" / "estuary-proposal"
    assert holdout["holdout_snapshot"] == str(holdout_dir / "holdout.arkive.db")
    assert (holdout_dir / "holdout.arkive.db").is_file()
    assert holdout["exemplar_paths"] == ["p101-harbour-lidar/harbour-memo.md"]
    rounds = holdout["rounds"]
    arms = ["zero_shot", "chunk_search", "arkive_mcp", "exemplars_in_context"]
    assert [list(r["arms"]) for r in rounds] == [arms] * 2
    assert [c["name"] for c in rounds[0]["arms"]["chunk_search"]["run"]["tool_calls"]] == [
        "search_chunks"
    ]
    # Both measures for every arm and round: the example proposal shares no house paragraph
    # with the included files, so recall is null; the exemplars draft copies a memo sentence,
    # and the other drafts are a heading alone, which has no prose shingles.
    assert holdout["house_units"] == 0 and holdout["house_shingles"] == 0
    for draft_round in rounds:
        for arm, cell in draft_round["arms"].items():
            overlap = cell["overlap"]
            assert overlap["house_text_recall"] is None, arm
            if arm == "exemplars_in_context":
                assert overlap["copied_text_contamination"] > 0
            else:
                assert overlap["copied_text_contamination"] is None, arm
    exemplar = rounds[1]["arms"]["exemplars_in_context"]["overlap"]
    assert exemplar["copied_units"] == [{"text_head": f"{copied}.", "stray_shingles": 11}]
    assert [d["check_id"] for d in holdout["rubric_diagnostics"]] == [
        c.check_id for c in holdout_task(example_pack).rubric
    ]
    blind_dir = holdout_dir / "blind-drafts"
    assert sorted(p.name for p in blind_dir.iterdir()) == ["round-1", "round-2"]
    assert sorted(p.name for p in (blind_dir / "round-2").iterdir()) == [
        "Candidate_1.md",
        "Candidate_2.md",
        "Candidate_3.md",
        "Candidate_4.md",
        "blind_key.json",
    ]
    report = (out / "report.md").read_text()
    assert "## Layer 3: holdout `estuary-proposal`" in report
    assert "2 draft rounds" in report
    assert "| raw_files | list_corpus_files | 3 |" in report
    assert "| arkive_mcp | grep | 3 |" in report
    assert "| chunk_search | search_chunks | 2 |" in report
    assert "Blind drafts: `holdouts/estuary-proposal/blind-drafts/round-<n>/`" in report


def drafting_round_script(api: ScriptedOpenAI) -> None:
    """One layer 3 round of the example pack: four drafts and a judge output covering them."""
    labels = [f"Candidate_{i}" for i in (1, 2, 3, 4)]
    api.script(
        "/responses",
        *[response_body([message(f"# Draft {i} of the estuary proposal")]) for i in range(4)],
        structured_body(
            DraftBlindComparison(
                candidates=[candidate_scores(label, 5) for label in labels], ranking=labels
            )
        ),
    )


def scripted_eval_gateways(api: ScriptedOpenAI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Holdout builds run on the fake compiler gateway; agents and judges are scripted."""

    def scripted_gateway(**options: Any) -> Any:
        return FakeGateway() if "cache" in options else api.gateway(**options)

    monkeypatch.setattr(eval_run, "OpenAIGateway", scripted_gateway)
    monkeypatch.setattr(
        eval_run, "_embedder", lambda *args: CountingEmbedder(FakeGateway().embed_query)
    )


def test_eval_runner_runs_each_holdout_into_its_own_folder_and_checks_all_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    second = EXAMPLE_HOLDOUT | {"holdout_id": "estuary-proposal-again"}
    pack = tmp_path / "two.eval.json"
    pack.write_text(json.dumps(pack_with(holdouts=[EXAMPLE_HOLDOUT, second])))
    api = ScriptedOpenAI()
    scripted_eval_gateways(api, monkeypatch)
    runner = CliRunner()

    def run(pack: Path, out: Path, *extra: str) -> Any:
        arguments = ["--pack", str(pack), "--snapshot", str(snapshot_path), "--out", str(out)]
        arguments += ["--corpus", str(tmp_path / "bathymetry-corpus"), "--layer", "3"]
        return runner.invoke(eval_app, [*arguments, *extra])

    drafting_round_script(api)
    drafting_round_script(api)
    both = run(pack, tmp_path / "both")
    assert both.exit_code == 0, both.output
    assert not api.scripts["/responses"]
    results = json.loads((tmp_path / "both" / "results.json").read_text())
    assert list(results["layer3"]) == ["estuary-proposal", "estuary-proposal-again"]
    for holdout_id in results["layer3"]:
        holdout_dir = tmp_path / "both" / "holdouts" / holdout_id
        assert (holdout_dir / "holdout.arkive.db").is_file()
        assert (holdout_dir / "blind-drafts" / "round-1" / "blind_key.json").is_file()

    drafting_round_script(api)
    selected = run(pack, tmp_path / "second", "--holdout", "estuary-proposal-again")
    assert selected.exit_code == 0, selected.output
    assert [p.name for p in (tmp_path / "second" / "holdouts").iterdir()] == [
        "estuary-proposal-again"
    ]

    requests = api.request_count
    bad_second = tmp_path / "bad.eval.json"
    broken = second | {"included_paths": ["p101-harbour-lidar", "p104"]}
    bad_second.write_text(json.dumps(pack_with(holdouts=[EXAMPLE_HOLDOUT, broken])))
    refused = run(bad_second, tmp_path / "refused")
    assert refused.exit_code == 2
    assert "holdout 'estuary-proposal-again'" in " ".join(refused.output.split())
    assert api.request_count == requests and not (tmp_path / "refused").exists()


def test_eval_runner_checks_the_credential_and_opens_the_cache_before_creating_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    out = tmp_path / "out"
    arguments = ["--pack", str(EXAMPLE_PACK_PATH), "--snapshot", str(snapshot_path)]
    arguments += ["--out", str(out), "--corpus", str(tmp_path / "bathymetry-corpus")]
    runner = CliRunner()

    no_key = runner.invoke(eval_app, [*arguments, "--layer", "1"])
    assert no_key.exit_code == 1 and "OPENAI_API_KEY is not set" in no_key.stderr
    assert not out.exists()

    api = ScriptedOpenAI()
    scripted_eval_gateways(api, monkeypatch)
    # Preflight cannot tell that a writable regular file is no SQLite database; opening does.
    garbage = tmp_path / "garbage.db"
    garbage.write_text("not a database")
    unopenable = runner.invoke(eval_app, [*arguments, "--cache", str(garbage)])
    assert unopenable.exit_code == 1 and "cannot open build cache" in unopenable.stderr
    assert api.request_count == 0 and not out.exists()


def snapshot_with(
    root: Path, files: dict[str, str], word_files: Sequence[str], monkeypatch: pytest.MonkeyPatch
) -> Path:
    """`compile_snapshot` of `files` plus Word files, each converted to a one-page PDF without
    LibreOffice."""
    corpus = root / "bathymetry-corpus"
    corpus.mkdir(parents=True)
    for rel_path in word_files:
        (corpus / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rel_path).write_bytes(b"word document bytes")
    pdf = write_pdf(root / "converted.pdf", [["Converted Word notes"]]).read_bytes()
    with monkeypatch.context() as patched:
        patched.setattr(compiler, "convert_office_to_pdf", lambda *args: pdf)
        return compile_snapshot(root, files, gateway=FakeGateway())


def test_eval_cli_refuses_outputs_it_cannot_write_and_inputs_it_cannot_convert_or_draft_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    api = ScriptedOpenAI()
    scripted_eval_gateways(api, monkeypatch)
    runner = CliRunner()

    def tree() -> list[Path]:
        return sorted(tmp_path.rglob("*"))

    def refused(snapshot_path: Path, out: Path, *extra: str, pack: Path = EXAMPLE_PACK_PATH) -> str:
        corpus = snapshot_path.parent / "bathymetry-corpus"
        before = tree()
        arguments = ["--pack", str(pack), "--snapshot", str(snapshot_path), "--out", str(out)]
        result = runner.invoke(eval_app, [*arguments, "--corpus", str(corpus), *extra])
        assert result.exit_code == 2, result.output
        assert api.request_count == 0 and tree() == before
        return " ".join(result.output.replace("\u2502", " ").split())

    snapshot_path = compile_snapshot(tmp_path / "plain", BATHYMETRY_CORPUS)
    nested = tmp_path / "nested"
    (nested / "report.md").mkdir(parents=True)
    assert "report.md is a directory" in refused(snapshot_path, nested, "--layer", "1")
    a_file = tmp_path / "a-file"
    a_file.write_text("not a directory")
    cache = tmp_path / "cache.db"
    assert "exists and is not a directory" in refused(
        snapshot_path, a_file, "--layer", "3", "--cache", str(cache)
    )
    assert "a-file exists and is not a directory" in refused(
        snapshot_path, a_file / "out", "--layer", "1"
    )
    layer3 = ("--layer", "3")
    # An absent cache below a file cannot be created.
    assert "--cache" in refused(
        snapshot_path, tmp_path / "o", *layer3, "--cache", str(a_file / "cache.db")
    )
    # A symlinked cache is refused, so the build never writes through it.
    linked_cache = tmp_path / "linked-cache.db"
    linked_cache.symlink_to(a_file)
    assert "is not a regular file" in refused(
        snapshot_path, tmp_path / "o", *layer3, "--cache", str(linked_cache)
    )
    if os.geteuid() != 0:
        # The evaluator must be able to create an absent --out and use an existing --cache.
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        assert "is not writable" in refused(snapshot_path, locked, "--layer", "1")
        assert "cannot be created: " in refused(snapshot_path, locked / "out", "--layer", "1")
        read_only_cache = tmp_path / "read-only.db"
        read_only_cache.write_bytes(b"")
        read_only_cache.chmod(0o400)
        assert "is not readable and writable" in refused(
            snapshot_path, tmp_path / "o", *layer3, "--cache", str(read_only_cache)
        )
        sealed = tmp_path / "sealed"
        sealed.mkdir()
        (sealed / "cache.db").write_bytes(b"")
        sealed.chmod(0o500)
        assert "SQLite writes -wal and -shm files" in refused(
            snapshot_path, tmp_path / "o", *layer3, "--cache", str(sealed / "cache.db")
        )
        assert "--cache" in refused(
            snapshot_path, tmp_path / "o", *layer3, "--cache", str(locked / "cache.db")
        )
        sealed.chmod(0o700)
        # An absent --out below a writable directory passes, with an absent or an existing
        # writable cache.
        writable_cache = tmp_path / "writable.db"
        writable_cache.write_bytes(b"")
        for usable in (cache, writable_cache):
            layers, _ = eval_run._check_inputs(
                EvalPack.load(EXAMPLE_PACK_PATH),
                "3",
                None,
                EXAMPLE_PACK_PATH,
                snapshot_path,
                snapshot_path.parent / "bathymetry-corpus",
                tmp_path / "fresh" / "out",
                usable,
            )
            assert layers == {"3"}
        assert not (tmp_path / "fresh").exists() and not cache.exists()

    # A Word file a selected layer converts needs LibreOffice on the path.
    word = snapshot_with(
        tmp_path / "word", BATHYMETRY_CORPUS, ["p104-dredge/notes.docx"], monkeypatch
    )
    included = snapshot_with(
        tmp_path / "included", BATHYMETRY_CORPUS, ["p101-harbour-lidar/notes.docx"], monkeypatch
    )
    with TopicSnapshot.open(included) as snapshot:
        (reference_id,) = [
            d.doc_id for d in snapshot.list_documents() if d.rel_path.startswith("p102")
        ]
    moved = tmp_path / "moved.eval.json"
    moved.write_text(json.dumps(pack_with(holdouts__0__reference_doc_id=reference_id)))
    with monkeypatch.context() as without_soffice:
        without_soffice.setattr(shutil, "which", lambda name: None)
        layer2 = refused(word, tmp_path / "o", "--layer", "2")
        assert "`soffice` is not on the path" in layer2
        assert "1 Word or PowerPoint files, such as p104-dredge/notes.docx" in layer2
        assert "such as p101-harbour-lidar/notes.docx" in refused(
            included, tmp_path / "o", "--layer", "3", pack=moved
        )
        # Layer 3 converts only the holdout's included files, which hold no Word file here.
        layers, _ = eval_run._check_inputs(
            EvalPack.load(EXAMPLE_PACK_PATH),
            "3",
            None,
            EXAMPLE_PACK_PATH,
            word,
            word.parent / "bathymetry-corpus",
            tmp_path / "o",
            None,
        )
        assert layers == {"3"}

    # A reference or exemplar document without sections cannot be drafted from.
    # The build refuses a delivered document without text, so the sections are removed from a
    # built snapshot, with the induced knowledge that could cite them.
    for name, empty in (
        ("reference", "p102-estuary-multibeam/estuary-proposal.md"),
        ("exemplar", "p101-harbour-lidar/harbour-memo.md"),
    ):
        hollow = compile_snapshot(tmp_path / name, BATHYMETRY_CORPUS)
        with closing(sqlite3.connect(hollow)) as db, db:
            (doc_id,) = db.execute("SELECT doc_id FROM documents WHERE rel_path = ?", (empty,))
            chunks = "SELECT chunk_id FROM chunks WHERE doc_id = ?"
            db.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({chunks})", doc_id)
            db.execute("DELETE FROM chunks WHERE doc_id = ?", doc_id)
            db.execute("DELETE FROM sections WHERE doc_id = ?", doc_id)
            db.execute("DELETE FROM glossary")
            db.execute("DELETE FROM blueprints")
            db.execute("UPDATE corpus_meta SET value = 'Guide.' WHERE key = 'guide_markdown'")
        message = refused(hollow, tmp_path / "o", "--layer", "3")
        assert f"holdout 'estuary-proposal': {empty} has no sections in --snapshot" in message


def test_eval_runner_stops_with_usage_and_no_report_when_the_spend_limit_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    api = ScriptedOpenAI()
    # The first QA turn costs $0.30 and asks for a tool, so the second turn hits the limit.
    api.script(
        "/responses",
        response_body(
            [function_call("c1", "list_corpus_files", "{}")], tokens=usage(100_000, 10_000)
        ),
    )

    def scripted_gateway(**options: Any) -> Any:
        # The fake snapshot's embedding model has no price; no query is embedded here.
        options.pop("embedding_model", None)
        options.pop("embedding_dimensions", None)
        return api.gateway(**options)

    monkeypatch.setattr(eval_run, "OpenAIGateway", scripted_gateway)
    out = tmp_path / "out"
    corpus = tmp_path / "bathymetry-corpus"
    arguments = ["--pack", str(EXAMPLE_PACK_PATH), "--snapshot", str(snapshot_path)]
    arguments += ["--out", str(out), "--corpus", str(corpus), "--layer", "2", "--max-cost", "0.3"]

    result = CliRunner().invoke(eval_app, arguments)

    assert result.exit_code == 1, result.output
    assert "reached the $0.3 limit" in result.stderr
    assert "usage: 1 provider calls" in result.stdout and "estimated $0.3000" in result.stdout
    assert api.request_count == 1
    assert list(out.iterdir()) == []


def test_eval_runner_prints_usage_when_a_layer_fails_after_calls_and_refuses_infinite_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    arguments = ["--pack", str(EXAMPLE_PACK_PATH), "--snapshot", str(snapshot_path)]
    arguments += ["--out", str(tmp_path / "out"), "--layer", "1"]
    runner = CliRunner()
    for limit in ("inf", "nan", "-1"):
        refused = runner.invoke(eval_app, [*arguments, "--max-cost", limit])
        assert refused.exit_code == 2 and "positive finite" in refused.output, refused.output

    def fails_after_calls(*args: object) -> None:
        raise ValueError("layer stopped after its calls")

    monkeypatch.setattr(eval_run, "_embedder", lambda *args: None)
    monkeypatch.setattr(eval_run, "OpenAIGateway", lambda **options: None)
    monkeypatch.setattr(eval_run, "run_layer1", fails_after_calls)
    result = runner.invoke(eval_app, arguments)

    assert result.exit_code == 1, result.output
    assert "usage: 0 provider calls" in result.stdout
    assert "layer stopped after its calls" in result.stderr
