from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import stat
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anyio
import httpx2
import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from typer.testing import CliRunner

from arkive_rag.llm import BudgetExceededError, Tool, ToolReply, ToolResult, UsageMeter
from arkive_rag.models import DocumentInfo, ProviderError, UnsafePathError
from arkive_rag.server import INSTRUCTIONS
from arkive_rag.snapshot import TopicSnapshot
from evals import agents
from evals import run as eval_run
from evals.ablation import first_satisfying_rank, run_layer1, summarize_ranks
from evals.agents import (
    QA_INSTRUCTIONS,
    AgentRun,
    CountingEmbedder,
    MemoBlindComparison,
    RawCorpus,
    arkive_mcp_tools,
    assert_holdout_clean,
    blind_assignment,
    check_reference_held_out,
    compile_holdout_snapshot,
    run_agent,
    run_layer2_qa,
    run_layer3_holdout_memo,
    score_memo_rubric,
    select_holdout_sources,
    with_server_instructions,
)
from evals.opencode_check.scenarios import ARKIVE_TOOLS
from evals.run import _layer2_report, make_private_directory, write_private_text
from evals.run import app as eval_app
from evals.schema import (
    EvalPack,
    HoldoutMemoTask,
    RetrievalCase,
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
    response_body,
    structured_body,
    usage,
    write_pdf,
)

EXAMPLE_PACK_PATH = Path(__file__).resolve().parents[1] / "evals" / "example_pack.eval.json"


@pytest.fixture(scope="module")
def example_pack() -> EvalPack:
    return EvalPack.load(EXAMPLE_PACK_PATH)


def holdout_task(pack: EvalPack) -> HoldoutMemoTask:
    assert pack.holdout_memo is not None
    return pack.holdout_memo


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
        ({"holdout_memo__included_paths": ["../outside"]}, "relative to the corpus root"),
        ({"holdout_memo__included_paths": ["/abs"]}, "relative to the corpus root"),
        ({"holdout_memo__included_paths": ["P102-notes"]}, "held-out marker"),
        ({"holdout_memo__included_paths": ["p101-harbour-lidar/"]}, "plain corpus-relative"),
        ({"holdout_memo__included_paths": ["./p101-harbour-lidar"]}, "plain corpus-relative"),
        ({"holdout_memo__included_paths": ["p101//memo.md"]}, "plain corpus-relative"),
        ({"holdout_memo__included_paths": ["."]}, "plain corpus-relative"),
        ({"holdout_memo__rubric__0__require_groups": []}, "no require_groups or forbid_any"),
        ({"holdout_memo__rubric__1__check_id": "scope_and_budget"}, "duplicate rubric check_id"),
    ],
)
def test_eval_pack_rejects_malformed_packs(overrides: dict[str, Any], message: str) -> None:
    EvalPack.model_validate(pack_with())
    with pytest.raises(ValidationError, match=message):
        EvalPack.model_validate(pack_with(**overrides))


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


FAITHFUL_MEMO = """\
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


def test_rubric_rewards_faithful_memos_and_catches_targeted_contamination(
    example_pack: EvalPack,
) -> None:
    rubric = holdout_task(example_pack).rubric
    assert all(score_memo_rubric(FAITHFUL_MEMO, rubric).values())
    borrowed_tiles = FAITHFUL_MEMO.replace("2000 x 2000 pixel", "500 x 500 pixel")
    borrowed_mask = FAITHFUL_MEMO + "\nShallows were masked using the harbour coastline land mask."
    assert score_memo_rubric(borrowed_tiles, rubric) == {
        **score_memo_rubric(FAITHFUL_MEMO, rubric),
        "tile_size_not_contaminated": False,
    }
    assert not score_memo_rubric(borrowed_mask, rubric)["no_borrowed_masking_claim"]
    dropped = FAITHFUL_MEMO.replace("estuary_depth_uncertainty.tif", "")
    assert not score_memo_rubric(dropped, rubric)["deliverables_and_area_kept"]
    assert not any(score_memo_rubric("", rubric).values())


def write_corpus(root: Path, files: dict[str, str | bytes]) -> Path:
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        (root / rel).write_bytes(data)
    return root


REFERENCE_PATH = "p102-estuary-multibeam/Estuary Proposal FINAL.md"
REFERENCE_TEXT = "# Estuary\n\nMultibeam 2000 x 2000 tiles."
HOLDOUT_FILES: dict[str, str | bytes] = {
    "p101-harbour-lidar/memo.md": "# Harbour\n\nLidar grid 1 m.",
    REFERENCE_PATH: REFERENCE_TEXT,
    "p103-reef-sonar/memo.md": "# Reef\n\nSidescan notes.",
    "AGENTS.md": "# Notes\n\nCorpus guidance.",
}


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
        check_reference_held_out(task, reference_doc(rel_path="p101-harbour-lidar/memo.md"))

    corpus = write_corpus(tmp_path / "corpus", HOLDOUT_FILES)
    selected = select_holdout_sources(corpus, task, reference)
    assert sorted(selected) == ["p101-harbour-lidar/memo.md", "p103-reef-sonar/memo.md"]

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
    # A file under the reference's top-level folder is held out even without a marker in it.
    sibling = task.model_copy(update={"included_paths": ("p101-harbour-lidar",)})
    in_folder = reference_doc(rel_path="p101-harbour-lidar/final/memo.md")
    with pytest.raises(ValueError, match="the reference's top-level folder 'p101-harbour-lidar'"):
        select_holdout_sources(corpus, sibling, in_folder)
    # A reference at the corpus root excludes the root's other files the same way.
    at_root = task.model_copy(update={"included_paths": ("AGENTS.md",)})
    root_reference = reference_doc(rel_path="P102 Proposal FINAL.md", sha256="0")
    check_reference_held_out(at_root, root_reference)
    with pytest.raises(ValueError, match=r"AGENTS\.md is held-out material: the corpus root"):
        select_holdout_sources(corpus, at_root, root_reference)
    with pytest.raises(ValueError, match="is a curator note"):
        check_reference_held_out(
            task, reference_doc(rel_path="p102-estuary-multibeam/AGENTS.md", is_curator_note=True)
        )
    linked = write_corpus(tmp_path / "linked", HOLDOUT_FILES)
    (linked / "p101-harbour-lidar" / "reef.md").symlink_to(linked / reference.rel_path)
    with pytest.raises(UnsafePathError, match="symlinks"):
        select_holdout_sources(linked, task, reference)
    missing = write_corpus(tmp_path / "missing", {"p101-harbour-lidar/memo.md": "x"})
    with pytest.raises(ValueError, match="'p103-reef-sonar' matches no corpus file"):
        select_holdout_sources(missing, task, reference)

    with pytest.raises(ValueError, match="held-out marker 'p102'"):
        assert_holdout_clean(task, reference, [reference_doc(rel_path="moved/P102.md", sha256="0")])
    with pytest.raises(ValueError, match="same bytes"):
        assert_holdout_clean(task, reference, [reference_doc(rel_path="p101/x.md")])
    with pytest.raises(ValueError, match="top-level folder 'p102-estuary-multibeam'"):
        assert_holdout_clean(
            task, reference, [reference_doc(rel_path="p102-estuary-multibeam/other.md", sha256="0")]
        )


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
            "p101-harbour-lidar/memo.md",
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

    monkeypatch.setattr(agents, "assert_holdout_clean", leak)
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
    retrieval_only.write_text(json.dumps(pack_with(qa_cases=[], holdout_memo=None)))
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

    def pack_file(name: str, **overrides: Any) -> Path:
        path = tmp_path / f"{name}.eval.json"
        path.write_text(json.dumps(pack_with(**overrides)))
        return path

    # An invalid pack or snapshot is a clean usage error, not a traceback.
    assert "is not a valid eval pack" in run(pack_file("v1", format="arkive-eval-pack/1"))
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
    note_reference = pack_file("note", holdout_memo__reference_doc_id="doc_003")
    assert "holdout_memo: reference document doc_003" in run(note_reference, *everything)
    assert "is a curator note" in run(note_reference, *everything)
    missing_part = pack_file("part", holdout_memo__included_paths=["p101-harbour-lidar", "p104"])
    assert "'p104' matches no corpus file" in run(missing_part, *everything)

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
    # A link at the destination is refused, not followed and truncated.
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    link = tmp_path / "results.json"
    link.symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        write_private_text(link, "new")
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
    arms = ["arm_a_zero_shot", "arm_b_chunk_only_rag", "arm_c_full_arkive_mcp"]
    key = blind_assignment(arms)
    assert key == blind_assignment(arms)
    assert list(key) == ["Candidate_1", "Candidate_2", "Candidate_3"]
    assert sorted(key.values()) == arms


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

    assert (run.answer, run.status, run.turns, run.tool_calls, run.error) == (
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
    assert (run.status, run.answer, run.tool_calls, run.input_tokens) == (
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
    assert f"| raw_files | 0/{len(cases)} | 0/{len(cases)} | n/a |" in report
    assert "harbour_median_rainfall / arkive_mcp: incomplete" in report


def test_layer3_compiles_the_holdout_off_the_event_loop_and_judges_completed_drafts(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    task = holdout_task(example_pack)
    corpus = write_corpus(tmp_path / "corpus", HOLDOUT_FILES)
    full = compile_snapshot(tmp_path, {k: cast(str, v) for k, v in HOLDOUT_FILES.items()})
    with TopicSnapshot.open(full) as snapshot:
        reference_id = next(
            d.doc_id for d in snapshot.list_documents() if d.rel_path.startswith("p102")
        )
    task = task.model_copy(update={"reference_doc_id": reference_id})
    out = tmp_path / "out"
    out.mkdir()
    verdict = MemoBlindComparison(candidates=[], ranking=["Candidate_1"])
    api = ScriptedOpenAI()
    api.script("/responses", *[response_body([message(f"draft {i}")]) for i in range(2)])
    api.script("/responses", response_body([message("cut")], "incomplete"))
    api.script("/responses", structured_body(verdict))
    holdout_gateway = FakeGateway()

    result = anyio.run(
        lambda: run_layer3_holdout_memo(
            api.gateway(),
            holdout_gateway=holdout_gateway,
            full_snapshot_path=full,
            corpus_dir=corpus,
            embedder=CountingEmbedder(FakeGateway().embed_query),
            output_dir=out,
            task=task,
        )
    )

    assert holdout_gateway.calls and (out / "holdout.arkive.db").is_file()
    arms = cast(dict[str, Any], result["arms"])
    assert [a["run"].status for a in arms.values()] == ["completed", "completed", "incomplete"]
    assert arms["arm_c_full_arkive_mcp"]["rubric"] is None
    assert sorted(cast(dict[str, str], result["blind_key"]).values()) == [
        "arm_a_zero_shot",
        "arm_b_chunk_only_rag",
    ]
    assert result["blind_comparison"] == verdict
    judge = api.requests("/responses")[-1]
    assert judge["text"]["format"]["name"] == "MemoBlindComparison"
    assert "draft 0" in judge["input"][0]["content"][0]["text"]


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
