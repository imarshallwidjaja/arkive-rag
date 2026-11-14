from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anyio
import numpy as np
import openai
import pytest
from numpy.typing import NDArray
from openai.types.responses import ResponseFunctionToolCall
from pydantic import ValidationError
from typer.testing import CliRunner

from arkive_rag.models import DocumentInfo, ProviderError, UnsafePathError
from arkive_rag.server import INSTRUCTIONS
from arkive_rag.snapshot import TopicSnapshot
from evals import agents
from evals.ablation import first_satisfying_rank, run_layer1, summarize_ranks
from evals.agents import (
    QA_INSTRUCTIONS,
    CountingEmbedder,
    RawCorpus,
    ToolOutput,
    arkive_mcp_tools,
    assert_holdout_clean,
    blind_assignment,
    check_reference_held_out,
    compile_holdout_snapshot,
    run_layer2_qa,
    run_tool_loop,
    score_memo_rubric,
    select_holdout_sources,
    with_server_instructions,
)
from evals.run import _layer2_report, write_private_text
from evals.run import app as eval_app
from evals.schema import (
    EvalPack,
    HoldoutMemoTask,
    RetrievalCase,
    normalize_for_match,
    satisfies_groups,
)
from tests.support import BATHYMETRY_CORPUS, FakeGateway, compile_snapshot, write_pdf

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
        ({"format": "arkive-eval-pack/0"}, "format"),
        ({"corpus_fingerprint": "abc"}, "corpus_fingerprint"),
        ({"surprise": 1}, "Extra inputs"),
        ({"retrieval_cases__1__case_id": "exact_identifiers_1"}, "duplicate retrieval case_id"),
        ({"retrieval_cases__0__required_groups": [[]]}, "at least 1 item"),
        ({"qa_cases__0__required_groups": [["  "]]}, "pattern"),
        ({"holdout_memo__included_paths": ["../outside"]}, "relative to the corpus root"),
        ({"holdout_memo__included_paths": ["/abs"]}, "relative to the corpus root"),
        ({"holdout_memo__included_paths": ["p103-notes"]}, "held-out marker"),
        ({"holdout_memo__rubric__0__require_groups": []}, "no require_groups or forbid_any"),
        ({"holdout_memo__rubric__1__check_id": "executive_summary"}, "duplicate rubric check_id"),
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
# Reef Sonar Habitat Survey
## Executive Summary
Recommendations: diver survey of high-likelihood reef.
Tiles were 2000 x 2000 pixels over 812 km2. Unlike the harbour project's 1 m lidar grid,
the sidescan mosaic was not resampled.
| File Name | Masking |
| reef_habitat_likelihood.tif | No masking applied. |
| reef_habitat_threshold.tif | No masking applied. |
## Limitations
Ground truth is needed over moorings and dredged channels.
"""


def test_rubric_rewards_faithful_memos_and_catches_targeted_contamination(
    example_pack: EvalPack,
) -> None:
    rubric = holdout_task(example_pack).rubric
    assert all(score_memo_rubric(FAITHFUL_MEMO, rubric).values())
    borrowed_tiles = FAITHFUL_MEMO.replace("2000 x 2000 pixels", "500 x 500 pixels")
    borrowed_mask = FAITHFUL_MEMO + "\nShallows were masked using the harbour coastline land mask."
    assert score_memo_rubric(borrowed_tiles, rubric) == {
        **score_memo_rubric(FAITHFUL_MEMO, rubric),
        "tile_size_not_contaminated": False,
    }
    assert not score_memo_rubric(borrowed_mask, rubric)["no_borrowed_masking_claim"]
    dropped = FAITHFUL_MEMO.replace("reef_habitat_threshold.tif", "")
    assert not score_memo_rubric(dropped, rubric)["deliverables_and_area_kept"]
    assert not any(score_memo_rubric("", rubric).values())


def write_corpus(root: Path, files: dict[str, str | bytes]) -> Path:
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        (root / rel).write_bytes(data)
    return root


HOLDOUT_FILES: dict[str, str | bytes] = {
    "p101-harbour-lidar/memo.md": "# Harbour\n\nLidar grid 1 m.",
    "p102-estuary-multibeam/memo.md": "# Estuary\n\nMultibeam fee.",
    "p103-reef-sonar/Reef Sonar Memo FINAL.md": "# Reef\n\nSidescan 2000 x 2000 tiles.",
    "AGENTS.md": "# Notes\n\nCorpus guidance.",
}


def reference_doc(**overrides: Any) -> DocumentInfo:
    fields = {
        "doc_id": "doc_003",
        "rel_path": "p103-reef-sonar/Reef Sonar Memo FINAL.md",
        "sha256": hashlib.sha256(b"# Reef\n\nSidescan 2000 x 2000 tiles.").hexdigest(),
        "format": "md",
        "project_id": "p103-reef-sonar",
        "client": "Contoso Ports",
        "stage": "memorandum",
        "title": "Reef Sonar Memo",
        "doc_date": None,
        "summary": "Reef memo.",
        "page_count": 1,
    }
    return DocumentInfo.model_validate(fields | overrides)


def test_holdout_selection_keeps_included_files_and_refuses_held_out_material(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    task = holdout_task(example_pack)
    reference = reference_doc()
    check_reference_held_out(task, reference)
    with pytest.raises(ValueError, match="no held-out marker"):
        check_reference_held_out(
            task, reference_doc(rel_path="p101-harbour-lidar/memo.md", project_id="p101-harbour")
        )

    corpus = write_corpus(tmp_path / "corpus", HOLDOUT_FILES)
    selected = select_holdout_sources(corpus, task, reference)
    assert sorted(selected) == ["p101-harbour-lidar/memo.md", "p102-estuary-multibeam/memo.md"]

    renamed = write_corpus(
        tmp_path / "renamed",
        HOLDOUT_FILES | {"p102-estuary-multibeam/old draft.md": HOLDOUT_FILES[reference.rel_path]},
    )
    with pytest.raises(ValueError, match="same bytes as the reference"):
        select_holdout_sources(renamed, task, reference)
    misfiled = write_corpus(
        tmp_path / "misfiled", HOLDOUT_FILES | {"p101-harbour-lidar/reef-sonar notes.md": "x"}
    )
    with pytest.raises(ValueError, match="held-out marker 'reef-sonar'"):
        select_holdout_sources(misfiled, task, reference)
    linked = write_corpus(tmp_path / "linked", HOLDOUT_FILES)
    (linked / "p101-harbour-lidar" / "reef.md").symlink_to(linked / reference.rel_path)
    with pytest.raises(UnsafePathError, match="symlinks"):
        select_holdout_sources(linked, task, reference)
    missing = write_corpus(tmp_path / "missing", {"p101-harbour-lidar/memo.md": "x"})
    with pytest.raises(ValueError, match="'p102-estuary-multibeam' matches no corpus file"):
        select_holdout_sources(missing, task, reference)

    with pytest.raises(ValueError, match="p103"):
        assert_holdout_clean(task, reference, [reference_doc(rel_path="moved/memo.md")])
    with pytest.raises(ValueError, match="same bytes"):
        assert_holdout_clean(
            task, reference, [reference_doc(rel_path="p101/x.md", project_id="p101")]
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
        assert holdout.projects == ["p101-harbour-lidar", "p102-estuary-multibeam"]
    output.unlink()

    def leak(*args: object) -> None:
        raise ValueError("holdout snapshot leaks held-out material")

    gateway = FakeGateway()
    renamed = write_corpus(
        tmp_path / "renamed",
        HOLDOUT_FILES
        | {"p101-harbour-lidar/copy.md": HOLDOUT_FILES["p103-reef-sonar/Reef Sonar Memo FINAL.md"]},
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

    def run(pack: Path, *extra: str, out: Path = out) -> str:
        result = runner.invoke(
            eval_app,
            ["--pack", str(pack), "--snapshot", str(snapshot_path), "--out", str(out), *extra],
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


def test_blind_assignment_is_a_seeded_permutation() -> None:
    arms = ["arm_a_zero_shot", "arm_b_chunk_only_rag", "arm_c_full_arkive_mcp"]
    key = blind_assignment(arms)
    assert key == blind_assignment(arms)
    assert list(key) == ["Candidate_1", "Candidate_2", "Candidate_3"]
    assert sorted(key.values()) == arms


def test_raw_corpus_tools_read_text_layers_and_page_images(tmp_path: Path) -> None:
    (tmp_path / "p1").mkdir()
    (tmp_path / "AGENTS.md").write_text("# Notes\nStill closing out.", encoding="utf-8")
    write_pdf(tmp_path / "p1" / "memo.pdf", [["Rainfall 330.9 mm"], ["Outputs depth_grid.tif"]])
    raw = RawCorpus(tmp_path)

    def call(name: str, **args: Any) -> ToolOutput:
        return anyio.run(raw.execute, name, args)

    assert json.loads(cast(str, call("list_corpus_files"))) == [
        {"rel_path": "AGENTS.md", "pages": 1},
        {"rel_path": "p1/memo.pdf", "pages": 2},
    ]
    assert json.loads(cast(str, call("grep_raw_text", pattern="330.9"))) == [
        {"rel_path": "p1/memo.pdf", "page": 1, "line": "Rainfall 330.9 mm"}
    ]
    assert "--- page 2 ---\nOutputs depth_grid.tif" in cast(
        str, call("read_raw_pages", rel_path="p1/memo.pdf", start_page=1, end_page=9)
    )
    image = call("view_raw_page_image", rel_path="p1/memo.pdf", page_number=2)
    assert isinstance(image, list) and image[1]["type"] == "input_image"
    assert cast(str, call("view_raw_page_image", rel_path="AGENTS.md", page_number=1)).startswith(
        "ERROR"
    )


def test_arkive_mcp_adapter_exposes_every_server_tool_and_forwards_images(tmp_path: Path) -> None:
    snapshot_path = compile_snapshot(
        tmp_path, BATHYMETRY_CORPUS, pdfs={"p101-harbour-lidar/chart.pdf": [["Figure 330.9"]]}
    )

    async def run() -> tuple[list[str], ToolOutput, ToolOutput, ToolOutput, str]:
        async with arkive_mcp_tools(snapshot_path, FakeGateway().embed_query) as (
            tools,
            execute,
            server_instructions,
        ):
            grep = await execute("grep", {"topic": "bathymetry", "pattern": "330.9"})
            page = await execute(
                "view_page", {"topic": "bathymetry", "doc_id": "doc_001", "page_number": 1}
            )
            missing = await execute("read_section", {"topic": "bathymetry", "section_id": "nope"})
            return [t["name"] for t in tools], grep, page, missing, server_instructions

    names, grep, page, missing, server_instructions = anyio.run(run)
    assert server_instructions == INSTRUCTIONS
    assert with_server_instructions("Answer.", "  ") == "Answer."
    assert sorted(names) == sorted(
        [
            "list_topics",
            "get_topic_guide",
            "list_documents",
            "get_document_outline",
            "search",
            "grep",
            "read_section",
            "view_page",
        ]
    )
    assert isinstance(grep, str) and "330.9" in grep
    assert isinstance(page, list) and [item["type"] for item in page] == [
        "input_text",
        "input_image",
    ]
    assert isinstance(missing, str) and missing.startswith("ERROR:")


@dataclass
class FakeResponses:
    script: list[SimpleNamespace]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return self.script.pop(0)


def response(
    rid: str, output: list[Any], text: str = "", status: str = "completed"
) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=100, output_tokens=10)
    return SimpleNamespace(id=rid, output=output, output_text=text, status=status, usage=usage)


def tool_call(call_id: str, arguments: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call", call_id=call_id, name="echo", arguments=arguments
    )


def test_tool_loop_feeds_outputs_back_and_accounts_tokens_turns_and_errors() -> None:
    responses = FakeResponses(
        [
            response("r1", [tool_call("c1", '{"value": 7}'), tool_call("c2", "not json")]),
            response("r2", [], text="final answer"),
        ]
    )
    client = cast(openai.OpenAI, SimpleNamespace(responses=responses))

    embedder = CountingEmbedder(FakeGateway().embed_query)

    async def execute(name: str, args: dict[str, Any]) -> ToolOutput:
        embedder(str(args["value"]))
        return f"{name}={args['value']}"

    run = anyio.run(
        lambda: run_tool_loop(
            client,
            instructions="i",
            prompt="q",
            tools=[],
            execute=execute,
            max_turns=4,
            embedder=embedder,
        )
    )
    assert (run.answer, run.status, run.turns, run.tool_calls) == (
        "final answer",
        "completed",
        2,
        2,
    )
    assert (run.input_tokens, run.output_tokens, run.search_embed_calls) == (200, 20, 1)
    assert run.error is None
    second = responses.requests[1]
    assert second["previous_response_id"] == "r1"
    assert second["input"][0] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "echo=7",
    }
    assert second["input"][1]["output"].startswith("ERROR: JSONDecodeError")

    looping = FakeResponses([response(f"r{i}", [tool_call("c", '{"value": 1}')]) for i in range(2)])
    capped = anyio.run(
        lambda: run_tool_loop(
            cast(openai.OpenAI, SimpleNamespace(responses=looping)),
            instructions="i",
            prompt="q",
            tools=[],
            execute=execute,
            max_turns=2,
            embedder=embedder,
        )
    )
    assert (capped.answer, capped.status, capped.turns) == ("", "max_turns", 2)
    assert capped.search_embed_calls == 2


def loop(script: list[SimpleNamespace], execute: Any) -> Any:
    client = cast(openai.OpenAI, SimpleNamespace(responses=FakeResponses(script)))
    return anyio.run(
        lambda: run_tool_loop(
            client,
            instructions="i",
            prompt="q",
            tools=[],
            execute=execute,
            max_turns=4,
            embedder=CountingEmbedder(FakeGateway().embed_query),
        )
    )


def test_tool_loop_reports_unfinished_runs_and_surfaces_non_input_failures() -> None:
    executed: list[str] = []

    async def execute(name: str, args: dict[str, Any]) -> ToolOutput:
        executed.append(name)
        if args.get("fail") == "provider":
            raise ProviderError("embedding service down")
        if args.get("fail") == "bug":
            raise RuntimeError("harness bug")
        return "ok"

    partial_call = response("r1", [tool_call("c1", "{}")], text="half a memo", status="incomplete")
    run = loop([partial_call], execute)
    assert (run.status, run.answer, executed) == ("incomplete", "half a memo", [])
    assert loop([response("r1", [], text="  ")], execute).status == "empty_answer"
    failed = loop([response("r1", [tool_call("c1", '{"fail": "provider"}')])], execute)
    assert failed.status == "provider_error" and "embedding service down" in (failed.error or "")
    with pytest.raises(RuntimeError, match="harness bug"):
        loop([response("r1", [tool_call("c1", '{"fail": "bug"}')])], execute)


def test_layer2_does_not_judge_or_credit_unfinished_runs(
    tmp_path: Path, example_pack: EvalPack
) -> None:
    snapshot_path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    cases = example_pack.qa_cases
    runs = 2 * len(cases)
    script = [response(f"r{i}", [], text="330.9 median", status="incomplete") for i in range(runs)]
    # No `parse` attribute: any judge call would fail the test.
    responses = FakeResponses(script)
    client = cast(openai.OpenAI, SimpleNamespace(responses=responses))
    embedder = CountingEmbedder(FakeGateway().embed_query)
    result = anyio.run(
        run_layer2_qa, client, snapshot_path, tmp_path / "bathymetry-corpus", embedder, cases
    )
    # Like an MCP host, only the Arkive arm sees the server's initialize instructions.
    mcp_instructions = f"{QA_INSTRUCTIONS}\n\nMCP SERVER INSTRUCTIONS:\n{INSTRUCTIONS.strip()}"
    assert [r["instructions"] for r in responses.requests] == [
        QA_INSTRUCTIONS,
        mcp_instructions,
    ] * len(cases)
    rows = cast(list[dict[str, Any]], result["rows"])
    assert [r["status"] for r in rows] == ["incomplete"] * runs
    assert not any(r["phrase_coverage"] or r["groups_matched"] or r["judge"] for r in rows)
    report = "\n".join(_layer2_report(result))
    assert f"| raw_files | 0/{len(cases)} | 0/{len(cases)} | n/a |" in report
    assert "harbour_median_rainfall / arkive_mcp: incomplete" in report
