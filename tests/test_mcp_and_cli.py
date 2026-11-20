from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult, ImageContent, TextContent
from typer.testing import CliRunner

from arkive_rag import __version__, cli
from arkive_rag.cli import app
from arkive_rag.models import ArkiveError, ObservationDraft, document_id
from arkive_rag.server import TopicCatalog, create_mcp_server
from arkive_rag.snapshot import QueryEmbedder, TopicSnapshot
from tests.support import (
    BATHYMETRY_CORPUS,
    HARBOUR_MEMO,
    MARKETING_CORPUS,
    FakeGateway,
    ScriptedOpenAI,
    compile_snapshot,
    structured_body,
    usage,
)

CHART_MEMO = document_id("p101-harbour-lidar/harbour-chart-memo.pdf")
PDF_PAGES = [
    ["# Rainfall", "Nine Year Median = 330.9 mm"],
    ["# Outputs", "Raster estuary_2024_grid.tif"],
]


@pytest.fixture
def snapshots(tmp_path: Path) -> tuple[Path, Path]:
    marine = compile_snapshot(
        tmp_path,
        BATHYMETRY_CORPUS,
        topic="bathymetry",
        pdfs={"p101-harbour-lidar/harbour-chart-memo.pdf": PDF_PAGES},
    )
    soils = compile_snapshot(
        tmp_path, {"p900-soils/soils-memo.md": "# Soils\n\nClay loam at 330 sites."}, topic="soils"
    )
    return marine, soils


def call_tools(
    paths: tuple[Path, ...], calls: list[tuple[str, dict[str, Any]]], embed: QueryEmbedder | None
) -> list[CallToolResult]:
    async def run() -> list[CallToolResult]:
        with TopicCatalog(paths) as catalog:
            async with Client(create_mcp_server(catalog, embed)) as client:
                return [await client.call_tool(name, args) for name, args in calls]

    return anyio.run(run)


def data(result: CallToolResult) -> Any:
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content.get("result", result.structured_content)


def error_text(result: CallToolResult) -> str:
    assert result.is_error
    (content,) = result.content
    assert isinstance(content, TextContent)
    return content.text


def test_mcp_session_exercises_every_tool_across_isolated_topics(
    snapshots: tuple[Path, Path],
) -> None:
    marine = "bathymetry"
    results = call_tools(
        snapshots,
        [
            ("list_topics", {}),
            ("get_topic_guide", {"topic": marine}),
            ("list_documents", {"topic": marine, "filters": {"project": "p101-harbour-lidar"}}),
            ("get_document_outline", {"topic": marine, "doc_id": CHART_MEMO}),
            ("search", {"topic": marine, "query": "nine year median rainfall", "top_k": 3}),
            ("grep", {"topic": marine, "pattern": "330.9"}),
            ("read_section", {"topic": marine, "section_id": f"{CHART_MEMO}_sec_002"}),
            ("view_page", {"topic": marine, "doc_id": CHART_MEMO, "page_number": 2}),
            ("list_documents", {"topic": "soils"}),
            ("grep", {"topic": "soils", "pattern": "330.9"}),
            ("lookup_glossary", {"topic": marine}),
            ("lookup_glossary", {"topic": marine, "query": "lowest astronomical"}),
            ("get_blueprint", {"topic": marine, "deliverable_type": "Survey Memorandum"}),
        ],
        FakeGateway().embed_query,
    )
    topics, guide, memos, outline, hits, matches, section, page, soils, soils_grep = results[:10]
    terms, lat, memo_blueprint = results[10:]

    assert [(t["topic"], t["document_count"]) for t in data(topics)] == [
        ("bathymetry", 4),
        ("soils", 1),
    ]
    assert [len(t["corpus_fingerprint"]) for t in data(topics)] == [64, 64]
    project = data(topics)[0]["facets"][0]
    assert (project["name"], project["values"]) == (
        "project",
        ["p101-harbour-lidar", "p102-estuary-multibeam", "p103-reef-sonar"],
    )
    assert f"[{CHART_MEMO}:" in data(guide)["guide_markdown"]
    # The guide carries blueprint summaries; get_blueprint returns one in full.
    assert data(guide)["blueprints"] == [
        {
            "deliverable_type": "proposal",
            "description": "How the collection writes a proposal.",
            "exemplar_count": 1,
        },
        {
            "deliverable_type": "survey memorandum",
            "description": "How the collection writes a survey memorandum.",
            "exemplar_count": 1,
        },
    ]
    blueprint = data(memo_blueprint)
    assert blueprint["deliverable_type"] == "survey memorandum"
    assert blueprint["outline"] and blueprint["conventions"]
    # A single exemplar shows no recurrence, so the memorandum has no standing passages.
    assert blueprint["standing_passages"] == []
    assert data(guide)["glossary_terms"] == ["LAT"]
    assert data(guide)["curator_notes"][0]["scope_path"] == "p103-reef-sonar/AGENTS.md"
    assert [d["rel_path"] for d in data(memos)] == [
        "p101-harbour-lidar/harbour-chart-memo.pdf",
        "p101-harbour-lidar/harbour-memo.md",
    ]
    assert [s["heading_path"] for s in data(outline)["sections"]] == ["Rainfall", "Outputs"]
    assert len(data(hits)) == 3 and "330.9" in data(hits)[0]["content"]
    assert {m["doc_id"] for m in data(matches)} == {CHART_MEMO, HARBOUR_MEMO}
    assert data(section)["markdown"] == "# Outputs\n\nRaster estuary_2024_grid.tif"
    assert data(section)["prev_section_id"] == f"{CHART_MEMO}_sec_001"
    text, image = page.content
    assert isinstance(text, TextContent) and "page 2 of 2" in text.text
    assert isinstance(image, ImageContent) and image.mime_type == "image/webp"
    assert base64.b64decode(image.data)[8:12] == b"WEBP"
    assert [d["rel_path"] for d in data(soils)] == ["p900-soils/soils-memo.md"]
    assert data(soils_grep) == []
    assert [(t["term"], t["defined"]) for t in data(terms)["terms"]] == [("LAT", True)]
    (entry,) = data(lat)["matches"]
    assert entry["aliases"] == ["lowest astronomical tide"]
    assert entry["senses"][0]["citations"] == [f"{HARBOUR_MEMO}_sec_003"]


def test_mcp_errors_name_valid_options_and_offline_search_explains_itself(
    snapshots: tuple[Path, Path],
) -> None:
    results = call_tools(
        snapshots,
        [
            ("search", {"topic": "bathymetry", "query": "rainfall"}),
            ("grep", {"topic": "bathymetry", "pattern": "Sentinel-2", "max_matches": 1}),
            ("read_section", {"topic": "bathymetry", "section_id": f"{CHART_MEMO}_sec_001"}),
            ("view_page", {"topic": "bathymetry", "doc_id": CHART_MEMO, "page_number": 1}),
            ("list_documents", {"topic": "mining"}),
            ("list_documents", {"topic": "bathymetry", "filters": {"project": "p999"}}),
            ("view_page", {"topic": "bathymetry", "doc_id": CHART_MEMO, "page_number": 9}),
            ("grep", {"topic": "bathymetry", "pattern": "x", "filters": {"stage": "memo"}}),
            ("lookup_glossary", {"topic": "bathymetry", "query": " "}),
            ("get_blueprint", {"topic": "bathymetry", "deliverable_type": "press release"}),
        ],
        None,
    )
    search, grep, section, page, bad_topic, bad_project, bad_page, bad_facet, blank = results[:9]
    (bad_type,) = results[9:]

    assert "OPENAI_API_KEY" in error_text(search) and "grep" in error_text(search)
    assert len(data(grep)) == 1
    assert data(section)["section_id"] == f"{CHART_MEMO}_sec_001"
    assert isinstance(page.content[1], ImageContent)
    assert error_text(bad_topic).endswith("unknown topic 'mining'; valid topics: bathymetry, soils")
    assert "valid values: p101-harbour-lidar" in error_text(bad_project)
    assert error_text(bad_page).endswith(f"{CHART_MEMO} has no page 9; it has pages 1-2")
    assert error_text(bad_facet).endswith("unknown facet 'stage'; valid facets: project, client")
    assert "glossary query must not be blank" in error_text(blank)
    assert error_text(bad_type).endswith("valid types: proposal, survey memorandum")


def test_catalog_rejects_two_snapshots_with_the_same_topic(
    snapshots: tuple[Path, Path], tmp_path: Path
) -> None:
    copy = tmp_path / "copy.arkive.db"
    copy.write_bytes(snapshots[0].read_bytes())
    with pytest.raises(ArkiveError, match="topic 'bathymetry' is served by both"):
        TopicCatalog([snapshots[0], copy])


def test_cli_inspect_and_grep_work_offline(
    snapshots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = CliRunner()

    inspected = runner.invoke(app, ["inspect", str(snapshots[0])])
    grepped = runner.invoke(
        app,
        [
            "grep",
            str(snapshots[0]),
            "sentinel-2",
            "--filter",
            "project=p101-harbour-lidar",
            "--filter",
            "client=Northwind",
        ],
    )
    bad_facet = runner.invoke(app, ["grep", str(snapshots[0]), "x", "--filter", "stage=draft"])
    twice = ["--filter", "project=p101-harbour-lidar", "--filter", "project=p103-reef-sonar"]
    repeated = runner.invoke(app, ["grep", str(snapshots[0]), "x", *twice])
    malformed = runner.invoke(app, ["grep", str(snapshots[0]), "x", "--filter", "project"])
    search = runner.invoke(app, ["search", str(snapshots[0]), "rainfall"])

    assert inspected.exit_code == 0, inspected.output
    summary = json.loads(inspected.stdout)
    assert (summary["topic"], summary["document_count"]) == ("bathymetry", 4)
    with TopicSnapshot.open(snapshots[0]) as snap:
        assert summary["corpus_fingerprint"] == snap.corpus_fingerprint
    assert summary["arkive_version"] == __version__
    source = snapshots[0].parent / "bathymetry-corpus" / "p101-harbour-lidar/harbour-chart-memo.pdf"
    assert summary["documents"][0] == {
        "doc_id": CHART_MEMO,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rel_path": "p101-harbour-lidar/harbour-chart-memo.pdf",
        "format": "pdf",
        "title": "Rainfall",
        "doc_date": None,
        "date": None,
        "deliverable_type": "document",
        "page_count": 2,
        "is_curator_note": False,
        "facets": {"project": ["p101-harbour-lidar"]},
        "sections": 2,
    }
    assert [f["name"] for f in summary["facets"]] == ["project", "client"]
    assert [e["term"] for e in summary["glossary"]] == ["LAT"]
    assert [b["deliverable_type"] for b in summary["blueprints"]] == [
        "proposal",
        "survey memorandum",
    ]
    assert summary["build_settings"]["efforts"]["observe"] == "low"
    assert summary["build_report"]["discarded_unlocated_quotes"] == 0
    assert grepped.exit_code == 0, grepped.output
    assert [m["doc_id"] for m in json.loads(grepped.stdout)] == [HARBOUR_MEMO]
    assert bad_facet.exit_code == 1 and "valid facets: project, client" in bad_facet.stderr
    assert repeated.exit_code == 2 and "given twice" in repeated.output
    assert malformed.exit_code == 2 and "not FACET=VALUE" in malformed.output
    assert search.exit_code == 1 and "OPENAI_API_KEY" in search.stderr


ARKIVE_RAG = Path(sys.executable).with_name("arkive-rag")
# No OPENAI_API_KEY, and no bytecode written into the checkout.
CHILD_ENV = {"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"}


def test_installed_cli_serves_snapshots_over_stdio(
    snapshots: tuple[Path, Path], tmp_path: Path
) -> None:
    params = StdioServerParameters(
        command=str(ARKIVE_RAG),
        args=["serve", str(snapshots[0]), str(snapshots[1])],
        env=CHILD_ENV,
        cwd=tmp_path,
    )

    async def run() -> tuple[str | None, CallToolResult, CallToolResult]:
        async with Client(params) as client:
            info = client.server_info
            grep = await client.call_tool("grep", {"topic": "bathymetry", "pattern": "330.9"})
            search = await client.call_tool("search", {"topic": "bathymetry", "query": "rainfall"})
            return (info.version if info else None), grep, search

    version, grep, search = anyio.run(run)

    assert version == __version__
    assert {m["doc_id"] for m in data(grep)} == {CHART_MEMO, HARBOUR_MEMO}
    assert "OPENAI_API_KEY" in error_text(search)


def test_cli_refuses_non_loopback_http_without_explicit_opt_in(
    snapshots: tuple[Path, Path], tmp_path: Path
) -> None:
    http = ["--transport", "streamable-http", "--host", "0.0.0.0"]
    result = subprocess.run(
        [str(ARKIVE_RAG), "serve", str(snapshots[0]), *http],
        capture_output=True,
        text=True,
        timeout=60,
        env=CHILD_ENV,
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 1
    assert "unauthenticated" in result.stderr and "--allow-nonlocal" in result.stderr


def test_cli_build_rejects_outputs_in_the_corpus_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    corpus = tmp_path / "corpus"
    (corpus / "p1").mkdir(parents=True)
    (corpus / "p1" / "memo.md").write_text("# Memo\n\nText")
    runner = CliRunner()

    inside = runner.invoke(app, ["build", str(corpus), "-o", str(corpus / "t.arkive.db")])
    shared = tmp_path / "t.db"
    same = runner.invoke(app, ["build", str(corpus), "-o", str(shared), "--cache", str(shared)])

    refused = [
        runner.invoke(app, ["build", str(corpus), "-o", str(shared), "--max-cost", limit])
        for limit in ("0", "inf", "nan")
    ]
    keyless = runner.invoke(app, ["build", str(corpus), "-o", str(shared), "--max-cost", "1"])

    assert inside.exit_code == 1 and "inside the corpus directory" in inside.stderr
    assert same.exit_code == 1 and "must be different files" in same.stderr
    for result in refused:
        message = " ".join(result.output.replace("\u2502", " ").split())
        assert result.exit_code == 2 and "--max-cost must be a positive finite" in message
    # A build that stops still reports what it spent.
    assert keyless.exit_code == 1 and "OPENAI_API_KEY" in keyless.stderr
    assert "usage: 0 provider calls" in keyless.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["corpus"]
    assert sorted(p.name for p in (corpus / "p1").iterdir()) == ["memo.md"]


def test_cli_build_that_reaches_the_spend_limit_exits_with_its_usage_and_no_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "p1").mkdir(parents=True)
    (corpus / "p1" / "memo.md").write_text("# Memo\n\nSeagrass at 59,494 ha.")
    api = ScriptedOpenAI()
    # Observing the one document costs $0.30, so the organisation call that follows is refused.
    card = ObservationDraft(
        title="Memo",
        doc_date=None,
        deliverable_type="memorandum",
        summary="Seagrass.",
        attributes=[],
        terms=[],
        craft=[],
    )
    api.script("/responses", structured_body(card, tokens=usage(100_000, 10_000)))
    monkeypatch.setattr(cli, "OpenAIGateway", api.gateway)
    output = tmp_path / "t.arkive.db"

    result = CliRunner().invoke(app, ["build", str(corpus), "-o", str(output), "--max-cost", "0.3"])

    assert result.exit_code == 1, result.output
    assert "reached the $0.3 limit" in result.stderr
    assert "usage: 1 provider calls" in result.stdout and "estimated $0.3000" in result.stdout
    assert api.request_count == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["corpus"]


def test_cli_builds_differently_organised_corpora_into_different_knowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "OpenAIGateway", lambda **options: FakeGateway())
    runner = CliRunner()
    summaries = {}
    for name, files in (("bathymetry", BATHYMETRY_CORPUS), ("campaigns", MARKETING_CORPUS)):
        corpus = tmp_path / name
        for rel_path, text in files.items():
            (corpus / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (corpus / rel_path).write_text(text, encoding="utf-8")
        output = tmp_path / f"{name}.arkive.db"
        built = runner.invoke(app, ["build", str(corpus), "-o", str(output)])
        assert built.exit_code == 0, built.output
        assert "build report: {" in built.stdout and f"wrote {output}" in built.stdout
        inspected = runner.invoke(app, ["inspect", str(output)])
        summaries[name] = json.loads(inspected.stdout)

    def knowledge(name: str) -> tuple[list[tuple[str, str]], list[str], list[str]]:
        summary = summaries[name]
        return (
            [(f["name"], f["binding"]) for f in summary["facets"]],
            [e["term"] for e in summary["glossary"]],
            [b["deliverable_type"] for b in summary["blueprints"]],
        )

    assert knowledge("bathymetry") == (
        [("project", "path"), ("client", "content")],
        ["LAT"],
        ["proposal", "survey memorandum"],
    )
    assert knowledge("campaigns") == (
        [("campaign", "path"), ("author", "content"), ("channel", "content")],
        ["Churn", "CTA", "ICP"],
        ["campaign brief", "review notes"],
    )
