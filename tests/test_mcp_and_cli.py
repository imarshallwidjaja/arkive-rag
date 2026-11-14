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

from arkive_rag import __version__
from arkive_rag.cli import app
from arkive_rag.models import ArkiveError
from arkive_rag.server import TopicCatalog, create_mcp_server
from arkive_rag.snapshot import QueryEmbedder, TopicSnapshot
from tests.support import BATHYMETRY_CORPUS, FakeGateway, compile_snapshot

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
            ("list_documents", {"topic": marine, "stage": "memorandum"}),
            ("get_document_outline", {"topic": marine, "doc_id": "doc_001"}),
            ("search", {"topic": marine, "query": "nine year median rainfall", "top_k": 3}),
            ("grep", {"topic": marine, "pattern": "330.9"}),
            ("read_section", {"topic": marine, "section_id": "doc_001_sec_002"}),
            ("view_page", {"topic": marine, "doc_id": "doc_001", "page_number": 2}),
            ("list_documents", {"topic": "soils"}),
            ("grep", {"topic": "soils", "pattern": "330.9"}),
        ],
        FakeGateway().embed_query,
    )
    topics, guide, memos, outline, hits, matches, section, page, soils, soils_grep = results

    assert [(t["topic"], t["document_count"]) for t in data(topics)] == [
        ("bathymetry", 4),
        ("soils", 1),
    ]
    assert [len(t["corpus_fingerprint"]) for t in data(topics)] == [64, 64]
    assert "[doc_00" in data(guide)["synthesis_markdown"]
    assert data(guide)["curator_notes"][0]["scope_path"] == "p103-reef-sonar/AGENTS.md"
    assert [d["rel_path"] for d in data(memos)] == [
        "p101-harbour-lidar/harbour-chart-memo.pdf",
        "p101-harbour-lidar/harbour-memo.md",
    ]
    assert [s["heading_path"] for s in data(outline)["sections"]] == ["Rainfall", "Outputs"]
    assert len(data(hits)) == 3 and "330.9" in data(hits)[0]["content"]
    assert {m["doc_id"] for m in data(matches)} == {"doc_001", "doc_002"}
    assert data(section)["markdown"] == "# Outputs\n\nRaster estuary_2024_grid.tif"
    assert data(section)["prev_section_id"] == "doc_001_sec_001"
    text, image = page.content
    assert isinstance(text, TextContent) and "page 2 of 2" in text.text
    assert isinstance(image, ImageContent) and image.mime_type == "image/webp"
    assert base64.b64decode(image.data)[8:12] == b"WEBP"
    assert [d["rel_path"] for d in data(soils)] == ["p900-soils/soils-memo.md"]
    assert data(soils_grep) == []


def test_mcp_errors_name_valid_options_and_offline_search_explains_itself(
    snapshots: tuple[Path, Path],
) -> None:
    results = call_tools(
        snapshots,
        [
            ("search", {"topic": "bathymetry", "query": "rainfall"}),
            ("grep", {"topic": "bathymetry", "pattern": "Sentinel-2", "max_matches": 1}),
            ("read_section", {"topic": "bathymetry", "section_id": "doc_001_sec_001"}),
            ("view_page", {"topic": "bathymetry", "doc_id": "doc_001", "page_number": 1}),
            ("list_documents", {"topic": "mining"}),
            ("list_documents", {"topic": "bathymetry", "project": "p999"}),
            ("view_page", {"topic": "bathymetry", "doc_id": "doc_001", "page_number": 9}),
        ],
        None,
    )
    search, grep, section, page, bad_topic, bad_project, bad_page = results

    assert "OPENAI_API_KEY" in error_text(search) and "grep" in error_text(search)
    assert len(data(grep)) == 1
    assert data(section)["section_id"] == "doc_001_sec_001"
    assert isinstance(page.content[1], ImageContent)
    assert error_text(bad_topic).endswith("unknown topic 'mining'; valid topics: bathymetry, soils")
    assert "valid projects: p101-harbour-lidar" in error_text(bad_project)
    assert error_text(bad_page).endswith("doc_001 has no page 9; it has pages 1-2")


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
        app, ["grep", str(snapshots[0]), "km2", "--project", "p101-harbour-lidar"]
    )
    bad_stage = runner.invoke(app, ["grep", str(snapshots[0]), "bathymetry", "--stage", "draft"])
    search = runner.invoke(app, ["search", str(snapshots[0]), "rainfall"])

    assert inspected.exit_code == 0, inspected.output
    summary = json.loads(inspected.stdout)
    assert (summary["topic"], summary["document_count"]) == ("bathymetry", 4)
    with TopicSnapshot.open(snapshots[0]) as snap:
        assert summary["corpus_fingerprint"] == snap.corpus_fingerprint
    assert summary["arkive_version"] == __version__
    source = snapshots[0].parent / "bathymetry-corpus" / "p101-harbour-lidar/harbour-chart-memo.pdf"
    assert summary["documents"][0] == {
        "doc_id": "doc_001",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rel_path": "p101-harbour-lidar/harbour-chart-memo.pdf",
        "format": "pdf",
        "project_id": "p101-harbour-lidar",
        "client": "Northwind Marine",
        "stage": "memorandum",
        "title": "Harbour Chart Memo",
        "doc_date": None,
        "page_count": 2,
        "sections": 2,
    }
    assert grepped.exit_code == 0
    assert "59,494 km²" in json.loads(grepped.stdout)[0]["snippet"]
    assert bad_stage.exit_code == 1 and "valid stages" in bad_stage.stderr
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
    assert {m["doc_id"] for m in data(grep)} == {"doc_001", "doc_002"}
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

    assert inside.exit_code == 1 and "inside the corpus directory" in inside.stderr
    assert same.exit_code == 1 and "must be different files" in same.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["corpus"]
    assert sorted(p.name for p in (corpus / "p1").iterdir()) == ["memo.md"]
