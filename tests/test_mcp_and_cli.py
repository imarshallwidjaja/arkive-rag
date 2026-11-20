from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
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
from arkive_rag.models import ArkiveError, ObservationDraft, TopicPage, document_id
from arkive_rag.server import TopicCatalog, create_mcp_server
from arkive_rag.snapshot import (
    RESPONSE_MAX_BYTES,
    RESPONSE_MAX_LINES,
    TOPIC_PAGE_MARGIN_BYTES,
    QueryEmbedder,
    TopicSnapshot,
    response_text,
)
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
            (
                "prepare_draft",
                {
                    "topic": marine,
                    "project_inputs": "A harbour lidar memo on rainfall and soundings.",
                    "deliverable_type": "Survey Memorandum",
                },
            ),
            ("list_documents", {"topic": marine, "limit": 1, "offset": 1}),
        ],
        FakeGateway().embed_query,
    )
    topics, guide, memos, outline, hits, matches, section, page, soils, soils_grep = results[:10]
    terms, lat, packet, page_two = results[10:]

    assert [(t["topic"], t["document_count"]) for t in data(topics)["topics"]] == [
        ("bathymetry", 4),
        ("soils", 1),
    ]
    assert [len(t["corpus_fingerprint"]) for t in data(topics)["topics"]] == [64, 64]
    project = data(topics)["topics"][0]["facets"][0]
    assert (project["name"], project["values"]) == (
        "project",
        ["p101-harbour-lidar", "p102-estuary-multibeam", "p103-reef-sonar"],
    )
    assert f"[{CHART_MEMO}:" in data(guide)["guide_markdown"]
    assert [
        (t["deliverable_type"], t["document_count"]) for t in data(guide)["deliverable_types"]
    ] == [("document", 1), ("proposal", 1), ("survey memorandum", 1)]
    assert data(topics)["topics"][0]["deliverable_types"] == [
        "document",
        "proposal",
        "survey memorandum",
    ]
    assert (data(topics)["total"], data(topics)["next_offset"]) == (2, None)
    draft = data(packet)
    assert draft["deliverable_type"] == "survey memorandum"
    assert draft["exemplars"][0]["doc_id"] == HARBOUR_MEMO
    assert all(s["included"] for s in draft["exemplars"][0]["sections"])
    assert draft["drafting_guidance"].startswith("- Take every fact")
    assert [d["rel_path"] for d in data(page_two)["documents"]] == [
        "p101-harbour-lidar/harbour-memo.md"
    ]
    assert (data(page_two)["total"], data(page_two)["next_offset"]) == (4, 2)
    assert data(guide)["glossary_terms"] == ["LAT"]
    assert data(guide)["curator_notes"][0]["scope_path"] == "p103-reef-sonar/AGENTS.md"
    assert [d["rel_path"] for d in data(memos)["documents"]] == [
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
    assert [d["rel_path"] for d in data(soils)["documents"]] == ["p900-soils/soils-memo.md"]
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
            (
                "prepare_draft",
                {"topic": "bathymetry", "project_inputs": "x", "deliverable_type": "press release"},
            ),
            ("prepare_draft", {"topic": "bathymetry", "project_inputs": "Rainfall memo."}),
            ("grep", {"topic": "bathymetry", "pattern": "x", "max_matches": 51}),
        ],
        None,
    )
    search, grep, section, page, bad_topic, bad_project, bad_page, bad_facet, blank = results[:9]
    bad_type, offline_draft, too_many = results[9:]

    assert "OPENAI_API_KEY" in error_text(search) and "grep" in error_text(search)
    assert len(data(grep)) == 1
    assert data(section)["section_id"] == f"{CHART_MEMO}_sec_001"
    assert isinstance(page.content[1], ImageContent)
    assert error_text(bad_topic).endswith("unknown topic 'mining'; valid topics: bathymetry, soils")
    assert "valid values: p101-harbour-lidar" in error_text(bad_project)
    assert error_text(bad_page).endswith(f"{CHART_MEMO} has no page 9; it has pages 1-2")
    assert error_text(bad_facet).endswith("unknown facet 'stage'; valid facets: project, client")
    assert "glossary query must not be blank" in error_text(blank)
    assert error_text(bad_type).endswith(
        "valid types: document (1), proposal (1), survey memorandum (1)"
    )
    # Drafting needs no API key: without one, exemplars are ranked by BM25 alone.
    assert data(offline_draft)["exemplars"]
    assert "between 1 and 50" in error_text(too_many)


# OpenCode cuts a tool result at 51,200 bytes or 2,000 lines; Arkive keeps every result
# within its own lower bounds.
OPENCODE_MAX_BYTES = 51_200
OPENCODE_MAX_LINES = 2_000


def _with_large_catalogue(path: Path, topic: str | None = None) -> None:
    """Edit a snapshot past what a build writes: a content facet with a 20,000-character
    description and 1,600 values of about 110 characters, 1,300 more deliverable types with
    2,000-character descriptions, and optionally another topic name."""
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO facets VALUES ('region', 99, ?, 'content')", ("Region. " * 2_500,))
        values = [f"region-{n:04d}-" + "x" * 100 for n in range(1_600)]
        db.executemany(
            "INSERT INTO facet_values VALUES ('region', ?, ?, 'A region.', '[]')",
            [(v, n) for n, v in enumerate(values, start=1)],
        )
        db.executemany(
            "INSERT INTO facet_aliases VALUES ('region', ?, ?, ?)", [(v, v, v) for v in values]
        )
        db.executemany(
            "INSERT INTO deliverable_types VALUES (?, ?, ?)",
            [(f"type {n:04d} " + "y" * 90, 1_000 + n, "Kind. " * 330) for n in range(1_300)],
        )
        if topic is not None:
            db.execute("UPDATE corpus_meta SET value = ? WHERE key = 'topic'", (topic,))
    db.close()


def _oversized_snapshot(tmp_path: Path) -> Path:
    """A topic built to overflow every tool result: a 600-section manual defining a term in
    each section, a full-length section of two-byte characters, 30 long curator notes, 15
    reviews with long lessons, 150 small documents and a two-page PDF. The snapshot is then
    edited past what a build writes: the manual's summary is 60,000 characters and its title
    and second clause heading 20,000 three-byte characters each, the small documents are
    drafts it supersedes, one glossary term has four senses citing 500 sections each, the
    guide is 24,000 bytes of three-byte characters, the PDF's second page has 2,100 lines of
    native text and 3,000 of visual summary, and the catalogue is large
    (`_with_large_catalogue`)."""
    clauses = "".join(
        f"\n## Clause {i}\n\n**TERM{i:04d}** means clause {i} of the field manual. "
        + "The crew records the gauge datum at each station. " * 3
        + "\n"
        for i in range(600)
    )
    files = {
        "handbook/manual.md": f"# Field Manual\n\nDeliverable: field manual\n{clauses}",
        # Sections hold at most 32,000 characters; in Cyrillic that is about 60,000 bytes.
        "long/section.md": "# Long\n\n"
        + "\n".join(f"\u0421\u0442\u0440\u043e\u043a\u0430 {i} " * 4 for i in range(1_500)),
        **{
            f"notes-{i:02d}/AGENTS.md": f"# Note {i}\n\n" + f"Curator guidance {i}. " * 150
            for i in range(30)
        },
        **{
            f"reviews/review-{k:02d}.md": (
                f"# Review {k}\n\nDeliverable: field manual review\n\n"
                f"Reviewer: State the datum of every station in each clause, lesson {k}, "
                + "and keep the gauge tables in one order. "
                * 8
            )
            for k in range(15)
        },
        **{f"zz-drafts/draft-{n:03d}.md": f"# Draft {n}\n\nDraft {n}." for n in range(150)},
    }
    path = compile_snapshot(
        tmp_path, files, topic="oversized", pdfs={"scans/scan.pdf": [["# Scan"], ["# Page"]]}
    )
    manual = document_id("handbook/manual.md")
    wide = "\u6587" * 20_000
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE documents SET summary = ?, title = ? WHERE doc_id = ?",
            ("Long. " * 10_000, wide, manual),
        )
        db.execute(
            "UPDATE sections SET heading_path = ? WHERE section_id = ?", (wide, f"{manual}_sec_002")
        )
        db.execute(
            "UPDATE corpus_meta SET value = ? WHERE key = 'guide_markdown'", ("\u6587" * 8_000,)
        )
        db.execute(
            "UPDATE pages SET native_text = ?, visual_summary = ? WHERE page_number = 2",
            (
                "\n".join(f"\u884c {i}" for i in range(2_100)),
                "\n".join(f"Shape {i}." for i in range(3_000)),
            ),
        )
        db.execute(
            "UPDATE documents SET family_id = ?, version_status = 'superseded',"
            " superseded_by = ? WHERE rel_path LIKE 'zz-drafts/%'",
            (manual, manual),
        )
        db.execute("UPDATE documents SET version_status = 'latest' WHERE doc_id = ?", (manual,))
        (ordinal, entry_json) = db.execute(
            "SELECT ordinal, entry_json FROM glossary WHERE term = 'TERM0001'"
        ).fetchone()
        entry = json.loads(entry_json)
        cited = [f"{manual}_sec_{k:03d}" for k in range(2, 502)]
        entry["senses"] = [{"definition": f"Sense {n}.", "citations": cited} for n in range(4)]
        db.execute(
            "UPDATE glossary SET entry_json = ? WHERE ordinal = ?", (json.dumps(entry), ordinal)
        )
    db.close()
    _with_large_catalogue(path)
    return path


def test_every_tool_result_fits_one_opencode_tool_result(tmp_path: Path) -> None:
    path = _oversized_snapshot(tmp_path)
    # Nineteen more topics, each with the large catalogue, so the topics need several pages.
    small = compile_snapshot(tmp_path / "small", {"p1/memo.md": "# Memo\n\nGauge datum."})
    others = []
    for n in range(19):
        copy = tmp_path / f"topic-{n:02d}.arkive.db"
        copy.write_bytes(small.read_bytes())
        _with_large_catalogue(copy, topic=f"topic-{n:02d}")
        others.append(copy)
    manual = document_id("handbook/manual.md")
    long_section = f"{document_id('long/section.md')}_sec_001"
    scan = document_id("scans/scan.pdf")
    topic = "oversized"
    calls: list[tuple[str, dict[str, Any]]] = [
        ("list_topics", {}),
        ("get_topic_guide", {"topic": topic}),
        ("list_documents", {"topic": topic, "limit": 100}),
        ("get_document_outline", {"topic": topic, "doc_id": manual}),
        ("search", {"topic": topic, "query": "gauge datum station clause", "top_k": 50}),
        ("grep", {"topic": topic, "pattern": "gauge datum", "max_matches": 50}),
        ("read_section", {"topic": topic, "section_id": long_section}),
        ("lookup_glossary", {"topic": topic}),
        ("lookup_glossary", {"topic": topic, "query": "term"}),
        ("lookup_glossary", {"topic": topic, "query": "TERM0001"}),
        (
            "prepare_draft",
            {
                "topic": topic,
                "project_inputs": "A field manual of gauge datum clauses for every station.",
                "deliverable_type": "field manual",
            },
        ),
        ("lookup_glossary", {"topic": topic, "query": "term " * 400}),
        ("view_page", {"topic": topic, "doc_id": scan, "page_number": 2}),
        ("grep", {"topic": topic, "pattern": "clause 0 of the field manual", "max_matches": 1}),
        ("read_section", {"topic": topic, "section_id": f"{manual}_sec_002"}),
        ("list_documents", {"topic": topic, "filters": {"region": "z" * 5_000}}),
        ("prepare_draft", {"topic": topic, "project_inputs": "x", "deliverable_type": "q" * 5_000}),
        ("get_topic_guide", {"topic": "t" * 5_000}),
        ("read_section", {"topic": topic, "section_id": "s" * 5_000}),
    ]

    results = call_tools((path, *others), calls, FakeGateway().embed_query)
    pages = [data(results[0])]
    while pages[-1]["next_offset"] is not None:
        (more,) = call_tools(
            (path, *others), [("list_topics", {"offset": pages[-1]["next_offset"]})], None
        )
        assert len(response_text_of(more).encode()) <= RESPONSE_MAX_BYTES
        pages.append(data(more))

    assert RESPONSE_MAX_BYTES < OPENCODE_MAX_BYTES and RESPONSE_MAX_LINES < OPENCODE_MAX_LINES
    for (name, _), result in zip(calls, results, strict=True):
        texts = [c.text for c in result.content if isinstance(c, TextContent)]
        rendered = "\n\n".join(texts)
        assert len(rendered.encode()) <= RESPONSE_MAX_BYTES, name
        assert rendered.count("\n") + 1 <= RESPONSE_MAX_LINES, name
        if not result.is_error and name != "view_page":
            # One compact JSON block, the same value as the structured result.
            (text,) = texts
            assert json.loads(text) == data(result), name
    topics, guide, page, outline, hits, matches, part, terms, matched, entry, packet = [
        data(r) for r in results[:11]
    ]
    assert "at most 1000" in error_text(results[11])
    view, long_heading, long_part = results[12], data(results[13]), data(results[14])
    bad_value, bad_type, bad_topic, bad_section = (error_text(r) for r in results[15:])
    # Twenty topics come in several pages, each topic once and in order.
    assert len(pages) > 1 and topics["total"] == 20
    listed = [t["topic"] for p in pages for t in p["topics"]]
    assert listed == ["oversized", *(f"topic-{n:02d}" for n in range(19))]
    (region,) = [f for f in pages[0]["topics"][0]["facets"] if f["name"] == "region"]
    assert len(region["values"]) + region["omitted_values"] == 1_600
    assert len(region["values"]) <= 50
    assert hits and matches
    # A 24,000-byte guide still comes with every facet name and the types that fit.
    assert guide["guide_markdown"] == "\u6587" * 8_000
    assert guide["omitted_deliverable_types"] > 0 and guide["deliverable_types"]
    assert "region" in [f["name"] for f in guide["facets"]]
    assert guide["omitted_curator_notes"] > 0 or guide["curator_notes"]
    assert guide["omitted_glossary_terms"] > 0
    # The page image comes whole beside a caption cut to the bound, saying what it left out.
    caption, image = view.content
    assert isinstance(caption, TextContent) and isinstance(image, ImageContent)
    assert caption.text.count("\n") < 1_500 and "lines omitted ..." in caption.text
    assert "\u884c 0" in caption.text and "Shape 0." in caption.text
    # A match in a section with a 20,000-character heading is returned with the heading cut.
    ((match,),) = [long_heading]
    assert match["section_id"] == f"{manual}_sec_002"
    assert match["heading_path"].endswith("\u2026") and len(match["heading_path"]) == 500
    assert match["doc_title"].endswith("\u2026") and len(match["doc_title"]) == 500
    assert long_part["heading_path"].endswith("\u2026") and long_part["markdown"]
    # Errors quote at most 200 characters of the input and name at most 40 choices.
    assert "valid values: region-0000-" in bad_value and "(+1560 more)" in bad_value
    assert re.search(r", \.\.\. \(\+\d+ more\)$", bad_type) and "(5000 characters)" in bad_type
    assert "unknown topic" in bad_topic and "(5000 characters)" in bad_topic
    assert "unknown section" in bad_section and "(5000 characters)" in bad_section
    assert page["documents"][0]["summary"].endswith("\u2026")
    assert (outline["total_sections"], outline["offset"]) == (601, 0)
    assert 0 < len(outline["sections"]) == outline["next_offset"] < 601
    assert part["next_offset"] == len(part["markdown"]) < part["total_chars"]
    assert part["markdown"].endswith("\n")
    assert terms["total"] == matched["total"] == 600 and matched["next_offset"]
    (term,) = entry["matches"]
    for sense in term["senses"]:
        assert sense["omitted_citations"] > 0
        assert len(sense["citations"]) + sense["omitted_citations"] == 500
    (exemplar,) = [e for e in packet["exemplars"] if e["doc_id"] == manual]
    assert len(exemplar["title"]) == 500
    assert exemplar["summary"].endswith("\u2026") and exemplar["omitted_family_members"] > 0
    assert len(exemplar["family_doc_ids"]) + exemplar["omitted_family_members"] == 150
    assert packet["lessons"] and packet["omitted_lessons"] > 0
    assert packet["curator_notes"] and packet["omitted_curator_notes"] > 0


def response_text_of(result: CallToolResult) -> str:
    return "\n\n".join(c.text for c in result.content if isinstance(c, TextContent))


def test_list_topics_summarises_only_its_page_and_inspect_shows_the_whole_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = compile_snapshot(tmp_path / "small", {"p1/memo.md": "# Memo\n\nGauge datum."})
    paths = []
    for n in range(12):
        copy = tmp_path / f"topic-{n:02d}.arkive.db"
        copy.write_bytes(small.read_bytes())
        _with_large_catalogue(copy, topic=f"topic-{n:02d}")
        paths.append(copy)
    # The first topic's title makes its summary 44,990 bytes uncut: a result on its own, but
    # not inside a page.
    with TopicSnapshot.open(paths[0]) as snap:
        rest = len(response_text(snap.summary()).encode()) - len(snap.title)
    title = "T" * (44_990 - rest)
    # The second topic has more facet values than one summary can name.
    with sqlite3.connect(paths[1]) as db:
        db.execute("UPDATE corpus_meta SET value = ? WHERE key = 'title'", (title,))
        for k in range(7):
            zone = f"zone{k}"
            values = [f"{zone}-{n:04d}-" + "z" * 100 for n in range(60)]
            db.execute("INSERT INTO facets VALUES (?, ?, 'Zone.', 'content')", (zone, 100 + k))
            db.executemany(
                "INSERT INTO facet_values VALUES (?, ?, ?, 'A zone.', '[]')",
                [(zone, v, n) for n, v in enumerate(values, start=1)],
            )
            db.executemany(
                "INSERT INTO facet_aliases VALUES (?, ?, ?, ?)", [(zone, v, v, v) for v in values]
            )
    db.close()
    with sqlite3.connect(paths[0]) as db:
        db.execute("UPDATE corpus_meta SET value = ? WHERE key = 'title'", (title,))
    db.close()
    summarised: list[str] = []
    summary = TopicSnapshot.summary

    def counted(self: TopicSnapshot) -> Any:
        summarised.append(self.topic)
        return summary(self)

    monkeypatch.setattr(TopicSnapshot, "summary", counted)

    pages: list[dict[str, Any]] = []
    offset: int | None = 0
    while offset is not None:
        summarised.clear()
        (result,) = call_tools(tuple(paths), [("list_topics", {"offset": offset})], None)
        assert len(response_text_of(result).encode()) <= RESPONSE_MAX_BYTES
        pages.append(data(result))
        listed = [t["topic"] for t in pages[-1]["topics"]]
        offset = pages[-1]["next_offset"]
        # Only the page's topics are summarised, and the one that did not fit after them.
        assert summarised == [*listed, *([f"topic-{offset:02d}"] if offset else [])]
    monkeypatch.undo()

    assert len(pages) > 1
    assert [t["topic"] for p in pages for t in p["topics"]] == [f"topic-{n:02d}" for n in range(12)]
    first, crowded = pages[0]["topics"][0], pages[1]["topics"][0]
    assert first["title"] == "T" * 499 + "\u2026" and crowded["title"] == first["title"]
    zones = [f for f in crowded["facets"] if f["name"].startswith("zone")]
    assert zones and all(f["omitted_values"] > 10 for f in zones)
    with TopicSnapshot.open(paths[1]) as snap:
        alone = TopicPage(topics=[snap.summary()], total=1, offset=0, next_offset=None)
    assert len(response_text(alone).encode()) <= RESPONSE_MAX_BYTES - TOPIC_PAGE_MARGIN_BYTES

    inspected = CliRunner().invoke(app, ["inspect", str(paths[1])])
    assert inspected.exit_code == 0, inspected.output
    whole = json.loads(inspected.stdout)
    assert whole["title"] == title
    assert [len(f["values"]) for f in whole["facets"] if f["name"].startswith("zone")] == [60] * 7
    assert len(whole["deliverable_types"]) > 1_300


def test_catalog_rejects_two_snapshots_with_the_same_topic(
    snapshots: tuple[Path, Path], tmp_path: Path
) -> None:
    copy = tmp_path / "copy.arkive.db"
    copy.write_bytes(snapshots[0].read_bytes())
    with pytest.raises(ArkiveError, match="topic 'bathymetry' is served by both"):
        TopicCatalog([snapshots[0], copy])


def test_cli_inspect_and_grep_work_offline(
    snapshots: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
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
        "is_feedback": False,
        "feedback_for_types": [],
        "family_id": CHART_MEMO,
        "version_status": summary["documents"][0]["version_status"],
        "superseded_by": None,
        "facets": {"project": ["p101-harbour-lidar"]},
        "sections": 2,
    }
    assert [f["name"] for f in summary["facets"]] == ["project", "client"]
    assert [e["term"] for e in summary["glossary"]] == ["LAT"]
    assert [t["deliverable_type"] for t in summary["deliverable_types"]] == [
        "document",
        "proposal",
        "survey memorandum",
    ]
    assert summary["feedback_lessons"] == []
    with TopicSnapshot.open(snapshots[0]) as snap:
        assert summary["recurring_units"] == snap.recurring_unit_count
    assert summary["build_report"]["dropped_feedback_lessons"] == 0
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
    monkeypatch.chdir(tmp_path)
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
            [t["deliverable_type"] for t in summary["deliverable_types"]],
        )

    assert knowledge("bathymetry") == (
        [("project", "path"), ("client", "content")],
        ["LAT"],
        ["proposal", "survey memorandum"],
    )
    assert [
        (lesson["statement"], lesson["target_types"])
        for lesson in summaries["campaigns"]["feedback_lessons"]
    ] == [("Cut the jargon from the opening line.", ["campaign brief"])]
    assert knowledge("campaigns") == (
        [("campaign", "path"), ("author", "content"), ("channel", "content")],
        ["Churn", "CTA", "ICP"],
        ["campaign brief", "review notes"],
    )
