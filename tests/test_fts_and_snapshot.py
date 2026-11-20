from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import stat
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from arkive_rag import __version__
from arkive_rag import snapshot as snapshot_module
from arkive_rag.compiler import compile_corpus
from arkive_rag.models import (
    DraftPacket,
    FamilyRecurrence,
    InvalidFilterError,
    NotFoundError,
    ProviderError,
    SearchHit,
    SnapshotIntegrityError,
    UnsafePathError,
    choices,
    document_id,
)
from arkive_rag.snapshot import (
    DRAFTING_GUIDANCE,
    MAX_GREP_MATCHES,
    MAX_OTHER_CANDIDATES,
    MAX_TOP_K,
    RESPONSE_MAX_BYTES,
    SCHEMA_VERSION,
    SNIPPET_CHARS,
    TopicSnapshot,
    build_fts5_query,
    fits_response,
    response_text,
    write_snapshot,
)
from tests.support import (
    BATHYMETRY_CORPUS,
    CAMPAIGN_NOTE,
    ESTUARY_PROPOSAL,
    LOYALTY_BRIEF,
    MARKETING_CORPUS,
    SPRING_BRIEF,
    SPRING_REVIEW,
    FakeGateway,
    compile_snapshot,
    write_pdf,
)

FTS_BATTERY = [
    "Sentinel-2",
    "sentinel 2",
    "SENTINEL\u20112",
    "\uff33\uff45\uff4e\uff54\uff49\uff4e\uff45\uff4c",
    "S\u00c9NTINEL",
    "p-value: 0.05",
    "p-value",
    "0.05",
    "Nine Year Median = 330.9 mm",
    "330.9",
    "$330.9",
    "harbour_dry_depth_thresh.tif",
    "NORTHWIND AND NOT Bathymetry?",
    "C12-SUM(C13:C15)",
    "km\u00b2",
    "km2",
    '"unbalanced rainfall quote',
    "NEAR(northwind sentinel)",
    "northwind*",
    "-northwind",
    "+rainfall",
    "(rainfall)",
    "rainfall)",
    "{rainfall}",
    "[rainfall]",
    "rainfall:",
    "column:rainfall",
    "^rainfall",
    "rainfall OR",
    "OR rainfall",
    "AND",
    "NOT rainfall",
    "What nine-year median rainfall reference does the chart show?",
    "Where is the reef Sentinel-2 case study?",
    "which file holds the dry season threshold raster",
    "how big is the harbour area",
    "2016\u20132017",
    "2016-2017",
    "lidar; DROP TABLE chunks;--",
    "rainfall'",
    "\u2018rainfall\u2019",
    "\u00abrainfall\u00bb",
    "rainfall\u2026",
    "rainfall\\",
    "median|rainfall",
    "median&rainfall%",
    "#Rainfall @Northwind",
    "fee 120,000",
    "tide gauge depth",
    "x" * 300 + " rainfall",
]


@pytest.fixture
def snapshot(tmp_path: Path) -> Iterator[TopicSnapshot]:
    with TopicSnapshot.open(compile_snapshot(tmp_path, BATHYMETRY_CORPUS)) as snap:
        yield snap


def query(snap: TopicSnapshot, text: str) -> list[SearchHit]:
    return snap.search(text, FakeGateway().embed_query)


def test_fts_battery_never_raises_and_always_finds_lexical_hits(snapshot: TopicSnapshot) -> None:
    assert len(FTS_BATTERY) == 50
    for text in FTS_BATTERY:
        expression = build_fts5_query(text)
        assert expression is not None, text
        assert any(hit.lexical_rank is not None for hit in query(snapshot, text)), text


@pytest.mark.parametrize("text", ["***", "!!! ???", "\u2014", "__"])
def test_queries_without_tokens_fall_back_to_dense_ranking_only(
    snapshot: TopicSnapshot, text: str
) -> None:
    assert build_fts5_query(text) is None
    hits = query(snapshot, text)
    assert hits and all(hit.lexical_rank is None for hit in hits)


def test_invalid_search_requests_fail_before_any_query_embedding(
    snapshot: TopicSnapshot,
) -> None:
    calls: list[str] = []

    def embed(text: str) -> NDArray[np.float32]:
        calls.append(text)
        return FakeGateway().embed_query(text)

    assert build_fts5_query("") is None
    bad: list[tuple[str, dict[str, str]]] = [
        ("", {}),
        (" \n ", {}),
        ("bathymetry", {"project": "p999"}),
        ("bathymetry", {"stage": "memorandum"}),
    ]
    for text, filters in bad:
        with pytest.raises(InvalidFilterError):
            snapshot.search(text, embed, filters=filters)
    with pytest.raises(InvalidFilterError, match="top_k"):
        snapshot.search("bathymetry", embed, top_k=MAX_TOP_K + 1)
    assert calls == []


def test_fts_query_quotes_tokens_adds_phrases_and_drops_filler() -> None:
    assert build_fts5_query("Sentinel-2") == '"sentinel" OR "2" OR "sentinel 2"'
    assert build_fts5_query("What is the median?") == (
        '"what" OR "is" OR "the" OR "median" OR "the median"'
    )
    assert build_fts5_query("where is the nine-year median") == (
        '"nine" OR "year" OR "median" OR "the nine" OR "nine year" OR "year median"'
    )
    assert build_fts5_query("km\u00b2") == '"km2"'
    assert build_fts5_query("harbour_dry.tif") == (
        '"harbour" OR "dry" OR "tif" OR "harbour dry" OR "dry tif"'
    )


def test_exact_phrase_ranks_the_owning_chunk_first(snapshot: TopicSnapshot) -> None:
    top = query(snapshot, "harbour_dry_depth_thresh.tif")[0]
    assert (top.lexical_rank, top.kind) == (1, "table")
    assert "harbour_dry_depth_thresh.tif" in top.content


class KeywordGateway(FakeGateway):
    """Embeds texts containing 'dense-match' on axis 0 and everything else on axis 1."""

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        rows = np.zeros((len(texts), self.embedding_dimensions), dtype=np.float32)
        for row, text in zip(rows, texts, strict=True):
            row[0 if "dense-match" in text else 1] = 1.0
        return rows

    def embed_query(self, query: str) -> NDArray[np.float32]:
        vector = np.zeros(self.embedding_dimensions, dtype=np.float32)
        vector[0] = 1.0
        return vector


def test_rrf_ranks_chunks_found_by_both_channels_above_single_channel_hits(
    tmp_path: Path,
) -> None:
    files = {
        "p1/a.md": "# A\n\nalpha appears in this longer lexical only paragraph",
        "p1/b.md": "# B\n\nalpha dense-match",
        "p1/c.md": "# C\n\nzeta dense-match only",
    }
    gateway = KeywordGateway()
    path = compile_snapshot(tmp_path, files, gateway=gateway)
    with TopicSnapshot.open(path) as snap:
        hits = snap.search("alpha", gateway.embed_query, top_k=3)

    assert [h.content for h in hits] == [
        "alpha dense-match",
        "alpha appears in this longer lexical only paragraph",
        "zeta dense-match only",
    ]
    assert (hits[0].lexical_rank, hits[0].dense_rank) == (1, 1)
    assert hits[0].rrf_score == pytest.approx(2 / 61)
    assert hits[1].dense_rank == 3 and hits[2].lexical_rank is None


def test_rank_channels_order_one_embedding_by_each_channel_and_handle_empty_snapshots(
    tmp_path: Path,
) -> None:
    files = {
        "p1/a.md": "# A\n\nalpha appears in this longer lexical only paragraph",
        "p1/b.md": "# B\n\nalpha dense-match",
        "p1/c.md": "# C\n\nzeta dense-match only",
    }
    gateway = KeywordGateway()
    calls: list[str] = []

    def embed(text: str) -> NDArray[np.float32]:
        calls.append(text)
        return gateway.embed_query(text)

    with TopicSnapshot.open(compile_snapshot(tmp_path, files, gateway=gateway)) as snap:
        channels = snap.rank_channels("alpha", embed, top_k=3)
        assert channels["hybrid_rrf"] == snap.search("alpha", gateway.embed_query, top_k=3)
    assert calls == ["alpha"]
    bm25, dense = channels["bm25_only"], channels["dense_only"]
    assert [h.lexical_rank for h in bm25] == list(range(1, len(bm25) + 1))
    assert {h.content for h in bm25} == {
        "alpha dense-match",
        "alpha appears in this longer lexical only paragraph",
    }
    assert [h.dense_rank for h in dense] == [1, 2, 3]
    assert dense[0].content == "alpha dense-match"

    (tmp_path / "empty" / "empty-corpus").mkdir(parents=True)
    empty = compile_snapshot(tmp_path / "empty", {}, topic="empty")
    with TopicSnapshot.open(empty) as snap:
        assert snap.rank_channels("alpha", FakeGateway().embed_query) == {
            "bm25_only": [],
            "dense_only": [],
            "hybrid_rrf": [],
        }


def test_error_choices_show_each_value_on_one_line() -> None:
    multiline = "Region\n\n  north\r\nof the\tbay\n" * 3
    listed = choices([multiline, "south", "x" * 300])

    assert "\n" not in listed and "\r" not in listed and "\t" not in listed
    assert listed.startswith("Region north of the bay Region north of the bay")
    assert listed.endswith(f", south, {'x' * 200}\u2026")


def test_filter_errors_name_the_valid_facets_and_values(snapshot: TopicSnapshot) -> None:
    vector = FakeGateway().embed_query
    with pytest.raises(
        InvalidFilterError, match="unknown facet 'stage'; valid facets: project, client"
    ):
        snapshot.list_documents(filters={"stage": "memorandum"})
    with pytest.raises(
        InvalidFilterError,
        match="unknown project value 'p999'; valid values: p101-harbour-lidar,"
        " p102-estuary-multibeam, p103-reef-sonar",
    ):
        snapshot.search("bathymetry", vector, filters={"project": "p999"})
    with pytest.raises(InvalidFilterError, match="unknown client value 'Fabrikam'"):
        snapshot.grep("bathymetry", filters={"client": "Fabrikam"})
    for blank in ("", " \t"):
        with pytest.raises(InvalidFilterError, match="client filter value must not be blank"):
            snapshot.list_documents(filters={"client": blank})
    with pytest.raises(InvalidFilterError, match="top_k"):
        snapshot.search("bathymetry", vector, top_k=0)
    with pytest.raises(ProviderError, match="dimension 64"):
        snapshot.search("bathymetry", lambda _: np.ones(3, dtype=np.float32))


def test_filters_combine_with_and_resolve_aliases_and_restrict_every_read(
    snapshot: TopicSnapshot,
) -> None:
    def paths(filters: dict[str, str]) -> list[str]:
        return [d.rel_path for d in snapshot.list_documents(filters=filters)]

    assert paths({}) == [d.rel_path for d in snapshot.list_documents()]
    assert len(paths({})) == 3
    assert paths({"project": "p103-reef-sonar"}) == ["p103-reef-sonar/AGENTS.md"]
    # Values match case-insensitively, and an alias resolves to its canonical value.
    assert paths({"client": "northwind"}) == ["p101-harbour-lidar/harbour-memo.md"]
    assert paths({"client": "NORTHWIND MARINE", "project": "p101-harbour-lidar"}) == [
        "p101-harbour-lidar/harbour-memo.md"
    ]
    assert paths({"client": "Northwind Marine", "project": "p102-estuary-multibeam"}) == []
    assert snapshot.resolve_filters({}) is None

    hits = snapshot.search("Sentinel-2", FakeGateway().embed_query, filters={"client": "Contoso"})
    assert hits and {tuple(h.facets["client"]) for h in hits} == {("Contoso Ports",)}
    assert {m.facets["project"][0] for m in snapshot.grep("sentinel-2")} == {
        "p101-harbour-lidar",
        "p102-estuary-multibeam",
        "p103-reef-sonar",
    }
    only_estuary = snapshot.grep("sentinel-2", filters={"project": "p102-estuary-multibeam"})
    assert {m.doc_id for m in only_estuary} == {ESTUARY_PROPOSAL}
    assert (
        snapshot.grep("sentinel-2", filters={"client": "Contoso", "project": "p101-harbour-lidar"})
        == []
    )


def test_every_read_follows_path_order_not_the_hash_order_of_document_ids(
    tmp_path: Path,
) -> None:
    paths = ("north.md", "south.md", "west.md")
    ids = [document_id(p) for p in paths]
    assert sorted(ids) == ids[::-1], "the paths' IDs must sort against path order"
    # Identical documents tie in every ranking, so only the tie-break orders them.
    same = "# Tide\n\nThe harbour tide gauge reads two metres at low water.\n"
    path = compile_snapshot(tmp_path, dict.fromkeys(paths, same), topic="tides")

    with TopicSnapshot.open(path) as snap:
        assert [d.doc_id for d in snap.list_documents()] == ids
        assert [m.doc_id for m in snap.grep("tide gauge")] == ids
        channels = snap.rank_channels("harbour tide gauge", FakeGateway().embed_query)
        for channel, hits in channels.items():
            assert [h.doc_id for h in hits] == ids, channel


def test_curator_guidance_answers_status_questions(snapshot: TopicSnapshot) -> None:
    text = "Where is the reef Sentinel-2 case study?"
    top = snapshot.search(text, FakeGateway().embed_query, top_k=1)[0]
    assert (top.kind, top.facets) == ("guidance", {"project": ["p103-reef-sonar"]})
    assert snapshot.document(snapshot.grep("case study")[0].doc_id).is_curator_note


def test_grep_folds_dashes_compatibility_forms_and_case(snapshot: TopicSnapshot) -> None:
    (dash,) = snapshot.grep("2016-2017")
    assert "2016\u20132017" in dash.snippet
    assert dash.heading_path == "Rainfall" and dash.line_number == 3
    assert snapshot.grep("KM2")[0].snippet.count("km\u00b2") == 1
    assert len(snapshot.grep("C12-SUM(C13:C15)")) == 1
    assert len(snapshot.grep("bathymetry", max_matches=2)) == 2
    assert snapshot.grep("no such text anywhere") == []
    with pytest.raises(InvalidFilterError, match="blank"):
        snapshot.grep("  ")


def test_sections_link_to_neighbours_and_missing_ids_raise(snapshot: TopicSnapshot) -> None:
    doc = next(d for d in snapshot.list_documents() if d.deliverable_type == "survey memorandum")
    outline = snapshot.outline(doc.doc_id)
    first, second = outline.sections[0], outline.sections[1]
    view = snapshot.read_section(second.section_id)

    assert second.heading_path == "Rainfall > Statistics"
    assert (view.prev_section_id, view.next_section_id) == (
        first.section_id,
        outline.sections[2].section_id,
    )
    assert snapshot.read_section(first.section_id).prev_section_id is None
    with pytest.raises(NotFoundError, match="doc_001_sec_999"):
        snapshot.read_section("doc_001_sec_999")
    with pytest.raises(NotFoundError, match="no rendered pages"):
        snapshot.view_page(doc.doc_id, 1)
    with pytest.raises(NotFoundError, match="doc_404"):
        snapshot.outline("doc_404")


def test_write_is_atomic_and_keeps_the_previous_snapshot_on_failure(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "p1").mkdir(parents=True)
    (corpus_dir / "p1" / "memo.md").write_text("# A\n\nText")
    corpus = compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T")
    output = tmp_path / "t.arkive.db"
    write_snapshot(corpus, output)
    before = output.read_bytes()

    broken = replace(corpus, embeddings=corpus.embeddings[:, :3])
    with pytest.raises(SnapshotIntegrityError, match="embeddings shape"):
        write_snapshot(broken, output)
    duplicate = replace(corpus, documents=corpus.documents * 2)
    with pytest.raises(SnapshotIntegrityError, match="UNIQUE constraint"):
        write_snapshot(replace(duplicate, embeddings=np.vstack([corpus.embeddings] * 2)), output)

    assert output.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["corpus", "t.arkive.db"]


def test_snapshot_files_are_private_replacements_keep_mode_and_stay_out_of_the_corpus(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "p1").mkdir(parents=True)
    (corpus_dir / "p1" / "memo.md").write_text("# A\n\nText")
    corpus = compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T")
    output = tmp_path / "new-dir" / "t.arkive.db"

    write_snapshot(corpus, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    output.chmod(0o640)
    write_snapshot(corpus, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o640
    with pytest.raises(UnsafePathError, match="inside the corpus"):
        write_snapshot(corpus, corpus_dir / "t.arkive.db")
    assert sorted(p.name for p in output.parent.iterdir()) == ["t.arkive.db"]
    assert sorted(p.name for p in (corpus_dir / "p1").iterdir()) == ["memo.md"]


def test_snapshot_reports_a_content_addressed_corpus_fingerprint(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "p1").mkdir(parents=True)
    (corpus_dir / "p1" / "memo.md").write_text("# A\n\nDepth 12.5 m")
    (corpus_dir / "AGENTS.md").write_text("# Notes\n\nCurated.")
    corpus = compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T")
    rel_paths = sorted(doc.rel_path for doc in corpus.documents)
    lines = [
        f"{rel}\0{hashlib.sha256((corpus_dir / rel).read_bytes()).hexdigest()}" for rel in rel_paths
    ]
    expected = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    assert corpus.corpus_fingerprint == expected
    rebuilt = compile_corpus(corpus_dir, KeywordGateway(), topic="other", title="Other")
    assert rebuilt.corpus_fingerprint == expected

    output = tmp_path / "t.arkive.db"
    write_snapshot(corpus, output)
    with TopicSnapshot.open(output) as snap:
        summary = snap.summary()
        sha256s = {d.rel_path: d.sha256 for d in snap.list_documents()}
    assert (summary.corpus_fingerprint, summary.arkive_version) == (expected, __version__)
    assert sha256s["p1/memo.md"] == hashlib.sha256(b"# A\n\nDepth 12.5 m").hexdigest()

    # Every metadata key is required.
    unrecorded = tmp_path / "unrecorded.arkive.db"
    shutil.copy(output, unrecorded)
    with sqlite3.connect(unrecorded) as db:
        db.execute("DELETE FROM corpus_meta WHERE key IN ('corpus_fingerprint', 'arkive_version')")
    db.close()
    with pytest.raises(SnapshotIntegrityError, match="lacks corpus_fingerprint, arkive_version"):
        TopicSnapshot.open(unrecorded)

    tampered = tmp_path / "tampered.arkive.db"
    shutil.copy(output, tampered)
    with sqlite3.connect(tampered) as db:
        db.execute("UPDATE documents SET sha256 = 'f00' WHERE rel_path = 'p1/memo.md'")
    db.close()
    with pytest.raises(SnapshotIntegrityError, match="does not match its documents"):
        TopicSnapshot.open(tampered)

    (corpus_dir / "p1" / "memo.md").write_text("# A\n\nDepth 12.6 m")
    edited = compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T")
    assert edited.corpus_fingerprint != expected


@pytest.mark.parametrize("name", ["a#b.arkive.db", "q?mode=rw.arkive.db", "pct%20x.arkive.db"])
def test_snapshot_paths_with_uri_syntax_open_read_only(tmp_path: Path, name: str) -> None:
    folder = tmp_path / "serve"
    folder.mkdir()
    target = folder / name
    shutil.copy(compile_snapshot(tmp_path / "build", BATHYMETRY_CORPUS), target)

    with TopicSnapshot.open(target) as snap:
        assert snap.topic == "bathymetry" and snap.grep("330.9")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            snap._db.execute("DELETE FROM chunks")
    assert [p.name for p in folder.iterdir()] == [name]


def test_grep_snippet_centres_on_a_match_beyond_the_first_400_characters(
    tmp_path: Path,
) -> None:
    line = "lead " * 200 + "needle-777 " + "tail " * 200
    path = compile_snapshot(tmp_path, {"p1/long.md": f"# Long\n\n{line}"})

    with TopicSnapshot.open(path) as snap:
        (match,) = snap.grep("NEEDLE\u2011777")

    assert "needle-777" in match.snippet
    assert match.snippet.startswith("...") and match.snippet.endswith("...")
    assert len(match.snippet) == SNIPPET_CHARS + 6


def test_snapshot_opens_read_only_and_rejects_other_schema_versions(tmp_path: Path) -> None:
    path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    # The read-only URI is the guarantee that serving never mutates a snapshot.
    with (
        TopicSnapshot.open(path) as snap,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        snap._db.execute("DELETE FROM chunks")

    stale = tmp_path / "stale.arkive.db"
    shutil.copy(path, stale)
    with sqlite3.connect(stale) as db:
        # A snapshot of the previous schema: its tables exist, but they lack this schema's
        # meaning, so opening it asks for a rebuild instead of reading it.
        db.execute("UPDATE corpus_meta SET value = '5.0' WHERE key = 'schema_version'")
    db.close()
    with pytest.raises(
        SnapshotIntegrityError,
        match=r"schema version '5\.0'; Arkive \S+ reads schema 6\.0 only, so rebuild",
    ):
        TopicSnapshot.open(stale)
    with pytest.raises(SnapshotIntegrityError, match="does not exist"):
        TopicSnapshot.open(tmp_path / "absent.arkive.db")
    garbage = tmp_path / "garbage.arkive.db"
    garbage.write_bytes(os.urandom(4096))
    with pytest.raises(SnapshotIntegrityError, match="not a valid snapshot"):
        TopicSnapshot.open(garbage)


def test_document_dates_round_trip_with_their_basis(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    for rel_path, text in {
        "a/memo.md": "# Memo\n\nDate: 20/11/2025\n\nSurvey notes.",
        "b/notes.md": "# Notes\n\nDate: 05/04/2026\n\nSurvey notes.",
    }.items():
        (corpus_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (corpus_dir / rel_path).write_text(text, encoding="utf-8")
    (corpus_dir / "c").mkdir()
    write_pdf(
        corpus_dir / "c" / "report.pdf",
        [["# Report", "Survey notes."]],
        {"CreationDate": "D:20250630"},
    )
    output = tmp_path / "dates.arkive.db"
    write_snapshot(compile_corpus(corpus_dir, FakeGateway(), topic="d", title="D"), output)

    with TopicSnapshot.open(output) as snap:
        dates = {
            d.rel_path: (d.doc_date, d.date.iso if d.date else None, d.date and d.date.basis)
            for d in snap.list_documents()
        }
    assert dates == {
        "a/memo.md": ("20/11/2025", "2025-11-20", "printed"),
        # Ambiguous as printed: kept as written, with no calendar date and no fallback.
        "b/notes.md": ("05/04/2026", None, None),
        "c/report.pdf": ("2025-06-30", "2025-06-30", "metadata"),
    }
    with sqlite3.connect(output) as db, pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        db.execute("UPDATE documents SET doc_date_basis = NULL WHERE doc_date_iso IS NOT NULL")
    db.close()


def test_facets_glossary_roles_lessons_units_and_build_records_round_trip(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    for rel_path, text in MARKETING_CORPUS.items():
        (corpus_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (corpus_dir / rel_path).write_text(text, encoding="utf-8")
    corpus = compile_corpus(corpus_dir, FakeGateway(), topic="campaigns", title="Campaigns")
    output = tmp_path / "campaigns.arkive.db"
    write_snapshot(corpus, output)

    with TopicSnapshot.open(output) as snap:
        assert SCHEMA_VERSION == "6.0"
        assert tuple(snap.facets) == corpus.facets
        assert tuple(snap.glossary) == corpus.glossary
        assert tuple(snap.deliverable_types) == corpus.deliverable_types
        assert tuple(snap.feedback_lessons) == corpus.feedback_lessons
        assert snap.build_settings == dict(corpus.build_settings)
        assert snap.build_report == corpus.build_report
        assert snap.guide_markdown == corpus.guide_markdown
        documents = {d.doc_id: d for d in snap.list_documents()}
        assert documents[SPRING_REVIEW].facets == {
            "campaign": ["spring-launch"],
            "author": ["Sam Okafor"],
            "channel": ["Social"],
        }
        for doc in corpus.documents:
            stored = documents[doc.doc_id]
            assert (
                stored.deliverable_type,
                stored.is_feedback,
                tuple(stored.feedback_for_types),
                stored.family_id,
                stored.version_status,
                stored.superseded_by,
            ) == tuple(doc.role.model_dump().values())
        summary = snap.summary()
        guide = snap.guide()
    assert [(f.name, f.binding) for f in summary.facets] == [
        ("campaign", "path"),
        ("author", "content"),
        ("channel", "content"),
    ]
    assert (summary.deliverable_types, summary.glossary_terms) == (
        ["campaign brief", "review notes"],
        3,
    )
    assert [t.model_dump() for t in guide.deliverable_types] == [
        {
            "deliverable_type": "campaign brief",
            "description": "A campaign brief.",
            "document_count": 2,
            "feedback_count": 1,
        },
        {
            "deliverable_type": "review notes",
            "description": "A review notes.",
            "document_count": 1,
            "feedback_count": 0,
        },
    ]


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        (
            f"UPDATE glossary SET entry_json = replace(entry_json, '{LOYALTY_BRIEF}_sec_002',"
            f" '{LOYALTY_BRIEF}_sec_099') WHERE ordinal = 1",
            f"cites '{LOYALTY_BRIEF}_sec_099'",
        ),
        # The curator note's sections are not citable.
        (
            f"UPDATE glossary SET entry_json = replace(entry_json, '{LOYALTY_BRIEF}_sec_002',"
            f" '{CAMPAIGN_NOTE}_sec_001') WHERE ordinal = 1",
            f"cites '{CAMPAIGN_NOTE}_sec_001'",
        ),
        ("UPDATE glossary SET entry_json = '{}' WHERE ordinal = 1", "row is malformed"),
        # `prepare_draft` resolves a folded type, so it must name one type.
        (
            "UPDATE deliverable_types SET name = ' Campaign BRIEF' WHERE name = 'review notes';"
            " UPDATE documents SET deliverable_type = ' Campaign BRIEF'"
            " WHERE deliverable_type = 'review notes'",
            "deliverable types 'campaign brief' and ' Campaign BRIEF' fold alike",
        ),
        (
            "UPDATE documents SET deliverable_type = 'press release'"
            f" WHERE doc_id = '{LOYALTY_BRIEF}'",
            f"document {LOYALTY_BRIEF} has an inconsistent role",
        ),
        (
            f"UPDATE documents SET is_feedback = 0 WHERE doc_id = '{SPRING_REVIEW}'",
            f"document {SPRING_REVIEW} has an inconsistent role",
        ),
        (
            f"UPDATE feedback_lessons SET section_id = '{LOYALTY_BRIEF}_sec_002'",
            "does not cite a section of a feedback document",
        ),
        (
            f"UPDATE documents SET family_id = '{LOYALTY_BRIEF}', version_status = 'unordered'"
            f" WHERE doc_id = '{SPRING_BRIEF}'",
            f"version family {LOYALTY_BRIEF}: a family of one must be sole",
        ),
        (
            f"UPDATE documents SET family_id = '{SPRING_BRIEF}' WHERE doc_id = '{LOYALTY_BRIEF}'",
            f"version family {SPRING_BRIEF}: its ID is not its first member in path order",
        ),
        (
            f"DELETE FROM cluster_sections WHERE section_id LIKE '{SPRING_BRIEF}%'",
            "is not held by citable sections of two documents",
        ),
        (
            "INSERT INTO cluster_sections SELECT cluster_id, section_id, 99, 'x' FROM"
            " (SELECT min(cluster_id) AS cluster_id FROM unit_clusters WHERE kind = 'prose'),"
            f" (SELECT section_id FROM sections WHERE doc_id = '{SPRING_REVIEW}' LIMIT 1)",
            "is held by a feedback document",
        ),
        (
            "UPDATE cluster_sections SET sample = ' ' WHERE cluster_id IN"
            " (SELECT cluster_id FROM unit_clusters WHERE kind = 'prose')",
            "each holder of a prose unit needs a nonblank sample",
        ),
        # Two briefs superseding each other: a cycle with no latest member.
        (
            f"UPDATE documents SET family_id = '{LOYALTY_BRIEF}', version_status = 'superseded',"
            f" superseded_by = CASE doc_id WHEN '{LOYALTY_BRIEF}' THEN '{SPRING_BRIEF}'"
            f" ELSE '{LOYALTY_BRIEF}' END WHERE doc_id IN ('{LOYALTY_BRIEF}', '{SPRING_BRIEF}')",
            f"version family {LOYALTY_BRIEF}: a member is superseded by a superseded member",
        ),
    ],
)
def test_knowledge_rows_are_validated_when_a_snapshot_opens(
    tmp_path: Path, sql: str, message: str
) -> None:
    path = compile_snapshot(tmp_path, MARKETING_CORPUS, topic="campaigns")
    with sqlite3.connect(path) as db:
        db.executescript(sql)
    db.close()

    with pytest.raises(SnapshotIntegrityError, match=re.escape(message)):
        TopicSnapshot.open(path)


def test_glossary_lookup_by_term_alias_list_and_miss(tmp_path: Path) -> None:
    path = compile_snapshot(tmp_path, MARKETING_CORPUS, topic="campaigns")

    with TopicSnapshot.open(path) as snap:
        listing = snap.lookup_glossary()
        by_term = snap.lookup_glossary("cta")
        by_alias = snap.lookup_glossary("Call To Action")
        miss = snap.lookup_glossary("net promoter score")
        past_end = snap.lookup_glossary(offset=3)
        with pytest.raises(InvalidFilterError, match="blank"):
            snap.lookup_glossary("  ")
        with pytest.raises(InvalidFilterError, match="offset"):
            snap.lookup_glossary(offset=-1)

    assert [(t.term, t.defined, t.aliases) for t in listing.terms] == [
        ("Churn", True, []),
        ("CTA", True, ["call to action"]),
        ("ICP", False, []),
    ]
    assert (listing.total, listing.offset, listing.next_offset) == (3, 0, None)
    assert listing.matches == []
    assert [e.term for e in by_term.matches] == ["CTA"] and by_term.terms == []
    assert by_alias.matches == by_term.matches
    assert len(by_term.matches[0].senses) == 2
    assert (miss.query, miss.matches, miss.terms, miss.total) == ("net promoter score", [], [], 0)
    assert (past_end.terms, past_end.total, past_end.next_offset) == ([], 3, None)


def test_a_glossary_page_stays_within_the_response_bound(tmp_path: Path) -> None:
    terms = "\n\n".join(
        f"**TERM{n:04d}** means a long defined measure number {n}. " + "Detail. " * 40
        for n in range(500)
    )
    path = compile_snapshot(tmp_path, {"p1/terms.md": f"# Terms\n\n{terms}"}, topic="terms")

    with TopicSnapshot.open(path) as snap:
        listing = snap.lookup_glossary()
        first = snap.lookup_glossary("term")
        assert first.next_offset is not None
        pages = [first]
        while pages[-1].next_offset is not None:
            pages.append(snap.lookup_glossary("term", offset=pages[-1].next_offset))

    assert len(listing.model_dump_json().encode()) <= RESPONSE_MAX_BYTES
    assert all(len(p.model_dump_json().encode()) <= RESPONSE_MAX_BYTES for p in pages)
    assert sum(len(p.matches) for p in pages) == first.total == 500


def test_document_pages_omit_hashes_validate_limits_and_shrink_to_the_response_bound(
    tmp_path: Path,
) -> None:
    files = {
        f"p{n:03d}/memo.md": f"# {'Detailed Harbour Survey Memo ' * 15}{n}\n\n"
        + f"Long summary line {n} of the memo. " * 20
        for n in range(60)
    }
    path = compile_snapshot(tmp_path, files, topic="memos")

    with TopicSnapshot.open(path) as snap:
        everything = snap.list_documents()
        page = snap.list_documents_page(limit=7, offset=51)
        last = snap.list_documents_page(limit=7, offset=58)
        wide = snap.list_documents_page(limit=100)
        for bad in ({"limit": 0}, {"limit": 101}, {"offset": -1}):
            with pytest.raises(InvalidFilterError):
                snap.list_documents_page(**bad)  # type: ignore[arg-type]

    assert len(everything) == 60 and all(d.sha256 for d in everything)
    assert [d.doc_id for d in page.documents] == [d.doc_id for d in everything[51:58]]
    assert (page.total, page.offset, page.limit, page.next_offset) == (60, 51, 7, 58)
    assert "sha256" not in page.documents[0].model_dump()
    assert (len(last.documents), last.next_offset) == (2, None)
    assert len(wide.model_dump_json().encode()) <= RESPONSE_MAX_BYTES
    assert wide.next_offset == len(wide.documents) < 60


def test_grep_returns_at_most_fifty_matches(snapshot: TopicSnapshot) -> None:
    assert MAX_GREP_MATCHES == 50
    assert len(snapshot.grep("a", max_matches=MAX_GREP_MATCHES)) <= 50
    for bad in (0, MAX_GREP_MATCHES + 1):
        with pytest.raises(InvalidFilterError, match="between 1 and 50"):
            snapshot.grep("bathymetry", max_matches=bad)


VERSIONED = {
    "p1/drafts/survey-report-v0.9.md": (
        "# Survey Report v0.9\n\nDate: 2025-03-01\n\n"
        + "\n\n".join(
            f"## Line {i}\n\nGauge reading {i} on the tidal survey line." for i in range(6)
        )
    ),
    "p1/survey-report.md": (
        "# Survey Report v1.0\n\nDate: 2025-04-01\n\n"
        + "\n\n".join(
            f"## Line {i}\n\nGauge reading {i} on the tidal survey line." for i in range(6)
        )
    ),
    "p2/other.md": "# Other\n\nA tidal note about something else entirely.",
}


def test_search_collapses_superseded_drafts_caps_families_and_backfills(tmp_path: Path) -> None:
    path = compile_snapshot(tmp_path, VERSIONED, topic="versions")
    draft, final = (
        document_id("p1/drafts/survey-report-v0.9.md"),
        document_id("p1/survey-report.md"),
    )
    embed = FakeGateway().embed_query

    with TopicSnapshot.open(path) as snap:
        assert snap.document(draft).superseded_by == final
        hits = snap.search("gauge reading tidal survey line", embed, top_k=4)
        alone = snap.search("gauge reading", embed, filters={"project": "p1"}, top_k=6)
        ranked = snap.rank_channels("gauge reading tidal survey line", embed, top_k=8)

    # The superseded draft is left behind its latest version, and the final report shows
    # three chunks before the other document gets a place.
    counts = {d: sum(h.doc_id == d for h in hits) for d in {h.doc_id for h in hits}}
    assert counts == {final: 3, document_id("p2/other.md"): 1}
    # Hits keep fused-rank order and count what their document has beyond the list.
    assert [h.rrf_score for h in hits] == sorted((h.rrf_score for h in hits), reverse=True)
    assert {h.doc_id: h.more_from_document for h in hits}[final] == 7 - 3
    # Filters that leave one family still fill top_k from it.
    assert len(alone) == 6 and {h.doc_id for h in alone} == {final}
    # The ablation channels stay raw: the draft's chunks rank there.
    assert draft in {h.doc_id for h in ranked["hybrid_rrf"]}


def test_a_long_superseded_draft_cannot_crowd_its_latest_version_out_of_search(
    tmp_path: Path,
) -> None:
    def report(version: str, date: str, lines: int, words: str) -> str:
        body = "\n\n".join(f"## Line {i}\n\n{words} {i}." for i in range(lines))
        return f"# Survey Report {version}\n\nDate: {date}\n\n{body}"

    files = {
        "p1/drafts/report-v1.md": report("v1", "2025-01-10", 320, "Gauge reading tidal line"),
        "p1/report-v2.md": report(
            "v2", "2025-02-10", 4, "The final record of the gauge reading on the tidal line"
        ),
    }
    path = compile_snapshot(tmp_path, files, topic="drafts")
    draft, final = (document_id(p) for p in files)

    with TopicSnapshot.open(path) as snap:
        assert snap.document(draft).superseded_by == final
        hits = snap.search("gauge reading tidal line", FakeGateway().embed_query, top_k=5)

    # The draft's 320 chunks would fill both ranking channels; only the latest is ranked.
    assert hits and {h.doc_id for h in hits} == {final}


def test_search_results_stay_within_the_response_bound(tmp_path: Path) -> None:
    files = {
        f"p{n:02d}/long.md": f"# Long {n}\n\n" + " ".join([f"tidal gauge reading site{n}"] * 300)
        for n in range(20)
    }
    path = compile_snapshot(tmp_path, files, topic="long")

    embed = FakeGateway().embed_query
    with TopicSnapshot.open(path) as snap:
        hits = snap.search("tidal gauge reading", embed, top_k=MAX_TOP_K)
        ranking = snap._rank(
            build_fts5_query("tidal gauge reading"),
            embed("tidal gauge reading"),
            snap._active_pool(None),
        )
        ranked = Counter(snap._hits[i]["doc_id"] for i in ranking.order)

    assert 1 <= len(hits) < MAX_TOP_K and fits_response(hits)
    assert len(response_text(hits).encode()) <= RESPONSE_MAX_BYTES
    # Each hit counts the ranked chunks of its document the cut list does not show.
    shown = Counter(h.doc_id for h in hits)
    assert all(h.more_from_document == ranked[h.doc_id] - shown[h.doc_id] for h in hits)
    assert any(h.more_from_document for h in hits)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (
            "INSERT INTO document_facets VALUES ('doc_001', 'client', 'Fabrikam')",
            "dangling foreign keys in document_facets",
        ),
        (
            "INSERT INTO facet_aliases VALUES ('client', 'fabrikam', 'Fabrikam', 'Fabrikam')",
            "dangling foreign keys in facet_aliases",
        ),
        (
            "DELETE FROM facet_aliases WHERE facet = 'client' AND alias = value",
            "value 'Contoso Ports' has no canonical alias row",
        ),
        # A blank folded key would let a blank filter resolve to the alias's value.
        (
            "UPDATE facet_aliases SET folded_alias = ''"
            " WHERE rowid = (SELECT min(rowid) FROM facet_aliases WHERE alias != value)",
            "has folded key '', which is not the alias's non-blank folded form",
        ),
        (
            "UPDATE facet_aliases SET folded_alias = 'someone else'"
            " WHERE rowid = (SELECT min(rowid) FROM facet_aliases WHERE alias != value)",
            "has folded key 'someone else', which is not the alias's non-blank folded form",
        ),
        ("DROP TABLE chunks_fts", "lacks the tables chunks_fts"),
        ("UPDATE documents SET page_count = 'many' WHERE ordinal = 1", "not a valid snapshot"),
        (
            "UPDATE documents SET doc_date = '2025-02-30', doc_date_iso = '2025-02-30',"
            " doc_date_basis = 'metadata' WHERE ordinal = 1",
            "'2025-02-30' is not a calendar date",
        ),
        # A date text that disagrees with its calendar date: a metadata date is its own text,
        # and a printed date's text parses to it.
        (
            "UPDATE documents SET doc_date = '2025-06-29', doc_date_iso = '2025-06-30',"
            " doc_date_basis = 'metadata' WHERE ordinal = 1",
            "has an inconsistent date",
        ),
        (
            "UPDATE documents SET doc_date = 'May 2025', doc_date_iso = '2025-06',"
            " doc_date_basis = 'printed' WHERE ordinal = 1",
            "has an inconsistent date",
        ),
        (
            "UPDATE documents SET doc_date = NULL, doc_date_iso = '2025-06-30',"
            " doc_date_basis = 'metadata' WHERE ordinal = 1",
            "has an inconsistent date",
        ),
        # A basis without a calendar date, even beside an ambiguous text that parses to none,
        # in a file written past the table's CHECK constraint.
        (
            "PRAGMA ignore_check_constraints = ON;"
            " UPDATE documents SET doc_date = '05/04/2025', doc_date_iso = NULL,"
            " doc_date_basis = 'printed' WHERE ordinal = 1",
            "has an inconsistent date",
        ),
        ("UPDATE facet_values SET folders_json = '{' WHERE ordinal = 1", "facet row is malformed"),
    ],
)
def test_a_damaged_snapshot_fails_to_open_with_an_integrity_error(
    tmp_path: Path, damage: str, message: str
) -> None:
    path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    with sqlite3.connect(path) as db:
        db.executescript(damage)
    db.close()

    with pytest.raises(SnapshotIntegrityError, match=message) as raised:
        TopicSnapshot.open(path)
    assert str(path) in str(raised.value)


def test_sqlite_errors_after_opening_are_integrity_errors(tmp_path: Path) -> None:
    path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    with TopicSnapshot.open(path) as snap:
        # The file is damaged under an open snapshot.
        with sqlite3.connect(path) as db:
            db.execute("DROP TABLE chunks_fts")
        db.close()
        with pytest.raises(SnapshotIntegrityError, match="cannot read the snapshot"):
            query(snap, "rainfall")


# Drafting packets.


def memo(title: str, deliverable: str, sections: Sequence[tuple[str, str]], date: str = "") -> str:
    head = f"# {title}\n\nDeliverable: {deliverable}\n" + (f"Date: {date}\n" if date else "")
    return head + "".join(f"\n## {heading}\n\n{body}\n" for heading, body in sections)


def packet_bytes(packet: DraftPacket) -> int:
    return len(response_text(packet).encode())


GDV_INPUTS = (
    "Map groundwater dependent vegetation for the Sentinel lease with NDVI anomaly trends,"
    " dry season composites and a field validation plan."
)


def test_draft_ranking_prefers_the_closer_document_over_the_longer_one(tmp_path: Path) -> None:
    generic = "Remote sensing analysis of water and vegetation across the region."
    files = {
        "a/waterbodies.md": memo(
            "Regional Waterbodies Memo",
            "technical memorandum",
            [(f"Basin {i}", f"{generic} Basin {i} vegetation map.") for i in range(40)],
        ),
        "b/gdv.md": memo(
            "Lease GDV Memo",
            "technical memorandum",
            [
                ("Method", "Groundwater dependent vegetation mapped with NDVI anomaly trends."),
                ("Composites", "Dry season composites of the Sentinel lease."),
                ("Validation", "A field validation plan for groundwater dependent vegetation."),
            ],
        ),
    }
    path = compile_snapshot(tmp_path, files, topic="gdv")
    gdv, waterbodies = document_id("b/gdv.md"), document_id("a/waterbodies.md")
    embedded: list[str] = []

    def embed(text: str) -> NDArray[np.float32]:
        embedded.append(text)
        return FakeGateway().embed_query(text)

    with TopicSnapshot.open(path) as snap:
        offline = snap.prepare_draft(GDV_INPUTS, None)
        online = snap.prepare_draft(GDV_INPUTS, embed, deliverable_type="Technical Memorandum")
        lexical = snap._draft_terms(GDV_INPUTS, snap.list_documents())[0] or ""
        matched = snap._lexical_ranks(lexical, None)
        chunks = {d: sum(snap._hits[i]["doc_id"] == d for i in matched) for d in (gdv, waterbodies)}

    # The long memo has many more matching chunks, but relevance sums each document's best
    # three, so the closer short memo ranks first, with and without an embedder.
    assert chunks[waterbodies] > chunks[gdv]
    assert [e.doc_id for e in offline.exemplars] == [gdv, waterbodies]
    assert [e.doc_id for e in online.exemplars] == [gdv, waterbodies]
    assert offline.exemplars[0].relevance_score > offline.exemplars[1].relevance_score
    assert embedded == [GDV_INPUTS] and online.deliverable_type == "technical memorandum"
    # The short memo is included whole, in order, without its visual transcripts.
    first = online.exemplars[0]
    assert all(s.included for s in first.sections) and online.omitted_sections >= 0
    assert [s.markdown for s in first.sections][
        1
    ] == "## Method\n\nGroundwater dependent vegetation mapped with NDVI anomaly trends."


def test_draft_requests_are_validated_before_any_embedding(tmp_path: Path) -> None:
    path = compile_snapshot(tmp_path, MARKETING_CORPUS, topic="campaigns")
    calls: list[str] = []

    def embed(text: str) -> NDArray[np.float32]:
        calls.append(text)
        return FakeGateway().embed_query(text)

    with TopicSnapshot.open(path) as snap:
        with pytest.raises(InvalidFilterError, match="project_inputs must not be blank"):
            snap.prepare_draft(" \n", embed)
        with pytest.raises(InvalidFilterError, match="unknown facet"):
            snap.prepare_draft("A brief.", embed, filters={"stage": "draft"})
        with pytest.raises(
            NotFoundError,
            match=re.escape(
                "unknown deliverable type 'press release'; valid types: campaign brief (2),"
                " review notes (1)"
            ),
        ):
            snap.prepare_draft("A brief.", embed, deliverable_type="press release")
        # A blank type is no type.
        untyped = snap.prepare_draft("A brief for lapsed members.", None, deliverable_type=" ")
    assert calls == []
    assert untyped.deliverable_type is None


def test_feedback_documents_give_lessons_and_draft_only_as_their_own_type(tmp_path: Path) -> None:
    path = compile_snapshot(tmp_path, MARKETING_CORPUS, topic="campaigns")

    with TopicSnapshot.open(path) as snap:
        brief = snap.prepare_draft(
            "A social brief for first-time buyers.", None, deliverable_type="campaign brief"
        )
        review = snap.prepare_draft(
            "Review notes on a social brief.", None, deliverable_type="review notes"
        )

    assert {e.doc_id for e in brief.exemplars} == {LOYALTY_BRIEF, SPRING_BRIEF}
    assert SPRING_REVIEW not in {c.doc_id for c in brief.other_candidates}
    (lesson,) = brief.lessons
    assert (lesson.statement, lesson.source_doc_id, lesson.target_types) == (
        "Cut the jargon from the opening line.",
        SPRING_REVIEW,
        ("campaign brief",),
    )
    assert [n.scope_path for n in brief.curator_notes] == ["AGENTS.md"]
    # Asked for its own type, the review is the first exemplar; at most one exemplar of the
    # requested type lets the best other deliverables fill the slots.
    assert review.exemplars[0].doc_id == SPRING_REVIEW
    assert {e.deliverable_type for e in review.exemplars[1:]} == {"campaign brief"}
    # The disclosure both briefs end with recurs in two version families of the topic.
    (loyalty,) = [e for e in brief.exemplars if e.doc_id == LOYALTY_BRIEF]
    disclosure = loyalty.sections[-1]
    assert disclosure.family_recurrence == FamilyRecurrence(
        recurring_unit_count=1,
        exemplar_doc_ids=[e.doc_id for e in brief.exemplars],
        corpus_family_count=2,
        # The review is feedback, so it is no family a unit could recur in.
        corpus_family_total=2,
        latest_holder_date=None,
    )
    assert loyalty.sections[1].family_recurrence is None
    assert brief.drafting_guidance == DRAFTING_GUIDANCE


PROPOSAL_CLAUSE = (
    "All survey data remain confidential to the client and are held for seven years under"
    " the house records policy."
)
DRAFT_ONLY_CLAUSE = (
    "This draft clause about interim staffing appears only in two versions of one proposal"
    " and nowhere else in the collection."
)


def test_family_recurrence_counts_version_families_not_duplicate_drafts(tmp_path: Path) -> None:
    def proposal(title: str, date: str, scope: str) -> str:
        return memo(title, "survey proposal", [("Scope", scope), ("Terms", PROPOSAL_CLAUSE)], date)

    estuary_scope = f"Survey of the estuary channel. {DRAFT_ONLY_CLAUSE}"
    files = {
        "a/proposal-v1.md": proposal("Estuary Proposal v1", "2025-01-10", estuary_scope),
        "a/proposal-v2.md": proposal("Estuary Proposal v2", "2025-02-10", estuary_scope),
        "b/proposal.md": proposal("Harbour Proposal", "2024-06-01", "Survey of the harbour."),
    }
    path = compile_snapshot(tmp_path, files, topic="proposals")
    v1, v2, harbour = (document_id(p) for p in files)

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft(
            "Estuary survey proposal.", None, deliverable_type="survey proposal"
        )

    # The superseded first version takes no exemplar slot; the latest one names it.
    assert {e.doc_id for e in packet.exemplars} == {v2, harbour}
    (latest,) = [e for e in packet.exemplars if e.doc_id == v2]
    assert (latest.version_status, latest.family_doc_ids) == ("latest", [v1])
    scope, terms = latest.sections[1], latest.sections[2]
    # Recurring only between two versions of one deliverable is no recurrence across families.
    assert scope.family_recurrence is None
    assert terms.family_recurrence is not None
    assert (
        terms.family_recurrence.corpus_family_count,
        terms.family_recurrence.corpus_family_total,
    ) == (2, 2)
    assert set(terms.family_recurrence.exemplar_doc_ids) == {v2, harbour}
    assert terms.family_recurrence.latest_holder_date == "2025-02-10"


def test_unordered_family_members_share_one_exemplar_slot(tmp_path: Path) -> None:
    body = [("Scope", "Survey of the canal locks and the towpath walls.")]
    files = {
        "a/report.md": memo("Canal Report", "survey report", body),
        "b/report.md": memo("Canal Report", "survey report", [*body, ("Annex", "Extra table.")]),
    }
    path = compile_snapshot(tmp_path, files, topic="canal")
    first, second = (document_id(p) for p in files)

    with TopicSnapshot.open(path) as snap:
        assert {snap.document(d).version_status for d in (first, second)} == {"unordered"}
        packet = snap.prepare_draft("Canal locks survey report.", None)

    (exemplar,) = packet.exemplars
    other = second if exemplar.doc_id == first else first
    assert exemplar.family_doc_ids == [other]
    assert other not in {c.doc_id for c in packet.other_candidates}


def test_a_competitive_neighbour_of_another_type_fills_a_free_slot(tmp_path: Path) -> None:
    night = "Night time noise from the depot yard measured at the nearest bedroom window."
    files = {
        "a/letter-1.md": memo("Depot Letter", "noise assessment letter", [("Noise", night)]),
        "b/letter-2.md": memo("Yard Letter", "noise assessment letter", [("Noise", night)]),
        "c/report.md": memo("Depot Report", "technical report", [("Results", night)]),
        "d/memo.md": memo("Fee Memo", "fee memorandum", [("Fees", "Hourly rates and invoicing.")]),
    }
    path = compile_snapshot(tmp_path, files, topic="acoustic")
    report, fees = document_id("c/report.md"), document_id("d/memo.md")

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft(
            "Depot yard night time noise at the nearest bedroom window.",
            None,
            deliverable_type="noise assessment letter",
        )

    types = [e.deliverable_type for e in packet.exemplars]
    # The report is as relevant as the letters, so it takes the third slot; the fee memo is
    # not competitive and stays a candidate.
    assert types == ["noise assessment letter", "noise assessment letter", "technical report"]
    assert packet.exemplars[2].doc_id == report
    assert [c.doc_id for c in packet.other_candidates] == [fees]


def test_a_long_exemplar_packs_whole_sections_by_priority_within_the_bound(
    tmp_path: Path,
) -> None:
    filler = " ".join(f"Observation {i} of the seabed condition was logged." for i in range(28))
    long_sections = [(f"Chapter {i}", f"Chapter {i}. {filler}") for i in range(60)]
    for k in (30, 31, 32):
        long_sections[k] = ("Dredge spoil", f"Berth pocket dredge spoil volumes, part {k}.")
    long_sections[-1] = ("Terms", f"{PROPOSAL_CLAUSE} {filler}")
    bio = "Priya Okafor leads hydrographic surveys and holds the chartered surveyor grade."
    files = {
        "a/big.md": memo("Berth Pocket Report", "survey report", long_sections),
        "b/small.md": memo(
            "Apron Report",
            "survey report",
            [("Scope", "Survey of the quay apron."), ("Terms", PROPOSAL_CLAUSE)],
        ),
        "c/other.md": memo(
            "Quay Report",
            "survey report",
            [(f"Berth {k}", f"Berth pocket survey of quay {k}.") for k in range(3)],
        ),
        "e/another.md": memo(
            "Jetty Report",
            "survey report",
            [(f"Pocket {k}", f"Berth pocket survey at jetty {k}.") for k in range(3)],
        ),
        "d/team.md": memo(
            "Lagoon Report", "survey report", [("Lagoon", "Lagoon depths."), ("Team", bio)]
        ),
    }
    path = compile_snapshot(tmp_path, files, topic="berth")
    big, team = document_id("a/big.md"), document_id("d/team.md")
    inputs = "Berth pocket dredge spoil survey report. Team: Priya Okafor (hydrographer)."

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft(inputs, None, deliverable_type="survey report")
        sections = {s.section_id: s for s in snap._sections.values()}

    assert fits_response(packet)
    first = packet.exemplars[0]
    assert first.doc_id == big and first.omitted_outline_sections == 0
    # The outline is complete; what is included is whole, never cut.
    assert len(first.sections) == 61
    included = [s for s in first.sections if s.included]
    assert 0 < len(included) < 61 and packet.omitted_sections > 0
    for entry in included:
        assert entry.markdown == sections[entry.section_id].markdown.strip()
    # Priority: the input-matched section, the recurring terms, and the first and last two.
    titles = {s.heading_path.split(" > ")[-1] for s in included}
    assert {"Dredge spoil", "Terms", "Chapter 0", "Chapter 58"} <= titles
    assert first.sections[0].included and first.sections[-1].included
    # The rare name in the inputs brings the team section of a document outside the exemplars.
    assert team not in {e.doc_id for e in packet.exemplars}
    (bio_section,) = packet.supplemental_sections
    assert (bio_section.doc_id, bio_section.markdown) == (team, f"## Team\n\n{bio}")


def test_an_outline_too_long_for_the_packet_is_cut_and_says_so(tmp_path: Path) -> None:
    sections = [(f"Clause {i} of the schedule of rates", f"Rate {i}.") for i in range(400)]
    files = {
        "a/schedule.md": memo("Rates Schedule", "schedule", sections),
        "b/schedule.md": memo("Other Schedule", "schedule", sections[:3]),
    }
    path = compile_snapshot(tmp_path, files, topic="rates")

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft("Schedule of rates clauses.", None)

    assert fits_response(packet)
    # The second exemplar's outline did not fit beside the first, so it is a candidate.
    (first,) = packet.exemplars
    assert first.doc_id == document_id("a/schedule.md")
    assert first.omitted_outline_sections > 0
    assert len(first.sections) + first.omitted_outline_sections == 401
    assert [c.doc_id for c in packet.other_candidates] == [document_id("b/schedule.md")]


def test_outlines_give_way_from_the_last_exemplar_after_notes_and_lessons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def report(name: str) -> str:
        parts = [
            (f"Part {i} {'w' * 20}", f"Harbour soundings part {i} for {name}.") for i in range(75)
        ]
        return memo(f"{name} Harbour Survey Report", "survey report", parts)

    files = {
        **{f"{c}/report.md": report(c) for c in ("a", "b", "c")},
        **{
            f"notes-{i}/AGENTS.md": f"# Note {i}\n\n" + f"Curator guidance {i}. " * 60
            for i in range(10)
        },
        **{
            f"reviews/r{i}.md": f"# Review {i}\n\nDeliverable: survey report review\n\nReviewer: "
            + f"State every harbour datum in each part, lesson {i}. " * 6
            for i in range(10)
        },
        **{
            f"other/o{i}.md": memo(f"Other {i}", "letter", [("Body", "Harbour letter. " * 30)])
            for i in range(10)
        },
    }
    path = compile_snapshot(tmp_path, files, topic="fair")
    long_parts = [
        (f"Part {i}", "Harbour soundings. " + "The seabed was logged. " * 20) for i in range(340)
    ]
    long_path = compile_snapshot(
        tmp_path / "long",
        {"long/long.md": memo("Long Harbour Report", "long report", long_parts)},
        topic="long",
    )
    measured = Counter[str]()
    real = snapshot_module.response_text

    def counting(result: Any) -> str:
        measured[type(result).__name__] += 1
        return real(result)

    monkeypatch.setattr(snapshot_module, "response_text", counting)
    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft(
            "Harbour survey report soundings.", None, deliverable_type="survey report"
        )
        drafts = measured["DraftPacket"]
    with TopicSnapshot.open(long_path) as snap:
        started = time.perf_counter()
        long = snap.prepare_draft("Harbour soundings.", None)
        elapsed = time.perf_counter() - started

    assert fits_response(packet) and fits_response(long)
    # Notes and lessons came first; the outlines of three, then two exemplars did not fit
    # beside them, so the last ones gave way and the first keeps its complete outline.
    assert packet.curator_notes and packet.lessons
    (first,) = packet.exemplars
    assert (len(first.sections), first.omitted_outline_sections) == (76, 0)
    # The moved exemplars lead the other candidates, which stay at most eight.
    moved = {document_id(f"{c}/report.md") for c in "abc"} - {first.doc_id}
    assert {c.doc_id for c in packet.other_candidates[:2]} == moved
    assert len(packet.other_candidates) == MAX_OTHER_CANDIDATES
    # Packing measures the packet a handful of times, not once per section.
    assert drafts <= 8
    (only,) = long.exemplars
    assert (
        only.omitted_outline_sections > 0
        and len(only.sections) + only.omitted_outline_sections == 341
    )
    assert elapsed < 0.5


def test_a_filter_that_leaves_only_a_copy_drafts_from_the_copy(tmp_path: Path) -> None:
    body = [("Scope", "Survey of the canal locks and the towpath walls.")]
    report = memo("Canal Report", "survey report", body)
    files = {"p1/report.md": report, "p2/report.md": report, "p3/other.md": memo("O", "x", body)}
    path = compile_snapshot(tmp_path, files, topic="copies")
    first, copy = document_id("p1/report.md"), document_id("p2/report.md")

    with TopicSnapshot.open(path) as snap:
        assert snap.document(copy).superseded_by == first
        both = snap.prepare_draft("Canal locks survey report.", None)
        filtered = snap.prepare_draft("Canal locks survey report.", None, filters={"project": "p2"})

    # With both in the pool the copy gives way to its first; filtered to the copy, it is the
    # exemplar.
    assert copy not in {e.doc_id for e in both.exemplars}
    assert [e.doc_id for e in filtered.exemplars] == [copy]


def test_house_lessons_apply_whatever_the_filters_and_follow_the_first_exemplars_type(
    tmp_path: Path,
) -> None:
    path = compile_snapshot(tmp_path, MARKETING_CORPUS, topic="campaigns")

    with TopicSnapshot.open(path) as snap:
        filtered = snap.prepare_draft(
            "A loyalty brief for lapsed members.",
            None,
            deliverable_type="campaign brief",
            filters={"campaign": "loyalty-relaunch"},
        )
        untyped = snap.prepare_draft("A loyalty brief for lapsed members.", None)

    # The review sits in another campaign, yet its lesson is a house rule for briefs.
    assert [e.doc_id for e in filtered.exemplars] == [LOYALTY_BRIEF]
    assert [lesson.source_doc_id for lesson in filtered.lessons] == [SPRING_REVIEW]
    assert untyped.exemplars[0].deliverable_type == "campaign brief"
    assert [lesson.source_doc_id for lesson in untyped.lessons] == [SPRING_REVIEW]


DELETED_CLAUSE = (
    "Interim staffing is drawn from the regional pool and approved by the project director"
    " before mobilisation."
)


def test_recurrence_counts_no_superseded_draft_and_samples_only_house_text(
    tmp_path: Path,
) -> None:
    def proposal(title: str, date: str, scope: str, terms: str) -> str:
        return memo(title, "survey proposal", [("Scope", scope), ("Terms", terms)], date)

    files = {
        # The first version held a clause its latest version dropped.
        "a/proposal-v1.md": proposal(
            "Estuary Proposal v1", "2025-01-10", DELETED_CLAUSE, PROPOSAL_CLAUSE
        ),
        "a/proposal-v2.md": proposal(
            "Estuary Proposal v2", "2025-02-10", "Survey of the estuary.", PROPOSAL_CLAUSE
        ),
        "b/proposal.md": proposal(
            "Harbour Proposal",
            "2024-06-01",
            f"Survey of the harbour for the port authority.\n\n{DELETED_CLAUSE}",
            PROPOSAL_CLAUSE,
        ),
    }
    path = compile_snapshot(tmp_path, files, topic="proposals")
    harbour = document_id("b/proposal.md")

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft(
            "Harbour survey proposal for the port authority.",
            None,
            deliverable_type="survey proposal",
        )

    (exemplar,) = [e for e in packet.exemplars if e.doc_id == harbour]
    scope, terms = exemplar.sections[1], exemplar.sections[2]
    # Only a superseded draft shares the dropped clause, so it recurs in no other family.
    assert scope.included and scope.family_recurrence is None
    assert terms.family_recurrence is not None
    assert packet.recurring_samples == [
        "all survey data remain confidential to the client and are held for seven years under the"
    ]


def test_recurring_samples_quote_the_included_exemplar_and_skip_table_headers(
    tmp_path: Path,
) -> None:
    tail = (
        " for seven years under the house records policy, after which the survey team destroys"
        " every copy and certifies the deletion to the client in writing."
    )
    table = "| Metric | Value |\n| --- | --- |\n| Depth | 3 m |"
    files = {
        # The first in path order, so its wording represents the recurring unit.
        "a/estuary.md": memo(
            "Estuary Proposal",
            "survey proposal",
            [
                ("Terms", f"The hydrographic survey records remain private to the client{tail}"),
                ("Results", table),
            ],
        ),
        "b/harbour.md": memo(
            "Harbour Proposal",
            "survey proposal",
            [
                (
                    "Terms",
                    "The hydro-\ngraphic survey records remain confidential to the client" + tail,
                ),
                ("Results", table),
            ],
        ),
    }
    path = compile_snapshot(tmp_path, files, topic="samples")

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft("Harbour proposal.", None, deliverable_type="survey proposal")

    assert packet.exemplars[0].doc_id == document_id("b/harbour.md")
    assert all(s.included for e in packet.exemplars for s in e.sections)
    # One sample per recurring prose unit, in the first exemplar's words with its line-end
    # hyphen joined; the table header recurs too but is no house text.
    assert packet.recurring_samples == [
        "the hydrographic survey records remain confidential to the client for seven years under"
        " the house records"
    ]


def test_a_recurring_sample_comes_from_its_own_holder_at_the_far_end_of_a_chain(
    tmp_path: Path,
) -> None:
    # Seven 8-word windows sliding one word at a time: each says the same as the next, so the
    # chain is one recurring unit, though its two ends share no 4-gram.
    words = "every survey record stays with the client under the house retention policy for seven"
    windows = [" ".join(words.split()[k : k + 8]) for k in range(7)]
    files = {
        f"p{k}/proposal.md": memo(
            f"Proposal {name}",
            "survey proposal",
            [("Scope", f"Survey of the {name} basin."), ("Terms", f"{window.capitalize()}.")],
        )
        for k, (name, window) in enumerate(
            zip(["Arden", "Brook", "Cove", "Dune", "Esker", "Fjord", "Gully"], windows, strict=True)
        )
    }
    path = compile_snapshot(tmp_path, files, topic="chain")

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft(
            "Gully basin proposal.", None, deliverable_type="survey proposal"
        )

    assert packet.exemplars[0].doc_id == document_id("p6/proposal.md")
    assert packet.recurring_samples == [windows[6]]


def test_a_recurring_section_keeps_its_priority_when_the_first_exemplar_shows_none(
    tmp_path: Path,
) -> None:
    house = "The seabed record covers the stated dates and certifies nothing later than those."
    cyrillic = "\u0417\u0430\u043f\u0438\u0441\u044c " * 4_500
    filler = "Filler text of the long chapter. " * 560
    files = {
        "a/alpha.md": memo(
            "Alpha Harbour", "report", [("Scope", "Short."), ("Record", f"{house}\n\n{cyrillic}")]
        ),
        "b/beta.md": memo(
            "Beta",
            "report",
            [
                ("Intro", "Short."),
                ("Filler one", filler),
                ("Terms", f"{house}\n\n" + "Standing terms of the house. " * 260),
                ("Filler two", filler),
                ("Filler three", filler),
            ],
        ),
    }
    path = compile_snapshot(tmp_path, files, topic="priority")
    alpha, beta = document_id("a/alpha.md"), document_id("b/beta.md")

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft("Alpha harbour", None)

    first, second = packet.exemplars
    assert (first.doc_id, second.doc_id) == (alpha, beta)
    record = first.sections[2]
    terms = next(s for s in second.sections if s.heading_path.endswith("Terms"))
    # The first exemplar's recurring section is too large to include, so the unit it holds is
    # not shown; the second exemplar's section holding it comes before its boundary sections.
    assert record.family_recurrence is not None and not record.included
    assert terms.included
    assert not all(s.included for s in second.sections)


def test_documents_of_the_requested_type_are_ranked_among_themselves(tmp_path: Path) -> None:
    dense = "Harbour tide gauge datum."
    files = {
        **{
            f"memos/memo-{n:02d}.md": memo(
                f"Memo {n}", "technical memorandum", [(f"Part {k}", dense) for k in range(10)]
            )
            for n in range(40)
        },
        "proposals/a.md": memo(
            "Coastal Proposal A", "survey proposal", [("Scope", "Works on the coast.")]
        ),
        "proposals/b.md": memo(
            "Coastal Proposal B",
            "survey proposal",
            [
                ("Scope", "Works on the coast."),
                ("Gauges", f"{'Context of the works. ' * 40} {dense}"),
            ],
        ),
    }
    path = compile_snapshot(tmp_path, files, topic="ranking")
    other, matching = document_id("proposals/a.md"), document_id("proposals/b.md")

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft(dense, None, deliverable_type="survey proposal")

    # Four hundred memo chunks rank above the proposal's matching chunk over the whole pool;
    # ranked among the proposals, its chunk counts, so it is the first exemplar.
    assert [e.doc_id for e in packet.exemplars[:2]] == [matching, other]
    assert packet.exemplars[0].relevance_score > packet.exemplars[1].relevance_score


def test_drafting_words_and_reviews_do_not_make_rare_terms(tmp_path: Path) -> None:
    files = {
        "a/memo.md": memo("Memo", "memo", [("Team", "Zephyrine leads the survey crew.")]),
        "b/review.md": memo("Review", "review notes", [("Notes", "Zephyrine was praised.")]),
        "c/other.md": memo("Other", "memo", [("Body", "Please write whether based on this.")]),
    }
    path = compile_snapshot(tmp_path, files, topic="rare")
    reviewed = {document_id("b/review.md")}

    with TopicSnapshot.open(path) as snap:
        deliverables = [d for d in snap.list_documents() if d.doc_id not in reviewed]
        query, rare = snap._draft_terms(
            "Please write the report based on this, whether or not Zephyrine leads.", deliverables
        )

    assert query is not None
    assert not {"please", "write", "whether", "based", "report"} & set(re.findall(r"\w+", query))
    # The review's section is no source of rare-word sections.
    assert rare["zephyrine"] == {f"{document_id('a/memo.md')}_sec_002"}


def test_a_free_slot_takes_only_a_relevant_document_of_another_type(tmp_path: Path) -> None:
    files = {
        "a/letter.md": memo(
            "Depot Letter", "noise letter", [("Noise", "Night noise at the yard.")]
        ),
        "b/fees.md": memo("Fee Schedule", "fee schedule", [("Rates", "Hourly rates.")]),
    }
    path = compile_snapshot(tmp_path, files, topic="slots")

    with TopicSnapshot.open(path) as snap:
        packet = snap.prepare_draft(
            "Night noise at the yard.", None, deliverable_type="noise letter"
        )

    # One exemplar of the requested type leaves free slots, but the fee schedule shares no
    # word with the inputs, so it stays a candidate.
    assert [e.doc_id for e in packet.exemplars] == [document_id("a/letter.md")]
    assert [c.doc_id for c in packet.other_candidates] == [document_id("b/fees.md")]
