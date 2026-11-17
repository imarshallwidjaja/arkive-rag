from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import stat
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from arkive_rag import __version__
from arkive_rag.compiler import compile_corpus
from arkive_rag.models import (
    InvalidFilterError,
    NotFoundError,
    ProviderError,
    SearchHit,
    SnapshotIntegrityError,
    UnsafePathError,
)
from arkive_rag.snapshot import (
    MAX_TOP_K,
    SCHEMA_VERSION,
    SNIPPET_CHARS,
    TopicSnapshot,
    build_fts5_query,
    write_snapshot,
)
from tests.support import BATHYMETRY_CORPUS, MARKETING_CORPUS, FakeGateway, compile_snapshot

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
    assert {m.doc_id for m in only_estuary} == {"doc_002"}
    assert (
        snapshot.grep("sentinel-2", filters={"client": "Contoso", "project": "p101-harbour-lidar"})
        == []
    )


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
        db.execute("UPDATE corpus_meta SET value = '2.0' WHERE key = 'schema_version'")
    db.close()
    with pytest.raises(
        SnapshotIntegrityError, match=r"schema version '2\.0'; Arkive \S+ reads schema 3\.0 only"
    ):
        TopicSnapshot.open(stale)
    with pytest.raises(SnapshotIntegrityError, match="does not exist"):
        TopicSnapshot.open(tmp_path / "absent.arkive.db")
    garbage = tmp_path / "garbage.arkive.db"
    garbage.write_bytes(os.urandom(4096))
    with pytest.raises(SnapshotIntegrityError, match="not a valid snapshot"):
        TopicSnapshot.open(garbage)


def test_facets_glossary_blueprints_and_build_records_round_trip(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    for rel_path, text in MARKETING_CORPUS.items():
        (corpus_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (corpus_dir / rel_path).write_text(text, encoding="utf-8")
    corpus = compile_corpus(corpus_dir, FakeGateway(), topic="campaigns", title="Campaigns")
    output = tmp_path / "campaigns.arkive.db"
    write_snapshot(corpus, output)

    with TopicSnapshot.open(output) as snap:
        assert SCHEMA_VERSION == "3.0"
        assert tuple(snap.facets) == corpus.facets
        assert tuple(snap.glossary) == corpus.glossary
        assert tuple(snap.blueprints) == corpus.blueprints
        assert snap.build_settings == dict(corpus.build_settings)
        assert snap.build_report == corpus.build_report
        assert snap.guide_markdown == corpus.guide_markdown
        assert {d.doc_id: d.facets for d in snap.list_documents()}["doc_004"] == {
            "campaign": ["spring-launch"],
            "author": ["Sam Okafor"],
            "channel": ["Social"],
        }
        summary = snap.summary()
    assert [(f.name, f.binding) for f in summary.facets] == [
        ("campaign", "path"),
        ("author", "content"),
        ("channel", "content"),
    ]
    assert (summary.blueprints, summary.glossary_terms) == (["campaign brief", "review notes"], 3)


@pytest.mark.parametrize(
    ("table", "column", "old", "new", "message"),
    [
        ("glossary", "entry_json", "doc_002_sec_002", "doc_002_sec_099", "cites 'doc_002_sec_099'"),
        # doc_001 is the curator note: its sections are not citable.
        ("glossary", "entry_json", "doc_002_sec_002", "doc_001_sec_001", "cites 'doc_001_sec_001'"),
        ("blueprints", "blueprint_json", '"doc_002"', '"doc_001"', "names document 'doc_001'"),
        ("blueprints", "blueprint_json", '"outline"', '"sketch"', "row is malformed"),
        # doc_003_sec_004 is cited only by the campaign brief's standing passage.
        (
            "blueprints",
            "blueprint_json",
            '"doc_003_sec_004"',
            '"doc_003_sec_099"',
            "cites 'doc_003_sec_099'",
        ),
        (
            "blueprints",
            "blueprint_json",
            '"doc_003_sec_004"',
            '"doc_004_sec_001"',
            "standing passage citing 'doc_004_sec_001', which is not a section of the"
            " blueprint's exemplars",
        ),
        # Both citations then name doc_002: a passage must recur in another exemplar.
        (
            "blueprints",
            "blueprint_json",
            '"doc_003_sec_004"',
            '"doc_002_sec_003"',
            "standing passage 'Disclosure' whose citations name fewer than two exemplar documents",
        ),
        # `get_blueprint` resolves a folded type, so it must name one blueprint.
        (
            "blueprints",
            "blueprint_json",
            '"campaign brief"',
            '" Review NOTES"',
            "blueprints ' Review NOTES' and 'review notes' have the same deliverable type",
        ),
    ],
)
def test_glossary_and_blueprint_rows_are_validated_when_a_snapshot_opens(
    tmp_path: Path, table: str, column: str, old: str, new: str, message: str
) -> None:
    corpus_dir = tmp_path / "corpus"
    for rel_path, text in MARKETING_CORPUS.items():
        (corpus_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (corpus_dir / rel_path).write_text(text, encoding="utf-8")
    output = tmp_path / "campaigns.arkive.db"
    write_snapshot(compile_corpus(corpus_dir, FakeGateway(), topic="c", title="C"), output)
    with sqlite3.connect(output) as db:
        changed = db.execute(
            f"UPDATE {table} SET {column} = replace({column}, ?, ?) WHERE ordinal = 1", (old, new)
        ).rowcount
    db.close()

    assert changed == 1
    with pytest.raises(SnapshotIntegrityError, match=message):
        TopicSnapshot.open(output)


def test_guide_summarises_blueprints_and_one_blueprint_is_read_in_full_by_folded_type(
    tmp_path: Path,
) -> None:
    path = compile_snapshot(tmp_path, MARKETING_CORPUS, topic="campaigns")

    with TopicSnapshot.open(path) as snap:
        guide = snap.guide()
        brief = snap.blueprint("  Campaign BRIEF ")
        with pytest.raises(
            NotFoundError,
            match="no blueprint for deliverable type 'press release'; valid types: campaign"
            " brief, review notes",
        ):
            snap.blueprint("press release")
        assert brief == snap.blueprints[0]

    assert [b.model_dump() for b in guide.blueprints] == [
        {
            "deliverable_type": "campaign brief",
            "description": "How the collection writes a campaign brief.",
            "exemplar_count": 2,
        },
        {
            "deliverable_type": "review notes",
            "description": "How the collection writes a review notes.",
            "exemplar_count": 1,
        },
    ]
    assert brief.outline and brief.conventions
    assert [p.citations for p in brief.standing_passages] == [
        ("doc_002_sec_004", "doc_003_sec_004")
    ]


def test_glossary_lookup_by_term_alias_list_and_miss(tmp_path: Path) -> None:
    path = compile_snapshot(tmp_path, MARKETING_CORPUS, topic="campaigns")

    with TopicSnapshot.open(path) as snap:
        listing = snap.lookup_glossary()
        by_term = snap.lookup_glossary("cta")
        by_alias = snap.lookup_glossary("Call To Action")
        miss = snap.lookup_glossary("net promoter score")
        with pytest.raises(InvalidFilterError, match="blank"):
            snap.lookup_glossary("  ")

    assert [(t.term, t.defined, t.aliases) for t in listing.terms] == [
        ("Churn", True, []),
        ("CTA", True, ["call to action"]),
        ("ICP", False, []),
    ]
    assert listing.matches == []
    assert [e.term for e in by_term.matches] == ["CTA"] and by_term.terms == []
    assert by_alias.matches == by_term.matches
    assert len(by_term.matches[0].senses) == 2
    assert (miss.query, miss.matches, miss.terms) == ("net promoter score", [], [])


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
        ("UPDATE facet_values SET folders_json = '{' WHERE ordinal = 1", "facet row is malformed"),
    ],
)
def test_a_damaged_snapshot_fails_to_open_with_an_integrity_error(
    tmp_path: Path, damage: str, message: str
) -> None:
    path = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    with sqlite3.connect(path) as db:
        db.execute(damage)
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
