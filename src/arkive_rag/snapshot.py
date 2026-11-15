"""The snapshot file: SQLite schema, atomic writer, and the read-only query surface."""

from __future__ import annotations

import errno
import json
import os
import re
import sqlite3
import stat
import tempfile
import threading
import unicodedata
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice, pairwise
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter, ValidationError

from arkive_rag import __version__
from arkive_rag.models import (
    Blueprint,
    BuildReport,
    CompiledCorpus,
    CuratorNote,
    DocumentInfo,
    DocumentOutline,
    Facet,
    FacetSummary,
    FacetValue,
    GlossaryEntry,
    GlossaryLookup,
    GlossaryTermSummary,
    GrepMatch,
    InvalidFilterError,
    NotFoundError,
    PageVisualView,
    ProviderError,
    SearchHit,
    SectionOutlineItem,
    SectionReadView,
    SnapshotIntegrityError,
    TopicGuide,
    TopicSummary,
    UnsafePathError,
    corpus_fingerprint,
    fold_facet_label,
    fold_text,
)

SCHEMA_VERSION = "3.0"
RRF_K = 60.0
CHANNEL_DEPTH = 100
MAX_TOP_K = 50
SNIPPET_CHARS = 400

QueryEmbedder = Callable[[str], NDArray[np.float32]]
RankingChannel = Literal["bm25_only", "dense_only", "hybrid_rrf"]

SCHEMA = """
CREATE TABLE corpus_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE curator_notes (
    ordinal INTEGER PRIMARY KEY, scope_path TEXT NOT NULL, content TEXT NOT NULL);
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, rel_path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL, format TEXT NOT NULL, title TEXT NOT NULL, doc_date TEXT,
    deliverable_type TEXT NOT NULL, summary TEXT NOT NULL, page_count INTEGER NOT NULL,
    is_curator_note INTEGER NOT NULL CHECK (is_curator_note IN (0, 1)));
CREATE TABLE facets (
    name TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE, description TEXT NOT NULL,
    binding TEXT NOT NULL CHECK (binding IN ('path', 'content')));
CREATE TABLE facet_values (
    facet TEXT NOT NULL REFERENCES facets(name), value TEXT NOT NULL, ordinal INTEGER NOT NULL,
    description TEXT NOT NULL, folders_json TEXT NOT NULL, PRIMARY KEY (facet, value));
CREATE TABLE facet_aliases (
    facet TEXT NOT NULL, folded_alias TEXT NOT NULL, alias TEXT NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY (facet, folded_alias),
    FOREIGN KEY (facet, value) REFERENCES facet_values(facet, value));
CREATE TABLE document_facets (
    doc_id TEXT NOT NULL REFERENCES documents(doc_id), facet TEXT NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY (doc_id, facet, value),
    FOREIGN KEY (facet, value) REFERENCES facet_values(facet, value));
CREATE TABLE glossary (ordinal INTEGER PRIMARY KEY, term TEXT NOT NULL, entry_json TEXT NOT NULL);
CREATE TABLE blueprints (
    ordinal INTEGER PRIMARY KEY, deliverable_type TEXT NOT NULL UNIQUE,
    blueprint_json TEXT NOT NULL);
CREATE TABLE sections (
    section_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    ordinal INTEGER NOT NULL, heading_path TEXT NOT NULL, level INTEGER NOT NULL,
    page_start INTEGER NOT NULL, page_end INTEGER NOT NULL, markdown TEXT NOT NULL);
CREATE TABLE pages (
    doc_id TEXT NOT NULL REFERENCES documents(doc_id), page_number INTEGER NOT NULL,
    native_text TEXT NOT NULL, visual_summary TEXT NOT NULL, webp_bytes BLOB NOT NULL,
    PRIMARY KEY (doc_id, page_number));
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    section_id TEXT NOT NULL REFERENCES sections(section_id), ordinal INTEGER NOT NULL,
    kind TEXT NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
    breadcrumb TEXT NOT NULL, content TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED, content_norm, breadcrumb_norm,
    tokenize='porter unicode61 remove_diacritics 2');
"""
# Every table a 3.0 snapshot holds; `TopicSnapshot.open` requires each one.
TABLES = frozenset(re.findall(r"CREATE (?:VIRTUAL )?TABLE (\w+)", SCHEMA))
# Every key is required when a snapshot opens.
META_KEYS = (
    "schema_version",
    "topic",
    "title",
    "created_at",
    "corpus_fingerprint",
    "arkive_version",
    "embedding_model",
    "embedding_dimensions",
    "guide_markdown",
    "build_settings",
    "build_report",
)

# Conversational filler dropped from FTS queries when at least two content tokens remain.
_FILLER_WORDS = """what which where when who how does did is are was were the a an of in for to on
with from by about show tell give list find and or not"""
_STOPWORDS = frozenset(_FILLER_WORDS.split())
_FOLDERS: TypeAdapter[tuple[str, ...]] = TypeAdapter(tuple[str, ...])
_SETTINGS: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def build_fts5_query(raw_query: str) -> str | None:
    """Turn arbitrary user text into an FTS5 expression that cannot be a syntax error.

    Every token is quoted, so FTS5 operators and punctuation in the input are inert.
    Adjacent-token phrases boost multi-part terms such as `Sentinel-2` or `330.9`.
    """
    # Letters and digits only: like the unicode61 tokenizer, `_` separates tokens.
    tokens = re.findall(r"[^\W_]+", _nfkc(raw_query).lower())
    if not tokens:
        return None
    content = [token for token in tokens if token not in _STOPWORDS]
    terms = list(dict.fromkeys(content if len(content) >= 2 else tokens))
    phrases = list(
        dict.fromkeys(
            f"{a} {b}" for a, b in pairwise(tokens) if a not in _STOPWORDS or b not in _STOPWORDS
        )
    )
    return " OR ".join(f'"{term}"' for term in [*terms, *phrases])


def check_build_paths(corpus_dir: Path, *outputs: Path) -> None:
    """Reject output files that coincide or sit inside the corpus they are built from."""
    root = corpus_dir.resolve()
    resolved = [path.resolve() for path in outputs]
    for path in resolved:
        if path.is_relative_to(root):
            raise UnsafePathError(f"{path} is inside the corpus directory {root}")
    if len(set(resolved)) != len(resolved):
        raise UnsafePathError(f"build outputs must be different files: {resolved}")


def write_snapshot(corpus: CompiledCorpus, output_path: Path) -> None:
    """Write the corpus to `output_path` atomically: readers see the old file or the new one.

    A new snapshot is private (0600); replacing a snapshot keeps the old file's mode.
    """
    chunks = corpus.all_chunks()
    if corpus.embeddings.shape != (len(chunks), corpus.embedding_dimensions):
        raise SnapshotIntegrityError(
            f"embeddings shape {corpus.embeddings.shape} does not match"
            f" ({len(chunks)}, {corpus.embedding_dimensions})"
        )
    check_build_paths(corpus.corpus_dir, output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # A unique 0600 sibling, so concurrent writers never share a temporary file.
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
    except OSError as exc:
        raise SnapshotIntegrityError(f"cannot create a file next to {output_path}: {exc}") from exc
    try:
        db = sqlite3.connect(tmp_path)
        try:
            _populate(db, corpus)
            if db.execute("PRAGMA foreign_key_check").fetchall():
                raise SnapshotIntegrityError("snapshot has dangling foreign keys")
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise SnapshotIntegrityError("snapshot failed integrity_check")
        finally:
            db.close()
        if output_path.exists():
            os.chmod(tmp_path, stat.S_IMODE(output_path.stat().st_mode))
        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    except BaseException as exc:
        tmp_path.unlink(missing_ok=True)
        if isinstance(exc, sqlite3.Error | OSError):
            raise SnapshotIntegrityError(f"cannot write snapshot {output_path}: {exc}") from exc
        raise
    _fsync_directory(output_path)


def _fsync_directory(output_path: Path) -> None:
    """Persist the rename; some filesystems cannot fsync a directory and say so."""
    try:
        directory = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        if exc.errno in (errno.EINVAL, errno.ENOTSUP):
            return
        raise SnapshotIntegrityError(
            f"{output_path} was published but its directory could not be synced: {exc}"
        ) from exc


def _populate(db: sqlite3.Connection, corpus: CompiledCorpus) -> None:
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(SCHEMA)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "topic": corpus.topic,
        "title": corpus.title,
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_fingerprint": corpus.corpus_fingerprint,
        "arkive_version": __version__,
        "embedding_model": corpus.embedding_model,
        "embedding_dimensions": str(corpus.embedding_dimensions),
        "guide_markdown": corpus.guide_markdown,
        "build_settings": json.dumps(corpus.build_settings, sort_keys=True),
        "build_report": corpus.build_report.model_dump_json(),
    }
    with db:
        db.executemany("INSERT INTO corpus_meta VALUES (?, ?)", meta.items())
        db.executemany(
            "INSERT INTO curator_notes VALUES (?, ?, ?)",
            [(i, n.scope_path, n.content) for i, n in enumerate(corpus.curator_notes, start=1)],
        )
        for ordinal, facet in enumerate(corpus.facets, start=1):
            db.execute(
                "INSERT INTO facets VALUES (?, ?, ?, ?)",
                (facet.name, ordinal, facet.description, facet.binding),
            )
            for value_ordinal, value in enumerate(facet.values, start=1):
                db.execute(
                    "INSERT INTO facet_values VALUES (?, ?, ?, ?, ?)",
                    (
                        facet.name,
                        value.value,
                        value_ordinal,
                        value.description,
                        json.dumps(list(value.folders)),
                    ),
                )
                # The canonical value is stored as its own alias, so the primary key also
                # refuses an alias that would name two values.
                db.executemany(
                    "INSERT INTO facet_aliases VALUES (?, ?, ?, ?)",
                    [
                        (facet.name, fold_facet_label(alias), alias, value.value)
                        for alias in dict.fromkeys((value.value, *value.aliases))
                    ],
                )
        db.executemany(
            "INSERT INTO glossary VALUES (?, ?, ?)",
            [
                (i, entry.term, entry.model_dump_json())
                for i, entry in enumerate(corpus.glossary, start=1)
            ],
        )
        db.executemany(
            "INSERT INTO blueprints VALUES (?, ?, ?)",
            [
                (i, blueprint.deliverable_type, blueprint.model_dump_json())
                for i, blueprint in enumerate(corpus.blueprints, start=1)
            ],
        )
        row = 0
        for ordinal, doc in enumerate(corpus.documents, start=1):
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.doc_id,
                    ordinal,
                    doc.rel_path,
                    doc.sha256,
                    doc.format,
                    doc.title,
                    doc.doc_date,
                    doc.deliverable_type,
                    doc.summary,
                    len(doc.pages),
                    int(doc.is_curator_note),
                ),
            )
            db.executemany(
                "INSERT INTO document_facets VALUES (?, ?, ?)",
                [
                    (doc.doc_id, name, value)
                    for name, values in doc.facets.items()
                    for value in values
                ],
            )
            db.executemany(
                "INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        s.section_id,
                        doc.doc_id,
                        s.ordinal,
                        s.heading_path,
                        s.level,
                        s.page_start,
                        s.page_end,
                        s.markdown,
                    )
                    for s in doc.sections
                ],
            )
            db.executemany(
                "INSERT INTO pages VALUES (?, ?, ?, ?, ?)",
                [
                    (doc.doc_id, p.page_number, p.native_text, p.visual_summary, p.webp_bytes)
                    for p in doc.pages
                ],
            )
            for chunk in doc.chunks:
                vector = corpus.embeddings[row].astype(np.float32).tobytes()
                row += 1
                db.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        doc.doc_id,
                        chunk.section_id,
                        chunk.ordinal,
                        chunk.kind,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.breadcrumb,
                        chunk.content,
                        vector,
                    ),
                )
                db.execute(
                    "INSERT INTO chunks_fts VALUES (?, ?, ?)",
                    (chunk.chunk_id, _nfkc(chunk.content), _nfkc(chunk.breadcrumb)),
                )


def _snippet(line: str, needle: str) -> str:
    """At most SNIPPET_CHARS of `line` centred on the first match, `...` where clipped."""
    if len(line) <= SNIPPET_CHARS:
        return line.strip()
    # Folding can change lengths (NFKC, casefold), so find the shortest original
    # prefix whose folded form contains the needle: that is where the match ends.
    low, high = 1, len(line)
    while low < high:
        middle = (low + high) // 2
        if needle in fold_text(line[:middle]):
            high = middle
        else:
            low = middle + 1
    centre = low - len(needle) // 2
    start = max(0, min(centre - SNIPPET_CHARS // 2, len(line) - SNIPPET_CHARS))
    end = start + SNIPPET_CHARS
    return f"{'...' if start else ''}{line[start:end]}{'...' if end < len(line) else ''}"


_HIT_COLUMNS = """c.chunk_id, c.doc_id, c.section_id, c.kind, d.title AS doc_title,
    s.heading_path, c.page_start, c.page_end, c.content"""
_SECTION_COLUMNS = """s.section_id, s.doc_id, d.title AS doc_title, s.heading_path,
    s.page_start, s.page_end"""
_FROM_CHUNKS = """FROM chunks c JOIN sections s ON s.section_id = c.section_id
    JOIN documents d ON d.doc_id = c.doc_id"""


@dataclass(frozen=True)
class _GrepSection:
    fields: dict[str, Any]
    doc_id: str
    lines: tuple[str, ...]
    normalized: tuple[str, ...]


class TopicSnapshot:
    """A read-only open snapshot. Thread-safe: queries share one connection under a lock."""

    def __init__(self, db: sqlite3.Connection, path: Path) -> None:
        self._db = db
        self._lock = threading.Lock()
        self.path = path
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        meta = (
            dict(db.execute("SELECT key, value FROM corpus_meta").fetchall())
            if "corpus_meta" in tables
            else {}
        )
        version = meta.get("schema_version")
        if version != SCHEMA_VERSION:
            raise SnapshotIntegrityError(
                f"{path}: schema version {version!r}; Arkive {__version__} reads schema"
                f" {SCHEMA_VERSION} only, so rebuild the snapshot from its corpus"
            )
        missing = [key for key in META_KEYS if key not in meta]
        if missing:
            raise SnapshotIntegrityError(f"{path}: corpus_meta lacks {', '.join(missing)}")
        absent = sorted(TABLES - tables)
        if absent:
            raise SnapshotIntegrityError(f"{path}: snapshot lacks the tables {', '.join(absent)}")
        dangling = db.execute("PRAGMA foreign_key_check").fetchall()
        if dangling:
            tables_named = sorted({row[0] for row in dangling})
            raise SnapshotIntegrityError(
                f"{path}: dangling foreign keys in {', '.join(tables_named)}"
            )
        try:
            self.embedding_dimensions = int(meta["embedding_dimensions"])
            self.build_settings: dict[str, Any] = _SETTINGS.validate_json(meta["build_settings"])
            self.build_report = BuildReport.model_validate_json(meta["build_report"])
        except (ValueError, ValidationError) as exc:
            raise SnapshotIntegrityError(f"{path}: corpus_meta is malformed: {exc}") from exc
        self.topic: str = meta["topic"]
        self.title: str = meta["title"]
        self.created_at: str = meta["created_at"]
        self.arkive_version: str = meta["arkive_version"]
        self.guide_markdown: str = meta["guide_markdown"]
        self.embedding_model: str = meta["embedding_model"]
        db.row_factory = sqlite3.Row
        self._load_facets(db)
        documents = db.execute(
            "SELECT doc_id, rel_path, sha256, format, title, doc_date, deliverable_type, summary,"
            " page_count, is_curator_note FROM documents ORDER BY ordinal"
        )
        self._documents = {
            r["doc_id"]: DocumentInfo.model_validate(
                dict(r)
                | {
                    "is_curator_note": bool(r["is_curator_note"]),
                    "facets": self.facets_of(r["doc_id"]),
                }
            )
            for r in documents
        }
        self.corpus_fingerprint = corpus_fingerprint(
            (d.rel_path, d.sha256) for d in self._documents.values()
        )
        if meta["corpus_fingerprint"] != self.corpus_fingerprint:
            raise SnapshotIntegrityError(
                f"{path}: recorded corpus_fingerprint {meta['corpus_fingerprint']} does not match"
                f" its documents ({self.corpus_fingerprint})"
            )
        self._load_knowledge(db)
        rows = db.execute(
            f"SELECT {_HIT_COLUMNS}, c.embedding {_FROM_CHUNKS} ORDER BY c.doc_id, c.ordinal"
        ).fetchall()
        self._hits = [{k: v for k, v in dict(row).items() if k != "embedding"} for row in rows]
        self._hit_index = {hit["chunk_id"]: i for i, hit in enumerate(self._hits)}
        dim = self.embedding_dimensions
        if any(len(row["embedding"]) != dim * 4 for row in rows):
            raise SnapshotIntegrityError(f"{path}: an embedding does not have {dim} dimensions")
        vectors = b"".join(row["embedding"] for row in rows)
        self._matrix = np.frombuffer(vectors, dtype=np.float32).reshape(len(rows), dim)
        self._grep_sections = [
            _GrepSection(
                {k: v for k, v in dict(r).items() if k != "markdown"},
                r["doc_id"],
                tuple(lines := r["markdown"].split("\n")),
                tuple(fold_text(line) for line in lines),
            )
            for r in db.execute(
                f"SELECT {_SECTION_COLUMNS}, s.markdown FROM sections s"
                " JOIN documents d ON d.doc_id = s.doc_id ORDER BY s.doc_id, s.ordinal"
            )
        ]

    def _load_facets(self, db: sqlite3.Connection) -> None:
        """Facets, aliases and the document-to-facets map, loaded once for every read path."""
        aliases: dict[str, dict[str, str]] = {}
        alias_lists: dict[tuple[str, str], list[str]] = {}
        for r in db.execute("SELECT facet, folded_alias, alias, value FROM facet_aliases"):
            # The resolver looks filters up by folded key; a key that is blank or is not its
            # alias's fold would answer a filter the alias does not name.
            if not r["folded_alias"] or r["folded_alias"] != fold_facet_label(r["alias"]):
                raise SnapshotIntegrityError(
                    f"{self.path}: facet {r['facet']!r} alias {r['alias']!r} has folded key"
                    f" {r['folded_alias']!r}, which is not the alias's non-blank folded form"
                )
            aliases.setdefault(r["facet"], {})[r["folded_alias"]] = r["value"]
            if r["alias"] != r["value"]:
                alias_lists.setdefault((r["facet"], r["value"]), []).append(r["alias"])
        values: dict[str, list[FacetValue]] = {}
        try:
            for r in db.execute(
                "SELECT facet, value, description, folders_json FROM facet_values"
                " ORDER BY facet, ordinal"
            ):
                values.setdefault(r["facet"], []).append(
                    FacetValue(
                        value=r["value"],
                        description=r["description"],
                        aliases=tuple(sorted(alias_lists.get((r["facet"], r["value"]), []))),
                        folders=_FOLDERS.validate_json(r["folders_json"]),
                    )
                )
            self.facets = [
                Facet(
                    name=r["name"],
                    description=r["description"],
                    binding=r["binding"],
                    values=tuple(values.get(r["name"], [])),
                )
                for r in db.execute(
                    "SELECT name, description, binding FROM facets ORDER BY ordinal"
                )
            ]
        except ValidationError as exc:
            raise SnapshotIntegrityError(f"{self.path}: a facet row is malformed: {exc}") from exc
        for facet in self.facets:
            for value in facet.values:
                if aliases.get(facet.name, {}).get(fold_facet_label(value.value)) != value.value:
                    raise SnapshotIntegrityError(
                        f"{self.path}: facet {facet.name!r} value {value.value!r} has no"
                        " canonical alias row"
                    )
        self._aliases = aliases
        self._doc_facets: dict[str, dict[str, list[str]]] = {}
        self._docs_by_value: dict[tuple[str, str], set[str]] = {}
        order = {f.name: i for i, f in enumerate(self.facets)}
        rows = db.execute("SELECT doc_id, facet, value FROM document_facets").fetchall()
        for r in sorted(rows, key=lambda r: (r["doc_id"], order[r["facet"]], r["value"])):
            self._doc_facets.setdefault(r["doc_id"], {}).setdefault(r["facet"], []).append(
                r["value"]
            )
            self._docs_by_value.setdefault((r["facet"], r["value"]), set()).add(r["doc_id"])

    def _load_knowledge(self, db: sqlite3.Connection) -> None:
        """Glossary and blueprint rows, validated and with every reference resolved."""
        citable = {
            r["section_id"]
            for r in db.execute(
                "SELECT s.section_id FROM sections s JOIN documents d ON d.doc_id = s.doc_id"
                " WHERE d.is_curator_note = 0"
            )
        }
        documents = {d for d, info in self._documents.items() if not info.is_curator_note}
        try:
            self.glossary = [
                GlossaryEntry.model_validate_json(r["entry_json"])
                for r in db.execute("SELECT entry_json FROM glossary ORDER BY ordinal")
            ]
            self.blueprints = [
                Blueprint.model_validate_json(r["blueprint_json"])
                for r in db.execute("SELECT blueprint_json FROM blueprints ORDER BY ordinal")
            ]
        except ValidationError as exc:
            raise SnapshotIntegrityError(
                f"{self.path}: a glossary or blueprint row is malformed: {exc}"
            ) from exc
        cited = [
            (f"glossary term {e.term!r}", c)
            for e in self.glossary
            for s in e.senses
            for c in s.citations
        ]
        for b in self.blueprints:
            label = f"blueprint {b.deliverable_type!r}"
            cited += [(label, c) for s in b.outline for c in s.citations]
            cited += [(label, c) for g in (*b.conventions, *b.pitfalls) for c in g.citations]
            for doc_id in (*b.exemplar_doc_ids, *b.feedback_doc_ids):
                if doc_id not in documents:
                    raise SnapshotIntegrityError(
                        f"{self.path}: {label} names document {doc_id!r}, which is not a"
                        " citable document of this snapshot"
                    )
        for label, section_id in cited:
            if section_id not in citable:
                raise SnapshotIntegrityError(
                    f"{self.path}: {label} cites {section_id!r}, which is not a citable section"
                    " of this snapshot"
                )

    @classmethod
    def open(cls, path: Path) -> TopicSnapshot:
        if not path.is_file():
            raise SnapshotIntegrityError(f"snapshot {path} does not exist")
        # Percent-encoding keeps `%`, `?` and `#` in the name from being read as URI syntax.
        uri = f"file:{urllib.parse.quote(str(path.resolve()), safe='/:')}?mode=ro"
        try:
            db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        except sqlite3.Error as exc:
            raise SnapshotIntegrityError(f"cannot open snapshot {path}: {exc}") from exc
        try:
            return cls(db, path)
        except (
            sqlite3.Error,
            KeyError,
            StopIteration,
            TypeError,
            ValueError,
        ) as exc:
            # A damaged build output: decoding or cross-table lookups failed while loading.
            db.close()
            raise SnapshotIntegrityError(f"{path} is not a valid snapshot: {exc!r}") from exc
        except BaseException:
            db.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _rows(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                return self._db.execute(sql, params).fetchall()
            except sqlite3.Error as exc:
                raise SnapshotIntegrityError(
                    f"{self.path}: cannot read the snapshot: {exc}"
                ) from exc

    def facets_of(self, doc_id: str) -> dict[str, list[str]]:
        """The document's values per facet, in facet order; facets it holds no value of omitted."""
        return {k: list(v) for k, v in self._doc_facets.get(doc_id, {}).items()}

    # Catalogue reads.

    def summary(self) -> TopicSummary:
        return TopicSummary(
            topic=self.topic,
            title=self.title,
            created_at=self.created_at,
            corpus_fingerprint=self.corpus_fingerprint,
            arkive_version=self.arkive_version,
            document_count=len(self._documents),
            facets=[
                FacetSummary(name=f.name, binding=f.binding, values=[v.value for v in f.values])
                for f in self.facets
            ],
            blueprints=[b.deliverable_type for b in self.blueprints],
            glossary_terms=len(self.glossary),
        )

    def curator_notes(self) -> list[CuratorNote]:
        notes = self._rows("SELECT scope_path, content FROM curator_notes ORDER BY ordinal")
        return [CuratorNote.model_validate(dict(n)) for n in notes]

    def guide(self) -> TopicGuide:
        return TopicGuide(
            topic=self.topic,
            title=self.title,
            guide_markdown=self.guide_markdown,
            curator_notes=self.curator_notes(),
            facets=self.facets,
            blueprints=self.blueprints,
            glossary_terms=[e.term for e in self.glossary],
        )

    def lookup_glossary(self, query: str | None = None) -> GlossaryLookup:
        """Every term without a query; with one, entries whose folded term or alias contains it."""
        if query is None:
            terms = [
                GlossaryTermSummary(
                    term=e.term,
                    kind=e.kind,
                    aliases=list(e.aliases),
                    defined=any(s.definition is not None for s in e.senses),
                )
                for e in self.glossary
            ]
            return GlossaryLookup(query=None, terms=terms, matches=[])
        needle = fold_text(query).strip()
        if not needle:
            raise InvalidFilterError("glossary query must not be blank; omit it to list terms")
        matches = [
            e for e in self.glossary if any(needle in fold_text(f) for f in (e.term, *e.aliases))
        ]
        return GlossaryLookup(query=query, terms=[], matches=matches)

    def resolve_filters(self, filters: Mapping[str, str] | None) -> frozenset[str] | None:
        """The documents holding every filter's value (keys combine with AND), or None when
        there is no filter. A value is a canonical facet value or one of its aliases."""
        if not filters:
            return None
        allowed: set[str] | None = None
        for facet, value in sorted(filters.items()):
            if facet not in self._aliases:
                valid = ", ".join(f.name for f in self.facets) or "none, this topic has no facets"
                raise InvalidFilterError(f"unknown facet {facet!r}; valid facets: {valid}")
            if not fold_facet_label(value):
                raise InvalidFilterError(f"{facet} filter value must not be blank")
            canonical = self._aliases[facet].get(fold_facet_label(value))
            if canonical is None:
                known = next(f for f in self.facets if f.name == facet).values
                raise InvalidFilterError(
                    f"unknown {facet} value {value!r}; valid values: "
                    + ", ".join(v.value for v in known)
                )
            docs = self._docs_by_value.get((facet, canonical), set())
            allowed = set(docs) if allowed is None else allowed & docs
        return frozenset(allowed or ())

    def list_documents(self, *, filters: Mapping[str, str] | None = None) -> list[DocumentInfo]:
        allowed = self.resolve_filters(filters)
        return [d for d in self._documents.values() if allowed is None or d.doc_id in allowed]

    def document(self, doc_id: str) -> DocumentInfo:
        if doc_id not in self._documents:
            raise NotFoundError(f"unknown document {doc_id!r}; call list_documents for valid IDs")
        return self._documents[doc_id]

    def outline(self, doc_id: str) -> DocumentOutline:
        document = self.document(doc_id)
        rows = self._rows(
            "SELECT section_id, ordinal, heading_path, level, page_start, page_end"
            " FROM sections WHERE doc_id = ? ORDER BY ordinal",
            (doc_id,),
        )
        sections = [SectionOutlineItem.model_validate(dict(r)) for r in rows]
        return DocumentOutline(document=document, sections=sections)

    # Retrieval.

    def search(
        self,
        query: str,
        embed_query: QueryEmbedder,
        *,
        filters: Mapping[str, str] | None = None,
        top_k: int = 8,
    ) -> list[SearchHit]:
        """Hybrid search: BM25 and dense ranks fused by Reciprocal Rank Fusion.

        The request is validated before `embed_query` runs, so a blank query or an invalid
        filter costs no provider call.
        """
        return self.rank_channels(query, embed_query, filters=filters, top_k=top_k)["hybrid_rrf"]

    def rank_channels(
        self,
        query: str,
        embed_query: QueryEmbedder,
        *,
        filters: Mapping[str, str] | None = None,
        top_k: int = 8,
    ) -> dict[RankingChannel, list[SearchHit]]:
        """The top `top_k` hits of the BM25 channel, the dense channel and their RRF fusion.

        All three come from one query embedding and one lexical query, so they differ only in
        how candidates are ordered. Every hit carries its fused score and both channel ranks.
        """
        if not query.strip():
            raise InvalidFilterError("search query must not be blank")
        allowed = self.resolve_filters(filters)
        if not 1 <= top_k <= MAX_TOP_K:
            raise InvalidFilterError(f"top_k must be between 1 and {MAX_TOP_K}")
        query_vector = embed_query(query)
        if query_vector.shape != (self.embedding_dimensions,):
            raise ProviderError(
                f"query embedding shape {query_vector.shape} does not match snapshot dimension"
                f" {self.embedding_dimensions} ({self.embedding_model})"
            )
        lexical = self._lexical_ranks(query, allowed)
        candidates = [
            i for i, hit in enumerate(self._hits) if allowed is None or hit["doc_id"] in allowed
        ]
        similarity = self._matrix[candidates] @ query_vector.astype(np.float32)
        cosine = dict(zip(candidates, similarity.tolist(), strict=True))
        order = np.argsort(-similarity, kind="stable")[:CHANNEL_DEPTH]
        dense = {candidates[i]: rank for rank, i in enumerate(order.tolist(), start=1)}
        fused = {
            index: sum(1.0 / (RRF_K + ranks[index]) for ranks in (lexical, dense) if index in ranks)
            for index in lexical.keys() | dense.keys()
        }
        ranked = sorted(fused, key=lambda i: (-fused[i], -cosine[i], self._hits[i]["chunk_id"]))

        def hits(order: list[int]) -> list[SearchHit]:
            return [
                SearchHit(
                    **self._hits[i],
                    facets=self.facets_of(self._hits[i]["doc_id"]),
                    rrf_score=fused[i],
                    lexical_rank=lexical.get(i),
                    dense_rank=dense.get(i),
                )
                for i in order[:top_k]
            ]

        return {
            "bm25_only": hits(sorted(lexical, key=lexical.__getitem__)),
            "dense_only": hits(sorted(dense, key=dense.__getitem__)),
            "hybrid_rrf": hits(ranked),
        }

    def _lexical_ranks(self, query: str, allowed: frozenset[str] | None) -> dict[int, int]:
        """Rank of each matching chunk, keyed by its row index in `self._hits`."""
        fts_query = build_fts5_query(query)
        if fts_query is None:
            return {}
        doc_ids = None if allowed is None else json.dumps(sorted(allowed))
        rows = self._rows(
            "SELECT c.chunk_id, bm25(chunks_fts, 0.0, 10.0, 2.0) AS score"
            " FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id"
            " WHERE chunks_fts MATCH ?"
            " AND (? IS NULL OR c.doc_id IN (SELECT value FROM json_each(?)))"
            " ORDER BY score, c.chunk_id LIMIT ?",
            (fts_query, doc_ids, doc_ids, CHANNEL_DEPTH),
        )
        return {self._hit_index[row["chunk_id"]]: rank for rank, row in enumerate(rows, start=1)}

    def grep(
        self,
        pattern: str,
        *,
        filters: Mapping[str, str] | None = None,
        max_matches: int = 20,
    ) -> list[GrepMatch]:
        """Literal, case-insensitive, dash- and NFKC-folded line search over section text."""
        if not pattern.strip():
            raise InvalidFilterError("grep pattern must not be blank")
        if max_matches < 1:
            raise InvalidFilterError("max_matches must be at least 1")
        allowed = self.resolve_filters(filters)
        needle = fold_text(pattern)
        matches = (
            GrepMatch(
                **section.fields,
                facets=self.facets_of(section.doc_id),
                line_number=number,
                snippet=_snippet(line, needle),
            )
            for section in self._grep_sections
            if allowed is None or section.doc_id in allowed
            for number, (line, folded) in enumerate(
                zip(section.lines, section.normalized, strict=True), start=1
            )
            if needle in folded
        )
        return list(islice(matches, max_matches))

    def read_section(self, section_id: str) -> SectionReadView:
        rows = self._rows(
            f"SELECT {_SECTION_COLUMNS}, s.markdown,"
            " (SELECT p.section_id FROM sections p WHERE p.doc_id = s.doc_id"
            "  AND p.ordinal = s.ordinal - 1) AS prev_section_id,"
            " (SELECT n.section_id FROM sections n WHERE n.doc_id = s.doc_id"
            "  AND n.ordinal = s.ordinal + 1) AS next_section_id"
            " FROM sections s JOIN documents d ON d.doc_id = s.doc_id WHERE s.section_id = ?",
            (section_id,),
        )
        if not rows:
            raise NotFoundError(
                f"unknown section {section_id!r}; get_document_outline lists section IDs"
            )
        row = dict(rows[0])
        return SectionReadView.model_validate(row | {"facets": self.facets_of(row["doc_id"])})

    def view_page(self, doc_id: str, page_number: int) -> PageVisualView:
        doc = self.document(doc_id)
        rows = self._rows(
            "SELECT native_text, visual_summary, webp_bytes FROM pages"
            " WHERE doc_id = ? AND page_number = ?",
            (doc_id, page_number),
        )
        if not rows:
            valid = f"pages 1-{doc.page_count}" if doc.page_count else "no rendered pages"
            raise NotFoundError(f"{doc_id} has no page {page_number}; it has {valid}")
        return PageVisualView(
            **dict(rows[0]),
            doc_id=doc_id,
            doc_title=doc.title,
            page_number=page_number,
            page_count=doc.page_count,
        )
