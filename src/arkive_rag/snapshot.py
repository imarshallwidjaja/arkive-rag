"""The snapshot file: SQLite schema, atomic writer, and the read-only query surface."""

from __future__ import annotations

import errno
import os
import re
import sqlite3
import stat
import tempfile
import threading
import unicodedata
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice, pairwise
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

from arkive_rag import __version__
from arkive_rag.models import (
    CompiledCorpus,
    CuratorNote,
    DocumentInfo,
    DocumentOutline,
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
)

SCHEMA_VERSION = "2.0"
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
    sha256 TEXT NOT NULL, format TEXT NOT NULL, project_id TEXT NOT NULL, client TEXT NOT NULL,
    stage TEXT NOT NULL, title TEXT NOT NULL, doc_date TEXT, summary TEXT NOT NULL,
    page_count INTEGER NOT NULL);
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

# Conversational filler dropped from FTS queries when at least two content tokens remain.
_FILLER_WORDS = """what which where when who how does did is are was were the a an of in for to on
with from by about show tell give list find and or not"""
_STOPWORDS = frozenset(_FILLER_WORDS.split())
_DASHES = re.compile(r"[\u2010-\u2015\u2212]")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _grep_normalize(text: str) -> str:
    return _DASHES.sub("-", _nfkc(text)).casefold()


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
        "synthesis_markdown": corpus.synthesis_markdown,
    }
    with db:
        db.executemany("INSERT INTO corpus_meta VALUES (?, ?)", meta.items())
        db.executemany(
            "INSERT INTO curator_notes VALUES (?, ?, ?)",
            [(i, n.scope_path, n.content) for i, n in enumerate(corpus.curator_notes, start=1)],
        )
        row = 0
        for ordinal, doc in enumerate(corpus.documents, start=1):
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.doc_id,
                    ordinal,
                    doc.rel_path,
                    doc.sha256,
                    doc.format,
                    doc.project_id,
                    doc.client,
                    doc.stage,
                    doc.title,
                    doc.doc_date,
                    doc.summary,
                    len(doc.pages),
                ),
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
        if needle in _grep_normalize(line[:middle]):
            high = middle
        else:
            low = middle + 1
    centre = low - len(needle) // 2
    start = max(0, min(centre - SNIPPET_CHARS // 2, len(line) - SNIPPET_CHARS))
    end = start + SNIPPET_CHARS
    return f"{'...' if start else ''}{line[start:end]}{'...' if end < len(line) else ''}"


_HIT_COLUMNS = """c.chunk_id, c.doc_id, c.section_id, c.kind, d.project_id, d.client, d.stage,
    d.title AS doc_title, s.heading_path, c.page_start, c.page_end, c.content"""
_SECTION_COLUMNS = """s.section_id, s.doc_id, d.project_id, d.stage, d.title AS doc_title,
    s.heading_path, s.page_start, s.page_end"""
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
        meta = dict(db.execute("SELECT key, value FROM corpus_meta").fetchall())
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise SnapshotIntegrityError(
                f"{path}: schema version {meta.get('schema_version')!r}, expected {SCHEMA_VERSION}"
            )
        try:
            self.topic: str = meta["topic"]
            self.title: str = meta["title"]
            self.created_at: str = meta["created_at"]
            self.synthesis_markdown: str = meta["synthesis_markdown"]
            self.embedding_model: str = meta["embedding_model"]
            self.embedding_dimensions = int(meta["embedding_dimensions"])
        except (KeyError, ValueError) as exc:
            raise SnapshotIntegrityError(f"{path}: corpus_meta is malformed: {exc!r}") from exc
        # Snapshots written before these keys were recorded still open: the fingerprint is
        # recomputed from the documents table, and an unrecorded writer version reads as "".
        self.arkive_version: str = meta.get("arkive_version", "")
        db.row_factory = sqlite3.Row
        documents = db.execute(
            "SELECT doc_id, rel_path, sha256, format, project_id, client, stage, title, doc_date,"
            " summary, page_count FROM documents ORDER BY ordinal"
        )
        self._documents = {r["doc_id"]: DocumentInfo.model_validate(dict(r)) for r in documents}
        self.corpus_fingerprint = corpus_fingerprint(
            (d.rel_path, d.sha256) for d in self._documents.values()
        )
        recorded = meta.get("corpus_fingerprint", self.corpus_fingerprint)
        if recorded != self.corpus_fingerprint:
            raise SnapshotIntegrityError(
                f"{path}: recorded corpus_fingerprint {recorded} does not match its documents"
                f" ({self.corpus_fingerprint})"
            )
        self.projects = sorted({d.project_id for d in self._documents.values()})
        self.stages = sorted({d.stage for d in self._documents.values()})
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
                tuple(_grep_normalize(line) for line in lines),
            )
            for r in db.execute(
                f"SELECT {_SECTION_COLUMNS}, s.markdown FROM sections s"
                " JOIN documents d ON d.doc_id = s.doc_id ORDER BY s.doc_id, s.ordinal"
            )
        ]

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
        except sqlite3.DatabaseError as exc:
            db.close()
            raise SnapshotIntegrityError(f"{path} is not a valid snapshot: {exc}") from exc
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
            return self._db.execute(sql, params).fetchall()

    # Catalogue reads.

    def summary(self) -> TopicSummary:
        return TopicSummary(
            topic=self.topic,
            title=self.title,
            created_at=self.created_at,
            corpus_fingerprint=self.corpus_fingerprint,
            arkive_version=self.arkive_version,
            document_count=len(self._documents),
            projects=self.projects,
            stages=self.stages,
        )

    def guide(self) -> TopicGuide:
        notes = self._rows("SELECT scope_path, content FROM curator_notes ORDER BY ordinal")
        return TopicGuide(
            topic=self.topic,
            title=self.title,
            synthesis_markdown=self.synthesis_markdown,
            curator_notes=[CuratorNote.model_validate(dict(n)) for n in notes],
            projects=self.projects,
            stages=self.stages,
        )

    def _check_filters(self, project: str | None, stage: str | None) -> None:
        if project is not None and project not in self.projects:
            raise InvalidFilterError(
                f"unknown project {project!r}; valid projects: {', '.join(self.projects)}"
            )
        if stage is not None and stage not in self.stages:
            raise InvalidFilterError(
                f"unknown stage {stage!r}; valid stages: {', '.join(self.stages)}"
            )

    def _matches(self, doc_id: str, project: str | None, stage: str | None) -> bool:
        doc = self._documents[doc_id]
        return project in (None, doc.project_id) and stage in (None, doc.stage)

    def list_documents(
        self, *, project: str | None = None, stage: str | None = None
    ) -> list[DocumentInfo]:
        self._check_filters(project, stage)
        return [d for d in self._documents.values() if self._matches(d.doc_id, project, stage)]

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
        project: str | None = None,
        stage: str | None = None,
        top_k: int = 8,
    ) -> list[SearchHit]:
        """Hybrid search: BM25 and dense ranks fused by Reciprocal Rank Fusion.

        The request is validated before `embed_query` runs, so a blank query or an invalid
        filter costs no provider call.
        """
        return self.rank_channels(query, embed_query, project=project, stage=stage, top_k=top_k)[
            "hybrid_rrf"
        ]

    def rank_channels(
        self,
        query: str,
        embed_query: QueryEmbedder,
        *,
        project: str | None = None,
        stage: str | None = None,
        top_k: int = 8,
    ) -> dict[RankingChannel, list[SearchHit]]:
        """The top `top_k` hits of the BM25 channel, the dense channel and their RRF fusion.

        All three come from one query embedding and one lexical query, so they differ only in
        how candidates are ordered. Every hit carries its fused score and both channel ranks.
        """
        if not query.strip():
            raise InvalidFilterError("search query must not be blank")
        self._check_filters(project, stage)
        if not 1 <= top_k <= MAX_TOP_K:
            raise InvalidFilterError(f"top_k must be between 1 and {MAX_TOP_K}")
        query_vector = embed_query(query)
        if query_vector.shape != (self.embedding_dimensions,):
            raise ProviderError(
                f"query embedding shape {query_vector.shape} does not match snapshot dimension"
                f" {self.embedding_dimensions} ({self.embedding_model})"
            )
        lexical = self._lexical_ranks(query, project, stage)
        candidates = [
            i for i, hit in enumerate(self._hits) if self._matches(hit["doc_id"], project, stage)
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

    def _lexical_ranks(self, query: str, project: str | None, stage: str | None) -> dict[int, int]:
        """Rank of each matching chunk, keyed by its row index in `self._hits`."""
        fts_query = build_fts5_query(query)
        if fts_query is None:
            return {}
        rows = self._rows(
            "SELECT c.chunk_id, bm25(chunks_fts, 0.0, 10.0, 2.0) AS score"
            " FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id"
            " JOIN documents d ON d.doc_id = c.doc_id WHERE chunks_fts MATCH ?"
            " AND (? IS NULL OR d.project_id = ?) AND (? IS NULL OR d.stage = ?)"
            " ORDER BY score, c.chunk_id LIMIT ?",
            (fts_query, project, project, stage, stage, CHANNEL_DEPTH),
        )
        return {self._hit_index[row["chunk_id"]]: rank for rank, row in enumerate(rows, start=1)}

    def grep(
        self,
        pattern: str,
        *,
        project: str | None = None,
        stage: str | None = None,
        max_matches: int = 20,
    ) -> list[GrepMatch]:
        """Literal, case-insensitive, dash- and NFKC-folded line search over section text."""
        if not pattern.strip():
            raise InvalidFilterError("grep pattern must not be blank")
        if max_matches < 1:
            raise InvalidFilterError("max_matches must be at least 1")
        self._check_filters(project, stage)
        needle = _grep_normalize(pattern)
        matches = (
            GrepMatch(**section.fields, line_number=number, snippet=_snippet(line, needle))
            for section in self._grep_sections
            if self._matches(section.doc_id, project, stage)
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
        return SectionReadView.model_validate(dict(rows[0]))

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
