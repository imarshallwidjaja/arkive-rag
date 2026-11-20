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
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice, pairwise
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, TypeAdapter, ValidationError

from arkive_rag import __version__
from arkive_rag.models import (
    BuildReport,
    CompiledCorpus,
    CuratorNote,
    DeliverableType,
    DeliverableTypeSummary,
    DocumentInfo,
    DocumentListItem,
    DocumentOutline,
    DocumentPage,
    DraftCandidate,
    DraftExemplar,
    DraftPacket,
    ExemplarSection,
    Facet,
    FacetSummary,
    FacetValue,
    FamilyRecurrence,
    FeedbackLesson,
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
    SupplementalSection,
    TopicGuide,
    TopicPage,
    TopicSummary,
    UnsafePathError,
    choices,
    corpus_fingerprint,
    echoed,
    fold_facet_label,
    fold_text,
    heading_context,
    parse_printed_date,
)
from arkive_rag.passages import without_visual_transcripts

SCHEMA_VERSION = "6.0"
RRF_K = 60.0
CHANNEL_DEPTH = 300
MAX_TOP_K = 50
MAX_GREP_MATCHES = 50
SNIPPET_CHARS = 400
# Every MCP read result that can grow with the corpus stays within this many UTF-8 bytes and
# lines as the MCP server renders it (`response_text`): a page of documents, an outline, a
# section, a search or grep result list, a glossary page, the topic guide, a drafting packet.
# A client such as OpenCode cuts a tool result at 51,200 bytes or 2,000 lines, so the budget
# keeps a margin below both.
RESPONSE_MAX_BYTES = 45_000
RESPONSE_MAX_LINES = 1_600
# A view caption (`page_caption`) is plain text, so its lines are real lines; it keeps a
# further margin below `RESPONSE_MAX_LINES`.
PAGE_CAPTION_MAX_LINES = 1_500
MAX_DOCUMENT_PAGE = 100
# A document title is shown beside other results cut to this many characters, as
# `heading_context` cuts a heading path.
TITLE_CHARS = 500
# `list_topics` names at most this many values per facet and deliverable types per topic.
SUMMARY_MAX_NAMES = 50
# A topic summary fits a `list_topics` page of its own with this many bytes to spare for the
# page's `total`, `offset` and `next_offset`.
TOPIC_PAGE_MARGIN_BYTES = 512
# The topic guide shows each facet and deliverable type description cut to this length.
GUIDE_DESCRIPTION_CHARS = 300
# A glossary query longer than this cannot be part of any term or alias worth looking up.
MAX_GLOSSARY_QUERY_CHARS = 1_000
# Search shows at most this many chunks of one version family before it backfills.
SEARCH_FAMILY_CAP = 3
# The drafting packet.
MAX_PACKET_EXEMPLARS = 3
MAX_OTHER_CANDIDATES = 8
MAX_SUPPLEMENTAL_SECTIONS = 2
# A document's relevance sums the fused-rank scores of at most this many of its chunks, so a
# long document does not outrank a closer short one by its length alone.
RELEVANCE_CHUNKS = 3
# A candidate of another deliverable type fills a free exemplar slot when its relevance is at
# least this share of the best exemplar's.
CROSS_TYPE_RELEVANCE = 0.6
# At most this share of a first-ranked chunk's score is added for input words in a document's
# title, type and summary, which separates documents whose chunks rank alike.
METADATA_BONUS = 0.25
# The lexical query of a drafting request uses at most this many of the inputs' distinct words,
# the rarest in the topic first, since every word adds to the cost of a BM25 query.
DRAFT_QUERY_TERMS = 48
DRAFT_INPUT_WORDS = 400
# Only the start of long inputs is embedded; the lexical query reads every word.
DRAFT_EMBED_CHARS = 8_000
# A word of the inputs held by at most this many documents is rare enough to look for in
# sections outside the exemplars.
RARE_TERM_DOCUMENTS = 2
MAX_RARE_TERMS = 20
# Words of drafting instructions and generic report prose that say nothing about a project;
# with the query filler they never make a lexical query or a rare word.
_DRAFT_FILLER = """also although among based because before being between both could deliverable
document draft drafting each following from further have include includes including into make
more must need needs other please prepare project provide report should since such than that
their them then there these they this those through under using whether which while will within
without would write written your"""
# The packet is measured once at the end; the running estimate of its size before that is
# exact for everything but its counters' digits, which this margin covers.
PACKET_MARGIN_BYTES = 512
# The curator notes and the lessons each take at most this many bytes of the packet.
PACKET_NOTES_BYTES = 6_000
PACKET_LESSONS_BYTES = 6_000
OTHER_CANDIDATE_SUMMARY_CHARS = 300
# A document summary is shown cut to this many characters (a card summary is a few sentences),
# and an exemplar names at most this many other members of its version family.
SUMMARY_CHARS = 2_000
MAX_FAMILY_DOC_IDS = 24
# Samples of the recurring units the included exemplar sections hold, at most this many.
MAX_RECURRING_SAMPLES = 12
BOUNDARY_SECTIONS = 2
# Sections whose chunks rank this high for the inputs count as matching them, at most this
# many per exemplar, so recurring and boundary sections keep their priority.
MATCHED_CHUNK_RANKS = 60
MATCHED_SECTIONS = 6

DRAFTING_GUIDANCE = """\
- Take every fact, measurement, date, quantity, parameter, personnel list and deliverable
  file name for the new work only from the user's inputs. Enumerate listed items in full and
  never invent unstated measurements or results.
- Reuse and adapt the exemplars' domain methodology explanations, equations, literature
  references, generic table structures and technical caveats wherever the user's inputs use
  the same method, instrument, sensor or workflow.
- Treat each exemplar's date, facets and deliverable_type and each section's
  family_recurrence (exemplar_doc_ids and corpus_family_count) as factual provenance. A
  clause shared across several independent version families is established house text; a
  clause found in only one family (corpus_family_count 1) may be a newly adopted standard or
  a project- or modality-specific requirement. Weigh dates, client and project scope and the
  input modality: for example, do not copy audio or speech retention rules when no audio was
  recorded, or tool-usage disclosures confirmed only for another project. When working with a
  user and an institutional policy is genuinely uncertain, ask; when producing a standalone
  deliverable directly, omit unconfirmed claims that could misstate the project or the
  organisation.
- Match the closest applicable exemplar's structure, heading hierarchy and concise register,
  without meta-commentary about the inputs. Apply the lessons and curator notes. Call
  read_section for any relevant outline entry with included false. Write a clean standalone
  deliverable with no doc_id or section_id tokens."""

QueryEmbedder = Callable[[str], NDArray[np.float32]]
RankingChannel = Literal["bm25_only", "dense_only", "hybrid_rrf"]

SCHEMA = """
CREATE TABLE corpus_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE curator_notes (
    ordinal INTEGER PRIMARY KEY, scope_path TEXT NOT NULL, content TEXT NOT NULL);
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE, rel_path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL, format TEXT NOT NULL, title TEXT NOT NULL, doc_date TEXT,
    doc_date_iso TEXT, doc_date_basis TEXT CHECK (doc_date_basis IN ('printed', 'metadata')),
    deliverable_type TEXT NOT NULL, summary TEXT NOT NULL, page_count INTEGER NOT NULL,
    is_curator_note INTEGER NOT NULL CHECK (is_curator_note IN (0, 1)),
    is_feedback INTEGER NOT NULL CHECK (is_feedback IN (0, 1)),
    feedback_for_json TEXT NOT NULL,
    family_id TEXT NOT NULL REFERENCES documents(doc_id) DEFERRABLE INITIALLY DEFERRED,
    version_status TEXT NOT NULL
        CHECK (version_status IN ('sole', 'latest', 'superseded', 'unordered')),
    superseded_by TEXT REFERENCES documents(doc_id) DEFERRABLE INITIALLY DEFERRED,
    CHECK ((doc_date_iso IS NULL) = (doc_date_basis IS NULL)),
    CHECK ((version_status = 'superseded') = (superseded_by IS NOT NULL)));
CREATE TABLE deliverable_types (
    name TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE, description TEXT NOT NULL);
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
CREATE TABLE feedback_lessons (
    ordinal INTEGER PRIMARY KEY, source_doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    section_id TEXT NOT NULL REFERENCES sections(section_id), statement TEXT NOT NULL,
    target_types_json TEXT NOT NULL);
CREATE TABLE unit_clusters (
    cluster_id INTEGER PRIMARY KEY, kind TEXT NOT NULL CHECK (kind IN ('prose', 'table_header')),
    text TEXT NOT NULL);
CREATE TABLE cluster_sections (
    cluster_id INTEGER NOT NULL REFERENCES unit_clusters(cluster_id),
    section_id TEXT NOT NULL REFERENCES sections(section_id), position INTEGER NOT NULL,
    sample TEXT, PRIMARY KEY (cluster_id, section_id));
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED, content_norm, breadcrumb_norm,
    tokenize='porter unicode61 remove_diacritics 2');
"""
# Every table a snapshot of this schema holds; `TopicSnapshot.open` requires each one.
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
_DRAFT_STOPWORDS = _STOPWORDS | frozenset(_DRAFT_FILLER.split())
_FOLDERS: TypeAdapter[tuple[str, ...]] = TypeAdapter(tuple[str, ...])
_SETTINGS: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])
_TYPES: TypeAdapter[tuple[str, ...]] = TypeAdapter(tuple[str, ...])
_TEXT: TypeAdapter[str] = TypeAdapter(str)


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
            "INSERT INTO deliverable_types VALUES (?, ?, ?)",
            [(t.name, i, t.description) for i, t in enumerate(corpus.deliverable_types, start=1)],
        )
        # Text tables first and page images last, so sections and chunks, which every read
        # path scans, lie together in the file instead of between the page BLOBs.
        row = 0
        for ordinal, doc in enumerate(corpus.documents, start=1):
            role = doc.role
            db.execute(
                f"INSERT INTO documents VALUES ({', '.join('?' * 18)})",
                (
                    doc.doc_id,
                    ordinal,
                    doc.rel_path,
                    doc.sha256,
                    doc.format,
                    doc.title,
                    doc.doc_date,
                    doc.date.iso if doc.date else None,
                    doc.date.basis if doc.date else None,
                    role.deliverable_type,
                    doc.summary,
                    len(doc.pages),
                    int(doc.is_curator_note),
                    int(role.is_feedback),
                    json.dumps(list(role.feedback_for_types)),
                    role.family_id,
                    role.version_status,
                    role.superseded_by,
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
        db.executemany(
            "INSERT INTO feedback_lessons VALUES (?, ?, ?, ?, ?)",
            [
                (
                    i,
                    lesson.source_doc_id,
                    lesson.section_id,
                    lesson.statement,
                    json.dumps(list(lesson.target_types)),
                )
                for i, lesson in enumerate(corpus.feedback_lessons, start=1)
            ],
        )
        db.executemany(
            "INSERT INTO unit_clusters VALUES (?, ?, ?)",
            [(c.cluster_id, c.kind, c.text) for c in corpus.unit_clusters],
        )
        db.executemany(
            "INSERT INTO cluster_sections VALUES (?, ?, ?, ?)",
            [
                (c.cluster_id, h.section_id, position, h.sample)
                for c in corpus.unit_clusters
                for position, h in enumerate(c.holders, start=1)
            ],
        )
        for doc in corpus.documents:
            db.executemany(
                "INSERT INTO pages VALUES (?, ?, ?, ?, ?)",
                [
                    (doc.doc_id, p.page_number, p.native_text, p.visual_summary, p.webp_bytes)
                    for p in doc.pages
                ],
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


@dataclass(frozen=True)
class _SectionRow:
    section_id: str
    doc_id: str
    ordinal: int
    heading_path: str
    level: int
    page_start: int
    page_end: int
    markdown: str


@dataclass(frozen=True)
class _Ranking:
    """One query's channels over the candidate chunks, keyed by chunk row index."""

    lexical: dict[int, int]
    dense: dict[int, int]
    fused: dict[int, float]
    # Fused order: score, then cosine similarity, then row index.
    order: list[int]


def response_text(result: BaseModel | Sequence[BaseModel]) -> str:
    """The one text block an MCP client reads for a tool result: the result as compact JSON,
    a list as one JSON array. The MCP server renders every structured result with this
    function, so a result that `fits_response` reaches the client whole."""
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return f"[{','.join(item.model_dump_json() for item in result)}]"


def fits_response(
    result: BaseModel | Sequence[BaseModel], *, max_bytes: int = RESPONSE_MAX_BYTES
) -> bool:
    """Whether `response_text(result)` is within `max_bytes` UTF-8 bytes and
    `RESPONSE_MAX_LINES` lines."""
    text = response_text(result)
    return text.count("\n") < RESPONSE_MAX_LINES and len(text.encode()) <= max_bytes


def _json_bytes(value: BaseModel | str) -> int:
    """UTF-8 bytes of `value` as `response_text` writes it inside a result."""
    if isinstance(value, BaseModel):
        return len(value.model_dump_json().encode())
    return len(_TEXT.dump_json(value))


def _most_that_fit(
    most: int,
    make: Callable[[int], BaseModel | Sequence[BaseModel]],
    what: str,
    *,
    max_bytes: int = RESPONSE_MAX_BYTES,
) -> int:
    """The largest `n` in 0..`most` whose `make(n)` fits one tool result (`fits_response`
    within `max_bytes`), where a larger `n` never yields a smaller result. `what` names the
    result for the error raised when not even `make(0)` fits: every value a snapshot holds is
    bounded so that it does, so that is a damaged snapshot."""
    if not fits_response(make(0), max_bytes=max_bytes):
        raise SnapshotIntegrityError(f"{what} does not fit one tool result")
    low, high = 0, most
    while low < high:
        middle = (low + high + 1) // 2
        if fits_response(make(middle), max_bytes=max_bytes):
            low = middle
        else:
            high = middle - 1
    return low


def _clipped(text: str, chars: int) -> str:
    return text if len(text) <= chars else f"{text[: max(0, chars - 1)]}\u2026"


def _text_prefix(text: str, max_bytes: int, max_lines: int) -> str:
    """`text` whole when it has at most `max_bytes` UTF-8 bytes and `max_lines` lines, else its
    longest prefix within both, then a line counting the characters and lines left out."""
    if len(text.encode()) <= max_bytes and text.count("\n") < max_lines:
        return text
    # The last line and 120 bytes are left for the marker line.
    prefix = "\n".join(text.split("\n")[: max(1, max_lines - 1)])
    prefix = prefix.encode()[: max(0, max_bytes - 120)].decode(errors="ignore")
    rest = text[len(prefix) :]
    lines = rest.count("\n") + 1
    return f"{prefix}\n[... {len(rest)} characters on {lines} lines omitted ...]"


def topic_page(snapshots: Sequence[TopicSnapshot], offset: int) -> TopicPage:
    """One page of `list_topics`: the summaries (`TopicSnapshot.summary`) of the topics from
    `offset` that fit one tool result together. Only the topics on the page are summarised:
    each summary's exact JSON bytes are added to the page's until the next would not fit."""
    total = len(snapshots)
    if offset < 0 or (offset and offset >= total):
        raise InvalidFilterError(f"offset {offset} is outside the {total} topics")

    def page(topics: list[TopicSummary], count: int) -> TopicPage:
        end = offset + count
        return TopicPage(
            topics=topics,
            total=total,
            offset=offset,
            next_offset=end if end < total else None,
        )

    shown: list[TopicSummary] = []
    # Bytes of the shown summaries and the commas between them.
    size = 0
    for snapshot in snapshots[offset:]:
        summary = snapshot.summary()
        added = _json_bytes(summary) + (1 if shown else 0)
        if _json_bytes(page([], len(shown) + 1)) + size + added > RESPONSE_MAX_BYTES:
            break
        shown.append(summary)
        size += added
    if offset < total and not shown:
        raise SnapshotIntegrityError(
            f"topic {snapshots[offset].topic!r} does not fit one tool result"
        )
    result = page(shown, len(shown))
    if not fits_response(result):
        raise AssertionError(f"a page of {len(shown)} topics was priced to fit but does not")
    return result


def page_caption(page: PageVisualView) -> str:
    """The text `view_page` returns beside the page image: the document title cut to
    `TITLE_CHARS`, the visual summary and the native text, together within
    `RESPONSE_MAX_BYTES` bytes and `PAGE_CAPTION_MAX_LINES` lines. The visual summary takes
    at most half of what remains after the heading, the native text the rest; a part that
    does not fit is cut and says how much it left out. The image itself shows the whole
    page."""
    head = (
        f"{_clipped(page.doc_title, TITLE_CHARS)} ({page.doc_id}) page {page.page_number} of"
        f" {page.page_count}\n\nVisual summary:\n"
    )
    middle = "\n\nNative text:\n"
    room = RESPONSE_MAX_BYTES - len(head.encode()) - len(middle.encode())
    lines = PAGE_CAPTION_MAX_LINES - head.count("\n") - middle.count("\n") - 1
    visual = _text_prefix(page.visual_summary or "(none)", room // 2, lines // 2)
    native = _text_prefix(
        page.native_text, room - len(visual.encode()), lines - visual.count("\n") - 1
    )
    return f"{head}{visual}{middle}{native}"


def _visual_count(markdown: str) -> int:
    return len(markdown) and sum(
        1 for line in markdown.splitlines() if line.lstrip().startswith(_VISUAL_PREFIX)
    )


_VISUAL_PREFIX = "> **[Visual"
_DRAFT_WORD = re.compile(r"[^\W_]+")


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
            "SELECT doc_id, ordinal, rel_path, sha256, format, title, doc_date, doc_date_iso,"
            " doc_date_basis, deliverable_type, summary, page_count, is_curator_note,"
            " is_feedback, feedback_for_json, family_id, version_status, superseded_by"
            " FROM documents ORDER BY ordinal"
        ).fetchall()
        self._documents = {r["doc_id"]: self._document_info(path, r) for r in documents}
        self._ordinal = {r["doc_id"]: i for i, r in enumerate(documents)}
        self.corpus_fingerprint = corpus_fingerprint(
            (d.rel_path, d.sha256) for d in self._documents.values()
        )
        if meta["corpus_fingerprint"] != self.corpus_fingerprint:
            raise SnapshotIntegrityError(
                f"{path}: recorded corpus_fingerprint {meta['corpus_fingerprint']} does not match"
                f" its documents ({self.corpus_fingerprint})"
            )
        sections = [
            _SectionRow(**dict(r))
            for r in db.execute(
                "SELECT s.section_id, s.doc_id, s.ordinal, s.heading_path, s.level,"
                " s.page_start, s.page_end, s.markdown FROM sections s"
                " JOIN documents d ON d.doc_id = s.doc_id ORDER BY d.ordinal, s.ordinal"
            )
        ]
        self._sections = {s.section_id: s for s in sections}
        self._doc_sections: dict[str, list[_SectionRow]] = {}
        for section in sections:
            self._doc_sections.setdefault(section.doc_id, []).append(section)
        self._load_families()
        self._load_knowledge(db)
        # Chunks and sections load in path order (document ordinal, then their own ordinal),
        # the order `list_documents` gives. A chunk's row index is its tie-break in every
        # ranking, so ties never fall to the hash order of document IDs.
        rows = db.execute(
            f"SELECT {_HIT_COLUMNS}, c.embedding {_FROM_CHUNKS} ORDER BY d.ordinal, c.ordinal"
        ).fetchall()
        # Hits show the title and heading path cut, as every other result does.
        self._hits = [
            {k: v for k, v in dict(row).items() if k != "embedding"}
            | {
                "doc_title": _clipped(row["doc_title"], TITLE_CHARS),
                "heading_path": heading_context(row["heading_path"]),
            }
            for row in rows
        ]
        self._hit_index = {hit["chunk_id"]: i for i, hit in enumerate(self._hits)}
        dim = self.embedding_dimensions
        if any(len(row["embedding"]) != dim * 4 for row in rows):
            raise SnapshotIntegrityError(f"{path}: an embedding does not have {dim} dimensions")
        vectors = b"".join(row["embedding"] for row in rows)
        self._matrix = np.frombuffer(vectors, dtype=np.float32).reshape(len(rows), dim)
        self._grep_sections = [
            _GrepSection(
                {
                    "section_id": s.section_id,
                    "doc_id": s.doc_id,
                    "doc_title": _clipped(self._documents[s.doc_id].title, TITLE_CHARS),
                    "heading_path": heading_context(s.heading_path),
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                },
                s.doc_id,
                tuple(lines := s.markdown.split("\n")),
                tuple(fold_text(line) for line in lines),
            )
            for s in sections
        ]

    def _document_info(self, path: Path, row: sqlite3.Row) -> DocumentInfo:
        """One `documents` row as a DocumentInfo. A calendar date and its basis come together,
        with a date text that agrees with it: a metadata date's text is the date itself, and a
        printed date's text parses to it. It must also be a real date; a row that breaks this is a
        damaged snapshot."""
        fields = dict(row)
        fields.pop("ordinal")
        iso, basis = fields.pop("doc_date_iso"), fields.pop("doc_date_basis")
        text = fields["doc_date"]
        agrees = (
            iso is None
            if basis is None
            else iso is not None
            and text is not None
            and (text == iso if basis == "metadata" else parse_printed_date(text) == iso)
        )
        if not agrees:
            raise SnapshotIntegrityError(
                f"{path}: document {row['doc_id']} has an inconsistent date"
                f" (doc_date={fields['doc_date']!r}, doc_date_iso={iso!r},"
                f" doc_date_basis={basis!r})"
            )
        # A malformed value, such as an impossible calendar date, fails validation here and
        # `open` reports the snapshot as not valid.
        return DocumentInfo.model_validate(
            fields
            | {
                "date": None if iso is None else {"iso": iso, "basis": basis},
                "is_curator_note": bool(row["is_curator_note"]),
                "is_feedback": bool(fields.pop("is_feedback")),
                "feedback_for_types": _TYPES.validate_json(fields.pop("feedback_for_json")),
                "facets": self.facets_of(row["doc_id"]),
            }
        )

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

    def _load_families(self) -> None:
        """Version families, each checked: its ID is its first member in path order, a
        superseded member names another member that is not superseded itself (so supersession
        takes one step and has no cycle), the `latest` member when the family has one, a family
        of one is `sole` and only a family of one is, at most one member is `latest`, and a
        curator note is a family of its own."""
        families: dict[str, list[str]] = {}
        for doc in self._documents.values():
            families.setdefault(doc.family_id, []).append(doc.doc_id)
        for family_id, members in families.items():
            statuses = [self._documents[m].version_status for m in members]
            problem = None
            if members[0] != family_id:
                problem = "its ID is not its first member in path order"
            elif (len(members) == 1) != (statuses == ["sole"]) or (
                len(members) > 1 and "sole" in statuses
            ):
                problem = "a family of one must be sole, and only a family of one is"
            elif statuses.count("latest") > 1:
                problem = "it has two latest members"
            elif any(self._documents[m].is_curator_note for m in members) and len(members) > 1:
                problem = "it holds a curator note"
            elif any(
                (d := self._documents[m]).superseded_by not in (None, *members)
                or d.superseded_by == m
                for m in members
            ):
                problem = "a member is superseded by a document outside the family"
            elif any(
                (target := self._documents[m].superseded_by) is not None
                and self._documents[target].version_status == "superseded"
                for m in members
            ):
                problem = "a member is superseded by a superseded member"
            elif "latest" in statuses and any(
                self._documents[m].version_status == "superseded"
                and self._documents[self._documents[m].superseded_by or m].version_status
                != "latest"
                for m in members
            ):
                problem = "a superseded member does not name the latest member"
            if problem:
                raise SnapshotIntegrityError(f"{self.path}: version family {family_id}: {problem}")
        self._families = families

    def _load_knowledge(self, db: sqlite3.Connection) -> None:
        """Glossary, deliverable types, feedback lessons and recurring units, validated and
        with every reference resolved."""
        citable = {
            s.section_id: s.doc_id
            for s in self._sections.values()
            if not self._documents[s.doc_id].is_curator_note
        }
        try:
            self.glossary = [
                GlossaryEntry.model_validate_json(r["entry_json"])
                for r in db.execute("SELECT entry_json FROM glossary ORDER BY ordinal")
            ]
            self.deliverable_types = [
                DeliverableType(name=r["name"], description=r["description"])
                for r in db.execute(
                    "SELECT name, description FROM deliverable_types ORDER BY ordinal"
                )
            ]
            self.feedback_lessons = [
                FeedbackLesson(
                    statement=r["statement"],
                    source_doc_id=r["source_doc_id"],
                    section_id=r["section_id"],
                    doc_title=self._documents[r["source_doc_id"]].title,
                    target_types=_TYPES.validate_json(r["target_types_json"]),
                )
                for r in db.execute(
                    "SELECT source_doc_id, section_id, statement, target_types_json"
                    " FROM feedback_lessons ORDER BY ordinal"
                )
            ]
        except ValidationError as exc:
            raise SnapshotIntegrityError(
                f"{self.path}: a glossary, deliverable type or lesson row is malformed: {exc}"
            ) from exc
        types: dict[str, str] = {}
        for t in self.deliverable_types:
            # `prepare_draft` resolves a type by this folded label, so two would be ambiguous.
            folded = fold_facet_label(t.name)
            if folded in types:
                raise SnapshotIntegrityError(
                    f"{self.path}: deliverable types {types[folded]!r} and {t.name!r} fold alike"
                )
            types[folded] = t.name
        self._types_by_fold = types
        names = set(types.values())
        for doc in self._documents.values():
            if doc.is_curator_note:
                if doc.is_feedback or doc.feedback_for_types:
                    raise SnapshotIntegrityError(
                        f"{self.path}: curator note {doc.doc_id} has a feedback role"
                    )
                continue
            unknown = [t for t in (doc.deliverable_type, *doc.feedback_for_types) if t not in names]
            if unknown or doc.is_feedback != bool(doc.feedback_for_types):
                raise SnapshotIntegrityError(
                    f"{self.path}: document {doc.doc_id} has an inconsistent role"
                    f" (deliverable_type={doc.deliverable_type!r}, is_feedback={doc.is_feedback},"
                    f" feedback_for_types={doc.feedback_for_types})"
                )
        for lesson in self.feedback_lessons:
            source = self._documents[lesson.source_doc_id]
            if (
                citable.get(lesson.section_id) != lesson.source_doc_id
                or not source.is_feedback
                or any(t not in names for t in lesson.target_types)
            ):
                raise SnapshotIntegrityError(
                    f"{self.path}: feedback lesson {lesson.statement!r} does not cite a section"
                    " of a feedback document or names an unknown deliverable type"
                )
        cited = [
            (f"glossary term {e.term!r}", c)
            for e in self.glossary
            for s in e.senses
            for c in s.citations
        ]
        for label, section_id in cited:
            if section_id not in citable:
                raise SnapshotIntegrityError(
                    f"{self.path}: {label} cites {section_id!r}, which is not a citable section"
                    " of this snapshot"
                )
        # Recurring prose units, by ID; the rest are table header rows.
        self._prose_clusters: set[int] = set()
        self._cluster_sections: dict[int, list[str]] = {}
        # Each holder's sample of a recurring prose unit, by unit and section (`ClusterHolder`).
        self._cluster_samples: dict[tuple[int, str], str] = {}
        for r in db.execute("SELECT cluster_id, kind FROM unit_clusters ORDER BY cluster_id"):
            if r["kind"] == "prose":
                self._prose_clusters.add(r["cluster_id"])
            self._cluster_sections[r["cluster_id"]] = []
        for r in db.execute(
            "SELECT cluster_id, section_id, sample FROM cluster_sections"
            " ORDER BY cluster_id, position"
        ):
            cluster_id, section_id, sample = r["cluster_id"], r["section_id"], r["sample"]
            prose = cluster_id in self._prose_clusters
            if not (bool(sample and sample.strip()) if prose else sample is None):
                rule = "needs a nonblank sample" if prose else "has no sample"
                raise SnapshotIntegrityError(
                    f"{self.path}: section {section_id} holds recurring unit {cluster_id};"
                    f" each holder of a {'prose unit' if prose else 'table header row'} {rule}"
                )
            self._cluster_sections[cluster_id].append(section_id)
            if prose:
                self._cluster_samples[cluster_id, section_id] = sample
        self._section_clusters: dict[str, list[int]] = {}
        # Each recurring unit's holders that count (`FamilyRecurrence`): no superseded draft.
        self._cluster_holders: dict[int, frozenset[str]] = {}
        for cluster_id, section_ids in self._cluster_sections.items():
            if len({citable.get(s) for s in section_ids} - {None}) < 2 or not all(
                s in citable for s in section_ids
            ):
                raise SnapshotIntegrityError(
                    f"{self.path}: recurring unit {cluster_id} is not held by citable sections of"
                    " two documents"
                )
            if any(self._documents[citable[s]].is_feedback for s in section_ids):
                raise SnapshotIntegrityError(
                    f"{self.path}: recurring unit {cluster_id} is held by a feedback document;"
                    " only deliverables hold recurring units"
                )
            for section_id in section_ids:
                self._section_clusters.setdefault(section_id, []).append(cluster_id)
            self._cluster_holders[cluster_id] = frozenset(
                d
                for d in (citable[s] for s in section_ids)
                if self._documents[d].version_status != "superseded"
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

    @property
    def recurring_unit_count(self) -> int:
        """How many text units recur across documents of this topic (`UnitCluster`)."""
        return len(self._cluster_sections)

    def _active_pool(self, allowed: frozenset[str] | None) -> frozenset[str]:
        """The documents of `allowed` (every document when None) that are not superseded by
        another document of it: a superseded draft stays only while the version superseding it
        is filtered out."""
        pool = frozenset(self._documents) if allowed is None else allowed
        return frozenset(d for d in pool if self._documents[d].superseded_by not in pool)

    def facets_of(self, doc_id: str) -> dict[str, list[str]]:
        """The document's values per facet, in facet order; facets it holds no value of omitted."""
        return {k: list(v) for k, v in self._doc_facets.get(doc_id, {}).items()}

    # Catalogue reads.

    def type_summaries(self) -> list[DeliverableTypeSummary]:
        citable = [d for d in self._documents.values() if not d.is_curator_note]
        return [
            DeliverableTypeSummary(
                deliverable_type=t.name,
                description=t.description,
                document_count=sum(d.deliverable_type == t.name for d in citable),
                feedback_count=sum(t.name in d.feedback_for_types for d in citable),
            )
            for t in self.deliverable_types
        ]

    def summary(self) -> TopicSummary:
        """The topic as `list_topics` shows it, within one tool result: at most
        `SUMMARY_MAX_NAMES` values per facet and deliverable types, fewer when more would not
        fit, each list counting what it leaves out. The title is cut to `TITLE_CHARS`, and the
        summary fits a `list_topics` page of its own with `TOPIC_PAGE_MARGIN_BYTES` to spare."""
        types = [t.name for t in self.deliverable_types]

        def make(names: int) -> TopicSummary:
            return TopicSummary(
                topic=self.topic,
                title=_clipped(self.title, TITLE_CHARS),
                created_at=self.created_at,
                corpus_fingerprint=self.corpus_fingerprint,
                arkive_version=self.arkive_version,
                document_count=len(self._documents),
                facets=[
                    FacetSummary(
                        name=f.name,
                        binding=f.binding,
                        values=[v.value for v in f.values[:names]],
                        omitted_values=max(0, len(f.values) - names),
                    )
                    for f in self.facets
                ],
                deliverable_types=types[:names],
                omitted_deliverable_types=max(0, len(types) - names),
                glossary_terms=len(self.glossary),
            )

        def page(names: int) -> TopicPage:
            return TopicPage(topics=[make(names)], total=1, offset=0, next_offset=None)

        names = _most_that_fit(
            SUMMARY_MAX_NAMES,
            page,
            f"{self.path}: the topic summary",
            max_bytes=RESPONSE_MAX_BYTES - TOPIC_PAGE_MARGIN_BYTES,
        )
        return make(names)

    def curator_notes(self) -> list[CuratorNote]:
        notes = self._rows("SELECT scope_path, content FROM curator_notes ORDER BY ordinal")
        return [CuratorNote.model_validate(dict(n)) for n in notes]

    def guide(self) -> TopicGuide:
        """The topic guide within one tool result. The guide text (at most 24,000 bytes, as the
        build checks) and the facet names and descriptions come first; then as many
        deliverable types as fit; then curator notes in path order (a note that does not fit
        is left out and later ones may still fit); then as many values per facet as fit; then
        glossary terms in order while they fit. Descriptions are cut to
        `GUIDE_DESCRIPTION_CHARS`."""
        notes = self.curator_notes()
        terms = [e.term for e in self.glossary]
        types = [
            t.model_copy(update={"description": _clipped(t.description, GUIDE_DESCRIPTION_CHARS)})
            for t in self.type_summaries()
        ]
        facets = [
            f.model_copy(update={"description": _clipped(f.description, GUIDE_DESCRIPTION_CHARS)})
            for f in self.facets
        ]
        most_values = max((len(f.values) for f in facets), default=0)
        shown: list[CuratorNote] = []

        def make(type_count: int, values: int, term_count: int) -> TopicGuide:
            return TopicGuide(
                topic=self.topic,
                title=self.title,
                guide_markdown=self.guide_markdown,
                curator_notes=shown,
                omitted_curator_notes=len(notes) - len(shown),
                facets=[f.model_copy(update={"values": f.values[:values]}) for f in facets],
                omitted_facet_values=sum(max(0, len(f.values) - values) for f in facets),
                deliverable_types=types[:type_count],
                omitted_deliverable_types=len(types) - type_count,
                glossary_terms=terms[:term_count],
                omitted_glossary_terms=len(terms) - term_count,
            )

        what = f"{self.path}: the topic guide"
        kinds = _most_that_fit(len(types), lambda n: make(n, 0, 0), what)
        for note in notes:
            shown.append(note)
            if not fits_response(make(kinds, 0, 0)):
                shown.pop()
        values = _most_that_fit(most_values, lambda n: make(kinds, n, 0), what)
        terms_shown = _most_that_fit(len(terms), lambda n: make(kinds, values, n), what)
        return make(kinds, values, terms_shown)

    def lookup_glossary(self, query: str | None = None, *, offset: int = 0) -> GlossaryLookup:
        """Every term without a query; with one, entries whose folded term or alias contains it.
        One page from `offset` within one tool result. An entry too large for a page of its own
        keeps as many citations per sense as fit, counting the rest in `omitted_citations`."""
        if offset < 0:
            raise InvalidFilterError("offset must not be negative")
        if query is None:
            items: list[Any] = [
                GlossaryTermSummary(
                    term=e.term,
                    kind=e.kind,
                    aliases=list(e.aliases),
                    defined=any(s.definition is not None for s in e.senses),
                )
                for e in self.glossary
            ]
        else:
            needle = fold_text(query).strip()
            if not needle:
                raise InvalidFilterError("glossary query must not be blank; omit it to list terms")
            if len(query) > MAX_GLOSSARY_QUERY_CHARS:
                raise InvalidFilterError(
                    f"glossary query has {len(query)} characters; give at most"
                    f" {MAX_GLOSSARY_QUERY_CHARS}, such as a term or part of one"
                )
            items = [
                e
                for e in self.glossary
                if any(needle in fold_text(f) for f in (e.term, *e.aliases))
            ]
        rest = items[offset:]

        def make(page: Sequence[Any]) -> GlossaryLookup:
            end = offset + len(page)
            return GlossaryLookup(
                query=query,
                terms=list(page) if query is None else [],
                matches=[] if query is None else list(page),
                total=len(items),
                offset=offset,
                next_offset=end if end < len(items) else None,
            )

        what = f"{self.path}: a glossary page"
        count = _most_that_fit(len(rest), lambda n: make(rest[:n]), what)
        if count or not rest:
            return make(rest[:count])
        first = rest[0]
        if isinstance(first, GlossaryEntry):
            most = max(len(s.citations) for s in first.senses)
            keep = _most_that_fit(most, lambda n: make([_with_citations(first, n)]), what)
            if keep:
                return make([_with_citations(first, keep)])
        raise SnapshotIntegrityError(
            f"{self.path}: glossary entry {first.term!r} does not fit one tool result"
        )

    def resolve_filters(self, filters: Mapping[str, str] | None) -> frozenset[str] | None:
        """The documents holding every filter's value (keys combine with AND), or None when
        there is no filter. A value is a canonical facet value or one of its aliases."""
        if not filters:
            return None
        allowed: set[str] | None = None
        for facet, value in sorted(filters.items()):
            if facet not in self._aliases:
                valid = choices(f.name for f in self.facets) or "none, this topic has no facets"
                raise InvalidFilterError(f"unknown facet {echoed(facet)}; valid facets: {valid}")
            if not fold_facet_label(value):
                raise InvalidFilterError(f"{facet} filter value must not be blank")
            canonical = self._aliases[facet].get(fold_facet_label(value))
            if canonical is None:
                known = next(f for f in self.facets if f.name == facet).values
                raise InvalidFilterError(
                    f"unknown {facet} value {echoed(value)}; valid values: "
                    + choices(v.value for v in known)
                )
            docs = self._docs_by_value.get((facet, canonical), set())
            allowed = set(docs) if allowed is None else allowed & docs
        return frozenset(allowed or ())

    def list_documents(self, *, filters: Mapping[str, str] | None = None) -> list[DocumentInfo]:
        """Every document passing `filters`, in path order, unpaginated."""
        allowed = self.resolve_filters(filters)
        return [d for d in self._documents.values() if allowed is None or d.doc_id in allowed]

    def list_documents_page(
        self, *, filters: Mapping[str, str] | None = None, limit: int = 50, offset: int = 0
    ) -> DocumentPage:
        """One page of `list_documents` without source hashes: at most `limit` documents from
        `offset`, fewer when more would not fit one tool result, each summary cut to
        `SUMMARY_CHARS`."""
        if not 1 <= limit <= MAX_DOCUMENT_PAGE:
            raise InvalidFilterError(f"limit must be between 1 and {MAX_DOCUMENT_PAGE}")
        if offset < 0:
            raise InvalidFilterError("offset must not be negative")
        documents = self.list_documents(filters=filters)
        items = [
            DocumentListItem.model_validate(
                d.model_dump(exclude={"sha256"})
                | {
                    "title": _clipped(d.title, TITLE_CHARS),
                    "summary": _clipped(d.summary, SUMMARY_CHARS),
                }
            )
            for d in documents[offset : offset + limit]
        ]

        def make(page: Sequence[DocumentListItem]) -> DocumentPage:
            end = offset + len(page)
            return DocumentPage(
                documents=list(page),
                total=len(documents),
                offset=offset,
                limit=limit,
                next_offset=end if end < len(documents) else None,
            )

        what = f"{self.path}: a document page"
        count = _most_that_fit(len(items), lambda n: make(items[:n]), what)
        if items and not count:
            raise SnapshotIntegrityError(
                f"{self.path}: document {items[0].doc_id} does not fit one tool result"
            )
        return make(items[:count])

    def document(self, doc_id: str) -> DocumentInfo:
        if doc_id not in self._documents:
            raise NotFoundError(
                f"unknown document {echoed(doc_id)}; call list_documents for valid IDs"
            )
        return self._documents[doc_id]

    def outline(self, doc_id: str, *, offset: int = 0) -> DocumentOutline:
        """One page of the document's outline from section `offset`, as many sections as fit
        one tool result, each heading path cut by `heading_context` and the summary cut to
        `SUMMARY_CHARS`."""
        document = self.document(doc_id)
        rows = self._doc_sections.get(doc_id, [])
        if offset < 0 or (offset and offset >= len(rows)):
            raise InvalidFilterError(
                f"offset {offset} is outside the {len(rows)} sections of {doc_id}"
            )
        sections = [
            SectionOutlineItem(
                section_id=s.section_id,
                ordinal=s.ordinal,
                heading_path=heading_context(s.heading_path),
                level=s.level,
                page_start=s.page_start,
                page_end=s.page_end,
            )
            for s in rows[offset:]
        ]

        shown = document.model_copy(
            update={
                "title": _clipped(document.title, TITLE_CHARS),
                "summary": _clipped(document.summary, SUMMARY_CHARS),
            }
        )

        def make(count: int) -> DocumentOutline:
            end = offset + count
            return DocumentOutline(
                document=shown,
                sections=sections[:count],
                total_sections=len(rows),
                offset=offset,
                next_offset=end if end < len(rows) else None,
            )

        count = _most_that_fit(len(sections), make, f"{self.path}: the outline of {doc_id}")
        if sections and not count:
            raise SnapshotIntegrityError(
                f"{self.path}: document {doc_id} does not fit one tool result"
            )
        return make(count)

    def full_outline(self, doc_id: str) -> list[SectionOutlineItem]:
        """Every section of the document in reading order with its whole heading path,
        unpaginated, for Python callers such as `inspect` and the evaluation harness;
        `outline` is the MCP view."""
        self.document(doc_id)
        return [
            SectionOutlineItem(
                section_id=s.section_id,
                ordinal=s.ordinal,
                heading_path=s.heading_path,
                level=s.level,
                page_start=s.page_start,
                page_end=s.page_end,
            )
            for s in self._doc_sections.get(doc_id, [])
        ]

    def section_markdown(self, section_id: str) -> str:
        """The section's whole Markdown, for Python callers; `read_section` is the MCP view."""
        return self._section(section_id).markdown

    def _section(self, section_id: str) -> _SectionRow:
        section = self._sections.get(section_id)
        if section is None:
            raise NotFoundError(
                f"unknown section {echoed(section_id)}; get_document_outline lists section IDs"
            )
        return section

    # Retrieval.

    def _rank(
        self,
        fts_query: str | None,
        query_vector: NDArray[np.float32] | None,
        allowed: frozenset[str] | None,
    ) -> _Ranking:
        """The BM25 channel, the dense channel when a query vector is given, and their RRF
        fusion over the chunks of `allowed` documents, each channel `CHANNEL_DEPTH` deep."""
        lexical = self._lexical_ranks(fts_query, allowed)
        dense: dict[int, int] = {}
        cosine: dict[int, float] = {}
        if query_vector is not None:
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
        ranked = sorted(fused, key=lambda i: (-fused[i], -cosine.get(i, 0.0), i))
        return _Ranking(lexical, dense, fused, ranked)

    def _hit(self, ranking: _Ranking, index: int, more: int) -> SearchHit:
        return SearchHit(
            **self._hits[index],
            facets=self.facets_of(self._hits[index]["doc_id"]),
            rrf_score=ranking.fused[index],
            lexical_rank=ranking.lexical.get(index),
            dense_rank=ranking.dense.get(index),
            more_from_document=more,
        )

    def _hits_of(self, ranking: _Ranking, order: list[int], top_k: int) -> list[SearchHit]:
        shown = order[:top_k]
        held = Counter(self._hits[i]["doc_id"] for i in order)
        returned = Counter(self._hits[i]["doc_id"] for i in shown)
        return [
            self._hit(ranking, i, held[d] - returned[d])
            for i in shown
            for d in [self._hits[i]["doc_id"]]
        ]

    def _query_vector(self, query: str, embed_query: QueryEmbedder) -> NDArray[np.float32]:
        vector = embed_query(query)
        if vector.shape != (self.embedding_dimensions,):
            raise ProviderError(
                f"query embedding shape {vector.shape} does not match snapshot dimension"
                f" {self.embedding_dimensions} ({self.embedding_model})"
            )
        return vector

    def search(
        self,
        query: str,
        embed_query: QueryEmbedder,
        *,
        filters: Mapping[str, str] | None = None,
        top_k: int = 8,
    ) -> list[SearchHit]:
        """Hybrid search: BM25 and dense ranks fused by Reciprocal Rank Fusion, diversified by
        version family.

        Only the active pool is ranked (`_active_pool`): a superseded document's chunks are
        left out before ranking when the member that supersedes it passes the filters, so they
        never take the channels' places. At most `SEARCH_FAMILY_CAP` chunks of one family are
        taken first; when that leaves fewer than `top_k` hits (as when the filters name one
        document), the held-back chunks fill the list. Hits keep fused-rank order, the list
        fits one tool result, and each hit counts the further ranked chunks of its document the
        list does not show. The request is validated before `embed_query` runs, so a blank
        query or an invalid filter costs no provider call.
        """
        allowed = self._validate_search(query, filters, top_k)
        ranking = self._rank(
            build_fts5_query(query),
            self._query_vector(query, embed_query),
            self._active_pool(allowed),
        )
        order = ranking.order
        taken: list[int] = []
        held_back: list[int] = []
        per_family: Counter[str] = Counter()
        for i in order:
            family = self._documents[self._hits[i]["doc_id"]].family_id
            if len(taken) >= top_k:
                break
            if per_family[family] < SEARCH_FAMILY_CAP:
                per_family[family] += 1
                taken.append(i)
            else:
                held_back.append(i)
        taken += held_back[: top_k - len(taken)]
        rank = {i: n for n, i in enumerate(order)}
        shown = sorted(taken, key=rank.__getitem__)

        def make(count: int) -> list[SearchHit]:
            kept = set(shown[:count])
            return self._hits_of(
                ranking, [*shown[:count], *(i for i in order if i not in kept)], count
            )

        count = _most_that_fit(len(shown), make, f"{self.path}: a search result")
        if shown and not count:
            raise SnapshotIntegrityError(f"{self.path}: a search hit does not fit one tool result")
        return make(count)

    def _validate_search(
        self, query: str, filters: Mapping[str, str] | None, top_k: int
    ) -> frozenset[str] | None:
        if not query.strip():
            raise InvalidFilterError("search query must not be blank")
        allowed = self.resolve_filters(filters)
        if not 1 <= top_k <= MAX_TOP_K:
            raise InvalidFilterError(f"top_k must be between 1 and {MAX_TOP_K}")
        return allowed

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
        how candidates are ordered; no family diversity or size limit applies. Every hit
        carries its fused score and both channel ranks.
        """
        allowed = self._validate_search(query, filters, top_k)
        ranking = self._rank(
            build_fts5_query(query), self._query_vector(query, embed_query), allowed
        )
        return {
            "bm25_only": self._hits_of(
                ranking, sorted(ranking.lexical, key=ranking.lexical.__getitem__), top_k
            ),
            "dense_only": self._hits_of(
                ranking, sorted(ranking.dense, key=ranking.dense.__getitem__), top_k
            ),
            "hybrid_rrf": self._hits_of(ranking, ranking.order, top_k),
        }

    def _lexical_ranks(
        self, fts_query: str | None, allowed: frozenset[str] | None
    ) -> dict[int, int]:
        """Rank of each matching chunk, keyed by its row index in `self._hits`."""
        if fts_query is None:
            return {}
        doc_ids = None if allowed is None else json.dumps(sorted(allowed))
        rows = self._rows(
            "SELECT c.chunk_id, bm25(chunks_fts, 0.0, 10.0, 2.0) AS score"
            " FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id"
            " JOIN documents d ON d.doc_id = c.doc_id"
            " WHERE chunks_fts MATCH ?"
            " AND (? IS NULL OR c.doc_id IN (SELECT value FROM json_each(?)))"
            " ORDER BY score, d.ordinal, c.ordinal LIMIT ?",
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
        if not 1 <= max_matches <= MAX_GREP_MATCHES:
            raise InvalidFilterError(f"max_matches must be between 1 and {MAX_GREP_MATCHES}")
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
        found = list(islice(matches, max_matches))
        count = _most_that_fit(len(found), lambda n: found[:n], f"{self.path}: a grep result")
        if found and not count:
            raise SnapshotIntegrityError(
                f"{self.path}: grep match in {found[0].section_id} does not fit one tool result"
            )
        return found[:count]

    def read_section(self, section_id: str, *, offset: int = 0) -> SectionReadView:
        """The section's Markdown from character `offset`, as much as fits one tool result: the
        rest of the section when it fits, else up to the last line break that fits (or a cut
        within a line longer than one result)."""
        section = self._section(section_id)
        text = section.markdown
        if offset < 0 or (offset and offset >= len(text)):
            raise InvalidFilterError(
                f"offset {offset} is outside the {len(text)} characters of {section_id}"
            )
        siblings = self._doc_sections[section.doc_id]
        position = siblings.index(section)

        def make(end: int) -> SectionReadView:
            return SectionReadView(
                section_id=section.section_id,
                doc_id=section.doc_id,
                doc_title=_clipped(self._documents[section.doc_id].title, TITLE_CHARS),
                heading_path=heading_context(section.heading_path),
                page_start=section.page_start,
                page_end=section.page_end,
                facets=self.facets_of(section.doc_id),
                markdown=text[offset:end],
                offset=offset,
                next_offset=end if end < len(text) else None,
                total_chars=len(text),
                prev_section_id=siblings[position - 1].section_id if position else None,
                next_section_id=(
                    siblings[position + 1].section_id if position + 1 < len(siblings) else None
                ),
            )

        end = offset + _most_that_fit(
            len(text) - offset, lambda n: make(offset + n), f"{self.path}: section {section_id}"
        )
        if end < len(text):
            line_end = text.rfind("\n", offset, end)
            end = line_end + 1 if line_end > offset else end
        if end == offset and text:
            raise SnapshotIntegrityError(
                f"{self.path}: section {section_id} metadata does not fit one tool result"
            )
        return make(end)

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

    # Drafting.

    def _deliverables(self, active: frozenset[str]) -> list[DocumentInfo]:
        """The documents of `active` a draft can follow: no curator note, no feedback."""
        return [
            d
            for d in self._documents.values()
            if d.doc_id in active and not d.is_curator_note and not d.is_feedback
        ]

    def _draft_terms(
        self, project_inputs: str, deliverables: Sequence[DocumentInfo]
    ) -> tuple[str | None, dict[str, set[str]]]:
        """The lexical query of a drafting request, and the inputs' rare words with the
        sections of `deliverables` whose chunks hold them.

        Words of the query filler and of generic drafting prose (`_DRAFT_FILLER`) are left
        out. The query ORs the `DRAFT_QUERY_TERMS` distinct input words held by the fewest
        chunks, words in no chunk left out. A rare word has four or more characters, is not a
        number, and is held by at least one and at most `RARE_TERM_DOCUMENTS` version families
        of `deliverables`, so drafts of one deliverable count once and reviews and curator
        notes do not count.
        """
        words = list(
            dict.fromkeys(
                w
                for w in _DRAFT_WORD.findall(_nfkc(project_inputs).lower())
                if w not in _DRAFT_STOPWORDS and len(w) >= 2
            )
        )[:DRAFT_INPUT_WORDS]
        counts = {
            w: self._rows(
                "SELECT count(*) AS n FROM chunks_fts WHERE chunks_fts MATCH ?", (f'"{w}"',)
            )[0]["n"]
            for w in words
        }
        present = sorted((w for w in words if counts[w]), key=lambda w: counts[w])
        chosen = present[:DRAFT_QUERY_TERMS]
        query = " OR ".join(f'"{w}"' for w in chosen) or None
        family_of = {d.doc_id: d.family_id for d in deliverables}
        rare: dict[str, set[str]] = {}
        # Words in the fewest chunks first; only the first few are looked up by document.
        for w in [w for w in present if len(w) >= 4 and not w.isdigit()][: 3 * MAX_RARE_TERMS]:
            if len(rare) >= MAX_RARE_TERMS:
                break
            holders = [
                r
                for r in self._rows(
                    "SELECT DISTINCT c.doc_id, c.section_id FROM chunks_fts"
                    " JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id WHERE chunks_fts MATCH ?",
                    (f'"{w}"',),
                )
                if r["doc_id"] in family_of
            ]
            if 1 <= len({family_of[r["doc_id"]] for r in holders}) <= RARE_TERM_DOCUMENTS:
                rare[w] = {r["section_id"] for r in holders}
        return query, rare

    def _resolve_type(self, deliverable_type: str | None) -> str | None:
        if deliverable_type is None or not deliverable_type.strip():
            return None
        found = self._types_by_fold.get(fold_facet_label(deliverable_type))
        if found is None:
            valid = choices(
                f"{t.deliverable_type} ({t.document_count})" for t in self.type_summaries()
            )
            raise NotFoundError(
                f"unknown deliverable type {echoed(deliverable_type)}; valid types:"
                f" {valid or 'none'}"
            )
        return found

    def _counted_holders(self, cluster_id: int, own_doc: str) -> frozenset[str]:
        """The documents holding a recurring unit as `FamilyRecurrence` counts them: its
        holders that are not superseded, and the exemplar itself."""
        return self._cluster_holders[cluster_id] | {own_doc}

    def _recurrence(
        self, section_id: str, exemplar_ids: Sequence[str], family_total: int
    ) -> FamilyRecurrence | None:
        """`FamilyRecurrence` of one exemplar section: its units held by another family."""
        own_doc = self._sections[section_id].doc_id
        own = self._documents[own_doc].family_id
        recurring: list[tuple[frozenset[str], set[str]]] = []
        for cluster_id in self._section_clusters.get(section_id, ()):
            docs = self._counted_holders(cluster_id, own_doc)
            families = {self._documents[d].family_id for d in docs}
            if families - {own}:
                recurring.append((docs, families))
        if not recurring:
            return None
        holders = set().union(*(docs for docs, _ in recurring))
        dates = [d.iso for h in holders if (d := self._documents[h].date) is not None]
        return FamilyRecurrence(
            recurring_unit_count=len(recurring),
            exemplar_doc_ids=[e for e in exemplar_ids if e in holders],
            corpus_family_count=max(len(families) for _, families in recurring),
            corpus_family_total=family_total,
            latest_holder_date=max(dates) if dates else None,
        )

    def _relevance(
        self,
        docs: Sequence[DocumentInfo],
        fts_query: str | None,
        vector: NDArray[np.float32] | None,
        input_words: set[str],
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Each of `docs` ranked among them: its relevance (the fused-rank scores of its best
        `RELEVANCE_CHUNKS` chunks, plus the metadata bonus), and each of their sections' best
        chunk position in that ranking."""
        ranking = self._rank(fts_query, vector, frozenset(d.doc_id for d in docs))
        chunk_ranks: dict[str, list[int]] = {}
        section_rank: dict[str, int] = {}
        for position, index in enumerate(ranking.order, start=1):
            hit = self._hits[index]
            chunk_ranks.setdefault(hit["doc_id"], []).append(position)
            section_rank.setdefault(hit["section_id"], position)
        scores: dict[str, float] = {}
        for doc in docs:
            score = sum(
                1.0 / (RRF_K + r) for r in chunk_ranks.get(doc.doc_id, [])[:RELEVANCE_CHUNKS]
            )
            label = f"{doc.title} {doc.deliverable_type} {doc.summary}"
            meta = set(_DRAFT_WORD.findall(_nfkc(label).lower())) - _STOPWORDS
            if meta:
                score += METADATA_BONUS * len(meta & input_words) / len(meta) / (RRF_K + 1)
            scores[doc.doc_id] = score
        return scores, section_rank

    def prepare_draft(
        self,
        project_inputs: str,
        embed_query: QueryEmbedder | None,
        *,
        deliverable_type: str | None = None,
        filters: Mapping[str, str] | None = None,
    ) -> DraftPacket:
        """The packet an agent drafts one deliverable from, within one tool result.

        Exemplars are ranked against `project_inputs` with hybrid search, or BM25 alone when
        `embed_query` is None; no model writes anything. The pool is every citable document of
        the active pool (`_active_pool` of the documents passing `filters`) that is not
        feedback; a feedback document of the requested type stays. A document's relevance sums
        the fused-rank scores of its best `RELEVANCE_CHUNKS` chunks, plus a small bonus for
        input words in its title, type and summary (`_relevance`). With a requested type, its
        documents are ranked among themselves, so many documents of other types cannot crowd
        them out of the ranking channels; they fill the exemplar slots first. A free slot then
        takes a relevant document of another type (relevance above zero, ranked over the whole
        pool) when at most one exemplar is of the requested type, or when it reaches
        `CROSS_TYPE_RELEVANCE` of the best exemplar's relevance. An unordered version family
        fills one slot, with its other members named.

        Lessons are the house rules for the requested type, or for the first exemplar's type
        when none is requested, from every feedback document of the topic whatever the
        filters. Section text is packed whole, never cut (`_pack`). Blank inputs or invalid
        filters raise `InvalidFilterError` and an unknown type `NotFoundError`, before any
        embedding call.
        """
        if not project_inputs.strip():
            raise InvalidFilterError("project_inputs must not be blank; pass the user's inputs")
        allowed = self.resolve_filters(filters)
        requested = self._resolve_type(deliverable_type)
        active = self._active_pool(allowed)
        pool = [
            d
            for d in self._documents.values()
            if d.doc_id in active
            and not d.is_curator_note
            and not (d.is_feedback and d.deliverable_type != requested)
        ]
        fts_query, rare_words = self._draft_terms(project_inputs, self._deliverables(active))
        vector = (
            None
            if embed_query is None
            else self._query_vector(project_inputs[:DRAFT_EMBED_CHARS], embed_query)
        )
        words = set(_DRAFT_WORD.findall(_nfkc(project_inputs).lower())) - _STOPWORDS
        scores, section_rank = self._relevance(pool, fts_query, vector, words)
        if requested is not None:
            typed = [d for d in pool if d.deliverable_type == requested]
            typed_ids = {d.doc_id for d in typed}
            typed_scores, typed_sections = self._relevance(typed, fts_query, vector, words)
            scores |= typed_scores
            section_rank = {
                sid: rank
                for sid, rank in section_rank.items()
                if self._sections[sid].doc_id not in typed_ids
            } | typed_sections
        ranked = sorted(pool, key=lambda d: (-scores[d.doc_id], self._ordinal[d.doc_id]))
        exemplars: list[DocumentInfo] = []
        taken_families: set[str] = set()
        for doc in ranked:
            if len(exemplars) == MAX_PACKET_EXEMPLARS:
                break
            if doc.family_id not in taken_families and requested in (None, doc.deliverable_type):
                exemplars.append(doc)
                taken_families.add(doc.family_id)
        if requested is not None:
            same_type = len(exemplars)
            top = scores[exemplars[0].doc_id] if exemplars else 0.0
            for doc in ranked:
                if len(exemplars) == MAX_PACKET_EXEMPLARS:
                    break
                score = scores[doc.doc_id]
                if doc.family_id in taken_families or doc.is_feedback or score <= 0:
                    continue
                if same_type <= 1 or score >= CROSS_TYPE_RELEVANCE * top:
                    exemplars.append(doc)
                    taken_families.add(doc.family_id)
        others = [d for d in ranked if d.family_id not in taken_families][:MAX_OTHER_CANDIDATES]
        lesson_type = requested or (exemplars[0].deliverable_type if exemplars else None)
        lessons = [lesson for lesson in self.feedback_lessons if lesson_type in lesson.target_types]
        return self._pack(
            requested,
            exemplars,
            others,
            scores,
            section_rank,
            self._supplemental(exemplars, requested, rare_words, active),
            lessons,
        )

    def _candidate(self, doc: DocumentInfo, scores: Mapping[str, float]) -> DraftCandidate:
        return DraftCandidate(
            doc_id=doc.doc_id,
            title=_clipped(doc.title, TITLE_CHARS),
            deliverable_type=doc.deliverable_type,
            date=doc.date,
            page_count=doc.page_count,
            summary=doc.summary[:OTHER_CANDIDATE_SUMMARY_CHARS],
            relevance_score=scores[doc.doc_id],
        )

    def _pack(
        self,
        requested: str | None,
        exemplars: list[DocumentInfo],
        others: list[DocumentInfo],
        scores: Mapping[str, float],
        section_rank: Mapping[str, int],
        supplements: list[_SectionRow],
        all_lessons: list[FeedbackLesson],
    ) -> DraftPacket:
        """Pack the chosen documents into one tool result.

        The packet is serialized a handful of times, never once per section: the envelope
        and the outlines are measured as served, and every later addition is priced by the
        exact bytes it adds (its JSON value and separator), within
        `RESPONSE_MAX_BYTES - PACKET_MARGIN_BYTES`. The finished packet is measured once; if
        it does not fit, the last additions are taken back until it does.

        1. The envelope, without outlines, notes or lessons, fits first: the last exemplars
           move to the other candidates (the first stays), then other candidates are
           dropped from the end. At most `MAX_OTHER_CANDIDATES` candidates are kept, moved
           exemplars first.
        2. Curator notes and lessons come in, each within its share of the packet.
        3. Outlines: while the exemplars' complete outlines do not fit, the last exemplar
           moves to the candidates; a first exemplar left alone keeps as many outline entries
           as fit, counting the rest in `omitted_outline_sections`. So an exemplar after the
           first always has its complete outline, and the first's is cut only when it is the
           only exemplar.
        4. Whole sections, each added only while the packet fits, a skipped one leaving room
           for later, smaller ones: the first exemplar's input-matched, recurring and first
           and last two sections; the rare-word supplemental sections; the rest of the first
           exemplar; then the other exemplars' input-matched sections, their recurring
           sections that show a unit no included section shows yet, their boundary sections,
           and the rest of them. Each included section adds its sample of each recurring
           prose unit it is the first to show (`ClusterHolder`), at most
           `MAX_RECURRING_SAMPLES`.

        Summaries are cut to `SUMMARY_CHARS` and family members listed up to
        `MAX_FAMILY_DOC_IDS`.
        """
        family_total = len(
            {
                d.family_id
                for d in self._documents.values()
                if not d.is_curator_note and not d.is_feedback
            }
        )
        all_notes = self.curator_notes()
        budget = RESPONSE_MAX_BYTES - PACKET_MARGIN_BYTES
        chosen = list(exemplars)
        demoted: list[DocumentInfo] = []
        candidates = [self._candidate(d, scores) for d in others][:MAX_OTHER_CANDIDATES]
        recurrence: dict[str, FamilyRecurrence | None] = {}
        entry_bytes: dict[str, int] = {}
        # Outline entries shown per exemplar; an exemplar not listed shows its whole outline.
        outline: dict[str, int] = {}
        bodies: dict[str, str] = {}
        visuals: dict[str, int] = {}
        included: dict[str, str] = {}
        # Recurring prose unit shown by an included section, with its opening words.
        samples: dict[int, str] = {}
        supplemental: list[SupplementalSection] = []
        lessons: list[FeedbackLesson] = []
        notes: list[CuratorNote] = []

        def entry(s: _SectionRow) -> ExemplarSection:
            return ExemplarSection(
                section_id=s.section_id,
                ordinal=s.ordinal,
                heading_path=heading_context(s.heading_path),
                level=s.level,
                page_start=s.page_start,
                page_end=s.page_end,
                char_count=len(s.markdown),
                visual_count=visuals[s.section_id],
                included=s.section_id in included,
                family_recurrence=recurrence[s.section_id],
                markdown=included.get(s.section_id),
            )

        def recount() -> None:
            """Recurrence and the bytes of each outline entry as not included, for the
            current exemplars: `exemplar_doc_ids` name only them."""
            ids = [d.doc_id for d in chosen]
            recurrence.clear()
            entry_bytes.clear()
            for doc in chosen:
                for s in self._doc_sections[doc.doc_id]:
                    if s.section_id not in bodies:
                        bodies[s.section_id] = without_visual_transcripts(s.markdown).strip()
                        visuals[s.section_id] = _visual_count(s.markdown)
                    recurrence[s.section_id] = self._recurrence(s.section_id, ids, family_total)
                    entry_bytes[s.section_id] = _json_bytes(entry(s)) + 1

        def exemplar(doc: DocumentInfo) -> DraftExemplar:
            sections = self._doc_sections[doc.doc_id]
            keep = outline.get(doc.doc_id, len(sections))
            family = [m for m in self._families[doc.family_id] if m != doc.doc_id]
            shown_family = family[:MAX_FAMILY_DOC_IDS]
            return DraftExemplar(
                doc_id=doc.doc_id,
                rel_path=doc.rel_path,
                title=_clipped(doc.title, TITLE_CHARS),
                doc_date=doc.doc_date,
                date=doc.date,
                deliverable_type=doc.deliverable_type,
                facets=doc.facets,
                page_count=doc.page_count,
                summary=_clipped(doc.summary, SUMMARY_CHARS),
                relevance_score=scores[doc.doc_id],
                version_status=doc.version_status,
                family_doc_ids=shown_family,
                omitted_family_members=len(family) - len(shown_family),
                sections=[entry(s) for s in sections[:keep]],
                omitted_outline_sections=len(sections) - keep,
            )

        def packet() -> DraftPacket:
            sections = [s for d in chosen for s in self._doc_sections[d.doc_id]]
            return DraftPacket(
                topic=self.topic,
                deliverable_type=requested,
                drafting_guidance=DRAFTING_GUIDANCE,
                lessons=list(lessons),
                omitted_lessons=len(all_lessons) - len(lessons),
                curator_notes=list(notes),
                omitted_curator_notes=len(all_notes) - len(notes),
                recurring_samples=list(samples.values()),
                exemplars=[exemplar(d) for d in chosen],
                supplemental_sections=list(supplemental),
                other_candidates=list(candidates),
                omitted_sections=sum(s.section_id not in included for s in sections),
            )

        def measured() -> int:
            return len(response_text(packet()).encode())

        def demote() -> None:
            demoted.insert(0, chosen.pop())
            candidates[:] = [self._candidate(d, scores) for d in [*demoted, *others]][
                :MAX_OTHER_CANDIDATES
            ]
            outline.pop(demoted[0].doc_id, None)
            recount()

        def outline_bytes(doc: DocumentInfo) -> int:
            return sum(entry_bytes[s.section_id] for s in self._doc_sections[doc.doc_id])

        # 1. The envelope.
        recount()
        outline.update({d.doc_id: 0 for d in chosen})
        while len(chosen) > 1 and measured() > budget:
            demote()
        while candidates and measured() > budget:
            candidates.pop()
        size = measured()
        if size > budget:
            raise SnapshotIntegrityError(
                f"{self.path}: the envelope of a drafting packet does not fit one tool result"
            )

        # 2. Notes and lessons, each within its share.
        def admit(items: Sequence[BaseModel], into: list[Any], share: int) -> None:
            nonlocal size
            spent = 0
            for item in items:
                cost = _json_bytes(item) + 1
                if spent + cost <= share and size + cost <= budget:
                    into.append(item)
                    spent += cost
                    size += cost

        admit(all_notes, notes, PACKET_NOTES_BYTES)
        admit(all_lessons, lessons, PACKET_LESSONS_BYTES)
        # 3. Outlines, the last exemplar giving way first.
        while len(chosen) > 1 and size + sum(map(outline_bytes, chosen)) > budget:
            demote()
            size = measured()
        if chosen:
            first_sections = self._doc_sections[chosen[0].doc_id]
            room = budget - size - sum(outline_bytes(d) for d in chosen[1:])
            keep = 0
            for s in first_sections:
                if entry_bytes[s.section_id] > room:
                    break
                room -= entry_bytes[s.section_id]
                keep += 1
            outline.clear()
            if keep < len(first_sections):
                outline[chosen[0].doc_id] = keep
        size = measured()
        # 4. Whole sections, priced by the bytes each adds.
        first = chosen[0] if chosen else None
        shown_units: set[int] = set()
        taken_back: list[Callable[[], None]] = []

        def outlined(doc: DocumentInfo) -> list[_SectionRow]:
            sections = self._doc_sections[doc.doc_id]
            return sections[: outline.get(doc.doc_id, len(sections))]

        def nonblank(doc: DocumentInfo) -> list[_SectionRow]:
            return [s for s in outlined(doc) if bodies[s.section_id]]

        def boundary(doc: DocumentInfo) -> list[_SectionRow]:
            sections = nonblank(doc)
            return [*sections[:BOUNDARY_SECTIONS], *sections[-BOUNDARY_SECTIONS:]]

        def matched(doc: DocumentInfo) -> list[_SectionRow]:
            hits = [
                s
                for s in nonblank(doc)
                if section_rank.get(s.section_id, MATCHED_CHUNK_RANKS + 1) <= MATCHED_CHUNK_RANKS
            ]
            return sorted(hits, key=lambda s: section_rank[s.section_id])[:MATCHED_SECTIONS]

        def recurring(doc: DocumentInfo) -> list[_SectionRow]:
            return [s for s in nonblank(doc) if recurrence[s.section_id] is not None]

        def shows_new_unit(section: _SectionRow) -> bool:
            return bool(set(self._section_clusters.get(section.section_id, ())) - shown_units)

        def new_samples(section: _SectionRow) -> dict[int, str]:
            found: dict[int, str] = {}
            for cluster_id in self._section_clusters.get(section.section_id, ()):
                if (
                    len(samples) + len(found) >= MAX_RECURRING_SAMPLES
                    or cluster_id in samples
                    or cluster_id not in self._prose_clusters
                ):
                    continue
                holders = self._counted_holders(cluster_id, section.doc_id)
                if len({self._documents[d].family_id for d in holders}) >= 2:
                    found[cluster_id] = self._cluster_samples[cluster_id, section.section_id]
            return found

        def include(section: _SectionRow) -> None:
            nonlocal size
            sid = section.section_id
            if sid in included or sid not in bodies:
                return
            body = bodies[sid]
            added = new_samples(section)
            # `"included":false` becomes `true` and `"markdown":null` the text.
            cost = _json_bytes(body) - len("null") - 1
            cost += sum(_json_bytes(t) + 1 for t in added.values())
            if size + cost > budget:
                return
            size += cost
            included[sid] = body
            samples.update(added)
            units = set(self._section_clusters.get(sid, ()))
            fresh = units - shown_units
            shown_units.update(fresh)

            def take_back() -> None:
                del included[sid]
                for cluster_id in added:
                    del samples[cluster_id]
                shown_units.difference_update(fresh)

            taken_back.append(take_back)

        def supplement(row: _SectionRow) -> None:
            nonlocal size
            item = SupplementalSection(
                doc_id=row.doc_id,
                doc_title=_clipped(self._documents[row.doc_id].title, TITLE_CHARS),
                date=self._documents[row.doc_id].date,
                section_id=row.section_id,
                heading_path=heading_context(row.heading_path),
                markdown=without_visual_transcripts(row.markdown).strip(),
            )
            cost = _json_bytes(item) + 1
            if size + cost > budget:
                return
            size += cost
            supplemental.append(item)
            taken_back.append(lambda: supplemental.remove(item))

        rest = chosen[1:]
        if first is not None:
            for s in [*matched(first), *recurring(first), *boundary(first)]:
                include(s)
        for row in supplements:
            supplement(row)
        if first is not None:
            for s in outlined(first):
                include(s)
        for doc in rest:
            for s in matched(doc):
                include(s)
        for doc in rest:
            for s in recurring(doc):
                if shows_new_unit(s):
                    include(s)
        for doc in rest:
            for s in boundary(doc):
                include(s)
        for doc in rest:
            for s in outlined(doc):
                include(s)
        result = packet()
        while not fits_response(result):
            if not taken_back:
                raise SnapshotIntegrityError(
                    f"{self.path}: a drafting packet does not fit one tool result"
                )
            taken_back.pop()()
            result = packet()
        return result

    def _supplemental(
        self,
        chosen: Sequence[DocumentInfo],
        requested: str | None,
        rare_words: Mapping[str, set[str]],
        active: frozenset[str],
    ) -> list[_SectionRow]:
        """Sections of deliverables of the active pool outside the exemplars' families that
        hold the most rare input words: of the requested type's documents when a type is
        requested, else of any deliverable. At most `MAX_SUPPLEMENTAL_SECTIONS`, best first,
        ties in path order."""
        families = {d.family_id for d in chosen}
        held = Counter(section for sections in rare_words.values() for section in sections)
        eligible = []
        for section_id, found in held.items():
            row = self._sections[section_id]
            doc = self._documents[row.doc_id]
            if (
                doc.is_curator_note
                or doc.is_feedback
                or doc.family_id in families
                or doc.doc_id not in active
                or (requested is not None and doc.deliverable_type != requested)
            ):
                continue
            eligible.append((-found, self._ordinal[doc.doc_id], row.ordinal, row))
        return [
            row for *_, row in sorted(eligible, key=lambda t: t[:3])[:MAX_SUPPLEMENTAL_SECTIONS]
        ]


def _with_citations(entry: GlossaryEntry, most: int) -> GlossaryEntry:
    """`entry` with at most `most` citations per sense, the rest counted per sense."""
    return entry.model_copy(
        update={
            "senses": tuple(
                s.model_copy(
                    update={
                        "citations": s.citations[:most],
                        "omitted_citations": max(0, len(s.citations) - most),
                    }
                )
                for s in entry.senses
            )
        }
    )
