"""Domain types, the shared text folding, and the error hierarchy."""

from __future__ import annotations

import datetime
import hashlib
import io
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ArkiveError(Exception):
    """Base class for every expected Arkive failure."""


class UnsupportedFormatError(ArkiveError):
    """A corpus file has a format the compiler does not accept."""


class MissingDependencyError(ArkiveError):
    """A native tool the compiler needs is not installed."""


class ConversionError(ArkiveError):
    """A source document could not be converted or rendered."""


class UnsafePathError(ConversionError):
    """A corpus entry is a symlink or escapes the corpus, or build outputs overlap it."""


class ProviderError(ArkiveError):
    """The model provider failed, refused, or returned unusable output."""


class TranscriptionFidelityError(ArkiveError):
    """A page transcription lost numbers or filenames present in the native text."""


class RejectedOutputError(ArkiveError):
    """Generated output still broke its structural rules after one rerun naming them.

    Examples: a citation that resolves to no citable section, an unknown folder or document
    ID, a facet that mixes path-bound and content-bound values, a document left
    unclassified, or an observation batch whose quotes could not be located at all.
    """


class SnapshotIntegrityError(ArkiveError):
    """A snapshot file is missing, malformed, or from another schema version."""


class InvalidFilterError(ArkiveError):
    """A query names a facet, a facet value, or a limit the snapshot cannot satisfy."""


class NotFoundError(ArkiveError):
    """A requested topic, document, section, or page does not exist."""


# An error message reaches an MCP client as one tool result, so it echoes at most this much
# of the caller's input and names at most this many valid choices.
ERROR_ECHO_CHARS = 200
ERROR_MAX_CHOICES = 40


def echoed(text: str) -> str:
    """The caller's input as an error message quotes it: `repr`, cut to `ERROR_ECHO_CHARS`
    characters of the input with an ellipsis."""
    if len(text) <= ERROR_ECHO_CHARS:
        return repr(text)
    return f"{text[:ERROR_ECHO_CHARS]!r}\u2026 ({len(text)} characters)"


def choices(names: Iterable[str]) -> str:
    """Valid choices as an error message lists them: the first `ERROR_MAX_CHOICES`, each on
    one line (whitespace runs, newlines included, become one space) and cut like `echoed`
    input, then a count of the rest."""
    every = list(names)
    lines = (" ".join(n.split()) for n in every[:ERROR_MAX_CHOICES])
    shown = ", ".join(
        n if len(n) <= ERROR_ECHO_CHARS else f"{n[:ERROR_ECHO_CHARS]}\u2026" for n in lines
    )
    rest = len(every) - ERROR_MAX_CHOICES
    return f"{shown}, ... (+{rest} more)" if rest > 0 else shown


SourceFormat = Literal["pdf", "docx", "pptx", "md", "xlsx", "html"]
# Formats read page by page from rendered pages; every other format is read from its own text
# and has no pages.
PAGED_FORMATS: frozenset[SourceFormat] = frozenset({"pdf", "docx", "pptx"})
BlockKind = Literal["heading", "paragraph", "table", "callout"]
ChunkKind = Literal["prose", "table", "visual", "guidance", "annotation"]
# A path-bound facet's values stand for corpus folders; a content-bound facet's values are
# assigned by reading each document.
FacetBinding = Literal["path", "content"]


# Text folding.

_DASHES = re.compile(r"[\u2010-\u2015\u2212]")
_LINE_HYPHENATION = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
_BLOCK_QUOTE_MARKERS = re.compile(r"^[ \t]*(?:>[ \t]?)+", re.MULTILINE)
# An asterisk run is emphasis unless whitespace (or the text edge) is on both sides, as in a
# literal `2 * 3`. An underscore run is emphasis only at a word edge, so `harbour_dry` keeps
# its underscore.
_EMPHASIS = re.compile(r"`|\*+(?=\S)|(?<=\S)\*+|(?<![^\W_])_+(?=[^\s_])|(?<=[^\s_])_+(?![^\W_])")
_WHITESPACE = re.compile(r"\s+")


def fold_text(text: str) -> str:
    """The comparison form of text: NFKC, Unicode dashes as `-`, and case-folded.

    `TopicSnapshot.grep`, the build-time `grep` tool and quote location all compare text in
    this form, so `2016-2017` finds `2016\u20132017` and `km2` finds `km\u00b2`.
    """
    return _DASHES.sub("-", unicodedata.normalize("NFKC", text)).casefold()


def fold_facet_label(text: str) -> str:
    """The identity of a facet value or alias: `fold_text` without surrounding whitespace.

    Organisation checks, the snapshot's alias table and the filter resolver all compare facet
    values and aliases in this form, so a label the build accepts is a label a filter finds.
    """
    return fold_text(text).strip()


def fold_for_quote(text: str) -> str:
    """`fold_text` with Markdown markup ignored, for locating a model's quote in source text.

    Unicode dashes become `-` first, so line-break hyphenation with any dash is joined; then
    block-quote markers, `*` and `_` emphasis, backticks and table pipes are dropped, and
    whitespace runs become one space, so a quote of a table row or of a visual transcript
    matches the Markdown it came from.
    """
    text = _DASHES.sub("-", unicodedata.normalize("NFKC", text))
    text = _LINE_HYPHENATION.sub("", text)
    text = _EMPHASIS.sub("", _BLOCK_QUOTE_MARKERS.sub("", text)).replace("|", " ")
    return _WHITESPACE.sub(" ", fold_text(text)).strip()


# Longest heading path shown as a label beside section text: in an observation batch, a chunk
# breadcrumb or any induction prompt. A heading the transcript ran on for pages keeps its whole
# Markdown in the section; only the label is cut, so it cannot blow a prompt or an embedding.
MAX_HEADING_CONTEXT_CHARS = 500


def heading_context(heading_path: str) -> str:
    """`heading_path` as a label beside section text: itself, or its first
    MAX_HEADING_CONTEXT_CHARS - 1 characters and an ellipsis."""
    if len(heading_path) <= MAX_HEADING_CONTEXT_CHARS:
        return heading_path
    return heading_path[: MAX_HEADING_CONTEXT_CHARS - 1] + "\u2026"


def document_id(rel_path: str) -> str:
    """The stable ID of the document at `rel_path`, a corpus-relative POSIX path.

    It depends on the NFC-normalised path only, never on the bytes or on the other files of
    the corpus, so adding, removing or editing another file leaves every other ID unchanged.
    """
    norm = unicodedata.normalize("NFC", rel_path)
    return f"doc_{hashlib.sha256(norm.encode('utf-8')).hexdigest()[:10]}"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# Provider structured-output schemas. OpenAI strict JSON schema needs every field typed
# explicitly, so these models avoid dict fields and defaults.


class VisualTranscript(_Frozen):
    label: str
    transcript: str


class PageBlock(_Frozen):
    """`heading_level` matters only for headings; the compiler reads 0 there as level 2."""

    kind: BlockKind
    heading_level: Literal[0, 1, 2, 3, 4]
    continues_from_previous_page: bool
    merges_into_previous_row: bool
    markdown: str


class PageExtraction(_Frozen):
    blocks: list[PageBlock]
    visuals: list[VisualTranscript]


class StatedAttribute(_Frozen):
    """A name/value pair the document states about itself, such as client or audience, with
    `quote`, the verbatim source text that states it."""

    name: str
    value: str
    quote: str


class TermDraft(_Frozen):
    term: str
    kind: str
    aliases: list[str]
    quote: str
    defines: bool


class ObservationDraft(_Frozen):
    """What one batch of a document's sections shows. Quotes are verbatim source text."""

    title: str
    doc_date: str | None
    deliverable_type: str
    summary: str
    attributes: list[StatedAttribute]
    terms: list[TermDraft]


class CardHeader(_Frozen):
    title: str
    doc_date: str | None
    deliverable_type: str
    summary: str


class FacetValueDraft(_Frozen):
    value: str
    description: str
    aliases: list[str]
    folders: list[str]


class FacetDraft(_Frozen):
    name: str
    description: str
    values: list[FacetValueDraft]


class DocumentRoleDraft(_Frozen):
    """How one batch reads one document: its deliverable type, and the types of deliverable
    it reviews or records lessons for when it is feedback."""

    doc_id: str
    deliverable_type: str
    is_feedback: bool
    feedback_for_types: list[str]


class OrganisationDraft(_Frozen):
    """The organisation one batch of document digests proposes."""

    facets: list[FacetDraft]
    documents: list[DocumentRoleDraft]


class CanonicalTypeDraft(_Frozen):
    name: str
    description: str
    labels: list[str]


class TypeCanonicalisationDraft(_Frozen):
    """Every proposed deliverable type label mapped to one canonical type."""

    types: list[CanonicalTypeDraft]


class FeedbackLessonDraft(_Frozen):
    statement: str
    quote: str


class FeedbackLessonsDraft(_Frozen):
    lessons: list[FeedbackLessonDraft]


class SenseDraft(_Frozen):
    definition: str | None
    citations: list[str]


class GlossaryEntryDraft(_Frozen):
    term: str
    kind: str
    aliases: list[str]
    senses: list[SenseDraft]


class GlossaryDraft(_Frozen):
    entries: list[GlossaryEntryDraft]


class GuideDraft(_Frozen):
    guide_markdown: str


# Reading: what one document says, addressed by its bytes. It holds no corpus IDs and no
# path, so it stays valid when files are added, removed or moved.


class TermObservation(_Frozen):
    term: str
    kind: str
    aliases: tuple[str, ...]
    quote: str
    defines: bool
    # Document-local ordinal of the first section containing the quote.
    section_ordinal: int


class LocatedAttribute(_Frozen):
    """A stated attribute whose quote was found in the document's text, outside its asides."""

    name: str
    value: str
    # Document-local ordinal of the first section containing the quote.
    section_ordinal: int


class ObservationCard(_Frozen):
    title: str
    doc_date: str | None
    deliverable_type: str
    summary: str
    attributes: tuple[LocatedAttribute, ...]
    terms: tuple[TermObservation, ...]


class ChunkPiece(_Frozen):
    kind: ChunkKind
    content: str
    page_start: int
    page_end: int


class ReadSection(_Frozen):
    """One section of a read document. `continues_table`: its first table continues the
    last table of the sections of its heading path before it, a table cut between rows."""

    ordinal: int
    heading_path: str
    level: int
    page_start: int
    page_end: int
    markdown: str
    pieces: tuple[ChunkPiece, ...]
    continues_table: bool = False


class RenderedPage(_Frozen):
    page_number: int
    native_text: str
    webp_bytes: bytes
    visual_summary: str


# `printed`: parsed from the date the document prints. `metadata`: from the original file's
# own metadata, used only when the document prints no date.
DateBasis = Literal["printed", "metadata"]


_CALENDAR_DATE = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?", re.ASCII)


def is_calendar_date(iso: str) -> bool:
    """Whether `iso` is a real calendar date written `YYYY-MM-DD`, or a month `YYYY-MM`."""
    match = _CALENDAR_DATE.fullmatch(iso)
    if match is None:
        return False
    try:
        datetime.date(int(match[1]), int(match[2]), int(match[3] or 1))
    except ValueError:
        return False
    return True


_MONTH_NAMES = (
    *("january", "february", "march", "april", "may", "june"),
    *("july", "august", "september", "october", "november", "december"),
)
_MONTH_NUMBERS = (
    {name: number for number, name in enumerate(_MONTH_NAMES, start=1)}
    | {name[:3]: number for number, name in enumerate(_MONTH_NAMES, start=1)}
    | {"sept": 9}
)
_MONTH = "(" + "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
# re.ASCII keeps every pattern to ASCII digits and makes IGNORECASE fold ASCII letters only,
# so a matched month name is a key of _MONTH_NUMBERS once lowered.
_ISO_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})", re.ASCII)
_ISO_MONTH = re.compile(r"(\d{4})-(\d{2})", re.ASCII)
_DAY_MONTH_YEAR = re.compile(rf"{_DAY} (?:of )?{_MONTH},? (\d{{4}})", re.ASCII | re.IGNORECASE)
_MONTH_DAY_YEAR = re.compile(rf"{_MONTH} {_DAY},? (\d{{4}})", re.ASCII | re.IGNORECASE)
_MONTH_YEAR = re.compile(rf"{_MONTH},? (\d{{4}})", re.ASCII | re.IGNORECASE)
_NUMERIC_DATE = re.compile(r"(\d{1,2})([/.-])(\d{1,2})\2(\d{4})", re.ASCII)
_YEAR_FIRST_DATE = re.compile(r"(\d{4})([/.])(\d{1,2})\2(\d{1,2})", re.ASCII)


def iso_date(year: str | int, month: str | int, day: str | int | None = None) -> str | None:
    """`YYYY-MM-DD`, or `YYYY-MM` without a day, when the parts name a real date."""
    try:
        parsed = datetime.date(int(year), int(month), 1 if day is None else int(day))
    except ValueError:
        return None
    return parsed.isoformat() if day is not None else parsed.isoformat()[:7]


def parse_printed_date(printed: str) -> str | None:
    """The ISO form of a printed date that has one reading, else None.

    After NFKC (so `19ᵗʰ` reads as `19th`) and with whitespace collapsed, the whole text must
    be one of: `YYYY-MM-DD` or `YYYY-MM`; a day (with or without an ordinal suffix), an
    English month name (full or abbreviated) and a year; a month name, day and year; a month
    name and year, which gives `YYYY-MM`; a numeric `D/M/YYYY` (also with `-` or `.`) where
    exactly one of the first two numbers is above 12, which makes it the day, or where both
    are the same, so either order reads alike (`01/01/2026`); or a year-first `YYYY/M/D` or
    `YYYY.M.D` with a year from 1900 to 2100. Anything else, such as `05/04/2026` or a
    range, has no single reading and gives None. A Unicode dash or minus sign reads as `-`,
    as `fold_text` reads it, so `2025\u201305\u201319` is `2025-05-19`.
    """
    text = " ".join(_DASHES.sub("-", unicodedata.normalize("NFKC", printed)).split())
    if match := _ISO_DAY.fullmatch(text):
        return iso_date(match[1], match[2], match[3])
    if match := _ISO_MONTH.fullmatch(text):
        return iso_date(match[1], match[2])
    if match := _DAY_MONTH_YEAR.fullmatch(text):
        return iso_date(match[3], _MONTH_NUMBERS[match[2].lower()], match[1])
    if match := _MONTH_DAY_YEAR.fullmatch(text):
        return iso_date(match[3], _MONTH_NUMBERS[match[1].lower()], match[2])
    if match := _MONTH_YEAR.fullmatch(text):
        return iso_date(match[2], _MONTH_NUMBERS[match[1].lower()])
    if match := _NUMERIC_DATE.fullmatch(text):
        first, second = int(match[1]), int(match[3])
        if first == second:
            return iso_date(match[4], first, first)
        if (first > 12) == (second > 12):
            return None
        day, month = (first, second) if first > 12 else (second, first)
        return iso_date(match[4], month, day)
    if (match := _YEAR_FIRST_DATE.fullmatch(text)) and 1900 <= int(match[1]) <= 2100:
        return iso_date(match[1], match[3], match[4])
    return None


class DocumentDate(_Frozen):
    """A document's calendar date, `YYYY-MM-DD` or `YYYY-MM`, and where it came from."""

    iso: str
    basis: DateBasis

    @field_validator("iso")
    @classmethod
    def _real_date(cls, iso: str) -> str:
        if not is_calendar_date(iso):
            raise ValueError(f"{iso!r} is not a calendar date written YYYY-MM-DD or YYYY-MM")
        return iso


# Longest side of a rendered page. A larger page (an A0 sheet, a foldout) renders at a smaller
# scale: libwebp refuses images over 16,383 px, and a 2x A0 bitmap holds about 130 MB.
MAX_RENDER_SIDE_PX = 4096


def _decodes_as_webp(data: bytes) -> bool:
    """Whether `data` is a WebP image within MAX_RENDER_SIDE_PX whose pixels decode. The
    format and size come from the header, so an image that is not one is refused before its
    pixels are decoded."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "WEBP" or not (
                1 <= image.width <= MAX_RENDER_SIDE_PX and 1 <= image.height <= MAX_RENDER_SIDE_PX
            ):
                return False
            image.load()
            return True
    except OSError, ValueError, SyntaxError, Image.DecompressionBombError:
        return False


@dataclass(frozen=True)
class RenderedSource:
    """Every page of a rendered PDF, Word or PowerPoint source as (native text, WebP bytes),
    numbered from 1, and the date its original file's own metadata carries.

    `metadata_date` is `YYYY-MM-DD` or `YYYY-MM`, or None when the file carries no date. A
    rendered source has at least one page, and every image is a WebP image no side of which
    exceeds MAX_RENDER_SIDE_PX that Pillow decodes in full, so a truncated, damaged or
    oversized cache entry is refused.
    """

    pages: tuple[tuple[str, bytes], ...]
    metadata_date: str | None

    def __post_init__(self) -> None:
        if not self.pages:
            raise ValueError("a rendered source has at least one page")
        for number, page in enumerate(self.pages, start=1):
            text, webp = page
            if not isinstance(text, str) or not isinstance(webp, bytes):
                raise ValueError(f"page {number} is not (native text, WebP bytes)")
            if not _decodes_as_webp(webp):
                raise ValueError(f"page {number} image is not a WebP image that decodes")
        if self.metadata_date is not None and not is_calendar_date(self.metadata_date):
            raise ValueError(f"metadata date {self.metadata_date!r} is not a calendar date")


class DocumentReading(_Frozen):
    """The reading of one source. `discarded_quotes` counts observations dropped because
    their quote was still not found in the source after one rerun. `date` is None when the
    printed date is ambiguous, or when nothing is printed and the file carries no date."""

    sha256: str
    format: SourceFormat
    is_curator_note: bool
    sections: tuple[ReadSection, ...]
    pages: tuple[RenderedPage, ...]
    card: ObservationCard
    date: DocumentDate | None
    discarded_quotes: int

    @property
    def date_label(self) -> str | None:
        """The date as prompts show it: the date as printed, else the metadata date marked as
        such, else None."""
        if self.card.doc_date:
            return self.card.doc_date
        if self.date is not None:
            return f"{self.date.iso} (file metadata)"
        return None


@dataclass(frozen=True)
class CorpusDocument:
    """A reading placed in the corpus. `doc_id` is `document_id(rel_path)`; documents are
    ordered by path."""

    doc_id: str
    rel_path: str
    reading: DocumentReading

    def section_id(self, ordinal: int) -> str:
        return f"{self.doc_id}_sec_{ordinal:03d}"

    @property
    def is_citable(self) -> bool:
        return not self.reading.is_curator_note


# Corpus knowledge, from induction.


class FacetValue(_Frozen):
    value: str
    description: str
    aliases: tuple[str, ...]
    # Corpus folders the value stands for; empty for a content-bound facet.
    folders: tuple[str, ...]


class Facet(_Frozen):
    name: str
    description: str
    binding: FacetBinding
    values: tuple[FacetValue, ...]


class GlossarySense(_Frozen):
    """One meaning of a term. `definition` is None when the corpus uses the term in the cited
    sections but does not establish what it means there. `omitted_citations` counts citations
    a glossary lookup left out to stay within one tool result; the stored glossary keeps every
    citation, so it is 0 there."""

    definition: str | None
    citations: Annotated[tuple[str, ...], Field(min_length=1)]
    omitted_citations: int = 0


class GlossaryEntry(_Frozen):
    term: str
    kind: str
    aliases: tuple[str, ...]
    senses: Annotated[tuple[GlossarySense, ...], Field(min_length=1)]


# `sole`: the only member of its version family. In a larger family, `latest` supersedes every
# other member, which is `superseded`; `unordered` members could not be ordered without guessing.
VersionStatus = Literal["sole", "latest", "superseded", "unordered"]


class DocumentRole(_Frozen):
    """What a citable document is within its corpus.

    `deliverable_type` is canonical across the corpus. A feedback document keeps its own type
    and also names, in `feedback_for_types`, the types whose deliverables it reviews. A version
    family groups drafts, revisions and exports of one deliverable; `family_id` is the ID of
    its first member in path order, and `superseded_by` names the family's latest member (or,
    for a byte-identical copy, the copy it duplicates).
    """

    deliverable_type: str
    is_feedback: bool
    feedback_for_types: tuple[str, ...]
    family_id: str
    version_status: VersionStatus
    superseded_by: str | None


class DeliverableType(_Frozen):
    name: str
    description: str


class FeedbackLesson(_Frozen):
    """A transferable reviewer criticism or resolution from a feedback document, located in
    `section_id` and confirmed there, for deliverables of `target_types`."""

    statement: str
    source_doc_id: str
    section_id: str
    doc_title: str
    target_types: tuple[str, ...]


UnitKind = Literal["prose", "table_header"]


class ClusterHolder(_Frozen):
    """One section holding a recurring unit. For a prose unit, `sample` is the opening words
    of the unit as this section writes it (`passages.recurring_units` defines them); a table
    header row has no sample."""

    section_id: str
    sample: str | None


class UnitCluster(_Frozen):
    """A text unit that recurs across documents. `text` is the first holder's unit exactly as
    written; `holders` are every section that holds it, in path order, the first holder's
    section first among its document's. Every holder of a prose unit has a nonblank sample,
    and no holder of a table header row has one."""

    cluster_id: int
    kind: UnitKind
    text: str
    holders: tuple[ClusterHolder, ...]

    @model_validator(mode="after")
    def _samples_match_kind(self) -> Self:
        prose = self.kind == "prose"
        for h in self.holders:
            valid = bool(h.sample and h.sample.strip()) if prose else h.sample is None
            if not valid:
                rule = "needs a nonblank sample" if prose else "has no sample"
                raise ValueError(
                    f"recurring unit {self.cluster_id}: holder {h.section_id} of a {self.kind}"
                    f" unit {rule}"
                )
        return self

    @property
    def section_ids(self) -> tuple[str, ...]:
        return tuple(h.section_id for h in self.holders)


class BuildReport(_Frozen):
    """What the build reduced or skipped without failing.

    The first three counts are the contracted reductions: observations and feedback lessons
    discarded because their quote could not be located after one rerun, glossary definitions
    the support check did not confirm (stored as `definition=None`), and feedback lessons it
    did not confirm (dropped). `judge_refusals` counts decision questions the decision model
    refused, by operation; `pruned_facet_values` and `pruned_facets` name what binding left
    empty. `truncated_curator_notes` counts curator notes an induction prompt could not show
    in full; the snapshot stores every note verbatim.
    """

    discarded_unlocated_quotes: int
    unsupported_definitions: int
    dropped_feedback_lessons: int
    judge_refusals: dict[str, int]
    pruned_facet_values: list[str]
    pruned_facets: list[str]
    truncated_curator_notes: int


# Compiled corpus.


class CuratorNote(_Frozen):
    scope_path: str
    content: str


class Section(_Frozen):
    section_id: str
    ordinal: int
    heading_path: str
    level: int
    page_start: int
    page_end: int
    markdown: str


class Chunk(_Frozen):
    chunk_id: str
    section_id: str
    ordinal: int
    kind: ChunkKind
    page_start: int
    page_end: int
    breadcrumb: str
    content: str


class CompiledDocument(_Frozen):
    doc_id: str
    rel_path: str
    sha256: str
    format: SourceFormat
    title: str
    # The date as printed, or the metadata date's ISO form when nothing is printed.
    doc_date: str | None
    date: DocumentDate | None
    summary: str
    # Facet name to the document's sorted values, in facet definition order.
    facets: dict[str, tuple[str, ...]]
    sections: tuple[Section, ...]
    chunks: tuple[Chunk, ...]
    pages: tuple[RenderedPage, ...]
    is_curator_note: bool
    # A curator note's role is its card's type, no feedback and a family of its own.
    role: DocumentRole


@dataclass(frozen=True)
class CompiledCorpus:
    """A whole compiled topic. `embeddings` row i belongs to the i-th chunk in document order."""

    corpus_dir: Path
    topic: str
    title: str
    embedding_model: str
    embedding_dimensions: int
    curator_notes: tuple[CuratorNote, ...]
    guide_markdown: str
    facets: tuple[Facet, ...]
    glossary: tuple[GlossaryEntry, ...]
    deliverable_types: tuple[DeliverableType, ...]
    feedback_lessons: tuple[FeedbackLesson, ...]
    unit_clusters: tuple[UnitCluster, ...]
    build_settings: Mapping[str, Any]
    build_report: BuildReport
    documents: tuple[CompiledDocument, ...]
    embeddings: NDArray[np.float32]

    def all_chunks(self) -> list[Chunk]:
        return [chunk for doc in self.documents for chunk in doc.chunks]

    @property
    def corpus_fingerprint(self) -> str:
        return corpus_fingerprint((doc.rel_path, doc.sha256) for doc in self.documents)


def corpus_fingerprint(documents: Iterable[tuple[str, str]]) -> str:
    """Content address of a corpus: SHA-256 over sorted `rel_path NUL sha256` lines.

    It depends only on source paths and bytes, so the same corpus always has the same
    fingerprint, whatever models or settings built the snapshot.
    """
    lines = sorted(f"{rel_path}\0{sha256}" for rel_path, sha256 in documents)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# Read results.


class FacetSummary(_Frozen):
    """A facet and its first values; `omitted_values` counts the rest, which
    `get_topic_guide` and an unknown-value error also name."""

    name: str
    binding: FacetBinding
    values: list[str]
    omitted_values: int


class TopicSummary(_Frozen):
    """One served topic as `list_topics` shows it. Facet values and deliverable types are
    listed up to `snapshot.SUMMARY_MAX_NAMES` each, the rest counted."""

    topic: str
    title: str
    created_at: str
    corpus_fingerprint: str
    arkive_version: str
    document_count: int
    facets: list[FacetSummary]
    deliverable_types: list[str]
    omitted_deliverable_types: int
    glossary_terms: int


class TopicPage(_Frozen):
    """One page of `list_topics`: the served topics from `offset` that fit one tool result,
    out of `total`; `next_offset` is None on the last page."""

    topics: list[TopicSummary]
    total: int
    offset: int
    next_offset: int | None


class DeliverableTypeSummary(_Frozen):
    """A canonical deliverable type: how many citable documents have it, and how many
    feedback documents review deliverables of it."""

    deliverable_type: str
    description: str
    document_count: int
    feedback_count: int


class TopicGuide(_Frozen):
    """The orientation view of a topic, within one tool result. Facet and deliverable type
    descriptions are cut to `snapshot.GUIDE_DESCRIPTION_CHARS`; deliverable types that do not
    fit are counted in `omitted_deliverable_types`; curator notes that do not fit are left out
    and counted in `omitted_curator_notes` (each note's sections stay readable with
    `read_section`); each facet lists at most as many values as fit, the rest counted in
    `omitted_facet_values`; glossary terms that do not fit are counted in
    `omitted_glossary_terms` (`lookup_glossary` pages them)."""

    topic: str
    title: str
    guide_markdown: str
    curator_notes: list[CuratorNote]
    omitted_curator_notes: int
    facets: list[Facet]
    omitted_facet_values: int
    deliverable_types: list[DeliverableTypeSummary]
    omitted_deliverable_types: int
    glossary_terms: list[str]
    omitted_glossary_terms: int


class DocumentListItem(_Frozen):
    """One document as `list_documents` pages it. `doc_date` is as printed (or the metadata
    date); `date` is its calendar form and where it came from, None when the printed date is
    ambiguous or no date is known. The version fields follow `DocumentRole`. A title longer
    than `snapshot.TITLE_CHARS` or a summary longer than `snapshot.LIST_SUMMARY_CHARS` ends with an
    ellipsis where it was cut."""

    doc_id: str
    rel_path: str
    format: str
    title: str
    doc_date: str | None
    date: DocumentDate | None
    deliverable_type: str
    summary: str
    page_count: int
    is_curator_note: bool
    is_feedback: bool
    feedback_for_types: list[str]
    family_id: str
    version_status: VersionStatus
    superseded_by: str | None
    facets: dict[str, list[str]]


class DocumentInfo(DocumentListItem):
    """A document with the SHA-256 of its source bytes."""

    sha256: str


class DocumentPage(_Frozen):
    """One page of `list_documents`. `next_offset` is None on the last page."""

    documents: list[DocumentListItem]
    total: int
    offset: int
    limit: int
    next_offset: int | None


class SectionOutlineItem(_Frozen):
    section_id: str
    ordinal: int
    heading_path: str
    level: int
    page_start: int
    page_end: int


class DocumentOutline(_Frozen):
    """One page of a document's outline: `sections` from `offset`, as many as fit one tool
    result, out of `total_sections`; `next_offset` is None on the last page."""

    document: DocumentInfo
    sections: list[SectionOutlineItem]
    total_sections: int
    offset: int
    next_offset: int | None


class SearchHit(_Frozen):
    chunk_id: str
    doc_id: str
    section_id: str
    kind: str
    doc_title: str
    heading_path: str
    page_start: int
    page_end: int
    facets: dict[str, list[str]]
    rrf_score: float
    lexical_rank: int | None
    dense_rank: int | None
    # Further ranked chunks of the same document that this result list does not show.
    more_from_document: int
    content: str


class GrepMatch(_Frozen):
    doc_id: str
    section_id: str
    doc_title: str
    heading_path: str
    page_start: int
    page_end: int
    facets: dict[str, list[str]]
    line_number: int
    snippet: str


class SectionReadView(_Frozen):
    """One section's Markdown from character `offset`, as much as fits one tool result, cut at
    a line break where one falls in it. `next_offset` is None once the part reaches the end of
    the section's `total_chars`."""

    section_id: str
    doc_id: str
    doc_title: str
    heading_path: str
    page_start: int
    page_end: int
    facets: dict[str, list[str]]
    markdown: str
    offset: int
    next_offset: int | None
    total_chars: int
    prev_section_id: str | None
    next_section_id: str | None


class PageVisualView(_Frozen):
    doc_id: str
    doc_title: str
    page_number: int
    page_count: int
    native_text: str
    visual_summary: str
    webp_bytes: bytes


class GlossaryTermSummary(_Frozen):
    term: str
    kind: str
    aliases: list[str]
    # False when no sense of the term has a definition the corpus supports.
    defined: bool


class GlossaryLookup(_Frozen):
    """Without a query, `terms` lists glossary terms and `matches` is empty. With one,
    `matches` holds the full entries whose term or an alias contains the query after folding
    (empty when none does) and `terms` is empty. Either list is one page from `offset`:
    `total` counts every term or match, and `next_offset` is None on the last page."""

    query: str | None
    terms: list[GlossaryTermSummary]
    matches: list[GlossaryEntry]
    total: int
    offset: int
    next_offset: int | None


# Drafting packet.


class FamilyRecurrence(_Frozen):
    """Where the recurring text units of one exemplar section also appear, counted by
    version family so that drafts of one deliverable count once.

    `recurring_unit_count` counts the section's units whose text another version family also
    holds. Only deliverables hold units (a feedback document quoting one does not), and a
    superseded draft is not counted as a holder. `exemplar_doc_ids` are the packet's exemplars
    holding at least one of them, this one included. `corpus_family_count` is the most version
    families holding any one of those units, out of `corpus_family_total` version families of
    deliverables in the topic, and `latest_holder_date` the newest calendar date among the
    documents holding them.
    """

    recurring_unit_count: int
    exemplar_doc_ids: list[str]
    corpus_family_count: int
    corpus_family_total: int
    latest_holder_date: str | None


class ExemplarSection(_Frozen):
    """One outline entry of an exemplar. `markdown` holds the section without its visual
    transcripts when `included`, else None; `visual_count` counts those transcripts."""

    section_id: str
    ordinal: int
    heading_path: str
    level: int
    page_start: int
    page_end: int
    char_count: int
    visual_count: int
    included: bool
    family_recurrence: FamilyRecurrence | None
    markdown: str | None


class DraftExemplar(_Frozen):
    """A prior deliverable chosen as a model for the new one, with its complete outline
    (`omitted_outline_sections` is 0 unless it is the only exemplar and its outline alone
    could not fit the packet).
    `family_doc_ids` are the other members of its version family, at most
    `snapshot.MAX_FAMILY_DOC_IDS`, with the rest counted in `omitted_family_members`. A summary
    longer than `snapshot.SUMMARY_CHARS` ends with an ellipsis where it was cut."""

    doc_id: str
    rel_path: str
    title: str
    doc_date: str | None
    date: DocumentDate | None
    deliverable_type: str
    facets: dict[str, list[str]]
    page_count: int
    summary: str
    relevance_score: float
    version_status: VersionStatus
    family_doc_ids: list[str]
    omitted_family_members: int
    sections: list[ExemplarSection]
    omitted_outline_sections: int


class SupplementalSection(_Frozen):
    """A section of a document outside the exemplars that holds rare terms of the inputs."""

    doc_id: str
    doc_title: str
    date: DocumentDate | None
    section_id: str
    heading_path: str
    markdown: str


class DraftCandidate(_Frozen):
    doc_id: str
    title: str
    deliverable_type: str
    date: DocumentDate | None
    page_count: int
    summary: str
    relevance_score: float


class DraftPacket(_Frozen):
    """Everything `prepare_draft` returns for drafting one deliverable, within one tool result
    (`snapshot.fits_response`); a section left out for that reason keeps its outline entry and
    counts in `omitted_sections`. The short guidance, lessons, curator notes and recurring
    samples come before the exemplars' bulky Markdown.

    `recurring_samples` are the opening words of the recurring prose units
    (`FamilyRecurrence`) that the included exemplar sections hold in two or more version
    families, one per unit, as the first included section holding it writes them, folded by
    `passages.fold_unit`: house text an agent can tell apart from a project's own paragraphs
    in a section that mixes both."""

    topic: str
    deliverable_type: str | None
    drafting_guidance: str
    lessons: list[FeedbackLesson]
    omitted_lessons: int
    curator_notes: list[CuratorNote]
    omitted_curator_notes: int
    recurring_samples: list[str]
    exemplars: list[DraftExemplar]
    supplemental_sections: list[SupplementalSection]
    other_candidates: list[DraftCandidate]
    omitted_sections: int
