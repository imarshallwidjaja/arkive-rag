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
from typing import Annotated, Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    ID, a facet that mixes path-bound and content-bound values, an empty blueprint outline,
    or an observation batch whose quotes could not be located at all.
    """


class SnapshotIntegrityError(ArkiveError):
    """A snapshot file is missing, malformed, or from another schema version."""


class InvalidFilterError(ArkiveError):
    """A query names a facet, a facet value, or a limit the snapshot cannot satisfy."""


class NotFoundError(ArkiveError):
    """A requested topic, document, section, or page does not exist."""


SourceFormat = Literal["pdf", "docx", "pptx", "md"]
BlockKind = Literal["heading", "paragraph", "table", "callout"]
ChunkKind = Literal["prose", "table", "visual", "guidance"]
CraftAspect = Literal[
    "structure", "convention", "caveat", "disclaimer", "reviewer_feedback", "tone"
]
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
    """A name/value pair the document states about itself, such as client or audience."""

    name: str
    value: str


class TermDraft(_Frozen):
    term: str
    kind: str
    aliases: list[str]
    quote: str
    defines: bool


class CraftDraft(_Frozen):
    aspect: CraftAspect
    statement: str
    quote: str


class ObservationDraft(_Frozen):
    """What one batch of a document's sections shows. Quotes are verbatim source text."""

    title: str
    doc_date: str | None
    deliverable_type: str
    summary: str
    attributes: list[StatedAttribute]
    terms: list[TermDraft]
    craft: list[CraftDraft]


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


class DeliverableTypeDraft(_Frozen):
    name: str
    description: str
    exemplar_doc_ids: list[str]
    feedback_doc_ids: list[str]


class OrganisationDraft(_Frozen):
    facets: list[FacetDraft]
    deliverable_types: list[DeliverableTypeDraft]


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


class BlueprintSectionDraft(_Frozen):
    heading: str
    purpose: str
    citations: list[str]


class GuidanceDraft(_Frozen):
    statement: str
    citations: list[str]


class StandingPassageSelection(_Frozen):
    """One standing passage candidate the agent keeps, by ID, with its label. The text and
    citations come from the candidate, never from the agent."""

    candidate_id: str
    label: str


class BlueprintDraft(_Frozen):
    description: str
    outline: list[BlueprintSectionDraft]
    conventions: list[GuidanceDraft]
    pitfalls: list[GuidanceDraft]
    standing_passages: list[StandingPassageSelection]


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


class CraftObservation(_Frozen):
    aspect: CraftAspect
    statement: str
    quote: str
    section_ordinal: int


class ObservationCard(_Frozen):
    title: str
    doc_date: str | None
    deliverable_type: str
    summary: str
    attributes: tuple[StatedAttribute, ...]
    terms: tuple[TermObservation, ...]
    craft: tuple[CraftObservation, ...]


class ChunkPiece(_Frozen):
    kind: ChunkKind
    content: str
    page_start: int
    page_end: int


class ReadSection(_Frozen):
    ordinal: int
    heading_path: str
    level: int
    page_start: int
    page_end: int
    markdown: str
    pieces: tuple[ChunkPiece, ...]


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
    sections but does not establish what it means there."""

    definition: str | None
    citations: Annotated[tuple[str, ...], Field(min_length=1)]


class GlossaryEntry(_Frozen):
    term: str
    kind: str
    aliases: tuple[str, ...]
    senses: Annotated[tuple[GlossarySense, ...], Field(min_length=1)]


class BlueprintSection(_Frozen):
    heading: str
    purpose: str
    citations: Annotated[tuple[str, ...], Field(min_length=1)]


class BlueprintGuidance(_Frozen):
    statement: str
    citations: Annotated[tuple[str, ...], Field(min_length=1)]


class StandingPassage(_Frozen):
    """A block of exemplar text, such as a method background, a reference entry or a
    disclaimer, that recurs in another exemplar document and that the blueprint agent selected
    as standing house content. `text` is exactly as it appears in the section `citations[0]`;
    every other citation is a section of another exemplar document where it recurs."""

    label: str
    text: str
    citations: Annotated[tuple[str, ...], Field(min_length=2)]


class Blueprint(_Frozen):
    """How the corpus writes one recurring deliverable type. Outline, conventions and pitfalls
    carry no project-specific values; standing passages are the only verbatim source text."""

    deliverable_type: str
    description: str
    exemplar_doc_ids: Annotated[tuple[str, ...], Field(min_length=1)]
    feedback_doc_ids: tuple[str, ...]
    outline: Annotated[tuple[BlueprintSection, ...], Field(min_length=1)]
    conventions: tuple[BlueprintGuidance, ...]
    pitfalls: tuple[BlueprintGuidance, ...]
    standing_passages: tuple[StandingPassage, ...]


class BuildReport(_Frozen):
    """What the build reduced or skipped without failing.

    The first three counts are the contracted reductions: observations discarded because
    their quote could not be located after one rerun, glossary definitions the support check
    did not confirm (stored as `definition=None`), and blueprint conventions or pitfalls it
    did not confirm (dropped). `judge_refusals` counts support and binding questions the
    decision model refused, by operation; `pruned_facet_values` and `pruned_facets` name what
    binding left empty.
    """

    discarded_unlocated_quotes: int
    unsupported_definitions: int
    dropped_blueprint_statements: int
    judge_refusals: dict[str, int]
    pruned_facet_values: list[str]
    pruned_facets: list[str]


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
    deliverable_type: str
    summary: str
    # Facet name to the document's sorted values, in facet definition order.
    facets: dict[str, tuple[str, ...]]
    sections: tuple[Section, ...]
    chunks: tuple[Chunk, ...]
    pages: tuple[RenderedPage, ...]
    is_curator_note: bool


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
    blueprints: tuple[Blueprint, ...]
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
    name: str
    binding: FacetBinding
    values: list[str]


class TopicSummary(_Frozen):
    topic: str
    title: str
    created_at: str
    corpus_fingerprint: str
    arkive_version: str
    document_count: int
    facets: list[FacetSummary]
    blueprints: list[str]
    glossary_terms: int


class BlueprintSummary(_Frozen):
    deliverable_type: str
    description: str
    exemplar_count: int


class TopicGuide(_Frozen):
    """The orientation view of a topic. Blueprints appear as summaries; `get_blueprint`
    returns one in full."""

    topic: str
    title: str
    guide_markdown: str
    curator_notes: list[CuratorNote]
    facets: list[Facet]
    blueprints: list[BlueprintSummary]
    glossary_terms: list[str]


class DocumentInfo(_Frozen):
    """`doc_date` is as printed (or the metadata date); `date` is its calendar form and
    where it came from, None when the printed date is ambiguous or no date is known."""

    doc_id: str
    rel_path: str
    sha256: str
    format: str
    title: str
    doc_date: str | None
    date: DocumentDate | None
    deliverable_type: str
    summary: str
    page_count: int
    is_curator_note: bool
    facets: dict[str, list[str]]


class SectionOutlineItem(_Frozen):
    section_id: str
    ordinal: int
    heading_path: str
    level: int
    page_start: int
    page_end: int


class DocumentOutline(_Frozen):
    document: DocumentInfo
    sections: list[SectionOutlineItem]


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
    section_id: str
    doc_id: str
    doc_title: str
    heading_path: str
    page_start: int
    page_end: int
    facets: dict[str, list[str]]
    markdown: str
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
    """Without a query, `terms` lists every glossary term and `matches` is empty. With one,
    `matches` holds the full entries whose term or an alias contains the query after folding
    (empty when none does) and `terms` is empty."""

    query: str | None
    terms: list[GlossaryTermSummary]
    matches: list[GlossaryEntry]
