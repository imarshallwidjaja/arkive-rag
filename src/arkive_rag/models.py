"""Domain types, the shared text folding, and the error hierarchy."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


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


class BlueprintDraft(_Frozen):
    description: str
    exemplar_doc_ids: list[str]
    outline: list[BlueprintSectionDraft]
    conventions: list[GuidanceDraft]
    pitfalls: list[GuidanceDraft]


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


class DocumentReading(_Frozen):
    """The reading of one source. `discarded_quotes` counts observations dropped because
    their quote was still not found in the source after one rerun."""

    sha256: str
    format: SourceFormat
    is_curator_note: bool
    sections: tuple[ReadSection, ...]
    pages: tuple[RenderedPage, ...]
    card: ObservationCard
    discarded_quotes: int


@dataclass(frozen=True)
class CorpusDocument:
    """A reading placed in the corpus: its ID follows sorted path order."""

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


class Blueprint(_Frozen):
    """How the corpus writes one recurring deliverable type, without project-specific values."""

    deliverable_type: str
    description: str
    exemplar_doc_ids: Annotated[tuple[str, ...], Field(min_length=1)]
    feedback_doc_ids: tuple[str, ...]
    outline: Annotated[tuple[BlueprintSection, ...], Field(min_length=1)]
    conventions: tuple[BlueprintGuidance, ...]
    pitfalls: tuple[BlueprintGuidance, ...]


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
    doc_date: str | None
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


class TopicGuide(_Frozen):
    topic: str
    title: str
    guide_markdown: str
    curator_notes: list[CuratorNote]
    facets: list[Facet]
    blueprints: list[Blueprint]
    glossary_terms: list[str]


class DocumentInfo(_Frozen):
    doc_id: str
    rel_path: str
    sha256: str
    format: str
    title: str
    doc_date: str | None
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
