"""Domain types, provider output schemas and the error hierarchy."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


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


class InvalidCitationError(ArkiveError):
    """The corpus synthesis cites sections that do not exist."""


class SnapshotIntegrityError(ArkiveError):
    """A snapshot file is missing, malformed, or from another schema version."""


class InvalidFilterError(ArkiveError):
    """A query names a project, stage, or limit the snapshot cannot satisfy."""


class NotFoundError(ArkiveError):
    """A requested topic, document, section, or page does not exist."""


SourceFormat = Literal["pdf", "docx", "pptx", "md"]
LifecycleStage = Literal[
    "proposal",
    "memorandum",
    "feedback",
    "summary",
    "case_study",
    "guidance",
    "reference",
]
BlockKind = Literal["heading", "paragraph", "table", "callout"]
ChunkKind = Literal["prose", "table", "visual", "guidance"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# Provider structured-output schemas. OpenAI strict JSON schema needs every field
# typed explicitly, so these models avoid dict fields and defaults.


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


class DocumentMetaExtraction(_Frozen):
    title: str
    client: str
    stage: LifecycleStage
    doc_date: str | None
    summary: str


class CorpusSynthesisExtraction(_Frozen):
    synthesis_markdown: str


# Compiled corpus.


class CuratorNote(_Frozen):
    scope_path: str
    content: str


class RenderedPage(_Frozen):
    page_number: int
    native_text: str
    webp_bytes: bytes
    visual_summary: str


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
    project_id: str
    client: str
    stage: LifecycleStage
    title: str
    doc_date: str | None
    summary: str
    sections: tuple[Section, ...]
    chunks: tuple[Chunk, ...]
    pages: tuple[RenderedPage, ...]
    is_curator_note: bool = False


@dataclass(frozen=True)
class CompiledCorpus:
    """A whole compiled topic. `embeddings` row i belongs to the i-th chunk in document order."""

    corpus_dir: Path
    topic: str
    title: str
    embedding_model: str
    embedding_dimensions: int
    curator_notes: tuple[CuratorNote, ...]
    synthesis_markdown: str
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


class TopicSummary(_Frozen):
    topic: str
    title: str
    created_at: str
    corpus_fingerprint: str
    # Empty when the snapshot was written before the version was recorded.
    arkive_version: str
    document_count: int
    projects: list[str]
    stages: list[str]


class TopicGuide(_Frozen):
    topic: str
    title: str
    synthesis_markdown: str
    curator_notes: list[CuratorNote]
    projects: list[str]
    stages: list[str]


class DocumentInfo(_Frozen):
    doc_id: str
    rel_path: str
    sha256: str
    format: str
    project_id: str
    client: str
    stage: str
    title: str
    doc_date: str | None
    summary: str
    page_count: int


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
    project_id: str
    client: str
    stage: str
    doc_title: str
    heading_path: str
    page_start: int
    page_end: int
    rrf_score: float
    lexical_rank: int | None
    dense_rank: int | None
    content: str


class GrepMatch(_Frozen):
    doc_id: str
    section_id: str
    project_id: str
    stage: str
    doc_title: str
    heading_path: str
    page_start: int
    page_end: int
    line_number: int
    snippet: str


class SectionReadView(_Frozen):
    section_id: str
    doc_id: str
    doc_title: str
    project_id: str
    stage: str
    heading_path: str
    page_start: int
    page_end: int
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
