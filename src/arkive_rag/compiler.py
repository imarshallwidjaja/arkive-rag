"""Compile a corpus directory into a `CompiledCorpus` in three passes.

1. Reading (`read_document`, per source, cached by content): render to WebP, transcribe with
   the generation model, check the transcript against the native text layer, stitch pages
   into sections and chunk pieces, and write an observation card. A reading holds no corpus
   IDs and no path.
2. Induction (`arkive_rag.induction.induce`, corpus-wide): facets, deliverable types and
   feedback roles, facet binding, version families, feedback lessons, the glossary, recurring
   text units and the topic guide.
3. Binding (`bind_document`): section and chunk IDs, breadcrumbs from the bound facets, then
   embeddings.

The request types below own every prompt, output schema and reasoning effort the reading
pass sends.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
import zlib
from bisect import bisect_right
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol
from xml.etree import ElementTree

import pypdfium2 as pdfium
from PIL import Image

from arkive_rag.induction import INDUCTION_EFFORTS, Checked, generate_checked, induce
from arkive_rag.llm import Effort, LanguageGateway
from arkive_rag.models import (
    MAX_RENDER_SIDE_PX,
    ArkiveError,
    CardHeader,
    Chunk,
    ChunkKind,
    ChunkPiece,
    CompiledCorpus,
    CompiledDocument,
    ConversionError,
    CorpusDocument,
    CraftObservation,
    CuratorNote,
    DocumentDate,
    DocumentReading,
    DocumentRole,
    Facet,
    MissingDependencyError,
    ObservationCard,
    ObservationDraft,
    PageBlock,
    PageExtraction,
    ReadSection,
    RenderedPage,
    RenderedSource,
    Section,
    SourceFormat,
    StatedAttribute,
    TermObservation,
    TranscriptionFidelityError,
    UnsafePathError,
    UnsupportedFormatError,
    document_id,
    fold_for_quote,
    fold_text,
    heading_context,
    iso_date,
    parse_printed_date,
)

SUPPORTED_SUFFIXES: dict[str, SourceFormat] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".md": "md",
}
CURATOR_NOTE_NAME = "AGENTS.md"
PROSE_CHUNK_CHARS = 1800
WHOLE_TABLE_CHARS = 2400
TABLE_PIECE_CHARS = 2000
# Hard ceiling for any chunk, well inside the embedding model's 8,192-token input limit.
MAX_CHUNK_CHARS = 6000
# Hard ceiling for a section's Markdown, so one `read_section` result, observation batch or
# support-check window never has to carry a whole unheaded transcript or a 20-page table.
MAX_SECTION_CHARS = 32_000
# The least room beside an oversized section's heading worth starting its body in; with less,
# the body starts in the next piece.
MIN_FIRST_SLICE_CHARS = 256
# The longest info word a reopened code fence repeats (`json`, `python`); a longer one is left
# off, so reopening a cut block adds only a few characters.
FENCE_INFO_CHARS = 32
OFFICE_TIMEOUT_SECONDS = 120
PREV_CONTEXT_CHARS = 1200
# Sections are packed in order into observation batches of about this many characters; a
# larger section is a batch of its own.
OBSERVE_BATCH_CHARS = 24_000
# Page rendering. The page image is part of every `transcribe_page` cache key, so changing the
# scale or quality re-transcribes every page of every cached corpus.
PDF_RENDER_SCALE = 2.0
WEBP_QUALITY = 85
# Names the render path's output. It is part of every render cache key, so a change in what a
# rendered page or its metadata date holds is never answered from an older render.
RENDER_CACHE_REVISION = "arkive-page-render/2"
# Largest `docProps/core.xml` read for a metadata date; a real one is a few kilobytes, and a
# larger entry (a decompression bomb) is refused before it is inflated.
MAX_CORE_PROPERTIES_BYTES = 1_000_000
# On a two-page document an edge line repeated on both pages is furniture only when it is at
# most this long, on pages of at least this many non-blank lines: a running header or footer
# is short and sits beside a body, while a short slide or a body line need not be furniture.
TWO_PAGE_FURNITURE_MAX_CHARS = 120
TWO_PAGE_MIN_LINES = 5

# libpdfium and pypdfium2's object tracker are not thread-safe: every pdfium call
# across all threads goes through this lock. It is reentrant so `_closing_pdf` can close a
# document from inside a block that already holds it.
_PDFIUM_LOCK = threading.RLock()

_FILENAME_ANCHOR = re.compile(r"\b[A-Za-z0-9_.-]+\.(?:tif|tiff|shp|nc|gpkg|csv)\b", re.IGNORECASE)
# An identifier fragment that ends where a following filename anchor begins, separated
# only by kerning spaces and at most one line break.
_WRAPPED_PREFIX = re.compile(r"([A-Za-z0-9_.-]+)[ \t]*\n?[ \t]*\Z")
# A narrow table cell can wrap one long filename over as many as this many lines before its
# extension, so each anchor also accepts joins with up to this many preceding fragments.
WRAPPED_FILENAME_FRAGMENTS = 6
# A leading `-` is a sign only when it does not join two tokens, as in `2016-2017`.
_SIGN = r"(?:(?<![\w-])-)?"
_FORMATTED_NUMBER = re.compile(rf"{_SIGN}\b\d+(?:[.,]\d+)+\b")
_LONG_INTEGER = re.compile(rf"{_SIGN}\b\d{{3,}}\b")
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# A bare number is a page number only on a page's first or last line; a labelled
# one ("Page 3", "3 of 12") is a page number on any edge line.
_BARE_PAGE_NUMBER = re.compile(r"\s*\d{1,4}\s*")
_PAGE_LABEL = re.compile(
    r"\s*(?:page\s+\d{1,4}(?:\s+of\s+\d{1,4})?|\d{1,4}\s+of\s+\d{1,4})\s*", re.IGNORECASE
)
_EDGE_PAGE_NUMBER = re.compile(r"(\d{1,4})\s+(.+)|(.+?)\s+(\d{1,4})")
_TABLE_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_HEADING = re.compile(r"(#{1,6})\s+(\S.*)")
# A stripped code fence opener (CommonMark): three or more backticks and an info string with
# no backtick, or three or more tildes and any info string. "```cli``` runs it" opens an
# inline code span, not a fence.
_FENCE_OPENER = re.compile(r"(`{3,})[^`]*|(~{3,}).*")


# Provider requests.

PAGE_INSTRUCTIONS = """\
You transcribe one rendered document page into structured Markdown blocks.
The page image is the authority for layout and for the characters a reader sees.
Use the native text layer to verify numbers, units, filenames and identifiers.
Some PDFs encode font glyphs wrongly in the text layer, for example \u019f or =
in place of "ti" (ENVes\u019fgator, Sen=nel-2) or a dropped ligature (aper):
repair those from the image. Never otherwise change the author's wording,
spelling or typos.

Rules:
- Emit blocks in reading order. kind is heading, paragraph, table or callout.
- heading_level is 1-4 for headings (1 = document title or top-level section)
  and 0 for every other kind. Heading markdown is the heading text without
  leading # marks and keeps any visible section number, for example
  "3.3. Wet and Dry Years Analysis".
- Tables are GitHub Markdown tables with a header row and a | --- | separator row.
  Never summarise a table: transcribe every row and cell.
- Set continues_from_previous_page when the first block on this page continues
  a paragraph or table that the previous page ended mid-way. A continued table
  may omit or repeat its header row. Set merges_into_previous_row when the first
  continued table row is the remainder of a row cut by the page break.
- Omit running headers, running footers and page numbers.
- For every chart, map, diagram, photo or figure add a visual: label is its
  caption or a short title; transcript states what it shows, every axis label,
  legend entry, annotation and printed value.
- Copy every number, unit and filename exactly as printed. Never invent content.
"""

OBSERVE_INSTRUCTIONS = """\
You read consecutive sections of one document and record what they show. Another
step organises a whole collection from these observations, so state only what the
text supports.

- title: the document's title as printed; when none is printed, a short descriptive
  title of what the document is. Never empty. doc_date: its date as printed, or null.
- deliverable_type: a short generic name for the kind of document, for example
  "technical memorandum", "proposal", "review comments", "campaign brief" or
  "slide deck". Never a project or client name.
- summary: two to four factual sentences on its subject, methods and results.
- attributes: name/value pairs the document states about itself, such as client,
  author, audience, project, region, sensor, product or channel. Use short
  lowercase names. Only values the text states.
- terms: domain terms, acronyms, named methods, datasets, tools and products the
  text uses. kind is a one-word category (acronym, method, dataset, tool, product,
  concept). defines is true only when the quote states what the term means.
- craft: how the document is written: structure (sections it always has), convention
  (house rules for tables, units, naming), caveat, disclaimer, reviewer_feedback
  (what a reviewer asked to change) and tone. statement is a generic rule a writer
  could follow, without this document's project-specific values.
- quote: copy one sentence or phrase of at most 300 characters verbatim from the
  sections, as contiguous text. Never quote section headers you were not shown,
  never paraphrase, and never give section numbers instead of a quote.
"""

CONSOLIDATE_INSTRUCTIONS = """\
You receive the title, date, deliverable type and summary written for each part of one
long document. Write the document's single title, date (null if none is printed),
deliverable type (a short generic name) and a factual two-to-four sentence summary. When a
part gives a date, copy the chosen part's Date exactly as written; do not reformat it.
"""


@dataclass(frozen=True)
class PageTranscription:
    """Transcribe one rendered page. `missing_values` names anchors an earlier attempt lost."""

    operation: ClassVar[str] = "transcribe_page"
    output: ClassVar[type[PageExtraction]] = PageExtraction
    instructions: ClassVar[str] = PAGE_INSTRUCTIONS
    native_text: str
    image_webp: bytes
    prev_context: str | None
    effort: Effort
    missing_values: tuple[str, ...] = ()

    def text(self) -> str:
        text = f"NATIVE TEXT LAYER:\n{self.native_text or '(none: the page has no text layer)'}\n"
        if self.prev_context:
            text += f"\nEND OF PREVIOUS PAGE TRANSCRIPT:\n{self.prev_context}\n"
        if self.missing_values:
            text += (
                "\nA previous transcription dropped these values. Each appears on the page;"
                f" transcribe them exactly where they occur: {', '.join(self.missing_values)}\n"
            )
        return text


@dataclass(frozen=True)
class BatchSection:
    """One section as an observation batch shows it, by document-local ordinal."""

    ordinal: int
    heading_path: str
    page_start: int
    page_end: int
    markdown: str


@dataclass(frozen=True)
class ObserveSections:
    """Observe one batch of a document's sections. `feedback` names a rejected answer's flaws."""

    operation: ClassVar[str] = "observe"
    output: ClassVar[type[ObservationDraft]] = ObservationDraft
    effort: ClassVar[Effort] = "low"
    instructions: ClassVar[str] = OBSERVE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    sections: tuple[BatchSection, ...]
    feedback: str | None = None

    def text(self) -> str:
        parts = [
            f"=== Section {s.ordinal}: {s.heading_path} (pp. {s.page_start}-{s.page_end}) ===\n"
            f"{s.markdown}"
            for s in self.sections
        ]
        text = "SECTIONS:\n" + "\n\n".join(parts)
        return f"{text}\n\n{self.feedback}" if self.feedback else text


@dataclass(frozen=True)
class ConsolidateCard:
    """One title, date, deliverable type and summary from the headers of several batches."""

    operation: ClassVar[str] = "consolidate_card"
    output: ClassVar[type[CardHeader]] = CardHeader
    effort: ClassVar[Effort] = "low"
    instructions: ClassVar[str] = CONSOLIDATE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    headers: tuple[CardHeader, ...]

    def text(self) -> str:
        return "\n\n".join(
            f"PART {i}:\nTitle: {h.title}\nDate: {h.doc_date or 'none'}\n"
            f"Deliverable type: {h.deliverable_type}\nSummary: {h.summary}"
            for i, h in enumerate(self.headers, start=1)
        )


@dataclass(frozen=True)
class SourceFile:
    rel_path: str
    path: Path
    format: SourceFormat
    # The size discovery saw; it only orders the reading work, largest first.
    size_bytes: int

    @property
    def is_curator_note(self) -> bool:
        return self.path.name == CURATOR_NOTE_NAME


class RenderCache(Protocol):
    """Rendered sources keyed by `render_cache_key`.

    `load_render` returns a complete stored rendering, or None when nothing complete and
    well-formed is stored. `store_render` keeps the first complete rendering stored under a
    key and returns it, so concurrent builds that render the same source all use one.
    """

    def load_render(self, cache_key: str) -> RenderedSource | None: ...

    def store_render(self, cache_key: str, rendered: RenderedSource) -> RenderedSource: ...


def discover_corpus(corpus_dir: Path) -> list[SourceFile]:
    """List corpus files in sorted order; reject unsupported files and symlinks before work."""
    if not corpus_dir.is_dir():
        raise ConversionError(f"corpus directory {corpus_dir} does not exist")
    root = corpus_dir.resolve()
    sources: list[SourceFile] = []
    # rglob does not descend into symlinked directories, so checking every entry
    # also covers symlinks among a file's parents.
    for path in sorted(corpus_dir.rglob("*")):
        rel = path.relative_to(corpus_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise UnsafePathError(f"{rel.as_posix()}: symlinks are not allowed in a corpus")
        if not path.is_file():
            continue
        fmt = SUPPORTED_SUFFIXES.get(path.suffix.lower())
        if fmt is None:
            raise UnsupportedFormatError(
                f"{rel.as_posix()}: supported formats are .pdf .docx .pptx .md"
            )
        sources.append(SourceFile(rel.as_posix(), path, fmt, path.lstat().st_size))
    return sources


def assign_document_ids(sources: Sequence[SourceFile]) -> list[str]:
    """`document_id` of each source, in order. Two paths with one ID, such as the NFC and NFD
    spellings of one name, stop the build before any file is read."""
    owners: dict[str, str] = {}
    for source in sources:
        doc_id = document_id(source.rel_path)
        if doc_id in owners:
            raise ArkiveError(
                f"{owners[doc_id]!r} and {source.rel_path!r} have the same document ID {doc_id};"
                " rename one of them"
            )
        owners[doc_id] = source.rel_path
    return list(owners)


def read_source(source: SourceFile) -> bytes:
    """Read a source once; O_NOFOLLOW refuses a symlink swapped in after discovery."""
    try:
        fd = os.open(source.path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise ConversionError(f"{source.rel_path}: cannot read: {exc}") from exc


def _decode_markdown(source: SourceFile, data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConversionError(f"{source.rel_path} is not UTF-8 text: {exc}") from exc


# Rendering.


def convert_office_to_pdf(data: bytes, suffix: str, name: str) -> bytes:
    """Convert a private copy of `data` so the file LibreOffice reads cannot change under it."""
    soffice = shutil.which("soffice")
    if soffice is None:
        raise MissingDependencyError("LibreOffice `soffice` is required for .docx and .pptx files")
    with tempfile.TemporaryDirectory(prefix="arkive-soffice-") as scratch:
        source = Path(scratch, f"source{suffix}")
        source.write_bytes(data)
        profile, out_dir = Path(scratch, "profile"), Path(scratch, "out")
        command = [
            soffice,
            "--headless",
            "--norestore",
            "--nolockcheck",
            f"-env:UserInstallation={profile.as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(source),
        ]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": scratch,
            "TMPDIR": scratch,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=OFFICE_TIMEOUT_SECONDS,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise ConversionError(f"{name}: soffice timed out") from exc
        pdf = out_dir / "source.pdf"
        if result.returncode != 0 or not pdf.is_file() or pdf.stat().st_size == 0:
            raise ConversionError(
                f"{name}: soffice exited {result.returncode}: {result.stderr.strip()}"
            )
        return pdf.read_bytes()


def _render_page(document: pdfium.PdfDocument, index: int) -> tuple[str, Image.Image]:
    page = document[index]
    try:
        textpage = page.get_textpage()
        try:
            text = textpage.get_text_bounded()
        finally:
            textpage.close()
        # A page whose longest side at PDF_RENDER_SCALE would pass MAX_RENDER_SIDE_PX renders
        # at the scale that fits it, so the oversized bitmap is never allocated. Every other
        # page renders exactly as before, keeping its transcription cache entries valid.
        longest_points = max(page.get_size())
        scale = PDF_RENDER_SCALE
        if longest_points * scale > MAX_RENDER_SIDE_PX:
            scale = MAX_RENDER_SIDE_PX / longest_points
        bitmap = page.render(scale=scale)
        try:
            # copy() detaches the pixels from pdfium's buffer before it is freed.
            image = bitmap.to_pil().copy()
        finally:
            bitmap.close()
    finally:
        page.close()
    return text.replace("\r\n", "\n"), image


def encode_page_image(image: Image.Image) -> bytes:
    """WebP bytes of a rendered page, its longest side clamped to MAX_RENDER_SIDE_PX.

    pdfium can round a clamped render a pixel past the limit, so the clamp is enforced here
    too. Pillow failures (a bitmap libwebp refuses, a decompression bomb, memory) are
    ConversionError, so they name the file like every other rendering failure.
    """
    try:
        if max(image.size) > MAX_RENDER_SIDE_PX:
            ratio = MAX_RENDER_SIDE_PX / max(image.size)
            size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=WEBP_QUALITY)
    except (ValueError, OSError, MemoryError, Image.DecompressionBombError) as exc:
        raise ConversionError(f"cannot encode a {image.width}x{image.height} page: {exc}") from exc
    return buffer.getvalue()


# What a pdfium or Pillow call can raise on a damaged or hostile file; each is ConversionError.
_PDFIUM_FAILURES = (pdfium.PdfiumError, ValueError, OSError, RuntimeError, MemoryError)


def _open_pdf(pdf_bytes: bytes) -> pdfium.PdfDocument:
    """Open a PDF; the caller holds `_PDFIUM_LOCK`."""
    try:
        return pdfium.PdfDocument(pdf_bytes)
    except _PDFIUM_FAILURES as exc:
        raise ConversionError(f"cannot open PDF: {exc}") from exc


def render_pdf(pdf_bytes: bytes) -> list[tuple[str, bytes]]:
    """Return (native text, WebP bytes) for each page; any pdfium failure is ConversionError."""
    with _PDFIUM_LOCK:
        document = _open_pdf(pdf_bytes)
    pages: list[tuple[str, bytes]] = []
    with _closing_pdf(document):
        with _PDFIUM_LOCK:
            try:
                page_count = len(document)
            except _PDFIUM_FAILURES as exc:
                raise ConversionError(f"cannot count pages: {exc}") from exc
        for index in range(page_count):
            with _PDFIUM_LOCK:
                try:
                    text, image = _render_page(document, index)
                except _PDFIUM_FAILURES as exc:
                    raise ConversionError(f"cannot render page {index + 1}: {exc}") from exc
            pages.append((text, encode_page_image(image)))
    return pages


@contextmanager
def _closing_pdf(document: pdfium.PdfDocument) -> Iterator[None]:
    """Close `document` when the block ends. After a block that raised, a failure to close is
    suppressed so it never replaces that error; after a clean block it is ConversionError."""
    try:
        yield
    except BaseException:
        with _PDFIUM_LOCK, suppress(*_PDFIUM_FAILURES):
            document.close()
        raise
    with _PDFIUM_LOCK:
        try:
            document.close()
        except _PDFIUM_FAILURES as exc:
            raise ConversionError(f"cannot close PDF: {exc}") from exc


def render_cache_key(source_format: SourceFormat, sha256: str) -> str:
    """The render cache key of a source: its format, its bytes' SHA-256 and every setting that
    shapes a rendered page."""
    parts = [
        RENDER_CACHE_REVISION,
        source_format,
        sha256,
        PDF_RENDER_SCALE,
        WEBP_QUALITY,
        MAX_RENDER_SIDE_PX,
    ]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def render_source(
    source: SourceFile, data: bytes, sha256: str, render_cache: RenderCache | None
) -> RenderedSource:
    """Every page of a PDF, Word or PowerPoint source and its file's metadata date.

    The metadata date is read from the original bytes first, so a file whose metadata is
    corrupt fails before LibreOffice or pdfium renders anything. With a render cache, a source
    rendered before is answered from it: neither LibreOffice nor pdfium runs.
    """
    key = render_cache_key(source.format, sha256)
    if render_cache is not None and (cached := render_cache.load_render(key)) is not None:
        return cached
    embedded = metadata_date(source, data)
    pdf = (
        data
        if source.format == "pdf"
        else convert_office_to_pdf(data, source.path.suffix.lower(), source.rel_path)
    )
    try:
        pages = render_pdf(pdf)
    except ConversionError as exc:
        raise ConversionError(f"{source.rel_path}: {exc}") from exc
    if not pages:
        raise ConversionError(f"{source.rel_path}: the document has no pages")
    rendered = RenderedSource(tuple(pages), embedded)
    return rendered if render_cache is None else render_cache.store_render(key, rendered)


# Fidelity gate.


EdgePosition = Literal["top", "bottom"]


def _edge_positions(lines: Sequence[str]) -> dict[int, set[EdgePosition]]:
    """Indexes of a page's edge lines and the edges they sit at: the first two non-blank
    lines are top lines and the last two bottom lines; on a page of at most four non-blank
    lines every line is both."""
    nonblank = [i for i, line in enumerate(lines) if line.strip()]
    if len(nonblank) <= 4:
        return {i: {"top", "bottom"} for i in nonblank}
    edges: dict[int, set[EdgePosition]] = {i: {"top"} for i in nonblank[:2]}
    for i in nonblank[-2:]:
        edges.setdefault(i, set()).add("bottom")
    return edges


def _furniture_keys(line: str, page_index: int, position: EdgePosition) -> set[str]:
    """The exact line at its edge, plus the line without an edge number that tracks the page
    index (`<rest> NUL <page index - number>`)."""
    text = " ".join(line.lower().split())
    keys = {f"{position}\x00{text}"}
    if match := _EDGE_PAGE_NUMBER.fullmatch(text):
        number, rest = (match[1], match[2]) if match[1] else (match[4], match[3])
        keys.add(f"{position}\x00{rest}\x00{page_index - int(number)}")
    return keys


def _page_furniture_keys(text: str, page_index: int) -> set[str]:
    lines = text.splitlines()
    return {
        key
        for i, positions in _edge_positions(lines).items()
        for position in positions
        for key in _furniture_keys(lines[i], page_index, position)
    }


def _two_page_furniture_keys(text: str, page_index: int) -> set[str]:
    """The furniture candidates of one page of a two-page document: edge lines of at most
    TWO_PAGE_FURNITURE_MAX_CHARS that hold no part of a data filename (a fragment of one
    wrapped over lines included, `_filename_lines`), on a page of at least
    TWO_PAGE_MIN_LINES non-blank lines; a numbered line counts only when its number is the
    page number itself."""
    lines = text.splitlines()
    if sum(1 for line in lines if line.strip()) < TWO_PAGE_MIN_LINES:
        return set()
    filename_lines = _filename_lines(text)
    keys: set[str] = set()
    for i, positions in _edge_positions(lines).items():
        line = lines[i]
        if len(line.strip()) > TWO_PAGE_FURNITURE_MAX_CHARS or i in filename_lines:
            continue
        for position in positions:
            keys |= {
                key
                for key in _furniture_keys(line, page_index, position)
                if key.count("\x00") == 1 or key.endswith("\x00-1")
            }
    return keys


def _filename_lines(text: str) -> set[int]:
    """Indexes of the lines of `text`, as `splitlines` numbers them, that hold part of a data
    filename, every fragment of one wrapped over lines included."""
    ends = list(accumulate(len(line) for line in text.splitlines(keepends=True)))
    return {
        line
        for occurrence in _filename_occurrences(text)
        for line in range(
            bisect_right(ends, occurrence.start), bisect_right(ends, occurrence.end - 1) + 1
        )
    }


def find_furniture(page_texts: Sequence[str]) -> frozenset[str]:
    """Keys of top lines repeated as top lines, and bottom lines repeated as bottom lines, on
    at least three pages.

    A two-page document has no third page to confirm a running header, so there a line is
    furniture only when both pages repeat it at the same edge and pass
    `_two_page_furniture_keys`.
    """
    if len(page_texts) == 2:
        first, second = (_two_page_furniture_keys(t, i) for i, t in enumerate(page_texts))
        return frozenset(first & second)
    counts: Counter[str] = Counter()
    for index, text in enumerate(page_texts):
        counts.update(_page_furniture_keys(text, index))
    return frozenset(key for key, count in counts.items() if count >= 3)


def strip_furniture(text: str, furniture: frozenset[str], page_index: int) -> str:
    """Drop furniture and page numbers from the page's edge lines only: a line in the body of
    the page stays even when it repeats an edge line."""
    lines = text.splitlines()
    edges = _edge_positions(lines)
    nonblank = sorted(edges)
    # A fragment of a filename wrapped over lines (`2026` above `-final.tif`) is never a page
    # number.
    bounds = ({nonblank[0], nonblank[-1]} if nonblank else set()) - _filename_lines(text)
    dropped = {
        i
        for i, positions in edges.items()
        if (i in bounds and _BARE_PAGE_NUMBER.fullmatch(lines[i]))
        or _PAGE_LABEL.fullmatch(lines[i])
        or any(_furniture_keys(lines[i], page_index, p) & furniture for p in positions)
    }
    return "\n".join(line for i, line in enumerate(lines) if i not in dropped)


def _normalize_numbers(text: str) -> str:
    return _THOUSANDS_COMMA.sub("", text.replace("\u2212", "-"))


def _number_anchors(text: str) -> set[str]:
    """Signed formatted numbers and long integers in number-normalized text."""
    integers = _LONG_INTEGER.findall(_FORMATTED_NUMBER.sub(" ", text))
    return {*_FORMATTED_NUMBER.findall(text), *integers}


@dataclass(frozen=True)
class FilenameAnchor:
    """One filename in native body text. `name` is the lowercased filename match, which for a
    filename wrapped over lines is only its last fragment; `accepted` holds the emitted
    filenames that satisfy it; `display` is the whole filename as native text wraps it, with
    the line breaks removed, as a retry or an error names it."""

    name: str
    accepted: frozenset[str]
    display: str


@dataclass(frozen=True)
class _FilenameOccurrence:
    """One filename match in a text: its anchor, the forms an emitted copy of it offers (the
    match with and without a leading `-` or `.` run), and the characters it covers, from the
    first fragment of a filename wrapped over lines to its end."""

    anchor: FilenameAnchor
    forms: frozenset[str]
    start: int
    end: int


def _filename_occurrences(text: str) -> list[_FilenameOccurrence]:
    """Every filename in `text`, in order (see `fidelity_anchors`)."""
    occurrences: list[_FilenameOccurrence] = []
    for match in _FILENAME_ANCHOR.finditer(text):
        # `\b` starts a match after a leading `-` or `.` (`2026\n-final.tif`); the run joins
        # the filename, so a fragment before it can still continue the name.
        start = match.start()
        while start > 0 and text[start - 1] in "-.":
            start -= 1
        bare = match.group(0).lower()
        name = joined = text[start : match.start()].lower() + bare
        accepted = {bare, name}
        wrapped_start = start
        line_broken = True
        continues = name[0] in "_.-"
        for _ in range(WRAPPED_FILENAME_FRAGMENTS):
            fragment = _WRAPPED_PREFIX.search(text, 0, start)
            if fragment is None:
                break
            joined = fragment.group(1).lower() + joined
            accepted.add(joined)
            # A fragment cut from a larger number (`1,234.5`) or joined by spaces alone ends
            # the run of line-broken fragments.
            line_broken = (
                line_broken
                and "\n" in fragment.group(0)
                and text[fragment.start() - 1 : fragment.start()] != ","
            )
            if line_broken and (continues or fragment.group(1)[-1] in "_.-"):
                wrapped_start = fragment.start()
            continues = False
            start = fragment.start()
        display = "".join(text[wrapped_start : match.end()].split())
        anchor = FilenameAnchor(name, frozenset(accepted), display)
        occurrences.append(
            _FilenameOccurrence(anchor, frozenset({bare, name}), wrapped_start, match.end())
        )
    return occurrences


def _numbers_outside(text: str, occurrences: Sequence[_FilenameOccurrence]) -> set[str]:
    """Number anchors of `text` with every filename occurrence blanked out."""
    outside = list(text)
    for occurrence in occurrences:
        outside[occurrence.start : occurrence.end] = " " * (occurrence.end - occurrence.start)
    return _number_anchors(_normalize_numbers("".join(outside)))


def fidelity_anchors(body_text: str) -> tuple[tuple[FilenameAnchor, ...], set[str]]:
    """Return (filenames, signed numbers) from native body text, without thousands commas.

    Each filename anchor is satisfied by itself, and by its joins with up to
    WRAPPED_FILENAME_FRAGMENTS identifier fragments directly before it. Native text can split
    a long filename mid-identifier across line breaks or kerning spaces, as a narrow table
    cell does, so the anchor holds only the tail while a faithful transcription carries the
    whole name. A tail that starts with `-` or `.` keeps it (`2026\\n-final.tif`), so the
    fragment before it still joins. Every occurrence is its own anchor, so two wrapped
    filenames that end in the same fragment each need their own transcription; occurrences
    with the same accepted forms are one anchor.

    A number inside a filename wrapped over lines, as `2026` in `dry_\\n2026\\nfinal.tif`, is
    part of the filename, so it is not a number anchor; the same number elsewhere on the page
    still is. A fragment belongs to the filename when line breaks join it to the tail and it,
    or a fragment before it, ends in `_`, `-` or `.`, or the tail starts with one: without
    that sign `Area 1204\\ngrid.tif` keeps 1204 as a value.
    """
    occurrences = _filename_occurrences(body_text)
    anchors: dict[frozenset[str], FilenameAnchor] = {}
    for occurrence in occurrences:
        anchors.setdefault(occurrence.anchor.accepted, occurrence.anchor)
    return tuple(anchors.values()), _numbers_outside(body_text, occurrences)


def _unmatched_filenames(
    required: Sequence[FilenameAnchor], emitted: Sequence[_FilenameOccurrence]
) -> list[FilenameAnchor]:
    """The required anchors left without an emitted filename by a maximum matching in which
    each emitted occurrence satisfies at most one anchor that accepts one of its forms.

    Each anchor in turn searches breadth first for an augmenting path: a free accepting
    occurrence, reached through occurrences whose owners can move to another. The search
    keeps its own queue, so a page with thousands of filenames that share a tail cannot
    exhaust the call stack."""
    accepts = [
        [j for j, occurrence in enumerate(emitted) if occurrence.forms & anchor.accepted]
        for anchor in required
    ]
    owner: list[int | None] = [None] * len(emitted)
    owned: list[int | None] = [None] * len(required)
    unmatched: list[FilenameAnchor] = []
    for i, anchor in enumerate(required):
        # reached[j] is the anchor whose search step reached occurrence j.
        reached: dict[int, int] = {}
        queue = deque([i])
        free: int | None = None
        while queue and free is None:
            current = queue.popleft()
            for j in accepts[current]:
                if j in reached:
                    continue
                reached[j] = current
                holder = owner[j]
                if holder is None:
                    free = j
                    break
                queue.append(holder)
        if free is None:
            unmatched.append(anchor)
            continue
        # Move each anchor on the path to the occurrence it reached, back to anchor i, the
        # only anchor on it that owned nothing.
        occurrence = free
        while True:
            moving, previous = reached[occurrence], owned[reached[occurrence]]
            owner[occurrence], owned[moving] = moving, occurrence
            if previous is None:
                break
            occurrence = previous
    return unmatched


def missing_anchors(
    page_number: int,
    extraction: PageExtraction,
    filenames: Sequence[FilenameAnchor],
    numbers: set[str],
) -> list[str]:
    """Anchors absent from the exact text `assemble_sections` emits for this page: each
    missing filename by its `display`, then each missing number.

    The emitted text is tokenized exactly as `fidelity_anchors` tokenizes native text, so
    a verbatim transcription has no missing anchors and a signed value never satisfies
    its unsigned counterpart. Each emitted filename occurrence satisfies one filename anchor
    at most, so a filename emitted once cannot stand in for two anchors that both accept it.
    Numbers are also read with filenames left in place, so a number that native text split
    from a line-wrapped filename still matches the rejoined filename.
    """
    emitted = "\n".join(part.markdown for part in page_parts(page_number, extraction))
    emitted = emitted.replace("\\", "")
    occurrences = _filename_occurrences(emitted)
    emitted_numbers = _numbers_outside(emitted, occurrences)
    emitted_numbers |= _number_anchors(_normalize_numbers(emitted))
    missing = sorted(a.display for a in _unmatched_filenames(filenames, occurrences))
    return missing + sorted(numbers - emitted_numbers)


def _fidelity_ok(
    missing: list[str], filenames: Sequence[FilenameAnchor], total: int, threshold: float
) -> bool:
    if any(token == anchor.display for anchor in filenames for token in missing):
        return False
    return total == 0 or (total - len(missing)) / total >= threshold


@dataclass(frozen=True)
class FidelityAttempt:
    """One row of the fidelity gate: the effort a transcription runs at and the recall it needs.

    `few_anchor_threshold` applies to pages with one or two anchors.
    """

    effort: Effort
    threshold: float
    few_anchor_threshold: float


# The first attempt runs at low effort; the retry runs at high effort and names the values the
# first attempt lost. A missing filename fails either attempt.
FIDELITY_ATTEMPTS = (
    FidelityAttempt(effort="low", threshold=0.95, few_anchor_threshold=1.0),
    FidelityAttempt(effort="high", threshold=0.85, few_anchor_threshold=0.85),
)


def transcribe_with_fidelity(
    gateway: LanguageGateway,
    *,
    rel_path: str,
    page_number: int,
    native_text: str,
    body_text: str,
    webp_bytes: bytes,
    prev_context: str | None,
) -> PageExtraction:
    filenames, numbers = fidelity_anchors(body_text)
    total = len(filenames) + len(numbers)
    missing: list[str] = []
    for attempt in FIDELITY_ATTEMPTS:
        extraction = gateway.run(
            PageTranscription(
                native_text=native_text,
                image_webp=webp_bytes,
                prev_context=prev_context,
                effort=attempt.effort,
                missing_values=tuple(missing),
            )
        )
        missing = missing_anchors(page_number, extraction, filenames, numbers)
        threshold = attempt.threshold if total >= 3 else attempt.few_anchor_threshold
        if _fidelity_ok(missing, filenames, total, threshold):
            return extraction
    raise TranscriptionFidelityError(
        f"{rel_path} page {page_number}: transcript is missing {', '.join(missing)}"
    )


# Section assembly.

PartKind = Literal["heading", "paragraph", "table", "callout", "visual"]


@dataclass
class Part:
    """One block of a section. A table also has `row_spans`: the (first, last) page of each
    line of its Markdown, header and separator lines included, so a slice of the table
    carries the pages of its own rows; every other kind has None."""

    kind: PartKind
    markdown: str
    page_start: int
    page_end: int
    row_spans: list[tuple[int, int]] | None = None


@dataclass
class SectionDraft:
    heading_path: str
    level: int
    page_start: int
    page_end: int
    parts: list[Part] = field(default_factory=list)

    def add(self, part: Part) -> None:
        self.parts.append(part)
        self.page_end = max(self.page_end, part.page_end)

    @property
    def markdown(self) -> str:
        return "\n\n".join(part.markdown for part in self.parts)


def _cells(line: str) -> list[str]:
    body = line.strip().removeprefix("|")
    if not body.endswith("\\|"):
        body = body.removesuffix("|")
    return [cell.strip() for cell in _TABLE_CELL_SPLIT.split(body)]


def _table_rows(markdown: str) -> list[list[str]]:
    """Cells of every row of a normalized table (one row per line)."""
    return [_cells(line) for line in markdown.strip().splitlines()]


def _is_separator(row: list[str]) -> bool:
    return bool(row) and all(_SEPARATOR_CELL.fullmatch(cell.replace(" ", "")) for cell in row)


def _render_rows(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def split_table_block(markdown: str) -> list[tuple[PartKind, str]]:
    """Normalize a table block; caption or footnote lines around the table stay as prose.

    The table is the first run of lines containing `|`. It counts as a table when its
    lines carry outer pipes or its second line is a separator row (GFM without pipes).
    """
    lines = [line.rstrip() for line in markdown.strip().splitlines()]
    has_pipe = [bool(_TABLE_CELL_SPLIT.search(line)) for line in lines]
    start = has_pipe.index(True) if True in has_pipe else len(lines)
    end = start
    while end < len(lines) and has_pipe[end]:
        end += 1
    run = lines[start:end]
    is_table = bool(run) and (
        all(line.lstrip().startswith("|") for line in run)
        or (len(run) >= 2 and _is_separator(_cells(run[1])))
    )
    if not is_table:
        return [("paragraph", markdown.strip())] if markdown.strip() else []
    parts: list[tuple[PartKind, str]] = []
    if before := "\n".join(lines[:start]).strip():
        parts.append(("paragraph", before))
    parts.append(("table", _render_rows([_cells(line) for line in run])))
    if after := "\n".join(lines[end:]).strip():
        parts.append(("paragraph", after))
    return parts


def _header_count(rows: list[list[str]]) -> int:
    """How many leading rows of a table are its header: the header and separator rows, or
    none when the second row is not a separator."""
    return 2 if len(rows) >= 2 and _is_separator(rows[1]) else 0


def _table_part(markdown: str, page: int) -> Part:
    rows = _table_rows(markdown)
    return Part("table", markdown, page, page, [(page, page)] * len(rows))


def _extend_table(
    table: Part, continuation: str, merges_into_previous_row: bool, page: int
) -> None:
    """Append a continued table's rows to `table`, dropping a repeated header. A row cut by
    the page break and merged back spans both pages; every appended row spans `page`."""
    assert table.row_spans is not None, "only a table part is extended"
    rows, cont_rows = _table_rows(table.markdown), _table_rows(continuation)
    if len(cont_rows) >= 2 and _is_separator(cont_rows[1]):
        cont_rows = cont_rows[2:]
    spans = table.row_spans
    last = rows[-1] if rows else None
    if (
        merges_into_previous_row
        and cont_rows
        and last is not None
        and not _is_separator(last)
        and len(last) == len(cont_rows[0])
    ):
        rows[-1] = [" ".join(t for t in pair if t) for pair in zip(last, cont_rows[0], strict=True)]
        spans[-1] = (spans[-1][0], page)
        cont_rows = cont_rows[1:]
    rows += cont_rows
    spans += [(page, page)] * len(cont_rows)
    table.markdown = _render_rows(rows)
    table.page_end = page


def _heading_text(markdown: str) -> str:
    return markdown.strip().lstrip("#").strip()


def _heading_level(block: PageBlock) -> int:
    return block.heading_level or 2


def _block_parts(block: PageBlock, page: int) -> list[Part]:
    if block.kind == "heading":
        text = f"{'#' * _heading_level(block)} {_heading_text(block.markdown)}"
        return [Part("heading", text, page, page)]
    if block.kind == "table":
        return [
            _table_part(text, page) if kind == "table" else Part(kind, text, page, page)
            for kind, text in split_table_block(block.markdown)
        ]
    text = block.markdown.strip()
    return [Part(block.kind, text, page, page)] if text else []


def _visual_part(page: int, label: str, transcript: str) -> Part:
    quoted = transcript.strip().replace("\n", "\n> ")
    text = f"> **[Visual \u2014 Page {page}: {label}]**\n> {quoted}"
    return Part("visual", text, page, page)


def page_parts(page: int, extraction: PageExtraction) -> list[Part]:
    """Every part a page contributes, in the exact form sections will contain."""
    parts = [part for block in extraction.blocks for part in _block_parts(block, page)]
    return parts + [_visual_part(page, v.label, v.transcript) for v in extraction.visuals]


def assemble_sections(
    pages: Sequence[tuple[int, PageExtraction]],
) -> list[SectionDraft]:
    """Stitch transcribed pages into heading-delimited sections of at most MAX_SECTION_CHARS
    (`split_oversized_section`)."""
    sections: list[SectionDraft] = []
    stack: dict[int, str] = {}

    def active(page: int) -> SectionDraft:
        if not sections:
            sections.append(SectionDraft("Overview", 1, page, page))
        return sections[-1]

    for page_number, extraction in pages:
        for index, block in enumerate(extraction.blocks):
            if block.kind == "heading":
                level, text = _heading_level(block), _heading_text(block.markdown)
                stack = {lvl: name for lvl, name in stack.items() if lvl < level}
                stack[level] = text
                path = " > ".join(stack[lvl] for lvl in sorted(stack))
                section = SectionDraft(path, level, page_number, page_number)
                section.parts += _block_parts(block, page_number)
                sections.append(section)
                continue
            parts = _block_parts(block, page_number)
            if not parts:
                continue
            section = active(page_number)
            previous = next((p for p in reversed(section.parts) if p.kind != "visual"), None)
            first = parts[0]
            if (
                index == 0
                and block.continues_from_previous_page
                and previous is not None
                and first.kind == previous.kind
                and first.kind in ("table", "paragraph")
            ):
                if first.kind == "table":
                    _extend_table(
                        previous, first.markdown, block.merges_into_previous_row, page_number
                    )
                else:
                    previous.markdown = f"{previous.markdown} {first.markdown}"
                    previous.page_end = page_number
                section.page_end = max(section.page_end, page_number)
                parts = parts[1:]
            for part in parts:
                section.add(part)
        for visual in extraction.visuals:
            active(page_number).add(_visual_part(page_number, visual.label, visual.transcript))
    return [piece for section in sections for piece in split_oversized_section(section)]


def _markdown_blocks(text: str) -> list[tuple[str, list[str]]]:
    """Group lines into (kind, lines); a fenced code block stays one literal paragraph."""
    groups: list[tuple[str, list[str]]] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if fence is not None:
            groups[-1][1].append(line)
            if _closes_fence(stripped, fence):
                fence = None
            continue
        if opening := _fence_opener(stripped):
            fence = opening[0]
            groups.append(("code", [line]))
            continue
        if not stripped:
            kind = "blank"
        elif _MARKDOWN_HEADING.fullmatch(stripped):
            kind = "heading"
        else:
            kind = {"|": "table", ">": "callout"}.get(stripped[0], "paragraph")
        if groups and groups[-1][0] == kind and kind not in ("heading", "code"):
            groups[-1][1].append(line)
        else:
            groups.append((kind, [line]))
    return groups


def parse_markdown_blocks(text: str) -> PageExtraction:
    """Deterministically split Markdown into heading, paragraph, table and callout blocks."""
    blocks: list[PageBlock] = []
    for kind, lines in _markdown_blocks(text):
        if kind == "heading":
            match = _MARKDOWN_HEADING.fullmatch(lines[0].strip())
            assert match is not None
            blocks.append(_block("heading", match[2], min(len(match[1]), 4)))
        elif kind == "code":
            blocks.append(_block("paragraph", "\n".join(lines)))
        elif kind != "blank":
            blocks.append(_block(kind, "\n".join(line.rstrip() for line in lines)))
    return PageExtraction(blocks=blocks, visuals=[])


def _block(kind: str, markdown: str, heading_level: int = 0) -> PageBlock:
    return PageBlock.model_validate(
        {
            "kind": kind,
            "heading_level": heading_level,
            "continues_from_previous_page": False,
            "merges_into_previous_row": False,
            "markdown": markdown,
        }
    )


# Chunking.


def wrap_text(text: str, limit: int, *, first_limit: int | None = None) -> list[str]:
    """Split text into pieces of at most `limit` characters at line breaks, else spaces; the
    first piece holds at most `first_limit` when it is given. Both limits must be at least 1,
    so every piece holds at least one character."""
    if limit < 1 or (first_limit is not None and first_limit < 1):
        raise ValueError(f"wrap limits must be positive: limit={limit}, first_limit={first_limit}")
    pieces: list[str] = []
    rest = text
    while len(rest) > (cap := limit if pieces or first_limit is None else first_limit):
        cut = rest.rfind("\n", 1, cap + 1)
        if cut <= 0:
            cut = rest.rfind(" ", 1, cap + 1)
        if cut <= 0:
            pieces.append(rest[:cap])
            rest = rest[cap:]
        else:
            # The line break or space at the cut is the only character dropped.
            pieces.append(rest[:cut])
            rest = rest[cut + 1 :]
    return [*pieces, rest] if rest else pieces


def _split_prose(text: str) -> list[str]:
    """Split at sentence ends; multi-line text (lists, callouts, code) splits at lines."""
    if len(text) <= PROSE_CHUNK_CHARS:
        return [text]
    if "\n" in text:
        return wrap_text(text, PROSE_CHUNK_CHARS)
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        for piece in wrap_text(sentence, PROSE_CHUNK_CHARS):
            if current and len(current) + 1 + len(piece) > PROSE_CHUNK_CHARS:
                pieces.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    return [*pieces, current] if current else pieces


def _wrap_cell(text: str, width: int) -> list[str]:
    """A cell's text in pieces of at most `width` characters, cut at spaces where it can be.
    A piece never ends in a backslash: the whole run of backslashes moves to the next piece,
    so an escaped pipe (`\\|`) is never cut from its backslash into a column break."""
    pieces: list[str] = []
    start = 0
    while len(text) - start > width:
        # wrap_text reads no further than one character past the cap, so each step reads only
        # that window and wrapping a long cell stays linear.
        window = text[start : start + width + 1]
        piece = wrap_text(window, width)[0]
        # wrap_text drops the space it cut at; a hard cut drops nothing.
        skip = 1 if window[len(piece)] == " " else 0
        trailing = len(piece) - len(piece.rstrip("\\"))
        if trailing == len(piece):
            # Only backslashes fit: cut after an even number of them, so each stays paired.
            piece, skip = piece[: max(1, len(piece) - len(piece) % 2)], 0
        elif trailing:
            piece, skip = piece[:-trailing], 0
        pieces.append(piece)
        start += len(piece) + skip
    return [*pieces, text[start:]]


def _wrap_row(line: str, room: int) -> list[str]:
    """A table row of at most `room` characters as itself; a longer one as consecutive rows
    with the same columns, each cell's text wrapped down them, so every piece stays a table
    row of that table. A row with too many columns for each cell to get two characters is cut
    as text: a one-character cell would cut an escaped pipe (`\\|`) from its backslash."""
    if len(line) <= room:
        return [line]
    cells = _cells(line)
    overhead = len(_render_rows([[""] * len(cells)]))
    width = (room - overhead) // len(cells)
    if width < 2:
        return wrap_text(line, room)
    wrapped = [_wrap_cell(cell, width) for cell in cells]
    return [
        _render_rows([[w[i] if i < len(w) else "" for w in wrapped]])
        for i in range(max(map(len, wrapped)))
    ]


def _table_slices(table: Part, first: int, rest: int) -> list[Part]:
    """`table` cut between rows into tables of at most `first` characters, then `rest`, each
    repeating the header and separator rows and spanning the pages of its own rows.

    A row too long for a slice is wrapped cell by cell into rows of the same columns
    (`_wrap_row`). A header that would leave a row less than a quarter of `rest` is not
    repeated: its lines are cut like rows. When the header and the first row do not fit in
    `first`, the first slice is sized to `rest` instead, so the caller starts it in a new
    piece; every slice stays within `rest`.
    """
    lines = table.markdown.splitlines()
    spans = table.row_spans or [(table.page_start, table.page_end)] * len(lines)
    header_count = _header_count(_table_rows(table.markdown))
    if len("\n".join(lines[:header_count])) + 1 + rest // 4 > rest:
        header_count = 0
    header = lines[:header_count]
    base = len("\n".join(header))
    room = rest - base - 1 if header else rest
    rows = [
        (row, span)
        for line, span in zip(lines[header_count:], spans[header_count:], strict=True)
        for row in _wrap_row(line, room)
    ]
    if not rows:
        return [table]
    if base + (1 if header else 0) + len(rows[0][0]) > first:
        first = rest
    slices: list[Part] = []
    current: list[tuple[str, tuple[int, int]]] = []

    def flush() -> None:
        row_spans = [span for _, span in current]
        start, end = min(span[0] for span in row_spans), max(span[1] for span in row_spans)
        # The header lines take the pages of the slice's rows, as the slice itself does, so a
        # later cut of a slice whose rows are all on page 2 never reaches back to page 1.
        slices.append(
            Part(
                "table",
                "\n".join([*header, *(line for line, _ in current)]),
                start,
                end,
                [(start, end)] * len(header) + row_spans,
            )
        )
        current.clear()

    size = base
    for row, span in rows:
        if current and size + 1 + len(row) > (rest if slices else first):
            flush()
            size = base
        size += len(row) + (1 if header or current else 0)
        current.append((row, span))
    flush()
    return slices


def _fence_opener(stripped: str) -> tuple[str, str] | None:
    """The delimiter of a stripped code fence opener line and the line that reopens its block
    after a cut (the delimiter and a short info word), or None for any other line."""
    match = _FENCE_OPENER.fullmatch(stripped)
    if match is None:
        return None
    delimiter = match[1] or match[2]
    words = stripped[len(delimiter) :].split()
    info = words[0] if words and len(words[0]) <= FENCE_INFO_CHARS else ""
    return delimiter, delimiter + info


def _closes_fence(stripped: str, delimiter: str) -> bool:
    return stripped.startswith(delimiter) and not stripped.strip(delimiter[0])


def _fence_slices(text: str, first: int, rest: int) -> list[str]:
    """Text cut at line breaks (else spaces) into pieces of at most `first` characters, then
    `rest`. A fenced code block cut in two is closed at the end of one piece and reopened at
    the start of the next with its delimiter and short info word (`_fence_opener`), so every
    piece has balanced fences and the text after the block is never read as code.

    A delimiter so long that `first` cannot hold it twice beside text gives a first piece
    sized to `rest` instead, so the caller starts it in a new piece. One so long that `rest`
    cannot either leaves the text cut as plain text, its fences unrepaired."""
    openers = [o for line in text.splitlines() if (o := _fence_opener(line.strip()))]
    # A cut block gains its reopening line and a closing delimiter, each on its own line.
    reserve = max((len(reopen) + len(delimiter) + 2 for delimiter, reopen in openers), default=0)
    if reserve >= rest:
        return wrap_text(text, rest, first_limit=first)
    pieces = wrap_text(
        text, rest - reserve, first_limit=first - reserve if first > reserve else None
    )
    if not openers:
        return pieces
    repaired: list[str] = []
    reopen: str | None = None
    for piece in pieces:
        if reopen is not None:
            piece = f"{reopen}\n{piece}"
        fence: str | None = None
        for line in piece.splitlines():
            stripped = line.strip()
            if fence is not None:
                if _closes_fence(stripped, fence):
                    fence = None
            elif opening := _fence_opener(stripped):
                fence, reopen = opening
        if fence is not None:
            piece = f"{piece}\n{fence}"
        else:
            reopen = None
        repaired.append(piece)
    return repaired


def _visual_slices(visual: str, page: int, first: int, rest: int) -> list[str]:
    """A visual transcript cut at line breaks into pieces of at most `first` characters, then
    `rest`, each opening with the transcript's `> **[Visual ...]**` line and quoting every
    line, so each piece is read as a visual transcript. A title longer than a quarter of a
    piece becomes the first quoted line, and each piece opens with `> **[Visual - Page N]**`
    instead."""
    title, *lines = visual.splitlines()
    limit = min(first, rest)
    if len(title) > limit // 4:
        title, lines = f"> **[Visual \u2014 Page {page}]**", [title, *lines]
    room = limit - len(title) - 1
    quoted = [
        f"> {piece}"
        for line in lines
        for piece in (wrap_text(line.removeprefix(">").strip(), room - 2) or [""])
    ]
    pieces: list[list[str]] = [[]]
    size = len(title)
    for line in quoted:
        if pieces[-1] and size + 1 + len(line) > (rest if len(pieces) > 1 else first):
            pieces.append([])
            size = len(title)
        pieces[-1].append(line)
        size += 1 + len(line)
    return ["\n".join([title, *piece]) for piece in pieces]


def _slices(part: Part, first: int, rest: int) -> list[Part]:
    """`part` as consecutive parts of at most `first` characters, then `rest`: a table cut
    between rows under its repeated header (`_table_slices`, whose first slice can take
    `rest`), a visual transcript under its repeated title, and other text at line breaks or
    spaces with every fenced code block kept balanced (`_fence_slices`, whose first slice
    can also take `rest`). `first` is at least
    MIN_FIRST_SLICE_CHARS."""
    if len(part.markdown) <= first:
        return [part]
    if part.kind == "table":
        return _table_slices(part, first, rest)
    texts = (
        _visual_slices(part.markdown, part.page_start, first, rest)
        if part.kind == "visual"
        else _fence_slices(part.markdown, first, rest)
    )
    return [Part(part.kind, text, part.page_start, part.page_end) for text in texts]


def _bounded_parts(parts: Sequence[Part], first: int) -> list[Part]:
    """`parts` with each one larger than MAX_SECTION_CHARS sliced (`_slices`), the first slice
    of the first part holding at most `first` where it can. A visual transcript that follows
    a sliced table, as the visuals of a table's pages do once the table runs across them, is
    sliced the same way and moves up to follow the slice that holds its page."""
    bounded: list[Part] = []
    pending: list[Part] = []
    room = first
    for part in parts:
        if part.kind == "visual" and pending:
            # The slice before the first that ends on or after the visual's page.
            target = next(
                (
                    i
                    for i, s in enumerate(pending)
                    if s.kind == "table" and s.page_end >= part.page_start
                ),
                len(pending) - 1,
            )
            at = _after_visuals(pending, target)
            pending[at:at] = _slices(part, MAX_SECTION_CHARS, MAX_SECTION_CHARS)
            continue
        bounded += pending
        pending = []
        sliced = _slices(part, room, MAX_SECTION_CHARS)
        room = MAX_SECTION_CHARS
        if part.kind == "table" and len(sliced) > 1:
            pending = sliced
        else:
            bounded += sliced
    return bounded + pending


def _after_visuals(parts: list[Part], index: int) -> int:
    """The position after `parts[index]` and the visual parts that directly follow it."""
    index += 1
    while index < len(parts) and parts[index].kind == "visual":
        index += 1
    return index


def split_oversized_section(draft: SectionDraft) -> list[SectionDraft]:
    """`draft` as consecutive sections of at most MAX_SECTION_CHARS of Markdown, cut between
    parts after an oversized part is sliced (`_bounded_parts`). Every piece keeps the heading
    path and level, and only the first holds the heading line. The first slice of the part
    after the heading is sized to fit beside it, so the first piece holds the start of the
    body, unless less than MIN_FIRST_SLICE_CHARS is left beside the heading: then the body
    starts in the next piece. A heading longer than a section is cut as text over pieces."""
    if len(draft.markdown) <= MAX_SECTION_CHARS:
        return [draft]
    lead = draft.parts[0] if draft.parts and draft.parts[0].kind == "heading" else None
    body = draft.parts[1:] if lead else draft.parts
    pieces: list[SectionDraft] = []
    size = 0
    for heading in _slices(lead, MAX_SECTION_CHARS, MAX_SECTION_CHARS) if lead else []:
        pieces.append(
            SectionDraft(draft.heading_path, draft.level, heading.page_start, heading.page_end)
        )
        pieces[-1].add(heading)
        size = len(heading.markdown)
    room = MAX_SECTION_CHARS - size - 2 if pieces else MAX_SECTION_CHARS
    for part in _bounded_parts(body, room if room >= MIN_FIRST_SLICE_CHARS else MAX_SECTION_CHARS):
        if pieces and size + 2 + len(part.markdown) <= MAX_SECTION_CHARS:
            pieces[-1].add(part)
            pieces[-1].page_start = min(pieces[-1].page_start, part.page_start)
            size += 2 + len(part.markdown)
            continue
        piece = SectionDraft(draft.heading_path, draft.level, part.page_start, part.page_end)
        piece.add(part)
        pieces.append(piece)
        size = len(part.markdown)
    return pieces


def section_pieces(draft: SectionDraft, prose_kind: ChunkKind) -> tuple[ChunkPiece, ...]:
    """The chunks of one section, without IDs or breadcrumbs: packed prose, table pieces and
    one piece per visual, each with its own page span (a table piece, that of its rows)."""
    pieces: list[tuple[ChunkKind, str, int, int]] = []
    prose: list[tuple[str, int, int]] = []

    def flush() -> None:
        if prose:
            text = "\n\n".join(t for t, _, _ in prose)
            first, last = min(p[1] for p in prose), max(p[2] for p in prose)
            pieces.append((prose_kind, text, first, last))
            prose.clear()

    for part in draft.parts:
        span = (part.page_start, part.page_end)
        if part.kind == "table":
            flush()
            tables = (
                [part]
                if len(part.markdown) <= WHOLE_TABLE_CHARS
                else _table_slices(part, TABLE_PIECE_CHARS, TABLE_PIECE_CHARS)
            )
            pieces += [("table", t.markdown, t.page_start, t.page_end) for t in tables]
        elif part.kind == "visual":
            flush()
            pieces.append(("visual", part.markdown, *span))
        elif part.kind != "heading":
            for text in _split_prose(part.markdown):
                packed = "\n\n".join([*(t for t, _, _ in prose), text])
                if prose and len(packed) > PROSE_CHUNK_CHARS:
                    flush()
                prose.append((text, *span))
    flush()
    return tuple(
        ChunkPiece(kind=kind, content=piece, page_start=first, page_end=last)
        for kind, content, first, last in pieces
        for piece in wrap_text(content, MAX_CHUNK_CHARS)
    )


# Reading.


def _transcribe_pages(
    gateway: LanguageGateway, source: SourceFile, rendered: Sequence[tuple[str, bytes]]
) -> list[tuple[int, PageExtraction]]:
    furniture = find_furniture([text for text, _ in rendered])
    results: list[tuple[int, PageExtraction]] = []
    prev_context: str | None = None
    for index, (native_text, webp) in enumerate(rendered):
        number = index + 1
        extraction = transcribe_with_fidelity(
            gateway,
            rel_path=source.rel_path,
            page_number=number,
            native_text=native_text,
            body_text=strip_furniture(native_text, furniture, index),
            webp_bytes=webp,
            prev_context=prev_context,
        )
        results.append((number, extraction))
        # Continuation flags use the preceding page's block tail. Changes propagate to the
        # following page only if this tail changes, and stop when a later tail is unchanged.
        prev_context = "\n\n".join(b.markdown for b in extraction.blocks)[-PREV_CONTEXT_CHARS:]
    return results


def _visual_summary(extraction: PageExtraction) -> str:
    lines = [
        f"Heading: {_heading_text(b.markdown)}" for b in extraction.blocks if b.kind == "heading"
    ]
    lines += [f"Visual: {v.label}: {v.transcript}" for v in extraction.visuals]
    return "\n".join(lines)


def observation_batches(sections: Sequence[ReadSection]) -> list[tuple[BatchSection, ...]]:
    """Sections packed in order into batches of at most `OBSERVE_BATCH_CHARS` characters.

    A section larger than the budget is split by `wrap_text` into pieces within it; every
    piece keeps the section's document-local ordinal, so a quote found in a piece is located
    in its section. Each piece is labelled with its heading path cut by `heading_context`.
    """
    batches: list[list[BatchSection]] = []
    size = 0
    for section in sections:
        label = heading_context(section.heading_path)
        # An empty section is one empty piece, shown as before.
        for piece in wrap_text(section.markdown, OBSERVE_BATCH_CHARS) or [section.markdown]:
            item = BatchSection(section.ordinal, label, section.page_start, section.page_end, piece)
            if batches and size + len(piece) <= OBSERVE_BATCH_CHARS:
                batches[-1].append(item)
                size += len(piece)
            else:
                batches.append([item])
                size = len(piece)
    return [tuple(batch) for batch in batches]


def locate_quote(quote: str, sections: Sequence[BatchSection]) -> int | None:
    """Ordinal of the first section whose text contains `quote`, compared by `fold_for_quote`."""
    needle = fold_for_quote(quote)
    if not needle:
        return None
    for section in sections:
        if needle in fold_for_quote(section.markdown):
            return section.ordinal
    return None


@dataclass(frozen=True)
class LocatedBatch:
    """One batch's observations with every quote located; `discarded` counts dropped ones."""

    draft: ObservationDraft
    terms: tuple[TermObservation, ...]
    craft: tuple[CraftObservation, ...]
    discarded: int


def locate_batch(
    draft: ObservationDraft, sections: Sequence[BatchSection], *, rerun_of_unlocated: bool
) -> Checked[LocatedBatch]:
    """Locate every quote of a batch in that batch's sections.

    Unlocated quotes are rejections. Keeping the located observations and discarding the rest
    is an accepted reduction, except when the batch quoted something and nothing could be
    located: that is a broken reading. With `rerun_of_unlocated` (the earlier answer quoted
    and located nothing), a rerun without any observation is that same broken reading.
    """
    if rerun_of_unlocated and not (draft.terms or draft.craft):
        return Checked.reject(
            [
                "the earlier answer quoted the sections but none of its quotes could be located,"
                " and this answer has no observations"
            ]
        )
    terms: list[TermObservation] = []
    craft: list[CraftObservation] = []
    unlocated: list[str] = []
    for term in draft.terms:
        ordinal = locate_quote(term.quote, sections)
        if ordinal is None:
            unlocated.append(term.quote)
            continue
        terms.append(
            TermObservation(
                term=term.term,
                kind=term.kind,
                aliases=tuple(term.aliases),
                quote=term.quote,
                defines=term.defines,
                section_ordinal=ordinal,
            )
        )
    for item in draft.craft:
        ordinal = locate_quote(item.quote, sections)
        if ordinal is None:
            unlocated.append(item.quote)
            continue
        craft.append(
            CraftObservation(
                aspect=item.aspect,
                statement=item.statement,
                quote=item.quote,
                section_ordinal=ordinal,
            )
        )
    located = LocatedBatch(draft, tuple(terms), tuple(craft), len(unlocated))
    if not unlocated:
        return Checked.accept(located)
    rejections = tuple(f"quote not found verbatim in the sections: {q!r}" for q in unlocated)
    if terms or craft:
        return Checked.reduce(rejections, located)
    return Checked.reject(rejections)


def observe_document(
    gateway: LanguageGateway, rel_path: str, sections: Sequence[ReadSection]
) -> tuple[ObservationCard, int]:
    """The observation card of a citable document, and the count of discarded observations."""
    batches = observation_batches(sections)
    if not batches:
        raise ConversionError(f"{rel_path}: the document has no text to read")

    def observe(batch: tuple[BatchSection, ...]) -> LocatedBatch:
        nothing_located = False

        def check(draft: ObservationDraft) -> Checked[LocatedBatch]:
            nonlocal nothing_located
            checked = locate_batch(draft, batch, rerun_of_unlocated=nothing_located)
            nothing_located = checked.result is None
            return checked

        return generate_checked(
            f"{rel_path}: observations",
            lambda feedback: gateway.run(ObserveSections(batch, feedback)),
            check,
        )

    located = [observe(batch) for batch in batches]
    attributes: dict[tuple[str, str], StatedAttribute] = {}
    for batch in located:
        for attribute in batch.draft.attributes:
            key = (fold_text(attribute.name).strip(), fold_text(attribute.value).strip())
            attributes.setdefault(key, attribute)
    headers = tuple(
        CardHeader(
            title=b.draft.title,
            doc_date=printed_date_or_none(b.draft.doc_date),
            deliverable_type=b.draft.deliverable_type,
            summary=b.draft.summary,
        )
        for b in located
    )
    header = headers[0] if len(headers) == 1 else gateway.run(ConsolidateCard(headers))
    card = ObservationCard(
        title=header.title,
        doc_date=consolidated_date(headers, header.doc_date),
        deliverable_type=header.deliverable_type,
        summary=header.summary,
        attributes=tuple(attributes.values()),
        terms=tuple(t for b in located for t in b.terms),
        craft=tuple(c for b in located for c in b.craft),
    )
    return card, sum(b.discarded for b in located)


def consolidated_date(parts: Sequence[CardHeader], answer: str | None) -> str | None:
    """The printed date of a document read in `parts`, given the consolidation's `answer`.

    The date is always one the parts reported as printed, as that part wrote it: the reported
    date the answer names, else the first reported. The answer names a reported date when it
    is the same text, else the same text once folded, else a different form of the same
    unambiguous date (`19th May 2025` for `19ᵗʰ May 2025`); each test runs over every reported
    date before the next, so an exact copy wins over an earlier part's other form of it. A
    placeholder such as `none` is not a reported date, and with none reported the document
    prints no date, whatever the answer.
    Consolidation can therefore neither invent a date, such as an ISO form of an ambiguous
    printed date, nor drop a printed one, which would let the metadata date stand in for it.
    """
    reported = [d for part in parts if (d := printed_date_or_none(part.doc_date)) is not None]
    if not reported:
        return None
    chosen = printed_date_or_none(answer)
    if chosen is None:
        return reported[0]
    if chosen in reported:
        return chosen
    folded = fold_text(chosen)
    if (same := next((r for r in reported if fold_text(r) == folded), None)) is not None:
        return same
    chosen_iso = parse_printed_date(chosen)
    if chosen_iso is None:
        return reported[0]
    return next((r for r in reported if parse_printed_date(r) == chosen_iso), reported[0])


def curator_note_card(text: str) -> ObservationCard:
    """A curator note's card, written without a model call."""
    """The title is the first line with text once heading marks are removed; the summary is
    the first line that is not a heading."""
    lines = [line.strip() for line in text.splitlines()]
    titles = [t for line in lines if (t := line.lstrip("#").strip())]
    prose = [line for line in lines if line and not line.startswith("#")]
    return ObservationCard(
        title=titles[0] if titles else "Curator note",
        doc_date=None,
        deliverable_type="curator note",
        summary=" ".join(prose[0].split())[:300] if prose else "",
        attributes=(),
        terms=(),
        craft=(),
    )


# Document dates.

# A whole PDF date string, `D:YYYYMM[DD[HH[mm[SS]]]]` (the `D:` prefix is optional in
# practice); a `Z`, `+HH'mm'` or `-HH:mm` offset may follow only an hour. The calendar date is
# read once every clock field is in range (`_metadata_iso`).
_PDF_DATE = re.compile(
    r"(?:D:)?(?P<year>\d{4})(?P<month>\d{2})(?:(?P<day>\d{2})"
    r"(?:(?P<hour>\d{2})(?:(?P<minute>\d{2})(?P<second>\d{2})?)?"
    r"(?:Z(?:00'?(?:00'?)?)?|[+-](?P<tz_hour>\d{2})(?:[':]?(?P<tz_minute>\d{2}))?'?)?)?)?",
    re.ASCII,
)
# A whole W3CDTF value as `docProps/core.xml` writes it: `YYYY-MM`, or `YYYY-MM-DD` with an
# optional time and offset; the calendar date is read once every clock field is in range.
_W3C_DATE = re.compile(
    r"(?P<year>\d{4})-(?P<month>\d{2})(?:-(?P<day>\d{2})"
    r"(?:T(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2})(?:\.\d+)?)?"
    r"(?:Z|[+-](?P<tz_hour>\d{2}):(?P<tz_minute>\d{2}))?)?)?",
    re.ASCII,
)
# The largest value of each clock field of a metadata timestamp.
_CLOCK_LIMITS = {"hour": 23, "minute": 59, "second": 59, "tz_hour": 23, "tz_minute": 59}
_CORE_PROPERTIES = "docProps/core.xml"
# The PDF info dictionary keys read for a metadata date, in order of preference.
_PDF_DATE_KEYS = ("CreationDate", "ModDate")
# Answers a model gives for a document date when nothing is printed, compared by `fold_text`.
_NO_DATE_ANSWERS = frozenset(
    {"none", "null", "n/a", "na", "nil", "not stated", "not given", "no date", "unknown", "undated"}
)
_DCTERMS = "{http://purl.org/dc/terms/}"


def _office_core_properties(source: SourceFile, data: bytes) -> ElementTree.Element | None:
    """The parsed `docProps/core.xml` of a Word or PowerPoint file, or None without one."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            try:
                info = archive.getinfo(_CORE_PROPERTIES)
            except KeyError:
                return None
            if info.file_size > MAX_CORE_PROPERTIES_BYTES:
                raise ConversionError(
                    f"{source.rel_path}: {_CORE_PROPERTIES} is {info.file_size} bytes; the"
                    f" limit is {MAX_CORE_PROPERTIES_BYTES}"
                )
            return ElementTree.fromstring(archive.read(info))
    except (
        OSError,
        zipfile.BadZipFile,
        UnicodeDecodeError,
        RuntimeError,
        NotImplementedError,
        EOFError,
        zlib.error,
        ElementTree.ParseError,
    ) as exc:
        raise ConversionError(f"{source.rel_path}: cannot read {_CORE_PROPERTIES}: {exc}") from exc


def metadata_date(source: SourceFile, data: bytes) -> str | None:
    """The date a file's own metadata carries, `YYYY-MM-DD` or `YYYY-MM`, read from its
    original bytes.

    PDF: the info dictionary's `CreationDate`, else `ModDate`. Word and PowerPoint:
    `dcterms:modified` in `docProps/core.xml`, else `dcterms:created`. Markdown has none. A
    value that is not a whole, well-formed date string with a real calendar date counts as
    absent. A file whose metadata container cannot be read (a PDF that does not open, an
    Office file that is not a zip archive, a `core.xml` that is not XML or is larger than
    MAX_CORE_PROPERTIES_BYTES) is corrupt and raises ConversionError.
    """
    if source.format == "md":
        return None
    if source.format == "pdf":
        with _PDFIUM_LOCK:
            try:
                document = _open_pdf(data)
                with _closing_pdf(document):
                    try:
                        values = [document.get_metadata_value(k) for k in _PDF_DATE_KEYS]
                    except _PDFIUM_FAILURES as exc:
                        raise ConversionError(f"cannot read PDF metadata: {exc}") from exc
            except ConversionError as exc:
                raise ConversionError(f"{source.rel_path}: {exc}") from exc
        pattern = _PDF_DATE
    else:
        core = _office_core_properties(source, data)
        if core is None:
            return None
        values = [core.findtext(f"{_DCTERMS}{k}") or "" for k in ("modified", "created")]
        pattern = _W3C_DATE
    for value in values:
        if (match := pattern.fullmatch(value.strip())) and (iso := _metadata_iso(match)):
            return iso
    return None


def _metadata_iso(match: re.Match[str]) -> str | None:
    """The calendar date of a matched metadata timestamp, or None when its date is not real
    or a clock field (hour, minute, second, offset) is out of range."""
    fields = match.groupdict()
    if any(int(fields[k] or 0) > limit for k, limit in _CLOCK_LIMITS.items()):
        return None
    return iso_date(fields["year"], fields["month"], fields["day"])


def printed_date_or_none(value: str | None) -> str | None:
    """A model's printed-date answer with surrounding space removed, or None when it is blank
    or a placeholder such as `none` or `not stated` rather than a printed date."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    return None if fold_text(text).strip(" .") in _NO_DATE_ANSWERS else text


def document_date(printed: str | None, embedded: str | None) -> DocumentDate | None:
    """A document's date from its printed date and its file's metadata date (`embedded`): the
    printed date when it has one reading, else nothing when a date is printed, else the
    metadata date.

    A printed date that is ambiguous never falls back to the metadata date, which can differ
    from the printed one by months. File system times are never read.
    """
    if printed is not None and printed.strip():
        parsed = parse_printed_date(printed)
        return None if parsed is None else DocumentDate(iso=parsed, basis="printed")
    return None if embedded is None else DocumentDate(iso=embedded, basis="metadata")


def read_document(
    source: SourceFile,
    data: bytes,
    gateway: LanguageGateway,
    *,
    render_cache: RenderCache | None = None,
) -> DocumentReading:
    """Read one source from `data`, the bytes read from it exactly once.

    The result depends on the bytes and the source format only: `source.rel_path` appears in
    error messages, and a file named `AGENTS.md` is read as a curator note. With a
    `render_cache`, a source rendered before skips LibreOffice and pdfium.
    """
    sha256 = hashlib.sha256(data).hexdigest()
    pages: tuple[RenderedPage, ...] = ()
    text, embedded = "", None
    if source.format == "md":
        text = _decode_markdown(source, data)
        drafts = assemble_sections([(1, parse_markdown_blocks(text))])
    else:
        rendered = render_source(source, data, sha256, render_cache)
        embedded = rendered.metadata_date
        transcribed = _transcribe_pages(gateway, source, rendered.pages)
        drafts = assemble_sections(transcribed)
        pages = tuple(
            RenderedPage(
                page_number=number,
                native_text=native_text,
                webp_bytes=webp,
                visual_summary=_visual_summary(extraction),
            )
            for (native_text, webp), (number, extraction) in zip(
                rendered.pages, transcribed, strict=True
            )
        )
    prose_kind: ChunkKind = "guidance" if source.is_curator_note else "prose"
    sections = tuple(
        ReadSection(
            ordinal=ordinal,
            heading_path=draft.heading_path,
            level=draft.level,
            page_start=draft.page_start,
            page_end=draft.page_end,
            markdown=draft.markdown,
            pieces=section_pieces(draft, prose_kind),
        )
        for ordinal, draft in enumerate(drafts, start=1)
    )
    if source.is_curator_note:
        card, discarded = curator_note_card(text), 0
    else:
        card, discarded = observe_document(gateway, source.rel_path, sections)
    return DocumentReading(
        sha256=sha256,
        format=source.format,
        is_curator_note=source.is_curator_note,
        sections=sections,
        pages=pages,
        card=card,
        date=document_date(card.doc_date, embedded),
        discarded_quotes=discarded,
    )


# Binding.


def breadcrumb_prefix(facets: Sequence[Facet], assignment: Mapping[str, Sequence[str]]) -> str:
    """`<facet>: v1, v2 | ...` in facet definition order, values sorted, empty facets left out."""
    return " | ".join(
        f"{facet.name}: {', '.join(sorted(assignment[facet.name]))}"
        for facet in facets
        if assignment.get(facet.name)
    )


def bind_document(
    document: CorpusDocument,
    facets: Sequence[Facet],
    assignment: Mapping[str, Sequence[str]],
    role: DocumentRole,
) -> CompiledDocument:
    """Bind one document: section and chunk IDs, breadcrumbs from its facets, and its role:
    the induced role of a citable document, or `curator_note_role` for a curator note."""
    reading, card = document.reading, document.reading.card
    prefix = breadcrumb_prefix(facets, assignment)
    sections: list[Section] = []
    chunks: list[Chunk] = []
    for read in reading.sections:
        section = Section(
            section_id=document.section_id(read.ordinal),
            ordinal=read.ordinal,
            heading_path=read.heading_path,
            level=read.level,
            page_start=read.page_start,
            page_end=read.page_end,
            markdown=read.markdown,
        )
        sections.append(section)
        crumb = (
            f"Doc: {card.title} (pp. {section.page_start}-{section.page_end})"
            f" | Section: {heading_context(section.heading_path)}"
        )
        crumb = f"[{prefix} | {crumb}]" if prefix else f"[{crumb}]"
        for piece in read.pieces:
            ordinal = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_chk_{ordinal:04d}",
                    section_id=section.section_id,
                    ordinal=ordinal,
                    kind=piece.kind,
                    page_start=piece.page_start,
                    page_end=piece.page_end,
                    breadcrumb=crumb,
                    content=piece.content,
                )
            )
    return CompiledDocument(
        doc_id=document.doc_id,
        rel_path=document.rel_path,
        sha256=reading.sha256,
        format=reading.format,
        title=card.title,
        doc_date=(
            card.doc_date
            if card.doc_date and card.doc_date.strip()
            else (reading.date.iso if reading.date else None)
        ),
        date=reading.date,
        summary=card.summary,
        facets={
            f.name: tuple(sorted(assignment[f.name])) for f in facets if assignment.get(f.name)
        },
        sections=tuple(sections),
        chunks=tuple(chunks),
        pages=reading.pages,
        is_curator_note=reading.is_curator_note,
        role=role,
    )


def curator_note_role(document: CorpusDocument) -> DocumentRole:
    """A curator note's role: its card's type, no feedback, and a version family of its own."""
    return DocumentRole(
        deliverable_type=document.reading.card.deliverable_type,
        is_feedback=False,
        feedback_for_types=(),
        family_id=document.doc_id,
        version_status="sole",
        superseded_by=None,
    )


def _effort_settings() -> dict[str, str]:
    retry = FIDELITY_ATTEMPTS[1].effort
    return {
        PageTranscription.operation: f"{FIDELITY_ATTEMPTS[0].effort} ({retry} on fidelity retry)",
        ObserveSections.operation: ObserveSections.effort,
        ConsolidateCard.operation: ConsolidateCard.effort,
        **INDUCTION_EFFORTS,
    }


def compile_corpus(
    corpus_dir: Path,
    gateway: LanguageGateway,
    *,
    topic: str,
    title: str,
    max_concurrency: int = 4,
    render_cache: RenderCache | None = None,
) -> CompiledCorpus:
    sources = discover_corpus(corpus_dir)
    doc_ids = assign_document_ids(sources)
    # Set by the first failed read: a read that a worker starts after it does no work, so
    # reads already handed to workers stop at once rather than when cancellation reaches them.
    failed = threading.Event()
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:

        def read(source: SourceFile) -> tuple[str | None, DocumentReading] | None:
            """The reading and, for a curator note only, its text: no other bytes outlive it.
            None when another read has already failed."""
            if failed.is_set():
                return None
            try:
                data = read_source(source)
                note = _decode_markdown(source, data) if source.is_curator_note else None
                return note, read_document(source, data, gateway, render_cache=render_cache)
            except BaseException:
                failed.set()
                raise

        # Rendered documents first, largest first, so a long PDF starts at once instead of
        # holding the build up at the end; results are gathered in path order.
        schedule = sorted(
            range(len(sources)),
            key=lambda i: (sources[i].format == "md", -sources[i].size_bytes, i),
        )
        futures: dict[int, Future[tuple[str | None, DocumentReading] | None]] = {
            i: pool.submit(read, sources[i]) for i in schedule
        }
        try:
            # Wait for every read or the first failure, so a failed read cancels the queued
            # ones at once instead of after the reads before it in path order finish.
            wait(futures.values(), return_when=FIRST_EXCEPTION)
            for i in schedule:
                if futures[i].done() and futures[i].exception() is not None:
                    futures[i].result()
            outcomes = [futures[i].result() for i in range(len(sources))]
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    # A read returns None only after another read failed, and that failure was raised above.
    results = [outcome for outcome in outcomes if outcome is not None]
    assert len(results) == len(sources), "a read skipped without a failed read"
    notes = tuple(
        CuratorNote(scope_path=source.rel_path, content=note)
        for source, (note, _) in zip(sources, results, strict=True)
        if note is not None
    )
    documents = tuple(
        CorpusDocument(doc_id, source.rel_path, reading)
        for doc_id, source, (_, reading) in zip(doc_ids, sources, results, strict=True)
    )
    knowledge = induce(documents, notes, gateway, topic=topic, max_concurrency=max_concurrency)
    compiled = tuple(
        bind_document(
            d,
            knowledge.facets,
            knowledge.assignments[d.doc_id],
            knowledge.roles[d.doc_id] if d.is_citable else curator_note_role(d),
        )
        for d in documents
    )
    chunks = [chunk for doc in compiled for chunk in doc.chunks]
    embeddings = gateway.embed_texts([f"{c.breadcrumb}\n{c.content}" for c in chunks])
    settings: dict[str, Any] = {
        **gateway.describe(),
        "efforts": _effort_settings(),
        "rendering": {
            "pdf_render_scale": PDF_RENDER_SCALE,
            "webp_quality": WEBP_QUALITY,
            "max_render_side_px": MAX_RENDER_SIDE_PX,
        },
    }
    return CompiledCorpus(
        corpus_dir=corpus_dir,
        topic=topic,
        title=title,
        embedding_model=gateway.embedding_model,
        embedding_dimensions=gateway.embedding_dimensions,
        curator_notes=notes,
        guide_markdown=knowledge.guide_markdown,
        facets=knowledge.facets,
        glossary=knowledge.glossary,
        deliverable_types=knowledge.deliverable_types,
        feedback_lessons=knowledge.feedback_lessons,
        unit_clusters=knowledge.unit_clusters,
        build_settings=settings,
        build_report=knowledge.report,
        documents=compiled,
        embeddings=embeddings,
    )
