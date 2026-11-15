"""Compile a corpus directory into a `CompiledCorpus`.

Every page of every PDF, Word and PowerPoint file goes through one pipeline:
render to WebP, transcribe with the generation model, check the transcript against
the native text layer, then stitch pages into sections and chunks. The request types
below own every prompt, output schema and reasoning effort the compiler sends.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import pypdfium2 as pdfium
from PIL import Image

from arkive_rag.llm import Effort, LanguageGateway
from arkive_rag.models import (
    Chunk,
    ChunkKind,
    CompiledCorpus,
    CompiledDocument,
    ConversionError,
    CorpusSynthesisExtraction,
    CuratorNote,
    DocumentMetaExtraction,
    InvalidCitationError,
    MissingDependencyError,
    PageBlock,
    PageExtraction,
    RenderedPage,
    Section,
    SourceFormat,
    TranscriptionFidelityError,
    UnsafePathError,
    UnsupportedFormatError,
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
OFFICE_TIMEOUT_SECONDS = 120
PREV_CONTEXT_CHARS = 1200

# libpdfium and pypdfium2's object tracker are not thread-safe: every pdfium call
# across all threads goes through this lock.
_PDFIUM_LOCK = threading.Lock()

_FILENAME_ANCHOR = re.compile(r"\b[A-Za-z0-9_.-]+\.(?:tif|tiff|shp|nc|gpkg|csv)\b", re.IGNORECASE)
# An identifier fragment that ends where a following filename anchor begins, separated
# only by kerning spaces and at most one line break.
_WRAPPED_PREFIX = re.compile(r"([A-Za-z0-9_.-]+)[ \t]*\n?[ \t]*\Z")
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
_CITATION_TOKEN = re.compile(r"\[doc[^\]]*\]")
_CITATION = re.compile(r"\[(doc_\d+):(doc_\d+_sec_\d+)\]")
_TABLE_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_HEADING = re.compile(r"(#{1,6})\s+(\S.*)")
_FENCE = re.compile(r"(`{3,}|~{3,})")


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

META_INSTRUCTIONS = """\
You catalogue one consulting document. From its path, project folder, outline
and opening text give: the document title, the client organisation, the
lifecycle stage (proposal, memorandum, feedback, summary, case_study, guidance
or reference), the document date as printed (null if absent) and a factual
two-to-four sentence summary naming its subject, methods and key results.
"""

SYNTHESIS_INSTRUCTIONS = """\
You write the orientation guide for a knowledge base that AI agents will search.
Using only the catalog and curator notes, describe in Markdown: the projects,
clients and how they relate; what each document type covers; recurring methods,
datasets and results; and caveats or gaps the curator notes record.
Cite every factual claim with the exact section citation tokens from the
catalog, written as [doc_XXX:doc_XXX_sec_YYY]. Use no other citation form.
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
class DocumentCatalogue:
    """Title, client, lifecycle stage, date and summary of one non-curator document."""

    operation: ClassVar[str] = "document_meta"
    output: ClassVar[type[DocumentMetaExtraction]] = DocumentMetaExtraction
    effort: ClassVar[Effort] = "low"
    instructions: ClassVar[str] = META_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    rel_path: str
    project_id: str
    outline_and_excerpt: str

    def text(self) -> str:
        return (
            f"PATH: {self.rel_path}\nPROJECT FOLDER: {self.project_id}\n\n"
            f"{self.outline_and_excerpt}"
        )


@dataclass(frozen=True)
class CorpusSynthesis:
    """The cited topic guide. `citation_feedback` lists a rejected draft's invalid citations."""

    operation: ClassVar[str] = "synthesize_corpus"
    output: ClassVar[type[CorpusSynthesisExtraction]] = CorpusSynthesisExtraction
    effort: ClassVar[Effort] = "medium"
    instructions: ClassVar[str] = SYNTHESIS_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    topic: str
    curator_notes_text: str
    catalog_text: str
    citation_feedback: str | None = None

    def text(self) -> str:
        text = (
            f"TOPIC: {self.topic}\n\nCURATOR NOTES:\n{self.curator_notes_text}"
            f"\n\nCATALOG:\n{self.catalog_text}"
        )
        if self.citation_feedback:
            text += f"\n\nYOUR PREVIOUS DRAFT HAD INVALID CITATIONS:\n{self.citation_feedback}"
        return text


@dataclass(frozen=True)
class SourceFile:
    rel_path: str
    path: Path
    format: SourceFormat
    project_id: str

    @property
    def is_curator_note(self) -> bool:
        return self.path.name == CURATOR_NOTE_NAME


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
        project_id = rel.parts[0] if len(rel.parts) >= 2 else "root"
        sources.append(SourceFile(rel.as_posix(), path, fmt, project_id))
    return sources


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
        bitmap = page.render(scale=2.0)
        try:
            # copy() detaches the pixels from pdfium's buffer before it is freed.
            image = bitmap.to_pil().copy()
        finally:
            bitmap.close()
    finally:
        page.close()
    return text.replace("\r\n", "\n"), image


def render_pdf(pdf_bytes: bytes) -> list[tuple[str, bytes]]:
    """Return (native text, WebP bytes) for each page."""
    with _PDFIUM_LOCK:
        try:
            document = pdfium.PdfDocument(pdf_bytes)
        except pdfium.PdfiumError as exc:
            raise ConversionError(f"cannot open PDF: {exc}") from exc
    pages: list[tuple[str, bytes]] = []
    try:
        with _PDFIUM_LOCK:
            page_count = len(document)
        for index in range(page_count):
            with _PDFIUM_LOCK:
                text, image = _render_page(document, index)
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=85)
            pages.append((text, buffer.getvalue()))
    finally:
        with _PDFIUM_LOCK:
            document.close()
    return pages


# Fidelity gate.


def _edge_lines(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[:2] + lines[-2:] if len(lines) > 4 else lines


def _furniture_keys(line: str, page_index: int) -> set[str]:
    """The exact line, plus the line without an edge number that tracks the page index."""
    text = " ".join(line.lower().split())
    keys = {text}
    if match := _EDGE_PAGE_NUMBER.fullmatch(text):
        number, rest = (match[1], match[2]) if match[1] else (match[4], match[3])
        keys.add(f"{rest}\x00{page_index - int(number)}")
    return keys


def find_furniture(page_texts: Sequence[str]) -> frozenset[str]:
    """Keys of leading/trailing lines repeated on at least three pages."""
    counts: Counter[str] = Counter()
    for index, text in enumerate(page_texts):
        counts.update({key for line in _edge_lines(text) for key in _furniture_keys(line, index)})
    return frozenset(key for key, count in counts.items() if count >= 3)


def strip_furniture(text: str, furniture: frozenset[str], page_index: int) -> str:
    """Drop repeated edge lines and page numbers."""
    nonblank = [line for line in text.splitlines() if line.strip()]
    edges = set(_edge_lines(text))
    bounds = {nonblank[0], nonblank[-1]} if nonblank else set()
    return "\n".join(
        line
        for line in text.splitlines()
        if not (
            (line in bounds and _BARE_PAGE_NUMBER.fullmatch(line))
            or (
                line in edges
                and (_PAGE_LABEL.fullmatch(line) or _furniture_keys(line, page_index) & furniture)
            )
        )
    )


def _normalize_numbers(text: str) -> str:
    return _THOUSANDS_COMMA.sub("", text.replace("\u2212", "-"))


def _number_anchors(text: str) -> set[str]:
    """Signed formatted numbers and long integers in number-normalized text."""
    integers = _LONG_INTEGER.findall(_FORMATTED_NUMBER.sub(" ", text))
    return {*_FORMATTED_NUMBER.findall(text), *integers}


def fidelity_anchors(body_text: str) -> tuple[dict[str, frozenset[str]], set[str]]:
    """Return (filenames, signed numbers) from native body text, without thousands commas.

    Each lowercased filename anchor maps to the emitted filenames that satisfy it: itself,
    and its joins with the one or two identifier fragments directly before it. Native text
    can split a long filename mid-identifier across a line break or kerning space, so the
    anchor holds only the tail while a faithful transcription carries the whole name.
    """
    forms: dict[str, set[str]] = {}
    for match in _FILENAME_ANCHOR.finditer(body_text):
        name = match.group(0).lower()
        accepted = forms.setdefault(name, set())
        accepted.add(name)
        before = body_text[: match.start()]
        for _ in range(2):
            fragment = _WRAPPED_PREFIX.search(before)
            if fragment is None:
                break
            name = fragment.group(1).lower() + name
            accepted.add(name)
            before = before[: fragment.start()]
    numbers = _number_anchors(_normalize_numbers(_FILENAME_ANCHOR.sub(" ", body_text)))
    return {name: frozenset(accepted) for name, accepted in forms.items()}, numbers


def missing_anchors(
    page_number: int,
    extraction: PageExtraction,
    filenames: dict[str, frozenset[str]],
    numbers: set[str],
) -> list[str]:
    """Anchors absent from the exact text `assemble_sections` emits for this page.

    The emitted text is tokenized exactly as `fidelity_anchors` tokenizes native text, so
    a verbatim transcription has no missing anchors and a signed value never satisfies
    its unsigned counterpart. A filename anchor is present when the emitted filenames
    hold any of its accepted forms. Numbers are also read with filenames left in place,
    so a number that native text split from a line-wrapped filename still matches the
    rejoined filename.
    """
    emitted = "\n".join(part.markdown for part in page_parts(page_number, extraction))
    emitted = emitted.replace("\\", "")
    emitted_files, emitted_numbers = fidelity_anchors(emitted)
    emitted_numbers |= _number_anchors(_normalize_numbers(emitted))
    missing = sorted(
        name for name, accepted in filenames.items() if not accepted & emitted_files.keys()
    )
    return missing + sorted(numbers - emitted_numbers)


def _fidelity_ok(
    missing: list[str], filenames: dict[str, frozenset[str]], total: int, threshold: float
) -> bool:
    if any(token in filenames for token in missing):
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
    kind: PartKind
    markdown: str
    page_start: int
    page_end: int


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


def merge_tables(previous: str, continuation: str, merges_into_previous_row: bool) -> str:
    """Append a continued table's rows to the previous table, dropping a repeated header."""
    prev_rows, cont_rows = _table_rows(previous), _table_rows(continuation)
    if len(cont_rows) >= 2 and _is_separator(cont_rows[1]):
        cont_rows = cont_rows[2:]
    last = prev_rows[-1] if prev_rows else None
    if (
        merges_into_previous_row
        and cont_rows
        and last is not None
        and not _is_separator(last)
        and len(last) == len(cont_rows[0])
    ):
        prev_rows[-1] = [
            " ".join(t for t in pair if t) for pair in zip(last, cont_rows[0], strict=True)
        ]
        cont_rows = cont_rows[1:]
    return _render_rows(prev_rows + cont_rows)


def _heading_text(markdown: str) -> str:
    return markdown.strip().lstrip("#").strip()


def _heading_level(block: PageBlock) -> int:
    return block.heading_level or 2


def _block_parts(block: PageBlock, page: int) -> list[Part]:
    if block.kind == "heading":
        text = f"{'#' * _heading_level(block)} {_heading_text(block.markdown)}"
        return [Part("heading", text, page, page)]
    if block.kind == "table":
        return [Part(kind, text, page, page) for kind, text in split_table_block(block.markdown)]
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
    """Stitch transcribed pages into heading-delimited sections."""
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
                    previous.markdown = merge_tables(
                        previous.markdown, first.markdown, block.merges_into_previous_row
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
    return sections


def _markdown_blocks(text: str) -> list[tuple[str, list[str]]]:
    """Group lines into (kind, lines); a fenced code block stays one literal paragraph."""
    groups: list[tuple[str, list[str]]] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if fence is not None:
            groups[-1][1].append(line)
            if stripped.startswith(fence) and not stripped.strip(fence[0]):
                fence = None
            continue
        if opening := _FENCE.match(stripped):
            fence = opening[1]
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


def wrap_text(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most `limit` characters at line breaks, else spaces."""
    pieces: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n", 1, limit + 1)
        if cut <= 0:
            cut = rest.rfind(" ", 1, limit + 1)
        if cut <= 0:
            pieces.append(rest[:limit])
            rest = rest[limit:]
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


def split_table(markdown: str) -> list[str]:
    """Split a large table on row boundaries, repeating its header rows in every piece.

    A row too long for one piece is wrapped into several pieces, each under the header.
    """
    if len(markdown) <= WHOLE_TABLE_CHARS:
        return [markdown]
    lines = markdown.splitlines()
    rows = _table_rows(markdown)
    header_count = 2 if len(rows) >= 2 and _is_separator(rows[1]) else 0
    header, body = lines[:header_count], lines[header_count:]
    room = max(TABLE_PIECE_CHARS - len("\n".join(header)) - 1, TABLE_PIECE_CHARS // 4)
    pieces: list[str] = []
    current: list[str] = []
    for line in (segment for row in body for segment in wrap_text(row, room)):
        if current and len("\n".join([*header, *current, line])) > TABLE_PIECE_CHARS:
            pieces.append("\n".join([*header, *current]))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join([*header, *current]))
    return pieces


@dataclass(frozen=True)
class ChunkContext:
    doc_id: str
    project_id: str
    client: str
    stage: str
    title: str
    prose_kind: ChunkKind


def _section_pieces(draft: SectionDraft) -> list[tuple[ChunkKind | None, str, int, int]]:
    """(kind, content, first page, last page) per chunk; kind None marks packed prose."""
    pieces: list[tuple[ChunkKind | None, str, int, int]] = []
    prose: list[tuple[str, int, int]] = []

    def flush() -> None:
        if prose:
            text = "\n\n".join(t for t, _, _ in prose)
            first, last = min(p[1] for p in prose), max(p[2] for p in prose)
            pieces.append((None, text, first, last))
            prose.clear()

    for part in draft.parts:
        span = (part.page_start, part.page_end)
        if part.kind == "table":
            flush()
            pieces += [("table", t, *span) for t in split_table(part.markdown)]
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
    return [
        (kind, piece, first, last)
        for kind, content, first, last in pieces
        for piece in wrap_text(content, MAX_CHUNK_CHARS)
    ]


def build_chunks(
    drafts: Sequence[SectionDraft], sections: Sequence[Section], ctx: ChunkContext
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for draft, section in zip(drafts, sections, strict=True):
        breadcrumb = (
            f"[Project: {ctx.project_id} | Client: {ctx.client} | Stage: {ctx.stage}"
            f" | Doc: {ctx.title} (pp. {section.page_start}-{section.page_end})"
            f" | Section: {section.heading_path}]"
        )
        for kind, content, first, last in _section_pieces(draft):
            ordinal = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{ctx.doc_id}_chk_{ordinal:04d}",
                    section_id=section.section_id,
                    ordinal=ordinal,
                    kind=kind or ctx.prose_kind,
                    page_start=first,
                    page_end=last,
                    breadcrumb=breadcrumb,
                    content=content,
                )
            )
    return chunks


# Document and corpus compilation.


def _transcribe_pages(
    gateway: LanguageGateway, source: SourceFile, rendered: list[tuple[str, bytes]]
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
        prev_context = "\n\n".join(b.markdown for b in extraction.blocks)[-PREV_CONTEXT_CHARS:]
    return results


def _visual_summary(extraction: PageExtraction) -> str:
    lines = [
        f"Heading: {_heading_text(b.markdown)}" for b in extraction.blocks if b.kind == "heading"
    ]
    lines += [f"Visual: {v.label}: {v.transcript}" for v in extraction.visuals]
    return "\n".join(lines)


def compile_document(
    source: SourceFile, doc_id: str, gateway: LanguageGateway, data: bytes
) -> CompiledDocument:
    """Compile one source from `data`, the bytes read from it exactly once."""
    pages: tuple[RenderedPage, ...] = ()
    text = _decode_markdown(source, data) if source.format == "md" else ""
    if source.format == "md":
        drafts = assemble_sections([(1, parse_markdown_blocks(text))])
    else:
        pdf = (
            data
            if source.format == "pdf"
            else convert_office_to_pdf(data, source.path.suffix.lower(), source.rel_path)
        )
        rendered = render_pdf(pdf)
        transcribed = _transcribe_pages(gateway, source, rendered)
        drafts = assemble_sections(transcribed)
        pages = tuple(
            RenderedPage(
                page_number=number,
                native_text=native_text,
                webp_bytes=webp,
                visual_summary=_visual_summary(extraction),
            )
            for (native_text, webp), (number, extraction) in zip(rendered, transcribed, strict=True)
        )
    sections = tuple(
        Section(
            section_id=f"{doc_id}_sec_{ordinal:03d}",
            ordinal=ordinal,
            heading_path=draft.heading_path,
            level=draft.level,
            page_start=draft.page_start,
            page_end=draft.page_end,
            markdown=draft.markdown,
        )
        for ordinal, draft in enumerate(drafts, start=1)
    )
    prose_kind: ChunkKind = "guidance" if source.is_curator_note else "prose"
    if source.is_curator_note:
        scope = source.rel_path.rpartition("/")[0] or "corpus root"
        first_line = next((ln.strip("# ") for ln in text.splitlines() if ln.strip()), "")
        meta = DocumentMetaExtraction(
            title=f"Curator guidance: {scope}",
            client="curator",
            stage="guidance",
            doc_date=None,
            summary=first_line[:300],
        )
    else:
        outline = "\n".join(
            f"- {s.heading_path} (pp. {s.page_start}-{s.page_end})" for s in sections
        )
        excerpt = "\n\n".join(s.markdown for s in sections)[:6000]
        meta = gateway.run(
            DocumentCatalogue(
                rel_path=source.rel_path,
                project_id=source.project_id,
                outline_and_excerpt=f"OUTLINE:\n{outline}\n\nOPENING TEXT:\n{excerpt}",
            )
        )
    ctx = ChunkContext(doc_id, source.project_id, meta.client, meta.stage, meta.title, prose_kind)
    return CompiledDocument(
        doc_id=doc_id,
        rel_path=source.rel_path,
        sha256=hashlib.sha256(data).hexdigest(),
        format=source.format,
        project_id=source.project_id,
        client=meta.client,
        stage=meta.stage,
        title=meta.title,
        doc_date=meta.doc_date,
        summary=meta.summary,
        sections=sections,
        chunks=tuple(build_chunks(drafts, sections, ctx)),
        pages=pages,
        is_curator_note=source.is_curator_note,
    )


def _citable(documents: Sequence[CompiledDocument]) -> list[CompiledDocument]:
    return [doc for doc in documents if not doc.is_curator_note]


def invalid_citations(markdown: str, documents: Sequence[CompiledDocument]) -> list[str]:
    """Bracketed `[doc...]` tokens that do not resolve, or a marker when one is needed."""
    valid = {(doc.doc_id, s.section_id) for doc in _citable(documents) for s in doc.sections}
    bad: list[str] = []
    resolved = 0
    for token in _CITATION_TOKEN.findall(markdown):
        match = _CITATION.fullmatch(token)
        if match is None or (match[1], match[2]) not in valid:
            bad.append(token)
        else:
            resolved += 1
    if valid and resolved == 0:
        bad.append("(no valid citation)")
    return bad


def _catalog_text(documents: Sequence[CompiledDocument]) -> str:
    lines: list[str] = []
    for doc in _citable(documents):
        lines.append(
            f"\n## {doc.doc_id}: {doc.title}\nProject: {doc.project_id} | Client: {doc.client}"
            f" | Stage: {doc.stage} | Date: {doc.doc_date or 'unknown'}\nSummary: {doc.summary}"
        )
        lines += [
            f"- [{doc.doc_id}:{s.section_id}] {s.heading_path} (pp. {s.page_start}-{s.page_end})"
            for s in doc.sections
        ]
    return "\n".join(lines)


def synthesize_with_citations(
    gateway: LanguageGateway,
    topic: str,
    notes: Sequence[CuratorNote],
    documents: Sequence[CompiledDocument],
) -> str:
    notes_text = "\n\n".join(f"### {n.scope_path}\n{n.content}" for n in notes) or "(none)"
    catalog = _catalog_text(documents)
    request = CorpusSynthesis(topic, notes_text, catalog)
    markdown = gateway.run(request).synthesis_markdown
    bad = invalid_citations(markdown, documents)
    if not bad:
        return markdown
    feedback = f"Invalid: {', '.join(bad)}. Cite only these section tokens:\n" + "\n".join(
        f"[{d.doc_id}:{s.section_id}]" for d in _citable(documents) for s in d.sections
    )
    retry = CorpusSynthesis(topic, notes_text, catalog, citation_feedback=feedback)
    markdown = gateway.run(retry).synthesis_markdown
    bad = invalid_citations(markdown, documents)
    if bad:
        raise InvalidCitationError(
            f"corpus synthesis still has invalid citations: {', '.join(bad)}"
        )
    return markdown


def compile_corpus(
    corpus_dir: Path,
    gateway: LanguageGateway,
    *,
    topic: str,
    title: str,
    max_concurrency: int = 4,
) -> CompiledCorpus:
    sources = discover_corpus(corpus_dir)
    note_bytes = {s.rel_path: read_source(s) for s in sources if s.is_curator_note}
    notes = tuple(
        CuratorNote(scope_path=s.rel_path, content=_decode_markdown(s, note_bytes[s.rel_path]))
        for s in sources
        if s.is_curator_note
    )

    def compile_source(source: SourceFile, doc_id: str) -> CompiledDocument:
        data = note_bytes[source.rel_path] if source.is_curator_note else read_source(source)
        return compile_document(source, doc_id, gateway, data)

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures: list[Future[CompiledDocument]] = [
            pool.submit(compile_source, source, f"doc_{index:03d}")
            for index, source in enumerate(sources, start=1)
        ]
        try:
            documents = tuple(future.result() for future in futures)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    synthesis = synthesize_with_citations(gateway, topic, notes, documents)
    chunks = [chunk for doc in documents for chunk in doc.chunks]
    embeddings = gateway.embed_texts([f"{c.breadcrumb}\n{c.content}" for c in chunks])
    return CompiledCorpus(
        corpus_dir=corpus_dir,
        topic=topic,
        title=title,
        embedding_model=gateway.embedding_model,
        embedding_dimensions=gateway.embedding_dimensions,
        curator_notes=notes,
        synthesis_markdown=synthesis,
        documents=documents,
        embeddings=embeddings,
    )
