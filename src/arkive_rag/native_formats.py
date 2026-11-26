"""Read what a file's own structure holds: Excel workbooks and HTML pages as Markdown, and the
reviewer comments, tracked changes, speaker notes and chart data of Word and PowerPoint files.

This is the only module that imports Docling or openpyxl, and the compiler imports it inside
the functions that read those formats, so serving a snapshot never loads either. Every reader
takes the bytes the compiler read once from the corpus file: no host path ever reaches Docling,
LibreOffice or openpyxl, so a file swapped after discovery cannot be read in its place.

Every Office file is a zip archive; `open_office_archive` inflates every member of it within
bounds before any member is parsed or the bytes reach Docling, openpyxl or LibreOffice. Every
failure to read a file is a ConversionError.
"""

from __future__ import annotations

import io
import re
import struct
import zipfile
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from posixpath import dirname, normpath
from posixpath import join as join_path
from typing import TYPE_CHECKING
from xml.etree import ElementTree

from lxml import etree

from arkive_rag.models import ConversionError
from arkive_rag.passages import annotation_markdown, render_table

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.items.node import NodeItem
    from docling_core.types.doc.items.table.table import TableItem

# Largest inflated size of all members of an Office archive, and the most members it may
# hold. The entry count is read from the central directory; the inflated size is measured by
# inflating every member, since a directory can understate it.
MAX_OFFICE_ARCHIVE_BYTES = 100_000_000
MAX_OFFICE_ARCHIVE_ENTRIES = 5_000
# How much compressed input and inflated output one step of the archive check handles.
ARCHIVE_CHUNK_BYTES = 64 * 1024
# Most populated cells (a value, an inline string or a formula) a workbook may hold. A larger
# workbook is refused, never cut.
MAX_WORKBOOK_CELLS = 250_000
# How much of a comment's anchor text its label repeats.
ANCHOR_LABEL_CHARS = 80

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
# A cell reference as a worksheet's `c/@r` writes it: column letters, then a row from 1.
_CELL_REF = re.compile(r"([A-Z]{1,3})([1-9]\d*)")
# A line Markdown would read as structure rather than text when it opens a paragraph.
_MARKDOWN_OPENER = re.compile(r"(#|>|\||```|~~~)")
# Member flags `ZipFile.open` cannot read: encrypted (0x1), compressed patch data (0x20) and
# strong encryption (0x40).
_UNREADABLE_FLAGS = 0x1 | 0x20 | 0x40

# The failures any member read or parse can raise; each becomes ConversionError.
ARCHIVE_FAILURES = (
    OSError,
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    ValueError,
    RuntimeError,
    NotImplementedError,
    EOFError,
    KeyError,
    zlib.error,
    ElementTree.ParseError,
)


@dataclass(frozen=True)
class Aside:
    """An annotation of a document: `label` names what it is and where it sits, `text` is its
    content, and `anchor` is document text near it (empty when it has none), which places it on
    a rendered page."""

    label: str
    text: str
    anchor: str

    @property
    def markdown(self) -> str:
        return annotation_markdown(self.label, self.text)


@dataclass(frozen=True)
class ChartTable:
    """The data series of one native chart as a table: a category column, then one column per
    series, or for series with x values of their own a row per point (`_chart_table`)."""

    caption: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SlideSupplement:
    """What one shown slide holds beyond its rendered page: chart data, then speaker notes
    and comments. `number` counts shown slides from 1, as the rendered pages do."""

    number: int
    charts: tuple[ChartTable, ...]
    asides: tuple[Aside, ...]


# Archives.


def check_office_archive(data: bytes) -> None:
    """Refuse an Office file that `open_office_archive` refuses."""
    with open_office_archive(data):
        pass


def open_office_archive(data: bytes) -> zipfile.ZipFile:
    """`data` opened as a zip archive once every member is checked by inflating it.

    Refused: an archive that is not readable, holds more than MAX_OFFICE_ARCHIVE_ENTRIES
    members, a member that is encrypted, holds patch data or is compressed other than stored
    or deflated (the methods OOXML allows), a member whose local header is missing or names
    another file, a member whose inflated bytes differ from the size and CRC its
    directory entry declares, or members that inflate to more than MAX_OFFICE_ARCHIVE_BYTES
    together. Each member is inflated in ARCHIVE_CHUNK_BYTES steps and its output dropped, so
    the check never holds more than one step of output."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except ARCHIVE_FAILURES as exc:
        raise ConversionError(f"cannot read the Office archive: {exc}") from exc
    try:
        infos = archive.infolist()
        if len(infos) > MAX_OFFICE_ARCHIVE_ENTRIES:
            raise ConversionError(
                f"the Office archive has {len(infos)} entries; the limit is"
                f" {MAX_OFFICE_ARCHIVE_ENTRIES}"
            )
        inflated = 0
        for info in infos:
            inflated += _inflated_size(data, info, MAX_OFFICE_ARCHIVE_BYTES - inflated)
            # Opening reads nothing but runs every structural check `ZipFile.open` makes,
            # members that overlap one another included.
            try:
                with archive.open(info):
                    pass
            except ARCHIVE_FAILURES as exc:
                raise ConversionError(f"cannot read the Office archive: {exc}") from exc
    except BaseException:
        archive.close()
        raise
    return archive


def _inflated_size(data: bytes, info: zipfile.ZipInfo, budget: int) -> int:
    """The size of one member inflated from its stored bytes, which must match the size and
    CRC its directory entry declares and stay within `budget`."""
    name = info.filename
    if info.flag_bits & _UNREADABLE_FLAGS:
        raise ConversionError(f"{name} is encrypted or holds patch data")
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ConversionError(
            f"{name} is compressed with method {info.compress_type}; an Office archive is"
            " stored or deflated"
        )
    offset = info.header_offset
    header = data[offset : offset + 30] if offset >= 0 else b""
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        raise ConversionError(f"cannot read the Office archive: {name} has no local header")
    (local_flags,) = struct.unpack("<H", header[6:8])
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    # `ZipFile.open` refuses a member whose local header names another file; refuse it here,
    # before any reader opens the archive.
    local_name = data[offset + 30 : offset + 30 + name_length]
    try:
        matches = local_name.decode("utf-8" if local_flags & 0x800 else "cp437") == (
            info.orig_filename
        )
    except UnicodeDecodeError:
        matches = False
    if not matches:
        raise ConversionError(
            f"cannot read the Office archive: the local header of {name} names another file"
        )
    start = offset + 30 + name_length + extra_length
    stored = memoryview(data)[start : start + info.compress_size]
    if len(stored) != info.compress_size:
        raise ConversionError(f"cannot read the Office archive: {name} is truncated")
    size, crc = 0, 0

    def count(output: bytes) -> None:
        nonlocal size, crc
        size += len(output)
        crc = zlib.crc32(output, crc)
        if size > budget:
            raise ConversionError(
                f"the Office archive inflates to more than {MAX_OFFICE_ARCHIVE_BYTES} bytes;"
                f" the limit is {MAX_OFFICE_ARCHIVE_BYTES}"
            )
        if size > info.file_size:
            raise ConversionError(
                f"{name} inflates to more than the {info.file_size} bytes its entry declares"
            )

    chunks = (
        stored[offset : offset + ARCHIVE_CHUNK_BYTES]
        for offset in range(0, len(stored), ARCHIVE_CHUNK_BYTES)
    )
    if info.compress_type == zipfile.ZIP_STORED:
        for chunk in chunks:
            count(bytes(chunk))
    else:
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        try:
            for chunk in chunks:
                pending: bytes | memoryview = chunk
                while pending and not inflater.eof:
                    count(inflater.decompress(pending, ARCHIVE_CHUNK_BYTES))
                    pending = inflater.unconsumed_tail
                if inflater.eof:
                    break
            while not inflater.eof and (output := inflater.decompress(b"", ARCHIVE_CHUNK_BYTES)):
                count(output)
        except zlib.error as exc:
            raise ConversionError(f"cannot read the Office archive: {name}: {exc}") from exc
        if not inflater.eof:
            raise ConversionError(f"cannot read the Office archive: {name} is truncated")
    if size != info.file_size or crc != info.CRC:
        raise ConversionError(
            f"{name} does not inflate to the {info.file_size} bytes and CRC its entry declares"
        )
    return size


def _read_xml(archive: zipfile.ZipFile, name: str) -> ElementTree.Element | None:
    """The parsed member `name`, or None when the archive has no such member."""
    try:
        info = archive.getinfo(name)
    except KeyError:
        return None
    try:
        return ElementTree.fromstring(archive.read(info))
    except ARCHIVE_FAILURES as exc:
        raise ConversionError(f"cannot read {name}: {exc}") from exc


def _rels_name(part: str) -> str:
    return join_path(dirname(part), "_rels", part.rsplit("/", 1)[-1] + ".rels")


def _relationships(archive: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """Each internal relationship of `part` by ID: (type, target member name)."""
    root = _read_xml(archive, _rels_name(part))
    if root is None:
        return {}
    found: dict[str, tuple[str, str]] = {}
    for rel in root.iter(f"{_PKG_REL}Relationship"):
        if rel.get("TargetMode") == "External":
            continue
        target = rel.get("Target", "")
        name = target.lstrip("/") if target.startswith("/") else join_path(dirname(part), target)
        found[rel.get("Id", "")] = (rel.get("Type", ""), normpath(name))
    return found


def _related(rels: Mapping[str, tuple[str, str]], kind: str) -> list[str]:
    """Targets of the relationships `rels` whose type ends with `/<kind>`."""
    return [target for rel_type, target in rels.values() if rel_type.endswith(kind)]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _texts(element: ElementTree.Element, local_name: str = "t") -> str:
    """The text of every descendant named `local_name` in any namespace, joined."""
    return "".join(e.text or "" for e in element.iter() if _local(e.tag) == local_name)


def _paragraph_lines(element: ElementTree.Element) -> str:
    """The text of each paragraph (`p` in any namespace) of `element` on its own line."""
    paragraphs = [p for p in element.iter() if _local(p.tag) == "p"]
    if not paragraphs:
        return _texts(element).strip()
    return "\n".join(_texts(p).strip() for p in paragraphs).strip()


def _when(value: str | None) -> str:
    """The calendar date of an OOXML timestamp, or empty."""
    return value[:10] if value and re.match(r"\d{4}-\d{2}-\d{2}", value) else ""


def _by(kind: str, author: str | None, date: str | None) -> str:
    label = f"{kind} by {author}" if author else kind
    return f"{label} ({_when(date)})" if _when(date) else label


def _escape_paragraph(text: str) -> str:
    """One line of text as a Markdown paragraph that cannot read as a heading, quote, table
    or fence."""
    line = " ".join(text.split())
    return f"\\{line}" if _MARKDOWN_OPENER.match(line) else line


# Docling.


def _excel_document(data: bytes) -> DoclingDocument:
    """A workbook converted by Docling's Excel backend, constructed directly on a byte stream;
    it neither renders pages nor runs LibreOffice."""
    from docling.backend.msexcel_backend import MsExcelDocumentBackend
    from docling.datamodel.backend_options import MsExcelBackendOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    try:
        document = InputDocument(
            path_or_stream=io.BytesIO(data),
            format=InputFormat.XLSX,
            backend=MsExcelDocumentBackend,
            filename="workbook.xlsx",
            backend_options=MsExcelBackendOptions.model_validate(
                {
                    "treat_singleton_as_text": True,
                    "parse_charts": False,
                    "render_chart_images": False,
                    "max_total_bytes": MAX_OFFICE_ARCHIVE_BYTES,
                    "max_member_count": MAX_OFFICE_ARCHIVE_ENTRIES,
                }
            ),
        )
        backend = document._backend
        if not isinstance(backend, MsExcelDocumentBackend):
            raise ConversionError("the workbook could not be loaded")
        # EMF and WMF images would otherwise be converted with LibreOffice.
        backend.xlsx_to_pdf_converter_init = True
        backend.xlsx_to_pdf_converter = None
        return backend.convert()
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"cannot convert the workbook: {exc}") from exc


def _html_document(data: bytes) -> DoclingDocument:
    """A page converted by Docling's HTML backend, constructed directly on a byte stream; it
    fetches nothing and renders nothing."""
    from docling.backend.html_backend import HTMLDocumentBackend
    from docling.datamodel.backend_options import HTMLBackendOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    try:
        document = InputDocument(
            path_or_stream=io.BytesIO(data),
            format=InputFormat.HTML,
            backend=HTMLDocumentBackend,
            filename="page.html",
            backend_options=HTMLBackendOptions.model_validate(
                {
                    "fetch_images": False,
                    "enable_remote_fetch": False,
                    "enable_local_fetch": False,
                    "render_page": False,
                    "source_uri": None,
                    # Inferred furniture would move everything before the first heading out
                    # of the body.
                    "infer_furniture": False,
                }
            ),
        )
        backend = document._backend
        if not isinstance(backend, HTMLDocumentBackend) or not backend.is_valid():
            raise ConversionError("the page could not be loaded")
        return backend.convert()
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"cannot convert the page: {exc}") from exc


def _children(document: DoclingDocument, node: NodeItem) -> Iterator[NodeItem]:
    for ref in node.children:
        yield ref.resolve(document)


def _table_grid(table: TableItem) -> list[list[str]]:
    """The table's cells by row and column; a merged cell's text fills every slot it spans."""
    data = table.data
    grid = [[""] * data.num_cols for _ in range(data.num_rows)]
    for cell in data.table_cells:
        for row in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
            for col in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                grid[row][col] = cell.text
    return grid


_BRACKETED = re.compile(r"\(([^()]+)\)|\[([^\[\]]+)\]")
# What an accounting cell holds in brackets: a number, maybe signed, with a currency sign or a
# percent sign, such as `(120.50)`, `($1,234)` or `(12%)`.
_BRACKETED_NUMBER = re.compile(r"[\s+\-\u2212$\u20ac\u00a3\u00a5]*\d[\d.,\s]*%?")


def _is_unit_cell(text: str) -> bool:
    """Whether `text` is a unit in paired brackets, such as `(GBP)`, `[m]` or `(%)`, and not
    a bracketed number."""
    match = _BRACKETED.fullmatch(text)
    return match is not None and not _BRACKETED_NUMBER.fullmatch(match[1] or match[2])


def _is_number_cell(text: str) -> bool:
    """Whether `text` is a number, bare or in brackets as an accounting cell writes it."""
    match = _BRACKETED.fullmatch(text)
    inner = (match[1] or match[2]) if match else text
    return _BRACKETED_NUMBER.fullmatch(inner) is not None


def _is_label_row(row: Sequence[str]) -> bool:
    """Whether `row` has a filled cell and every filled cell is text, none a number."""
    filled = [cell.strip() for cell in row if cell.strip()]
    return bool(filled) and not any(_is_number_cell(cell) for cell in filled)


def _header_rows(table: TableItem, grid: Sequence[Sequence[str]]) -> tuple[int, int]:
    """(title rows, header rows) at the top of a table. A title row is one merged cell across
    the whole width. The header is the first row after the titles, deepened to the last row
    any of its merged cells reaches, to two rows when a cell of it spans columns and a data
    row would remain, then by each row whose filled cells are all units in brackets."""
    data = table.data
    titles = 0
    while titles < len(grid) - 1 and any(
        c.start_row_offset_idx == titles and c.col_span == data.num_cols and data.num_cols > 1
        for c in data.table_cells
    ):
        titles += 1
    depth = max(
        (
            c.end_row_offset_idx - titles
            for c in data.table_cells
            if c.start_row_offset_idx == titles and c.row_span > 1
        ),
        default=1,
    )
    if len(grid) - titles >= 3 and any(
        c.start_row_offset_idx == titles and c.col_span > 1 for c in data.table_cells
    ):
        depth = max(depth, 2)
    return titles, _with_unit_rows(grid, titles, min(depth, len(grid) - titles))


def _with_unit_rows(grid: Sequence[Sequence[str]], titles: int, depth: int) -> int:
    """`depth` header rows deepened by each following row whose filled cells are all units in
    brackets, while a data row remains."""
    while titles + depth < len(grid) - 1:
        filled = [cell for cell in grid[titles + depth] if cell.strip()]
        if not filled or not all(_is_unit_cell(cell.strip()) for cell in filled):
            break
        depth += 1
    return depth


def _declared_header_rows(table: TableItem, grid: Sequence[Sequence[str]]) -> tuple[int, int]:
    """(title rows, header rows) of a table whose source marks its header cells, as HTML
    `<th>` and `<thead>` do: the header is every leading row whose cells are all marked
    column headers, a full-width row of it a title while a header row follows, then a row of
    text labels under a header cell that spans columns, then rows of units in brackets. A
    table that marks none is read by `_header_rows`."""
    data = table.data
    marked = 0
    while marked < len(grid) - 1 and all(
        c.column_header
        for c in data.table_cells
        if c.start_row_offset_idx <= marked < c.end_row_offset_idx
    ):
        marked += 1
    if not marked:
        return _header_rows(table, grid)
    titles = 0
    while titles < marked - 1 and any(
        c.start_row_offset_idx == titles and c.col_span == data.num_cols and data.num_cols > 1
        for c in data.table_cells
    ):
        titles += 1
    depth = marked - titles
    # A header cell spanning columns names a group; a row of text labels under it, such as
    # `<td>` labels in a `<thead>`, completes the header while a data row follows.
    groups = any(
        c.col_span > 1 and titles <= c.start_row_offset_idx < marked for c in data.table_cells
    )
    if groups and marked < len(grid) - 1 and _is_label_row(grid[marked]):
        depth += 1
    return titles, _with_unit_rows(grid, titles, depth)


def _column_names(grid: Sequence[Sequence[str]], first: int, depth: int) -> list[str]:
    """One name per column from `depth` header rows: the distinct texts top to bottom joined
    `Group \u2014 Column`, a bracketed unit added after the name it follows."""
    names: list[str] = []
    for col in range(len(grid[0]) if grid else 0):
        parts: list[str] = []
        for row in range(first, first + depth):
            text = " ".join(grid[row][col].split())
            if not text or (parts and parts[-1].endswith(text)):
                continue
            if parts and _is_unit_cell(text):
                parts[-1] = f"{parts[-1]} {text}"
            else:
                parts.append(text)
        names.append(" \u2014 ".join(parts))
    return names


def _flat_table(
    table: TableItem,
    fill: Mapping[tuple[int, int], str],
    header_rows: Callable[[TableItem, Sequence[Sequence[str]]], tuple[int, int]],
) -> tuple[list[str], str]:
    """The title lines and the GFM table of a Docling table with flattened headers; `fill`
    gives text for empty cells by (row, column) offset, and `header_rows` finds the title and
    header rows."""
    grid = _table_grid(table)
    if not grid or not grid[0]:
        return [], ""
    for (row, col), text in fill.items():
        if row < len(grid) and col < len(grid[row]) and not grid[row][col].strip():
            grid[row][col] = text
    titles, depth = header_rows(table, grid)
    title_lines = [" ".join(grid[r][0].split()) for r in range(titles) if grid[r][0].strip()]
    headers = _column_names(grid, titles, depth)
    return title_lines, render_table(headers, grid[titles + depth :])


# Excel.


@dataclass(frozen=True)
class _Sheet:
    name: str
    part: str


def _workbook_sheets(archive: zipfile.ZipFile, rels: Mapping[str, tuple[str, str]]) -> list[_Sheet]:
    workbook = _read_xml(archive, "xl/workbook.xml")
    if workbook is None:
        raise ConversionError("the workbook has no xl/workbook.xml")
    sheets: list[_Sheet] = []
    for sheet in workbook.iter():
        if _local(sheet.tag) != "sheet":
            continue
        rel_type, part = rels.get(sheet.get(f"{_REL}id", ""), ("", ""))
        if rel_type.endswith("/worksheet"):
            sheets.append(_Sheet(sheet.get("name", ""), part))
    return sheets


def _scan_cells(archive: zipfile.ZipFile, sheets: Sequence[_Sheet]) -> dict[str, dict[str, str]]:
    """Count the populated cells of every worksheet against MAX_WORKBOOK_CELLS, streaming
    each sheet's XML, and return the formula (`=...`) of each formula cell without a cached
    value, by sheet name and cell reference. A cell without `r` takes the column after the
    previous cell of its row, as a row without `r` takes the row after the previous row.

    A cell of a shared formula gets the formula of the group's master cell, the one whose
    `f` carries the group's `ref` range, translated to it; any text of its own is ignored,
    as ECMA-376 directs."""
    from openpyxl.formula.tokenizer import TokenizerError
    from openpyxl.formula.translate import Translator, TranslatorError

    total = 0
    formulas: dict[str, dict[str, str]] = {}
    for sheet in sheets:
        # Shared formula index -> (formula, master cell); uncached cells of a shared group.
        masters: dict[str, tuple[str, str]] = {}
        sharing: list[tuple[str, str]] = []
        found: dict[str, str] = {}
        row, column = 0, 0
        try:
            with archive.open(sheet.part) as stream:
                for event, element in ElementTree.iterparse(stream, ("start", "end")):
                    tag = _local(element.tag)
                    if event == "start":
                        if tag == "row":
                            row = _row_number(element.get("r"), row + 1)
                            column = 0
                        continue
                    if tag != "c":
                        continue
                    ref = element.get("r")
                    if ref is None:
                        column += 1
                        ref = f"{_column_letters(column)}{row}"
                    else:
                        row, column = _cell_position(ref)
                    formula = _child(element, "f")
                    value = _child(element, "v")
                    # A formula written without recalculation can carry an empty value; an
                    # empty value of a string formula is its computed empty string.
                    cached = _child(element, "is") is not None or (
                        value is not None and (bool(value.text) or element.get("t") == "str")
                    )
                    if cached or formula is not None:
                        total += 1
                        if total > MAX_WORKBOOK_CELLS:
                            raise ConversionError(
                                f"the workbook has more than {MAX_WORKBOOK_CELLS} populated"
                                " cells; it is refused rather than cut"
                            )
                    if formula is not None:
                        text = f"={formula.text}" if formula.text else ""
                        index = formula.get("si") if formula.get("t") == "shared" else None
                        if index is not None and formula.get("ref") is not None and text:
                            masters.setdefault(index, (text, ref))
                        elif index is not None:
                            text = ""
                        if not cached and text:
                            found[ref] = text
                        elif not cached and index is not None:
                            sharing.append((ref, index))
                    element.clear()
            translators: dict[str, Translator] = {}
            for ref, index in sharing:
                if index not in masters:
                    raise ConversionError(f"{ref} uses shared formula {index}, which no cell holds")
                if index not in translators:
                    text, origin = masters[index]
                    translators[index] = Translator(text, origin=origin)
                found[ref] = translators[index].translate_formula(ref)
        except ConversionError:
            raise
        except (*ARCHIVE_FAILURES, TranslatorError, TokenizerError) as exc:
            raise ConversionError(f"cannot read {sheet.part}: {exc}") from exc
        if found:
            formulas[sheet.name] = dict(sorted(found.items(), key=lambda f: _cell_position(f[0])))
    return formulas


def _row_number(value: str | None, implied: int) -> int:
    """A worksheet `row/@r`, or `implied` when it has none; rows count from 1."""
    if value is None:
        return implied
    if not re.fullmatch(r"[1-9]\d*", value):
        raise ConversionError(f"{value!r} is not a row number")
    return int(value)


def _column_number(letters: str) -> int:
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - ord("A") + 1
    return number


def _cell_position(ref: str) -> tuple[int, int]:
    """(row, column) of an A1 reference, both from 1."""
    match = _CELL_REF.fullmatch(ref)
    if match is None:
        raise ConversionError(f"{ref!r} is not a cell reference")
    return int(match[2]), _column_number(match[1])


def _column_letters(number: int) -> str:
    letters = ""
    while number:
        number, rest = divmod(number - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return letters


def _persons(
    archive: zipfile.ZipFile, workbook_rels: Mapping[str, tuple[str, str]]
) -> dict[str, str]:
    """The display name of each person a threaded comment can name, by person ID."""
    persons: dict[str, str] = {}
    for part in _related(workbook_rels, "/person"):
        if (root := _read_xml(archive, part)) is not None:
            persons |= {
                p.get("id", ""): p.get("displayName", "")
                for p in root.iter()
                if _local(p.tag) == "person"
            }
    return persons


def _sheet_comments(
    archive: zipfile.ZipFile, sheet: _Sheet, persons: Mapping[str, str]
) -> list[Aside]:
    """The notes and threaded comments of one worksheet, in document order. A note Excel
    writes beside a threaded comment for older versions is left out."""
    asides: list[Aside] = []
    threaded_cells: set[str] = set()
    rels = _relationships(archive, sheet.part)
    for part in _related(rels, "/threadedComment"):
        root = _read_xml(archive, part)
        if root is None:
            continue
        for comment in root.iter():
            if _local(comment.tag) != "threadedComment":
                continue
            ref = comment.get("ref", "")
            threaded_cells.add(ref)
            kind = "reply" if comment.get("parentId") else "comment"
            author = persons.get(comment.get("personId", ""))
            label = f"Sheet {sheet.name}: {ref} {_by(kind, author, comment.get('dT'))}"
            asides.append(Aside(label, _texts(comment, "text").strip(), ""))
    for part in _related(rels, "/comments"):
        root = _read_xml(archive, part)
        if root is None:
            continue
        authors = [a.text or "" for a in root.iter() if _local(a.tag) == "author"]
        for comment in root.iter():
            if _local(comment.tag) != "comment" or comment.get("ref", "") in threaded_cells:
                continue
            index = comment.get("authorId", "")
            author = authors[int(index)] if index.isdigit() and int(index) < len(authors) else None
            label = f"Sheet {sheet.name}: {comment.get('ref', '')} {_by('note', author, None)}"
            asides.append(Aside(label, _texts(comment).strip(), ""))
    return asides


def xlsx_markdown(data: bytes) -> str:
    """An Excel workbook (`.xlsx`) as Markdown: one level-1 heading per worksheet in workbook
    order, then each data region as a GFM table under its range, flattened headers and all,
    then the sheet's notes and threaded comments as annotations.

    A formula cell whose file holds no computed value shows its formula, so a workbook saved
    without recalculation never reads as empty cells.
    """
    with open_office_archive(data) as archive:
        workbook_rels = _relationships(archive, "xl/workbook.xml")
        sheets = _workbook_sheets(archive, workbook_rels)
        formulas = _scan_cells(archive, sheets)
        persons = _persons(archive, workbook_rels)
        comments = {sheet.name: _sheet_comments(archive, sheet, persons) for sheet in sheets}
    document = _excel_document(data)
    from docling_core.types.doc.items.group import GroupItem
    from docling_core.types.doc.items.table.table import TableItem
    from docling_core.types.doc.items.text import TextItem
    from docling_core.types.doc.labels import GroupLabel

    blocks: list[str] = []
    for node in _children(document, document.body):
        # Docling also files each cell comment as a group; comments are read from the archive.
        if not isinstance(node, GroupItem) or node.label != GroupLabel.SHEET:
            continue
        name = node.name
        blocks.append(f"# {' '.join(name.split()) or 'Sheet'}")
        shown: set[str] = set()
        for item in _children(document, node):
            if isinstance(item, TableItem) and item.prov:
                box = item.prov[0].bbox
                left, top = int(box.l), int(box.t)
                fill: dict[tuple[int, int], str] = {}
                for ref, formula in formulas.get(name, {}).items():
                    row, col = _cell_position(ref)
                    if top < row <= int(box.b) and left < col <= int(box.r):
                        fill[row - 1 - top, col - 1 - left] = formula
                        shown.add(ref)
                titles, table = _flat_table(item, fill, _header_rows)
                blocks += [_escape_paragraph(t) for t in titles]
                first = f"{_column_letters(left + 1)}{top + 1}"
                last = f"{_column_letters(int(box.r))}{int(box.b)}"
                blocks += [f"Range {first}:{last}", table]
            elif isinstance(item, TextItem) and item.text.strip():
                blocks.append(_escape_paragraph(item.text))
        blocks += [
            _escape_paragraph(f"{ref}: {formula}")
            for ref, formula in formulas.get(name, {}).items()
            if ref not in shown
        ]
        blocks += [aside.markdown for aside in comments.get(name, [])]
    return "\n\n".join(block for block in blocks if block)


# HTML.


def html_markdown(data: bytes) -> str:
    """An HTML page as Markdown in document order: headings, paragraphs, lists, code blocks
    and tables. Images are never fetched, and text before the first heading stays body text."""
    document = _html_document(data)
    from docling_core.types.doc.common.content_layer import ContentLayer
    from docling_core.types.doc.items.code import CodeItem
    from docling_core.types.doc.items.group import InlineGroup, ListGroup
    from docling_core.types.doc.items.node import DocItem
    from docling_core.types.doc.items.picture.picture import PictureItem
    from docling_core.types.doc.items.table.table import TableItem
    from docling_core.types.doc.items.text import ListItem, SectionHeaderItem, TextItem, TitleItem

    captions = {
        ref.cref
        for item, _ in document.iterate_items()
        if isinstance(item, TableItem)
        for ref in item.captions
    }
    blocks: list[str] = []

    def inline_text(node: NodeItem) -> str:
        if isinstance(node, TextItem):
            return node.text
        return " ".join(inline_text(child) for child in _children(document, node))

    def list_lines(group: NodeItem, depth: int) -> list[str]:
        lines: list[str] = []
        number = 0
        for child in _children(document, group):
            if isinstance(child, ListItem):
                number += 1
                marker = f"{number}." if child.enumerated else "-"
                text = child.text or " ".join(
                    inline_text(c)
                    for c in _children(document, child)
                    if not isinstance(c, ListGroup)
                )
                lines.append(f"{'   ' * depth}{marker} {' '.join(text.split())}")
                for nested in _children(document, child):
                    if isinstance(nested, ListGroup):
                        lines += list_lines(nested, depth + 1)
            elif isinstance(child, ListGroup):
                lines += list_lines(child, depth + 1)
        return lines

    def walk(node: NodeItem) -> None:
        for item in _children(document, node):
            if isinstance(item, DocItem) and item.content_layer != ContentLayer.BODY:
                continue
            if isinstance(item, InlineGroup):
                if text := inline_text(item).strip():
                    blocks.append(_escape_paragraph(text))
                continue
            if isinstance(item, ListGroup):
                blocks.append("\n".join(list_lines(item, 0)))
                continue
            if isinstance(item, TitleItem):
                blocks.append(f"# {' '.join(item.text.split())}")
            elif isinstance(item, SectionHeaderItem):
                blocks.append(f"{'#' * min(item.level + 1, 6)} {' '.join(item.text.split())}")
            elif isinstance(item, TableItem):
                titles, table = _flat_table(item, {}, _declared_header_rows)
                caption = " ".join(c.resolve(document).text for c in item.captions)
                blocks.extend(_escape_paragraph(t) for t in [caption, *titles] if t.strip())
                blocks.append(table)
            elif isinstance(item, CodeItem):
                fence = "```"
                while fence in item.text:
                    fence += "`"
                blocks.append(f"{fence}\n{item.text}\n{fence}")
            elif isinstance(item, PictureItem):
                continue
            elif isinstance(item, TextItem) and item.self_ref not in captions and item.text.strip():
                blocks.append(_escape_paragraph(item.text))
            walk(item)

    walk(document.body)
    return "\n\n".join(block for block in blocks if block)


# Word.


_W_REMOVED = {f"{_W}del", f"{_W}moveFrom"}
_W_UNWRAPPED = {f"{_W}ins", f"{_W}moveTo"}
_WORD_TEXT_PARTS = re.compile(r"word/(document|header\d*|footer\d*|footnotes|endnotes)\.xml")


def _accept_revisions(element: etree._Element) -> bool:
    """Accept every tracked change below `element` in place: deleted and moved-away content
    and deleted table rows go, inserted and moved-in content stays without its revision
    wrapper. Whether anything changed. A row is deleted by its own `w:trPr`, never by one of
    a table nested in it."""
    changed = False
    children: list[etree._Element] = []
    for child in element:
        if child.tag in _W_REMOVED or (
            child.tag == f"{_W}tr"
            and (row_properties := child.find(f"{_W}trPr")) is not None
            and row_properties.find(f"{_W}del") is not None
        ):
            changed = True
            continue
        changed |= _accept_revisions(child)
        if child.tag in _W_UNWRAPPED:
            changed = True
            children += [_keep_declarations(inner, element) for inner in child]
        else:
            children.append(child)
    if changed:
        element[:] = children
    return changed


def _keep_declarations(inner: etree._Element, parent: etree._Element) -> etree._Element:
    """`inner`, a child of a revision wrapper about to move up to `parent`, declaring every
    prefix in scope at it (the wrapper's included) that `parent` does not bind the same way.
    lxml carries the declarations an element's own names use, but not a prefix only an
    attribute value names, as `mc:Choice/@Requires` and `mc:Ignorable` do."""
    if not isinstance(inner.tag, str):
        return inner
    missing = {
        prefix: uri for prefix, uri in inner.nsmap.items() if parent.nsmap.get(prefix) != uri
    }
    if not missing:
        return inner
    copy = etree.Element(inner.tag, dict(inner.attrib), nsmap=missing)
    copy.text, copy.tail = inner.text, inner.tail
    copy.extend(list(inner))
    return copy


def accepted_revision_copy(data: bytes) -> bytes:
    """A Word file (`.docx`) with every tracked change accepted in its body, headers,
    footers, footnotes and endnotes; `data` itself when it tracks none.

    LibreOffice draws a tracked deletion and its replacement both into a PDF's text, so
    `2176019` replaced by `1998` would read `21760191998`; the rendered copy carries only the
    accepted text. The original bytes stay the document's identity and metadata source.
    """
    with open_office_archive(data) as archive:
        try:
            members = [(info, archive.read(info)) for info in archive.infolist()]
        except ARCHIVE_FAILURES as exc:
            raise ConversionError(f"cannot read the Word archive: {exc}") from exc
    # lxml keeps each element's own namespace declarations, so every prefix, including one
    # only `mc:Ignorable` names, is written back where its source declared it.
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    replaced: dict[str, bytes] = {}
    for info, raw in members:
        if not _WORD_TEXT_PARTS.fullmatch(info.filename):
            continue
        try:
            root = etree.fromstring(raw, parser)
        except etree.XMLSyntaxError as exc:
            raise ConversionError(f"cannot read {info.filename}: {exc}") from exc
        if _accept_revisions(root):
            replaced[info.filename] = etree.tostring(
                root, xml_declaration=True, encoding="UTF-8", standalone=True
            )
    if not replaced:
        return data
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as copy:
        for info, raw in members:
            copy.writestr(info, replaced.get(info.filename, raw))
    return buffer.getvalue()


def _accepted_text(element: ElementTree.Element) -> str:
    """The text of `element` once its tracked changes are accepted."""
    if element.tag in _W_REMOVED:
        return ""
    text = element.text or "" if element.tag == f"{_W}t" else ""
    if element.tag == f"{_W}tab":
        text = " "
    return text + "".join(_accepted_text(child) for child in element)


def docx_asides(data: bytes) -> list[Aside]:
    """The reviewer comments and tracked changes of a Word file's body, in document order:
    each tracked insertion and deletion with its author, date and text, anchored to its
    paragraph's accepted text, then each comment anchored to the text it comments on."""
    with open_office_archive(data) as archive:
        document = _read_xml(archive, "word/document.xml")
        rels = _relationships(archive, "word/document.xml")
        comments_part = next(iter(_related(rels, "/comments")), None)
        comments = None if comments_part is None else _read_xml(archive, comments_part)
    if document is None:
        raise ConversionError("the Word file has no word/document.xml")
    asides: list[Aside] = []
    anchors: dict[str, list[str]] = {}
    open_ranges: set[str] = set()

    def visit(element: ElementTree.Element, paragraph: str) -> None:
        if element.tag == f"{_W}p":
            paragraph = " ".join(_accepted_text(element).split())
        if element.tag in (f"{_W}p", f"{_W}tab", f"{_W}br"):
            # A range over several paragraphs, or across a tab or line break, keeps its words
            # apart.
            for comment_id in open_ranges:
                anchors[comment_id].append(" ")
        if element.tag == f"{_W}commentRangeStart":
            open_ranges.add(element.get(f"{_W}id", ""))
            anchors.setdefault(element.get(f"{_W}id", ""), [])
        elif element.tag == f"{_W}commentRangeEnd":
            open_ranges.discard(element.get(f"{_W}id", ""))
        elif element.tag == f"{_W}t" and element.text:
            for comment_id in open_ranges:
                anchors[comment_id].append(element.text)
        if element.tag in (f"{_W}ins", f"{_W}del"):
            changed = _texts(element, "t" if element.tag == f"{_W}ins" else "delText").strip()
            if changed:
                kind = "Tracked insertion" if element.tag == f"{_W}ins" else "Tracked deletion"
                label = _by(kind, element.get(f"{_W}author"), element.get(f"{_W}date"))
                asides.append(Aside(label, changed, paragraph))
        for child in element:
            visit(child, paragraph)

    visit(document, "")
    for comment in [] if comments is None else comments.iter(f"{_W}comment"):
        comment_id = comment.get(f"{_W}id", "")
        anchor = " ".join("".join(anchors.get(comment_id, [])).split())
        label = _by("Comment", comment.get(f"{_W}author"), comment.get(f"{_W}date"))
        if anchor:
            shown = anchor[:ANCHOR_LABEL_CHARS] + (
                "\u2026" if len(anchor) > ANCHOR_LABEL_CHARS else ""
            )
            label = f"{label} on \u201c{shown}\u201d"
        asides.append(Aside(label, _paragraph_lines(comment), anchor))
    return asides


# PowerPoint.


def _shown_slides(archive: zipfile.ZipFile) -> list[str]:
    """The slide parts of a presentation in slide order, hidden slides left out as
    LibreOffice leaves them out of a PDF."""
    presentation = _read_xml(archive, "ppt/presentation.xml")
    if presentation is None:
        raise ConversionError("the presentation has no ppt/presentation.xml")
    rels = _relationships(archive, "ppt/presentation.xml")
    slides: list[str] = []
    for slide_id in presentation.iter():
        if _local(slide_id.tag) != "sldId":
            continue
        rel_type, part = rels.get(slide_id.get(f"{_REL}id", ""), ("", ""))
        if not rel_type.endswith("/slide"):
            continue
        slide = _read_xml(archive, part)
        if slide is not None and slide.get("show", "1").lower() not in ("0", "false"):
            slides.append(part)
    return slides


def _comment_authors(archive: zipfile.ZipFile) -> dict[str, str]:
    """The name of each comment author by ID, from the legacy and modern author parts the
    presentation relates."""
    rels = _relationships(archive, "ppt/presentation.xml")
    authors: dict[str, str] = {}
    for part in _related(rels, "/commentAuthors") + _related(rels, "/authors"):
        root = _read_xml(archive, part)
        if root is not None:
            authors |= {
                a.get("id", ""): a.get("name", "")
                for a in root.iter()
                if _local(a.tag) in ("cmAuthor", "author")
            }
    return authors


def _speaker_notes(archive: zipfile.ZipFile, part: str) -> str:
    """The text of the body placeholders of a notes slide."""
    root = _read_xml(archive, part)
    if root is None:
        return ""
    texts: list[str] = []
    for shape in root.iter():
        if _local(shape.tag) != "sp":
            continue
        kinds = {p.get("type") for p in shape.iter() if _local(p.tag) == "ph"}
        if "body" in kinds:
            texts.append(_paragraph_lines(shape))
    return "\n".join(t for t in texts if t)


def _series_points(element: ElementTree.Element | None) -> dict[int, str]:
    """The cached points of a chart data reference by index."""
    if element is None:
        return {}
    return {
        int(pt.get("idx", "0")): _texts(pt, "v")
        for pt in element.iter()
        if _local(pt.tag) == "pt" and pt.get("idx", "").isdigit()
    }


def _child(element: ElementTree.Element, *names: str) -> ElementTree.Element | None:
    return next((c for c in element if _local(c.tag) in names), None)


def _chart_table(root: ElementTree.Element, number: int) -> ChartTable | None:
    """The series of a chart part as a table, or None when it holds no cached data. Series
    that share one set of categories (or x values) are columns beside a category column;
    series with x values of their own, as a scatter chart's can be, or with none cached
    beside series that have them, are rows of a series, x and y table, so no value is ever
    placed against another series' x value."""
    chart = _child(root, "chart")
    if chart is None:
        return None
    title_element = _child(chart, "title")
    title = " ".join(_texts(title_element).split()) if title_element is not None else ""
    names: list[str] = []
    columns: list[dict[int, str]] = []
    axes: list[dict[int, str]] = []
    scatter = False
    for plot in chart.iter():
        if _local(plot.tag) != "ser":
            continue
        name = _child(plot, "tx")
        names.append(
            " ".join(_texts(name, "v").split()) if name is not None else f"Series {len(names) + 1}"
        )
        columns.append(_series_points(_child(plot, "val", "yVal")))
        axis = _child(plot, "cat", "xVal")
        scatter |= axis is not None and _local(axis.tag) == "xVal"
        axes.append(_series_points(axis))
    if not columns or not any(columns):
        return None
    caption = f"Chart: {title}" if title else f"Chart {number}"

    def place(axis: Mapping[int, str], index: int) -> str:
        # A missing x value or category is blank: never a point number that would read as
        # one, except for a category chart that caches no categories at all.
        return axis.get(index, "" if scatter or axis else str(index + 1))

    if all(axis == axes[0] for axis in axes):
        categories = axes[0]
        indexes = sorted(set(categories).union(*columns))
        rows = tuple(
            (place(categories, i), *(column.get(i, "") for column in columns)) for i in indexes
        )
        return ChartTable(caption, ("Category", *names), rows)
    # A series without cached x values has blank ones, never another series' values.
    pairs = tuple(
        (name, place(axis, i), column.get(i, ""))
        for name, axis, column in zip(names, axes, columns, strict=True)
        for i in sorted(set(axis).union(column))
    )
    headers = ("Series", "X", "Y") if scatter else ("Series", "Category", "Value")
    return ChartTable(caption, headers, pairs)


def _slide_comments(
    root: ElementTree.Element, authors: Mapping[str, str], number: int
) -> list[Aside]:
    """The comments of one slide's comment part, legacy or modern, in order; each reply of a
    modern comment follows it under its own author and date."""
    asides: list[Aside] = []

    def add(kind: str, element: ElementTree.Element, text: str) -> None:
        author = authors.get(element.get("authorId", ""))
        when = element.get("dt") or element.get("created")
        asides.append(Aside(f"{_by(kind, author, when)}, slide {number}", text, ""))

    for comment in root.iter():
        if _local(comment.tag) != "cm":
            continue
        body = _child(comment, "txBody")
        add(
            "Comment",
            comment,
            _texts(comment, "text").strip() if body is None else _paragraph_lines(body),
        )
        for replies in (c for c in comment if _local(c.tag) == "replyLst"):
            for reply in (r for r in replies if _local(r.tag) == "reply"):
                reply_body = _child(reply, "txBody")
                add("Reply", reply, "" if reply_body is None else _paragraph_lines(reply_body))
    return asides


def pptx_supplements(data: bytes) -> list[SlideSupplement]:
    """For each shown slide of a PowerPoint file, in order: its native charts as tables, its
    speaker notes and its comments (legacy and modern) as annotations."""
    supplements: list[SlideSupplement] = []
    with open_office_archive(data) as archive:
        authors = _comment_authors(archive)
        for number, part in enumerate(_shown_slides(archive), start=1):
            rels = _relationships(archive, part)
            charts: list[ChartTable] = []
            for chart_part in _related(rels, "/chart"):
                root = _read_xml(archive, chart_part)
                table = None if root is None else _chart_table(root, len(charts) + 1)
                if table is not None:
                    charts.append(table)
            asides: list[Aside] = []
            for notes_part in _related(rels, "/notesSlide"):
                if notes := _speaker_notes(archive, notes_part):
                    asides.append(Aside(f"Speaker notes, slide {number}", notes, ""))
            for comments_part in _related(rels, "/comments"):
                if (root := _read_xml(archive, comments_part)) is not None:
                    asides += _slide_comments(root, authors, number)
            supplements.append(SlideSupplement(number, tuple(charts), tuple(asides)))
    return supplements
