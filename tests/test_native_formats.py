"""Reading Excel workbooks and HTML pages from their own structure, and the comments, tracked
changes, speaker notes and chart data of Word and PowerPoint files, all offline."""

from __future__ import annotations

import io
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from lxml import etree
from openpyxl import Workbook
from openpyxl.comments import Comment

from arkive_rag import native_formats
from arkive_rag.compiler import (
    SourceFile,
    assemble_sections,
    compile_corpus,
    metadata_date,
    office_to_pdf,
    page_parts,
    page_supplements,
    render_pdf,
    section_pieces,
)
from arkive_rag.models import ConversionError, PageExtraction
from arkive_rag.native_formats import (
    accepted_revision_copy,
    check_office_archive,
    docx_asides,
    html_markdown,
    pptx_supplements,
    xlsx_markdown,
)
from arkive_rag.passages import standing_units, tables, without_asides
from tests.support import FakeGateway

needs_soffice = pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice missing")

_CORE = (
    '<?xml version="1.0" encoding="UTF-8"?><cp:coreProperties xmlns:cp='
    '"http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
    ' xmlns:dcterms="http://purl.org/dc/terms/"'
    ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-02-11T09:00:00Z</dcterms:modified>'
    "</cp:coreProperties>"
)
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_THREADED_NS = "http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments"


def rewrite_zip(data: bytes, edit: Callable[[dict[str, bytes]], None]) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    edit(members)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return buffer.getvalue()


def add_relationship(rels: bytes, rel_id: str, rel_type: str, target: str) -> bytes:
    entry = f'<Relationship Id="{rel_id}" Type="{rel_type}" Target="{target}"/>'
    return rels.replace(b"</Relationships>", entry.encode() + b"</Relationships>")


def workbook_bytes() -> bytes:
    """Two sheets: a budget with a title row, a merged two-row header with units, formulas
    saved without computed values, a second region beside it and a legacy note; and a staff
    list whose cell carries an Excel 365 threaded comment with a reply."""
    book = Workbook()
    budget = book.active
    assert budget is not None
    budget.title = "Budget"
    budget["A1"] = "Quarterly survey budget"
    budget.merge_cells("A1:D1")
    budget["A2"], budget["B2"], budget["D2"] = "Item", "Cost", "Total"
    budget.merge_cells("A2:A3")
    budget.merge_cells("B2:C2")
    budget.merge_cells("D2:D3")
    budget["B3"], budget["C3"] = "Labour (GBP)", "Materials (GBP)"
    budget.append(["Hydrographic survey", 4200, 1150, "=B4+C4"])
    budget.append(["Tide gauge hire", 600, 85, "=B5+C5"])
    budget["B4"].comment = Comment("Day rate agreed with the harbour master", "Rowan Vale")
    budget["G2"], budget["H2"] = "Vessel", "Draught (m)"
    budget["G3"], budget["H3"] = "Kittiwake", 1.8
    budget["A8"] = "=SUM(D4:D5)"
    staff = book.create_sheet("Staff")
    staff.append(["Name", "Role"])
    staff.append(["Imogen Hale", "Surveyor"])
    staff.append(["Tobias Wren", "Skipper"])
    buffer = io.BytesIO()
    book.save(buffer)

    def edit(members: dict[str, bytes]) -> None:
        members["docProps/core.xml"] = _CORE.encode()
        members["xl/persons/person.xml"] = (
            f'<personList xmlns="{_THREADED_NS}"><person displayName="Rowan Vale" id="{{P1}}"'
            ' userId="rv" providerId="None"/></personList>'
        ).encode()
        members["xl/threadedComments/threadedComment1.xml"] = (
            f'<ThreadedComments xmlns="{_THREADED_NS}">'
            '<threadedComment ref="B2" dT="2026-01-20T10:00:00.00" personId="{P1}" id="{C1}">'
            "<text>Confirm the skipper's ticket is current</text></threadedComment>"
            '<threadedComment ref="B2" dT="2026-01-21T10:00:00.00" personId="{P1}" id="{C2}"'
            ' parentId="{C1}"><text>Checked against the register</text></threadedComment>'
            "</ThreadedComments>"
        ).encode()
        members["xl/_rels/workbook.xml.rels"] = add_relationship(
            members["xl/_rels/workbook.xml.rels"],
            "rIdPerson",
            "http://schemas.microsoft.com/office/2017/10/relationships/person",
            "persons/person.xml",
        )
        members["xl/worksheets/_rels/sheet2.xml.rels"] = (
            f'<Relationships xmlns="{_REL_NS}"><Relationship Id="rIdTc" Type='
            '"http://schemas.microsoft.com/office/2017/10/relationships/threadedComment"'
            ' Target="../threadedComments/threadedComment1.xml"/></Relationships>'
        ).encode()

    return rewrite_zip(buffer.getvalue(), edit)


def test_a_workbook_reads_as_sheets_of_flat_tables_with_formulas_and_comments() -> None:
    markdown = xlsx_markdown(workbook_bytes())

    assert [line for line in markdown.splitlines() if line.startswith("# ")] == [
        "# Budget",
        "# Staff",
    ]
    budget, staff = tables(markdown)[0], tables(markdown)[2]
    assert budget.caption == "Range A2:D5"
    assert budget.headers == (
        "Item",
        "Cost \u2014 Labour (GBP)",
        "Cost \u2014 Materials (GBP)",
        "Total",
    )
    # Formulas saved without a computed value show their formula, never an empty cell.
    assert budget.rows == (
        ("Hydrographic survey", "4200", "1150", "=B4+C4"),
        ("Tide gauge hire", "600", "85", "=B5+C5"),
    )
    assert "Quarterly survey budget" in markdown
    # A lone formula cell outside every table still shows its formula.
    assert "A8: =SUM(D4:D5)" in markdown.split("# Staff")[0]
    assert tables(markdown)[1].caption == "Range G2:H3"
    assert tables(markdown)[1].rows == (("Kittiwake", "1.8"),)
    assert staff.rows == (("Imogen Hale", "Surveyor"), ("Tobias Wren", "Skipper"))
    assert "> **[Annotation \u2014 Sheet Budget: B4 note by Rowan Vale]**" in markdown
    assert (
        "> **[Annotation \u2014 Sheet Staff: B2 comment by Rowan Vale (2026-01-20)]**\n"
        "> Confirm the skipper's ticket is current" in markdown
    )
    assert "Sheet Staff: B2 reply by Rowan Vale (2026-01-21)" in markdown
    # Asides are not the workbook's own text.
    assert "harbour master" not in without_asides(markdown)
    assert not any("ticket" in unit.text for unit in standing_units(markdown))


def test_a_workbook_over_the_cell_limit_is_refused_not_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_formats, "MAX_WORKBOOK_CELLS", 10)

    with pytest.raises(ConversionError, match="more than 10 populated cells; it is refused"):
        xlsx_markdown(workbook_bytes())


def test_uncached_and_shared_formulas_are_read_from_the_sheet_without_expanding_merges() -> None:
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.append(["Item", "Labour", "Materials", "Total", "Note"])
    for row, name in enumerate(("Survey", "Gauge", "Vessel"), start=2):
        sheet.append([name, row * 100, row * 10, None, None])
    buffer = io.BytesIO()
    book.save(buffer)
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

    def edit(members: dict[str, bytes]) -> None:
        xml = members["xl/worksheets/sheet1.xml"].decode()
        # D2 holds a shared formula D3 and D4 reuse; none was recalculated. E2 is a string
        # formula whose computed value is the empty string.
        cells = {
            "D2": '<c r="D2"><f t="shared" ref="D2:D4" si="0">B2+C2</f><v></v></c>',
            "D3": '<c r="D3"><f t="shared" si="0"/></c>',
            "D4": '<c r="D4"><f t="shared" si="0"/><v></v></c>',
            "E2": '<c r="E2" t="str"><f>IF(A2="","",1)</f><v/></c>',
        }
        for ref, cell in cells.items():
            row = int(ref[1:])
            before = f'<c r="C{row}" t="n"><v>{row * 10}</v></c>'
            xml = (
                xml.replace(before, before + cell)
                if ref[0] == "D"
                else xml.replace(cells[f"D{row}"], cells[f"D{row}"] + cell)
            )
        xml = xml.replace(
            "</sheetData>",
            '</sheetData><mergeCells count="1"><mergeCell ref="G10:Z20000"/></mergeCells>',
        )
        assert xml.count("<f") == 4 and f'xmlns="{ns}"' in xml
        members["xl/worksheets/sheet1.xml"] = xml.encode()

    started = time.monotonic()
    markdown = xlsx_markdown(rewrite_zip(buffer.getvalue(), edit))

    assert time.monotonic() - started < 20
    (table,) = tables(markdown)
    assert table.headers == ("Item", "Labour", "Materials", "Total", "Note")
    assert [row[3:] for row in table.rows] == [
        ("=B2+C2", ""),
        ("=B3+C3", ""),
        ("=B4+C4", ""),
    ]
    assert "IF(" not in markdown


def _workbook_with_rows(rows: str) -> bytes:
    """A one-sheet workbook whose sheet data is `rows`, a header row of inline strings
    `Item, B, C, D, E` before them."""
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.append(["placeholder"])
    buffer = io.BytesIO()
    book.save(buffer)
    header = "".join(
        f'<c r="{col}1" t="inlineStr"><is><t>{text}</t></is></c>'
        for col, text in zip("ABCDE", ("Item", "B", "C", "D", "E"), strict=True)
    )

    def edit(members: dict[str, bytes]) -> None:
        xml = members["xl/worksheets/sheet1.xml"].decode()
        start, end = xml.index("<sheetData"), xml.index("</sheetData>")
        sheet_data = f'<sheetData><row r="1">{header}</row>{rows}'
        members["xl/worksheets/sheet1.xml"] = (xml[:start] + sheet_data + xml[end:]).encode()

    return rewrite_zip(buffer.getvalue(), edit)


def test_a_shared_formula_is_translated_from_its_master_over_rows_and_columns() -> None:
    def name(text: str, ref: str = "") -> str:
        place = f' r="{ref}"' if ref else ""
        return f'<c{place} t="inlineStr"><is><t>{text}</t></is></c>'

    # The master D2 shares B2+C2 over D2:E3. E2 carries stale text of its own, which the
    # master overrides; row 3 and its cells have no `r`, so their places follow from order.
    rows = (
        f'<row r="2">{name("One", "A2")}<c r="B2"><v>1</v></c>'
        '<c r="C2"><v>2</v></c><c r="D2"><f t="shared" ref="D2:E3" si="0">B2+C2</f></c>'
        '<c r="E2"><f t="shared" si="0">STALE()</f></c></row>'
        f'<row>{name("Two")}<c><v>3</v></c><c><v>4</v></c><c><f t="shared" si="0"/></c>'
        '<c><f t="shared" si="0"/></c></row>'
    )

    (table,) = tables(xlsx_markdown(_workbook_with_rows(rows)))

    assert table.headers == ("Item", "B", "C", "D", "E")
    assert table.rows == (
        ("One", "1", "2", "=B2+C2", "=C2+D2"),
        ("Two", "3", "4", "=B3+C3", "=C3+D3"),
    )


def test_a_cell_reference_or_shared_formula_that_cannot_be_read_is_refused() -> None:
    def follower(master: str) -> str:
        return (
            f'<row r="2"><c r="D2"><f t="shared" ref="D2:D3" si="0">{master}</f></c></row>'
            '<row r="3"><c r="D3"><f t="shared" si="0"/></c></row>'
        )

    for ref in ("$D$2", "D0", "d2"):
        with pytest.raises(ConversionError, match=f"'{re.escape(ref)}' is not a cell reference"):
            xlsx_markdown(_workbook_with_rows(f'<row r="2"><c r="{ref}"><v>1</v></c></row>'))
    with pytest.raises(ConversionError, match="'0' is not a row number"):
        xlsx_markdown(_workbook_with_rows('<row r="0"><c><v>1</v></c></row>'))
    with pytest.raises(ConversionError, match="while parsing string"):
        xlsx_markdown(_workbook_with_rows(follower('"abc')))
    with pytest.raises(ConversionError, match="uses shared formula 0, which no cell holds"):
        xlsx_markdown(_workbook_with_rows(follower("").replace(' ref="D2:D3"', "")))


def test_an_office_archive_too_large_to_inflate_or_too_many_entries_is_refused() -> None:
    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml", b"\0" * (native_formats.MAX_OFFICE_ARCHIVE_BYTES + 1)
        )
    crowded = io.BytesIO()
    with zipfile.ZipFile(crowded, "w") as archive:
        for n in range(native_formats.MAX_OFFICE_ARCHIVE_ENTRIES + 1):
            archive.writestr(f"x/{n}.xml", b"")

    assert len(bomb.getvalue()) < 1_000_000
    with pytest.raises(ConversionError, match="inflates to more than 100000000 bytes; the limit"):
        check_office_archive(bomb.getvalue())
    with pytest.raises(ConversionError, match="has 5001 entries; the limit is 5000"):
        xlsx_markdown(crowded.getvalue())
    with pytest.raises(ConversionError, match="cannot read the Office archive"):
        docx_asides(b"PK\x03\x04 truncated")


def _patch_sizes(data: bytes, name: bytes, size: int) -> bytes:
    """`data` with the inflated size of member `name` written as `size` in its local header
    and its central directory entry."""
    patched = bytearray(data)
    for signature, offset in ((b"PK\x03\x04", 22), (b"PK\x01\x02", 24)):
        at = 0
        while (at := patched.find(signature, at)) != -1:
            name_at = at + (30 if signature == b"PK\x03\x04" else 46)
            if patched[name_at : name_at + len(name)] == name:
                patched[at + offset : at + offset + 4] = size.to_bytes(4, "little")
            at += 4
    return bytes(patched)


def test_an_archive_whose_directory_understates_a_member_or_uses_another_method_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"<x>" + b"\0" * 20_000_000 + b"</x>"
    deflated = io.BytesIO()
    with zipfile.ZipFile(deflated, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", payload)
    lying = _patch_sizes(deflated.getvalue(), b"word/document.xml", 10)
    with zipfile.ZipFile(io.BytesIO(lying)) as archive:
        assert archive.getinfo("word/document.xml").file_size == 10
    lzma = io.BytesIO()
    with zipfile.ZipFile(lzma, "w", zipfile.ZIP_LZMA) as archive:
        archive.writestr("word/document.xml", b"<x/>")
    inflated: list[int] = []
    decompressobj = zlib.decompressobj

    class Counting:
        def __init__(self, wbits: int) -> None:
            self._inner = decompressobj(wbits)

        def decompress(self, data: bytes, max_length: int = 0) -> bytes:
            out = self._inner.decompress(data, max_length)
            inflated.append(len(out))
            return out

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    monkeypatch.setattr(zlib, "decompressobj", Counting)

    with pytest.raises(ConversionError, match="inflates to more than the 10 bytes its entry"):
        check_office_archive(lying)
    # The check stopped at the first step of output, never inflating the whole member.
    assert sum(inflated) <= native_formats.ARCHIVE_CHUNK_BYTES
    with pytest.raises(ConversionError, match="compressed with method 14"):
        docx_asides(lzma.getvalue())


class _Unseekable(io.RawIOBase):
    """A write-only stream that cannot seek, so `zipfile` writes each member's sizes and CRC
    in a data descriptor after its bytes."""

    def __init__(self) -> None:
        self.written = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        self.written += bytes(data)
        return len(data)


def test_an_archive_member_zipfile_could_not_open_as_its_directory_names_it_is_refused() -> None:
    plain = io.BytesIO()
    with zipfile.ZipFile(plain, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"<x/>")
    data = plain.getvalue()
    # The directory placed after the end of the file: every member's local header offset is
    # negative, the first one's pointing back at the first local header by Python slicing.
    end = data.rindex(b"PK\x05\x06")
    directory = int.from_bytes(data[end + 16 : end + 20], "little")
    shifted = data[: end + 16] + (directory + len(data)).to_bytes(4, "little") + data[end + 20 :]
    with zipfile.ZipFile(io.BytesIO(shifted)) as archive:
        assert archive.getinfo("word/document.xml").header_offset == -len(shifted)

    def with_flags(flags: int) -> bytes:
        at = data.index(b"PK\x01\x02")
        return data[: at + 8] + flags.to_bytes(2, "little") + data[at + 10 :]

    def with_local_name(name: bytes, flags: int = 0) -> bytes:
        local = data[:6] + flags.to_bytes(2, "little") + data[8:30] + name
        return local + data[30 + len(name) :]

    streamed = _Unseekable()
    with (
        zipfile.ZipFile(streamed, "w", zipfile.ZIP_DEFLATED) as archive,
        archive.open("word/document.xml", "w") as member,
    ):
        member.write(b"<x>streamed</x>")

    with pytest.raises(ConversionError, match=r"document\.xml has no local header"):
        check_office_archive(shifted)
    for flags in (0x20, 0x40):
        with pytest.raises(ConversionError, match="encrypted or holds patch data"):
            check_office_archive(with_flags(flags))
    for local in (
        with_local_name(b"word/documenX.xml"),
        with_local_name(b"word/doc\xffment.xml", 0x800),
    ):
        with pytest.raises(
            ConversionError, match=r"local header of word/document\.xml names another"
        ):
            check_office_archive(local)
    # A member whose sizes and CRC follow its bytes in a data descriptor is read as written.
    assert (
        zipfile.ZipFile(io.BytesIO(bytes(streamed.written))).getinfo("word/document.xml").flag_bits
        & 0x08
    )
    check_office_archive(bytes(streamed.written))


def test_an_archive_whose_members_overlap_is_refused() -> None:
    def stored(name: str, body: bytes) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(name, body)
        return buffer.getvalue()

    # The outer member stores a whole inner member, local header and all, and a second
    # directory entry points at that inner header: each member reads correctly on its own,
    # but the outer one's bytes run over the inner one.
    inner = stored("word/document.xml", b"<x/>")
    inner_entry = inner[inner.index(b"PK\x01\x02") : inner.rindex(b"PK\x05\x06")]
    outer = stored("word/outer.xml", inner[: inner.index(b"PK\x01\x02")])
    inner_offset = 30 + len("word/outer.xml")
    assert outer[inner_offset : inner_offset + 4] == b"PK\x03\x04"
    entry = inner_entry[:42] + inner_offset.to_bytes(4, "little") + inner_entry[46:]
    end = outer.rindex(b"PK\x05\x06")
    record = bytearray(outer[end:])
    record[8:12] = (2).to_bytes(2, "little") * 2
    record[12:16] = (int.from_bytes(record[12:16], "little") + len(entry)).to_bytes(4, "little")
    overlapping = outer[:end] + entry + bytes(record)
    with zipfile.ZipFile(io.BytesIO(overlapping)) as archive:
        assert archive.read("word/document.xml") == b"<x/>"

    with pytest.raises(ConversionError, match="Overlapped entries"):
        check_office_archive(overlapping)


HTML_PAGE = """<!doctype html><html><head><title>Survey notes</title>
<style>p {{ color: red }}</style><script>var tracking = 1;</script></head><body>
<p>Prepared for the estuary partnership; read with the 2025 tide tables.</p>
<img src="http://127.0.0.1:{port}/plan.png" alt="plan"><img src="{local}" alt="local">
<h1>Survey results</h1><p>Soundings reached <b>12.4 m</b> below chart datum.</p>
<ul><li>Northern channel</li><li>Southern channel</li></ul>
<h2>Tide gauges</h2><table><caption>Gauge readings</caption>
<tr><th>Gauge</th><th>Level (m)</th></tr><tr><td>North | quay</td><td>3.2</td></tr></table>
<pre>depth = 12.4</pre></body></html>"""


def test_an_html_page_reads_in_order_without_fetching_anything(tmp_path: Path) -> None:
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"\x89PNG not really")
    events: list[tuple[str, Any]] = []
    watching = False

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if watching and (event.startswith("socket.") or event == "urllib.Request"):
            events.append((event, args))
        if watching and event == "open" and str(args[0]).endswith("secret.png"):
            events.append((event, args))

    sys.addaudithook(audit)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.setblocking(False)
        page = HTML_PAGE.format(port=server.getsockname()[1], local=secret.as_uri())
        html_markdown(b"<p>warm up</p>")
        watching = True
        try:
            markdown = html_markdown(page.encode())
        finally:
            watching = False
        with pytest.raises(BlockingIOError):
            server.accept()

    assert events == []
    assert markdown.startswith(
        "Prepared for the estuary partnership; read with the 2025 tide tables."
    )
    assert "# Survey results\n\nSoundings reached 12.4 m below chart datum." in markdown
    assert "- Northern channel\n- Southern channel" in markdown
    (table,) = tables(markdown)
    assert (table.caption, table.headers, table.rows) == (
        "Gauge readings",
        ("Gauge", "Level (m)"),
        (("North | quay", "3.2"),),
    )
    assert "## Tide gauges" in markdown and "```\ndepth = 12.4\n```" in markdown
    assert "tracking" not in markdown
    # An image's alt text is not body text.
    assert not {"plan", "local"} & set(markdown.split("\n\n"))


def test_bracketed_numbers_are_data_and_a_merged_header_never_takes_the_only_data_row() -> None:
    page = (
        "<table><tr><td>Item</td><td>Amount</td></tr><tr><td>(120.50)</td><td>($1,234)</td>"
        "</tr><tr><td>Credit</td><td>(12%)</td></tr></table>"
        "<table><tr><th>Item</th><th colspan=2>Cost</th></tr>"
        "<tr><td>Survey</td><td>1</td><td>2</td></tr></table>"
        "<table><tr><th>Item</th><th>Depth</th></tr><tr><td></td><td>(m)</td></tr>"
        "<tr><td>Quay</td><td>3.2</td></tr></table>"
    )

    accounts, costs, depths = tables(html_markdown(page.encode()))

    assert accounts.headers == ("Item", "Amount")
    assert accounts.rows == (("(120.50)", "($1,234)"), ("Credit", "(12%)"))
    assert costs.headers == ("Item", "Cost", "Cost")
    assert costs.rows == (("Survey", "1", "2"),)
    assert depths.headers == ("Item", "Depth (m)")
    assert depths.rows == (("Quay", "3.2"),)


def test_formats_without_pages_compile_into_sections_and_annotation_chunks(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "budget.xlsx").write_bytes(workbook_bytes())
    (corpus_dir / "notes.htm").write_bytes(
        HTML_PAGE.format(port=9, local="file:///nowhere.png").encode()
    )

    corpus = compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T")

    budget, notes = corpus.documents
    assert (budget.format, notes.format) == ("xlsx", "html")
    assert budget.pages == () and notes.pages == ()
    assert {(s.page_start, s.page_end) for d in corpus.documents for s in d.sections} == {(1, 1)}
    assert [s.heading_path for s in budget.sections] == ["Budget", "Staff"]
    assert budget.date is not None and budget.date.iso == "2026-02-11"
    assert notes.date is None and notes.sections[0].heading_path == "Overview"
    kinds = {c.kind for c in budget.chunks}
    assert {"table", "annotation"} <= kinds
    assert all(
        c.content.startswith("> **[Annotation") for c in budget.chunks if c.kind == "annotation"
    )


def test_converting_workbooks_and_pages_never_loads_torch_and_serving_never_loads_docling(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "book.xlsx"
    workbook.write_bytes(workbook_bytes())
    script = (
        "import json, sys\n"
        "import arkive_rag.server, arkive_rag.snapshot\n"
        "served = sorted(m for m in sys.modules if m.split('.')[0] in"
        " ('docling', 'docling_core', 'openpyxl', 'pandas', 'torch', 'arkive_rag.compiler'))\n"
        "from arkive_rag import native_formats\n"
        f"native_formats.xlsx_markdown(open({str(workbook)!r}, 'rb').read())\n"
        "native_formats.html_markdown(b'<h1>T</h1><table><tr><td>1</td></tr></table>')\n"
        "print(json.dumps({'served': served, 'docling': 'docling' in sys.modules,"
        " 'torch': [m for m in sys.modules if m.split('.')[0] == 'torch']}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=300
    )

    seen = json.loads(result.stdout.strip().splitlines()[-1])
    assert seen == {"served": [], "docling": True, "torch": []}


# Word.

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def docx_bytes() -> bytes:
    """A Word file whose body replaces `2176019` with `1998` as a tracked change and carries a
    reviewer comment on its first paragraph; `mc:Ignorable` names a prefix the body never
    uses, as Word writes it."""
    body = (
        '<w:p><w:commentRangeStart w:id="0"/><w:r><w:t>Survey area: </w:t></w:r>'
        '<w:del w:id="1" w:author="Imogen Hale" w:date="2026-01-05T09:00:00Z">'
        "<w:r><w:delText>2176019</w:delText></w:r></w:del>"
        '<w:ins w:id="2" w:author="Imogen Hale" w:date="2026-01-05T09:00:00Z">'
        '<w:r><w:t>1998</w:t></w:r></w:ins><w:r><w:t xml:space="preserve"> hectares.</w:t>'
        '</w:r><w:commentRangeEnd w:id="0"/></w:p>'
        "<w:p><w:r><w:t>Status: Final issue for client review.</w:t></w:r></w:p>"
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W_NS}" xmlns:r="{_DOC_REL}"'
        ' xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
        ' xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"'
        f' mc:Ignorable="w14"><w:body>{body}</w:body></w:document>'
    )
    comments = (
        f'<w:comments xmlns:w="{_W_NS}"><w:comment w:id="0" w:author="Tobias Wren"'
        ' w:date="2026-01-06T10:00:00Z"><w:p><w:r><w:t>Check the area against the'
        " licence.</w:t></w:r></w:p></w:comment></w:comments>"
    )
    return word_file(document, comments)


def word_file(document: str, comments: str) -> bytes:
    """A Word file of `document` (`word/document.xml`) and `comments` (`word/comments.xml`)."""
    word = "application/vnd.openxmlformats-officedocument.wordprocessingml"
    members = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns='
            '"http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType='
            '"application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/word/document.xml" ContentType="{word}.document.main+xml"/>'
            f'<Override PartName="/word/comments.xml" ContentType="{word}.comments+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            f'<Relationships xmlns="{_REL_NS}"><Relationship Id="rId1" Type='
            f'"{_DOC_REL}/officeDocument" Target="word/document.xml"/></Relationships>'
        ),
        "word/_rels/document.xml.rels": (
            f'<Relationships xmlns="{_REL_NS}"><Relationship Id="rId1" Type='
            f'"{_DOC_REL}/comments" Target="comments.xml"/></Relationships>'
        ),
        "word/document.xml": document,
        "word/comments.xml": comments,
        "docProps/core.xml": _CORE,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def test_accepting_tracked_changes_keeps_only_the_accepted_text_and_every_prefix() -> None:
    original = docx_bytes()

    accepted = accepted_revision_copy(original)

    with zipfile.ZipFile(io.BytesIO(accepted)) as archive:
        document = archive.read("word/document.xml")
        assert archive.read("word/comments.xml") == docx_bytes_member("word/comments.xml")
    root = ElementTree.fromstring(document)
    text = "".join(t.text or "" for t in root.iter(f"{{{_W_NS}}}t"))
    assert text == "Survey area: 1998 hectares.Status: Final issue for client review."
    assert b"delText" not in document and b"w:ins" not in document
    assert document.startswith(b"<?xml") and b'mc:Ignorable="w14"' in document
    assert b'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"' in document
    assert accepted_revision_copy(accepted) == accepted
    assert metadata_date(SourceFile("a.docx", Path("a.docx"), "docx", 0), original) == "2026-02-11"


def docx_bytes_member(name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(docx_bytes())) as archive:
        return archive.read(name)


_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"


def _word_document(body: str, declarations: str = "") -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{_W_NS}"'
        f"{declarations}><w:body>{body}</w:body></w:document>"
    )


def _row(text: str, deleted: bool = False, nested: str = "") -> str:
    properties = '<w:trPr><w:del w:id="9" w:author="A"/></w:trPr>' if deleted else ""
    return f"<w:tr>{properties}<w:tc><w:p><w:r><w:t>{text}</w:t></w:r></w:p>{nested}</w:tc></w:tr>"


def _accepted_document(document: str) -> bytes:
    no_comments = f'<w:comments xmlns:w="{_W_NS}"/>'
    with zipfile.ZipFile(io.BytesIO(accepted_revision_copy(word_file(document, no_comments)))) as z:
        return z.read("word/document.xml")


def test_a_deleted_nested_row_never_deletes_the_row_that_holds_its_table() -> None:
    nested = f"<w:tbl>{_row('Inner kept')}{_row('Inner deleted', deleted=True)}</w:tbl>"
    body = (
        f"<w:tbl>{_row('Outer kept', nested=nested)}{_row('Outer deleted', deleted=True)}</w:tbl>"
    )

    root = ElementTree.fromstring(_accepted_document(_word_document(body)))

    texts = [t.text for t in root.iter(f"{{{_W_NS}}}t")]
    assert texts == ["Outer kept", "Inner kept"]


def test_accepting_changes_keeps_each_namespace_declaration_where_its_source_put_it() -> None:
    # Two prefixes for one URI, both named by `mc:Ignorable`, and default namespaces declared
    # on separate subtrees.
    declarations = (
        f' xmlns:mc="{_MC_NS}" xmlns:w14="{_W14_NS}" xmlns:wx="{_W14_NS}" mc:Ignorable="w14 wx"'
    )
    body = (
        '<w:p><w:ins w:id="1" w:author="A"><w:r><w:t>New</w:t></w:r></w:ins></w:p>'
        '<w:sectPr><first xmlns="urn:first"><leaf/></first><second xmlns="urn:second"/>'
        "</w:sectPr>"
    )

    document = _accepted_document(_word_document(body, declarations))

    assert b'mc:Ignorable="w14 wx"' in document
    assert f'xmlns:w14="{_W14_NS}" xmlns:wx="{_W14_NS}"'.encode() in document
    assert b'<first xmlns="urn:first"><leaf/></first><second xmlns="urn:second"/>' in document
    assert b"w:ins" not in document and b"<w:t>New</w:t>" in document


def test_a_comment_over_several_paragraphs_keeps_their_words_apart() -> None:
    body = (
        '<w:p><w:commentRangeStart w:id="0"/><w:r><w:t>First line</w:t><w:br/>'
        "<w:t>after break</w:t><w:tab/><w:t>tabbed</w:t></w:r></w:p>"
        '<w:p><w:r><w:t>second paragraph</w:t></w:r><w:commentRangeEnd w:id="0"/></w:p>'
    )
    comments = (
        f'<w:comments xmlns:w="{_W_NS}"><w:comment w:id="0" w:author="R"><w:p><w:r>'
        "<w:t>Reword</w:t></w:r></w:p></w:comment></w:comments>"
    )

    (comment,) = docx_asides(word_file(_word_document(body), comments))

    assert comment.anchor == "First line after break tabbed second paragraph"


@needs_soffice
def test_libreoffice_renders_only_the_accepted_value_of_a_tracked_change() -> None:
    source = SourceFile("a.docx", Path("a.docx"), "docx", 0)

    ((text, _),) = render_pdf(office_to_pdf(source, docx_bytes()))

    assert "1998" in text and "2176019" not in text


def test_word_comments_and_tracked_changes_become_annotations_on_their_page() -> None:
    source = SourceFile("memo.docx", Path("memo.docx"), "docx", 0)
    pages = ["Cover page", "Survey area: 1998 hectares.\nStatus: Final issue for client review."]

    supplements = page_supplements(source, docx_bytes(), pages)

    assert list(supplements) == [2]
    labels = [part.markdown.splitlines()[0] for part in supplements[2]]
    assert labels == [
        "> **[Annotation \u2014 Tracked deletion by Imogen Hale (2026-01-05)]**",
        "> **[Annotation \u2014 Tracked insertion by Imogen Hale (2026-01-05)]**",
        "> **[Annotation \u2014 Comment by Tobias Wren (2026-01-06) on"
        " \u201cSurvey area: 1998 hectares.\u201d]**",
    ]
    assert [a.text for a in docx_asides(docx_bytes())] == [
        "2176019",
        "1998",
        "Check the area against the licence.",
    ]
    extraction = PageExtraction.model_validate(
        {
            "blocks": [
                {
                    "kind": "paragraph",
                    "heading_level": 0,
                    "continues_from_previous_page": False,
                    "merges_into_previous_row": False,
                    "markdown": pages[1],
                }
            ],
            "visuals": [],
        }
    )
    (draft,) = assemble_sections([(2, extraction)], supplements)
    assert [p.kind for p in draft.parts] == ["paragraph", "annotation", "annotation", "annotation"]
    pieces = section_pieces(draft, "prose")
    assert [p.kind for p in pieces] == ["prose", "annotation", "annotation", "annotation"]
    assert "2176019" not in without_asides(draft.markdown)
    assert [u.text for u in standing_units(draft.markdown)] == [pages[1]]
    assert page_parts(2, extraction)[0].kind == "paragraph"


# PowerPoint.

_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
_MODERN_AUTHORS_REL = "http://schemas.microsoft.com/office/2018/10/relationships/authors"


def _slide(text: str, show: str = "") -> str:
    show = f' show="{show}"' if show else ""
    return (
        f'<p:sld xmlns:p="{_P_NS}" xmlns:a="{_A_NS}"{show}><p:cSld><p:spTree><p:sp><p:txBody>'
        f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )


def _rels(entries: Mapping[str, tuple[str, str]]) -> str:
    rels = "".join(
        f'<Relationship Id="{rid}" Type="{rel_type}" Target="{target}"/>'
        for rid, (rel_type, target) in entries.items()
    )
    return f'<Relationships xmlns="{_REL_NS}">{rels}</Relationships>'


def pptx_bytes(hidden: str = "0") -> bytes:
    """Three slides, the second hidden: the first has speaker notes and a modern comment,
    the third a native bar chart of two series and a legacy comment."""
    chart = (
        f'<c:chartSpace xmlns:c="{_C_NS}" xmlns:a="{_A_NS}"><c:chart><c:title><c:tx><c:rich>'
        "<a:p><a:r><a:t>Soundings by quarter</a:t></a:r></a:p></c:rich></c:tx></c:title>"
        "<c:plotArea><c:barChart>"
        + "".join(
            f'<c:ser><c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>{name}</c:v></c:pt>'
            "</c:strCache></c:strRef></c:tx><c:cat><c:strRef><c:strCache>"
            '<c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt>'
            "</c:strCache></c:strRef></c:cat><c:val><c:numRef><c:numCache>"
            f'<c:pt idx="0"><c:v>{a}</c:v></c:pt><c:pt idx="1"><c:v>{b}</c:v></c:pt>'
            "</c:numCache></c:numRef></c:val></c:ser>"
            for name, a, b in (("North", "120", "135"), ("South", "98", "101"))
        )
        + "</c:barChart></c:plotArea></c:chart></c:chartSpace>"
    )
    notes = (
        f'<p:notes xmlns:p="{_P_NS}" xmlns:a="{_A_NS}"><p:cSld><p:spTree>'
        '<p:sp><p:nvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr></p:sp>'
        '<p:sp><p:nvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:txBody>'
        "<a:p><a:r><a:t>Mention the storm delay first.</a:t></a:r></a:p>"
        "<a:p><a:r><a:t>Then the revised dates.</a:t></a:r></a:p></p:txBody></p:sp>"
        '<p:sp><p:nvSpPr><p:nvPr><p:ph type="sldNum"/></p:nvPr></p:nvSpPr><p:txBody>'
        "<a:p><a:r><a:t>1</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>"
    )
    modern = (
        '<p188:cmLst xmlns:p188="http://schemas.microsoft.com/office/powerpoint/2018/8/main"'
        f' xmlns:a="{_A_NS}"><p188:cm id="{{X}}" authorId="{{A1}}"'
        ' created="2026-01-07T08:00:00.000"><p188:txBody><a:p><a:r><a:t>Add the vessel name'
        "</a:t></a:r></a:p></p188:txBody></p188:cm></p188:cmLst>"
    )
    legacy = (
        f'<p:cmLst xmlns:p="{_P_NS}"><p:cm authorId="0" dt="2026-01-08T08:00:00.000"'
        ' idx="1"><p:pos x="0" y="0"/><p:text>Use the final figures</p:text></p:cm></p:cmLst>'
    )
    rel = _DOC_REL
    members = {
        "ppt/presentation.xml": (
            f'<p:presentation xmlns:p="{_P_NS}" xmlns:r="{rel}"><p:sldIdLst>'
            '<p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/>'
            '<p:sldId id="258" r:id="rId3"/></p:sldIdLst></p:presentation>'
        ),
        "ppt/_rels/presentation.xml.rels": _rels(
            {
                **{f"rId{n}": (f"{rel}/slide", f"slides/slide{n}.xml") for n in (1, 2, 3)},
                "rId4": (_MODERN_AUTHORS_REL, "authors.xml"),
                "rId5": (f"{rel}/commentAuthors", "commentAuthors.xml"),
            }
        ),
        "ppt/slides/slide1.xml": _slide("Survey overview"),
        "ppt/slides/slide2.xml": _slide("Hidden backup", show=hidden),
        "ppt/slides/slide3.xml": _slide("Quarterly soundings"),
        "ppt/slides/_rels/slide1.xml.rels": _rels(
            {
                "rId1": (f"{rel}/notesSlide", "../notesSlides/notesSlide1.xml"),
                "rId2": (
                    "http://schemas.microsoft.com/office/2018/10/relationships/comments",
                    "../comments/modernComment_100_1.xml",
                ),
            }
        ),
        "ppt/slides/_rels/slide3.xml.rels": _rels(
            {
                "rId1": (f"{rel}/chart", "../charts/chart1.xml"),
                "rId2": (f"{rel}/comments", "../comments/comment1.xml"),
            }
        ),
        "ppt/notesSlides/notesSlide1.xml": notes,
        "ppt/charts/chart1.xml": chart,
        "ppt/comments/modernComment_100_1.xml": modern,
        "ppt/comments/comment1.xml": legacy,
        "ppt/authors.xml": (
            '<p188:authorLst xmlns:p188="http://schemas.microsoft.com/office/powerpoint/2018/8/'
            'main"><p188:author id="{A1}" name="Imogen Hale"/></p188:authorLst>'
        ),
        "ppt/commentAuthors.xml": (
            f'<p:cmAuthorLst xmlns:p="{_P_NS}"><p:cmAuthor id="0" name="Tobias Wren"/>'
            "</p:cmAuthorLst>"
        ),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def test_speaker_notes_comments_and_chart_data_belong_to_their_shown_slide() -> None:
    first, second = pptx_supplements(pptx_bytes())

    assert (first.number, second.number) == (1, 2)
    assert first.charts == ()
    assert [(a.label, a.text) for a in first.asides] == [
        ("Speaker notes, slide 1", "Mention the storm delay first.\nThen the revised dates."),
        ("Comment by Imogen Hale (2026-01-07), slide 1", "Add the vessel name"),
    ]
    (chart,) = second.charts
    assert (chart.caption, chart.headers, chart.rows) == (
        "Chart: Soundings by quarter",
        ("Category", "North", "South"),
        (("Q1", "120", "98"), ("Q2", "135", "101")),
    )
    assert [(a.label, a.text) for a in second.asides] == [
        ("Comment by Tobias Wren (2026-01-08), slide 2", "Use the final figures")
    ]


def test_slide_supplements_become_annotation_chunks_and_a_table_on_the_slide_page() -> None:
    source = SourceFile("deck.pptx", Path("deck.pptx"), "pptx", 0)
    supplements = page_supplements(source, pptx_bytes(), ["Survey overview", "Quarterly"])

    def page(text: str) -> PageExtraction:
        block = {
            "kind": "heading",
            "heading_level": 1,
            "continues_from_previous_page": False,
            "merges_into_previous_row": False,
            "markdown": text,
        }
        return PageExtraction.model_validate({"blocks": [block], "visuals": []})

    drafts = assemble_sections(
        [(1, page("Survey overview")), (2, page("Quarterly soundings"))], supplements
    )

    overview, quarterly = drafts
    assert [p.kind for p in overview.parts] == ["heading", "annotation", "annotation"]
    assert [(p.kind, p.page_start) for p in quarterly.parts] == [
        ("heading", 2),
        ("paragraph", 2),
        ("table", 2),
        ("annotation", 2),
    ]
    pieces = section_pieces(quarterly, "prose")
    assert [(p.kind, p.page_start, p.page_end) for p in pieces] == [
        ("prose", 2, 2),
        ("table", 2, 2),
        ("annotation", 2, 2),
    ]
    assert "| Q2 | 135 | 101 |" in pieces[1].content
    assert re.search(r"Speaker notes, slide 1", overview.markdown)
    assert "storm delay" not in without_asides(overview.markdown)
    # Pages that are not one per shown slide put everything on the last page.
    assert list(page_supplements(source, pptx_bytes(), ["only page"])) == [1]


def test_a_slide_shown_false_is_hidden_and_scatter_points_keep_their_own_series() -> None:
    scatter = (
        f'<c:chartSpace xmlns:c="{_C_NS}"><c:chart><c:plotArea><c:scatterChart>'
        + "".join(
            f"<c:ser><c:tx><c:v>{name}</c:v></c:tx><c:xVal><c:numLit>"
            + "".join(f'<c:pt idx="{i}"><c:v>{x}</c:v></c:pt>' for i, (x, _) in enumerate(points))
            + "</c:numLit></c:xVal><c:yVal><c:numLit>"
            + "".join(f'<c:pt idx="{i}"><c:v>{y}</c:v></c:pt>' for i, (_, y) in enumerate(points))
            + "</c:numLit></c:yVal></c:ser>"
            for name, points in (("North", [("1", "10"), ("2", "20")]), ("South", [("5", "50")]))
        )
        + "</c:scatterChart></c:plotArea></c:chart></c:chartSpace>"
    )
    replies = (
        '<p188:cmLst xmlns:p188="http://schemas.microsoft.com/office/powerpoint/2018/8/main"'
        f' xmlns:a="{_A_NS}"><p188:cm id="{{X}}" authorId="{{A1}}" created="2026-01-07T08:00:00">'
        '<p188:replyLst><p188:reply id="{R}" authorId="{A2}" created="2026-01-09T08:00:00">'
        "<p188:txBody><a:p><a:r><a:t>Added it</a:t></a:r></a:p></p188:txBody></p188:reply>"
        "</p188:replyLst><p188:txBody><a:p><a:r><a:t>Add the vessel name</a:t></a:r></a:p>"
        "</p188:txBody></p188:cm></p188:cmLst>"
    )

    def edit(members: dict[str, bytes]) -> None:
        members["ppt/charts/chart1.xml"] = scatter.encode()
        members["ppt/comments/modernComment_100_1.xml"] = replies.encode()
        members["ppt/authors.xml"] = members["ppt/authors.xml"].replace(
            b"</p188:authorLst>", b'<p188:author id="{A2}" name="Tobias Wren"/></p188:authorLst>'
        )

    first, second = pptx_supplements(rewrite_zip(pptx_bytes(hidden="false"), edit))

    assert [(a.label, a.text) for a in first.asides[1:]] == [
        ("Comment by Imogen Hale (2026-01-07), slide 1", "Add the vessel name"),
        ("Reply by Tobias Wren (2026-01-09), slide 1", "Added it"),
    ]
    (chart,) = second.charts
    assert (chart.headers, chart.rows) == (
        ("Series", "X", "Y"),
        (("North", "1", "10"), ("North", "2", "20"), ("South", "5", "50")),
    )


def test_scatter_points_without_cached_x_values_have_blank_ones() -> None:
    def series(name: str, points: list[tuple[str, str]], cached_x: int) -> str:
        """A scatter series whose first `cached_x` points cache their x value."""
        x = "".join(
            f'<c:pt idx="{i}"><c:v>{x}</c:v></c:pt>' for i, (x, _) in enumerate(points[:cached_x])
        )
        y = "".join(f'<c:pt idx="{i}"><c:v>{y}</c:v></c:pt>' for i, (_, y) in enumerate(points))
        x_values = (
            f"<c:xVal><c:numRef><c:f>Sheet1!$A$2</c:f><c:numCache>{x}"
            "</c:numCache></c:numRef></c:xVal>"
        )
        return (
            f"<c:ser><c:tx><c:v>{name}</c:v></c:tx>{x_values}<c:yVal><c:numLit>{y}"
            "</c:numLit></c:yVal></c:ser>"
        )

    def chart_of(*plotted: str) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        scatter = (
            f'<c:chartSpace xmlns:c="{_C_NS}"><c:chart><c:plotArea><c:scatterChart>'
            f"{''.join(plotted)}</c:scatterChart></c:plotArea></c:chart></c:chartSpace>"
        )

        def edit(members: dict[str, bytes]) -> None:
            members["ppt/charts/chart1.xml"] = scatter.encode()

        _, second = pptx_supplements(rewrite_zip(pptx_bytes(), edit))
        (chart,) = second.charts
        return chart.headers, chart.rows

    # A series without cached x values never borrows another series' values.
    assert chart_of(
        series("North", [("1", "10"), ("2", "20")], 2),
        series("South", [("7", "50"), ("8", "60")], 0),
    ) == (
        ("Series", "X", "Y"),
        (("North", "1", "10"), ("North", "2", "20"), ("South", "", "50"), ("South", "", "60")),
    )
    # Series sharing one sparse x cache never get point numbers for the x values it lacks.
    assert chart_of(
        series("North", [("5", "10"), ("6", "20")], 1),
        series("South", [("5", "50"), ("6", "60")], 1),
    ) == (("Category", "North", "South"), (("5", "10", "50"), ("", "20", "60")))


def test_comment_authors_are_read_from_the_parts_the_presentation_relates() -> None:
    def edit(members: dict[str, bytes]) -> None:
        members["ppt/people/modern.xml"] = members.pop("ppt/authors.xml")
        members["ppt/people/legacy.xml"] = members.pop("ppt/commentAuthors.xml")
        members["ppt/_rels/presentation.xml.rels"] = (
            members["ppt/_rels/presentation.xml.rels"]
            .replace(b'Target="authors.xml"', b'Target="people/modern.xml"')
            .replace(b'Target="commentAuthors.xml"', b'Target="people/legacy.xml"')
        )

    first, second = pptx_supplements(rewrite_zip(pptx_bytes(), edit))

    assert first.asides[1].label == "Comment by Imogen Hale (2026-01-07), slide 1"
    assert second.asides[0].label == "Comment by Tobias Wren (2026-01-08), slide 2"


def test_a_prefix_declared_only_on_an_insertion_stays_declared_once_it_is_accepted() -> None:
    body = (
        f'<w:p><w:ins w:id="1" w:author="A" xmlns:mc="{_MC_NS}" xmlns:w14="{_W14_NS}">'
        '<mc:AlternateContent><mc:Choice Requires="w14"><w:r><w:t>New</w:t></w:r></mc:Choice>'
        "<mc:Fallback><w:r><w:t>Old</w:t></w:r></mc:Fallback></mc:AlternateContent></w:ins>"
        "<w:r><w:t> kept</w:t></w:r></w:p>"
    )

    document = _accepted_document(_word_document(body))

    root = etree.fromstring(document)
    (choice,) = root.iter(f"{{{_MC_NS}}}Choice")
    assert b"w:ins" not in document
    assert choice.nsmap["w14"] == _W14_NS and choice.get("Requires") == "w14"
    (run,) = (r for r in root.iter(f"{{{_W_NS}}}r") if r.findtext(f"{{{_W_NS}}}t") == " kept")
    assert "w14" not in run.nsmap


def test_a_spanning_header_cell_marked_in_html_takes_a_label_row_but_no_data_row() -> None:
    data = "<tr><td>Survey</td><td>1</td><td>2</td></tr><tr><td>Gauge</td><td>3</td><td>4</td></tr>"
    page = (
        f"<table><tr><th>Item</th><th colspan=2>Cost</th></tr>{data}</table>"
        "<table><thead><tr><th>Item</th><th colspan=2>Cost</th></tr>"
        "<tr><td></td><td>Labour</td><td>Materials</td></tr></thead>"
        f"<tbody>{data}</tbody></table>"
    )

    single, grouped = tables(html_markdown(page.encode()))

    assert single.headers == ("Item", "Cost", "Cost")
    assert single.rows == (("Survey", "1", "2"), ("Gauge", "3", "4"))
    assert grouped.headers == ("Item", "Cost \u2014 Labour", "Cost \u2014 Materials")
    assert grouped.rows == single.rows
