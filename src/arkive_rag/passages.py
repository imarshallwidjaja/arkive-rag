"""Passages: what a text unit is, which units of a set of documents recur, and how two texts
compare word by word.

The blueprint step of induction offers recurring units as standing passage candidates, and
the evaluation harness measures drafts with the same units, shingles and containment. Every
function here is pure: it takes Markdown and plain document records, and does no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from arkive_rag.models import fold_for_quote

# Shorter prose is a phrase, not a passage worth reproducing.
STANDING_MIN_WORDS = 8
# A table header row is a unit once it fixes this many columns.
STANDING_MIN_TABLE_COLUMNS = 2
# Recurrence compares word 4-grams: long enough that shared vocabulary alone does not match.
STANDING_SHINGLE_WORDS = 4
# Share of a unit's 4-grams a section of another document must hold. Standing blocks measured
# 0.8 to 1.0 between projects, so light edits such as a changed year still count as recurrence.
STANDING_RECURRENCE_CONTAINMENT = 0.8

WORD = re.compile(r"\w+")
_MARKDOWN_HEADING_LINE = re.compile(r"#{1,6}\s")
# A list item's leading marker and the horizontal space after it, followed by content on the
# same line. Besides Markdown markers, the bullet glyphs PDF text layers carry into
# transcripts; U+F0B7 is the Private Use Area bullet that Word's Symbol font writes into PDF
# text layers.
LIST_MARKER = re.compile(
    r"(?:[-*+\u2022\u25e6\u25aa\u2023\u2043\uf0b7]|\d{1,3}[.)])[^\S\r\n]+(?=\S)"
)
_LIST_ITEM = re.compile(rf"\s*{LIST_MARKER.pattern}")
# A fenced block opens with three or more backticks or tildes and closes at a line of at least
# as many of the same character (`_closes_fence`).
_FENCE_LINE = re.compile(r"\s*(`{3,}|~{3,})")
_TABLE_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
# How the compiler opens a visual transcript; a transcript is a description, not house text.
VISUAL_TRANSCRIPT_PREFIX = "> **[Visual"
_TABLE_KINDS = ("outer_pipe_table", "pipeless_table")


def _closes_fence(fence: str, stripped: str) -> bool:
    return stripped.startswith(fence) and not stripped.strip(fence[0])


@dataclass(frozen=True)
class StandingUnit:
    """One block of a section's Markdown: a paragraph or list item (`header` false), or a
    table's header row with its separator row (`header` true). `text` is exactly as in the
    section, without surrounding whitespace."""

    text: str
    header: bool


def _table_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _is_table_separator(row: str) -> bool:
    return "|" in row and all(
        _TABLE_SEPARATOR_CELL.fullmatch(c.replace(" ", "")) for c in _table_cells(row)
    )


def standing_units(markdown: str) -> list[StandingUnit]:
    """The units of one section in order. A blank line, heading or fenced block ends a unit.
    A table starts at a line opening with `|` (outer pipes), or at a line holding `|` right
    above a separator row, whatever its first character. A table with outer pipes continues
    through lines that hold `|` and do not open with a list marker (`LIST_MARKER`), so a list
    item written right after it is a list item; a table without outer pipes continues
    through every line holding `|`, so its rows stay in it whatever their first character.
    Any other line that opens with a list marker starts a unit; an unmarked line right after
    a list item continues it. Headings, fenced blocks, visual transcripts and table body rows
    are not units; a table without a separator row has no header unit. A block quote that is
    not a visual transcript is a paragraph. A unit's text keeps its inner bytes, such as the
    trailing spaces of a hard line break."""
    units: list[StandingUnit] = []
    kind: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal kind
        text = "\n".join(lines).strip()
        if kind in ("paragraph", "item") or (
            kind == "quote" and not text.startswith(VISUAL_TRANSCRIPT_PREFIX)
        ):
            units.append(StandingUnit(text, header=False))
        elif kind in _TABLE_KINDS and len(lines) >= 2 and _is_table_separator(lines[1]):
            units.append(StandingUnit("\n".join(lines[:2]).strip(), header=True))
        kind = None
        lines.clear()

    fence: str | None = None
    all_lines = markdown.splitlines()
    for i, line in enumerate(all_lines):
        stripped = line.strip()
        if fence is not None:
            if _closes_fence(fence, stripped):
                fence = None
            continue
        if opening := _FENCE_LINE.match(line):
            flush()
            fence = opening[1]
            continue
        if not stripped or _MARKDOWN_HEADING_LINE.match(stripped):
            flush()
            continue
        if "|" in stripped and (
            kind == "pipeless_table" or (kind == "outer_pipe_table" and not _LIST_ITEM.match(line))
        ):
            pass
        elif stripped.startswith("|") or (
            "|" in stripped and i + 1 < len(all_lines) and _is_table_separator(all_lines[i + 1])
        ):
            flush()
            kind = "outer_pipe_table" if stripped.startswith("|") else "pipeless_table"
        elif _LIST_ITEM.match(line):
            flush()
            kind = "item"
        elif stripped.startswith(">"):
            if kind != "quote":
                flush()
                kind = "quote"
        elif kind not in ("paragraph", "item"):
            flush()
            kind = "paragraph"
        lines.append(line)
    flush()
    return units


def without_visual_transcripts(markdown: str) -> str:
    """`markdown` without the visual transcripts the compiler writes: a block quote whose first
    line opens with `VISUAL_TRANSCRIPT_PREFIX`, through its last `>` line. This agrees with
    `standing_units` on what a transcript is: a prefix line inside an ordinary block quote
    starts nothing, and a fenced block is kept as written."""
    kept: list[str] = []
    in_quote = in_transcript = False
    fence: str | None = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if fence is not None:
            if _closes_fence(fence, stripped):
                fence = None
            kept.append(line)
            continue
        if opening := _FENCE_LINE.match(line):
            fence = opening[1]
            in_quote = in_transcript = False
            kept.append(line)
            continue
        quote_line = stripped.startswith(">")
        if not quote_line:
            in_transcript = False
        elif not in_quote:
            in_transcript = stripped.startswith(VISUAL_TRANSCRIPT_PREFIX)
        in_quote = quote_line
        if not in_transcript:
            kept.append(line)
    return "\n".join(kept)


def shingles(folded: str) -> frozenset[tuple[str, ...]]:
    """The word `STANDING_SHINGLE_WORDS`-grams of text already in `fold_for_quote` form."""
    words = WORD.findall(folded)
    n = STANDING_SHINGLE_WORDS
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def containment(part: frozenset[tuple[str, ...]], whole: frozenset[tuple[str, ...]]) -> float:
    """Share of `part`'s 4-grams that `whole` holds; `part` is never empty for a unit."""
    return len(part & whole) / len(part)


# Recurring candidates.


@dataclass(frozen=True)
class ExemplarSection:
    section_id: str
    markdown: str


@dataclass(frozen=True)
class ExemplarDocument:
    """One exemplar document as candidate discovery reads it: its sections in reading order."""

    doc_id: str
    rel_path: str
    sections: tuple[ExemplarSection, ...]


def _top_folder(rel_path: str) -> str:
    """The first path component, or "" for a document at the corpus root."""
    return rel_path.split("/")[0] if "/" in rel_path else ""


@dataclass(frozen=True)
class _PlacedUnit:
    doc: ExemplarDocument
    section_id: str
    unit: StandingUnit
    # `fold_for_quote` of the text, or of the header row alone for a table header.
    folded: str
    grams: frozenset[tuple[str, ...]]

    def says_the_same_as(self, other: _PlacedUnit) -> bool:
        if self.unit.header or other.unit.header:
            return self.unit.header == other.unit.header and self.folded == other.folded
        return self.folded == other.folded or (
            containment(self.grams, other.grams) >= STANDING_RECURRENCE_CONTAINMENT
            and containment(other.grams, self.grams) >= STANDING_RECURRENCE_CONTAINMENT
        )


@dataclass(frozen=True)
class StandingCandidate:
    """A recurring block of exemplar text. `text` is exactly as in `citations[0]`, the first
    holder in canonical order; the other citations are the sections of other exemplar
    documents where it recurs, in canonical order."""

    candidate_id: str
    text: str
    citations: tuple[str, ...]


def standing_candidates(documents: Sequence[ExemplarDocument]) -> tuple[StandingCandidate, ...]:
    """The recurring units of a deliverable type's exemplar documents.

    Canonical order is document path (compared component by component, as corpus discovery
    orders files), then each document's sections in the order given, then position in the
    section; the order of `documents` does not matter, and a document ID may appear once.
    Document IDs are path hashes, so ordering by ID would scramble the path order. A unit
    (`standing_units`) is eligible with at least `STANDING_MIN_WORDS` words, or as a table
    header row of at least `STANDING_MIN_TABLE_COLUMNS` columns. It
    recurs in a section of another document that holds at least
    `STANDING_RECURRENCE_CONTAINMENT` of its word 4-grams, or, for a table header, that has a
    table header with the same folded row. When the documents span more than one top-level
    folder, that document must sit under another top-level folder than the unit's own: the
    folder is the only sign of a different project the code has. Recurring units that say the
    same thing (the same folded text, or containment at the threshold both ways) are one
    candidate, represented by the first in canonical order.
    """
    docs = sorted(documents, key=lambda d: (d.rel_path.split("/"), d.doc_id))
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"exemplar document IDs repeat: {ids}")
    spans_folders = len({_top_folder(d.rel_path) for d in docs}) > 1
    sections = [(doc, s) for doc in docs for s in doc.sections]
    section_grams = {s.section_id: shingles(fold_for_quote(s.markdown)) for _, s in sections}
    placed: list[_PlacedUnit] = []
    for doc, section in sections:
        for unit in standing_units(section.markdown):
            if unit.header:
                header = unit.text.splitlines()[0]
                if len(_table_cells(header)) < STANDING_MIN_TABLE_COLUMNS:
                    continue
                folded = fold_for_quote(header)
                if not folded:
                    continue
            else:
                folded = fold_for_quote(unit.text)
                if len(WORD.findall(folded)) < STANDING_MIN_WORDS:
                    continue
            placed.append(_PlacedUnit(doc, section.section_id, unit, folded, shingles(folded)))
    headers: dict[str, set[str]] = {}
    for p in placed:
        if p.unit.header:
            headers.setdefault(p.section_id, set()).add(p.folded)

    def recurrences(p: _PlacedUnit) -> tuple[str, ...]:
        found: list[str] = []
        for doc, section in sections:
            sid = section.section_id
            if doc.doc_id == p.doc.doc_id or (
                spans_folders and _top_folder(doc.rel_path) == _top_folder(p.doc.rel_path)
            ):
                continue
            if p.unit.header:
                held = p.folded in headers.get(sid, ())
            else:
                held = containment(p.grams, section_grams[sid]) >= STANDING_RECURRENCE_CONTAINMENT
            if held:
                found.append(sid)
        return tuple(found)

    groups: list[tuple[_PlacedUnit, tuple[str, ...]]] = []
    for p in placed:
        where = recurrences(p)
        if where and not any(p.says_the_same_as(holder) for holder, _ in groups):
            groups.append((p, where))
    return tuple(
        StandingCandidate(f"candidate_{n:02d}", holder.unit.text, (holder.section_id, *where))
        for n, (holder, where) in enumerate(groups, start=1)
    )
