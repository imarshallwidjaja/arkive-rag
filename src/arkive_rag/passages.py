"""Passages: what a text unit is, which units of a set of documents recur, which wording a
recurring unit takes, which units of one section may be house text seen once, and how two
texts compare word by word.

The blueprint step of induction offers recurring units as passage candidates and takes the
units of the sections its agent nominates as house matter; the evaluation harness measures
drafts with the same units, shingles and containment. Every function here is pure: it takes
Markdown and plain document records, and does no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from arkive_rag.models import DocumentDate, WordingSource, fold_for_quote, is_strictly_newer

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
    """One exemplar document as candidate discovery reads it: its sections in reading order,
    and its date (None when it has none). Discovery never reads the date; only the choice of
    a candidate's wording does (`word_candidates`)."""

    doc_id: str
    rel_path: str
    date: DocumentDate | None
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


def _recurrence_key(unit: StandingUnit) -> str | None:
    """The folded text recurrence compares (for a table header, its header row alone), or
    None when the unit cannot recur: prose under `STANDING_MIN_WORDS` words, or a table header
    row under `STANDING_MIN_TABLE_COLUMNS` columns or without text."""
    if unit.header:
        header = unit.text.splitlines()[0]
        if len(_table_cells(header)) < STANDING_MIN_TABLE_COLUMNS:
            return None
        return fold_for_quote(header) or None
    folded = fold_for_quote(unit.text)
    return folded if len(WORD.findall(folded)) >= STANDING_MIN_WORDS else None


def _eligible_units(doc: ExemplarDocument, section: ExemplarSection) -> list[_PlacedUnit]:
    """The units of `section` that can recur: prose of at least `STANDING_MIN_WORDS` words,
    or a table header row of at least `STANDING_MIN_TABLE_COLUMNS` columns."""
    placed: list[_PlacedUnit] = []
    for unit in standing_units(section.markdown):
        folded = _recurrence_key(unit)
        if folded is not None:
            placed.append(_PlacedUnit(doc, section.section_id, unit, folded, shingles(folded)))
    return placed


def is_passage_unit(text: str, markdown: str) -> bool:
    """True when `text` is exactly one of the units of `markdown` that can recur (prose of at
    least `STANDING_MIN_WORDS` words, or a table header row of at least
    `STANDING_MIN_TABLE_COLUMNS` columns) and occurs in `markdown` byte for byte: what every
    stored passage and variant is. A unit's lines are rejoined with `\\n`, so a unit of
    Markdown that breaks lines otherwise is not in it."""
    return text in markdown and any(
        unit.text == text and _recurrence_key(unit) is not None for unit in standing_units(markdown)
    )


@dataclass(frozen=True)
class StandingCandidate:
    """A recurring block of exemplar text. `text` is exactly as in `citations[0]`, the first
    holder in canonical order; the other citations are the sections of other exemplar
    documents where it recurs, in canonical order. Its wording for a blueprint is
    `word_candidates`'s choice."""

    candidate_id: str
    text: str
    citations: tuple[str, ...]


def standing_candidates(documents: Sequence[ExemplarDocument]) -> tuple[StandingCandidate, ...]:
    """The recurring units of a deliverable type's exemplar documents.

    Canonical order is document ID, then each document's sections in the order given, then
    position in the section; the order of `documents` does not matter, and a document ID may
    appear once. A unit (`standing_units`) is eligible with at least `STANDING_MIN_WORDS`
    words, or as a table header row of at least `STANDING_MIN_TABLE_COLUMNS` columns. It
    recurs in a section of another document that holds at least
    `STANDING_RECURRENCE_CONTAINMENT` of its word 4-grams, or, for a table header, that has a
    table header with the same folded row. When the documents span more than one top-level
    folder, that document must sit under another top-level folder than the unit's own: the
    folder is the only sign of a different project the code has. Recurring units that say the
    same thing (the same folded text, or containment at the threshold both ways) are one
    candidate, represented by the first in canonical order.
    """
    docs = sorted(documents, key=lambda d: d.doc_id)
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"exemplar document IDs repeat: {ids}")
    spans_folders = len({_top_folder(d.rel_path) for d in docs}) > 1
    sections = [(doc, s) for doc in docs for s in doc.sections]
    section_grams = {s.section_id: shingles(fold_for_quote(s.markdown)) for _, s in sections}
    placed = [p for doc, section in sections for p in _eligible_units(doc, section)]
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


# A candidate's wording.


@dataclass(frozen=True)
class HolderUnit:
    """A unit of a candidate's holding section that says the same thing as the candidate's
    first unit, with the date of the document that holds it. `header` is true for a table
    header unit (the header row and its separator row), as for every holder of a table
    header candidate."""

    section_id: str
    date: DocumentDate | None
    text: str
    header: bool


@dataclass(frozen=True)
class HoldingSection:
    """One section a candidate cites, the document that holds it, and its holder units in
    section order. A section that holds the candidate's 4-grams only across several units has
    none."""

    doc_id: str
    section_id: str
    units: tuple[HolderUnit, ...]


# Why a candidate keeps its first holder's wording:
# - `holder_units_not_one`: a holding section has zero or several holder units;
# - `dates_show_no_order`: a holding document is undated, or two are not ordered by date;
# - `newest_holds_it_twice`: the newest document holds it in several sections;
# - `newest_unit_taken`: the newest holder's unit is already an earlier candidate's wording.
FirstHolderReason = Literal[
    "holder_units_not_one", "dates_show_no_order", "newest_holds_it_twice", "newest_unit_taken"
]


@dataclass(frozen=True)
class CandidateWording:
    """The wording a blueprint stores for a candidate: `text` exactly as in
    `source_section_id`, and `variants`, the other holder units that word it differently
    (`_wording_key`), in citation and section order. `first_holder_reason` says why a
    `first_holder` wording was not the newest holder's; it is None for the other sources."""

    text: str
    source_section_id: str
    wording_from: WordingSource
    variants: tuple[HolderUnit, ...]
    first_holder_reason: FirstHolderReason | None


@dataclass(frozen=True)
class WordedCandidate:
    """A candidate with its holding sections and the wording a passage made from it stores
    unless a selection names another holder (`holder_wording`)."""

    candidate: StandingCandidate
    holding: tuple[HoldingSection, ...]
    wording: CandidateWording


def holding_sections(
    documents: Sequence[ExemplarDocument], candidate: StandingCandidate
) -> tuple[HoldingSection, ...]:
    """Each section `candidate` cites, in citation order, with its holder units: the units of
    that section that say the same thing as the candidate's first unit (the same folded text,
    or containment of at least `STANDING_RECURRENCE_CONTAINMENT` both ways; for a table
    header, the same folded header row). `documents` are the ones discovery read."""
    by_id = {s.section_id: (d, s) for d in documents for s in d.sections}
    unknown = [sid for sid in candidate.citations if sid not in by_id]
    if unknown:
        raise ValueError(f"{candidate.candidate_id} cites sections not in the documents: {unknown}")
    first_doc, first_section = by_id[candidate.citations[0]]
    first = next(
        (p for p in _eligible_units(first_doc, first_section) if p.unit.text == candidate.text),
        None,
    )
    if first is None:
        raise ValueError(f"{candidate.candidate_id}'s text is not a unit of its first section")
    holding = []
    for sid in candidate.citations:
        doc, section = by_id[sid]
        units = tuple(
            HolderUnit(sid, doc.date, p.unit.text, p.unit.header)
            for p in _eligible_units(doc, section)
            if p.says_the_same_as(first)
        )
        holding.append(HoldingSection(doc.doc_id, sid, units))
    return tuple(holding)


def _alignment(separator_cell: str) -> str:
    cell = separator_cell.replace(" ", "")
    left, right = cell.startswith(":"), cell.endswith(":")
    return "centre" if left and right else "left" if left else "right" if right else "none"


def _wording_key(unit: HolderUnit) -> str | tuple[str, tuple[str, ...]]:
    """What two holder units compare to tell whether they word a candidate alike: a prose
    unit's folded text (`fold_for_quote`); a table header unit's folded header row and each
    separator cell's alignment (left, right, centre or none), so `| --- |` and `| ----- |`
    word it alike and `| :--- |` and `| ---: |` do not."""
    if not unit.header:
        return fold_for_quote(unit.text)
    header, separator = unit.text.split("\n")
    return fold_for_quote(header), tuple(map(_alignment, _table_cells(separator)))


def _wording(
    holding: Sequence[HoldingSection],
    chosen: HolderUnit,
    wording_from: WordingSource,
    reason: FirstHolderReason | None = None,
) -> CandidateWording:
    key = _wording_key(chosen)
    variants = tuple(u for s in holding for u in s.units if _wording_key(u) != key)
    return CandidateWording(chosen.text, chosen.section_id, wording_from, variants, reason)


def _newest_holder(
    holding: Sequence[HoldingSection],
) -> HolderUnit | FirstHolderReason:
    if any(len(s.units) != 1 for s in holding):
        return "holder_units_not_one"
    dates = {s.doc_id: s.units[0].date for s in holding}
    ordered = all(
        is_strictly_newer(dates[a], dates[b]) or is_strictly_newer(dates[b], dates[a])
        for a in dates
        for b in dates
        if a < b
    )
    newest = [
        d for d in dates if all(is_strictly_newer(dates[d], dates[o]) for o in dates if o != d)
    ]
    if not ordered or len(newest) != 1:
        return "dates_show_no_order"
    held = [s.units[0] for s in holding if s.doc_id == newest[0]]
    return held[0] if len(held) == 1 else "newest_holds_it_twice"


def word_candidates(
    documents: Sequence[ExemplarDocument], candidates: Sequence[StandingCandidate]
) -> tuple[WordedCandidate, ...]:
    """Each candidate, in the order given (canonical order), with its holding sections and
    its wording: the newest holder's unit when the holders and their dates show which one is
    newest, else the first holder's (`candidate.text`) with the reason.

    `newest_holder` needs every holding section to have exactly one holder unit, every two
    holding documents' dates to be ordered (`is_strictly_newer` one way), the newest document
    to hold the candidate in one section, and that unit's folded text (`fold_for_quote`) not
    to be the wording of an earlier candidate, from whatever section. Equal dates, a month
    against a day of that month, and a missing date show no order. `check_blueprint` still
    refuses a selection whose wording folds like an earlier selected one's.
    """
    taken: set[str] = set()
    worded: list[WordedCandidate] = []
    for candidate in candidates:
        holding = holding_sections(documents, candidate)
        first = next(u for u in holding[0].units if u.text == candidate.text)
        newest = _newest_holder(holding)
        if isinstance(newest, str):
            wording = _wording(holding, first, "first_holder", newest)
        elif fold_for_quote(newest.text) in taken:
            wording = _wording(holding, first, "first_holder", "newest_unit_taken")
        else:
            wording = _wording(holding, newest, "newest_holder")
        taken.add(fold_for_quote(wording.text))
        worded.append(WordedCandidate(candidate, holding, wording))
    return tuple(worded)


def holder_wording(worded: WordedCandidate, section_id: str) -> CandidateWording:
    """The wording of `worded` taken from the holder unit of `section_id`, as a selection that
    names that holder stores it (`selected_holder`). Raises ValueError when `section_id` is
    not one of the candidate's holding sections or holds it in zero or several units."""
    cid = worded.candidate.candidate_id
    by_id = {s.section_id: s for s in worded.holding}
    if section_id not in by_id:
        raise ValueError(f"{section_id!r} is not a section that holds {cid}")
    units = by_id[section_id].units
    if len(units) != 1:
        raise ValueError(
            f"{section_id!r} holds {cid} in {len(units)} units, so it has no one wording to take"
        )
    return _wording(worded.holding, units[0], "selected_holder")


# House text seen in one exemplar.

# A unit of a nominated section that shares this share of its 4-grams with a holder unit of a
# recurring candidate, either way, belongs to that candidate's house text rather than to a new
# one.
HOUSE_OVERLAP_CONTAINMENT = 0.5


@dataclass(frozen=True)
class HouseUnit:
    """A unit of nominated section `section_id`, numbered by its position among the section's
    units (`standing_units`, from 1), with its text exactly as in the section."""

    section_id: str
    ordinal: int
    text: str


def house_units(
    documents: Sequence[ExemplarDocument],
    section_ids: Sequence[str],
    candidates: Sequence[StandingCandidate],
) -> tuple[HouseUnit, ...]:
    """The prose units of the nominated `section_ids`, each a section of one of `documents`,
    that may be house text seen in one exemplar, in section and unit order: units of at least
    `STANDING_MIN_WORDS` words, except one whose containment with any holder unit of any of
    `candidates` reaches `HOUSE_OVERLAP_CONTAINMENT` in either direction, one that a section
    of another of `documents` holds at `STANDING_RECURRENCE_CONTAINMENT`, and one whose folded
    text equals that of a unit already taken."""
    owners = {s.section_id: (d, s) for d in documents for s in d.sections}
    unknown = [sid for sid in section_ids if sid not in owners]
    if unknown:
        raise ValueError(f"{unknown} are not sections of the documents")
    holder_grams = [
        shingles(fold_for_quote(u.text))
        for c in candidates
        for s in holding_sections(documents, c)
        for u in s.units
    ]
    kept: list[HouseUnit] = []
    seen: set[str] = set()
    for section_id in section_ids:
        owner, section = owners[section_id]
        other_sections = [
            shingles(fold_for_quote(s.markdown))
            for d in documents
            if d.doc_id != owner.doc_id
            for s in d.sections
        ]
        for ordinal, unit in enumerate(standing_units(section.markdown), start=1):
            folded = fold_for_quote(unit.text)
            if unit.header or len(WORD.findall(folded)) < STANDING_MIN_WORDS or folded in seen:
                continue
            grams = shingles(folded)
            # A table header holder can have no 4-gram; it then contains nothing.
            overlaps = any(
                containment(grams, hg) >= HOUSE_OVERLAP_CONTAINMENT
                or (bool(hg) and containment(hg, grams) >= HOUSE_OVERLAP_CONTAINMENT)
                for hg in holder_grams
            )
            recurs = any(
                containment(grams, other) >= STANDING_RECURRENCE_CONTAINMENT
                for other in other_sections
            )
            if not overlaps and not recurs:
                seen.add(folded)
                kept.append(HouseUnit(section_id, ordinal, unit.text))
    return tuple(kept)
