"""Passages: what a text unit is, which units of a corpus recur, and how two texts compare
word by word.

Induction records which units recur across documents, the drafting packet reports that
recurrence as provenance, and the evaluation harness measures drafts with the same units,
shingles and containment. Every function here is pure: it takes Markdown and plain document
records, and does no I/O.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from arkive_rag.models import ClusterHolder, UnitCluster, fold_for_quote

# Shorter prose is a phrase, not a passage worth reproducing.
STANDING_MIN_WORDS = 8
# A table header row is a unit once it fixes this many columns.
STANDING_MIN_TABLE_COLUMNS = 2
# Recurrence compares word 4-grams: long enough that shared vocabulary alone does not match.
STANDING_SHINGLE_WORDS = 4
# Share of a unit's 4-grams a section of another document must hold. Standing blocks measured
# 0.8 to 1.0 between projects, so light edits such as a changed year still count as recurrence.
STANDING_RECURRENCE_CONTAINMENT = 0.8
# A holder's sample of a recurring prose unit is at most this many folded words.
RECURRING_SAMPLE_WORDS = 16

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


# Recurring units.


@dataclass(frozen=True)
class TextSection:
    section_id: str
    markdown: str


@dataclass(frozen=True)
class TextDocument:
    """One document as recurrence reads it: its sections in reading order."""

    doc_id: str
    rel_path: str
    sections: tuple[TextSection, ...]


@dataclass(frozen=True)
class _PlacedUnit:
    doc_index: int
    section_index: int
    unit: StandingUnit
    # `fold_for_quote` of the text, or of the header row alone for a table header.
    folded: str
    grams: frozenset[int]

    def says_the_same_as(self, other: _PlacedUnit) -> bool:
        if self.unit.header or other.unit.header:
            return self.unit.header == other.unit.header and self.folded == other.folded
        return self.folded == other.folded or (
            _held(self.grams, other.grams) and _held(other.grams, self.grams)
        )


def _held(part: frozenset[int], whole: frozenset[int]) -> bool:
    return len(part & whole) / len(part) >= STANDING_RECURRENCE_CONTAINMENT


def _hashed(grams: frozenset[tuple[str, ...]]) -> frozenset[int]:
    # In-process hashes only test membership; no result depends on their values or order.
    return frozenset(map(hash, grams))


def _probe_count(size: int) -> int:
    """How many of a unit's `size` 4-grams to look up so that every section holding at least
    `STANDING_RECURRENCE_CONTAINMENT` of them holds one of the looked-up ones: such a section
    misses at most `size - needed` grams."""
    needed = math.ceil(size * STANDING_RECURRENCE_CONTAINMENT)
    while needed > 1 and (needed - 1) / size >= STANDING_RECURRENCE_CONTAINMENT:
        needed -= 1
    while needed / size < STANDING_RECURRENCE_CONTAINMENT:
        needed += 1
    return size - needed + 1


def fold_unit(text: str) -> str:
    """A prose unit's text as recurrence compares it: without its leading list marker
    (`LIST_MARKER`), then `fold_for_quote`, so items numbered differently in two documents
    still say the same thing."""
    marker = LIST_MARKER.match(text)
    return fold_for_quote(text[marker.end() :] if marker else text)


def recurring_units(documents: Sequence[TextDocument]) -> tuple[UnitCluster, ...]:
    """The text units of `documents` that recur in another document, grouped into clusters.

    Canonical order is document path (compared component by component, as corpus discovery
    orders files), then each document's sections in the order given, then position in the
    section; the order of `documents` does not matter, and a document ID may appear once.
    Document IDs are path hashes, so ordering by ID would scramble the path order. A unit
    (`standing_units`) is eligible with at least `STANDING_MIN_WORDS` words after `fold_unit`,
    or as a table header row of at least `STANDING_MIN_TABLE_COLUMNS` columns. A section holds
    a unit when it holds at least `STANDING_RECURRENCE_CONTAINMENT` of the unit's word 4-grams,
    counting the 4-grams of its whole text without visual transcripts and of its own units,
    or, for a table header, when it has a table header with the same folded row. A visual
    transcript describes a figure, so its words never make a section hold prose. Where
    documents sit plays no part.

    Units that say the same thing (the same folded text, or containment at the threshold both
    ways) are linked, and a cluster is everything linked to everything else, directly or
    through other units: a chain of light edits across several documents is one cluster even
    when its two ends differ more than the threshold allows. A cluster's sections are every
    section of its units and every section holding one of them, its own document's included;
    it is kept when they lie in two documents or more. It is represented by its first unit in
    canonical order, whose section comes first among its document's sections; the others follow
    in canonical order. Clusters are numbered from 1 in the order of their representatives.

    Each holder of a prose cluster carries a sample: at most `RECURRING_SAMPLE_WORDS` words of
    the cluster as that section writes it, after `fold_unit`, so a line-end hyphen is joined
    and a list marker gone. A section with its own unit of the cluster gives that unit's first
    words; a section holding a unit inside other text gives the words it shares with it
    (`_held_sample`). A table header row has no sample.

    An inverted index of 4-grams finds the sections that could hold a unit by looking up only
    its rarest grams (`_probe_count`), so the search stays near linear in corpus size and still
    finds every section that holds the unit.
    """
    docs = sorted(documents, key=lambda d: (d.rel_path.split("/"), d.doc_id))
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"document IDs repeat: {ids}")
    sections = [(i, s) for i, doc in enumerate(docs) for s in doc.sections]
    placed: list[_PlacedUnit] = []
    header_sections: dict[str, set[int]] = {}
    for index, (doc_index, section) in enumerate(sections):
        for unit in standing_units(section.markdown):
            grams: frozenset[int] = frozenset()
            if unit.header:
                header = unit.text.splitlines()[0]
                if len(_table_cells(header)) < STANDING_MIN_TABLE_COLUMNS:
                    continue
                folded = fold_for_quote(header)
                if not folded:
                    continue
                header_sections.setdefault(folded, set()).add(index)
            else:
                folded = fold_unit(unit.text)
                if len(WORD.findall(folded)) < STANDING_MIN_WORDS:
                    continue
                grams = _hashed(shingles(folded))
            placed.append(_PlacedUnit(doc_index, index, unit, folded, grams))
    # What a section carries: its whole text folded (tables and headings included, visual
    # transcripts removed) together with its own units' grams. Folding the whole text joins
    # a line-end hyphen with the next unit's first word, so neither form alone holds every
    # gram the other does; with both, every unit's grams lie in its own section's grams and
    # in those of every section holding the same text.
    unit_grams: dict[int, list[frozenset[int]]] = {}
    for p in placed:
        unit_grams.setdefault(p.section_index, []).append(p.grams)
    section_grams = [
        _hashed(shingles(fold_for_quote(without_visual_transcripts(s.markdown)))).union(
            *unit_grams.get(index, ())
        )
        for index, (_, s) in enumerate(sections)
    ]
    postings: dict[int, list[int]] = {}
    for index, held in enumerate(section_grams):
        for gram in held:
            postings.setdefault(gram, []).append(index)

    def holders(p: _PlacedUnit) -> set[int]:
        """Every section other than the unit's own that holds it."""
        if p.unit.header:
            return header_sections[p.folded] - {p.section_index}
        probes = sorted(p.grams, key=lambda g: len(postings.get(g, ())))
        return {
            c
            for g in probes[: _probe_count(len(p.grams))]
            for c in postings.get(g, ())
            if c != p.section_index and _held(p.grams, section_grams[c])
        }

    where = [holders(p) for p in placed]
    by_section: dict[int, list[int]] = {}
    for n, p in enumerate(placed):
        by_section.setdefault(p.section_index, []).append(n)
    parent = list(range(len(placed)))

    def root(n: int) -> int:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for n, p in enumerate(placed):
        # A unit that says the same as `p` lies in a section holding `p`, or in its own.
        for c in sorted({p.section_index, *where[n]}):
            for m in by_section.get(c, ()):
                a, b = root(n), root(m)
                if a != b and p.says_the_same_as(placed[m]):
                    parent[max(a, b)] = min(a, b)
    members: dict[int, list[int]] = {}
    for n in range(len(placed)):
        members.setdefault(root(n), []).append(n)
    clusters: list[UnitCluster] = []
    for group in members.values():
        rep = placed[group[0]]
        holding = {placed[n].section_index for n in group}.union(*(where[n] for n in group))
        if len({sections[c][0] for c in holding}) < 2:
            continue
        ordered = sorted(holding, key=lambda c: (sections[c][0], c != rep.section_index, c))
        # Per holding section, its first own unit of the cluster and the first unit it holds.
        own_unit: dict[int, _PlacedUnit] = {}
        held_unit: dict[int, _PlacedUnit] = {}
        for n in group:
            own_unit.setdefault(placed[n].section_index, placed[n])
            for c in where[n]:
                held_unit.setdefault(c, placed[n])
        cluster_holders: list[ClusterHolder] = []
        for c in ordered:
            sample: str | None = None
            if not rep.unit.header:
                sample = (
                    " ".join(WORD.findall(own_unit[c].folded)[:RECURRING_SAMPLE_WORDS])
                    if c in own_unit
                    else _held_sample(held_unit[c].grams, sections[c][1].markdown)
                )
            cluster_holders.append(
                ClusterHolder(section_id=sections[c][1].section_id, sample=sample)
            )
        clusters.append(
            UnitCluster(
                cluster_id=len(clusters) + 1,
                kind="table_header" if rep.unit.header else "prose",
                text=rep.unit.text,
                holders=tuple(cluster_holders),
            )
        )
    return tuple(clusters)


def _held_sample(grams: frozenset[int], markdown: str) -> str:
    """The sample of a prose unit with 4-grams `grams` for a section, of Markdown `markdown`,
    that holds the unit inside other text, such as one sentence of a longer paragraph, and has
    no unit of the cluster itself (that section's sample is its own unit's first words).

    The sample is the longest run of words in which every 4-gram is one of `grams`, cut to
    `RECURRING_SAMPLE_WORDS`, so it never runs on into the section's own words. The run is
    looked for in the section's units after `fold_unit`, then in its whole text without visual
    transcripts, which is taken only when its run is longer, as when the unit spans two of the
    section's units. A holder holds at least `STANDING_RECURRENCE_CONTAINMENT` of `grams` in
    one of those forms, so a run always exists.
    """
    n = STANDING_SHINGLE_WORDS
    best: list[str] = []
    candidates = [
        *(WORD.findall(fold_unit(u.text)) for u in standing_units(markdown) if not u.header),
        WORD.findall(fold_for_quote(without_visual_transcripts(markdown))),
    ]
    for words in candidates:
        start = 0
        for i in range(len(words) - n + 2):
            if i <= len(words) - n and hash(tuple(words[i : i + n])) in grams:
                continue
            # The 4-grams from `start` to `i - 1` are all shared; they cover the words from
            # `start` to `i + n - 2`.
            if i > start and i - start + n - 1 > len(best):
                best = words[start : i + n - 1]
            start = i + 1
    if not best:
        raise AssertionError("a holder of a recurring unit shares none of its 4-grams")
    return " ".join(best[:RECURRING_SAMPLE_WORDS])
