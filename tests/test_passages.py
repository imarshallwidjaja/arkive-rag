from __future__ import annotations

import importlib
import itertools
from collections.abc import Sequence
from dataclasses import replace

import pytest

from arkive_rag import passages
from arkive_rag.models import DocumentDate, fold_for_quote
from arkive_rag.passages import (
    STANDING_RECURRENCE_CONTAINMENT,
    CandidateWording,
    ExemplarDocument,
    ExemplarSection,
    HolderUnit,
    HoldingSection,
    HouseUnit,
    StandingCandidate,
    StandingUnit,
    containment,
    holder_wording,
    holding_sections,
    house_units,
    is_passage_unit,
    shingles,
    standing_candidates,
    standing_units,
    without_visual_transcripts,
    word_candidates,
)

# One section covering every block the unit rules tell apart: a paragraph, Markdown and glyph
# list items with continuation lines, a table with outer pipes followed by a list item, a
# pipeless table whose rows open with a list marker, fenced blocks of both characters, an
# ordinary block quote, visual transcripts, a quote whose prefix line is not its first, and a
# backtick fence holding a line that opens with the fence followed by text (which does not
# close it) and a visual prefix line.
PIN_MARKDOWN = (
    "## Method\n\n"
    "Northwind Marine reduces every sounding to the stated vertical datum before it grids.\n"
    "A second line of the same paragraph.\n\n"
    "- Multibeam lines run at 10 m spacing across the whole channel.\n"
    "  continuing the first item on its own line\n"
    "1. Tide gauges are read every ten minutes during the survey.\n"
    "\u2022 Grid delivered as GeoTIFF on the projected grid of the client.\n"
    "  \uf0b7 Lines delivered as Shapefile with attributes for every run.\n\n"
    "| Deliverable | File name | Format |\n| --- | --- | --- |\n| Grid | `g.tif` | GeoTIFF |\n"
    "- A list item right after the outer-pipe table ends it.\n\n"
    "Deliverable | Coordinate reference system\n--- | ---\n"
    "- Grid | ETRS89 / UTM zone 30N\nMosaic | OSGB36\n\n"
    "```\nNorthwind Marine reduces every sounding inside a fence.\n> **[Visual x]**\n```\n\n"
    "~~~~\n``` not a close\n~~~~\n\n"
    "> Survey results describe the seabed on the survey dates only.\n"
    "> They certify nothing later.\n\n"
    "> **[Visual \u2014 Page 1: Chart]**\n> The chart compares the inner and outer approaches.\n"
    "After the transcript, a closing paragraph of house text follows here.\n\n"
    "> **[Visual \u2014 Page 2: Map]** opens this quote, so the whole quote is a transcript\n\n"
    "> An ordinary quote line comes first.\n> **[Visual \u2014 Page 3: Plan]** labels a figure.\n\n"
    "```\n``` note\n> **[Visual \u2014 Page 4: Plan]**\n"
    "A fenced line that the fence keeps as written.\n```\n\n"
    "Quillfen Geomatics Ltd  \nRegistered surveyors under the house quality manual since 2009."
)
PIN_VARIANT = (
    PIN_MARKDOWN.replace("vertical datum", "chart datum")
    .replace("ten minutes", "five minutes")
    .replace("since 2009", "since 2011")
    .replace("whole channel", "inner channel and the outer approach")
)


def test_text_units_shingles_containments_and_transcript_removal_are_pinned() -> None:
    units = standing_units(PIN_MARKDOWN)
    grams = [shingles(fold_for_quote(u.text)) for u in units]
    variant = shingles(fold_for_quote(PIN_VARIANT))

    assert units == [
        StandingUnit(
            "Northwind Marine reduces every sounding to the stated vertical datum before it"
            " grids.\nA second line of the same paragraph.",
            header=False,
        ),
        StandingUnit(
            "- Multibeam lines run at 10 m spacing across the whole channel.\n"
            "  continuing the first item on its own line",
            header=False,
        ),
        StandingUnit("1. Tide gauges are read every ten minutes during the survey.", header=False),
        StandingUnit(
            "\u2022 Grid delivered as GeoTIFF on the projected grid of the client.", header=False
        ),
        StandingUnit(
            "\uf0b7 Lines delivered as Shapefile with attributes for every run.", header=False
        ),
        StandingUnit("| Deliverable | File name | Format |\n| --- | --- | --- |", header=True),
        StandingUnit("- A list item right after the outer-pipe table ends it.", header=False),
        StandingUnit("Deliverable | Coordinate reference system\n--- | ---", header=True),
        StandingUnit(
            "> Survey results describe the seabed on the survey dates only.\n"
            "> They certify nothing later.",
            header=False,
        ),
        StandingUnit(
            "After the transcript, a closing paragraph of house text follows here.", header=False
        ),
        StandingUnit(
            "> An ordinary quote line comes first.\n"
            "> **[Visual \u2014 Page 3: Plan]** labels a figure.",
            header=False,
        ),
        StandingUnit(
            "Quillfen Geomatics Ltd  \n"
            "Registered surveyors under the house quality manual since 2009.",
            header=False,
        ),
    ]
    assert [len(g) for g in grams] == [17, 16, 8, 8, 6, 1, 8, 1, 11, 8, 10, 9]
    assert [containment(g, variant) for g in grams] == [
        13 / 17,
        11 / 16,
        4 / 8,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        8 / 9,
    ]
    assert without_visual_transcripts(PIN_MARKDOWN) == (
        PIN_MARKDOWN.replace(
            "> **[Visual \u2014 Page 1: Chart]**\n"
            "> The chart compares the inner and outer approaches.\n",
            "",
        ).replace(
            "> **[Visual \u2014 Page 2: Map]** opens this quote, so the whole quote is a"
            " transcript\n",
            "",
        )
    )


MOVED_NAMES = (
    "standing_units",
    "standing_candidates",
    "StandingCandidate",
    "StandingUnit",
    "ExemplarDocument",
    "ExemplarSection",
    "shingles",
    "containment",
    "WORD",
    "LIST_MARKER",
    "without_visual_transcripts",
    "VISUAL_TRANSCRIPT_PREFIX",
    "STANDING_MIN_WORDS",
    "STANDING_MIN_TABLE_COLUMNS",
    "STANDING_SHINGLE_WORDS",
    "STANDING_RECURRENCE_CONTAINMENT",
)


@pytest.mark.parametrize("module", ["arkive_rag.induction", "evals.drafting"])
def test_the_passage_rules_are_reachable_only_from_the_passages_module(module: str) -> None:
    assert all(hasattr(passages, name) for name in MOVED_NAMES)
    imported = importlib.import_module(module)
    assert [name for name in MOVED_NAMES if hasattr(imported, name)] == []


def test_visual_transcripts_are_block_quotes_whose_first_line_opens_with_the_prefix() -> None:
    house = (
        "Northwind Marine reduces every sounding to the stated vertical datum before it grids"
        " the surface."
    )
    ordinary = f"> {house}\n> **[Visual \u2014 Page 2: Chart]** is how the compiler labels figures."
    transcript = (
        "> **[Visual \u2014 Page 1: Map]**\n"
        "> The survey boundary map marks every mooring buoy inside the estuary channel."
    )
    markdown = f"{ordinary}\n\n{transcript}\n\nAfter."

    assert without_visual_transcripts(markdown) == f"{ordinary}\n\n\nAfter."
    # `standing_units` reads the same block as the one transcript.
    assert [u.text for u in standing_units(markdown)] == [ordinary, "After."]


# Recurring candidates.

# Thirteen words, so ten word 4-grams: changing word 11 breaks two of them (0.8 recurs),
# changing word 10 breaks three (0.7 does not).
TERMS = "Quillfen Geomatics carries out every survey under its quality manual and audit schedule"
AT_THRESHOLD = TERMS.replace("audit", "review")
BELOW_THRESHOLD = TERMS.replace("and audit", "plus audit")
HEADER = "| Deliverable | File name | Format |\n| --- | --- | --- |"
DISCLAIMER = (
    "Survey results describe the seabed on the survey dates only and certify nothing later."
)


Documents = dict[str, ExemplarDocument]


def documents(*docs: tuple[str, Sequence[str]]) -> Documents:
    """Documents doc_001, doc_002, ... in argument order, keyed by ID; each section's Markdown
    as given, with sections doc_00N_sec_001, doc_00N_sec_002, ..."""
    built = (
        ExemplarDocument(
            f"doc_{n:03d}",
            rel_path,
            None,
            tuple(
                ExemplarSection(f"doc_{n:03d}_sec_{k:03d}", markdown)
                for k, markdown in enumerate(markdowns, start=1)
            ),
        )
        for n, (rel_path, markdowns) in enumerate(docs, start=1)
    )
    return {d.doc_id: d for d in built}


def candidates(docs: Documents, exemplars: Sequence[str]) -> tuple[StandingCandidate, ...]:
    return standing_candidates([docs[d] for d in exemplars])


def texts(docs: Documents, exemplars: Sequence[str]) -> list[str]:
    return [c.text for c in candidates(docs, exemplars)]


def test_prose_recurs_at_exactly_the_containment_threshold_and_not_below() -> None:
    assert STANDING_RECURRENCE_CONTAINMENT == 0.8
    at = documents(("p1/a.md", [TERMS]), ("p2/b.md", [AT_THRESHOLD]))
    below = documents(("p1/a.md", [TERMS]), ("p2/b.md", [BELOW_THRESHOLD]))

    (candidate,) = candidates(at, ("doc_001", "doc_002"))

    # The two wordings say the same thing, so they are one candidate held by the first.
    assert candidate == StandingCandidate(
        "candidate_01", TERMS, ("doc_001_sec_001", "doc_002_sec_001")
    )
    assert candidates(below, ("doc_001", "doc_002")) == ()


def test_recurrence_needs_another_top_level_folder_only_when_the_exemplars_span_folders() -> None:
    view = documents(
        ("p1/a.md", [DISCLAIMER]), ("p1/b.md", [DISCLAIMER]), ("p2/c.md", ["Scope only."])
    )
    flat = documents(("a.md", [DISCLAIMER]), ("b.md", [DISCLAIMER]))

    assert texts(view, ("doc_001", "doc_002", "doc_003")) == []
    assert texts(view, ("doc_001", "doc_002")) == [DISCLAIMER]
    assert [c.citations for c in candidates(flat, ("doc_001", "doc_002"))] == [
        ("doc_001_sec_001", "doc_002_sec_001")
    ]
    # Recurrence within one document is not recurrence.
    same_doc = documents(("a.md", [DISCLAIMER, DISCLAIMER]), ("b.md", ["Other."]))
    assert texts(same_doc, ("doc_001", "doc_002")) == []


def test_units_are_paragraphs_list_items_and_table_header_rows_only() -> None:
    reference = "- Harlow, J. (2019). *Multibeam practice for inland waters*. Survey Press."
    heading = "## Quality assurance under the survey quality manual of the house"
    visual = f"> **[Visual \u2014 Page 1: Chart]**\n> {DISCLAIMER}"
    first = [
        f"{heading}\n\n{reference}\n- Field notes of the first crew on the estuary.",
        f"{HEADER}\n| Grid | `p1_grid.tif` | GeoTIFF |",
        f"{visual}\n\n```\n{TERMS}\n```",
        "| Name |\n| --- |\n| Grid |",
    ]
    second = [
        f"{heading}\n\n- Moreau, L. (2021). Tide reduction.\n{reference}",
        f"{HEADER}\n| Lines | `p2_lines.shp` | Shapefile |",
        f"{visual}\n\n```\n{TERMS}\n```",
        "| Name |\n| --- |\n| Lines |",
    ]
    view = documents(("p1/a.md", first), ("p2/b.md", second))

    # Each list item is its own unit; the header row keeps its separator row and the body
    # differs; headings, visual transcripts, fenced blocks and one-column headers are not units.
    assert texts(view, ("doc_001", "doc_002")) == [reference, HEADER]
    other_header = documents(("p1/a.md", [HEADER]), ("p2/b.md", [HEADER.replace("Format", "Size")]))
    assert texts(other_header, ("doc_001", "doc_002")) == []
    # A table without outer pipes is a table too: a short one still has its header unit, and
    # the body rows of a long one never join it, even when they recur.
    short = "Name | Value\n--- | ---"
    long_header = (
        "Deliverable | File name | Format | Coordinate reference system\n--- | --- | --- | ---"
    )
    body = "Bathymetric grid | `grid.tif` | GeoTIFF | ETRS89 / UTM zone 30N with OD heights"
    pipeless = documents(
        ("p1/a.md", [f"{short}\nGrid | 1", f"{long_header}\n{body}"]),
        ("p2/b.md", [f"{short}\nLines | 2", f"{long_header}\n{body}"]),
    )
    assert texts(pipeless, ("doc_001", "doc_002")) == [short, long_header]


@pytest.mark.parametrize("glyph", ["\u2022", "\u25e6", "\u25aa", "\u2023", "\u2043", "\uf0b7"])
def test_bullet_glyphs_of_pdf_transcripts_start_list_items(glyph: str) -> None:
    lines = [f"{glyph} Grid delivered as GeoTIFF.", f"  {glyph} Lines delivered as Shapefile."]

    # Consecutive glyph bullets without blank lines are separate units, like `-` items.
    assert standing_units("\n".join(lines)) == [
        StandingUnit(line.strip(), header=False) for line in lines
    ]
    # A glyph inside a sentence starts nothing.
    inline = f"Deliverables:\nGrid {glyph} Lines {glyph} Mosaic."
    assert standing_units(inline) == [StandingUnit(inline, header=False)]


@pytest.mark.parametrize("marker", ["\u2022", "-"])
def test_a_table_row_or_header_that_opens_with_a_list_marker_stays_in_its_table(
    marker: str,
) -> None:
    header = "Deliverable | Format\n--- | ---"
    row = f"{header}\n{marker} Bathymetric grid | GeoTIFF\nMosaic | GeoTIFF"
    assert standing_units(row) == [StandingUnit(header, header=True)]
    marked_header = f"{marker} Deliverable | Format\n--- | ---"
    assert standing_units(f"{marked_header}\nBathymetric grid | GeoTIFF") == [
        StandingUnit(marked_header, header=True)
    ]


def test_a_list_item_right_after_a_table_with_outer_pipes_ends_the_table() -> None:
    header = "| Measure | Basis |\n| --- | --- |"
    item = "- Each measurement report records current site baseline | as supplied"

    assert standing_units(f"{header}\n| Noise | LAeq |\n{item}") == [
        StandingUnit(header, header=True),
        StandingUnit(item, header=False),
    ]


def test_a_unit_keeps_the_inner_bytes_of_its_holder_section() -> None:
    # Two trailing spaces before a newline are a Markdown hard line break.
    broken = (
        "Quillfen Geomatics Ltd  \nRegistered surveyors under the house quality manual since 2009."
    )
    view = documents(("p1/a.md", [f"Intro.\n\n{broken}  \n"]), ("p2/b.md", [broken]))

    (candidate,) = candidates(view, ("doc_001", "doc_002"))

    assert candidate.text == broken
    (holder,) = [s for s in view["doc_001"].sections if s.section_id == candidate.citations[0]]
    assert candidate.text in holder.markdown


def test_candidates_group_duplicates_in_canonical_order_whatever_the_input_order() -> None:
    emphasised = DISCLAIMER.replace("seabed", "**seabed**")
    docs = [
        ("p1/a.md", ["# Scope\n\nThe p1 harbour.", emphasised]),
        ("p2/b.md", [f"{DISCLAIMER}\n\n{HEADER}"]),
        ("p3/c.md", [f"{HEADER}\n| Grid | x | y |", f"Intro.\n\n{DISCLAIMER}"]),
    ]
    expected = (
        StandingCandidate(
            "candidate_01", emphasised, ("doc_001_sec_002", "doc_002_sec_001", "doc_003_sec_002")
        ),
        StandingCandidate("candidate_02", HEADER, ("doc_002_sec_001", "doc_003_sec_001")),
    )

    results = {
        candidates(documents(*docs), order)
        for order in itertools.permutations(("doc_001", "doc_002", "doc_003"))
    }

    assert results == {expected}
    # One exemplar cannot show recurrence.
    assert candidates(documents(*docs), ("doc_002",)) == ()
    # A document given twice is a caller error, not a second exemplar.
    with pytest.raises(ValueError, match="repeat"):
        candidates(documents(*docs), ("doc_001", "doc_002", "doc_001"))


# A candidate's wording.

# Sixteen words; the later wordings change only the last word, so they keep 12 of 13 4-grams
# and say the same thing.
REDUCTION = "Soundings are reduced to chart datum using the tide gauge records of the survey week."
FORTNIGHT = REDUCTION.replace("week.", "fortnight.")
MONTH = REDUCTION.replace("week.", "month.")


def dated(*docs: tuple[str | None, Sequence[str]]) -> list[ExemplarDocument]:
    """Documents doc_001, doc_002, ... under folders p1, p2, ..., each with a printed date
    (or none) and its sections' Markdown."""
    built = documents(*((f"p{n}/r.md", s) for n, (_, s) in enumerate(docs, start=1)))
    return [
        replace(doc, date=None if iso is None else DocumentDate(iso=iso, basis="printed"))
        for doc, (iso, _) in zip(built.values(), docs, strict=True)
    ]


def wording(docs: Sequence[ExemplarDocument]) -> CandidateWording:
    (worded,) = word_candidates(docs, standing_candidates(docs))
    return worded.wording


def test_the_newest_dated_holder_supplies_the_wording_and_the_others_are_variants() -> None:
    docs = dated(("2024-03-01", [REDUCTION]), ("2026-01", [FORTNIGHT]), ("2025-06-10", [MONTH]))
    (candidate,) = standing_candidates(docs)

    # Identity stays discovery's: the first holder's text and every holding section.
    assert candidate.text == REDUCTION
    (worded,) = word_candidates(docs, (candidate,))
    assert worded.holding == holding_sections(docs, candidate)
    assert worded.holding == tuple(
        HoldingSection(doc.doc_id, sid, (HolderUnit(sid, doc.date, text, False),))
        for sid, doc, text in zip(
            candidate.citations, docs, (REDUCTION, FORTNIGHT, MONTH), strict=True
        )
    )
    assert worded.wording == CandidateWording(
        FORTNIGHT,
        "doc_002_sec_001",
        "newest_holder",
        (
            HolderUnit("doc_001_sec_001", docs[0].date, REDUCTION, False),
            HolderUnit("doc_003_sec_001", docs[2].date, MONTH, False),
        ),
        None,
    )
    # A holder whose wording folds like the chosen one is no variant.
    same = dated(("2024-03-01", [REDUCTION.upper()]), ("2025-01-01", [REDUCTION]))
    assert wording(same) == CandidateWording(
        REDUCTION, "doc_002_sec_001", "newest_holder", (), None
    )


@pytest.mark.parametrize(
    "dates",
    [
        pytest.param((None, "2025-06-10"), id="undated"),
        pytest.param(("2025-06-10", "2025-06-10"), id="tied"),
        pytest.param(("2025-06", "2025-06-10"), id="month-against-a-day-of-it"),
        # The newest is later than both, but the two older dates show no order between them.
        pytest.param(("2025-06", "2025-06-10", "2026-01-01"), id="older-pair-unordered"),
    ],
)
def test_dates_that_show_no_order_keep_the_first_holder(dates: tuple[str | None, ...]) -> None:
    texts = (REDUCTION, FORTNIGHT, MONTH)[: len(dates)]

    chosen = wording(dated(*((d, [t]) for d, t in zip(dates, texts, strict=True))))

    assert (chosen.text, chosen.source_section_id, chosen.wording_from) == (
        REDUCTION,
        "doc_001_sec_001",
        "first_holder",
    )
    assert chosen.first_holder_reason == "dates_show_no_order"
    assert [v.text for v in chosen.variants] == list(texts[1:])


def test_a_holding_section_without_exactly_one_holder_unit_keeps_the_first_holder() -> None:
    # Two units of the newest section say the same thing: which one is the wording is unknown.
    twice = dated(("2024-03-01", [REDUCTION]), ("2025-06-10", [f"{FORTNIGHT}\n\n{MONTH}"]))
    (candidate,) = standing_candidates(twice)
    assert [len(s.units) for s in holding_sections(twice, candidate)] == [1, 2]
    assert (wording(twice).wording_from, wording(twice).first_holder_reason) == (
        "first_holder",
        "holder_units_not_one",
    )
    assert [v.text for v in wording(twice).variants] == [FORTNIGHT, MONTH]
    # The newest section holds the candidate's 4-grams only across two units: it has no
    # holder unit and gives no variant. (Its second unit recurs on its own as a candidate.)
    split = REDUCTION.replace(" using", ".\n\nUsing")
    spread = dated(("2024-03-01", [REDUCTION]), ("2025-06-10", [split]))
    candidate = standing_candidates(spread)[0]
    assert candidate.text == REDUCTION
    assert [len(s.units) for s in holding_sections(spread, candidate)] == [1, 0]
    assert word_candidates(spread, (candidate,))[0].wording == CandidateWording(
        REDUCTION, "doc_001_sec_001", "first_holder", (), "holder_units_not_one"
    )
    # The newest document holds it in two sections, each with one holder unit.
    two_sections = dated(("2024-03-01", [REDUCTION]), ("2025-06-10", [FORTNIGHT, MONTH]))
    assert (wording(two_sections).wording_from, wording(two_sections).first_holder_reason) == (
        "first_holder",
        "newest_holds_it_twice",
    )


# Twenty-nine words. EARLY and LATE each change one word of NEWEST, so each keeps 23 of its 27
# 4-grams and says the same thing as NEWEST; between them only 19 survive, so they do not.
NEWEST = (
    "Every survey we deliver is reduced to the stated vertical datum, checked against an"
    " independent crossline, reviewed by a second surveyor and archived with its raw files"
    " for seven years."
)
EARLY = NEWEST.replace("stated", "declared")
LATE = NEWEST.replace("raw", "source")


def test_two_candidates_never_store_the_same_newest_holder_unit() -> None:
    docs = dated(("2024-03-01", [EARLY]), ("2025-03-01", [LATE]), ("2026-03-01", [NEWEST]))
    found = standing_candidates(docs)
    # Both candidates are held by the newest report's one unit.
    assert [(c.text, c.citations) for c in found] == [
        (EARLY, ("doc_001_sec_001", "doc_003_sec_001")),
        (LATE, ("doc_002_sec_001", "doc_003_sec_001")),
    ]

    first, second = word_candidates(docs, found)

    # Worded in canonical order, the first takes the newest unit; the second keeps its own.
    assert (first.wording.text, first.wording.wording_from) == (NEWEST, "newest_holder")
    assert second.wording == CandidateWording(
        LATE,
        "doc_002_sec_001",
        "first_holder",
        (HolderUnit("doc_003_sec_001", docs[2].date, NEWEST, False),),
        "newest_unit_taken",
    )
    # Two projects whose final reports both hold the newest wording: each candidate's newest
    # holder is another section, but the wording folds alike, so it is still taken.
    projects = documents(
        ("p1/01-draft.md", [EARLY]),
        ("p1/02-final.md", [NEWEST]),
        ("p2/01-draft.md", [LATE]),
        ("p2/02-final.md", [NEWEST.upper()]),
    )
    finals = [
        replace(doc, date=DocumentDate(iso=iso, basis="printed"))
        for doc, iso in zip(
            projects.values(), ("2024-03-01", "2026-01-15", "2025-03-01", "2026-02-15"), strict=True
        )
    ]
    first, second = word_candidates(finals, standing_candidates(finals))
    assert (first.wording.source_section_id, first.wording.text) == (
        "doc_004_sec_001",
        NEWEST.upper(),
    )
    assert (second.wording.source_section_id, second.wording.text) == ("doc_003_sec_001", LATE)
    assert second.wording.first_holder_reason == "newest_unit_taken"


def test_a_table_header_variant_compares_the_header_row_and_each_columns_alignment() -> None:
    aligned = HEADER.replace("| --- | --- | --- |", "| :--- | ---: | --- |")
    docs = dated(("2024-03-01", [HEADER]), ("2025-03-01", [aligned]))

    (worded,) = word_candidates(docs, standing_candidates(docs))

    # The header rows fold alike, so they recur as one candidate; the alignment differs.
    assert worded.candidate.citations == ("doc_001_sec_001", "doc_002_sec_001")
    assert (worded.wording.text, worded.wording.wording_from) == (aligned, "newest_holder")
    assert worded.wording.variants == (HolderUnit("doc_001_sec_001", docs[0].date, HEADER, True),)
    # A padded separator row aligns its columns as the compact one does: no variant, so a
    # passage of it may be standing.
    padded = HEADER.replace("| --- | --- | --- |", "| ----------- | --------- | ------ |")
    centred = aligned.replace(":--- |", ":---: |", 1)
    assert wording(dated(("2024-03-01", [HEADER]), ("2025-03-01", [padded]))) == CandidateWording(
        padded, "doc_002_sec_001", "newest_holder", (), None
    )
    recentred = dated(("2024-03-01", [aligned]), ("2025-03-01", [centred]))
    assert [v.text for v in wording(recentred).variants] == [aligned]


def test_a_selection_takes_the_wording_of_a_holder_that_holds_it_in_one_unit() -> None:
    docs = dated(
        ("2024-03-01", [REDUCTION]),
        ("2025-03-01", [FORTNIGHT]),
        ("2026-03-01", [f"{MONTH}\n\n{MONTH.upper()}", "Scope only."]),
    )
    (worded,) = word_candidates(docs, standing_candidates(docs))

    assert holder_wording(worded, "doc_001_sec_001") == CandidateWording(
        REDUCTION,
        "doc_001_sec_001",
        "selected_holder",
        tuple(
            HolderUnit(f"doc_00{n}_sec_001", docs[n - 1].date, text, False)
            for n, text in ((2, FORTNIGHT), (3, MONTH), (3, MONTH.upper()))
        ),
        None,
    )
    with pytest.raises(ValueError, match="'doc_003_sec_001' holds candidate_01 in 2 units"):
        holder_wording(worded, "doc_003_sec_001")
    with pytest.raises(ValueError, match="'doc_003_sec_002' is not a section that holds"):
        holder_wording(worded, "doc_003_sec_002")


# House text seen in one exemplar.


def test_nominated_units_skip_any_holders_wording_and_repeat_no_unit() -> None:
    # The candidate's first holder says "declared"; the newer holder says "stated". The
    # nominated unit shares 8 of its 14 4-grams with the newer holder's wording, 5 with the
    # first holder's.
    near_holder = (
        "Each crew notes that every survey we deliver is reduced to the stated vertical datum"
        " before lunch."
    )
    policy = "Personal data collected on site is deleted thirty days after the final report."
    docs = dated(
        ("2024-03-01", [EARLY]),
        ("2025-03-01", [NEWEST]),
        (
            "2026-03-01",
            [f"{near_holder}\n\n{policy}\n\n{policy}", policy.upper(), "Scope only here."],
        ),
    )
    found = standing_candidates(docs)
    assert [c.citations for c in found] == [("doc_001_sec_001", "doc_002_sec_001")]

    units = house_units(docs, ("doc_003_sec_001", "doc_003_sec_002"), found)

    # The near-holder unit belongs to the candidate's house text; the policy is taken once,
    # though it repeats in its section and, folded alike, in the next nominated section.
    assert units == (HouseUnit("doc_003_sec_001", 2, policy),)
    with pytest.raises(ValueError, match="doc_009_sec_001"):
        house_units(docs, ("doc_009_sec_001",), found)


def test_a_passage_unit_is_a_recurrable_unit_of_its_section_exactly() -> None:
    section = f"## Method\n\n{REDUCTION}\n\nShort line.\n\n{HEADER}\n| Grid | g.tif | GeoTIFF |"
    assert is_passage_unit(REDUCTION, section)
    assert is_passage_unit(HEADER, section)
    # A substring of a unit, a unit too short to recur, and a heading are not passage units.
    assert not is_passage_unit(REDUCTION[:-1], section)
    assert not is_passage_unit("Short line.", section)
    assert not is_passage_unit("## Method", section)
    # A unit rejoins its lines with `\n`, so in a section that breaks them otherwise it is a
    # unit but not in the section byte for byte.
    two_lines = REDUCTION.replace(" using", "\nusing")
    assert is_passage_unit(two_lines, two_lines)
    assert [u.text for u in standing_units(two_lines.replace("\n", "\r\n"))] == [two_lines]
    assert not is_passage_unit(two_lines, two_lines.replace("\n", "\r\n"))
