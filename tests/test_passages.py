from __future__ import annotations

import importlib
import itertools
import time
from collections.abc import Sequence

import pytest

from arkive_rag import passages
from arkive_rag.models import ClusterHolder, UnitCluster, fold_for_quote
from arkive_rag.passages import (
    STANDING_RECURRENCE_CONTAINMENT,
    StandingUnit,
    TextDocument,
    TextSection,
    containment,
    recurring_units,
    shingles,
    standing_units,
    without_visual_transcripts,
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
    "recurring_units",
    "StandingUnit",
    "TextDocument",
    "TextSection",
    "shingles",
    "containment",
    "fold_unit",
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


# Recurring units.

# Thirteen words, so ten word 4-grams: changing word 11 breaks two of them (0.8 recurs),
# changing word 10 breaks three (0.7 does not).
TERMS = "Quillfen Geomatics carries out every survey under its quality manual and audit schedule"
AT_THRESHOLD = TERMS.replace("audit", "review")
BELOW_THRESHOLD = TERMS.replace("and audit", "plus audit")
HEADER = "| Deliverable | File name | Format |\n| --- | --- | --- |"
DISCLAIMER = (
    "Survey results describe the seabed on the survey dates only and certify nothing later."
)


Documents = dict[str, TextDocument]


def documents(*docs: tuple[str, Sequence[str]]) -> Documents:
    """Documents doc_001, doc_002, ... in argument order, keyed by ID; each section's Markdown
    as given, with sections doc_00N_sec_001, doc_00N_sec_002, ..."""
    built = (
        TextDocument(
            f"doc_{n:03d}",
            rel_path,
            tuple(
                TextSection(f"doc_{n:03d}_sec_{k:03d}", markdown)
                for k, markdown in enumerate(markdowns, start=1)
            ),
        )
        for n, (rel_path, markdowns) in enumerate(docs, start=1)
    )
    return {d.doc_id: d for d in built}


def clusters(docs: Documents, chosen: Sequence[str] | None = None) -> tuple[UnitCluster, ...]:
    return recurring_units([docs[d] for d in (chosen or docs)])


def texts(docs: Documents, chosen: Sequence[str] | None = None) -> list[str]:
    return [c.text for c in clusters(docs, chosen)]


def test_prose_recurs_at_exactly_the_containment_threshold_and_not_below() -> None:
    assert STANDING_RECURRENCE_CONTAINMENT == 0.8
    at = documents(("p1/a.md", [TERMS]), ("p2/b.md", [AT_THRESHOLD]))
    below = documents(("p1/a.md", [TERMS]), ("p2/b.md", [BELOW_THRESHOLD]))

    (cluster,) = clusters(at)

    # The two wordings say the same thing, so they are one cluster held by the first.
    assert cluster == UnitCluster(
        cluster_id=1,
        kind="prose",
        text=TERMS,
        holders=(
            ClusterHolder(section_id="doc_001_sec_001", sample=fold_for_quote(TERMS)),
            ClusterHolder(section_id="doc_002_sec_001", sample=fold_for_quote(AT_THRESHOLD)),
        ),
    )
    assert clusters(below) == ()


def test_recurrence_needs_another_document_wherever_the_documents_sit() -> None:
    # No folder rule: two documents of one folder, or at the root, show recurrence alike.
    same_folder = documents(
        ("p1/a.md", [DISCLAIMER]), ("p1/b.md", [DISCLAIMER]), ("p2/c.md", ["Scope only."])
    )
    flat = documents(("a.md", [DISCLAIMER]), ("b.md", [DISCLAIMER]))

    assert [c.section_ids for c in clusters(same_folder)] == [
        ("doc_001_sec_001", "doc_002_sec_001")
    ]
    assert [c.section_ids for c in clusters(flat)] == [("doc_001_sec_001", "doc_002_sec_001")]
    # Recurrence within one document is not recurrence.
    same_doc = documents(("a.md", [DISCLAIMER, DISCLAIMER]), ("b.md", ["Other."]))
    assert texts(same_doc) == []


def test_numbered_items_recur_whatever_their_numbers_and_a_hyphen_before_the_next_item() -> None:
    # `widget-` at a line end joins the next item's number in the section's folded text, so
    # the item's last 4-gram is in no section; looking it up must not fail.
    hyphenated = (
        "1. Mount the pressure sensor on the calibrated survey widget-\n"
        "2. Calibrate the pressure sensor against the reference gauge every day."
    )
    numbered = documents(
        ("p1/a.md", [f"1. {TERMS}", hyphenated]),
        ("p2/b.md", [f"2. {AT_THRESHOLD}", hyphenated.replace("1.", "3.").replace("2.", "4.")]),
    )

    found = clusters(numbered)

    # Without their markers the two items hold eight of ten 4-grams, which recurs; with
    # them, the numbers would break a ninth.
    assert [(c.text, c.section_ids) for c in found] == [
        (f"1. {TERMS}", ("doc_001_sec_001", "doc_002_sec_001")),
        (
            "1. Mount the pressure sensor on the calibrated survey widget-",
            ("doc_001_sec_002", "doc_002_sec_002"),
        ),
        (
            "2. Calibrate the pressure sensor against the reference gauge every day.",
            ("doc_001_sec_002", "doc_002_sec_002"),
        ),
    ]


def test_a_unit_whose_edges_join_neighbouring_words_still_recurs() -> None:
    # In the whole section's folded text, the hyphenated heading joins the paragraph's first
    # word and the paragraph's last word joins the next item's number: two of the paragraph's
    # eight 4-grams (0.75) are in neither section's whole text, though both hold it verbatim.
    section = (
        "## Scope of the hydro-\n"
        "The survey crew records every sounding against the chart datum daily-\n"
        "2. The second item of the list follows the paragraph."
    )
    twin = documents(("p1/a.md", [section]), ("p2/b.md", [section]))

    found = clusters(twin)

    assert [(c.text, c.section_ids) for c in found] == [
        (
            "The survey crew records every sounding against the chart datum daily-",
            ("doc_001_sec_001", "doc_002_sec_001"),
        ),
        (
            "2. The second item of the list follows the paragraph.",
            ("doc_001_sec_001", "doc_002_sec_001"),
        ),
    ]


def test_a_visual_transcript_never_makes_a_section_hold_prose() -> None:
    # The figure's description repeats a paragraph of the other document word for word.
    transcript = f"> **[Visual: figure 1]**\n> {DISCLAIMER}"
    docs = documents(("p1/a.md", [DISCLAIMER]), ("p2/b.md", [f"Intro.\n\n{transcript}"]))

    assert clusters(docs) == ()
    assert len(clusters(documents(("p1/a.md", [DISCLAIMER]), ("p2/b.md", [DISCLAIMER])))) == 1


def test_a_chain_of_light_edits_is_one_cluster_holding_every_document() -> None:
    # Each neighbour breaks two of ten 4-grams (0.8, recurs); the two ends break four (0.6).
    first = TERMS
    middle = TERMS.replace("audit", "review")
    last = middle.replace("Geomatics", "Surveying")
    chain = documents(("p1/a.md", [first]), ("p2/b.md", [middle]), ("p3/c.md", [last]))

    (cluster,) = clusters(chain)

    assert (cluster.text, cluster.section_ids) == (
        first,
        ("doc_001_sec_001", "doc_002_sec_001", "doc_003_sec_001"),
    )


def test_a_cluster_lists_every_holding_section_of_its_own_documents_once() -> None:
    # The disclaimer twice in one document, and a variant of the terms beside the terms.
    repeated = documents(
        ("p1/a.md", [DISCLAIMER, f"{TERMS}\n\nAnd more.", f"Intro.\n\n{DISCLAIMER}"]),
        ("p1/b.md", [AT_THRESHOLD, DISCLAIMER]),
    )

    assert [(c.text, c.section_ids) for c in clusters(repeated)] == [
        (DISCLAIMER, ("doc_001_sec_001", "doc_001_sec_003", "doc_002_sec_002")),
        (TERMS, ("doc_001_sec_002", "doc_002_sec_001")),
    ]
    variants = documents(("p1/a.md", [TERMS, AT_THRESHOLD]), ("p2/b.md", [TERMS]))
    (cluster,) = clusters(variants)
    assert cluster.section_ids == ("doc_001_sec_001", "doc_001_sec_002", "doc_002_sec_001")


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
    assert texts(view) == [reference, HEADER]
    assert [c.kind for c in clusters(view)] == ["prose", "table_header"]
    other_header = documents(("p1/a.md", [HEADER]), ("p2/b.md", [HEADER.replace("Format", "Size")]))
    assert texts(other_header) == []
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
    assert texts(pipeless) == [short, long_header]


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

    (cluster,) = clusters(view)

    assert cluster.text == broken
    (holder,) = [s for s in view["doc_001"].sections if s.section_id == cluster.section_ids[0]]
    assert cluster.text in holder.markdown


def test_clusters_group_duplicates_in_canonical_order_whatever_the_input_order() -> None:
    emphasised = DISCLAIMER.replace("seabed", "**seabed**")
    docs = [
        ("p1/a.md", ["# Scope\n\nThe p1 harbour.", emphasised]),
        ("p2/b.md", [f"{DISCLAIMER}\n\n{HEADER}"]),
        ("p3/c.md", [f"{HEADER}\n| Grid | x | y |", f"Intro.\n\n{DISCLAIMER}"]),
    ]
    # A sample is the folded words without punctuation.
    sample = "survey results describe the seabed on the survey dates only and certify nothing later"
    expected = (
        UnitCluster(
            cluster_id=1,
            kind="prose",
            text=emphasised,
            holders=tuple(
                ClusterHolder(section_id=s, sample=sample)
                for s in ("doc_001_sec_002", "doc_002_sec_001", "doc_003_sec_002")
            ),
        ),
        UnitCluster(
            cluster_id=2,
            kind="table_header",
            text=HEADER,
            holders=tuple(
                ClusterHolder(section_id=s, sample=None)
                for s in ("doc_002_sec_001", "doc_003_sec_001")
            ),
        ),
    )

    results = {
        clusters(documents(*docs), order)
        for order in itertools.permutations(("doc_001", "doc_002", "doc_003"))
    }

    assert results == {expected}
    # One document cannot show recurrence.
    assert clusters(documents(*docs), ("doc_002",)) == ()
    # A document given twice is a caller error, not a second holder.
    with pytest.raises(ValueError, match="repeat"):
        clusters(documents(*docs), ("doc_001", "doc_002", "doc_001"))


def test_a_holders_sample_stops_where_the_recurring_words_end() -> None:
    recurring = "Every survey record stays with the client for seven years."
    own_opening = "The harbour team opens the section with its own sentence."
    own_tail = "Our crew then archives the raw soundings on two separate drives."
    view = documents(
        ("p1/a.md", [recurring]),
        # The recurring sentence inside a longer paragraph of the section's own words.
        ("p2/b.md", [f"{own_opening} {recurring} {own_tail}"]),
    )

    (cluster,) = clusters(view)

    sample = "every survey record stays with the client for seven years"
    assert cluster.holders == (
        ClusterHolder(section_id="doc_001_sec_001", sample=sample),
        ClusterHolder(section_id="doc_002_sec_001", sample=sample),
    )


def test_canonical_order_is_path_order_whatever_the_document_ids() -> None:
    def document(doc_id: str, rel_path: str, markdown: str) -> TextDocument:
        return TextDocument(doc_id, rel_path, (TextSection(f"{doc_id}_s", markdown),))

    # Path-hash IDs can sort against their paths; canonical order must not follow them.
    reversed_ids = [
        document("doc_ccc", "p1/a.md", DISCLAIMER),
        document("doc_bbb", "p2/b.md", f"Intro.\n\n{DISCLAIMER}"),
        document("doc_aaa", "p3/c.md", DISCLAIMER),
    ]
    (cluster,) = recurring_units(reversed_ids[::-1])
    assert cluster.section_ids == ("doc_ccc_s", "doc_bbb_s", "doc_aaa_s")
    # Paths compare component by component, as discovery orders files: `p1/x.md` precedes
    # `p1-old.md`, although the string `p1-old.md` sorts first.
    by_component = [
        document("doc_aaa", "p1-old.md", AT_THRESHOLD),
        document("doc_bbb", "p1/x.md", TERMS),
    ]
    (holder,) = recurring_units(by_component)
    assert (holder.text, holder.section_ids) == (TERMS, ("doc_bbb_s", "doc_aaa_s"))


def test_recurrence_over_a_large_corpus_stays_fast_and_finds_every_holder() -> None:
    # 160 documents of 12 sections each: every document carries the house disclaimer, one in
    # eight a shared table header, and the rest of each section is its own text.
    def section(n: int, k: int) -> str:
        own = " ".join(f"term{n}x{k}y{w}" for w in range(60))
        return f"## Part {k}\n\n{own}."

    docs = [
        TextDocument(
            f"doc_{n:04d}",
            f"project-{n % 9}/report-{n}.md",
            tuple(
                TextSection(
                    f"doc_{n:04d}_sec_{k:03d}",
                    section(n, k)
                    + (f"\n\n{DISCLAIMER}" if k == 12 else "")
                    + (f"\n\n{HEADER}\n| a | b | c |" if k == 3 and n % 8 == 0 else ""),
                )
                for k in range(1, 13)
            ),
        )
        for n in range(160)
    ]

    started = time.perf_counter()
    found = recurring_units(docs)
    elapsed = time.perf_counter() - started

    # The header holder's third section precedes the disclaimer's twelfth in canonical order.
    assert [c.text for c in found] == [HEADER, DISCLAIMER]
    assert [len(c.section_ids) for c in found] == [20, 160]
    # The inverted index keeps this near linear; testing every unit against every section of
    # 160 documents would take far longer.
    assert elapsed < 10.0
