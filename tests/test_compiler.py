from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from arkive_rag.compiler import (
    FIDELITY_ATTEMPTS,
    MAX_CHUNK_CHARS,
    OBSERVE_BATCH_CHARS,
    SUPPORTED_SUFFIXES,
    BatchSection,
    ObserveSections,
    SectionDraft,
    SourceFile,
    assemble_sections,
    bind_document,
    compile_corpus,
    curator_note_card,
    discover_corpus,
    document_date,
    embedded_date,
    fidelity_anchors,
    find_furniture,
    locate_quote,
    missing_anchors,
    observation_batches,
    parse_markdown_blocks,
    parse_printed_date,
    read_document,
    render_pdf,
    section_pieces,
    split_table,
    split_table_block,
    strip_furniture,
    transcribe_with_fidelity,
)
from arkive_rag.llm import BuildCache
from arkive_rag.models import (
    BlueprintDraft,
    BlueprintSectionDraft,
    ChunkPiece,
    ConversionError,
    CorpusDocument,
    CraftDraft,
    DeliverableTypeDraft,
    DocumentDate,
    DocumentReading,
    Facet,
    FacetDraft,
    FacetValue,
    FacetValueDraft,
    GuidanceDraft,
    GuideDraft,
    HouseSectionNomination,
    ObservationCard,
    ObservationDraft,
    OrganisationDraft,
    PageBlock,
    PageExtraction,
    ReadSection,
    RejectedOutputError,
    TermDraft,
    TranscriptionFidelityError,
    UnsafePathError,
    UnsupportedFormatError,
    VisualTranscript,
    is_strictly_newer,
)
from arkive_rag.passages import without_visual_transcripts
from arkive_rag.snapshot import TopicSnapshot, write_snapshot
from tests.support import (
    FakeGateway,
    ScriptedOpenAI,
    core_properties,
    write_docx,
    write_flat_presentation,
    write_pdf,
)

needs_soffice = pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice missing")


def block(
    kind: str, markdown: str, *, level: int = 0, continues: bool = False, merges: bool = False
) -> PageBlock:
    return PageBlock.model_validate(
        {
            "kind": kind,
            "heading_level": level,
            "continues_from_previous_page": continues,
            "merges_into_previous_row": merges,
            "markdown": markdown,
        }
    )


def page(*blocks: PageBlock, visuals: tuple[VisualTranscript, ...] = ()) -> PageExtraction:
    return PageExtraction(blocks=list(blocks), visuals=list(visuals))


def write_files(root: Path, files: dict[str, str]) -> Path:
    for rel_path, text in files.items():
        (root / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (root / rel_path).write_text(text, encoding="utf-8")
    return root


# Discovery.


def test_discovery_lists_supported_files_in_sorted_order_and_skips_hidden(tmp_path: Path) -> None:
    (tmp_path / "p102-estuary-multibeam" / "drafts").mkdir(parents=True)
    (tmp_path / "p102-estuary-multibeam" / "P102 summary.docx").write_bytes(b"x")
    (tmp_path / "p102-estuary-multibeam" / "drafts" / "notes.MD").write_text("x")
    (tmp_path / "overview.md").write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.xlsx").write_text("x")
    (tmp_path / ".DS_Store").write_text("x")

    sources = discover_corpus(tmp_path)

    assert [(s.rel_path, s.format) for s in sources] == [
        ("overview.md", "md"),
        ("p102-estuary-multibeam/P102 summary.docx", "docx"),
        ("p102-estuary-multibeam/drafts/notes.MD", "md"),
    ]


def test_discovery_rejects_unsupported_files_before_any_model_call(tmp_path: Path) -> None:
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "memo.md").write_text("# Memo\n\nText")
    (tmp_path / "p1" / "model.xlsx").write_bytes(b"x")
    gateway = FakeGateway()

    with pytest.raises(UnsupportedFormatError, match=r"p1/model\.xlsx"):
        compile_corpus(tmp_path, gateway, topic="t", title="T")
    assert gateway.calls == []


def test_discovery_rejects_symlinked_files_and_directories(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("# Secret")
    corpus = tmp_path / "corpus"
    (corpus / "p1").mkdir(parents=True)
    (corpus / "p1" / "memo.md").write_text("# Memo")
    (corpus / "p1" / "link.md").symlink_to(outside / "secret.md")

    with pytest.raises(ConversionError, match=r"p1/link\.md: symlinks are not allowed"):
        discover_corpus(corpus)
    (corpus / "p1" / "link.md").unlink()
    (corpus / "p2").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafePathError, match="p2: symlinks are not allowed"):
        discover_corpus(corpus)


def test_unreadable_documents_are_conversion_errors(tmp_path: Path) -> None:
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "memo.md").write_bytes(b"# Memo\n\n\xff\xfe broken")

    with pytest.raises(ConversionError, match=r"p1/memo\.md is not UTF-8"):
        compile_corpus(tmp_path, FakeGateway(), topic="t", title="T")
    (tmp_path / "p1" / "memo.md").write_text("  \n")
    with pytest.raises(ConversionError, match=r"p1/memo\.md: the document has no text"):
        compile_corpus(tmp_path, FakeGateway(), topic="t", title="T")


def test_curator_notes_get_a_deterministic_card_and_searchable_guidance_chunks(
    tmp_path: Path,
) -> None:
    note = "# Status\n\nThe reef Sentinel-2 case study has not been written yet."
    write_files(
        tmp_path,
        {
            "p103-reef-sonar/AGENTS.md": note,
            "p103-reef-sonar/estuary-proposal.md": "# Scope\n\nSentinel-2 bathymetry.",
        },
    )
    gateway = FakeGateway()

    corpus = compile_corpus(tmp_path, gateway, topic="bathymetry", title="Bathymetry")

    assert [(n.scope_path, n.content) for n in corpus.curator_notes] == [
        ("p103-reef-sonar/AGENTS.md", note)
    ]
    guidance, proposal = corpus.documents
    assert (guidance.doc_id, guidance.is_curator_note) == ("doc_001", True)
    assert (guidance.title, guidance.deliverable_type) == ("Status", "curator note")
    assert guidance.summary == "The reef Sentinel-2 case study has not been written yet."
    assert {c.kind for c in guidance.chunks} == {"guidance"}
    assert "case study has not been written" in guidance.chunks[0].content
    assert not proposal.is_curator_note and {c.kind for c in proposal.chunks} == {"prose"}
    # Only the proposal is observed; the note's card needs no model call.
    (observed,) = gateway.of_type(ObserveSections)
    assert [s.heading_path for s in observed.sections] == ["Scope"]
    assert corpus.embeddings.shape == (len(corpus.all_chunks()), FakeGateway.embedding_dimensions)


@pytest.mark.parametrize(
    ("text", "title", "summary"),
    [
        (
            "#\nThe reef study is unwritten.\n",
            "The reef study is unwritten.",
            "The reef study is unwritten.",
        ),
        (
            "# Status\nThe reef study is unwritten.\nAsk Dana.",
            "Status",
            "The reef study is unwritten.",
        ),
        ("#\n\n", "Curator note", ""),
    ],
)
def test_curator_note_cards_take_the_first_text_as_title_and_first_prose_line_as_summary(
    text: str, title: str, summary: str
) -> None:
    card = curator_note_card(text)
    assert (card.title, card.summary) == (title, summary)


# Section assembly and chunking.


def test_sections_follow_the_heading_stack_with_overview_for_leading_text() -> None:
    sections = assemble_sections(
        [
            (1, page(block("paragraph", "Cover text"), block("heading", "Methods", level=1))),
            (2, page(block("heading", "Rainfall", level=2), block("paragraph", "Median."))),
            (3, page(block("heading", "Results", level=1), block("paragraph", "Bathymetry."))),
        ]
    )

    assert [(s.heading_path, s.level, s.page_start, s.page_end) for s in sections] == [
        ("Overview", 1, 1, 1),
        ("Methods", 1, 1, 1),
        ("Methods > Rainfall", 2, 2, 2),
        ("Results", 1, 3, 3),
    ]
    assert sections[2].markdown == "## Rainfall\n\nMedian."


# Every line break `str.splitlines` recognises besides `\n`.
LINE_BREAKS = ("\r\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")


def test_transcribed_line_breaks_become_newlines_so_a_binary_build_passage_round_trips(
    tmp_path: Path,
) -> None:
    # A house paragraph whose lines a transcription separates with every other line break; the
    # same transcription for two reports in two project folders, so the paragraph recurs.
    lines = [
        f"House rule line {word} applies to every survey we deliver." for word in "abcdefghijk"
    ]
    paragraph = lines[0] + "".join(b + line for b, line in zip(LINE_BREAKS, lines[1:], strict=True))
    transcribed = page(
        block("heading", f"Survey{LINE_BREAKS[-2]}Report", level=1),
        block("paragraph", "Deliverable: survey report"),
        block("paragraph", paragraph),
        visuals=(
            VisualTranscript(
                label=f"Depth{LINE_BREAKS[1]}chart", transcript=f"Depth{LINE_BREAKS[0]}rises."
            ),
        ),
    )
    corpus_dir = tmp_path / "corpus"
    for project in ("p1", "p2"):
        (corpus_dir / project).mkdir(parents=True)
        write_pdf(corpus_dir / project / "report.pdf", [["Survey Report", "House rule line"]])
    gateway = FakeGateway(page_script=[transcribed, transcribed])

    corpus = compile_corpus(corpus_dir, gateway, topic="t", title="T", max_concurrency=1)

    sections = [s for d in corpus.documents for s in d.sections]
    assert sections and all(s.markdown.splitlines() == s.markdown.split("\n") for s in sections)
    # In a heading or visual label another line break becomes a space, so neither adds a line
    # that units would read as prose or that would end the visual block.
    assert sections[0].heading_path == "Survey Report"
    assert "> **[Visual \u2014 Page 1: Depth chart]**\n> Depth\n> rises." in sections[0].markdown
    assert without_visual_transcripts(sections[0].markdown).count("Depth") == 0
    (passage,) = corpus.blueprints[0].passages
    assert passage.text == "\n".join(lines) and passage.text in sections[0].markdown
    output = tmp_path / "t.arkive.db"
    write_snapshot(corpus, output)
    with TopicSnapshot.open(output) as snap:
        assert snap.blueprints[0].passages == (passage,)


def test_a_transcription_with_newlines_only_compiles_as_before_line_breaks_were_normalised(
    tmp_path: Path,
) -> None:
    # The expected values are the output of this build at commit 399aa4c, before transcribed
    # line breaks were normalised: a `\n` in a heading or visual label stays where it is, so
    # sections, chunks and the observation request (whose text keys the reading cache) do
    # not change.
    transcribed = page(
        block("heading", "Survey\nReport", level=1),
        block("paragraph", "Deliverable: survey report"),
        block("heading", "## Method\nand scope\n", level=2),
        block("paragraph", "House rule line one applies to every survey.\nLine two follows.\n"),
        block("table", "| Item | Value |\n| --- | --- |\n| Grid | GeoTIFF |\n"),
        visuals=(VisualTranscript(label="Chart\nof depth", transcript="Depth\nrises.\n"),),
    )
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "p1").mkdir(parents=True)
    write_pdf(corpus_dir / "p1" / "report.pdf", [["Survey Report", "House rule line"]])
    gateway = FakeGateway(page_script=[transcribed])

    (doc,) = compile_corpus(corpus_dir, gateway, topic="t", title="T").documents

    method = (
        "## Method\nand scope\n\nHouse rule line one applies to every survey.\nLine two follows."
        "\n\n| Item | Value |\n| --- | --- |\n| Grid | GeoTIFF |\n\n"
        "> **[Visual \u2014 Page 1: Chart\nof depth]**\n> Depth\n> rises."
    )
    assert [(s.heading_path, s.markdown) for s in doc.sections] == [
        ("Survey\nReport", "# Survey\nReport\n\nDeliverable: survey report"),
        ("Survey\nReport > Method\nand scope", method),
    ]
    method_crumb = "[Doc: Survey\nReport (pp. 1-1) | Section: Survey\nReport > Method\nand scope]"
    assert [(c.breadcrumb, c.content) for c in doc.chunks] == [
        ("[Doc: Survey\nReport (pp. 1-1) | Section: Survey\nReport]", "Deliverable: survey report"),
        (method_crumb, "House rule line one applies to every survey.\nLine two follows."),
        (method_crumb, "| Item | Value |\n| --- | --- |\n| Grid | GeoTIFF |"),
        (method_crumb, "> **[Visual \u2014 Page 1: Chart\nof depth]**\n> Depth\n> rises."),
    ]
    (observed,) = gateway.of_type(ObserveSections)
    assert observed.text() == (
        "SECTIONS:\n=== Section 1: Survey\nReport (pp. 1-1) ===\n# Survey\nReport\n\n"
        "Deliverable: survey report\n\n"
        f"=== Section 2: Survey\nReport > Method\nand scope (pp. 1-1) ===\n{method}"
    )


def test_continued_table_merges_split_row_and_drops_repeated_header() -> None:
    first = (
        "| Site | Area (ha) |\n| --- | --- |\n| Jimblebar | 1,204 |\n| Mining Area | 59,494 |\n"
        "| Eastern | 33"
    )
    repeated_header = "| Site | Area (ha) |\n| --- | --- |\n| Ridge | 0.5 |\n| Newman | 812 |"
    sections = assemble_sections(
        [
            (4, page(block("heading", "Areas", level=1), block("table", first))),
            (5, page(block("table", repeated_header, continues=True, merges=True))),
        ]
    )

    assert sections[0].markdown.split("\n\n")[1] == (
        "| Site | Area (ha) |\n| --- | --- |\n| Jimblebar | 1,204 |\n| Mining Area | 59,494 |\n"
        "| Eastern Ridge | 33 0.5 |\n| Newman | 812 |"
    )
    assert (sections[0].page_start, sections[0].page_end) == (4, 5)


def test_headless_continuation_skips_visuals_and_appends_rows() -> None:
    chart = VisualTranscript(label="Figure 2", transcript="Nine Year Median = 330.9 mm")
    table = "| Year | Rain |\n| --- | --- |\n| 2016 | 210 |"
    sections = assemble_sections(
        [
            (1, page(block("table", table), visuals=(chart,))),
            (2, page(block("table", "| 2017 | 402 |", continues=True), block("paragraph", "End"))),
        ]
    )

    assert sections[0].markdown == (
        "| Year | Rain |\n| --- | --- |\n| 2016 | 210 |\n| 2017 | 402 |\n\n"
        "> **[Visual \u2014 Page 1: Figure 2]**\n> Nine Year Median = 330.9 mm\n\nEnd"
    )


def test_continued_paragraph_joins_previous_paragraph() -> None:
    sections = assemble_sections(
        [
            (1, page(block("paragraph", "The model was"))),
            (2, page(block("paragraph", "validated in 2024.", continues=True))),
        ]
    )
    assert sections[0].markdown == "The model was validated in 2024."


def test_table_blocks_keep_empty_cells_captions_and_pipeless_rows() -> None:
    raw = (
        "Table 3: Areas\n|| Site | Area |\n| --- | --- | --- |\n|| Jimblebar | 1,204 |\n"
        "Source: survey 2024"
    )

    assert split_table_block(raw) == [
        ("paragraph", "Table 3: Areas"),
        ("table", "|  | Site | Area |\n| --- | --- | --- |\n|  | Jimblebar | 1,204 |"),
        ("paragraph", "Source: survey 2024"),
    ]
    assert split_table_block("Year | Rain\n--- | ---\n2016 | 210") == [
        ("table", "| Year | Rain |\n| --- | --- |\n| 2016 | 210 |")
    ]
    assert split_table_block("Ratio 5 | 6 overall") == [("paragraph", "Ratio 5 | 6 overall")]
    (section,) = assemble_sections([(1, page(block("table", raw)))])
    assert "Table 3: Areas" in section.markdown and section.markdown.endswith("survey 2024")


def pieces(drafts: Sequence[SectionDraft]) -> list[ChunkPiece]:
    return [piece for draft in drafts for piece in section_pieces(draft, "prose")]


def test_chunk_pieces_stitched_across_a_page_break_span_both_pages() -> None:
    drafts = assemble_sections(
        [
            (4, page(block("heading", "Areas", level=1), block("table", "| a |\n| --- |\n| 1 |"))),
            (5, page(block("table", "| 2 |", continues=True), block("paragraph", "The model"))),
            (6, page(block("paragraph", "was validated.", continues=True))),
        ]
    )

    chunks = pieces(drafts)

    assert [(c.kind, c.content, c.page_start, c.page_end) for c in chunks] == [
        ("table", "| a |\n| --- |\n| 1 |\n| 2 |", 4, 5),
        ("prose", "The model was validated.", 5, 6),
    ]


def test_every_chunk_is_bounded_for_oversized_sentences_rows_and_visuals() -> None:
    sentence = " ".join(["word"] * 3000)
    table = "| h |\n| --- |\n| " + " ".join(["cell"] * 2000) + " |\n| tail |"
    legend = VisualTranscript(label="Map", transcript="\n".join(["legend entry"] * 2000))
    drafts = assemble_sections(
        [
            (
                1,
                page(
                    block("heading", "Big", level=1),
                    block("paragraph", sentence),
                    block("table", table),
                    visuals=(legend,),
                ),
            )
        ]
    )

    chunks = pieces(drafts)

    assert max(len(c.content) for c in chunks) <= MAX_CHUNK_CHARS
    by_kind = {k: [c.content for c in chunks if c.kind == k] for k in ("prose", "table", "visual")}
    assert " ".join(by_kind["prose"]).split() == sentence.split()
    assert all(text.startswith("| h |\n| --- |\n") for text in by_kind["table"])
    assert sum(text.count("cell") for text in by_kind["table"]) == 2000
    assert sum(text.count("legend entry") for text in by_kind["visual"]) == 2000


def test_heading_level_is_ignored_off_headings_and_zero_reads_as_level_two() -> None:
    sections = assemble_sections(
        [
            (
                1,
                page(
                    block("heading", "Intro", level=1),
                    block("heading", "3.3. Wet and Dry Years Analysis"),
                    block("paragraph", "Body", level=3),
                ),
            )
        ]
    )

    assert [(s.heading_path, s.level) for s in sections] == [
        ("Intro", 1),
        ("Intro > 3.3. Wet and Dry Years Analysis", 2),
    ]
    # OpenAI strict structured outputs reject schemas that carry `default`.
    assert '"default"' not in json.dumps(PageExtraction.model_json_schema())


def test_markdown_fenced_code_stays_one_literal_block() -> None:
    text = "# Run\n\n```sh\n# not a heading\n| not | a table |\n\n> nor a callout\n```\nAfter."

    blocks = parse_markdown_blocks(text).blocks

    assert [(b.kind, b.markdown) for b in blocks] == [
        ("heading", "Run"),
        ("paragraph", "```sh\n# not a heading\n| not | a table |\n\n> nor a callout\n```"),
        ("paragraph", "After."),
    ]


def test_large_tables_split_on_rows_with_header_on_every_piece() -> None:
    rows = "\n".join(f"| Site {i:03d} | {'x' * 60} | {i * 7} |" for i in range(120))
    table = f"| Site | Notes | Area |\n| --- | --- | --- |\n{rows}"

    pieces = split_table(table)

    assert len(pieces) > 2
    for piece in pieces:
        assert piece.startswith("| Site | Notes | Area |\n| --- | --- | --- |\n| Site ")
        assert len(piece) <= 2000
    body_rows = [line for piece in pieces for line in piece.splitlines()[2:]]
    assert body_rows == rows.splitlines()
    assert split_table("| a |\n| --- |\n| 1 |") == ["| a |\n| --- |\n| 1 |"]


# Fidelity gate.


def test_running_header_and_footer_numbers_are_not_fidelity_anchors() -> None:
    pages = [
        f"Project 2025-118 Memorandum\nSection {name}\nBody value {n * 111}\nMore {name} text"
        f"\nPage {n} of 3\nCONFIDENTIAL 2025"
        for n, name in ((1, "Alpha"), (2, "Beta"), (3, "Gamma"))
    ]
    furniture = find_furniture(pages)
    body = strip_furniture(pages[1], furniture, 1)

    assert body == "Section Beta\nBody value 222\nMore Beta text"
    filenames, numbers = fidelity_anchors(body + "\nharbour_dry_depth_thresh.tif covers 59,494 ha")
    assert filenames.keys() == {"harbour_dry_depth_thresh.tif"}
    assert numbers == {"222", "59494"}
    assert fidelity_anchors(pages[1])[1] == {"2025", "118", "222"}
    assert find_furniture(pages[:2]) == frozenset()


def test_furniture_keeps_varying_data_lines_and_strips_page_tracking_titles() -> None:
    pages = [f"Survey Memo {i + 3}\nArea {area}" for i, area in enumerate([812, 1204, 5949])]
    furniture = find_furniture(pages)

    assert [strip_furniture(text, furniture, i) for i, text in enumerate(pages)] == [
        "Area 812",
        "Area 1204",
        "Area 5949",
    ]
    assert strip_furniture("12\nResults 330.9\n7", frozenset(), 0) == "Results 330.9"


def test_fidelity_anchors_keep_signs_and_short_filenames() -> None:
    filenames, numbers = fidelity_anchors("Min \u2212330.9 m in a.tif; 2016-2017; v1.2.nc")

    assert filenames.keys() == {"a.tif", "v1.2.nc"}
    assert numbers == {"-330.9", "2016", "2017"}
    dropped_sign = page(block("paragraph", "Min 330.9 m, 2016-2017"))
    assert missing_anchors(1, dropped_sign, {}, numbers) == ["-330.9"]
    assert missing_anchors(1, page(block("paragraph", "Gain -330.9")), {}, {"330.9"}) == ["330.9"]
    # The caption line of a table block survives assembly, so its value counts.
    captioned = page(block("table", "Range 2016-2017, min -330.9\n| a |\n| --- |\n| 1 |"))
    assert missing_anchors(1, captioned, {}, numbers) == []


def test_missing_anchors_accept_verbatim_dotted_identifiers_and_rejoined_filenames() -> None:
    doi = "See https://doi.org/10.1016/j.rse.2025.114694. for 59,494 ha"
    filenames, numbers = fidelity_anchors(doi)

    assert "2025.114694" in numbers
    assert missing_anchors(1, page(block("paragraph", doi)), filenames, numbers) == []
    # Native text wraps the filename mid-extension; the transcription rejoins it.
    filenames, numbers = fidelity_anchors("Inputs NWM-2016-stats.cs\nv and grid.tif")
    rejoined = page(block("paragraph", "Inputs NWM-2016-stats.csv and grid.tif"))

    assert (set(filenames), numbers) == ({"grid.tif"}, {"2016"})
    assert missing_anchors(1, rejoined, filenames, numbers) == []
    dropped = page(block("paragraph", "Inputs NWM-stats.csv and grid.tiff"))
    assert missing_anchors(1, dropped, filenames, numbers) == ["grid.tif", "2016"]


def test_missing_anchors_accept_filenames_rejoined_across_native_wraps() -> None:
    # Narrow table cells wrap a long filename mid-identifier; the anchor holds only its tail.
    filenames, numbers = fidelity_anchors(
        "Counts aoi_tiles_v2_1000x1000_\nSentinel_scene_counts.shp"
    )
    whole = page(block("paragraph", "Counts aoi_tiles_v2_1000x1000_Sentinel_scene_counts.shp"))

    assert filenames.keys() == {"sentinel_scene_counts.shp"}
    assert missing_anchors(1, whole, filenames, numbers) == []
    # A wrap followed by a kerning space splits the name into three fragments.
    filenames, numbers = fidelity_anchors(
        "In NWM_harbour_depth_frequency_all_y\nears _2016_2024.tif"
    )
    whole = page(block("paragraph", "In NWM_harbour_depth_frequency_all_years_2016_2024.tif"))

    assert filenames.keys() == {"_2016_2024.tif"}
    assert missing_anchors(1, whole, filenames, numbers) == []


def test_missing_anchors_reject_a_dropped_filename_beside_a_kept_one() -> None:
    for native, emitted in (
        ("Compare a.tif with data.tif", "Compare with data.tif"),
        ("Header line\na.tif and data.tif", "Header line\ndata.tif"),
    ):
        filenames, numbers = fidelity_anchors(native)
        dropped = page(block("paragraph", emitted))

        assert missing_anchors(1, dropped, filenames, numbers) == ["a.tif"]


def test_fidelity_retry_at_high_effort_names_and_recovers_missing_numbers() -> None:
    native = "Nine Year Median = 330.9 mm\nArea 59,494 ha\nThreshold 0.05"
    lossy = page(block("paragraph", "Nine Year Median rainfall. Area 59494 ha."))
    faithful = page(block("paragraph", "Nine Year Median = 330.9 mm. Area 59,494 ha. p 0.05"))
    gateway = FakeGateway(page_script=[lossy, faithful])

    result = transcribe_with_fidelity(
        gateway,
        rel_path="memo.pdf",
        page_number=3,
        native_text=native,
        body_text=native,
        webp_bytes=b"RIFF....WEBP",
        prev_context=None,
    )

    assert result == faithful
    assert gateway.transcribe_calls == [("low", ()), ("high", ("0.05", "330.9"))]
    assert [(a.effort, a.threshold, a.few_anchor_threshold) for a in FIDELITY_ATTEMPTS] == [
        ("low", 0.95, 1.0),
        ("high", 0.85, 0.85),
    ]


def test_fidelity_gate_accepts_the_first_attempt_at_its_own_threshold() -> None:
    # 20 anchors, one dropped: 0.95 recall passes the first attempt's threshold.
    native = " ".join(str(100 + i) for i in range(20))
    lossy = page(block("paragraph", native.replace("119", "")))
    gateway = FakeGateway(page_script=[lossy])

    transcribe_with_fidelity(
        gateway,
        rel_path="memo.pdf",
        page_number=1,
        native_text=native,
        body_text=native,
        webp_bytes=b"RIFF....WEBP",
        prev_context=None,
    )

    assert gateway.transcribe_calls == [("low", ())]


def test_fidelity_gate_fails_loudly_when_a_filename_stays_missing() -> None:
    native = "Output raster harbour_dry_depth_thresh.tif"
    lossy = page(block("paragraph", "Output raster file."))
    gateway = FakeGateway(page_script=[lossy, lossy])

    with pytest.raises(TranscriptionFidelityError, match=r"memo\.pdf page 7.*thresh\.tif"):
        transcribe_with_fidelity(
            gateway,
            rel_path="memo.pdf",
            page_number=7,
            native_text=native,
            body_text=native,
            webp_bytes=b"RIFF....WEBP",
            prev_context=None,
        )
    assert gateway.transcribe_calls == [("low", ()), ("high", ("harbour_dry_depth_thresh.tif",))]


# Document dates.


@pytest.mark.parametrize(
    ("printed", "iso"),
    [
        # Every printed date the observe step returned for the private evaluation corpus.
        ("15 April 2026", "2026-04-15"),
        ("15/04/2026", "2026-04-15"),
        ("19\u1d57\u02b0 Feb 2026", "2026-02-19"),
        ("20/11/2025", "2025-11-20"),
        ("20/2/2026", "2026-02-20"),
        ("21\u02e2\u1d57 October 2025", "2025-10-21"),
        ("February 2025", "2025-02"),
        ("June 12, 2026", "2026-06-12"),
        # The other unambiguous forms.
        ("2026-03-05", "2026-03-05"),
        ("3rd  march 2026", "2026-03-03"),
        ("Sept 2026", None),
        ("Sep 2026", "2026-09"),
        ("12-31-2025", "2025-12-31"),
        # Ambiguous or not a date: never guessed.
        ("12/02/2026", None),
        ("02/12/2026", None),
        ("31/13/2026", None),
        ("30/02/2026", None),
        ("2026-02-30", None),
        ("12/02/26", None),
        ("Q3 2026", None),
        ("Spring 2026", None),
        ("2026", None),
        ("Issued 15 April 2026", None),
        # Month names are ASCII English only: a dotted capital I is not an I.
        ("19 APR\u0130L 2026", None),
        ("APR\u0130L 2026", None),
        ("\u0661\u0669 April 2026", None),
    ],
)
def test_printed_dates_parse_only_in_unambiguous_forms(printed: str, iso: str | None) -> None:
    assert parse_printed_date(printed) == iso


def source_of(rel_path: str) -> SourceFile:
    path = Path(rel_path)
    return SourceFile(rel_path, path, SUPPORTED_SUFFIXES[path.suffix])


def pptx_bytes(tmp_path: Path, core: str | None) -> bytes:
    path = tmp_path / "deck.pptx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/presentation.xml", "<p:presentation/>")
        if core is not None:
            archive.writestr("docProps/core.xml", core)
    return path.read_bytes()


def test_embedded_dates_come_from_pdf_info_and_office_core_properties(tmp_path: Path) -> None:
    # The calendar date in the timestamp's own offset: 01:30 at +10:00 is still 20 February.
    dated_pdf = write_pdf(tmp_path / "a.pdf", [["Text"]], creation_date="D:20260220013000+10'00'")
    undated_pdf = write_pdf(tmp_path / "b.pdf", [["Text"]])
    # `modified`, not the template's `created` (2019-01-01).
    docx = write_docx(tmp_path / "a.docx", ["Text"], modified="2026-02-19T23:30:00Z")
    bare_docx = write_docx(tmp_path / "b.docx", ["Text"])
    pdf, deck = source_of("a.pdf"), source_of("deck.pptx")

    assert embedded_date(pdf, dated_pdf.read_bytes()) == "2026-02-20"
    assert embedded_date(pdf, undated_pdf.read_bytes()) is None
    assert embedded_date(source_of("a.docx"), docx.read_bytes()) == "2026-02-19"
    assert embedded_date(source_of("b.docx"), bare_docx.read_bytes()) is None
    deck_bytes = pptx_bytes(tmp_path, core_properties("2025-11-03T08:00:00Z"))
    assert embedded_date(deck, deck_bytes) == "2025-11-03"
    assert embedded_date(deck, pptx_bytes(tmp_path, None)) is None
    assert embedded_date(source_of("memo.md"), b"# Memo\n\nDate: 2026-01-01\n") is None


@pytest.mark.parametrize(
    ("creation_date", "iso"),
    [
        ("D:20260220", "2026-02-20"),
        ("D:20260220013000Z", "2026-02-20"),
        ("D:20260220013000-05'00", "2026-02-20"),
        ("D:202602", "2026-02"),
        ("D:2026", None),
        # A valid-looking prefix with anything else after it is not a PDF date.
        ("D:20260220nonsense", None),
        ("D:20260220013000+10'00'junk", None),
        ("20260220013000Z", None),
        ("D:20260220253000Z", None),
        ("D:20260230", None),
    ],
)
def test_a_pdf_date_gives_a_date_only_when_the_whole_value_is_a_pdf_date(
    tmp_path: Path, creation_date: str, iso: str | None
) -> None:
    pdf = write_pdf(tmp_path / "a.pdf", [["Text"]], creation_date=creation_date)
    assert embedded_date(source_of("a.pdf"), pdf.read_bytes()) == iso


@pytest.mark.parametrize(
    ("modified", "iso"),
    [
        ("2026-02-19", "2026-02-19"),
        ("2026-02-19T23:30Z", "2026-02-19"),
        ("2026-02-19T23:30:00.125+10:00", "2026-02-19"),
        # No zone: the written calendar date.
        ("2026-02-19T23:30:00", "2026-02-19"),
        (" 2026-02-19T23:30:00Z\n", "2026-02-19"),
        # A valid-looking prefix with anything else after it is not a W3CDTF value.
        ("2026-02-19garbage", None),
        ("2026-02-19T23:30:00Zjunk", None),
        ("2026-02-19T24:00:00Z", None),
        ("2026-02", None),
        ("yesterday", None),
        ("", None),
    ],
)
def test_an_office_modified_date_gives_a_date_only_when_the_whole_value_is_w3cdtf(
    tmp_path: Path, modified: str, iso: str | None
) -> None:
    deck = pptx_bytes(tmp_path, core_properties(modified))
    assert embedded_date(source_of("deck.pptx"), deck) == iso


def test_corrupt_embedded_metadata_is_a_conversion_error_naming_the_file(tmp_path: Path) -> None:
    no_modified = core_properties("2026-02-19").replace("dcterms:modified", "dcterms:other")
    assert embedded_date(source_of("deck.pptx"), pptx_bytes(tmp_path, no_modified)) is None

    with pytest.raises(ConversionError, match=r"^p/deck\.pptx: docProps/core\.xml is not"):
        embedded_date(source_of("p/deck.pptx"), pptx_bytes(tmp_path, "<cp:coreProperties>"))
    with pytest.raises(ConversionError, match=r"^p/memo\.docx: cannot read as a zip archive"):
        embedded_date(source_of("p/memo.docx"), b"not a zip archive")
    with pytest.raises(ConversionError, match=r"^p/scan\.pdf: cannot open PDF"):
        embedded_date(source_of("p/scan.pdf"), b"not a pdf")


def test_a_printed_date_wins_and_an_unparsed_one_never_falls_back_to_the_embedded_date(
    tmp_path: Path,
) -> None:
    pdf = write_pdf(tmp_path / "a.pdf", [["Text"]], creation_date="D:20250701000000Z").read_bytes()
    source, memo = source_of("a.pdf"), source_of("memo.md")

    assert document_date(source, pdf, "15/04/2026") == DocumentDate(
        iso="2026-04-15", basis="printed"
    )
    assert document_date(source, pdf, "12/02/2026") is None
    assert document_date(source, pdf, None) == DocumentDate(iso="2025-07-01", basis="embedded")
    assert document_date(source, pdf, "  ") == DocumentDate(iso="2025-07-01", basis="embedded")
    assert document_date(memo, b"# Memo", None) is None
    assert document_date(memo, b"# Memo", "February 2025") == DocumentDate(
        iso="2025-02", basis="printed"
    )


def test_reading_a_document_that_prints_a_non_english_month_gives_no_date() -> None:
    text = "# Memo\n\nDate: 19 APR\u0130L 2026\n\nBody text.\n"
    reading = read_document(source_of("memo.md"), text.encode(), FakeGateway())
    assert (reading.card.doc_date, reading.date) == ("19 APR\u0130L 2026", None)


def test_dates_order_documents_only_where_they_show_an_order() -> None:
    def day(iso: str) -> DocumentDate:
        return DocumentDate(iso=iso, basis="printed")

    later, earlier = day("2026-04-15"), day("2026-02-19")
    assert is_strictly_newer(later, earlier) and not is_strictly_newer(earlier, later)
    assert is_strictly_newer(day("2026-04"), earlier)
    # Equal dates, a month and a day inside it, and a missing date give no order either way.
    for a, b in [(later, day("2026-04-15")), (day("2026-04"), later), (later, None)]:
        assert not is_strictly_newer(a, b) and not is_strictly_newer(b, a)
    # A month is compared by the days it covers: equal in its first and last day, newer than
    # the day before it starts, older than the day after it ends.
    for edge in ("2026-02-01", "2026-02-28"):
        assert not is_strictly_newer(day("2026-02"), day(edge))
        assert not is_strictly_newer(day(edge), day("2026-02"))
    assert is_strictly_newer(day("2026-02"), day("2026-01-31"))
    assert is_strictly_newer(day("2026-03-01"), day("2026-02"))
    assert is_strictly_newer(day("2025-01"), day("2024-12"))
    for invalid in ("2026-13", "\u0662\u0660\u0662\u0666-04-15", "2026-0\uff14"):
        with pytest.raises(ValueError, match="not a YYYY-MM-DD date"):
            day(invalid)


# Binding.


def reading_of(drafts: Sequence[SectionDraft], title: str) -> DocumentReading:
    return DocumentReading(
        sha256="0" * 64,
        format="pdf",
        is_curator_note=False,
        sections=tuple(
            ReadSection(
                ordinal=i,
                heading_path=d.heading_path,
                level=d.level,
                page_start=d.page_start,
                page_end=d.page_end,
                markdown=d.markdown,
                pieces=section_pieces(d, "prose"),
            )
            for i, d in enumerate(drafts, start=1)
        ),
        pages=(),
        card=ObservationCard(
            title=title,
            doc_date=None,
            deliverable_type="memorandum",
            summary="Rainfall.",
            attributes=(),
            terms=(),
            craft=(),
        ),
        date=None,
        discarded_quotes=0,
    )


def facet(name: str, *values: str) -> Facet:
    return Facet(
        name=name,
        description=name,
        binding="content",
        values=tuple(FacetValue(value=v, description=v, aliases=(), folders=()) for v in values),
    )


def test_binding_numbers_chunks_and_builds_breadcrumbs_from_bound_facets() -> None:
    chart = VisualTranscript(label="Rainfall chart", transcript="Median 330.9 mm")
    long_paragraph = " ".join(f"Sentence {i} about groundwater." for i in range(150))
    drafts = assemble_sections(
        [
            (2, page(block("heading", "Climate", level=1), block("paragraph", long_paragraph))),
            (3, page(block("paragraph", "Short note."), visuals=(chart,))),
        ]
    )
    facets = (facet("project", "p1", "p2"), facet("client", "Northwind Marine", "Contoso Ports"))
    assignment = {"project": (), "client": ("Northwind Marine", "Contoso Ports")}

    doc = bind_document(
        CorpusDocument("doc_007", "p1/memo.pdf", reading_of(drafts, "Memo")), facets, assignment
    )

    assert doc.facets == {"client": ("Contoso Ports", "Northwind Marine")}
    assert [s.section_id for s in doc.sections] == ["doc_007_sec_001"]
    chunks = doc.chunks
    assert [c.chunk_id for c in chunks][:2] == ["doc_007_chk_0001", "doc_007_chk_0002"]
    # Facets in definition order, values sorted, a facet without values left out.
    assert {c.breadcrumb for c in chunks} == {
        "[client: Contoso Ports, Northwind Marine | Doc: Memo (pp. 2-3) | Section: Climate]"
    }
    prose = [c for c in chunks if c.kind == "prose"]
    assert len(prose) >= 3 and all(len(c.content) <= 1800 for c in prose)
    assert " ".join(c.content for c in prose).startswith("Sentence 0 about")
    (visual,) = [c for c in chunks if c.kind == "visual"]
    assert (visual.page_start, visual.page_end) == (3, 3)
    assert "Rainfall chart" in visual.content and "330.9" in visual.content
    unfaceted = bind_document(
        CorpusDocument("doc_001", "memo.pdf", reading_of(drafts, "M")), (), {}
    )
    assert unfaceted.chunks[0].breadcrumb == "[Doc: M (pp. 2-3) | Section: Climate]"


# Observation cards.

LOCATION_SECTIONS = (
    BatchSection(
        1,
        "Areas",
        1,
        1,
        "| Layer | File |\n| --- | --- |\n| Dry season | harbour_dry_depth_thresh.tif |",
    ),
    BatchSection(
        2,
        "Rainfall",
        2,
        2,
        "> **[Visual \u2014 Page 2: Rainfall chart]**\n> Nine Year Median = 330.9 mm\n"
        "> for 2016\u20132017",
    ),
    BatchSection(3, "Methods", 3, 3, "The multi-\nbeam survey used `LAT` as the *vertical* datum."),
    BatchSection(
        4,
        "Grid",
        4,
        4,
        "The multi\u2010\nswath run used _LAT_ on a 2 * 3 m grid, file dry_run.tif.",
    ),
)


@pytest.mark.parametrize(
    ("quote", "ordinal"),
    [
        ("Dry season harbour_dry_depth_thresh.tif", 1),
        ("| Dry season | harbour_dry_depth_thresh.tif |", 1),
        ("NINE YEAR MEDIAN = 330.9 mm for 2016-2017", 2),
        ("Rainfall chart] Nine Year Median", 2),
        ("multibeam survey used LAT as the vertical datum", 3),
        # A Unicode hyphen at a line break joins; `_` emphasis is ignored.
        ("multiswath run used LAT", 4),
        ("2 * 3 m grid", 4),
        ("file dry_run.tif", 4),
        # A literal operator between spaces and an intraword underscore are text, not markup.
        ("on a 2 3 m grid", None),
        ("file dryrun.tif", None),
        ("survey used lidar", None),
        ("  ", None),
    ],
)
def test_quotes_are_located_through_markup_case_dashes_and_hyphenation(
    quote: str, ordinal: int | None
) -> None:
    assert locate_quote(quote, LOCATION_SECTIONS) == ordinal


def draft(*quotes: str, craft: Sequence[str] = ()) -> ObservationDraft:
    return ObservationDraft(
        title="Datum memo",
        doc_date=None,
        deliverable_type="memorandum",
        summary="Datum.",
        attributes=[],
        terms=[
            TermDraft(term=q.split()[0], kind="acronym", aliases=[], quote=q, defines=True)
            for q in quotes
        ],
        craft=[CraftDraft(aspect="convention", statement=q, quote=q) for q in craft],
    )


DATUM_CORPUS = {"p1/memo.md": "# Datum\n\nLAT is the datum of every sounding.\n"}
LOCATED, UNLOCATED = "LAT is the datum", "MSL is mean sea level"


def test_unlocated_quotes_get_one_rerun_that_names_them(tmp_path: Path) -> None:
    gateway = FakeGateway(observe_script=[draft(LOCATED, UNLOCATED), draft(LOCATED)])

    corpus = compile_corpus(write_files(tmp_path, DATUM_CORPUS), gateway, topic="t", title="T")

    first, rerun = gateway.of_type(ObserveSections)
    assert first.feedback is None
    assert rerun.feedback is not None and repr(UNLOCATED) in rerun.feedback
    assert LOCATED not in rerun.feedback
    assert corpus.build_report.discarded_unlocated_quotes == 0


def test_quotes_still_unlocated_after_the_rerun_are_discarded_and_counted(tmp_path: Path) -> None:
    kept_then_lost = draft(LOCATED, UNLOCATED, craft=["Always state the datum loudly"])
    gateway = FakeGateway(observe_script=[kept_then_lost, kept_then_lost])

    corpus = compile_corpus(write_files(tmp_path, DATUM_CORPUS), gateway, topic="t", title="T")

    assert gateway.calls.count("observe") == 2
    assert corpus.build_report.discarded_unlocated_quotes == 2
    (entry,) = corpus.glossary
    assert entry.term == "LAT"


def test_a_batch_whose_quotes_are_never_located_raises(tmp_path: Path) -> None:
    gateway = FakeGateway(observe_script=[draft(UNLOCATED), draft(UNLOCATED)])

    with pytest.raises(RejectedOutputError, match=r"p1/memo\.md: observations.*MSL is mean"):
        compile_corpus(write_files(tmp_path, DATUM_CORPUS), gateway, topic="t", title="T")
    assert gateway.calls == ["observe", "observe"]
    # A rerun that drops every observation is the same broken reading, not an empty batch.
    silent = FakeGateway(observe_script=[draft(UNLOCATED), draft()])
    with pytest.raises(RejectedOutputError, match="none of its quotes could be located"):
        compile_corpus(write_files(tmp_path / "silent", DATUM_CORPUS), silent, topic="t", title="T")
    assert silent.calls == ["observe", "observe"]


def test_long_documents_are_observed_in_batches_then_consolidated(tmp_path: Path) -> None:
    body = " ".join(["sounding"] * (OBSERVE_BATCH_CHARS // 20))
    text = "".join(f"# Part {i}\n\n{body}\n\n" for i in range(1, 4))
    gateway = FakeGateway()

    corpus = compile_corpus(write_files(tmp_path, {"memo.md": text}), gateway, topic="t", title="T")

    batches = gateway.of_type(ObserveSections)
    assert [[s.ordinal for s in b.sections] for b in batches] == [[1, 2], [3]]
    assert gateway.calls.count("consolidate_card") == 1
    assert corpus.documents[0].title == "Part 1"


def test_an_oversized_section_is_observed_in_pieces_that_keep_its_ordinal() -> None:
    long_text = " ".join(f"sounding{i}" for i in range(OBSERVE_BATCH_CHARS // 4)) + " closing note"
    sections = [
        ReadSection(
            ordinal=ordinal,
            heading_path=heading,
            level=1,
            page_start=1,
            page_end=1,
            markdown=markdown,
            pieces=(),
        )
        for ordinal, heading, markdown in ((1, "Intro", "Short."), (2, "Log", long_text))
    ]

    batches = observation_batches(sections)

    assert len(long_text) > 2 * OBSERVE_BATCH_CHARS
    assert all(sum(len(s.markdown) for s in batch) <= OBSERVE_BATCH_CHARS for batch in batches)
    pieces = [s for batch in batches for s in batch if s.ordinal == 2]
    assert len(pieces) >= 3 and " ".join(p.markdown for p in pieces) == long_text
    assert locate_quote("closing note", batches[-1]) == 2


# Build cache: provider requests through `OpenAIGateway` and the scripted provider.

CLIENTS = FacetDraft(
    name="client",
    description="Client organisation.",
    values=[
        FacetValueDraft(value="Northwind Marine", description="NM", aliases=["NM"], folders=[]),
        FacetValueDraft(value="Contoso Ports", description="CP", aliases=[], folders=[]),
    ],
)
MEMO_TYPE = DeliverableTypeDraft(
    name="memorandum", description="Memo.", exemplar_doc_ids=["doc_001"], feedback_doc_ids=[]
)
MEMO_BLUEPRINT = BlueprintDraft(
    description="A survey memo.",
    outline=[
        BlueprintSectionDraft(heading="Results", purpose="Findings.", citations=["doc_001_sec_001"])
    ],
    conventions=[
        GuidanceDraft(statement="State areas in hectares.", citations=["doc_001_sec_001"])
    ],
    pitfalls=[],
    passages=[],
    house_sections=[HouseSectionNomination(section_id="doc_001_sec_001", label="Field safety")],
)
_TOKEN = re.compile(r"\[doc_\d+:doc_\d+_sec_\d+\]")


def cite_last_section(body: dict[str, Any]) -> GuideDraft:
    """A guide that cites the last section token of its own request's catalog."""
    text = body["input"][0]["content"][0]["text"]
    return GuideDraft(guide_markdown=f"Overview {_TOKEN.findall(text)[-1]}.")


def scripted_api() -> ScriptedOpenAI:
    api = ScriptedOpenAI()
    api.defaults["OrganisationDraft"] = OrganisationDraft(
        facets=[CLIENTS], deliverable_types=[MEMO_TYPE]
    )
    api.defaults["final_output"] = MEMO_BLUEPRINT
    api.defaults["GuideDraft"] = cite_last_section
    return api


def build(api: ScriptedOpenAI, corpus_dir: Path, cache: Path) -> Any:
    with BuildCache(cache) as build_cache:
        return compile_corpus(corpus_dir, api.gateway(cache=build_cache), topic="t", title="T")


def formats(api: ScriptedOpenAI, start: int = 0) -> list[str]:
    bodies = [e.body for e in api.exchanges[start:] if e.path == "/responses"]
    return sorted(b["text"]["format"]["name"] for b in bodies)


MEMO_CORPUS = {
    "p1/2024-memo.md": (
        "# Results\n\nSeagrass at 59,494 ha.\n\nEvery survey follows the house field safety plan."
    ),
    "p1/AGENTS.md": "Memo is final.",
    "p2/proposal.md": "# Scope\n\nA follow-up survey.",
}


def test_unchanged_corpus_rebuilds_from_the_cache_without_provider_requests(
    tmp_path: Path,
) -> None:
    corpus_dir = write_files(tmp_path / "corpus", MEMO_CORPUS)
    api = scripted_api()

    first = build(api, corpus_dir, tmp_path / "cache.db")
    requests_after_first = api.request_count
    second = build(api, corpus_dir, tmp_path / "cache.db")

    assert api.request_count == requests_after_first
    assert (second.documents, second.guide_markdown) == (first.documents, first.guide_markdown)
    assert (second.facets, second.blueprints) == (first.facets, first.blueprints)
    # Every operation of all three passes was made once: two observations, organisation, one
    # binding judgement per citable document, one blueprint agent run with its support check
    # and the house-text check of its nominated section's one unit, the guide, and the
    # embeddings.
    assert formats(api) == [
        "GuideDraft",
        "ObservationDraft",
        "ObservationDraft",
        "OrganisationDraft",
        "final_output",
    ]
    assert len(api.requests("/decisions")) == 4 and len(api.requests("/embeddings")) == 1
    efforts = {
        b["text"]["format"]["name"]: b["reasoning"]["effort"] for b in api.requests("/responses")
    }
    assert efforts == {
        "ObservationDraft": "low",
        "OrganisationDraft": "medium",
        "final_output": "medium",
        "GuideDraft": "medium",
    }
    assert first.blueprints[0].conventions[0].statement == "State areas in hectares."
    (house,) = first.blueprints[0].passages
    assert (house.kind, house.text) == (
        "single_exemplar",
        "Every survey follows the house field safety plan.",
    )
    assert first.build_settings["efforts"]["write_blueprint"] == "medium"
    assert first.build_settings["model"] == "gpt-6.1-sol"


def test_adding_a_file_requests_no_reading_of_the_existing_files(tmp_path: Path) -> None:
    corpus_dir = write_files(tmp_path / "corpus", MEMO_CORPUS)
    api = scripted_api()
    build(api, corpus_dir, tmp_path / "cache.db")
    start = len(api.exchanges)

    # Sorts first, so every existing document gets a new ID.
    write_files(corpus_dir, {"p0/new.md": "# Brief\n\nA new brief."})
    corpus = build(api, corpus_dir, tmp_path / "cache.db")

    observed = [
        e.body
        for e in api.exchanges[start:]
        if e.path == "/responses" and e.body["text"]["format"]["name"] == "ObservationDraft"
    ]
    assert len(observed) == 1 and "A new brief." in observed[0]["input"][0]["content"][0]["text"]
    assert "OrganisationDraft" in formats(api, start)
    assert corpus.documents[0].rel_path == "p0/new.md"


def test_a_subset_corpus_does_not_reuse_the_full_corpus_induction(tmp_path: Path) -> None:
    full = write_files(
        tmp_path / "full", {"a.md": "# Alpha\n\nAlpha survey.", "b.md": "# Beta\n\nBeta survey."}
    )
    subset = write_files(tmp_path / "subset", {"a.md": "# Alpha\n\nAlpha survey."})
    api = ScriptedOpenAI()
    api.defaults["GuideDraft"] = cite_last_section
    assert "[doc_002:doc_002_sec_001]" in build(api, full, tmp_path / "cache.db").guide_markdown
    start = len(api.exchanges)

    corpus = build(api, subset, tmp_path / "cache.db")

    # The reading of a.md is reused; organisation and the guide are requested again, and the
    # guide cites only the subset's documents.
    assert formats(api, start) == ["GuideDraft", "OrganisationDraft"]
    assert _TOKEN.findall(corpus.guide_markdown) == ["[doc_001:doc_001_sec_001]"]


# Rendered formats end to end.


def test_pdf_compiles_through_render_transcribe_and_observation(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus" / "p101-harbour-lidar"
    corpus_dir.mkdir(parents=True)
    write_pdf(
        corpus_dir / "harbour-memo.pdf",
        [
            [
                "# Rainfall",
                "Nine Year Median = 330.9 mm",
                "| Year | Rain |",
                "| --- | --- |",
                "| 2016 | 210 |",
            ],
            ["## Outputs", "Raster harbour_dry_depth_thresh.tif", "Deliverable: memorandum"],
        ],
    )

    corpus = compile_corpus(tmp_path / "corpus", FakeGateway(), topic="t", title="T")

    (doc,) = corpus.documents
    assert (doc.format, doc.title, doc.deliverable_type) == ("pdf", "Rainfall", "memorandum")
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert all(p.webp_bytes[8:12] == b"WEBP" for p in doc.pages)
    assert "330.9" in doc.pages[0].native_text
    assert doc.pages[1].visual_summary == "Heading: Outputs"
    assert [s.heading_path for s in doc.sections] == ["Rainfall", "Rainfall > Outputs"]
    assert {c.kind for c in doc.chunks} == {"prose", "table"}


def test_render_pdf_is_safe_and_deterministic_across_threads(tmp_path: Path) -> None:
    pdfs = [
        write_pdf(
            tmp_path / f"doc{i}.pdf", [[f"Doc {i} page {p} value {i * 100 + p}"] for p in range(4)]
        ).read_bytes()
        for i in range(6)
    ]
    expected = [render_pdf(pdf) for pdf in pdfs]

    with ThreadPoolExecutor(max_workers=6) as pool:
        for _ in range(3):
            assert list(pool.map(render_pdf, pdfs)) == expected
    assert expected[5][3][0].strip() == "Doc 5 page 3 value 503"


@needs_soffice
def test_word_and_powerpoint_compile_through_libreoffice(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus" / "p102-estuary-multibeam"
    corpus_dir.mkdir(parents=True)
    write_docx(corpus_dir / "estuary-memo.docx", ["# Memorandum", "Mapped 1,234 ha of seagrass."])
    fodp = write_flat_presentation(
        tmp_path / "summary.fodp", [["# Project Summary", "Seagrass 4567 ha"], ["Next steps"]]
    )
    profile = (tmp_path / "profile").as_uri()
    subprocess.run(
        [
            "soffice",
            "--headless",
            f"-env:UserInstallation={profile}",
            "--convert-to",
            "pptx",
            "--outdir",
            str(corpus_dir),
            str(fodp),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )

    corpus = compile_corpus(tmp_path / "corpus", FakeGateway(), topic="t", title="T")

    memo, deck = corpus.documents
    assert (memo.format, deck.format) == ("docx", "pptx")
    assert "1,234" in memo.sections[0].markdown
    assert len(deck.pages) == 2
    assert deck.sections[0].heading_path == "Project Summary"
    assert "4567" in deck.chunks[0].content
