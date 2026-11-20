from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import time
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pytest
from PIL import Image

from arkive_rag import compiler
from arkive_rag.compiler import (
    FIDELITY_ATTEMPTS,
    MAX_CHUNK_CHARS,
    MAX_CORE_PROPERTIES_BYTES,
    MAX_SECTION_CHARS,
    OBSERVE_BATCH_CHARS,
    BatchSection,
    FilenameAnchor,
    ObserveSections,
    PageTranscription,
    SectionDraft,
    SourceFile,
    assemble_sections,
    bind_document,
    compile_corpus,
    consolidated_date,
    curator_note_card,
    discover_corpus,
    document_date,
    encode_page_image,
    fidelity_anchors,
    find_furniture,
    locate_quote,
    metadata_date,
    missing_anchors,
    observation_batches,
    observe_document,
    parse_markdown_blocks,
    printed_date_or_none,
    read_source,
    render_pdf,
    render_source,
    section_pieces,
    split_table_block,
    strip_furniture,
    transcribe_with_fidelity,
    wrap_text,
)
from arkive_rag.induction import OrganiseCorpus, WriteGuide, binding_evidence
from arkive_rag.llm import BuildCache
from arkive_rag.models import (
    MAX_HEADING_CONTEXT_CHARS,
    MAX_RENDER_SIDE_PX,
    ArkiveError,
    CardHeader,
    ChunkPiece,
    ConversionError,
    CorpusDocument,
    CraftDraft,
    DocumentDate,
    DocumentReading,
    DocumentRole,
    DocumentRoleDraft,
    Facet,
    FacetDraft,
    FacetValue,
    FacetValueDraft,
    GuideDraft,
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
    document_id,
    parse_printed_date,
)
from arkive_rag.passages import standing_units, without_visual_transcripts
from arkive_rag.snapshot import TopicSnapshot, write_snapshot
from tests.support import (
    FakeGateway,
    ScriptedOpenAI,
    sec,
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


def sole_role(document: CorpusDocument) -> DocumentRole:
    """The role of a document that is its own version family and reviews nothing."""
    return DocumentRole(
        deliverable_type=document.reading.card.deliverable_type,
        is_feedback=False,
        feedback_for_types=(),
        family_id=document.doc_id,
        version_status="sole",
        superseded_by=None,
    )


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
    assert (guidance.doc_id, guidance.is_curator_note) == (
        document_id("p103-reef-sonar/AGENTS.md"),
        True,
    )
    assert (guidance.title, guidance.role.deliverable_type) == ("Status", "curator note")
    assert (guidance.role.family_id, guidance.role.version_status) == (guidance.doc_id, "sole")
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

    chunks = pieces(assemble_sections([(1, page(block("table", table)))]))

    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.content.startswith("| Site | Notes | Area |\n| --- | --- | --- |\n| Site ")
        assert len(chunk.content) <= 2000
    body_rows = [line for chunk in chunks for line in chunk.content.splitlines()[2:]]
    assert body_rows == rows.splitlines()
    (small,) = pieces(assemble_sections([(1, page(block("table", "| a |\n| --- |\n| 1 |")))]))
    assert small.content == "| a |\n| --- |\n| 1 |"


# Fidelity gate.


def names(filenames: Sequence[FilenameAnchor]) -> set[str]:
    return {anchor.name for anchor in filenames}


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
    assert names(filenames) == {"harbour_dry_depth_thresh.tif"}
    assert numbers == {"222", "59494"}
    assert fidelity_anchors(pages[1])[1] == {"2025", "118", "222"}
    # Two pages: a short edge line on both is furniture, a page's only line never is.
    assert strip_furniture(pages[1], find_furniture(pages[:2]), 1) == (
        "Section Beta\nBody value 222\nMore Beta text"
    )
    assert find_furniture(["Title 2025", "Title 2025"]) == frozenset()
    long_line = "Body " + "x" * 130
    assert find_furniture([f"{long_line}\nA 101", f"{long_line}\nB 202"]) == frozenset()


def test_furniture_keeps_varying_data_lines_and_strips_page_tracking_titles() -> None:
    pages = [f"Survey Memo {i + 3}\nArea {area}" for i, area in enumerate([812, 1204, 5949])]
    furniture = find_furniture(pages)

    assert [strip_furniture(text, furniture, i) for i, text in enumerate(pages)] == [
        "Area 812",
        "Area 1204",
        "Area 5949",
    ]
    assert strip_furniture("12\nResults 330.9\n7", frozenset(), 0) == "Results 330.9"
    # A number that starts a filename wrapped over lines is part of the name, not a page number.
    wrapped = "2026\n-final.tif\nResults 330.9\n7"
    assert strip_furniture(wrapped, frozenset(), 0) == "2026\n-final.tif\nResults 330.9"


def test_fidelity_anchors_keep_signs_and_short_filenames() -> None:
    filenames, numbers = fidelity_anchors("Min \u2212330.9 m in a.tif; 2016-2017; v1.2.nc")

    assert names(filenames) == {"a.tif", "v1.2.nc"}
    assert numbers == {"-330.9", "2016", "2017"}
    dropped_sign = page(block("paragraph", "Min 330.9 m, 2016-2017"))
    assert missing_anchors(1, dropped_sign, (), numbers) == ["-330.9"]
    assert missing_anchors(1, page(block("paragraph", "Gain -330.9")), (), {"330.9"}) == ["330.9"]
    # The caption line of a table block survives assembly, so its value counts.
    captioned = page(block("table", "Range 2016-2017, min -330.9\n| a |\n| --- |\n| 1 |"))
    assert missing_anchors(1, captioned, (), numbers) == []


def test_missing_anchors_accept_verbatim_dotted_identifiers_and_rejoined_filenames() -> None:
    doi = "See https://doi.org/10.1016/j.rse.2025.114694. for 59,494 ha"
    filenames, numbers = fidelity_anchors(doi)

    assert "2025.114694" in numbers
    assert missing_anchors(1, page(block("paragraph", doi)), filenames, numbers) == []
    # Native text wraps the filename mid-extension; the transcription rejoins it.
    filenames, numbers = fidelity_anchors("Inputs NWM-2016-stats.cs\nv and grid.tif")
    rejoined = page(block("paragraph", "Inputs NWM-2016-stats.csv and grid.tif"))

    assert (names(filenames), numbers) == ({"grid.tif"}, {"2016"})
    assert missing_anchors(1, rejoined, filenames, numbers) == []
    dropped = page(block("paragraph", "Inputs NWM-stats.csv and grid.tiff"))
    assert missing_anchors(1, dropped, filenames, numbers) == ["grid.tif", "2016"]


def test_missing_anchors_accept_filenames_rejoined_across_native_wraps() -> None:
    # Narrow table cells wrap a long filename mid-identifier; the anchor holds only its tail.
    filenames, numbers = fidelity_anchors(
        "Counts aoi_tiles_v2_1000x1000_\nSentinel_scene_counts.shp"
    )
    whole = page(block("paragraph", "Counts aoi_tiles_v2_1000x1000_Sentinel_scene_counts.shp"))

    assert names(filenames) == {"sentinel_scene_counts.shp"}
    assert missing_anchors(1, whole, filenames, numbers) == []
    # A wrap followed by a kerning space splits the name into three fragments.
    filenames, numbers = fidelity_anchors(
        "In NWM_harbour_depth_frequency_all_y\nears _2016_2024.tif"
    )
    whole = page(block("paragraph", "In NWM_harbour_depth_frequency_all_years_2016_2024.tif"))

    assert names(filenames) == {"_2016_2024.tif"}
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


# Binding.


def reading_of(
    drafts: Sequence[SectionDraft],
    title: str,
    *,
    printed_date: str | None = None,
    date: DocumentDate | None = None,
) -> DocumentReading:
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
            doc_date=printed_date,
            deliverable_type="memorandum",
            summary="Rainfall.",
            attributes=(),
            terms=(),
            craft=(),
        ),
        date=date,
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

    memo = document_id("p1/memo.pdf")
    printed = DocumentDate(iso="2025-11-20", basis="printed")
    source = CorpusDocument(
        memo, "p1/memo.pdf", reading_of(drafts, "Memo", printed_date="20/11/2025", date=printed)
    )
    doc = bind_document(source, facets, assignment, sole_role(source))

    assert doc.facets == {"client": ("Contoso Ports", "Northwind Marine")}
    assert (doc.doc_date, doc.date) == ("20/11/2025", printed)
    assert [s.section_id for s in doc.sections] == [f"{memo}_sec_001"]
    chunks = doc.chunks
    assert [c.chunk_id for c in chunks][:2] == [f"{memo}_chk_0001", f"{memo}_chk_0002"]
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
    # A document that prints no date shows its metadata date as written.
    embedded = DocumentDate(iso="2024-03-01", basis="metadata")
    plain = CorpusDocument(
        document_id("memo.pdf"), "memo.pdf", reading_of(drafts, "M", date=embedded)
    )
    unfaceted = bind_document(plain, (), {}, sole_role(plain))
    assert unfaceted.chunks[0].breadcrumb == "[Doc: M (pp. 2-3) | Section: Climate]"
    assert (unfaceted.doc_date, unfaceted.date) == ("2024-03-01", embedded)


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


def test_a_runaway_heading_is_cut_only_where_it_labels_section_text() -> None:
    heading = "Results " + "of the survey " * 400
    drafts = assemble_sections(
        [(1, page(block("heading", heading, level=1), block("paragraph", "B.")))]
    )
    reading = reading_of(drafts, "Memo")

    (batch,) = observation_batches(reading.sections)
    source = CorpusDocument(document_id("m.pdf"), "m.pdf", reading)
    doc = bind_document(source, (), {}, sole_role(source))

    label = heading.strip()[: MAX_HEADING_CONTEXT_CHARS - 1] + "…"
    assert [s.heading_path for s in batch] == [label]
    assert {c.breadcrumb for c in doc.chunks} == {f"[Doc: Memo (pp. 1-1) | Section: {label}]"}
    # The section keeps its whole heading.
    assert doc.sections[0].heading_path == heading.strip()
    assert heading.strip() in doc.sections[0].markdown


# Build cache: provider requests through `OpenAIGateway` and the scripted provider.

CLIENTS = FacetDraft(
    name="client",
    description="Client organisation.",
    values=[
        FacetValueDraft(value="Northwind Marine", description="NM", aliases=["NM"], folders=[]),
        FacetValueDraft(value="Contoso Ports", description="CP", aliases=[], folders=[]),
    ],
)
MEMO = "p1/2024-memo.md"


def memo_organisation(body: dict[str, Any]) -> OrganisationDraft:
    """The client facet, and every document of the request's batch: the memo a memorandum,
    any other document a proposal."""
    text = body["input"][0]["content"][0]["text"]
    roles = [
        DocumentRoleDraft(
            doc_id=doc_id,
            deliverable_type="memorandum" if doc_id == document_id(MEMO) else "proposal",
            is_feedback=False,
            feedback_for_types=[],
        )
        for doc_id in re.findall(r"^(doc_[0-9a-f]{10}) \|", text, re.MULTILINE)
    ]
    return OrganisationDraft(facets=[CLIENTS], documents=roles)


_TOKEN = re.compile(r"\[doc_[0-9a-f]{10}:doc_[0-9a-f]{10}_sec_\d+\]")


def cite_last_section(body: dict[str, Any]) -> GuideDraft:
    """A guide that cites the last section token of its own request's catalog."""
    text = body["input"][0]["content"][0]["text"]
    return GuideDraft(guide_markdown=f"Overview {_TOKEN.findall(text)[-1]}.")


def scripted_api() -> ScriptedOpenAI:
    api = ScriptedOpenAI()
    api.defaults["OrganisationDraft"] = memo_organisation
    api.defaults["GuideDraft"] = cite_last_section
    return api


def build(api: ScriptedOpenAI, corpus_dir: Path, cache: Path) -> Any:
    with BuildCache(cache) as build_cache:
        return compile_corpus(corpus_dir, api.gateway(cache=build_cache), topic="t", title="T")


def formats(api: ScriptedOpenAI, start: int = 0) -> list[str]:
    bodies = [e.body for e in api.exchanges[start:] if e.path == "/responses"]
    return sorted(b["text"]["format"]["name"] for b in bodies)


MEMO_CORPUS = {
    MEMO: "# Results\n\nSeagrass at 59,494 ha.",
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
    assert (second.facets, second.deliverable_types) == (first.facets, first.deliverable_types)
    assert second.unit_clusters == first.unit_clusters
    # Every operation of all three passes was made once: two observations, one organisation
    # batch and its type canonicalisation, one binding judgement per citable document, the
    # guide, and the embeddings. The two documents share no text or title, so no version pair
    # is judged.
    assert formats(api) == [
        "GuideDraft",
        "ObservationDraft",
        "ObservationDraft",
        "OrganisationDraft",
        "TypeCanonicalisationDraft",
    ]
    assert len(api.requests("/decisions")) == 2 and len(api.requests("/embeddings")) == 1
    efforts = {
        b["text"]["format"]["name"]: b["reasoning"]["effort"] for b in api.requests("/responses")
    }
    assert efforts == {
        "ObservationDraft": "low",
        "OrganisationDraft": "medium",
        "TypeCanonicalisationDraft": "medium",
        "GuideDraft": "medium",
    }
    assert [t.name for t in first.deliverable_types] == ["memorandum", "proposal"]
    assert first.build_settings["efforts"]["canonicalise_types"] == "medium"
    assert first.build_settings["efforts"]["extract_feedback_lessons"] == "low"
    assert first.build_settings["model"] == "gpt-6.1-sol"


def test_adding_a_file_requests_no_reading_of_the_existing_files(tmp_path: Path) -> None:
    corpus_dir = write_files(tmp_path / "corpus", MEMO_CORPUS)
    api = scripted_api()
    before = build(api, corpus_dir, tmp_path / "cache.db")
    start = len(api.exchanges)

    # Sorts first, yet every existing document keeps its ID: an ID depends on its path only.
    write_files(corpus_dir, {"p0/new.md": "# Brief\n\nA new brief."})
    corpus = build(api, corpus_dir, tmp_path / "cache.db")
    assert [(d.rel_path, d.doc_id) for d in corpus.documents[1:]] == [
        (d.rel_path, d.doc_id) for d in before.documents
    ]
    assert [c.chunk_id for d in corpus.documents[1:] for c in d.chunks] == [
        c.chunk_id for d in before.documents for c in d.chunks
    ]

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
    b_token = f"[{document_id('b.md')}:{sec('b.md')}]"
    assert b_token in build(api, full, tmp_path / "cache.db").guide_markdown
    start = len(api.exchanges)

    corpus = build(api, subset, tmp_path / "cache.db")

    # The reading of a.md is reused; organisation and the guide are requested again, and the
    # guide cites only the subset's documents.
    assert formats(api, start) == ["GuideDraft", "OrganisationDraft", "TypeCanonicalisationDraft"]
    assert _TOKEN.findall(corpus.guide_markdown) == [f"[{document_id('a.md')}:{sec('a.md')}]"]


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
    assert (doc.format, doc.title, doc.role.deliverable_type) == ("pdf", "Rainfall", "memorandum")
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


# Document identity.


def test_document_ids_hash_the_nfc_path_and_a_collision_stops_the_build_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert document_id("p1/memo.md") == f"doc_{hashlib.sha256(b'p1/memo.md').hexdigest()[:10]}"
    # The NFC and NFD spellings of one folder name are two directories on Linux.
    nfc, nfd = "caf\u00e9/memo.md", "cafe\u0301/memo.md"
    assert document_id(nfc) == document_id(nfd) != document_id("cafe/memo.md")
    write_files(tmp_path, {nfc: "# A\n\nText.", nfd: "# B\n\nText."})
    gateway = FakeGateway()

    def refuse(source: Any) -> bytes:
        raise AssertionError(f"read {source.rel_path}")

    monkeypatch.setattr(compiler, "read_source", refuse)
    with pytest.raises(ArkiveError, match="have the same document ID") as raised:
        compile_corpus(tmp_path, gateway, topic="t", title="T")
    assert nfc in str(raised.value) and nfd in str(raised.value)
    assert gateway.calls == []


# Document dates.


@pytest.mark.parametrize(
    ("printed", "iso"),
    [
        ("2026-02-19", "2026-02-19"),
        ("19\u1d57\u02b0 Feb 2026", "2026-02-19"),
        ("21\u02e2\u1d57 October 2025", "2025-10-21"),
        ("June 12, 2026", "2026-06-12"),
        ("20/11/2025", "2025-11-20"),
        ("15/04/2026", "2026-04-15"),
        ("20/2/2026", "2026-02-20"),
        ("4/20/2026", "2026-04-20"),
        ("February 2025", "2025-02"),
        ("2025-02", "2025-02"),
        # Day and month alike read the same either way round.
        ("01/01/2026", "2026-01-01"),
        ("12.12.2026", "2026-12-12"),
        ("2026/02/19", "2026-02-19"),
        ("2026.2.9", "2026-02-09"),
        # A Unicode dash reads as a hyphen.
        ("2025\u201305\u201319", "2025-05-19"),
        ("19\u221205\u22122025", "2025-05-19"),
        # No single reading: both numbers could be the month, a range, an impossible day.
        ("05/04/2026", None),
        ("March\u2013April 2026", None),
        ("31 February 2026", None),
        ("Q3 2025", None),
        ("2026/19/02", None),
        ("1899/01/02", None),
        ("13/13/2026", None),
    ],
)
def test_printed_dates_parse_only_when_they_have_one_reading(printed: str, iso: str | None) -> None:
    assert parse_printed_date(printed) == iso


def source_file(rel_path: str, fmt: Any) -> SourceFile:
    return SourceFile(rel_path, Path(rel_path), fmt, 0)


def test_metadata_dates_come_from_the_original_file_and_never_replace_a_printed_date(
    tmp_path: Path,
) -> None:
    created = write_pdf(
        tmp_path / "a.pdf",
        [["Body"]],
        {"CreationDate": "D:20250630164500+00'00'", "ModDate": "D:20250701090000Z"},
    ).read_bytes()
    modified_only = write_pdf(
        tmp_path / "b.pdf", [["Body"]], {"ModDate": "D:20250701"}
    ).read_bytes()
    undated = write_pdf(tmp_path / "c.pdf", [["Body"]]).read_bytes()
    pdf = source_file("a.pdf", "pdf")
    docx = write_docx(
        tmp_path / "d.docx",
        ["Body"],
        {"created": "2024-03-04T09:40:00Z", "modified": "2024-03-05T11:57:00+10:00"},
    ).read_bytes()
    created_only = write_docx(
        tmp_path / "e.docx", ["Body"], {"created": "2024-03-04T09:40:00Z"}
    ).read_bytes()
    word, deck = source_file("d.docx", "docx"), source_file("d.pptx", "pptx")

    assert metadata_date(pdf, created) == "2025-06-30"
    assert metadata_date(pdf, modified_only) == "2025-07-01"
    assert metadata_date(pdf, undated) is None
    assert metadata_date(word, docx) == metadata_date(deck, docx) == "2024-03-05"
    assert metadata_date(word, created_only) == "2024-03-04"
    assert metadata_date(word, write_docx(tmp_path / "f.docx", ["Body"]).read_bytes()) is None
    assert metadata_date(source_file("m.md", "md"), b"# Memo") is None
    with pytest.raises(ConversionError, match=r"d\.docx: cannot read docProps/core\.xml"):
        metadata_date(word, b"not a zip archive")

    metadata = DocumentDate(iso="2025-06-30", basis="metadata")
    assert document_date(None, "2025-06-30") == metadata
    assert document_date("  ", "2025-06-30") == metadata
    assert document_date("21st October 2025", "2025-06-30") == DocumentDate(
        iso="2025-10-21", basis="printed"
    )
    # An ambiguous printed date leaves the document undated rather than taking the file's date.
    assert document_date("05/04/2026", "2025-06-30") is None


def test_every_acoustic_example_file_carries_a_metadata_date() -> None:
    corpus = (
        Path(__file__).resolve().parents[1] / "evals/example_corpora/acoustic-consultancy-files"
    )

    dates = {s.rel_path: metadata_date(s, read_source(s)) for s in discover_corpus(corpus)}

    assert dates == {
        "corvane-depot-night-noise-report.pdf": "2025-06-30",
        "corvane-depot-quotation.docx": "2025-04-14",
        "corvane-depot-response-to-planning-comments.docx": "2025-08-25",
        "fernmoss-school-heat-pump-report.pdf": "2024-10-07",
        "fernmoss-school-quotation.pdf": "2024-08-12",
        "hartwell-tap-room-noise-report.docx": "2024-05-20",
        "hartwell-tap-room-quotation.docx": "2024-03-04",
        "pellow-mill-facade-report.docx": "2026-01-19",
        "pellow-mill-quotation.pdf": "2025-11-03",
    }


# Rendering and the render cache.


def test_a_render_cache_hit_runs_neither_libreoffice_nor_pdfium(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "p1").mkdir(parents=True)
    write_docx(corpus_dir / "p1" / "memo.docx", ["Memo"])
    converted = write_pdf(
        tmp_path / "converted.pdf", [["# Memo", "Value 330.9"], ["## Next", "Area 1204"]]
    ).read_bytes()
    conversions: list[object] = []

    def convert(*args: object) -> bytes:
        conversions.append(args)
        return converted

    monkeypatch.setattr(compiler, "convert_office_to_pdf", convert)
    cache_path = tmp_path / "cache.db"
    with BuildCache(cache_path) as cache:
        first = compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T", render_cache=cache)

    def refuse(*args: object) -> bytes:
        raise AssertionError("rendered again")

    monkeypatch.setattr(compiler, "convert_office_to_pdf", refuse)
    monkeypatch.setattr(compiler, "render_pdf", refuse)
    with BuildCache(cache_path) as cache:
        second = compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T", render_cache=cache)

    assert len(conversions) == 1
    (doc,) = second.documents
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert doc.pages == first.documents[0].pages and doc.sections == first.documents[0].sections
    output = tmp_path / "t.arkive.db"
    write_snapshot(second, output)
    with TopicSnapshot.open(output) as snapshot:
        page = snapshot.view_page(doc.doc_id, 2)
    assert (page.webp_bytes, page.native_text) == (
        doc.pages[1].webp_bytes,
        doc.pages[1].native_text,
    )
    # SQLite folds the write-ahead log back into the file when the last connection closes.
    wal = cache_path.with_name("cache.db-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    with pytest.raises(AssertionError, match="rendered again"):
        compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T")


def test_an_oversized_page_renders_within_the_side_limit() -> None:
    document = pdfium.PdfDocument.new()
    document.new_page(12_000, 600)
    buffer = io.BytesIO()
    document.save(buffer)
    document.close()

    ((_, webp),) = render_pdf(buffer.getvalue())

    width, height = Image.open(io.BytesIO(webp)).size
    assert width == MAX_RENDER_SIDE_PX and abs(height - 600 * MAX_RENDER_SIDE_PX / 12_000) <= 1
    wide = encode_page_image(Image.new("RGB", (MAX_RENDER_SIDE_PX + 904, 50), "white"))
    assert Image.open(io.BytesIO(wide)).size == (MAX_RENDER_SIDE_PX, 41)


def test_a_page_pillow_cannot_encode_is_a_conversion_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("encoder error -2")

    monkeypatch.setattr(Image.Image, "save", refuse)
    with pytest.raises(ConversionError, match="cannot encode a 10x10 page: encoder error"):
        encode_page_image(Image.new("RGB", (10, 10)))


def test_a_filename_wrapped_over_six_lines_of_a_narrow_cell_is_one_anchor() -> None:
    native = "| Layer | File |\n| Depth | harbour_\ndry_\nseason_\ndepth_\nthresh_\nfinal.tif |"

    filenames, _ = fidelity_anchors(native)

    (anchor,) = filenames
    assert anchor.name == "final.tif"
    assert "harbour_dry_season_depth_thresh_final.tif" in anchor.accepted
    whole = page(block("paragraph", "Depth raster harbour_dry_season_depth_thresh_final.tif"))
    assert missing_anchors(1, whole, filenames, set()) == []


def test_an_oversized_section_is_split_with_every_table_piece_under_its_header() -> None:
    header = "| Station | Depth | Notes |\n| --- | --- | --- |"
    rows = [f"| S{i:05d} | {i * 3} m | sounding {'x' * 60} |" for i in range(900)]
    prose = " ".join(f"Sentence {i} about the survey." for i in range(1500))

    sections = assemble_sections(
        [
            (
                1,
                page(
                    block("heading", "Soundings", level=1),
                    block("table", "\n".join([header, *rows])),
                ),
            ),
            (2, page(block("paragraph", prose))),
        ]
    )

    assert len(sections) >= 3
    assert all(len(s.markdown) <= MAX_SECTION_CHARS for s in sections)
    assert {(s.heading_path, s.level) for s in sections} == {("Soundings", 1)}
    assert sections[0].markdown.startswith("# Soundings\n\n" + header)
    tables = [p.markdown for s in sections for p in s.parts if p.kind == "table"]
    assert len(tables) >= 2 and all(t.startswith(header + "\n") for t in tables)
    assert [line for t in tables for line in t.splitlines()[2:]] == rows
    paragraphs = [p.markdown for s in sections for p in s.parts if p.kind == "paragraph"]
    assert " ".join(paragraphs) == prose and sections[-1].page_end == 2


def test_rendered_documents_are_read_largest_first_and_kept_in_path_order(tmp_path: Path) -> None:
    write_files(tmp_path, {"m/notes.md": "# Notes\n\nShort."})
    (tmp_path / "a").mkdir()
    (tmp_path / "z").mkdir()
    write_pdf(tmp_path / "a" / "small.pdf", [["Small page"]])
    write_pdf(tmp_path / "z" / "large.pdf", [[f"Large page {n} " + "x" * 400] for n in range(3)])
    gateway = FakeGateway()

    corpus = compile_corpus(tmp_path, gateway, topic="t", title="T", max_concurrency=1)

    transcribed = [r.native_text.split()[0] for r in gateway.of_type(PageTranscription)]
    assert transcribed == ["Large", "Large", "Large", "Small"]
    assert [d.rel_path for d in corpus.documents] == ["a/small.pdf", "m/notes.md", "z/large.pdf"]


def test_a_failed_read_cancels_queued_reads_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_files(tmp_path, {"a.md": "# A\n\nShort."})
    for size, name in enumerate(("b", "c", "d", "e"), start=1):
        write_pdf(tmp_path / f"{name}.pdf", [[f"Page {name} " + "x" * (100 * size)]])
    reads: list[str] = []

    def read(source: SourceFile, data: bytes, gateway: object, **kwargs: object) -> object:
        reads.append(source.rel_path)
        if len(reads) == 1:
            raise ConversionError(f"{source.rel_path}: damaged")
        time.sleep(0.2)
        raise AssertionError(f"{source.rel_path} read after a failure")

    monkeypatch.setattr(compiler, "read_document", read)
    with pytest.raises(ConversionError, match=r"e\.pdf: damaged"):
        compile_corpus(tmp_path, FakeGateway(), topic="t", title="T", max_concurrency=1)
    # The largest file is read first, and the worker that ran it starts no other read: the
    # failure stops it before the cancellation of the queued reads reaches the pool.
    assert reads == ["e.pdf"]


# Section splitting: headings, fences, visuals and tables.


def test_a_headed_section_too_large_for_one_piece_keeps_its_body_beside_the_heading() -> None:
    body = " ".join(f"sounding{i}" for i in range(5000))
    assert len(body) > MAX_SECTION_CHARS

    drafts = assemble_sections(
        [(1, page(block("heading", "Soundings", level=1), block("paragraph", body)))]
    )

    assert len(drafts) == 2
    assert all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    assert drafts[0].markdown.startswith("# Soundings\n\nsounding0 ")
    assert all(section_pieces(d, "prose") for d in drafts)
    assert " ".join(p.markdown for d in drafts for p in d.parts if p.kind == "paragraph") == body


def test_a_split_fenced_block_stays_balanced_and_the_prose_after_it_stays_prose() -> None:
    code = "\n".join(f"value_{i} = {i}  # sample line" for i in range(2000))
    after = "The survey team reviewed every sounding against the tide gauge record."
    text = f"# Run\n\n```python\n{code}\n```\n\n{after}"

    drafts = assemble_sections([(1, parse_markdown_blocks(text))])

    assert len(drafts) >= 2 and all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    for draft in drafts:
        fences = [line for line in draft.markdown.splitlines() if line.startswith("```")]
        assert len(fences) % 2 == 0 and all(f in ("```python", "```") for f in fences)
    assert standing_units(drafts[-1].markdown)[-1].text == after
    assert not any(after in u.text for d in drafts[:-1] for u in standing_units(d.markdown))
    lines = [line for d in drafts for line in d.markdown.splitlines() if line.startswith("value_")]
    assert lines == code.splitlines()


def test_a_fence_delimiter_too_long_to_repeat_still_splits_within_the_ceiling() -> None:
    code = "\n".join(f"value_{i} = {i}" for i in range(4000))
    # 300 backticks leave no room to reopen beside the heading; 20,000 leave none in a section.
    for length in (300, 20_000):
        fence = "`" * length
        heading = block("heading", "h" * (MAX_SECTION_CHARS - 400), level=1)
        body = block("paragraph", f"{fence}\n{code}\n{fence}")

        drafts = assemble_sections([(1, page(heading, body))])

        assert len(drafts) >= 2 and all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
        lines = [line for d in drafts for line in d.markdown.splitlines()]
        assert [line for line in lines if line.startswith("value_")] == code.splitlines()


def test_an_oversized_visual_is_split_into_transcripts_each_under_its_title() -> None:
    legend = VisualTranscript(
        label="Map", transcript="\n".join(f"legend entry {i}" for i in range(4000))
    )

    drafts = assemble_sections([(3, page(block("heading", "Maps", level=1), visuals=(legend,)))])

    visuals = [p.markdown for d in drafts for p in d.parts if p.kind == "visual"]
    assert len(visuals) >= 2 and all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    for text in visuals:
        assert text.startswith("> **[Visual \u2014 Page 3: Map]**\n> legend entry")
        assert all(line.startswith(">") for line in text.splitlines())
        assert without_visual_transcripts(text) == "" and standing_units(text) == []
    entries = [line for text in visuals for line in text.splitlines()[1:]]
    assert entries == [f"> legend entry {i}" for i in range(4000)]


def test_a_table_continued_over_twenty_pages_splits_with_each_slice_on_its_own_pages() -> None:
    header = "| Station | Depth | Notes |\n| --- | --- | --- |"
    pages: list[tuple[int, PageExtraction]] = []
    for number in range(1, 21):
        rows = "\n".join(f"| S{number:02d}-{i:02d} | {i} m | {'x' * 60} |" for i in range(80))
        chart = VisualTranscript(label=f"Profile {number}", transcript=f"Profile of page {number}")
        table = block("table", f"{header}\n{rows}" if number == 1 else rows, continues=number > 1)
        heading = [block("heading", "Soundings", level=1)] if number == 1 else []
        pages.append((number, page(*heading, table, visuals=(chart,))))

    drafts = assemble_sections(pages)

    assert len(drafts) >= 4 and all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    assert drafts[0].page_start == 1 and drafts[-1].page_end == 20
    seen_charts = []
    for draft in drafts:
        assert draft.page_end - draft.page_start <= 7
        for part in draft.parts:
            if part.kind == "table":
                stations = {int(row[3:5]) for row in part.markdown.splitlines()[2:]}
                assert (min(stations), max(stations)) == (part.page_start, part.page_end)
            if part.kind == "visual":
                # Each page's chart stays with the slice that holds that page's rows.
                assert draft.page_start <= part.page_start <= draft.page_end
                seen_charts.append(part.page_start)
    assert seen_charts == list(range(1, 21))
    chunks = [c for d in drafts for c in section_pieces(d, "prose") if c.kind == "table"]
    assert max(c.page_end - c.page_start for c in chunks) <= 1


def test_a_table_row_too_long_for_a_slice_wraps_into_rows_of_its_columns() -> None:
    header = "| Site | Notes |\n| --- | --- |"
    long_row = "| S1 | " + " ".join(["note"] * 9000) + " |"
    table = f"{header}\n{long_row}\n| S2 | short |"

    drafts = assemble_sections([(1, page(block("table", table)))])

    tables = [p.markdown for d in drafts for p in d.parts if p.kind == "table"]
    assert len(tables) >= 2 and all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    for text in tables:
        assert text.startswith(header + "\n")
        assert all(row.startswith("| ") and row.count(" | ") == 1 for row in text.splitlines()[2:])
    assert sum(text.count("note") for text in tables) == 9000
    # A table with a header and no rows is kept, never dropped.
    (only_header,) = assemble_sections([(1, page(block("table", header)))])
    assert only_header.markdown == header


def test_wrap_text_refuses_a_limit_below_one_and_always_advances() -> None:
    for limit, first_limit in ((0, None), (4, 0), (4, -5)):
        with pytest.raises(ValueError, match="wrap limits must be positive"):
            wrap_text("abc def ghi", limit, first_limit=first_limit)
    pieces = wrap_text("abc def\nghijklmnop", 3, first_limit=1)
    assert pieces == ["a", "bc", "def", "ghi", "jkl", "mno", "p"]


def test_a_line_opening_with_an_inline_code_span_is_prose_not_a_fence() -> None:
    lead = "```arkive-rag``` is the command line entry point."
    prose = " ".join([lead, *(f"word{i}" for i in range(9000))])

    extraction = parse_markdown_blocks(f"{lead}\n\n# Usage\n\nRun it.")
    drafts = assemble_sections([(1, page(block("paragraph", prose)))])

    assert [b.kind for b in extraction.blocks] == ["paragraph", "heading", "paragraph"]
    assert len(drafts) >= 2 and all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    # No fence is closed or reopened around text that never opened one.
    assert sum(d.markdown.count("```") for d in drafts) == 2
    assert " ".join(d.markdown for d in drafts) == prose


def test_a_cut_code_block_reopens_with_its_delimiter_and_a_short_info_word() -> None:
    code = "\n".join(f"value_{i} = {i}" for i in range(4000))
    long_info = "```" + "x" * 16_000
    drafts = assemble_sections(
        [
            (
                1,
                page(
                    block("paragraph", f"```json extra words\n{code}\n```"),
                    block("paragraph", f"{long_info}\n{code}\n```"),
                ),
            )
        ]
    )

    assert all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    fences = [line for d in drafts for line in d.markdown.splitlines() if line.startswith("```")]
    assert set(fences) == {"```json extra words", "```json", "```", long_info}
    assert fences.count(long_info) == 1


def test_a_heading_with_little_or_no_room_beside_it_keeps_every_piece_bounded() -> None:
    body = " ".join(f"sounding{i}" for i in range(4000))
    for heading in ("h" * (MAX_SECTION_CHARS - 100), "h " * MAX_SECTION_CHARS):
        drafts = assemble_sections(
            [(1, page(block("heading", heading, level=1), block("paragraph", body)))]
        )

        assert all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
        assert drafts[0].markdown.startswith("# h")
        paragraphs = [p.markdown for d in drafts for p in d.parts if p.kind == "paragraph"]
        assert " ".join(paragraphs) == body


def test_an_oversized_visual_after_a_split_table_is_split_too() -> None:
    header = "| Station | Notes |\n| --- | --- |"
    rows = "\n".join(f"| S{i:04d} | {'x' * 60} |" for i in range(1200))
    legend = VisualTranscript(label="Map", transcript="\n".join(f"entry {i}" for i in range(9000)))

    drafts = assemble_sections([(1, page(block("table", f"{header}\n{rows}"), visuals=(legend,)))])

    assert all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    visuals = [p.markdown for d in drafts for p in d.parts if p.kind == "visual"]
    assert len(visuals) >= 2
    entries = [line for text in visuals for line in text.splitlines()[1:]]
    assert entries == [f"> entry {i}" for i in range(9000)]


def test_a_visual_title_too_long_to_repeat_is_quoted_once_under_a_short_title() -> None:
    label = "L" * 20_000
    legend = VisualTranscript(label=label, transcript="\n".join(f"entry {i}" for i in range(6000)))

    drafts = assemble_sections([(4, page(visuals=(legend,)))])

    visuals = [p.markdown for d in drafts for p in d.parts if p.kind == "visual"]
    assert len(visuals) >= 2 and all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    assert all(text.startswith("> **[Visual \u2014 Page 4]**\n> ") for text in visuals)
    assert "".join(visuals).count(label) == 1


def test_table_slices_never_pass_the_section_ceiling() -> None:
    rows = "\n".join(f"| S{i:04d} | {'x' * 60} |" for i in range(800))
    # A header wider than a section: its lines are cut like rows instead of repeated.
    wide = "| " + " | ".join(f"column {i}" for i in range(4000)) + " |"
    wide_separator = "| " + " | ".join(["---"] * 4000) + " |"
    # A header that fits a section but leaves too little room beside a heading for a row.
    narrow_header = "| Station | " + "n" * (MAX_SECTION_CHARS - 400) + " |\n| --- | --- |"
    cases = [
        ([block("table", f"{wide}\n{wide_separator}")], 0),
        ([block("table", f"{wide}\n{wide_separator}\n{rows}")], 800),
        ([block("heading", "h" * (MAX_SECTION_CHARS - 600), level=1), block("table", rows)], 800),
        ([block("heading", "Soundings", level=1), block("table", f"{narrow_header}\n{rows}")], 800),
    ]
    for blocks, station_rows in cases:
        drafts = assemble_sections([(1, page(*blocks))])

        assert all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
        tables = [p.markdown for d in drafts for p in d.parts if p.kind == "table"]
        assert sum(t.count("| S0") for t in tables) == station_rows


def test_a_wrapped_cell_keeps_every_escaped_pipe_and_its_row_inside_the_slice() -> None:
    # Two full-width columns whose first cut falls right after a backslash, then escaped pipes
    # after backslash runs of both parities, with no space to cut at.
    runs = "".join(f"v{i}" + "\\" * (1 + i % 2) + "|" for i in range(3000))
    cell = "x" * 15_980 + "\\|" + runs
    table = f"| Notes | More |\n| --- | --- |\n| {cell} | {cell} |"

    drafts = assemble_sections([(1, page(block("table", table)))])

    tables = [p.markdown for d in drafts for p in d.parts if p.kind == "table"]
    assert len(tables) >= 2 and all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    rows = [compiler._cells(row) for text in tables for row in text.splitlines()[2:]]
    # No piece starts with a pipe its backslash was cut from, so every row keeps two columns.
    assert all(len(cells) == 2 for cells in rows)
    assert ["".join(cells[i] for cells in rows) for i in (0, 1)] == [cell, cell]


def test_a_header_less_table_slice_that_starts_like_a_header_keeps_its_row_pages() -> None:
    # Every row after the first two looks like a separator row, so a slice that starts with a
    # data row reads that row as its header.
    rows = ["| a0 | b0 |", "| a1 | b1 |"] + [f"| {'-' * 60} | --- |" for _ in range(1200)]
    pages = [
        (number, page(block("table", "\n".join(rows[i : i + 100]), continues=number > 1)))
        for number, i in enumerate(range(0, len(rows), 100), start=1)
    ]

    drafts = assemble_sections(pages)
    chunks = [c for d in drafts for c in section_pieces(d, "prose") if c.kind == "table"]

    assert all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    assert chunks and all(1 <= c.page_start <= c.page_end <= len(pages) for c in chunks)


def test_a_repeated_header_takes_the_pages_of_its_slice_rows() -> None:
    # A header too wide to repeat in a chunk is cut like rows there, so a chunk of a slice
    # whose rows are all on page 2 would start on the header's page if the header kept it.
    header = "| Station | " + "n" * 1700 + " |\n| --- | --- |"
    rows = [f"| S{i:04d} | {'x' * 60} |" for i in range(900)]
    pages = [
        (1, page(block("table", "\n".join([header, *rows[:10]])))),
        (2, page(block("table", "\n".join(rows[10:]), continues=True))),
    ]

    drafts = assemble_sections(pages)

    later = [d for d in drafts if d.page_start == 2]
    assert later and all(len(d.markdown) <= MAX_SECTION_CHARS for d in drafts)
    chunks = [c for d in later for c in section_pieces(d, "prose")]
    assert chunks and all(c.page_start == c.page_end == 2 for c in chunks)


def test_a_cell_too_narrow_for_an_escaped_pipe_is_cut_as_text() -> None:
    line = "| a\\|b | c |"
    # Rows with one-character cells would cut `\|` from its backslash into a third column.
    pieces = compiler._wrap_row(line, len("|  |  |") + 3)

    assert all(len(compiler._cells(piece)) <= 2 for piece in pieces)
    assert any("a\\|b" in piece for piece in pieces)


def test_wrapping_a_long_cell_into_narrow_columns_keeps_every_character() -> None:
    cell = " ".join(f"w{i}" for i in range(6000))
    line = "| " + " | ".join([cell, *(["v"] * 999)]) + " |"

    rows = compiler._wrap_row(line, MAX_SECTION_CHARS)

    assert len(rows) > 1000 and all(len(row) <= MAX_SECTION_CHARS for row in rows)
    assert " ".join(compiler._cells(row)[0] for row in rows).split() == cell.split()


# Furniture and filename anchors.


def test_two_page_furniture_needs_body_pages_matching_edges_and_true_page_numbers() -> None:
    # Two short slides repeat a value line: it is data, not furniture.
    slides = ["Seagrass extent\nArea 1204 ha", "Seagrass change\nArea 1204 ha"]
    assert find_furniture(slides) == frozenset()
    body = ["Body line one", "Body line two", "Body line three"]

    def pair(tops: tuple[str, str], bottoms: tuple[str, str]) -> list[str]:
        return [
            "\n".join([top, f"Section {name}", *body, f"Notes {name}", bottom])
            for top, bottom, name in zip(tops, bottoms, ("alpha", "beta"), strict=True)
        ]

    pages = pair(("Harbour Memo", "Harbour Memo"), ("Report page 1", "Report page 2"))
    assert strip_furniture(pages[1], find_furniture(pages), 1) == "\n".join(
        ["Section beta", *body, "Notes beta"]
    )
    # A number that is not the page number is data, even when it steps with the page.
    assert find_furniture(pair(("Title A", "Title B"), ("Depth 450", "Depth 451"))) == frozenset()
    # A line at the top of one page and at the bottom of the other is not furniture.
    swapped = pair(("Harbour Memo", "Start B"), ("End A", "Harbour Memo"))
    assert find_furniture(swapped) == frozenset()
    # A line naming a data file is never furniture, nor is a fragment of a filename wrapped
    # over lines, such as `north_` above `2024_grid.tif`.
    assert find_furniture(pair(("grid.tif", "grid.tif"), ("End A", "End B"))) == frozenset()
    wrapped = [f"north_\n2024_grid.tif\n{text}" for text in pair(("A", "B"), ("End", "End"))]
    assert "top\x00north_" not in find_furniture(wrapped)
    assert strip_furniture(wrapped[1], find_furniture(wrapped), 1).startswith("north_\n2024_")


def test_furniture_is_stripped_from_page_edges_only() -> None:
    pages = [
        f"CONFIDENTIAL\nSection {n}\nBody {n}\nCONFIDENTIAL\nMore {n}\nEnd {n}\nCONFIDENTIAL"
        for n in ("alpha", "beta", "gamma")
    ]

    stripped = strip_furniture(pages[1], find_furniture(pages), 1)

    assert stripped == "Section beta\nBody beta\nCONFIDENTIAL\nMore beta\nEnd beta"


def test_two_wrapped_filenames_with_one_tail_are_two_anchors_and_hold_their_numbers() -> None:
    native = (
        "North north_\n2024_\ngrid.tif\nSouth south_\n2024_\ngrid.tif\n"
        "Dry harbour_\ndry_\n2026\nfinal.tif"
    )

    filenames, numbers = fidelity_anchors(native)

    assert sorted(a.name for a in filenames) == ["final.tif", "grid.tif", "grid.tif"]
    assert numbers == set()
    north_only = page(block("paragraph", "north_2024_grid.tif and harbour_dry_2026final.tif"))
    # A missing wrapped filename is named whole, as a retry asks for it.
    assert missing_anchors(1, north_only, filenames, numbers) == ["south_2024_grid.tif"]
    both = page(
        block("paragraph", "north_2024_grid.tif, south_2024_grid.tif, harbour_dry_2026final.tif")
    )
    assert missing_anchors(1, both, filenames, numbers) == []
    # The same number standing alone elsewhere on the page is still an anchor.
    assert fidelity_anchors(native + "\nYear 2026")[1] == {"2026"}
    # A value on the line before a filename, with no sign of a wrapped name, stays a value.
    assert fidelity_anchors("Area 1204\ngrid.tif")[1] == {"1204"}


def test_a_wrapped_filename_whose_tail_starts_with_a_hyphen_holds_its_number() -> None:
    filenames, numbers = fidelity_anchors("Dry dry_\n2026\n-final.tif")

    (anchor,) = filenames
    assert (anchor.display, numbers) == ("dry_2026-final.tif", set())
    whole = page(block("paragraph", "Dry dry_2026-final.tif"))
    assert missing_anchors(1, whole, filenames, numbers) == []
    assert missing_anchors(1, page(block("paragraph", "Dry")), filenames, numbers) == [
        "dry_2026-final.tif"
    ]


def test_one_emitted_filename_satisfies_only_one_anchor() -> None:
    filenames, numbers = fidelity_anchors("Compare grid.tif\nSouth south_\n2024_\ngrid.tif")

    assert len(filenames) == 2
    # Both anchors accept `grid.tif`, but one emitted copy cannot stand in for both.
    once = page(block("paragraph", "Compare grid.tif\nSouth"))
    assert missing_anchors(1, once, filenames, numbers) == ["south_2024_grid.tif"]
    both = page(block("paragraph", "Compare grid.tif\nSouth south_2024_grid.tif"))
    assert missing_anchors(1, both, filenames, numbers) == []
    # The matching is maximal whatever the order: the copy the wrapped anchor needs is not
    # taken by the anchor that also accepts the bare name.
    reordered = page(block("paragraph", "South south_2024_grid.tif\nCompare grid.tif"))
    assert missing_anchors(1, reordered, filenames, numbers) == []


def test_matching_thousands_of_filenames_with_one_tail_keeps_the_call_stack_flat() -> None:
    # Every anchor accepts every emitted `grid.tif`, so each new anchor's augmenting path runs
    # through all the anchors matched before it.
    native = "\n".join(f"n{i}_\ngrid.tif" for i in range(2000))
    filenames, numbers = fidelity_anchors(native)
    assert len(filenames) == 2000

    emitted = page(block("paragraph", " ".join(["grid.tif"] * 2000)))
    assert missing_anchors(1, emitted, filenames, numbers) == []
    short = page(block("paragraph", " ".join(["grid.tif"] * 1999)))
    assert len(missing_anchors(1, short, filenames, numbers)) == 1


# Dates and metadata.


def test_metadata_dates_read_only_whole_well_formed_values(tmp_path: Path) -> None:
    pdf, word = source_file("a.pdf", "pdf"), source_file("d.docx", "docx")

    def pdf_date(value: str) -> str | None:
        data = write_pdf(tmp_path / "x.pdf", [["B"]], {"CreationDate": value}).read_bytes()
        return metadata_date(pdf, data)

    def word_date(modified: str, created: str | None = None) -> str | None:
        dates = {"modified": modified} | ({"created": created} if created else {})
        return metadata_date(word, write_docx(tmp_path / "x.docx", ["B"], dates).read_bytes())

    assert pdf_date("D:20250630164500-10'00'") == "2025-06-30"
    assert pdf_date("D:20250630164500-10:00") == "2025-06-30"
    assert pdf_date("20250630") == "2025-06-30"
    assert pdf_date("D:202506") == "2025-06"
    assert pdf_date("D:20250630 garbage") is None
    assert pdf_date("D:20250631") is None
    # An offset follows an hour only, and every clock field must be in range; a value that
    # fails falls through to ModDate.
    for created in ("202403-2025", "20240301-15", "D:20240301250000Z", "D:2024030112+24'00'"):
        data = write_pdf(
            tmp_path / "x.pdf", [["B"]], {"CreationDate": created, "ModDate": "D:20240501"}
        ).read_bytes()
        assert metadata_date(pdf, data) == "2024-05-01", created
    assert word_date("2024-03") == "2024-03"
    assert word_date("2024-03-05T11:57:00.25+10:00") == "2024-03-05"
    # A range or trailing text is not a date; the next field is read instead.
    assert word_date("2024-03-04/2024-05-01", "2024-02-01T00:00:00Z") == "2024-02-01"
    assert word_date("2024-03-04 approx") is None
    assert word_date("2024-03-04T99:99:99Z", "2024-02-01") == "2024-02-01"
    assert word_date("2024-03-04T10:00:00+24:00", "2024-02-01") == "2024-02-01"


def write_zip(path: Path, entries: dict[str, bytes]) -> bytes:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path.read_bytes()


def undecodable_member_name(path: Path) -> bytes:
    """A zip archive whose second member name is flagged UTF-8 but is not UTF-8."""
    data = write_zip(path, {"docProps/core.xml": b"<x/>", "\u00e9.xml": b"x"})
    return data.replace("\u00e9".encode(), b"\xff\xfe")


def test_unreadable_office_metadata_stops_the_build_before_any_model_or_render_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = b'<cp:coreProperties xmlns:cp="x"/>'
    valid = write_zip(tmp_path / "v.docx", {"docProps/core.xml": core * 50})
    # Flip the deflated bytes of core.xml, which follow its name in the local file header.
    start = valid.index(b"docProps/core.xml") + len("docProps/core.xml")
    flipped = bytes(b ^ 0xFF for b in valid[start : start + 12])
    damaged = valid[:start] + flipped + valid[start + 12 :]
    oversized = core + b" " * MAX_CORE_PROPERTIES_BYTES
    cases = {
        "big.docx": (write_zip(tmp_path / "b.docx", {"docProps/core.xml": oversized}), "limit is"),
        "bad.docx": (write_zip(tmp_path / "x.docx", {"docProps/core.xml": b"<no"}), "cannot read"),
        "crc.docx": (damaged, "cannot read"),
        "zip.pptx": (b"PK\x03\x04 truncated", "cannot read"),
        "name.docx": (undecodable_member_name(tmp_path / "n.docx"), "cannot read"),
    }

    def refuse(*args: object) -> bytes:
        raise AssertionError("rendered")

    monkeypatch.setattr(compiler, "convert_office_to_pdf", refuse)
    for name, (data, message) in cases.items():
        corpus_dir = tmp_path / name.replace(".", "-")
        corpus_dir.mkdir()
        (corpus_dir / name).write_bytes(data)
        gateway = FakeGateway()

        with pytest.raises(ConversionError, match=rf"{re.escape(name)}: .*{message}"):
            compile_corpus(corpus_dir, gateway, topic="t", title="T")
        assert gateway.calls == [], name


def test_every_pdfium_failure_is_a_conversion_error_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = write_pdf(tmp_path / "a.pdf", [["Body 330.9"]]).read_bytes()
    pdf = source_file("p1/a.pdf", "pdf")

    def broken(*args: object, **kwargs: object) -> object:
        raise pdfium.PdfiumError("damaged")

    with monkeypatch.context() as patch:
        patch.setattr(pdfium.PdfDocument, "get_metadata_value", broken)
        with pytest.raises(ConversionError, match=r"p1/a\.pdf: cannot read PDF metadata: damaged"):
            metadata_date(pdf, data)
    with monkeypatch.context() as patch:
        patch.setattr(compiler, "_render_page", broken)
        with pytest.raises(ConversionError, match=r"p1/a\.pdf: cannot render page 1: damaged"):
            render_source(pdf, data, "0" * 64, None)
    with monkeypatch.context() as patch:
        patch.setattr(pdfium.PdfDocument, "__len__", broken)
        with pytest.raises(ConversionError, match=r"p1/a\.pdf: cannot count pages: damaged"):
            render_source(pdf, data, "0" * 64, None)
    # A failure to close is a ConversionError naming the file, and never replaces an earlier
    # failure.
    with monkeypatch.context() as patch:
        patch.setattr(pdfium.PdfDocument, "close", broken)
        with pytest.raises(ConversionError, match=r"p1/a\.pdf: cannot close PDF: damaged"):
            metadata_date(pdf, data)
        with pytest.raises(ConversionError, match=r"p1/a\.pdf: cannot close PDF: damaged"):
            render_source(pdf, data, "0" * 64, None)
        patch.setattr(compiler, "_render_page", broken)
        with pytest.raises(ConversionError, match=r"^cannot render page 1: damaged"):
            render_pdf(data)


def test_a_warm_render_cache_answers_a_pdf_and_its_metadata_date_without_pdfium(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    write_pdf(corpus_dir / "memo.pdf", [["# Memo", "Value 330.9"]], {"CreationDate": "D:20250630"})
    cache_path = tmp_path / "cache.db"
    with BuildCache(cache_path) as cache:
        first = compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T", render_cache=cache)

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("pdfium opened")

    monkeypatch.setattr(pdfium, "PdfDocument", refuse)
    with BuildCache(cache_path) as cache:
        second = compile_corpus(corpus_dir, FakeGateway(), topic="t", title="T", render_cache=cache)

    metadata = DocumentDate(iso="2025-06-30", basis="metadata")
    assert first.documents[0].date == second.documents[0].date == metadata
    assert second.documents[0].pages == first.documents[0].pages


def test_placeholders_and_consolidation_never_invent_or_drop_a_printed_date() -> None:
    answers = (None, " ", "None", "N/A", "not stated.", " 5/4/26 ")
    assert [printed_date_or_none(a) for a in answers] == [None] * 5 + ["5/4/26"]

    def header(doc_date: str | None) -> CardHeader:
        return CardHeader(title="T", doc_date=doc_date, deliverable_type="memo", summary="S")

    ambiguous = (header(None), header("05/04/2026"))
    # Consolidation cannot turn an ambiguous printed date into an ISO date, or drop it.
    assert consolidated_date(ambiguous, "2026-04-05") == "05/04/2026"
    assert consolidated_date(ambiguous, None) == "05/04/2026"
    assert consolidated_date((header("May 2025"), header("2025-05-02")), "2025-05-02") == (
        "2025-05-02"
    )
    # With no printed date in any part, an answer is not a date, and a placeholder part is
    # not a printed date.
    assert consolidated_date((header(None), header(None)), "2025-01-01") is None
    assert consolidated_date((header("none"), header("N/A")), "2025-01-01") is None
    # A reformatted copy of a reported date keeps the date as that part printed it.
    printed = (header("April 2025"), header("19\u1d57\u02b0 May 2025"))
    for answer in ("19th May 2025", "2025-05-19", "19\u1d57\u02b0 may 2025"):
        assert consolidated_date(printed, answer) == "19\u1d57\u02b0 May 2025", answer
    # An answer that copies a later part exactly keeps that part's form, even when an earlier
    # part printed the same date another way.
    forms = (header("2025-05-19"), header("19 May 2025"), header("19 MAY 2025"))
    assert consolidated_date(forms, "19 May 2025") == "19 May 2025"
    assert consolidated_date(forms, "19 may 2025") == "19 May 2025"

    draft = ObservationDraft(
        title="Memo",
        doc_date="none",
        deliverable_type="memo",
        summary="S.",
        attributes=[],
        terms=[],
        craft=[],
    )
    section = ReadSection(
        ordinal=1, heading_path="Memo", level=1, page_start=1, page_end=1, markdown="B.", pieces=()
    )
    card, _ = observe_document(FakeGateway(observe_script=[draft]), "memo.pdf", (section,))
    assert card.doc_date is None


def test_a_metadata_date_reaches_organisation_binding_and_the_guide(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "p1").mkdir(parents=True)
    write_pdf(
        corpus_dir / "p1" / "memo.pdf",
        [["# Memo", "Survey notes."]],
        {"CreationDate": "D:20250630"},
    )
    gateway = FakeGateway()

    compile_corpus(corpus_dir, gateway, topic="t", title="T")

    label = "2025-06-30 (file metadata)"
    (organise,) = gateway.of_type(OrganiseCorpus)
    (guide,) = gateway.of_type(WriteGuide)
    assert label in organise.text() and label in guide.text()
    drafts = assemble_sections([(1, page(block("paragraph", "Body.")))])
    embedded = DocumentDate(iso="2025-06-30", basis="metadata")
    doc = CorpusDocument(document_id("m.pdf"), "m.pdf", reading_of(drafts, "M", date=embedded))
    assert "DATE: 2025-06-30 (file metadata)" in binding_evidence(doc)
    printed = reading_of(drafts, "M", printed_date="05/04/2026")
    assert "DATE: 05/04/2026" in binding_evidence(CorpusDocument(doc.doc_id, "m.pdf", printed))
