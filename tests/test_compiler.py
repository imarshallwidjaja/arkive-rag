from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import openai
import pytest

from arkive_rag.compiler import (
    MAX_CHUNK_CHARS,
    ChunkContext,
    SectionDraft,
    assemble_sections,
    build_chunks,
    compile_corpus,
    discover_corpus,
    fidelity_anchors,
    find_furniture,
    invalid_citations,
    missing_anchors,
    parse_markdown_blocks,
    render_pdf,
    split_table,
    split_table_block,
    strip_furniture,
    synthesize_with_citations,
    transcribe_with_fidelity,
)
from arkive_rag.llm import BuildCache, OpenAIGateway
from arkive_rag.models import (
    CompiledDocument,
    ConversionError,
    DocumentMetaExtraction,
    InvalidCitationError,
    PageBlock,
    PageExtraction,
    Section,
    TranscriptionFidelityError,
    UnsafePathError,
    UnsupportedFormatError,
    VisualTranscript,
)
from tests.support import (
    FakeGateway,
    FakeOpenAIClient,
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


# Discovery.


def test_discovery_derives_project_from_top_folder_and_skips_hidden(tmp_path: Path) -> None:
    (tmp_path / "p102-estuary-multibeam" / "drafts").mkdir(parents=True)
    (tmp_path / "p102-estuary-multibeam" / "P102 summary.docx").write_bytes(b"x")
    (tmp_path / "p102-estuary-multibeam" / "drafts" / "notes.MD").write_text("x")
    (tmp_path / "overview.md").write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.xlsx").write_text("x")
    (tmp_path / ".DS_Store").write_text("x")

    sources = discover_corpus(tmp_path)

    assert [(s.rel_path, s.project_id, s.format) for s in sources] == [
        ("overview.md", "root", "md"),
        ("p102-estuary-multibeam/P102 summary.docx", "p102-estuary-multibeam", "docx"),
        ("p102-estuary-multibeam/drafts/notes.MD", "p102-estuary-multibeam", "md"),
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


def test_markdown_that_is_not_utf8_is_a_conversion_error(tmp_path: Path) -> None:
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "memo.md").write_bytes(b"# Memo\n\n\xff\xfe broken")

    with pytest.raises(ConversionError, match=r"p1/memo\.md is not UTF-8"):
        compile_corpus(tmp_path, FakeGateway(), topic="t", title="T")


def test_curator_notes_are_kept_and_compiled_as_searchable_guidance(tmp_path: Path) -> None:
    (tmp_path / "p103-reef-sonar").mkdir()
    note = "# Status\n\nThe reef Sentinel-2 case study has not been written yet."
    (tmp_path / "p103-reef-sonar" / "AGENTS.md").write_text(note)
    (tmp_path / "p103-reef-sonar" / "estuary-proposal.md").write_text(
        "# Scope\n\nSentinel-2 bathymetry."
    )
    gateway = FakeGateway()

    corpus = compile_corpus(tmp_path, gateway, topic="bathymetry", title="Bathymetry")

    assert [(n.scope_path, n.content) for n in corpus.curator_notes] == [
        ("p103-reef-sonar/AGENTS.md", note)
    ]
    guidance, proposal = corpus.documents
    assert (guidance.doc_id, guidance.stage, guidance.client) == ("doc_001", "guidance", "curator")
    assert guidance.title == "Curator guidance: p103-reef-sonar"
    assert {c.kind for c in guidance.chunks} == {"guidance"}
    assert "case study has not been written" in guidance.chunks[0].content
    assert proposal.stage == "proposal"
    assert gateway.calls.count("extract_document_meta") == 1
    assert corpus.embeddings.shape == (len(corpus.all_chunks()), FakeGateway.embedding_dimensions)


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


def _sections(drafts: Sequence[SectionDraft]) -> list[Section]:
    return [
        Section(
            section_id=f"doc_001_sec_{i:03d}",
            ordinal=i,
            heading_path=d.heading_path,
            level=d.level,
            page_start=d.page_start,
            page_end=d.page_end,
            markdown=d.markdown,
        )
        for i, d in enumerate(drafts, start=1)
    ]


CTX = ChunkContext(
    "doc_001", "p101-harbour-lidar", "Northwind Marine", "memorandum", "Memo", "prose"
)


def test_chunks_stitched_across_a_page_break_span_both_pages() -> None:
    drafts = assemble_sections(
        [
            (4, page(block("heading", "Areas", level=1), block("table", "| a |\n| --- |\n| 1 |"))),
            (5, page(block("table", "| 2 |", continues=True), block("paragraph", "The model"))),
            (6, page(block("paragraph", "was validated.", continues=True))),
        ]
    )

    chunks = build_chunks(drafts, _sections(drafts), CTX)

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

    chunks = build_chunks(drafts, _sections(drafts), CTX)

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


def test_chunks_carry_breadcrumbs_and_emit_dedicated_visual_chunks() -> None:
    chart = VisualTranscript(label="Rainfall chart", transcript="Median 330.9 mm")
    long_paragraph = " ".join(f"Sentence {i} about groundwater." for i in range(150))
    drafts = assemble_sections(
        [
            (2, page(block("heading", "Climate", level=1), block("paragraph", long_paragraph))),
            (3, page(block("paragraph", "Short note."), visuals=(chart,))),
        ]
    )
    sections = [
        Section(
            section_id="doc_001_sec_001",
            ordinal=1,
            heading_path="Climate",
            level=1,
            page_start=2,
            page_end=3,
            markdown=drafts[0].markdown,
        )
    ]
    ctx = ChunkContext(
        "doc_001", "p101-harbour-lidar", "Northwind Marine", "memorandum", "Memo", "prose"
    )
    chunks = build_chunks(drafts, sections, ctx)

    assert [c.chunk_id for c in chunks][:2] == ["doc_001_chk_0001", "doc_001_chk_0002"]
    assert all(
        c.breadcrumb
        == "[Project: p101-harbour-lidar | Client: Northwind Marine | Stage: memorandum"
        " | Doc: Memo (pp. 2-3) | Section: Climate]"
        for c in chunks
    )
    prose = [c for c in chunks if c.kind == "prose"]
    assert len(prose) >= 3 and all(len(c.content) <= 1800 for c in prose)
    assert " ".join(c.content for c in prose).startswith("Sentence 0 about")
    visual = [c for c in chunks if c.kind == "visual"]
    assert len(visual) == 1
    assert (visual[0].page_start, visual[0].page_end) == (3, 3)
    assert "Rainfall chart" in visual[0].content and "330.9" in visual[0].content


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


def test_fidelity_retry_with_strong_model_recovers_missing_numbers() -> None:
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
    assert gateway.transcribe_calls == [(False, ()), (True, ("0.05", "330.9"))]


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
    assert len(gateway.transcribe_calls) == 2


# Cited synthesis.


def _documents(tmp_path: Path) -> tuple[CompiledDocument, ...]:
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "memo.md").write_text("# Results\n\nSeagrass mapped.")
    return compile_corpus(tmp_path, FakeGateway(), topic="t", title="T").documents


def test_synthesis_retries_once_with_feedback_on_invalid_citations(tmp_path: Path) -> None:
    documents = _documents(tmp_path)
    gateway = FakeGateway(
        synthesis_script=["See [doc_001:bad_sec].", "See [doc_001:doc_001_sec_001]."]
    )

    markdown = synthesize_with_citations(gateway, "t", (), documents)

    assert markdown == "See [doc_001:doc_001_sec_001]."
    feedback = gateway.synthesis_feedback[1]
    assert feedback is not None
    assert "[doc_001:bad_sec]" in feedback and "[doc_001:doc_001_sec_001]" in feedback


def test_synthesis_without_valid_citations_after_retry_raises(tmp_path: Path) -> None:
    documents = _documents(tmp_path)
    gateway = FakeGateway(synthesis_script=["No citations.", "Still [doc_002:doc_002_sec_001]."])

    with pytest.raises(InvalidCitationError, match="doc_002_sec_001"):
        synthesize_with_citations(gateway, "t", (), documents)


class GuidanceStageGateway(FakeGateway):
    def extract_document_meta(
        self, rel_path: str, project_id: str, outline_and_excerpt: str
    ) -> DocumentMetaExtraction:
        return DocumentMetaExtraction(
            title="Style guide",
            client="Contoso Ports",
            stage="guidance",
            doc_date=None,
            summary="How.",
        )


def test_citations_resolve_exactly_to_sections_of_non_curator_documents(tmp_path: Path) -> None:
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "AGENTS.md").write_text("# Note\n\nDraft only.")
    (tmp_path / "p1" / "style-guide.md").write_text("# Style\n\nWrite plainly.")

    corpus = compile_corpus(tmp_path, GuidanceStageGateway(), topic="t", title="T")
    docs = corpus.documents

    # A non-curator document the model classes as guidance is still catalogued and citable.
    assert corpus.synthesis_markdown.endswith("[doc_002:doc_002_sec_001].")
    assert invalid_citations("See [doc_002:doc_002_sec_001].", docs) == []
    assert invalid_citations(
        "[doc_002:doc_002_sec_001] [doc_002: doc_002_sec_001] [doc_001:doc_001_sec_001]", docs
    ) == ["[doc_002: doc_002_sec_001]", "[doc_001:doc_001_sec_001]"]
    assert invalid_citations("Only [docs] here.", docs) == ["[docs]", "(no valid citation)"]


def test_second_compile_with_build_cache_makes_no_provider_requests(tmp_path: Path) -> None:
    (tmp_path / "corpus" / "p1").mkdir(parents=True)
    # Sorts before AGENTS.md, so it is doc_001: the section the fake synthesis cites.
    (tmp_path / "corpus" / "p1" / "2024-memo.md").write_text("# Results\n\nSeagrass at 59,494 ha.")
    (tmp_path / "corpus" / "p1" / "AGENTS.md").write_text("Memo is final.")
    client = FakeOpenAIClient()

    def build() -> list[str]:
        with BuildCache(tmp_path / "cache.db") as cache:
            gateway = OpenAIGateway(
                client=cast(openai.OpenAI, client), cache=cache, embedding_dimensions=8
            )
            corpus = compile_corpus(tmp_path / "corpus", gateway, topic="t", title="T")
        return [corpus.synthesis_markdown, *(c.content for c in corpus.all_chunks())]

    first = build()
    requests_after_first = client.call_count
    second = build()

    assert requests_after_first == 3
    assert client.call_count == requests_after_first
    assert second == first


# Rendered formats end to end.


def test_pdf_compiles_through_render_transcribe_and_metadata(tmp_path: Path) -> None:
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
            ["## Outputs", "Raster harbour_dry_depth_thresh.tif"],
        ],
    )

    corpus = compile_corpus(tmp_path / "corpus", FakeGateway(), topic="t", title="T")

    (doc,) = corpus.documents
    assert (doc.format, doc.project_id, doc.stage) == ("pdf", "p101-harbour-lidar", "memorandum")
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
