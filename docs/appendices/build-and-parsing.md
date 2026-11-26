# Appendix A: Parsing, induction, and build pipeline

> Technical appendix. For the plain-language summary, read [How folders become searchable knowledge](../how-it-works.md). The authoritative specification is in [Reading](../architecture.md#reading), [Induction](../architecture.md#induction), and [Snapshot file](../architecture.md#snapshot-file).

`arkive-rag build` turns a folder of `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, and `.md` documents into one read-only SQLite snapshot (`*.arkive.db`, schema `7.0`). The compiler assumes no folder hierarchy, domain vocabulary, or heading convention: a flat folder of Word and PDF files goes through the same three passes as a 126-file multi-client archive.

`compile_corpus` in `src/arkive_rag/compiler.py` runs the passes in order: per-document reading, corpus induction, and binding with embeddings.

![Arkive three-pass build and parsing pipeline](../assets/build-pipeline.svg)

[Open the interactive diagram](../assets/build-pipeline.html)

## Pass 1: per-document reading (`compiler.py`)

`read_document` processes each source file independently in a thread pool and produces a `DocumentReading` that depends only on the file's bytes and format. The relative path is carried only for diagnostics.

### Discovery, path-hash IDs, and document dates

`discover_corpus` walks the corpus in sorted POSIX relative-path order, rejects symlinks and unsupported file extensions before any model call, and hashes each file's bytes with SHA-256.

- **Stable document IDs.** `document_id(rel_path)` hashes the NFC-normalised POSIX path to `doc_<10hex>`. For example, `q102-oskary-quarry/q102-technical-memorandum.pdf` in `coastal-survey-projects` becomes `doc_cc01ac49e1`. Adding or moving another file never renumbers existing documents. SQLite queries keep the sorted-path order through an `ordinal` column.
- **Document dates.** Arkive stores a validated `DocumentDate` (`iso` and `basis`). A printed date comes from the document's observed text (`basis: "printed"`). When a file has no printed date, the compiler reads embedded PDF metadata (`/CreationDate`, `/ModDate`) or Office `docProps/core.xml` properties (`basis: "metadata"`). In the flat `acoustic-consultancy-files` collection, which has no printed dates and no folders, `corvane-depot-night-noise-report.pdf` (`doc_d81e70766d`) receives `iso="2025-06-30", basis="metadata"`.
- **Curator notes.** Every `AGENTS.md` file in the corpus is marked `is_curator_note = True` and read directly, without a model call. Curator notes steer induction and appear in `get_topic_guide` and `prepare_draft`. The induced guide, glossary, and feedback lessons never cite them.

### Office conversion, rendering, and the render cache

`convert_office_to_pdf` converts `.docx` and `.pptx` files to PDF in an isolated headless LibreOffice (`soffice`) temporary directory. Checkboxes, floating text boxes, embedded charts, and slide layouts then follow the same page pipeline as native PDFs.

Before any part of an Office archive (`.docx`, `.pptx`, `.xlsx`) reaches LibreOffice, Docling, or openpyxl, `open_office_archive` in `native_formats.py` inflates every member in 64 KiB steps and refuses an archive that is not a readable zip file, declares more than 5,000 entries (`MAX_OFFICE_ARCHIVE_ENTRIES`), holds an encrypted member or one whose inflated size or CRC differs from its directory entry, or inflates to more than 100,000,000 bytes (`MAX_OFFICE_ARCHIVE_BYTES`).

A Word file is converted from an accepted-revision copy (`accepted_revision_copy`): deleted and moved-away runs are removed and inserted runs kept, because LibreOffice otherwise draws a deletion and its replacement into one PDF text run. The original bytes stay the document's SHA-256 and metadata source. `page_supplements` in `compiler.py` then adds what the file's own XML holds as quoted `[Annotation — ...]` blocks on the page that holds the marked text: each Word tracked insertion and deletion (with author and date) and each comment, and each shown PowerPoint slide's speaker notes and comments. A PowerPoint slide's native chart series become a Markdown table under a `Chart: <title>` line. Hidden slides are skipped, as LibreOffice skips them. Annotations are their own `annotation` chunks, and an observed document attribute never counts a quote found only inside an annotation or visual transcript.

### Excel workbooks and HTML pages

`.xlsx`, `.html`, and `.htm` files have no pages. `xlsx_markdown` and `html_markdown` read them from an in-memory stream through Docling's spreadsheet and HTML backends. The dependency is `docling-slim` with only its `format-xlsx` and `format-html` extras, so no PyTorch is installed and serving never loads Docling.

- **Workbooks.** Each worksheet becomes a level-1 heading, and each data region a `Range A2:D5` line and a GFM table. A merged multi-row header is flattened into one row (`Group — Column (Unit)`). A formula saved without a computed value shows its formula (`=B4+C4`). Cell notes and Excel 365 threaded comments with their replies follow the sheet's tables as annotations. A workbook with more than 250,000 populated cells (`MAX_WORKBOOK_CELLS`, counted while streaming each sheet) is refused with `ConversionError` instead of being cut. A legacy `.xls` file stops the build as an unsupported format.
- **HTML pages.** Every image fetch, remote or local, is off, as are page rendering and furniture inference, so text before the first heading stays body text. Headings, paragraphs, lists, code blocks, and tables follow in document order. Images and their alt text are left out.

Their Markdown is split into sections by `parse_markdown_blocks` like a `.md` file, and each section spans page 1. They have no page images for `view_page`.

`render_pdf` uses `pypdfium2` to extract each page's Unicode native text layer and render a 2x WebP image (`webp_bytes`, longest side clamped to `MAX_RENDER_SIDE_PX = 4096`). With `--cache`, `BuildCache` stores rendered pages and metadata dates in SQLite (`page_render_cache` and `page_render_meta`) keyed by source SHA-256, and checks the cache before invoking LibreOffice or `pypdfium2`. Rebuilding an unchanged corpus renders zero pages.

### Vision transcription, fidelity checks, and section stitching

For each rendered page, `transcribe_with_fidelity` sends `gpt-6.1-sol` (low reasoning effort) three inputs: the 2x WebP page image, the native text layer, and the last 1,200 characters (`PREV_CONTEXT_CHARS`) of the preceding page's transcribed blocks, excluding visual transcripts. The model returns structured blocks (headings, paragraphs, tables, lists) and separate `VisualTranscript` blocks for charts, maps, and diagrams. A value that exists only inside a chart image enters a `VisualTranscript`.

Two mechanical checks guard transcription:

1. **Numeric and filename fidelity.** The compiler extracts anchor tokens (numbers, measurements, filenames) from the native text layer and checks that they survive transcription. When a page drops anchors or shows ligature corruption, Arkive retries it once at high reasoning effort. When the retry changes the page's trailing 1,200-character tail, page `N + 1` is transcribed again. Propagation stops as soon as a tail is unchanged.
2. **Cross-page section and table stitching.** `assemble_sections` builds Markdown sections with hierarchical `heading_path` strings and `page_start..page_end` spans. A paragraph or table that continues across a page break (`continues_from_previous_page=True`) is joined onto the previous part. A table row split mid-cell (`merges_into_previous_row=True`) has its cells joined column by column, and each row keeps the pages it came from.

Sections longer than `MAX_SECTION_CHARS = 32_000` split at paragraph or table boundaries, never inside fenced code blocks or visual transcripts. `section_pieces` then splits each section into `prose`, `table`, `visual`, `annotation`, or `guidance` chunk pieces of at most `MAX_CHUNK_CHARS = 6_000`. The section itself stays whole for reading.

### Per-document observation cards

`ObserveSections` asks `gpt-6.1-sol` (low effort; batched over long documents with one consolidation request) for an observation card per citable document: title, printed date, proposed deliverable type, summary, the attributes the document states about itself, and the terms it uses. Every quoted term or attribute is matched back to a section through `fold_for_quote`. Unlocated quotes get one rerun that names them, then are discarded and counted in `build_report.discarded_unlocated_quotes`. Each located attribute (such as `reference`, `client`, `site`, `status`, `author`, or `instruments`) is stored in `document_attributes` with the section that states it; a quote found only inside an annotation or visual transcript does not count.

## Pass 2: bounded corpus induction (`induction.py`)

`induce` runs once for the whole corpus over the observation cards, folder paths, and curator notes, with documents in sorted path order. Every Pass 2 prompt has a hard ceiling of `MAX_PROMPT_CHARS = 90_000` (about 30k tokens), so a large archive never overflows the context window.

1. **Facets and uncapped deliverable types (`OrganiseCorpus`, `CanonicaliseFacets`, `CanonicaliseTypes`).** Document digests go to the model in batches of at most 44,000 characters (`MAX_ORGANISE_BATCH_CHARS`). The model proposes 0 to 6 corpus-specific facets (`MAX_FACETS`), each either `path`-bound (from folders) or `content`-bound (from document text), and gives each document a deliverable type. A review is marked feedback for the types it reviews and keeps its own type. One facet canonicalisation request receives every distinct facet the batches proposed and merges synonymous facets and values into at most 6 facets. One type canonicalisation request maps every proposed type label to a canonical type, with no cap on the number of types.
2. **Facet binding (`bind_facets`).** Path-bound facets take values from folder segments. `gpt-6-luna` on the Decisions API judges content-bound values for each citable document against its card and outline. Unused values and facets with fewer than two populated values are pruned and logged in `build_report`.
3. **Version families (`PartitionVersions`).** Python proposes scored pairs of documents with high word 4-gram overlap, or overlapping title and filename words within one type, ignoring boilerplate held by more than 12 documents (`VERSION_COMMON_POSTING`). Pairs join documents into components of at most 16 (`MAX_VERSION_COMPONENT_DOCS`); weaker pairs that would exceed it are dropped and counted in `build_report.dropped_version_edges`. For each component, one `gpt-6.1-sol` request partitions the documents into families and names a latest member only when one clearly supersedes the others; a file metadata date never proves supersession. The latest member is `latest` and the others `superseded`. When the order is ambiguous, every distinct member is `unordered`.
4. **Verified feedback lessons (`ExtractFeedbackLessons`).** For each feedback document, `gpt-6.1-sol` extracts transferable reviewer lessons with verbatim quotes. Python locates each quote in a section, and `gpt-6-luna` confirms the lessons of each section in one request. Unconfirmed lessons are dropped and recorded in `build_report.dropped_feedback_lessons`.
5. **Cited glossary (`WriteGlossary`).** Term observations and occurrences produce cited senses. `gpt-6-luna` checks each proposed definition against its cited evidence. An unsupported definition is demoted to `null` and counted in `build_report.unsupported_definitions`. A `null` definition means no definition was established from that sense's cited evidence. It does not prove the corpus never defines the term, and `unsupported_definitions` counts only demoted definitions, not every `null` sense.
6. **Recurring text units (`recurring_units` in `passages.py`).** Pure Python extracts normalised paragraphs, list items, and table header rows from every citable document, indexes their word 4-grams, and clusters units that recur in more than one document into `unit_clusters` and `cluster_sections`, keeping each holder's matched sample text. No model call is made.
7. **Bounded topic guide (`WriteGuide`).** `gpt-6.1-sol` writes a 4,000 to 8,000 character overview (hard ceiling 16,000 characters and 24,000 JSON bytes) from a bounded catalog of documents, facets, types, glossary terms, and curator notes. Python verifies that every `[doc_id:section_id]` citation resolves to a citable section of the document it names.

## Pass 3: binding, embeddings, and atomic snapshot

`bind_document` in `compiler.py` gives sections and chunks their final IDs (`<doc_id>_sec_001`, `<doc_id>_chk_0001`) and prepends a breadcrumb to every chunk with its bound facets, document title, page span, and heading path. `text-embedding-3-large` (1,536 dimensions) embeds `breadcrumb + "\n" + content` in batches, and each vector is L2-normalised to `float32`.

`write_snapshot` in `snapshot.py` fills a temporary sibling SQLite file (mode `0600`) in one transaction, builds the FTS5 index `chunks_fts` (NFKC-normalised content weighted 10, breadcrumbs weighted 2), runs foreign-key and integrity checks, and atomically renames the file into place. Replacing an existing snapshot keeps that file's mode. A snapshot of an older schema is rebuilt from its corpus.

### Snapshot contents

The snapshot holds the corpus metadata and topic guide, the verbatim curator notes, one row per source document with its facets, type, date, and version family, its verified attributes (`document_attributes`), every Markdown table of its sections stored row by row with an FTS5 index (`document_tables`, `table_rows`, `table_rows_fts`), the glossary and reviewer lessons, the recurring text clusters, every stitched section and rendered page, and the chunks with their embeddings and FTS5 index. [Snapshot file](../architecture.md#snapshot-file) in the architecture reference gives every table and column.

### Model calls during a build

| Operation | Model | When it runs |
| --- | --- | --- |
| Page transcription | `gpt-6.1-sol`, low effort (high on fidelity retry) | Once per rendered page. |
| Observation | `gpt-6.1-sol`, low effort | Once per batch of a citable document's sections, plus one consolidation for a multi-batch document. |
| Organisation, facet and type canonicalisation, version partition, glossary, guide | `gpt-6.1-sol`, medium effort | Organisation once per batch of digests; each canonicalisation and the guide once per topic; the version partition once per component of candidate versions; the glossary once per batch of terms. |
| Feedback lessons | `gpt-6.1-sol`, low effort | Once per batch of a feedback document's sections. |
| Facet binding, support checks | `gpt-6-luna` on the Decisions API | Once per citable document for content-bound facets, once per glossary definition, and once per section holding feedback lessons. |
| Chunk embeddings | `text-embedding-3-large` | Batched once per topic. |

With `--cache`, provider responses are stored under a hash of their inputs. An unchanged rebuild makes no provider calls. Adding a file reads only that file and requests induction over the changed corpus, where identical requests still hit the cache.

`build` prints the provider calls, tokens, and estimated cost, then the build report: observations and lessons dropped for an unlocated quote, definitions and lessons the support checks did not confirm, refused decision questions, pruned facet values and facets, and curator notes a prompt could not show in full.

## Worked example from a public corpus

The file `archive/m2409-northbray-askern-viaduct/m2409-installation-and-baseline-report.docx` in the 126-file fictional `structural-monitoring-consultancy` collection shows what the three passes produce. The figures come from the metered schema `7.0` cold build of the collection.

1. **Discovery and identity.** `document_id` hashes the path to `doc_4957546a24`. LibreOffice converts the Word file to PDF, and `pypdfium2` renders all 52 pages.
2. **Transcription and assembly.** Transcription and stitching produce 86 sections (`doc_4957546a24_sec_001` to `doc_4957546a24_sec_086`), keeping the document-control tables and multi-page instrument schedules.
3. **Induction.** Across the collection, induction creates 28 canonical deliverable types, 30 verified reviewer lessons, and a 10,395-character cited topic guide. For `doc_4957546a24` it assigns:
   - `deliverable_type`: `Installation and baseline report`
   - `doc_date`: `14 October 2024`
   - `version_status`: `sole`
   - bound facets: `issue_status: issued`, `job_cohort: archive`, `project: M2409 Askern Viaduct`, `sensing_technology: fibre_bragg_grating`
   - 17 verified attributes, including `reference: MI-IR-2409`, `status: Issue 1, final`, and `instruments: Prisma FI-4 interrogator, serial PF4-1876`
   - 61 stored tables in `document_tables`

   Facet names come from induction and can differ between independent builds of the same collection (see [Appendix C](evaluation-benchmarks.md#version-families-and-facet-coverage)).

   In the same folder, the calibration sheet `prisma-fi-4-calibration-pf4-1876.md` (`doc_a971f575a6`) is byte-identical to a copy in the `m2505-halverton-greyscar-spillway` folder (`doc_fed78f012b`). Induction groups them as one family and marks `doc_a971f575a6` `superseded` by `doc_fed78f012b`.
4. **Stored chunk breadcrumb.** Chunk `doc_4957546a24_chk_0001` (`kind="table"`, page 1) is indexed and embedded as:

```text
[job_cohort: archive | sensing_technology: fibre_bragg_grating | issue_status: issued | project: M2409 Askern Viaduct | Doc: Askern Viaduct: fibre Bragg grating installation and baseline report (pp. 1-1) | Section: Askern Viaduct: fibre Bragg grating installation and baseline report]
| Item | Detail |
| --- | --- |
| Our ref | MI-IR-2409 |
| Client | Northbray Rail Infrastructure |
| Client contact | Mr Jarrow Keld |
| Author | Maren Lisk |
...
```

5. **Linked reviewer lesson.** From the unlabelled QA register `active/m2505-halverton-greyscar-spillway/qa-disposition-register.docx` (`doc_4329fae0a3`), Pass 2 extracts and verifies seven lessons that target `Installation and baseline report`, including: *"Apply strain-free reference grating temperature compensation before reporting wavelength shifts as strain, and state this step in the method."*

Rebuilding the same collection with `--cache` and no changes made 0 provider calls ($0.0000). On the earlier schema `6.0` build, inserting a 127th file, `a00-new-intake/a00-scoping-note.md`, at the start of the directory tree left all 126 existing `doc_id` values unchanged.
