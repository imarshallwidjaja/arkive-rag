# Appendix A: Parsing, induction, and build pipeline

> Technical appendix. For the plain-language summary, read [How folders become searchable knowledge](../how-it-works.md). The authoritative specification is in [Reading](../architecture.md#reading), [Induction](../architecture.md#induction), and [Snapshot file](../architecture.md#snapshot-file).

`arkive-rag build` turns a folder of `.pdf`, `.docx`, `.pptx`, and `.md` documents into one read-only SQLite snapshot (`*.arkive.db`, schema `6.0`). The compiler assumes no folder hierarchy, domain vocabulary, or heading convention: a flat folder of Word and PDF files goes through the same three passes as a 126-file multi-client archive.

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

`render_pdf` uses `pypdfium2` to extract each page's Unicode native text layer and render a 2x WebP image (`webp_bytes`, longest side clamped to `MAX_RENDER_SIDE_PX = 4096`). With `--cache`, `BuildCache` stores rendered pages and metadata dates in SQLite (`page_render_cache` and `page_render_meta`) keyed by source SHA-256, and checks the cache before invoking LibreOffice or `pypdfium2`. Rebuilding an unchanged corpus renders zero pages.

### Vision transcription, fidelity checks, and section stitching

For each rendered page, `transcribe_with_fidelity` sends `gpt-6.1-sol` (low reasoning effort) three inputs: the 2x WebP page image, the native text layer, and the last 1,200 characters (`PREV_CONTEXT_CHARS`) of the preceding page's transcribed blocks, excluding visual transcripts. The model returns structured blocks (headings, paragraphs, tables, lists) and separate `VisualTranscript` blocks for charts, maps, and diagrams. A value that exists only inside a chart image enters a `VisualTranscript`.

Two mechanical checks guard transcription:

1. **Numeric and filename fidelity.** The compiler extracts anchor tokens (numbers, measurements, filenames) from the native text layer and checks that they survive transcription. When a page drops anchors or shows ligature corruption, Arkive retries it once at high reasoning effort. When the retry changes the page's trailing 1,200-character tail, page `N + 1` is transcribed again. Propagation stops as soon as a tail is unchanged.
2. **Cross-page section and table stitching.** `assemble_sections` builds Markdown sections with hierarchical `heading_path` strings and `page_start..page_end` spans. A paragraph or table that continues across a page break (`continues_from_previous_page=True`) is joined onto the previous part. A table row split mid-cell (`merges_into_previous_row=True`) has its cells joined column by column, and each row keeps the pages it came from.

Sections longer than `MAX_SECTION_CHARS = 32_000` split at paragraph or table boundaries, never inside fenced code blocks or visual transcripts. `section_pieces` then splits each section into `prose`, `table`, `visual`, or `guidance` chunk pieces of at most `MAX_CHUNK_CHARS = 6_000`. The section itself stays whole for reading.

### Per-document observation cards

`ObserveSections` asks `gpt-6.1-sol` (low effort; batched over long documents with one consolidation request) for an observation card per citable document: title, printed date, proposed deliverable type, whether the document is reviewer feedback and which types it reviews, summary, observed attributes, domain terms, and craft notes. Every quoted term or craft observation is matched back to a section through `fold_for_quote`. Unlocated quotes are discarded and counted in `build_report.discarded_unlocated_quotes`.

## Pass 2: bounded corpus induction (`induction.py`)

`induce` runs once for the whole corpus over the observation cards, folder paths, and curator notes, with documents in sorted path order. Every Pass 2 prompt has a hard ceiling of `MAX_PROMPT_CHARS = 90_000` (about 30k tokens), so a large archive never overflows the context window.

1. **Facets and uncapped deliverable types (`OrganiseCorpus`, `CanonicaliseTypes`).** Document digests go to the model in batches of at most 44,000 characters (`MAX_ORGANISE_BATCH_CHARS`). The model proposes 0 to 6 corpus-specific facets (`MAX_FACETS`), each either `path`-bound (from folders) or `content`-bound (from document text), and gives each document a deliverable type. A review is marked feedback for the types it reviews and keeps its own type. One canonicalisation request maps every proposed type label to a canonical type, with no cap on the number of types.
2. **Facet binding (`bind_facets`).** Path-bound facets take values from folder segments. `gpt-6-luna` on the Decisions API judges content-bound values for each citable document against its card and outline. Unused values and facets with fewer than two populated values are pruned and logged in `build_report`.
3. **Version families (`version_families`).** Python proposes pairs of documents with high word 4-gram overlap, or overlapping title and filename words within one type. `gpt-6-luna` confirms which pairs are drafts, revisions, or exports of the same deliverable. Validated dates mark one member `latest` and older members `superseded`. When dates tie or are missing, `gpt-6-luna` orders the family. When nothing orders it, every member is `unordered`.
4. **Verified feedback lessons (`ExtractFeedbackLessons`).** For each feedback document, `gpt-6.1-sol` extracts transferable reviewer lessons with verbatim quotes. Python locates each quote in a section, and `gpt-6-luna` confirms the lessons of each section in one request. Unconfirmed lessons are dropped and recorded in `build_report.dropped_feedback_lessons`.
5. **Cited glossary (`WriteGlossary`).** Term observations and occurrences produce cited senses. `gpt-6-luna` checks each proposed definition against its cited evidence. An unsupported definition is demoted to `null` and counted in `build_report.unsupported_definitions`. A `null` definition means no definition was established from that sense's cited evidence. It does not prove the corpus never defines the term, and `unsupported_definitions` counts only demoted definitions, not every `null` sense.
6. **Recurring text units (`recurring_units` in `passages.py`).** Pure Python extracts normalised paragraphs, list items, and table header rows from every citable document, indexes their word 4-grams, and clusters units that recur in more than one document into `unit_clusters` and `cluster_sections`, keeping each holder's matched sample text. No model call is made.
7. **Bounded topic guide (`WriteGuide`).** `gpt-6.1-sol` writes a 4,000 to 8,000 character overview (hard ceiling 16,000 characters and 24,000 JSON bytes) from a bounded catalog of documents, facets, types, glossary terms, and curator notes. Python verifies that every `[doc_id:section_id]` citation resolves to a citable section of the document it names.

## Pass 3: binding, embeddings, and atomic snapshot

`bind_document` in `compiler.py` gives sections and chunks their final IDs (`<doc_id>_sec_001`, `<doc_id>_chk_0001`) and prepends a breadcrumb to every chunk with its bound facets, document title, page span, and heading path. `text-embedding-3-large` (1,536 dimensions) embeds `breadcrumb + "\n" + content` in batches, and each vector is L2-normalised to `float32`.

`write_snapshot` in `snapshot.py` fills a temporary sibling SQLite file (mode `0600`) in one transaction, builds the FTS5 index `chunks_fts` (NFKC-normalised content weighted 10, breadcrumbs weighted 2), runs foreign-key and integrity checks, and atomically renames the file into place. Replacing an existing snapshot keeps that file's mode. A snapshot of an older schema is rebuilt from its corpus.

### Snapshot contents

The snapshot holds the corpus metadata and topic guide, the verbatim curator notes, one row per source document with its facets, type, date, and version family, the glossary and reviewer lessons, the recurring text clusters, every stitched section and rendered page, and the chunks with their embeddings and FTS5 index. [Snapshot file](../architecture.md#snapshot-file) in the architecture reference gives every table and column.

### Model calls during a build

| Operation | Model | When it runs |
| --- | --- | --- |
| Page transcription | `gpt-6.1-sol`, low effort (high on fidelity retry) | Once per rendered page. |
| Observation | `gpt-6.1-sol`, low effort | Once per batch of a citable document's sections, plus one consolidation for a multi-batch document. |
| Organisation, type canonicalisation, glossary, guide | `gpt-6.1-sol`, medium effort | Organisation once per batch of digests; canonicalisation and the guide once per topic; the glossary once per batch of terms. |
| Feedback lessons | `gpt-6.1-sol`, low effort | Once per batch of a feedback document's sections. |
| Facet binding, version judgements, support checks | `gpt-6-luna` on the Decisions API | Once per citable document for content-bound facets, once per window of candidate version pairs, once per family that dates cannot order, once per glossary definition, and once per section holding feedback lessons. |
| Chunk embeddings | `text-embedding-3-large` | Batched once per topic. |

With `--cache`, provider responses are stored under a hash of their inputs. An unchanged rebuild makes no provider calls. Adding a file reads only that file and requests induction over the changed corpus, where identical requests still hit the cache.

`build` prints the provider calls, tokens, and estimated cost, then the build report: observations and lessons dropped for an unlocated quote, definitions and lessons the support checks did not confirm, refused decision questions, pruned facet values and facets, and curator notes a prompt could not show in full.

## Worked example from a public corpus

The file `archive/m2409-northbray-askern-viaduct/m2409-installation-and-baseline-report.docx` in the 126-file fictional `structural-monitoring-consultancy` collection shows what the three passes produce:

1. **Discovery and identity.** `document_id` hashes the path to `doc_4957546a24`. LibreOffice converts the Word file to PDF, and `pypdfium2` renders all 52 pages.
2. **Transcription and assembly.** Transcription and stitching produce 86 sections (`doc_4957546a24_sec_001` to `doc_4957546a24_sec_086`), keeping the document-control tables and multi-page instrument schedules.
3. **Induction.** Across the collection, induction creates 28 canonical deliverable types, 29 verified reviewer lessons, and a 9,724-character cited topic guide. For `doc_4957546a24` it assigns:
   - `deliverable_type`: `Installation and baseline report`
   - `doc_date`: `14 October 2024`
   - `version_status`: `sole`
   - bound facets: `job_status: archived`, `project: M2409 — Askern Viaduct`, `sensing_technology: fibre_bragg_grating`, `monitoring_technology: Fibre Bragg grating`

   In the same folder, the calibration sheet `prisma-fi-4-calibration-pf4-1876.md` (`doc_a971f575a6`) is byte-identical to a copy in the `m2505-halverton-greyscar-spillway` folder (`doc_fed78f012b`). Induction groups them as one family and marks `doc_a971f575a6` `superseded` by `doc_fed78f012b`.
4. **Stored chunk breadcrumb.** Chunk `doc_4957546a24_chk_0001` (`kind="table"`, page 1) is indexed and embedded as:

```text
[job_status: archived | sensing_technology: fibre_bragg_grating | project: M2409 — Askern Viaduct | monitoring_technology: Fibre Bragg grating | Doc: Askern Viaduct: fibre Bragg grating installation and baseline report (pp. 1-1) | Section: Askern Viaduct: fibre Bragg grating installation and baseline report]
| Item | Detail |
| --- | --- |
| Our ref | MI-IR-2409 |
| Client | Northbray Rail Infrastructure |
| Client contact | Mr Jarrow Keld |
| Author | Maren Lisk |
```

5. **Linked reviewer lesson.** From the unlabelled QA register `active/m2505-halverton-greyscar-spillway/qa-disposition-register.docx` (`doc_4329fae0a3`), Pass 2 extracts and verifies a lesson that targets `Installation and baseline report`: *"Explain that strain is calculated only after subtracting the strain-free reference grating response for temperature compensation."*

Rebuilding the same collection with `--cache` and no changes made 0 provider calls ($0.0000). Inserting a 127th file, `a00-new-intake/a00-scoping-note.md`, at the start of the directory tree left all 126 existing `doc_id` values unchanged.
