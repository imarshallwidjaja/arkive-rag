# Product goal and system walkthrough

Arkive RAG turns a human-curated folder of reference documents into a single SQLite file (`*.arkive.db`) that an MCP-capable agent can query and read.

## Curating a topic without data-preparation overhead

Domain leads, engineers, and marketing managers already hold examples of strong work: final technical reports, slide decks, proposals, reviewer feedback tables, and campaign briefs. Requiring them to clean text layers, design chunking rules, or label documents before building a knowledge base blocks adoption.

Arkive keeps curation in folders and prose:

1. A person collects the reference documents that represent good work on a specific topic (typically 10 to 200 `.pdf`, `.docx`, `.pptx`, or `.md` files) and arranges them in a directory, in whatever folder structure suits the collection: one folder per project, per campaign, per author, or none.
2. Optionally, they add `AGENTS.md` files in plain language explaining how the collection fits together, how it should be filtered, what is still missing, or how later work should use those examples.
3. Running `arkive-rag build` calls OpenAI vision, text, decision, and embedding models at build time to transcribe pages, repair broken PDF font encodings, extract chart and map text, stitch tables across page breaks, observe each document, and induce from the whole collection the facets it can be filtered by, each document's canonical deliverable type and feedback role, version families of drafts and copies, reviewer lessons, the text units that recur across documents, and a cited glossary. It writes one read-only `.arkive.db` file.
4. `arkive-rag serve` exposes one or several `.arkive.db` files over MCP. An agent can mount multiple topic snapshots in one session (for example, a technical survey collection alongside an executive presentation or marketing collection) without scanning dozens of raw source files on every turn.

## Two ways agents use a snapshot

A curated snapshot supports two distinct agent workflows through the same MCP surface:

- **Pinpoint factual and visual answers:** finding exact numbers, filenames, parameter tables, reviewer answers, or chart legend values across the collection, narrowing by facet (client, project, campaign, author) when the question names one, and citing the source `doc_id`, `section_id`, and page numbers.
- **Deliverable authoring aligned to house craft:** calling `prepare_draft` with the new project's inputs returns up to three prior deliverables ranked against them, each with its complete outline and as much whole section text as fits in 45,000 bytes, plus reviewer lessons, curator notes and drafting guidance. Each section reports how many independent version families share its recurring text, so the agent can tell established house text from a clause one project used, adopt the team's structure and table columns, and take every project fact from the new project's own inputs.

## End-to-end trace of one PDF

Consider a curated marine-survey folder containing `AGENTS.md` at the root and `p101-harbour-lidar/harbour-survey-memorandum.pdf` (a 14-page PDF with multi-page deliverable tables and a raster rainfall chart on page 5 whose legend value `330.9 mm` exists only inside the chart image).

### 1. Build (`arkive-rag build`)

```sh
uv run --locked arkive-rag build path/to/corpus -o bathymetry.arkive.db --cache build-cache.db
```

`compile_corpus` in `src/arkive_rag/compiler.py` runs three passes.

**Reading** (`read_document`, one document at a time in a thread pool) produces a `DocumentReading` that depends only on the file's bytes:

1. **Discovery and lineage (`discover_corpus`):** walks the corpus in sorted path order, rejects symlinks and unsupported file types before calling OpenAI, and reads the source bytes once to compute the file's SHA-256 hash. (For `.docx` and `.pptx` files, `convert_office_to_pdf` first converts those bytes to PDF in an isolated headless LibreOffice directory so Word checkboxes, floating textboxes, and slide layouts follow the same page pipeline.)
2. **Page rendering (`render_pdf`):** `pypdfium2` extracts each page's Unicode native text layer and renders a 2x WebP image (`webp_bytes`).
3. **Structured transcription and fidelity check (`transcribe_with_fidelity`):** the model receives the page image, native text and the preceding page's block tail, and returns ordered blocks and visual transcripts. The chart value `330.9 mm` can enter a `VisualTranscript`. The compiler checks fidelity anchors and retries when required. Page N receives the last 1,200 characters of page N-1's transcribed blocks, excluding visual transcripts; a changed tail causes the next page to be re-transcribed, and propagation stops when a tail is unchanged.
4. **Cross-page section and table stitching (`assemble_sections`):** headings open sections with their full heading path and page span. When a paragraph or Markdown table continues across a page break (`continues_from_previous_page=True`), `assemble_sections` joins it onto the previous part and updates the page span; when a table row was split mid-cell across pages (`merges_into_previous_row=True`), the split cells are joined column by column, and each row keeps the pages it came from.
5. **Chunk pieces (`section_pieces`):** each section stays intact for reading and is split into prose, table and visual pieces for search. Curator notes produce guidance pieces.
6. **Observation card (`ObserveSections`):** the model records the memorandum's title, date, deliverable type, summary, attributes, terms and craft observations. Term and craft observations carry quotes that the compiler locates in the source. The root `AGENTS.md` gets its card from its own text, without a model call.

The [reading architecture](architecture.md#reading) describes the fidelity, chunking and quote checks.

**Induction** (`induce` in `src/arkive_rag/induction.py`) runs once for the whole corpus, after each document gets its ID, a hash of its path (`document_id`, such as `doc_a1b2c3d4e5`), with documents kept in sorted path order:

1. **Organise (`OrganiseCorpus`, then `CanonicaliseTypes`):** the model reads the folders, curator notes and a one-line digest of each document, in batches of at most 60,000 characters, so every prompt stays bounded on a corpus of any size. It might propose a path-bound `project` facet for the survey folders and a content-bound `client` facet, and gives each document a deliverable type, such as `technical memorandum`; a review is marked feedback for the types it reviews and keeps its own type. One canonicalisation request maps every proposed type label to one canonical type, without any cap on how many types there are.
2. **Bind:** the memorandum gets `project: p101-harbour-lidar` from its folder. The Decisions API judges content-bound values against the memorandum's card and outline.
3. **Version families:** code proposes pairs of documents that share most of their text, or most title and file-name words within one type; the Decisions API confirms the pairs that are drafts, revisions or exports of the same deliverable, and dates (or the Decisions API, when dates cannot decide) mark one member `latest` and the rest `superseded`, or all `unordered` when nothing orders them.
4. **Feedback lessons (`ExtractFeedbackLessons`):** each feedback document yields transferable reviewer criticisms with verbatim quotes; the compiler locates each quote, and the Decisions API confirms the lessons of each section in one request.
5. **Glossary (`WriteGlossary`):** term observations and occurrences produce cited senses. A null definition means no definition was established from that sense's cited evidence: the generator supplied none, or the support check demoted it. It does not prove the corpus never defines the term. The build report's `unsupported_definitions` counts proposed definitions demoted by the support check, not every null sense.
6. **Recurring units (`recurring_units`):** Python finds the paragraphs, list items and table header rows that recur in another document, through an inverted index of word 4-grams, and records every section that holds each one.
7. **Topic guide (`WriteGuide`):** a cited Markdown overview of the collection from a bounded catalog of the documents, the facets, the deliverable types, glossary terms and curator notes; Python verifies that every `[doc_a1b2c3d4e5:doc_a1b2c3d4e5_sec_001]` citation resolves to a citable section of the document it names.

**Binding** (`bind_document`) gives sections their IDs (`doc_a1b2c3d4e5_sec_001`, ...) and chunks their IDs and breadcrumbs (`[project: p101-harbour-lidar | client: Northwind Marine | Doc: Harbour Survey Memorandum (pp. 4-6) | Section: 3 Rainfall]`). Finally, `text-embedding-3-large` (1,536 dimensions) embeds `breadcrumb + "\n" + content` in batches and L2-normalises each vector.

The [induction architecture](architecture.md#induction) describes validation, support checks, version ordering and recurrence.

With `--cache`, provider responses are stored under a hash of their inputs. Rebuilding an unchanged corpus makes no provider calls; adding a file reads only that file again and requests induction over the changed corpus. Identical requests can use cached responses.

### 2. Snapshot contents and model calls

`write_snapshot` in `src/arkive_rag/snapshot.py` writes a temporary sibling SQLite file (mode `0600`) in a single transaction, checks foreign keys and database integrity, and atomically renames it into place (retaining the existing file mode when replacing a previous snapshot). The schema version is `5.0`; older snapshots are rebuilt from their corpus.

| Table | Contents |
| --- | --- |
| `corpus_meta` | `topic`, `title`, `corpus_fingerprint` (SHA-256 over sorted `rel_path` + file `sha256`), `arkive_version`, `embedding_model`, `embedding_dimensions`, `guide_markdown`, `build_settings` (model IDs and the effort of each operation), and `build_report`. |
| `curator_notes` | Verbatim `AGENTS.md` files with their path. |
| `documents` | One row per source file with `sha256`, `format`, `title`, `doc_date`, `doc_date_iso`, `doc_date_basis`, `deliverable_type`, `summary`, `page_count`, `is_curator_note`, the feedback role and the version family. |
| `deliverable_types` | The canonical deliverable types with their descriptions. |
| `facets`, `facet_values`, `facet_aliases`, `document_facets` | The induced facets, their values and aliases, and which document holds which value. |
| `glossary` | One validated JSON document per glossary entry; every citation is checked when the snapshot opens. |
| `feedback_lessons` | Each confirmed reviewer lesson with its source section and the types it reviews. |
| `unit_clusters`, `cluster_sections` | Each recurring text unit and every section that holds it. |
| `sections` | Full Markdown of every stitched section with `heading_path` and `page_start`..`page_end`. |
| `pages` | Per-page `native_text`, `visual_summary`, and `webp_bytes` image for visual inspection. |
| `chunks` | Chunks (`prose`, `table`, `visual`, `guidance`), breadcrumbs, page spans, and `float32` embedding blobs. |
| `chunks_fts` | SQLite FTS5 BM25 index over NFKC-normalised chunk content (weight 10) and breadcrumb (weight 2). |

Model calls are concentrated in `build`:

| Operation | Model | When it runs |
| --- | --- | --- |
| Page transcription | `gpt-6.1-sol`, low effort (high effort on fidelity retry) | Once per rendered page (cached by `--cache`). |
| Observation | `gpt-6.1-sol`, low effort | Once per batch of a citable document's sections, plus one consolidation for a multi-batch document (cached). |
| Organisation, type canonicalisation, glossary, guide | `gpt-6.1-sol`, medium effort | Organisation once per batch of document digests, canonicalisation and the guide once per topic, the glossary once per batch of terms (cached). |
| Feedback lessons | `gpt-6.1-sol`, low effort | Once per batch of a feedback document's sections (cached). |
| Facet binding, version judgements and support checks | `gpt-6-luna` on the Decisions API | Once per citable document for content-bound facets, once per window of candidate version pairs, once per family that dates cannot order, once per glossary definition and once per section holding feedback lessons (cached). |
| Chunk embeddings | `text-embedding-3-large` | Batched once per topic (cached). |
| Query embedding | `text-embedding-3-large` | Once per `search` call during `serve` or CLI `search`. |

`build` prints the provider calls, tokens, and estimated cost it used, and the build report: observations and lessons dropped for an unlocated quote, definitions and lessons the support checks did not confirm, refused decision questions, pruned facet values and facets, and curator notes a prompt could not show in full. With `--max-cost`, no provider call starts once the estimate has reached the limit, though calls already in flight can take the total past it and the estimate ignores the long-context surcharge. `list_topics`, `get_topic_guide`, `prepare_draft`, `list_documents`, `get_document_outline`, `grep`, `read_section`, `view_page`, and `lookup_glossary` read local SQLite data without calling OpenAI; `prepare_draft` embeds the inputs only when the server has a key.

### 3. Serving and querying over MCP (`arkive-rag serve`)

`TopicSnapshot.open` opens each `.arkive.db` file via a percent-encoded read-only SQLite URI (`?mode=ro`), checks that every table is present, that no foreign key dangles and that every facet value has its canonical alias, verifies `corpus_fingerprint`, every glossary citation, the document roles, version families, lessons and recurring units, and loads the facets and the normalised `(N, 1536)` `float32` embedding matrix into memory. `create_mcp_server` in `src/arkive_rag/server.py` exposes ten tools across six levels of detail:

1. **Collection map:** `list_topics` pages through the topics, each with its facets and allowed values and its deliverable types (the first 50 of each, the rest counted); `get_topic_guide` returns the cited guide, curator notes, facet definitions, glossary terms, and each deliverable type with its document and feedback counts; `list_documents` pages through documents with their facets, types and version families, and `get_document_outline` shows a section tree.
2. **Terminology:** `lookup_glossary` lists the glossary's terms, or returns matching entries with cited senses. A `null` definition means none was established from the cited evidence; the corpus may still define the term elsewhere.
3. **Exact string lookup:** `grep` runs a case-insensitive, NFKC- and dash-normalised substring scan across section Markdown for exact codes, numbers, and filenames.
4. **Hybrid chunk search:** `search` validates filters locally, embeds the query once, ranks the top 300 FTS5 BM25 hits (using `build_fts5_query` to quote tokens and adjacent phrases so punctuation or operators never raise FTS5 syntax errors) and the top 300 exact cosine-similarity hits (`matrix @ query_vector`), and merges them with Reciprocal Rank Fusion (`1 / (60 + rank)`). It leaves out a superseded draft behind its latest version and shows at most three chunks of one version family before filling the list.
5. **Drafting packet:** `prepare_draft` ranks the topic's deliverables against the user's inputs, by each document's best three chunks, and returns up to three exemplars with complete outlines and whole sections in priority order (input-matched, recurring, first and last), lessons, curator notes, sections from other documents that hold the inputs' rare words, and drafting guidance, within 45,000 bytes.
6. **Full context and page visuals:** `read_section` returns the complete section Markdown with links to its previous and next sections; `view_page` returns the rendered `image/webp` page alongside its visual summary and native text (cut, with a count of what was left out, when they would not fit one tool result) so a multimodal client model can inspect charts, maps, or layouts directly.

`list_documents`, `search`, `grep` and `prepare_draft` take `filters`, for example `{"project": "p101-harbour-lidar", "client": "Northwind"}`: each key is a facet, each value a canonical value or alias, and the keys combine with AND. An unknown facet or value returns an error naming the valid ones. The CLI's `search` and `grep` take the same filters as repeated `--filter FACET=VALUE`.

During the MCP initialisation handshake, `INSTRUCTIONS` tells the client to call `list_topics` when topics or facets are unknown, to answer questions with the read tools, citing `doc_id`, `section_id` and pages and saying so when the corpus does not establish a fact, and to draft by calling `prepare_draft` with the user's full inputs and following the packet's guidance. The guidance takes every fact of the new work only from the user's inputs and treats dates, version families and recurrence counts as provenance: text shared by several independent families is established house text, while a clause from one family may be project-specific, so the agent weighs scope and modality and, in an interactive session, asks when an institutional policy is genuinely uncertain.

### 4. Proving value over raw files (`python -m evals.run`)

`evals/` tests whether a compiled snapshot outperforms giving the same model direct file tools over the raw corpus directory. Benchmark data lives in an external `*.eval.json` pack (`evals/schema.py`, format `arkive-eval-pack/3`) that can optionally pin the snapshot's `corpus_fingerprint`, keeping private documents and facts out of Git. Before any output or model call, the runner checks the pack, the snapshot, the pinned fingerprint, the retrieval filters and the holdout guards for every layer it will run:

- **Layer 1 (`evals/ablation.py`):** compares `bm25_only`, `dense_only`, and `hybrid_rrf` from the same query embedding across exact identifiers, conceptual paraphrases, visual-only chart/map questions, repaired font ligatures, and disambiguation cases that set facet `filters` (`Completion@1/3/5` and `CompletionMRR@5`).
- **Layer 2 (`evals/agents.py`):** runs the same `gpt-6.1-sol` agent conversation (OpenAI Agents SDK, through `OpenAIGateway.converse`) on questions of several kinds under two tool sets (raw file tools `list_corpus_files`, `grep_raw_text`, `read_raw_pages`, `view_raw_page_image` versus the ten Arkive MCP tools), then grades completed answers with phrase coverage and a blinded `gpt-6.1-sol` judge that sees the reference facts and the snapshot's text of the anchored pages and lists each claim as supported, contradicted or unverifiable, while recording token usage, each tool call, and latency.
- **Layer 3 (`evals/drafting.py`):** for each holdout, builds `holdout.arkive.db` from the selected files after guards exclude files whose path holds a held-out marker and files identical to the reference. The guards infer nothing from folders and do not scan content; the pack author must exclude changed copies and documents that restate held-out facts. In each of `--drafts` rounds, three `gpt-6.1-sol` arms always draft from project run notes: no tools, chunk search, and the Arkive MCP tools. A fourth, with the pack's named exemplar documents placed in the prompt, runs when the holdout names any. Completed drafts receive rubric checks, two lexical-overlap measures over their prose (how much of the reference's house text a draft reproduces, and how much text it copies from included documents that the reference does not carry) and one blinded comparison with the reference per round. The [holdout procedure](architecture.md#evaluation) describes the guards, arms and measures.
