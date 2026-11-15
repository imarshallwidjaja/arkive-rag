# Product goal and system walkthrough

Arkive RAG turns a human-curated folder of reference documents into a single SQLite file (`*.arkive.db`) that an MCP-capable agent can query and read.

## Curating a topic without data-preparation overhead

Domain leads, engineers, and marketing managers already hold examples of strong work: final technical reports, slide decks, proposals, reviewer feedback tables, and campaign briefs. Requiring them to clean text layers, design chunking rules, or label documents before building a knowledge base blocks adoption.

Arkive keeps curation in folders and prose:

1. A person collects the reference documents that represent good work on a specific topic (typically 10 to 200 `.pdf`, `.docx`, `.pptx`, or `.md` files) and arranges them in a directory, in whatever folder structure suits the collection: one folder per project, per campaign, per author, or none.
2. Optionally, they add `AGENTS.md` files in plain language explaining how the collection fits together, how it should be filtered, what is still missing, or how later work should use those examples.
3. Running `arkive-rag build` calls OpenAI vision, text, decision, and embedding models at build time to transcribe pages, repair broken PDF font encodings, extract chart and map text, stitch tables across page breaks, observe each document, and induce from the whole collection the facets it can be filtered by, a cited glossary, and a blueprint for each recurring deliverable type. It writes one read-only `.arkive.db` file.
4. `arkive-rag serve` exposes one or several `.arkive.db` files over MCP. An agent can mount multiple topic snapshots in one session (for example, a technical survey collection alongside an executive presentation or marketing collection) without scanning dozens of raw source files on every turn.

## Two ways agents use a snapshot

A curated snapshot supports two distinct agent workflows through the same MCP surface:

- **Pinpoint factual and visual answers:** finding exact numbers, filenames, parameter tables, reviewer answers, or chart legend values across the collection, narrowing by facet (client, project, campaign, author) when the question names one, and citing the source `doc_id`, `section_id`, and page numbers.
- **Deliverable authoring aligned to house craft:** fetching the blueprint for the deliverable type with `get_blueprint` (its section outline, conventions, reviewer pitfalls, and standing passages), then reading cited exemplar sections for their layout, so a newly drafted report, deck, or campaign brief adopts the team's structure and table columns, reproduces its standing house text (method background, references, acknowledgements, disclaimers) word for word, and takes every project fact from the new project's own inputs.

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
4. **Cross-page section and table stitching (`assemble_sections`):** headings open sections with their full heading path and page span. When a paragraph or Markdown table continues across a page break (`continues_from_previous_page=True`), `assemble_sections` joins it onto the previous part and updates the page span; when a table row was split mid-cell across pages (`merges_into_previous_row=True`), `merge_tables` joins the split cells column by column.
5. **Chunk pieces (`section_pieces`):** each section stays intact for reading and is split into prose, table and visual pieces for search. Curator notes produce guidance pieces.
6. **Observation card (`ObserveSections`):** the model records the memorandum's title, date, deliverable type, summary, attributes, terms and craft observations. Term and craft observations carry quotes that the compiler locates in the source. The root `AGENTS.md` gets its card from its own text, without a model call.

The [reading architecture](architecture.md#reading) describes the fidelity, chunking and quote checks.

**Induction** (`induce` in `src/arkive_rag/induction.py`) runs once for the whole corpus, after documents are numbered in sorted path order (`doc_001` for `AGENTS.md`, `doc_002` for `harbour-survey-memorandum.pdf`):

1. **Organise (`OrganiseCorpus`):** the model reads the folder tree, curator notes and document cards. It might propose a path-bound `project` facet for the survey folders and a content-bound `client` facet.
2. **Bind:** the memorandum gets `project: p101-harbour-lidar` from its folder. The Decisions API judges content-bound values against the memorandum's card and outline.
3. **Glossary (`WriteGlossary`):** term observations and occurrences produce cited senses. A null definition means no definition was established from that sense's cited evidence: the generator supplied none, or the support check demoted it. It does not prove the corpus never defines the term. The build report's `unsupported_definitions` counts proposed definitions demoted by the support check, not every null sense.
4. **Blueprints (`WriteBlueprint`):** for each recurring deliverable type, such as survey memoranda, an agent reads exemplars and reviewer feedback through `outline`, `read_section`, and `grep`. It returns a cited outline, conventions and reviewer pitfalls, and selects standing house text from candidates found in the exemplars. The Decisions API checks conventions and pitfalls against their cited sections.
5. **Topic guide (`WriteGuide`):** a cited Markdown overview of the collection from the cards, facets, glossary terms, blueprint summaries, and curator notes; Python verifies that every `[doc_XXX:doc_XXX_sec_YYY]` citation resolves to a citable section.

**Binding** (`bind_document`) gives sections their IDs (`doc_002_sec_001`, ...) and chunks their IDs and breadcrumbs (`[project: p101-harbour-lidar | client: Northwind Marine | Doc: Harbour Survey Memorandum (pp. 4-6) | Section: 3 Rainfall]`). Finally, `text-embedding-3-large` (1,536 dimensions) embeds `breadcrumb + "\n" + content` in batches and L2-normalises each vector.

The [induction architecture](architecture.md#induction) describes validation, support checks and standing passage selection.

With `--cache`, provider responses are stored under a hash of their inputs. Rebuilding an unchanged corpus makes no provider calls; adding a file reads only that file again and requests induction over the changed corpus. Identical requests can use cached responses.

### 2. Snapshot contents and model calls

`write_snapshot` in `src/arkive_rag/snapshot.py` writes a temporary sibling SQLite file (mode `0600`) in a single transaction, checks foreign keys and database integrity, and atomically renames it into place (retaining the existing file mode when replacing a previous snapshot). The schema version is `3.0`; older snapshots are rebuilt from their corpus.

| Table | Contents |
| --- | --- |
| `corpus_meta` | `topic`, `title`, `corpus_fingerprint` (SHA-256 over sorted `rel_path` + file `sha256`), `arkive_version`, `embedding_model`, `embedding_dimensions`, `guide_markdown`, `build_settings` (model IDs and the effort of each operation), and `build_report`. |
| `curator_notes` | Verbatim `AGENTS.md` files with their path. |
| `documents` | One row per source file with `sha256`, `format`, `title`, `doc_date`, `deliverable_type`, `summary`, `page_count`, and `is_curator_note`. |
| `facets`, `facet_values`, `facet_aliases`, `document_facets` | The induced facets, their values and aliases, and which document holds which value. |
| `glossary`, `blueprints` | One validated JSON document per glossary entry and per blueprint (with its standing passages); every citation is checked when the snapshot opens. |
| `sections` | Full Markdown of every stitched section with `heading_path` and `page_start`..`page_end`. |
| `pages` | Per-page `native_text`, `visual_summary`, and `webp_bytes` image for visual inspection. |
| `chunks` | Chunks (`prose`, `table`, `visual`, `guidance`), breadcrumbs, page spans, and `float32` embedding blobs. |
| `chunks_fts` | SQLite FTS5 BM25 index over NFKC-normalised chunk content (weight 10) and breadcrumb (weight 2). |

Model calls are concentrated in `build`:

| Operation | Model | When it runs |
| --- | --- | --- |
| Page transcription | `gpt-6.1-sol`, low effort (high effort on fidelity retry) | Once per rendered page (cached by `--cache`). |
| Observation | `gpt-6.1-sol`, low effort | Once per batch of a citable document's sections, plus one consolidation for a multi-batch document (cached). |
| Organisation, glossary, guide | `gpt-6.1-sol`, medium effort | Once per topic; the glossary once per batch of terms (cached). |
| Blueprints | `gpt-6.1-sol`, medium effort, Agents SDK run | Once per recurring deliverable type (cached). |
| Facet binding and support checks | `gpt-6-luna` on the Decisions API | Once per citable document for content-bound facets, and once per glossary definition and blueprint statement (cached). |
| Chunk embeddings | `text-embedding-3-large` | Batched once per topic (cached). |
| Query embedding | `text-embedding-3-large` | Once per `search` call during `serve` or CLI `search`. |

`build` prints the provider calls, tokens, and estimated cost it used, and the build report: observations dropped for an unlocated quote, definitions and blueprint statements the support checks did not confirm, refused decision questions, and pruned facet values and facets. With `--max-cost`, no provider call starts once the estimate has reached the limit, though calls already in flight can take the total past it and the estimate ignores the long-context surcharge. `--trace` sends traces of the blueprint agent runs to the OpenAI dashboard. `list_topics`, `get_topic_guide`, `get_blueprint`, `list_documents`, `get_document_outline`, `grep`, `read_section`, `view_page`, and `lookup_glossary` read local SQLite data without calling OpenAI.

### 3. Serving and querying over MCP (`arkive-rag serve`)

`TopicSnapshot.open` opens each `.arkive.db` file via a percent-encoded read-only SQLite URI (`?mode=ro`), checks that every table is present, that no foreign key dangles and that every facet value has its canonical alias, verifies `corpus_fingerprint` and every glossary and blueprint citation, and loads the facets and the normalised `(N, 1536)` `float32` embedding matrix into memory. `create_mcp_server` in `src/arkive_rag/server.py` exposes ten tools across five levels of detail:

1. **Collection map:** `list_topics` lists each topic's facets and allowed values and the deliverable types that have a blueprint; `get_topic_guide` returns the cited guide, curator notes, facet definitions, glossary terms, and a short summary of each blueprint (type, description, number of exemplars); `get_blueprint` returns one blueprint in full, matching the type regardless of case; `list_documents` and `get_document_outline` show documents, their facets, and their section trees.
2. **Terminology:** `lookup_glossary` lists the glossary's terms, or returns matching entries with cited senses. A `null` definition means none was established from the cited evidence; the corpus may still define the term elsewhere.
3. **Exact string lookup:** `grep` runs a case-insensitive, NFKC- and dash-normalised substring scan across section Markdown for exact codes, numbers, and filenames.
4. **Hybrid chunk search:** `search` validates filters locally, embeds the query once, ranks the top 100 FTS5 BM25 hits (using `build_fts5_query` to quote tokens and adjacent phrases so punctuation or operators never raise FTS5 syntax errors) and the top 100 exact cosine-similarity hits (`matrix @ query_vector`), and merges them with Reciprocal Rank Fusion (`1 / (60 + rank)`).
5. **Full context and page visuals:** `read_section` returns the complete section Markdown with links to its previous and next sections; `view_page` returns the rendered `image/webp` page alongside its visual summary and native text so a multimodal client model can inspect charts, maps, or layouts directly.

`list_documents`, `search`, and `grep` take `filters`, for example `{"project": "p101-harbour-lidar", "client": "Northwind"}`: each key is a facet, each value a canonical value or alias, and the keys combine with AND. An unknown facet or value returns an error naming the valid ones. The CLI's `search` and `grep` take the same filters as repeated `--filter FACET=VALUE`.

During the MCP initialisation handshake, `INSTRUCTIONS` separates corpus Q&A (which cites `doc_id`, `section_id`, and pages, and reads the guide only when scope or missing material matters) from standalone deliverable drafting. A draft calls `get_blueprint`, follows its outline, conventions, and pitfalls, mirrors a cited exemplar section's heading levels and table column headers, and reproduces the standing passages with their wording, references, and attributions, without commentary about their origin. Every project-specific fact comes only from the user's inputs, and that rule takes precedence: a standing passage holding a project fact the user did not give is adapted or left out. Names, references, and attributions from the corpus may be reused only from standing passages; every item the user's inputs enumerate is listed in full; and the draft carries no Arkive document or section IDs.

### 4. Proving value over raw files (`python -m evals.run`)

`evals/` tests whether a compiled snapshot outperforms giving the same model direct file tools over the raw corpus directory. Benchmark data lives in an external `*.eval.json` pack (`evals/schema.py`, format `arkive-eval-pack/2`) that can optionally pin the snapshot's `corpus_fingerprint`, keeping private documents and facts out of Git. Before any output or model call, the runner checks the pack, the snapshot, the pinned fingerprint, the retrieval filters and the holdout guards for every layer it will run:

- **Layer 1 (`evals/ablation.py`):** compares `bm25_only`, `dense_only`, and `hybrid_rrf` from the same query embedding across exact identifiers, conceptual paraphrases, visual-only chart/map questions, repaired font ligatures, and disambiguation cases that set facet `filters` (`Completion@1/3/5` and `CompletionMRR@5`).
- **Layer 2 (`evals/agents.py`):** runs the same `gpt-6.1-sol` agent conversation (OpenAI Agents SDK, through `OpenAIGateway.converse`) on multi-document questions under two tool sets (raw file tools `list_corpus_files`, `grep_raw_text`, `read_raw_pages`, `view_raw_page_image` versus the ten Arkive MCP tools), then grades completed answers with phrase coverage and a blinded `gpt-6.1-sol` judge anchored to expected source documents and pages while recording token usage, tool calls, and latency.
- **Layer 3 (`evals/agents.py`):** builds `holdout.arkive.db` from selected files after guards exclude marker paths, the reference's top-level folder (or root files when the reference is there), and files identical to the reference. The guards do not scan content; the curator must exclude changed copies and documents that restate held-out facts. Three `gpt-6.1-sol` arms draft from project run notes; completed drafts receive deterministic checks and a blinded comparison with the reference. The [holdout procedure](architecture.md#evaluation) describes the guards and arms.
