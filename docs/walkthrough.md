# Product goal and system walkthrough

Arkive RAG turns a human-curated folder of reference documents into a single SQLite file (`*.arkive.db`) that an MCP-capable agent can query and read.

## Curating a topic without data-preparation overhead

Domain leads, engineers, and marketing managers already hold examples of strong work: final technical reports, slide decks, proposals, reviewer feedback tables, and campaign briefs. Requiring them to clean text layers, design chunking rules, or label documents before building a knowledge base blocks adoption.

Arkive keeps curation in folders and prose:

1. A person collects the reference documents that represent good work on a specific topic (typically 10 to 200 `.pdf`, `.docx`, `.pptx`, or `.md` files) and arranges them in a directory, in whatever folder structure suits the collection: one folder per project, per campaign, per author, or none.
2. Optionally, they add `AGENTS.md` files in plain language explaining how the collection fits together, how it should be filtered, what is still missing, or how later work should use those examples.
3. Running `arkive-rag build` uses OpenAI vision, text, decision, and embedding models offline to transcribe pages, repair broken PDF font encodings, extract chart and map text, stitch tables across page breaks, observe each document, and induce from the whole collection the facets it can be filtered by, a cited glossary, and a blueprint for each recurring deliverable type. It writes one read-only `.arkive.db` file.
4. `arkive-rag serve` exposes one or several `.arkive.db` files over MCP. An agent can mount multiple topic snapshots in one session (for example, a technical survey collection alongside an executive presentation or marketing collection) without scanning dozens of raw source files on every turn.

## Two ways agents use a snapshot

A curated snapshot supports two distinct agent workflows through the same MCP surface:

- **Pinpoint factual and visual answers:** finding exact numbers, filenames, parameter tables, reviewer answers, or chart legend values across the collection, narrowing by facet (client, project, campaign, author) when the question names one, and citing the source `doc_id`, `section_id`, and page numbers.
- **Deliverable authoring aligned to house craft:** starting from the blueprint for the deliverable type (its section outline, conventions, and reviewer pitfalls), then reading the cited exemplar sections, the reviewer feedback documents, and the glossary, so a newly drafted report, deck, or campaign brief adopts the team's structure, methodological framing, caveats, and standard disclaimers while grounding every fact in the new project's own inputs.

## End-to-end trace of one PDF

Consider a curated marine-survey folder containing `AGENTS.md` at the root and `p101-harbour-lidar/harbour-survey-memorandum.pdf` (a 14-page PDF with multi-page deliverable tables and a raster rainfall chart on page 5 whose legend value `330.9 mm` exists only inside the chart image).

### 1. Offline build (`arkive-rag build`)

```sh
uv run --locked arkive-rag build path/to/corpus -o bathymetry.arkive.db --cache build-cache.db
```

`compile_corpus` in `src/arkive_rag/compiler.py` runs three passes.

**Reading** (`read_document`, one document at a time in a thread pool) produces a `DocumentReading` that depends only on the file's bytes:

1. **Discovery and lineage (`discover_corpus`):** walks the corpus in sorted path order, rejects symlinks and unsupported file types before calling OpenAI, and reads the source bytes once to compute the file's SHA-256 hash. (For `.docx` and `.pptx` files, `convert_office_to_pdf` first converts those bytes to PDF in an isolated headless LibreOffice directory so Word checkboxes, floating textboxes, and slide layouts follow the same page pipeline.)
2. **Page rendering (`render_pdf`):** `pypdfium2` extracts each page's Unicode native text layer and renders a 2x WebP image (`webp_bytes`).
3. **Structured transcription and fidelity check (`transcribe_with_fidelity`):** for each page, the compiler sends a `PageTranscription` request through `OpenAIGateway.run`: the page's WebP image, its native text, and the tail of the previous page's text go to `gpt-6.1-sol` at low reasoning effort. The model returns ordered blocks (`heading`, `paragraph`, `table`, `callout`) plus `visuals` transcripts for charts, maps, and diagrams. Reading from the rendered image repairs broken PDF glyph mappings in prose while capturing image-only values such as `330.9 mm` in a `VisualTranscript`. Python then strips repeated running headers, footers, and page numbers from the native text, collects GIS/data filenames plus formatted and 3+-digit numbers as fidelity anchors, and checks recall against the emitted page Markdown: every filename must survive and combined filename-and-number recall must reach at least 0.95 (1.0 when only one or two anchors exist). If that check fails, Python retries once at high effort naming the missing anchors, after which any missing filename or combined recall below 0.85 raises `TranscriptionFidelityError`. Because each page's request carries the previous page's transcript tail, editing page 3 re-transcribes pages 3 to 14 on the next build.
4. **Cross-page section and table stitching (`assemble_sections`):** headings open sections with their full heading path and page span. When a paragraph or Markdown table continues across a page break (`continues_from_previous_page=True`), `assemble_sections` joins it onto the previous part and updates the page span; when a table row was split mid-cell across pages (`merges_into_previous_row=True`), `merge_tables` joins the split cells column by column.
5. **Chunk pieces (`section_pieces`):** each section is kept intact for reading, and split into pieces for search (capped at 6,000 characters):
   - `prose` pieces packed up to 1,800 characters;
   - `table` pieces kept whole when up to 2,400 characters, otherwise split into ~2,000-character pieces with the Markdown header and separator rows repeated on every piece (wrapping oversized rows when needed to stay within the 6,000-character ceiling);
   - `visual` pieces for each chart, map, or figure transcript;
   - `guidance` pieces for `AGENTS.md` curator notes.
6. **Observation card (`ObserveSections`):** `gpt-6.1-sol` at low effort reads the memorandum's sections in batches of at most 24,000 characters (a longer section is split into pieces) and records its title, date, deliverable type (`survey memorandum`), summary, the attributes it states about itself (for example `client: Northwind Marine`), the terms it uses (such as `LAT`, with whether a quote defines it), and craft observations (for example "report depths in metres below the stated datum"). Each term and craft observation carries a verbatim quote, which Python locates in the batch's sections, ignoring case, dashes, whitespace, emphasis, table pipes, block-quote markers and line-break hyphenation. A quote that cannot be found gets one rerun; if it is still missing, the observation is dropped and counted in the build report. The root `AGENTS.md` gets its card from its own text, without a model call.

**Induction** (`induce` in `src/arkive_rag/induction.py`) runs once for the whole corpus, after documents are numbered in sorted path order (`doc_001` for `AGENTS.md`, `doc_002` for `harbour-survey-memorandum.pdf`):

1. **Organise (`OrganiseCorpus`):** `gpt-6.1-sol` at medium effort reads the folder tree, the curator notes, and every document's card, and proposes up to six facets and up to eight recurring deliverable types. Here it might propose a path-bound `project` facet whose values stand for the `p101-harbour-lidar` and `p102-estuary-multibeam` folders, and a content-bound `client` facet. The answer is validated by the same folded form that filters use (no unknown folders or document IDs, no facet mixing folder-bound and content-bound values, no value given twice, no alias naming two values; repeated or blank aliases of one value are simply dropped) and rerun once with the problems named if it fails.
2. **Bind:** the memorandum gets `project: p101-harbour-lidar` from its folder. For `client`, the Decisions API (`gpt-6-luna`) answers one predicate per value over the memorandum's card and outline; a probability of 0.5 or more assigns the value. Values no document holds, and facets left with fewer than two populated values, are pruned.
3. **Glossary (`WriteGlossary`):** term observations are merged by term and alias, and the terms seen in two or more sections or defined somewhere are sent with their quotes and further occurrence snippets. Each entry gets one sense per meaning, each with section citations and a definition, or `null` when the corpus uses the term without defining it. The Decisions API then checks each definition against its cited sections; an unsupported definition becomes `null` and is counted.
4. **Blueprints (`WriteBlueprint`):** for each recurring deliverable type, such as the survey memoranda, an OpenAI Agents SDK run reads the exemplars and reviewer feedback through read-only `outline`, `read_section`, and `grep` tools (curator notes are not readable), with the craft observations noted while reading those documents listed as leads it must check in the source, and returns the section outline, conventions, and reviewer pitfalls, each citing sections. The Decisions API checks every convention and pitfall against its complete cited sections (long evidence is judged in windows); unsupported ones are dropped and counted.
5. **Topic guide (`WriteGuide`):** a cited Markdown overview of the collection from the cards, facets, glossary terms, blueprint summaries, and curator notes; Python verifies that every `[doc_XXX:doc_XXX_sec_YYY]` citation resolves to a citable section.

**Binding** (`bind_document`) gives sections their IDs (`doc_002_sec_001`, ...) and chunks their IDs and breadcrumbs (`[project: p101-harbour-lidar | client: Northwind Marine | Doc: Harbour Survey Memorandum (pp. 4-6) | Section: 3 Rainfall]`). Finally, `text-embedding-3-large` (1,536 dimensions) embeds `breadcrumb + "\n" + content` in batches and L2-normalises each vector.

With `--cache`, every provider response is stored under a hash of its inputs. Rebuilding an unchanged corpus makes no provider calls; adding a file reads only that file again, while induction reruns over the changed corpus.

### 2. What the `.arkive.db` file stores and what calls OpenAI

`write_snapshot` in `src/arkive_rag/snapshot.py` writes a temporary sibling SQLite file (mode `0600`) in a single transaction, checks foreign keys and database integrity, and atomically renames it into place (retaining the existing file mode when replacing a previous snapshot). The schema version is `3.0`; older snapshots are rebuilt from their corpus.

| Table | Contents |
| --- | --- |
| `corpus_meta` | `topic`, `title`, `corpus_fingerprint` (SHA-256 over sorted `rel_path` + file `sha256`), `arkive_version`, `embedding_model`, `embedding_dimensions`, `guide_markdown`, `build_settings` (model IDs and the effort of each operation), and `build_report`. |
| `curator_notes` | Verbatim `AGENTS.md` files with their path. |
| `documents` | One row per source file with `sha256`, `format`, `title`, `doc_date`, `deliverable_type`, `summary`, `page_count`, and `is_curator_note`. |
| `facets`, `facet_values`, `facet_aliases`, `document_facets` | The induced facets, their values and aliases, and which document holds which value. |
| `glossary`, `blueprints` | One validated JSON document per glossary entry and per blueprint; every citation is checked when the snapshot opens. |
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

`build` prints the provider calls, tokens, and estimated cost it used, and the build report: observations dropped for an unlocated quote, definitions and blueprint statements the support checks did not confirm, refused decision questions, and pruned facet values and facets. With `--max-cost`, no provider call starts once the estimate has reached the limit, though calls already in flight can take the total past it and the estimate ignores the long-context surcharge. `--trace` sends traces of the blueprint agent runs to the OpenAI dashboard. `list_topics`, `get_topic_guide`, `list_documents`, `get_document_outline`, `grep`, `read_section`, `view_page`, and `lookup_glossary` read local SQLite data without calling OpenAI.

### 3. Serving and querying over MCP (`arkive-rag serve`)

`TopicSnapshot.open` opens each `.arkive.db` file via a percent-encoded read-only SQLite URI (`?mode=ro`), checks that every table is present, that no foreign key dangles and that every facet value has its canonical alias, verifies `corpus_fingerprint` and every glossary and blueprint citation, and loads the facets and the normalised `(N, 1536)` `float32` embedding matrix into memory. `create_mcp_server` in `src/arkive_rag/server.py` exposes nine tools across five levels of detail:

1. **Collection map:** `list_topics` lists each topic's facets and allowed values; `get_topic_guide` returns the cited guide, curator notes, facet definitions, blueprints, and glossary terms; `list_documents` and `get_document_outline` show documents, their facets, and their section trees.
2. **Terminology:** `lookup_glossary` lists the glossary's terms, or returns the entries whose term or alias matches a query, each sense with its citations and a `null` definition where the corpus never defines the term.
3. **Exact string lookup:** `grep` runs a case-insensitive, NFKC- and dash-normalised substring scan across section Markdown for exact codes, numbers, and filenames.
4. **Hybrid chunk search:** `search` validates filters locally, embeds the query once, ranks the top 100 FTS5 BM25 hits (using `build_fts5_query` to quote tokens and adjacent phrases so punctuation or operators never raise FTS5 syntax errors) and the top 100 exact cosine-similarity hits (`matrix @ query_vector`), and merges them with Reciprocal Rank Fusion (`1 / (60 + rank)`).
5. **Full context and page visuals:** `read_section` returns the complete section Markdown with links to its previous and next sections; `view_page` returns the rendered `image/webp` page alongside its visual summary and native text so a multimodal client model can inspect charts, maps, or layouts directly.

`list_documents`, `search`, and `grep` take `filters`, for example `{"project": "p101-harbour-lidar", "client": "Northwind"}`: each key is a facet, each value a canonical value or alias, and the keys combine with AND. An unknown facet or value returns an error naming the valid ones. The CLI's `search` and `grep` take the same filters as repeated `--filter FACET=VALUE`.

During the MCP initialisation handshake, `INSTRUCTIONS` separates corpus Q&A (which cites `doc_id`, `section_id`, and pages) from standalone deliverable drafting (which starts from the blueprint for the deliverable type and adopts structure, caveats, tables, and disclaimers from completed exemplars without copying another project's names, dates, figures, or internal `doc_id`s).

### 4. Proving value over raw files (`python -m evals.run`)

`evals/` tests whether a compiled snapshot outperforms giving the same model direct file tools over the raw corpus directory. Benchmark data lives in an external `*.eval.json` pack (`evals/schema.py`, format `arkive-eval-pack/2`) that can optionally pin the snapshot's `corpus_fingerprint`, keeping private documents and facts out of Git. Before any output or model call, the runner checks the pack, the snapshot, the pinned fingerprint, the retrieval filters and the holdout guards for every layer it will run:

- **Layer 1 (`evals/ablation.py`):** compares `bm25_only`, `dense_only`, and `hybrid_rrf` from the same query embedding across exact identifiers, conceptual paraphrases, visual-only chart/map questions, repaired font ligatures, and disambiguation cases that set facet `filters` (`Completion@1/3/5` and `CompletionMRR@5`).
- **Layer 2 (`evals/agents.py`):** runs the same `gpt-6.1-sol` agent conversation (OpenAI Agents SDK, through `OpenAIGateway.converse`) on multi-document questions under two tool sets (raw file tools `list_corpus_files`, `grep_raw_text`, `read_raw_pages`, `view_raw_page_image` versus the nine Arkive MCP tools), then grades completed answers with phrase coverage and a blinded `gpt-6.1-sol` judge anchored to expected source documents and pages while recording token usage, tool calls, and latency.
- **Layer 3 (`evals/agents.py`):** holds out one completed deliverable (never a curator note), verifies before copying or calling any model that no included file carries a held-out marker in its path (case- and dash-insensitive), shares the reference's top-level folder (or the corpus root, when the reference sits there), or has the reference's bytes, compiles `holdout.arkive.db` from the included files alone (so its facets, glossary, and blueprints never see the held-out material), and asks three `gpt-6.1-sol` arms (zero-shot, top-5 chunk-only search, and full Arkive MCP) to draft the deliverable from bare project run notes. Completed drafts are scored with deterministic structure/craft/anti-contamination checks and compared side by side in shuffled blind order (`Candidate_1..3`) against the human-written reference deliverable.
