# Product goal and system walkthrough

Arkive RAG turns a human-curated folder of reference documents into a single SQLite file (`*.arkive.db`) that an MCP-capable agent can query and read.

## Curating a topic without data-preparation overhead

Domain leads, engineers, and marketing managers already hold examples of strong work: final technical reports, slide decks, proposals, reviewer feedback tables, and campaign briefs. Requiring them to clean text layers, design chunking rules, or label documents before building a knowledge base blocks adoption.

Arkive keeps curation in folders and prose:

1. A person collects the reference documents that represent good work on a specific topic (typically 10 to 200 `.pdf`, `.docx`, `.pptx`, or `.md` files) and arranges them in a directory.
2. Optionally, they add `AGENTS.md` files in plain language explaining how the collection fits together, what is still missing, or how later work should use those examples.
3. Running `arkive-rag build` uses OpenAI vision, text, and embedding models offline to transcribe pages, repair broken PDF font encodings, extract chart and map text, stitch tables across page breaks, catalogue documents, and write one read-only `.arkive.db` file.
4. `arkive-rag serve` exposes one or several `.arkive.db` files over MCP. An agent can mount multiple topic snapshots in one session (for example, a technical survey collection alongside an executive presentation or marketing collection) without scanning dozens of raw source files on every turn.

## Two ways agents use a snapshot

A curated snapshot supports two distinct agent workflows through the same MCP surface:

- **Pinpoint factual and visual answers:** finding exact numbers, filenames, parameter tables, reviewer answers, or chart legend values across the collection and citing the source `doc_id`, `section_id`, and page numbers.
- **Deliverable authoring aligned to house craft:** reading the topic guide, reviewer feedback documents, document outlines, and full sections of completed deliverables so a newly drafted report, deck, or campaign brief adopts the team's structure, methodological framing, caveats, and standard disclaimers while grounding every fact in the new project's own inputs.

## End-to-end trace of one PDF

Consider a curated marine-survey folder containing `AGENTS.md` at the root and `p101-harbour-lidar/harbour-survey-memorandum.pdf` (a 14-page PDF with multi-page deliverable tables and a raster rainfall chart on page 5 whose legend value `330.9 mm` exists only inside the chart image).

### 1. Offline build (`arkive-rag build`)

```sh
uv run --locked arkive-rag build path/to/corpus -o bathymetry.arkive.db --cache build-cache.db
```

`compile_corpus` in `src/arkive_rag/compiler.py` runs six steps:

1. **Discovery and lineage (`discover_corpus`):** walks the corpus in sorted path order, rejects symlinks and unsupported file types before calling OpenAI, assigns deterministic IDs (`doc_001` for `AGENTS.md`, `doc_002` for `harbour-survey-memorandum.pdf`), records `project_id` from the top-level subfolder (`p101-harbour-lidar`), and reads the source bytes once to compute the file's SHA-256 hash. (For `.docx` and `.pptx` files, `convert_office_to_pdf` first converts those bytes to PDF in an isolated headless LibreOffice directory so Word checkboxes, floating textboxes, and slide layouts follow the same page pipeline.)
2. **Page rendering (`render_pdf`):** `pypdfium2` extracts each page's Unicode native text layer and renders a 2x WebP image (`webp_bytes`).
3. **Structured transcription and fidelity check (`transcribe_with_fidelity`):** for each page, `OpenAIGateway.transcribe_page` sends the page's WebP image, its native text, and the tail of the previous page's text to `gpt-5.4-mini` (cached in `BuildCache` by SHA-256 of model, prompt, schema, and image bytes). The model returns ordered blocks (`heading`, `paragraph`, `table`, `callout`) plus `visuals` transcripts for charts, maps, and diagrams. Reading from the rendered image repairs broken PDF glyph mappings in prose while capturing image-only values such as `330.9 mm` in a `VisualTranscript`. Python then strips repeated running headers, footers, and page numbers from the native text, collects GIS/data filenames plus formatted and 3+-digit numbers as fidelity anchors, and checks recall against the emitted page Markdown: every filename must survive and combined filename-and-number recall must reach at least 0.95 (1.0 when only one or two anchors exist). If that check fails, Python retries once with `gpt-5.4` naming the missing anchors, after which any missing filename or combined recall below 0.85 raises `TranscriptionFidelityError`.
4. **Cross-page section and table stitching (`assemble_sections`):** headings open `Section` records (`doc_002_sec_001`, `doc_002_sec_002`, ...) with their full heading path and page span. When a paragraph or Markdown table continues across a page break (`continues_from_previous_page=True`), `assemble_sections` joins it onto the previous part and updates the page span; when a table row was split mid-cell across pages (`merges_into_previous_row=True`), `merge_tables` joins the split cells column by column.
5. **Document catalogue and hierarchical chunking (`extract_document_meta`, `build_chunks`):** one `gpt-5.4-mini` call reads the document's headings and opening excerpt to extract `title`, `client`, lifecycle `stage`, `doc_date`, and `summary` (`AGENTS.md` curator notes skip this call and become `guidance` documents directly). `build_chunks` keeps each parent `Section` intact for reading while creating child `Chunk` rows (capped at 6,000 characters) for search:
   - `prose` chunks packed up to 1,800 characters;
   - `table` chunks kept whole when up to 2,400 characters, otherwise split into ~2,000-character pieces with the Markdown header and separator rows repeated on every piece (wrapping oversized rows when needed to stay within the 6,000-character ceiling);
   - `visual` chunks for each chart, map, or figure transcript;
   - `guidance` chunks for `AGENTS.md` curator notes.
   Every chunk carries a contextual breadcrumb (`[Project | Client | Stage | Doc (pp. X-Y) | Section]`).
6. **Cited topic guide and embeddings (`synthesize_corpus`, `embed_texts`):** `gpt-5.4` writes a cited overview of the collection from the document catalogue and curator notes; Python verifies that every `[doc_XXX:doc_XXX_sec_YYY]` citation resolves to a compiled section. Finally, `text-embedding-3-large` (1,536 dimensions) embeds `breadcrumb + "\n" + content` in batches and L2-normalises each vector.

### 2. What the `.arkive.db` file stores and what calls OpenAI

`write_snapshot` in `src/arkive_rag/snapshot.py` writes a temporary sibling SQLite file (mode `0600`) in a single transaction, checks foreign keys and database integrity, and atomically renames it into place (retaining the existing file mode when replacing a previous snapshot):

| Table | Contents |
| --- | --- |
| `corpus_meta` | `topic`, `title`, `corpus_fingerprint` (SHA-256 over sorted `rel_path` + file `sha256`), `arkive_version`, `embedding_model`, `embedding_dimensions`, and `synthesis_markdown`. |
| `curator_notes` | Verbatim `AGENTS.md` files with their directory scope. |
| `documents` | One row per source file with `sha256`, `format`, `project_id`, `client`, `stage`, `title`, `doc_date`, `summary`, and `page_count`. |
| `sections` | Full Markdown of every stitched parent section with `heading_path` and `page_start`..`page_end`. |
| `pages` | Per-page `native_text`, `visual_summary`, and `webp_bytes` image for visual inspection. |
| `chunks` | Child chunks (`prose`, `table`, `visual`, `guidance`), breadcrumbs, page spans, and `float32` embedding blobs. |
| `chunks_fts` | SQLite FTS5 BM25 index over NFKC-normalised chunk content (weight 10) and breadcrumb (weight 2). |

Model calls are concentrated in `build`:

| Operation | Model | When it runs |
| --- | --- | --- |
| Page transcription | `gpt-5.4-mini` (`gpt-5.4` on fidelity retry) | Once per rendered page during `build` (cached by `--cache`). |
| Document catalogue | `gpt-5.4-mini` | Once per non-curator document during `build` (cached). |
| Topic guide synthesis | `gpt-5.4` | Once per topic during `build` (cached). |
| Chunk embeddings | `text-embedding-3-large` | Batched once per topic during `build` (cached). |
| Query embedding | `text-embedding-3-large` | Once per `search` call during `serve` or CLI `search`. |

`list_topics`, `get_topic_guide`, `list_documents`, `get_document_outline`, `grep`, `read_section`, and `view_page` read local SQLite data without calling OpenAI.

### 3. Serving and querying over MCP (`arkive-rag serve`)

`TopicSnapshot.open` opens each `.arkive.db` file via a percent-encoded read-only SQLite URI (`?mode=ro`), verifies `corpus_fingerprint`, and loads the normalised `(N, 1536)` `float32` embedding matrix into memory. `create_mcp_server` in `src/arkive_rag/server.py` exposes eight tools across four levels of detail:

1. **Collection map:** `list_topics`, `get_topic_guide`, `list_documents`, and `get_document_outline` show what projects, lifecycle stages, curator notes, and section trees exist.
2. **Exact string lookup:** `grep` runs a case-insensitive, NFKC- and dash-normalised substring scan across section Markdown for exact codes, numbers, and filenames.
3. **Hybrid chunk search:** `search` validates filters locally, embeds the query once, ranks the top 100 FTS5 BM25 hits (using `build_fts5_query` to quote tokens and adjacent phrases so punctuation or operators never raise FTS5 syntax errors) and the top 100 exact cosine-similarity hits (`matrix @ query_vector`), and merges them with Reciprocal Rank Fusion (`1 / (60 + rank)`).
4. **Full context and page visuals:** `read_section` returns the complete parent section Markdown with links to its previous and next sections; `view_page` returns the rendered `image/webp` page alongside its visual summary and native text so a multimodal client model can inspect charts, maps, or layouts directly.

During the MCP initialisation handshake, `INSTRUCTIONS` separates corpus Q&A (which cites `doc_id`, `section_id`, and pages) from standalone deliverable drafting (which adopts structure, caveats, tables, and disclaimers from completed exemplars without copying another project's dates, recipients, or internal `doc_id`s).

### 4. Proving value over raw files (`python -m evals.run`)

`evals/` tests whether a compiled snapshot outperforms giving the same model direct file tools over the raw corpus directory. Benchmark data lives in an external `*.eval.json` pack (`evals/schema.py`) that can optionally pin the snapshot's `corpus_fingerprint` (checked before running when set), keeping private documents and facts out of Git:

- **Layer 1 (`evals/ablation.py`):** compares `bm25_only`, `dense_only`, and `hybrid_rrf` from the same query embedding across exact identifiers, conceptual paraphrases, visual-only chart/map questions, repaired font ligatures, and cross-project disambiguation (`Completion@1/3/5` and `CompletionMRR@5`).
- **Layer 2 (`evals/agents.py`):** runs the same `gpt-5.4` tool loop on multi-document questions under two tool sets (raw file tools `list_corpus_files`, `grep_raw_text`, `read_raw_pages`, `view_raw_page_image` versus the eight Arkive MCP tools), then grades completed answers with phrase coverage and a blinded `gpt-5.4` judge anchored to expected source documents and pages while recording token usage, tool calls, and latency.
- **Layer 3 (`evals/agents.py`):** holds out one completed project deliverable, verifies before copying or calling any model that zero files or hashes from the held-out project enter `holdout.arkive.db`, and asks three `gpt-5.4` arms (zero-shot, top-5 chunk-only search, and full Arkive MCP) to draft the deliverable from bare project run notes. Completed drafts are scored with deterministic structure/craft/anti-contamination checks and compared side by side in shuffled blind order (`Candidate_1..3`) against the human-written reference deliverable.
