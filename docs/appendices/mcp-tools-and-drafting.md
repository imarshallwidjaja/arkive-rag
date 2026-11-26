# Appendix B: MCP tools, drafting packet, and cost scaling

> Technical appendix. For the plain-language summary, read [How folders become searchable knowledge](../how-it-works.md) and [Data boundaries, security, and cost](../security-privacy-and-cost.md). The authoritative specification is in [Retrieval](../architecture.md#retrieval), [Drafting packet](../architecture.md#drafting-packet), and [MCP server](../architecture.md#mcp-server).

`arkive-rag serve` mounts one or more `.arkive.db` snapshots read-only and exposes eleven MCP tools over `stdio` (default) or `streamable-http`. The server makes no generative model calls. It serves local SQLite data, bounded Markdown and WebP payloads, and, when `OPENAI_API_KEY` is set, one query embedding per `search` or `prepare_draft` call.

![Two bounded runtime paths: pinpoint Q&A and the 45 KB prepare_draft packet](../assets/mcp-serving-and-drafting.svg)

[Open the interactive diagram](../assets/mcp-serving-and-drafting.html)

## Startup checks and transport safety

`TopicSnapshot.open` opens each snapshot through a percent-encoded read-only SQLite URI (`?mode=ro`) and verifies that:

- `SCHEMA_VERSION == "7.0"` and every required table is present;
- no foreign key dangles and every facet value has its canonical alias;
- `corpus_fingerprint`, every glossary citation, the document roles, version families, feedback lessons, and recurring clusters pass validation;
- every stored document attribute cites a section of its own document, and every stored table is exactly a table of its section's Markdown and matches its rows and its `table_rows_fts` index;
- the normalised `(N, 1536)` `float32` chunk embedding matrix loads into memory.

`create_mcp_server` in `src/arkive_rag/server.py` registers the tools. With `OPENAI_API_KEY` set, all served snapshots must share one embedding model and dimension, or `serve` refuses to start.

The server starts without `OPENAI_API_KEY`. Ten tools then work unchanged, `prepare_draft` ranks with BM25 alone, and `search` reports that it needs a key (the CLI `grep` is the offline alternative for exact matches). With `--transport streamable-http`, the server binds to `127.0.0.1:8000` by default. It refuses a host outside `LOOPBACK_HOSTS` (`127.0.0.1`, `localhost`, `::1`) unless `--allow-nonlocal` is passed, because the HTTP transport has no built-in authentication.

## The eleven tools and their bounds

Every tool returns one compact JSON text block of at most 45,000 bytes and 1,600 lines (`RESPONSE_MAX_BYTES`, `RESPONSE_MAX_LINES` in `snapshot.py`). That limit sits below the size at which MCP clients such as OpenCode truncate tool results, so an agent never silently loses the end of a search result, outline, or drafting packet.

During the MCP initialisation handshake, `INSTRUCTIONS` tells the client to call `list_topics` when topics or facets are unknown, to answer questions with the read tools (and `query_tables` for values in tables) while citing `doc_id`, `section_id`, and pages and saying so when the corpus does not establish a fact, to repeat a call with `versions="all"` when `excluded_superseded_count` is above zero and the user asks about earlier drafts, and to draft by calling `prepare_draft` with the user's full inputs. When the server serves a single topic, `server_instructions` appends the topic slug, its facet names, and up to 20 deliverable types, so the agent can skip a first `list_topics` call.

| Tool | Purpose | Defaults and bounds |
| --- | --- | --- |
| `list_topics` | List served topics with their facets, allowed values, deliverable types, and how many citable documents hold each facet (`document_count` of `citable_document_count`). | First 50 values and types per topic inline, the rest counted; paginated. |
| `get_topic_guide` | Return the cited guide, curator notes, facet definitions with their document coverage, glossary terms, and each deliverable type with its document and feedback counts. | Within 45 KB; omitted items are reported. |
| `list_documents` | Page through documents with dates, deliverable types, version families, verified attributes, and bound facets. | `limit` 50 by default, at most 100 (`MAX_DOCUMENT_PAGE`), fewer when more would not fit; summaries cut to 240 characters (`LIST_SUMMARY_CHARS`); at most 12 attributes per document (`LIST_ATTRIBUTES`) within 4,000 bytes, names cut to 120 characters and values to 300. |
| `get_document_outline` | Return a document's metadata and attributes and its section tree (`section_id`, `heading_path`, page span, character count). | Paginated with `offset`; attributes within 8,000 bytes. |
| `lookup_glossary` | List glossary terms, or return matching entries with cited senses. A `null` definition means none was established from the cited evidence. | Query at most 1,000 characters; paginated within 45 KB. |
| `grep` | Case-insensitive, NFKC- and dash-normalised substring scan of section Markdown for exact codes, filenames, and numbers. A match in a table row shows the table's header row first. | `max_matches` 12 by default, at most 50; `truncated` says whether more lines match. |
| `query_tables` | Rows of the Markdown tables stored at build time: `columns` keeps tables whose headers contain every given name and shows those columns plus the row labels; `query` matches header and cells together and shows the rows whose cells match, or the leading rows of a table whose header alone matches (a table with a header and no rows included); a query with no words matches nothing. | `limit` 20 by default, at most 50; at most 24 columns and 20 rows per table within 12,000 bytes, cells cut to 300 characters, the rest counted. |
| `search` | Hybrid chunk search (below). | `top_k` 5 by default, at most 50; at most 3 chunks per version family before backfilling. |
| `read_section` | Return full section Markdown with `prev_section_id` and `next_section_id` links. | Paginated with `offset`. |
| `view_page` | Return the rendered `image/webp` page with its visual summary and native text, so a multimodal model can inspect charts, maps, and layouts. | Text fields cut, with counts of what was left out, to fit 45 KB. |
| `prepare_draft` | Assemble one drafting packet for a new deliverable from the user's inputs. | 45,000-byte budget; up to 3 exemplars with outlines, inline sections, lessons, notes, and provenance. |

`list_documents`, `search`, `grep`, `query_tables`, and `prepare_draft` accept `filters`, for example `{"job_cohort": "archive", "sensing_technology": "fibre_bragg_grating"}` on the measured `structural-monitoring-consultancy` build (facet names are induced per build, so another build may name them differently). Each key is a facet and each value a canonical value or alias. Keys combine with AND. An unknown facet or value returns a structured error that lists the valid options, so the agent can recover in one turn. The CLI `search` and `grep` take the same filters as repeated `--filter FACET=VALUE`.

`list_documents`, `search`, `grep`, and `query_tables` also accept `deliverable_type` (a canonical type, matched ignoring case) and `versions`. With `versions="current"`, the default, a document superseded by another document in scope is left out and counted in the result's `excluded_superseded_count`; `versions="all"` keeps every version. Every document, hit, match, and table carries the same version fields: `deliverable_type`, `doc_date`, `family_id`, `version_status`, `superseded_by`, and `superseded_doc_ids`. `prepare_draft` always drafts from current versions. The CLI `search` and `grep` commands take `--type` and `--versions`.

### Hybrid search

`search` validates filters locally and embeds the query once. It then ranks the top 300 FTS5 BM25 hits and the top 300 exact cosine-similarity hits (`matrix @ query_vector`) and merges them with Reciprocal Rank Fusion, `1 / (60 + rank)` (`RRF_K = 60`). `build_fts5_query` quotes tokens and adjacent phrases, so punctuation and operators in a query never raise FTS5 syntax errors. The result leaves out a superseded draft behind its latest version and shows at most three chunks of one version family before filling the list.

## Inside `prepare_draft`

`prepare_draft(topic, project_inputs, deliverable_type, filters)` selects and packs reference material at request time.

1. **Task-ranked exemplars.** Eligible latest-version deliverables are scored against `project_inputs` by each document's best three chunk matches (hybrid BM25 and cosine with a key, BM25 alone without). Up to 3 exemplars are admitted (`MAX_PACKET_EXEMPLARS`) and up to 8 more are listed as `other_candidates` (`MAX_OTHER_CANDIDATES`).
2. **Related-type backfill.** When fewer than two exemplars of the requested `deliverable_type` exist, a high-relevance document of another type can fill an open slot. Once two or more same-type exemplars are present, a third cross-type document is admitted only when its type shares a content word with the requested type (for example, `Noise assessment letter` for `Noise assessment report`) and it scores at least 0.6 (`CROSS_TYPE_RELEVANCE`) of the top exemplar.
3. **Room for long exemplars.** A 52-page report with 86 sections could fill the packet with its outline alone. Each exemplar's outline is capped at 8,000 bytes (`EXEMPLAR_OUTLINE_MAX_BYTES`), keeping top-level, recurring, input-matched, and boundary rows first and reporting `omitted_outline_sections`. At least 16,000 bytes (`PACKET_MIN_INLINE_BYTES`) are reserved for inline section Markdown when more than one exemplar is kept.
4. **Section priority.** Within the inline budget, whole sections are packed in tiers: the first exemplar's input-matched sections, supplemental sections, the other exemplars' input-matched sections, the first exemplar's opening and closing sections, sections that show a recurring cluster not yet shown, the other exemplars' opening and closing sections, then the remaining recurring sections and the rest of each exemplar. A 52-page first exemplar therefore cannot crowd out a compact sibling's matched sections. Up to 2 `supplemental_sections` (`MAX_SUPPLEMENTAL_SECTIONS`) come from outside the chosen exemplars when they hold rare words of `project_inputs`, such as a specialist biography or a method appendix; they come only from documents of the requested type, or of the first exemplar's type when none is requested.
5. **Curator notes and lessons.** Up to 6,000 bytes each of relevant curator notes (`PACKET_NOTES_BYTES`) and verified feedback lessons for the requested type (`PACKET_LESSONS_BYTES`), with `omitted_curator_notes` and `omitted_lessons` counted.
6. **Provenance.** Every section reports its document date, version status, and how many independent version families share its recurring text (`family_recurrence`, whose `corpus_family_count` gives the largest number of families holding any one recurring unit). The packet's `recurring_samples` list the opening words of those recurring passages. Text shared by several families is established house text that the agent can reproduce. A clause from one family may be project-specific, so the agent weighs its scope and, in an interactive session, asks the user when an institutional policy is uncertain. Every project-specific fact (client, site, dates, measured values, fees, serials, coordinates) comes from `project_inputs`. House-wide equipment, method, and standard technical constants that appear identically across independent version families or in the lessons are carried into the draft unless the inputs give a project value, which always wins.

## Hosted models, costs, and scaling

`src/arkive_rag/llm.py` fixes three OpenAI models. Every snapshot records the exact model IDs and reasoning efforts in `corpus_meta.build_settings`. Prices are the standard list prices the cost meter uses, in USD per million tokens.

| Model | Input / cached input / output | Used during `build` | Used during `serve` |
| --- | --- | --- | --- |
| `gpt-6.1-sol` | $2.00 / $0.10 / $10.00 | Page transcription (low effort, high on retry), observation cards, organisation, facet and type canonicalisation, version partition, feedback lessons, glossary, topic guide | Never called by the server |
| `gpt-6-luna` | $0.10 / $0.01 / $0.00 | Decisions API: content-bound facets, glossary and lesson support checks | Never called |
| `text-embedding-3-large` | $0.13 input | Batched chunk embeddings (1,536 dimensions) | One query embedding per `search` or `prepare_draft` call |

Cost scales with these drivers, measured on the public fictional corpora with schema `7.0` snapshots:

- **Cold build.** Page transcription and per-document observation make up most of the cost, so it grows mainly with rendered pages and document text. The 126-file `structural-monitoring-consultancy` archive cost $4.67 and each 9-file collection $0.18 to $0.37. [Metered build cost](evaluation-benchmarks.md#metered-build-cost) has the per-operation ledger.
- **Cached rebuild.** An unchanged rebuild with `--cache` makes no provider call. An added file is read alone, and the collection-level induction reruns.
- **Runtime tokens.** Every result is capped, so a single answer's tokens grow slowly with archive size. Long drafting is the exception: an agent matching a long report's house wording reads many sections. [Holdout drafting](evaluation-benchmarks.md#holdout-drafting-layer-3) gives the token counts.

`arkive-rag build` and `python -m evals.run` accept `--max-cost <USD>`. No new provider call starts once the estimated spend reaches the limit. Calls already in flight can finish past it, and the estimate excludes the long-context surcharge.
