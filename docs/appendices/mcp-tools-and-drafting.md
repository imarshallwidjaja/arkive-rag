# Appendix B: Server tools and drafting reference

This appendix lists what `arkive-rag serve` offers an AI assistant: every tool with its options and limits, how search ranks results, how the drafting packet is assembled, and what the models cost. [Building and serving a snapshot](../how-it-works.md) and [Data boundaries, security, and cost](../security-privacy-and-cost.md) are the plain-language summaries. Maintainers will find the authoritative specification under [Retrieval](../architecture.md#retrieval), [Drafting packet](../architecture.md#drafting-packet) and [MCP server](../architecture.md#mcp-server).

The server opens one or more snapshots read-only and offers eleven tools over MCP, through standard input and output (the default) or HTTP (`--transport streamable-http`). It makes no generative model calls. It returns data from the local snapshots as compact JSON, page images as WebP, and, when `OPENAI_API_KEY` is set, computes one query embedding per `search` or `prepare_draft` call.

![Two server paths: pinpoint questions through the read tools, and drafting through one bounded packet](../assets/mcp-serving-and-drafting.svg)

[Open the interactive diagram](../assets/mcp-serving-and-drafting.html)

## Startup checks and network safety

The server refuses to start unless every snapshot passes these checks:

- it uses the current snapshot format and has every required table;
- its stored records are consistent: every filter value, glossary citation, document type, version group, reviewer lesson and group of standard wording refers to something that exists;
- every stored fact about a document cites a section of that document, and every stored table matches the table in its section's text exactly;
- its chunk vectors load into memory.

With `OPENAI_API_KEY` set, all served snapshots must use the same embedding model and vector size, or the server refuses to start.

The server also starts without `OPENAI_API_KEY`. Ten tools then work unchanged, `prepare_draft` ranks by keywords alone, and `search` returns an error that points to the offline tools. For exact matches without a key, use `grep`, as a tool or on the command line.

Over HTTP, the server listens on `127.0.0.1:8000` by default. It refuses any host other than `127.0.0.1`, `localhost` or `::1` unless `--allow-nonlocal` is given, because the HTTP mode has no built-in authentication.

## The eleven tools and their limits

Every tool returns one compact JSON text block of at most 45,000 bytes and 1,600 lines. That size is below the point at which MCP clients such as OpenCode cut off tool results, so an assistant never silently loses the end of a search result, outline or drafting packet.

When the assistant connects, the server sends short instructions. They tell it to:

- call `list_topics` when the topics or filters are unknown;
- answer questions with the read tools, using `query_tables` for values in tables;
- cite `doc_id`, `section_id` and pages, and say so when the collection does not establish a fact;
- repeat a call with `versions="all"` when the result reports superseded drafts left out (`excluded_superseded_count` above zero) and the user asks about earlier drafts;
- draft by calling `prepare_draft` with the user's full inputs.

When the server serves a single topic, the instructions also name that topic, its filters and up to 20 document types, so the assistant can skip a first `list_topics` call.

| Tool | Returns | Defaults and limits |
| --- | --- | --- |
| `list_topics` | The served topics, each with its filters, allowed values, document types, and how many documents each filter covers. | The first 50 values and types per topic, the rest counted; paged. |
| `get_topic_guide` | The cited collection guide, `AGENTS.md` notes, filter definitions with their coverage, glossary terms, and each document type with its document and feedback counts. | Fits within 45,000 bytes; left-out items are counted. |
| `list_documents` | A page of documents with dates, types, version groups, stated facts and filter values. | 50 per page by default, at most 100, fewer when more would not fit. Summaries cut to 240 characters. At most 12 facts per document within 4,000 bytes, with names cut to 120 characters and values to 300. |
| `get_document_outline` | A document's details, stated facts and section tree, with each section's `section_id`, heading path, pages and length. | Paged with `offset`; facts within 8,000 bytes. |
| `lookup_glossary` | All glossary terms, or the entries matching a query with their cited meanings. An empty (`null`) definition means the cited evidence did not establish one. | Query up to 1,000 characters; paged within 45,000 bytes. |
| `grep` | Exact substring matches in section text, for codes, filenames and numbers. Matching ignores case and treats Unicode dash and width variants as equal. A match in a table row is shown under the table's header row. | 12 matches by default, at most 50. `truncated` says whether more lines match. |
| `query_tables` | Rows of the tables stored at build time. `columns` keeps tables whose headers contain every given name, and shows those columns plus the row labels. `query` matches headers and cells together, and shows the matching rows, or the first rows of a table whose header alone matches. A query with no words matches nothing. | 20 tables by default, at most 50. Per table at most 24 columns and 20 rows within 12,000 bytes, cells cut to 300 characters, the rest counted. |
| `search` | Passages ranked by keyword and meaning together (see below). | 5 results by default, at most 50. At most 3 passages from one version group before other documents fill the list. |
| `read_section` | A section's full text with links to the previous and next sections. | Paged with `offset`. |
| `view_page` | A rendered page as a WebP image, with its chart description and text layer, so a model that reads images can inspect charts, maps and layouts. | Text cut to fit 45,000 bytes, with counts of what was left out. |
| `prepare_draft` | One drafting packet for a new deliverable, built from the user's inputs (see below). | 45,000 bytes; up to 3 past examples with outlines, sections, lessons, notes and provenance. |

### Filters, document types and versions

`list_documents`, `search`, `grep`, `query_tables` and `prepare_draft` accept `filters`, a mapping from filter name to one value. For example, on the measured build of `structural-monitoring-consultancy`, `{"job_cohort": "archive", "sensing_technology": "fibre_bragg_grating"}` keeps archived fibre Bragg grating jobs. Filter names are generated per build, so another build may name them differently. A value can be the canonical value or another spelling of it. Several filters combine with AND. An unknown filter or value returns an error that lists the valid options, so the assistant can correct itself on its next call. On the command line, `search` and `grep` take the same filters as repeated `--filter FACET=VALUE` options.

`list_documents`, `search`, `grep` and `query_tables` also accept:

- `deliverable_type`, a document type, matched ignoring case;
- `versions`, either `"current"` (the default) or `"all"`.

With `"current"`, a document superseded by another document in scope is left out and counted in `excluded_superseded_count`. With `"all"`, every version is kept. Every document, search result, match and table carries the same version fields: `deliverable_type`, `doc_date`, `family_id`, `version_status`, `superseded_by` and `superseded_doc_ids`. `prepare_draft` always drafts from current versions. On the command line, `search` and `grep` take `--type` and `--versions`.

### Search ranking

`search` checks the filters, then computes one embedding for the query. It takes the top 300 passages by keyword relevance (BM25) and the top 300 by meaning (cosine similarity of the vectors). It merges the two lists by reciprocal rank fusion: each list adds `1 / (60 + rank)` to a passage's score. Punctuation and search operators in the query are treated as plain text, so a query such as `Sentinel-2` never causes a syntax error. A superseded draft is left out when its latest version is in scope, and at most three passages come from one version group before other documents fill the list.

## Drafting packet assembly

`prepare_draft(topic, project_inputs, deliverable_type, filters)` chooses and packs reference material when it is called. It makes no generative model call.

1. **Ranking past work.** Each current deliverable is scored against `project_inputs` by its three best-matching passages, ranked by keyword and meaning with a key, or by keywords alone without one. Up to 3 past examples are included, and up to 8 more are listed as `other_candidates`.
2. **Filling from related types.** When fewer than two past examples of the requested type exist, the most relevant document of another type can fill a free slot. When two or more of the requested type are already included, a third document of another type is added only when its type shares a content word with the requested type, such as `Noise assessment letter` for `Noise assessment report`, and its score reaches at least 0.6 of the best example's.
3. **Room for long examples.** A 52-page report with 86 sections could fill the packet with its outline alone. Each example's outline is therefore capped at 8,000 bytes. The cap keeps top-level, recurring, input-matched and first and last entries, and counts the rest in `omitted_outline_sections`. When the packet holds more than one example, at least 16,000 bytes are kept free for section text.
4. **Which sections go in first.** Whole sections are added in this order while they fit:
   1. the first example's sections that match the inputs;
   2. up to 2 extra sections from outside the chosen examples (below);
   3. the other examples' sections that match the inputs;
   4. the first example's opening and closing sections;
   5. sections with standard wording that no included section shows yet;
   6. the other examples' opening and closing sections;
   7. the remaining sections with standard wording, then the rest of each example.

   A 52-page first example therefore cannot crowd out a shorter example's matching sections. The extra sections come from documents of the requested type, or of the first example's type when none is requested, and are the sections that hold the most rare words from `project_inputs`, such as a specialist's biography or a method appendix.
5. **Notes and lessons.** Up to 6,000 bytes each of the `AGENTS.md` notes and of the confirmed reviewer lessons for the requested type, with `omitted_curator_notes` and `omitted_lessons` counting what was left out.
6. **Provenance.** Every section reports its document date, version status, and how many independent version groups share its recurring text (`family_recurrence`). The packet's `recurring_samples` give the opening words of those recurring passages. Text shared by several version groups is established house text that the assistant can reproduce. A clause that only one group holds may be specific to that project, so the assistant weighs its scope and, in an interactive session, asks the user when a company policy is uncertain. Every project-specific fact (client, site, dates, measured values, fees, serial numbers, coordinates) must come from `project_inputs`. House-wide equipment, methods and standard technical constants that appear identically in independent version groups, or in the lessons, carry into the draft unless the inputs give a project value, which always wins.

## Models, prices and cost growth

Arkive uses three fixed OpenAI models. Each snapshot records the exact model IDs and reasoning efforts it was built with, which `arkive-rag inspect` prints. The prices below are the standard list prices that Arkive's cost estimate uses, in USD per million tokens.

| Model | Input / cached input / output | Used by `build` | Used by `serve` |
| --- | --- | --- | --- |
| `gpt-6.1-sol` | $2.00 / $0.10 / $10.00 | Page transcription (low effort, high on a retry), document records, filters and types, version groups, reviewer lessons, glossary, guide | Never |
| `gpt-6-luna` | $0.10 / $0.01 / $0.00 | Text-based filter values and the glossary and lesson support checks, through OpenAI's Decisions API | Never |
| `text-embedding-3-large` | $0.13 input | Chunk vectors, 1,536 dimensions, in batches | One query embedding per `search` or `prepare_draft` call |

Measured on the fictional example collections with the current version, cost grows as follows:

- **First build.** Page transcription and the per-document records make up most of the cost, so it grows mainly with rendered pages and document text. The 126-file `structural-monitoring-consultancy` archive cost $4.67, and each 9-file collection $0.18 to $0.37. [Metered build cost](evaluation-benchmarks.md#metered-build-cost) breaks this down by step.
- **Cached rebuild.** Rebuilding an unchanged folder with `--cache` makes no paid call. An added file is read on its own, and the collection-wide organising pass runs again.
- **Tokens while serving.** Every result is capped, so the tokens behind a single answer grow slowly with the size of the archive. Long drafting is the exception: an assistant matching a long report's house wording reads many sections. [Drafting tests](evaluation-benchmarks.md#drafting-tests) gives the token counts.

`arkive-rag build` and `python -m evals.run` accept `--max-cost <USD>`. No new paid call starts once the estimated spend reaches the limit. Calls already in progress can finish past it, and the estimate leaves out OpenAI's surcharge for very long prompts.
