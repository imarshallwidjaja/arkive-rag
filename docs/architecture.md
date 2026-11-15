# Architecture

Arkive RAG turns a folder of reference documents into one SQLite file, the
*snapshot*, and serves one or more snapshots to AI agents over MCP. Building a
snapshot calls OpenAI models. Reading one needs no network, except that the
`search` tool embeds each query with the OpenAI embedding model.

The build does not assume how a collection is organised. It reads every
document, induces from the whole corpus the facets an agent can filter by, a
cited glossary and a blueprint for each recurring deliverable type, and stores
them in the snapshot next to the sections, pages and chunks.

## Modules

| Module | Owns |
| --- | --- |
| `models.py` | Domain types (readings, observation cards, facets, glossary entries, blueprints, the build report, read results), the provider output schemas, the shared text folding (`fold_text`, `fold_for_quote`), and the `ArkiveError` hierarchy. |
| `llm.py` | The request protocols (`StructuredRequest`, `AgentRequest`), the provider-neutral `Tool`, the `LanguageGateway` protocol, `OpenAIGateway` (every provider call: structured Responses calls, Agents SDK runs, Decisions, embeddings, the retry policy), `BuildCache`, `UsageMeter` with the spend limit, and `.env` loading. |
| `compiler.py` | Reading (discovery, rendering, the transcription fidelity gate, section stitching, chunk pieces, observation cards with quote location) and binding (IDs, breadcrumbs, embeddings). `compile_corpus` composes reading, induction and binding into a `CompiledCorpus`. Its request types (`PageTranscription`, `ObserveSections`, `ConsolidateCard`) own their prompts, output schemas and reasoning efforts. |
| `induction.py` | Induction: corpus organisation, facet binding, the glossary, blueprints and the topic guide; the validate-and-retry helper `generate_checked`; support checks; the read-only source tools of the blueprint agent. Its request types (`OrganiseCorpus`, `WriteGlossary`, `WriteBlueprint`, `WriteGuide`) own their prompts, schemas and efforts. |
| `snapshot.py` | The SQLite schema, the atomic writer, and `TopicSnapshot`, the read-only query surface (facet filters, hybrid search, grep, sections, pages, glossary lookup). |
| `server.py` | `TopicCatalog` (open snapshots keyed by topic) and the nine MCP tools. |
| `cli.py` | The `arkive-rag` command. It is the only package module that reads `.env` and constructs `OpenAIGateway`; the evaluation harness in `evals/` does the same for its own runs. |

Dependencies point one way: `cli` → `server` → `snapshot`, and `cli` →
`compiler` → `induction` → `llm`; every module uses `models`. `snapshot` and
`server` import neither `compiler`, `induction` nor `llm`; the CLI passes the
query embedder in. The compiler and induction receive a `LanguageGateway`, so
tests drive them with a deterministic fake and no network. Only `llm.py`
imports the OpenAI SDK and the OpenAI Agents SDK (`openai-agents`).

## Corpus layout

Every non-hidden file under the corpus directory must be `.pdf`, `.docx`,
`.pptx` or `.md`. Any other file stops the build with `UnsupportedFormatError`
before the first model call. Paths starting with `.` are skipped. A symlink
anywhere in the corpus stops the build with `UnsafePathError`, and each source
is read once without following symlinks; Word and PowerPoint files are
converted from a private copy of those bytes. The snapshot and the build cache
must be different files outside the corpus directory.

- Folders carry no fixed meaning. Induction decides whether the folder
  structure encodes a facet (projects, campaigns, authors) and binds that
  facet to the folders it names.
- Every `AGENTS.md` is a curator note. Its text is stored verbatim in the
  topic guide and given to induction as the curators' stated intent, which
  outranks the model's inference. It is also compiled as a document
  (`is_curator_note`, chunk kind `guidance`), so status questions such as
  "where is the reef case study?" match the note in both `search` and `grep`.
  A curator note is not citable: no glossary entry, blueprint or guide
  citation may point at it, and the blueprint tools cannot read it. A
  path-bound facet value still covers a note in its folder, so a project
  filter also finds that project's note.
- Documents are numbered in sorted path order: `doc_001`, sections
  `doc_001_sec_001`, chunks `doc_001_chk_0001`. The same corpus produces the
  same IDs.

## Build pipeline

`compile_corpus` runs three passes in order. Reading runs per document in a
thread pool (`--max-concurrency`, default 4); induction is corpus-wide and
one executor of the same size bounds all its provider calls in flight;
binding needs both.

### Reading

`read_document(source, data, gateway)` returns a `DocumentReading`: the file's
SHA-256, its sections and chunk pieces, its rendered pages and its observation
card. A reading holds no corpus IDs and no path, so the provider responses it
is built from stay valid in the build cache when files are added, removed or
moved. Each PDF, Word and PowerPoint file follows one path:

1. **Render.** Word and PowerPoint files are converted to PDF by LibreOffice
   (`soffice --headless`) with a private temporary profile, a minimal
   environment and a 120 s timeout. Each PDF page is rendered by pypdfium2 at
   scale 2.0 to WebP (quality 85), and its full Unicode native text layer is
   extracted. libpdfium is not thread-safe, so every pdfium call runs under one
   process-wide lock; WebP encoding runs outside it.
2. **Transcribe.** `gpt-6.1-sol` at low reasoning effort receives the page
   image, the native text, and the tail of the previous page's transcript. It
   returns blocks (`heading`, `paragraph`, `table`, `callout`) and visual
   transcripts for charts, maps and figures. The image decides what the page
   says, which repairs broken glyph encodings in the text layer; headings keep
   their visible section numbers. The previous page's tail lets the model set
   the continuation flags below, so page N's request depends on page N-1's
   transcript: a changed page also changes the cache keys, and so the
   transcription, of every later page of its document.
3. **Check fidelity.** Running headers, footers and page numbers are removed
   before anchors are collected: a top or bottom line repeated exactly on
   three or more pages, or repeated apart from a leading or trailing number
   that advances with the page; a labelled page number (`Page 3`, `3 of 12`)
   on an edge line; and a bare number on a page's first or last line. The
   anchors are GIS/data filenames (`.tif`, `.tiff`, `.shp`, `.nc`, `.gpkg`,
   `.csv`), signed formatted numbers such as `-330.9` or `59,494`, and integers
   of three or more digits. They are checked against the exact text the page
   contributes to its sections. `FIDELITY_ATTEMPTS` holds the two attempts. If
   a filename is missing from the first (low effort) transcript, or recall is
   below 0.95 (1.0 when the page has one or two anchors), the page is
   transcribed once more at high effort and the missing values are named in
   the request. After that retry, a missing filename or recall below 0.85
   raises `TranscriptionFidelityError` naming the file, page and missing
   values.
4. **Stitch sections.** `assemble_sections` keeps a heading stack (levels 1-4;
   a heading with level 0 counts as level 2) and starts a section at every
   heading; text before the first heading is `Overview`. A table block keeps
   empty cells and accepts GFM tables without outer pipes; caption or footnote
   lines around the table stay as paragraphs. A block flagged
   `continues_from_previous_page` joins the previous paragraph or table, and the
   joined part spans both pages. A continued table loses its repeated header
   rows, and when `merges_into_previous_row` is set, its first row is merged
   cell by cell into the row cut by the page break. Visual transcripts become
   quoted `[Visual — Page N: label]` blocks in the section.
5. **Chunk pieces.** `section_pieces` packs paragraphs and callouts into prose
   pieces of up to 1,800 characters, splitting an oversized paragraph at
   sentence ends. A table up to 2,400 characters is one `table` piece; a larger
   table is split on row boundaries into pieces of up to 2,000 characters, each
   repeating the header and separator rows. Each visual transcript is its own
   `visual` piece. No piece exceeds 6,000 characters: an oversized sentence,
   table row or visual is wrapped at line breaks or spaces, and table pieces
   keep the header. A piece has its own page span and no ID or breadcrumb yet.
6. **Observe.** The document's sections are packed in order into batches of
   at most 24,000 characters (a larger section is split at line breaks or
   spaces into pieces within that size, each keeping the section's ordinal), and
   `gpt-6.1-sol` at low effort reads each batch (`observe`). It returns the
   title, date, deliverable type (free text, such as "technical memorandum"
   or "campaign brief"), summary, the attributes the document states about
   itself (client, author, audience, channel and similar), the terms it uses
   (with kind, aliases and whether the quote defines the term), and craft
   observations (structure, convention, caveat, disclaimer, reviewer
   feedback, tone). Every term and craft observation carries a verbatim
   quote, never a section number. Code locates each quote in the batch's
   sections with `fold_for_quote` (NFKC, case, Unicode dashes, whitespace,
   `*` and `_` emphasis, backticks, table pipes, block-quote markers and
   line-break hyphenation ignored; a `*` with spaces on both sides and an
   underscore inside a word stay text) and records the first containing
   section's ordinal. Unlocated quotes get one rerun that names them. After
   the rerun, observations whose quote is still unlocated are discarded and
   counted in the build report; a batch that quoted something and had no
   quote located raises `RejectedOutputError`, also when its rerun returns
   no observations at all. A document with several batches has its terms,
   craft and attributes unioned in code, and one `consolidate_card` request
   (low effort) writes its single title, date, deliverable type and summary
   from the batch headers. A curator note gets a card from its own text and
   no model call: the first line with text is its title and the first line
   that is not a heading its summary.

Markdown files skip steps 1-3: `parse_markdown_blocks` splits them
deterministically, keeping a fenced code block as one literal paragraph, and
they have no page images for `view_page`. A document with no text raises
`ConversionError`.

### Induction

`induce(documents, notes, gateway)` receives the readings with their IDs and
runs five steps: organise, then bind, then the glossary and the blueprints
concurrently, then the guide. One executor with `--max-concurrency` workers
runs the binding questions, the blueprint runs, the glossary batches and the
definition checks; each task makes one provider call at a time, so the limit
bounds them together. Results are gathered in submission order, and the first
failure of any task cancels the tasks not yet started. Every generating step
goes through
`generate_checked`: run, validate, rerun once with the rejections appended to
the request, then raise `RejectedOutputError`. Only three outcomes reduce the
output without failing, and the build report counts each: an observation
whose quote is still unlocated after its rerun, a glossary definition the
support check does not confirm, and a blueprint convention or pitfall it does
not confirm.

1. **Organise** (`organise_corpus`, medium effort). The request holds the
   folder tree, the curator notes and every citable document's card. The
   model returns up to six facets and up to eight recurring deliverable types
   (each type costs one blueprint run, and `get_topic_guide` returns every
   blueprint). A facet has a snake_case name, a description and a closed
   list of values, each with a description and aliases. A facet is
   *path-bound* when every value names the corpus folders it stands for, or
   *content-bound* when no value names a folder. Values and aliases are
   compared by `fold_facet_label`, the folded form the snapshot's alias table
   and the filter resolver use. Repetition within one value is normalised:
   values and aliases are stripped, blank aliases dropped and one alias kept
   per folded form. Validation rejects a facet that mixes the two bindings,
   an unknown folder, a duplicate facet name, a blank value, a value given
   twice, a folded form that names two values of one facet, more than six
   facets, more than eight deliverable types, and a deliverable type with no
   exemplar or with an unknown document ID. An organisation that passes is
   always storable. A
   path-bound value also answers to the paths of its folders, unless a path
   already names a value of the facet or belongs to several values.
2. **Bind.** A path-bound value is assigned to every document under its
   folders, curator notes included. Content-bound values are judged per
   citable document: one `bind_facets` call to the Decisions API whose
   evidence is the document's path, title, date, deliverable type, summary,
   attributes and outline, with one predicate per value. A probability of
   0.5 or more assigns the value; a refused question assigns nothing and is
   counted. A document may hold several values of a facet or none. Values no
   document holds are then pruned, and so are facets with fewer than two
   populated values; the build report names both.
3. **Glossary** (`write_glossary`, medium effort, structured requests).
   Code merges term observations by folded term and alias, keeps a term seen
   in two or more sections or quoted where a document defines it, and scans
   the other citable sections for further occurrences (up to six snippets,
   case-folded). A term's name, aliases and quotes are chosen in a total
   order (most frequent name, ties broken by folded then literal text), so
   the same observations in any order give the same request. Terms are sent
   in batches with their quotes and snippets, each tagged with its section
   ID. Each entry has a term, kind, aliases and
   one or more senses; a sense has a definition, or `null` when the cited
   sections use the term without establishing its meaning, and section
   citations. Conflicting meanings are separate senses. Validation requires
   one entry per term given, at least one sense and one citation per sense,
   citations that resolve to citable sections, and the entry's own term or
   one of the aliases it publishes in every cited section; an entry's aliases
   are stripped, one per folded form and sorted. Then `check_definitions`
   asks the Decisions API whether the cited sections establish each
   definition; below 0.5 or refused, the definition becomes `null` and is
   counted.
4. **Blueprints** (`write_blueprint`, medium effort, one Agents SDK run of
   at most 12 turns per recurring deliverable type). The agent reads the
   citable documents through three read-only tools, `outline`,
   `read_section` and `grep`, which refuse an argument that is not a string;
   its request lists the exemplars' outlines, the feedback documents and,
   as leads the agent must verify in the source before using, the craft
   observations located in those documents (aspect, statement, section ID),
   in sorted order. It returns a description, exemplar document IDs,
   a section outline (heading, purpose, citations), conventions and reviewer
   pitfalls (statement, citations), stated without project-specific values.
   Validation requires a non-empty outline, citable exemplars and citations
   that resolve to citable sections. `check_blueprint` then asks the
   Decisions API whether the cited sections support each convention and
   pitfall; unsupported or refused ones are dropped and counted.
5. **Guide** (`write_guide`, medium effort). A cited Markdown overview from
   the cards, the bound facets, the glossary terms, the blueprint summaries
   and the curator notes. Every bracketed `[doc...]` token must have the exact
   form `[doc_XXX:doc_XXX_sec_YYY]` and resolve to a section of a citable
   document, and at least one valid citation is required when such a section
   exists. A corpus with no citable section (curator notes only) gets a guide
   without citation tokens.

A support check judges the complete cited sections. Evidence longer than
48,000 characters is sent in windows (a longer section in parts that overlap
by 2,000 characters), and each predicate takes its highest probability over
the windows; it counts as refused only when every window refused it.

### Binding

`bind_document` gives each section and chunk its ID and each chunk the
breadcrumb `[<facet>: v1, v2 | ... | Doc: <title> (pp. X-Y) | Section: <heading
path>]`, with facets in definition order, values sorted and facets the
document holds no value of left out. Every chunk is then embedded as
`breadcrumb + "\n" + content` with `text-embedding-3-large` at 1,536
dimensions.

`build` prints the usage line and the build report. The build report has
`discarded_unlocated_quotes`, `unsupported_definitions`,
`dropped_blueprint_statements`, `judge_refusals` (per Decisions operation),
`pruned_facet_values` and `pruned_facets`.

## Provider calls

`OpenAIGateway` is the only provider caller for builds and the evaluation
harness, and the single owner of retry policy; the SDK clients run with
`max_retries=0`. Domain code describes each call with a request object that
owns its prompt, output schema and reasoning effort; the gateway owns
transport. Model IDs are constants in `llm.py`: `DEFAULT_MODEL`
(`gpt-6.1-sol`), `DECISION_MODEL` (`gpt-6-luna`) and `text-embedding-3-large`
at 1,536 dimensions. The CLI has no model options.

- `run` sends one structured `responses.parse` call with the request's
  `reasoning.effort`.
- `run_agent` runs a bounded OpenAI Agents SDK agent with read-only tools and
  structured output on its own event loop with its own async client. Each
  model turn is one provider operation under the retry policy below; the
  Agents SDK's own retries stay off. Invalid tool input (`NotFoundError`,
  `InvalidFilterError`, `ValueError` or `KeyError`) and a call to a tool name
  the agent does not have go back to the model as text. Not finishing within
  `max_turns`, a response that is not `completed`, or a provider or
  spend-limit error raises `ProviderError`; any other tool exception,
  including `SnapshotIntegrityError` and a `ProviderError` from a tool, ends
  the run and propagates. Tools use non-strict JSON schemas.
- `judge` asks the Decisions API (`POST /v1/decisions`) one `predicate`
  question per statement over shared evidence and returns one probability
  per predicate in order, `None` where the model refused that question. No
  predicates means no request; more than 400 are split across requests.
- `converse` is the free-text agent conversation the evaluation harness uses
  (see Evaluation). It runs on the caller's event loop and is never cached.
- Hosted Agents SDK tracing is off unless the caller passes `trace=True`
  (`--trace`). The trace export key is passed with each run, not set on the
  SDK's process-wide exporter. `trace=True` while
  `OPENAI_AGENTS_DISABLE_TRACING` disables tracing fails when the gateway is
  constructed.
- Rate-limit, connection, timeout and server errors are retried, up to five
  attempts per operation, waiting 2, 5, 12 and 25 s or the provider's
  `Retry-After` (a value that is not a finite number is ignored). All attempts
  share one deadline (600 s for builds, 60 s for query embedding in `search`
  and `serve`), and each request's timeout (180 s, or 30 s for queries) is
  capped by the time left. A wait that would pass the deadline raises
  `ProviderError` at once, and no attempt starts after it. An agent turn's
  whole attempt, not only each socket read, ends at the deadline. The spend
  limit is checked before every attempt.
- A structured response that stops at `max_output_tokens` (starting at
  16,384) is retried once with double the budget. Agent turns allow 16,384
  output tokens.
- Every other failure, a refusal, a body that is not JSON, missing or
  malformed usage, or output that fails schema validation raises
  `ProviderError`, on every call type. A `judge` probability that is not a
  finite number within 0..1 is malformed too.
- Embeddings are requested in batches of up to 128 inputs and 100,000
  characters. Response rows must carry each index exactly once and are
  reordered by it; every row is L2-normalised. A wrong shape or a zero or
  non-finite vector raises `ProviderError`.

`--cache <file>` enables `BuildCache`, a SQLite table of validated responses
keyed by SHA-256 of the call's inputs:

- `run`: operation, model, effort, instructions, text, output schema and the
  image hash;
- `run_agent`: the same plus `max_turns`, each tool's name, description and
  parameter schema, and the request's `source_digest` (a hash of everything
  the tools can read), so a result computed while the tools could read one
  source set never answers a request over another. The blueprint request's
  digest also covers `SOURCE_TOOLS_REVISION`, which names the tools'
  behaviour, so a change in what the tools return is never answered from an
  older run;
- `judge`: operation, decision model, evidence and predicates;
- embeddings: model, dimensions and the batch's texts.

A rebuild of an unchanged corpus with the same cache makes no provider
requests: every collection that reaches a prompt or a cache key is in a
canonical sorted order, and concurrent results are gathered in submission
order. Adding a file requests the reading of that file only; induction and
the embeddings are requested again because the corpus changed. A build of a
subset of a corpus never reuses the full corpus's induction results, because
each induction request carries the cards, IDs or source digest of the
documents it covers. The first valid entry stored under a key is canonical: when two build
threads miss the same key at once, the second writer returns the first
writer's value. An entry that no longer decodes or validates is a miss and is
replaced. A new cache file is created with mode `0600`.

Every provider response adds its operation, model, input, cached-input and
output tokens to a `UsageMeter`, which estimates USD from Standard-tier list
prices in `llm.PRICES` (`gpt-6.1-sol`, `gpt-6-luna` on Decisions, and
`text-embedding-3-large`; the long-context surcharge is not modelled).
`build` prints the totals. With `--max-cost`, no provider call starts once the
estimate has reached the limit: it raises `BudgetExceededError`, a
`ProviderError`, and `build` exits with an error after printing the totals.
Calls already in flight can take the total past the limit. A spend limit with a
model that has no price is refused when the gateway is constructed.

## Snapshot file

`write_snapshot` writes a uniquely named sibling temporary file (mode
`0600`), inserts everything in one transaction, runs `PRAGMA
foreign_key_check` and `PRAGMA integrity_check`, fsyncs, and renames it over
the output, then fsyncs the directory. A replaced snapshot keeps its file
mode. A failed build leaves any previous snapshot untouched, and SQLite or
file-system failures raise `SnapshotIntegrityError`.

Tables (schema version `3.0`):

| Table | Contents |
| --- | --- |
| `corpus_meta` | `schema_version`, `topic`, `title`, `created_at`, `corpus_fingerprint`, `arkive_version`, `embedding_model`, `embedding_dimensions`, `guide_markdown`, `build_settings` (JSON: the model IDs and the effort of each operation) and `build_report` (JSON, as printed by `build`). |
| `curator_notes` | Verbatim `AGENTS.md` files with their path. |
| `documents` | One row per source file: `doc_id`, `ordinal`, `rel_path`, `sha256`, `format`, `title`, `doc_date`, `deliverable_type`, `summary`, `page_count`, `is_curator_note`. |
| `facets` | Name, order, description and binding (`path` or `content`). |
| `facet_values` | Each facet's values with their order, description and folders (JSON). |
| `facet_aliases` | Every folded form of a value, the canonical value included, keyed by facet and folded alias, so a form that would name two values cannot be stored. |
| `document_facets` | Document-to-value assignments, with foreign keys to `documents` and `facet_values`. |
| `glossary` | One validated `GlossaryEntry` JSON document per row. |
| `blueprints` | One validated `Blueprint` JSON document per row. |
| `sections`, `pages`, `chunks` | Section Markdown with heading path and pages; native text, visual summary and WebP per page; chunk content, breadcrumb and float32 embedding. |
| `chunks_fts` | FTS5 over NFKC-normalised chunk content and breadcrumb (`porter unicode61 remove_diacritics 2`). |

The corpus fingerprint is the SHA-256 hex digest of the lines `rel_path NUL
sha256`, one per document sorted by `rel_path` and joined by newlines. It
depends only on source paths and bytes, not on models, prompts or build time,
so it is the content address of the corpus a snapshot was built from
(`CompiledCorpus.corpus_fingerprint`).

`TopicSnapshot.open` uses a read-only SQLite URI with the path
percent-encoded, so names containing `%`, `?` or `#` open the right file. It
refuses any schema version other than `3.0`, naming the file's version and the
one it reads; a `2.0` snapshot is rebuilt from its corpus. Every
`corpus_meta` key and every table is required, and `open` runs `PRAGMA
foreign_key_check`, so a dangling facet assignment or alias is refused. Each
facet value must have its canonical alias row. A snapshot is a build output
that may be truncated or damaged: any decoding or cross-table failure while
it opens, and any SQLite error from a later read, is `SnapshotIntegrityError`
naming the file. `open` recomputes the fingerprint from the
`documents` table and raises `SnapshotIntegrityError` when the recorded one
differs. It validates every glossary and blueprint row through its model and
checks that every cited section belongs to a citable document of the
snapshot and that every blueprint exemplar and feedback document is one. It
loads the facets, aliases and document-to-facets map once, with chunk
metadata and the embedding matrix. `list_topics` and `arkive-rag inspect`
report the fingerprint and version; `inspect` also prints the facets,
glossary, blueprints, build settings, build report and each document's
`sha256`.

## Private corpus and eval versioning

Code, the eval pack schema and the eval runners live in this repository, which
holds no client documents, client facts or personal data. Corpora, eval packs
with corpus facts, and snapshots live in a private store outside it (a private
Git LFS or DVC repository, or a private bucket):

```text
private-store/
  bathymetry/corpus/...                         source documents
  bathymetry/bathymetry.eval.json               eval pack, pinned to a corpus_fingerprint
  bathymetry/bathymetry-1a2b3c4d5e6f.arkive.db  snapshot, named by corpus_fingerprint[:12]
```

A snapshot file is immutable once published: a changed corpus gets a new
fingerprint and so a new file name. To check that a snapshot matches a corpus
folder, compare `inspect`'s `corpus_fingerprint` and per-document `sha256`
values with the folder's files. An eval pack that sets `corpus_fingerprint`
refuses any snapshot built from a different corpus. `.gitignore` excludes
`*.arkive.db`, `*.arkive.db.*` and `*.eval.local.json` so that snapshots and
local packs are not staged by accident.

## Retrieval

- **Filters.** `list_documents`, `search`, `rank_channels` and `grep` take
  `filters`, a mapping from facet name to one value. A value matches a
  canonical facet value or one of its aliases after `fold_text`. Several
  facets combine with AND; an empty mapping means no filter.
  `TopicSnapshot.resolve_filters` turns the mapping into the set of matching
  document IDs once, and every read path uses that set. An unknown facet or
  value raises `InvalidFilterError` naming the valid facets or values.
- **search** fuses two rankings with Reciprocal Rank Fusion
  (each ranking adds `1 / (60 + rank)` to a chunk's score), taking the top 100 of each:
  - BM25 over `chunks_fts`, weighting content 10 and breadcrumb 2.
    `build_fts5_query` quotes every letter-and-digit token (`_` separates
    tokens, as in the `unicode61` tokenizer) and adds adjacent-token phrases,
    so user text such as `Sentinel-2`, `C12-SUM(C13:C15)` or `NOT` can never be
    an FTS5 syntax error. Filler words (`what`, `the`, `of` and similar) are
    dropped when at least two other tokens remain, and a phrase of two filler
    words is never added.
  - Exact cosine similarity between the query vector and every chunk that
    passes the filters.

  Ties break on cosine similarity, then chunk ID. Each hit reports its fused
  score, both ranks and its document's facets. A blank query or an invalid
  filter or `top_k` raises `InvalidFilterError` before the query is embedded.
- **grep** is a literal, case-insensitive line search over section Markdown.
  Pattern and text are compared by `fold_text` (NFKC, Unicode dashes as `-`,
  case-folded), so `2016-2017` finds `2016–2017` and `km2` finds `km²`. A
  snippet is at most 400 characters of the line, centred on the first match.
- **Glossary lookup.** Without a query, `lookup_glossary` lists every term
  with its kind, aliases and whether any sense has a definition. With a
  query, it returns the full entries whose folded term or alias contains the
  folded query; no match is an empty list.

## MCP server

`arkive-rag serve a.arkive.db b.arkive.db` opens every snapshot into a
`TopicCatalog` and refuses two snapshots with the same topic. The tools are
`list_topics` (each topic's facets and allowed values), `get_topic_guide`
(guide, curator notes, facet definitions, blueprints and glossary terms),
`list_documents`, `get_document_outline`, `search`, `grep`, `read_section`,
`view_page` and `lookup_glossary`. `list_documents`, `search` and `grep` take
`filters: dict[str, str] | None`. `view_page` returns the caption as text
content and the page as `image/webp` image content. Domain errors become tool
errors whose message names the valid topics, facets, values or pages.

The server's instructions describe two modes. Answering questions cites
`doc_id`, `section_id` and pages. Drafting a deliverable starts from the
blueprint for its deliverable type, reads the blueprint's feedback documents
and exemplars, and keeps every project fact to the user's own inputs.

Over `streamable-http` the server has no authentication, so `serve` refuses a
host other than `127.0.0.1`, `localhost` or `::1` unless `--allow-nonlocal` is
given. The server reports the package version to clients.

`search` needs `OPENAI_API_KEY` in the server environment and all served
snapshots must share one embedding model and dimension. Without a key the
server still starts; `search` returns an error that points to the offline
tools.

The CLI's `search` and `grep` take repeated `--filter FACET=VALUE`; naming a
facet twice is a usage error, because each facet takes one value.

## Evaluation

`evals/` is a harness outside the `arkive_rag` package; it is not in the wheel.
`python -m evals.run --pack <file> --snapshot <db> --out <dir> [--corpus <dir>]`
runs an eval pack and writes `results.json`, `report.md` and, for layer 3,
`blind-drafts/` to `--out`. `--out` is required and should be a private
directory outside the repository; the runner creates it with mode 0700 and
writes 0600 files. Every layer calls OpenAI, so it needs `OPENAI_API_KEY`.
Before creating any output or calling any model the runner checks, for every
layer it will run (so `--layer all` checks layer 3 before layer 1 calls
anything), that the pack is a valid `arkive-eval-pack/2` file, that it
defines the requested layers, that layers 2 and 3 have `--corpus`, that
`--out` and `--cache` overlap none of the pack, snapshot and corpus paths,
that the snapshot opens, that its `corpus_fingerprint` matches the pack's
when the pack sets one, that every retrieval case's `filters` resolve
against it, that the fingerprint of the `--corpus` files (found by
`discover_corpus`) equals the snapshot's, and that the holdout's reference
and included files pass the layer 3 guards below. A failure is a usage error
(exit 2) without a traceback. A failure after model calls have been made
prints the usage line before the error.

Eval agents and judges go through `OpenAIGateway` at medium reasoning effort
on `gpt-6.1-sol`: agents through `converse`, judges through `run`, under the
gateway's retry policy and metering. An agent run records its final status:
only `completed` counts as an answer; `incomplete` (or another response
status), `empty_answer`, `max_turns` and `provider_error` (including a
`ProviderError` raised by a tool) are reported and never credited. A query
embedding that fails behind the MCP `search` tool also ends its run as
`provider_error`: the MCP server hands the model a tool error, so the harness
reads the failure from its query embedder after the conversation. Invalid
input to a harness tool goes back to the model as an error message; any other
exception from a harness tool stops the evaluation. An unexpected exception
inside an MCP server tool reaches the model as a tool error, as any MCP host
would see it, and does not stop the evaluation. Each run reports its tokens,
turns, tool calls and the query embeddings that returned a vector
(`search_embed_calls`). `--max-cost` sets one spend limit, a positive finite
number of USD, over every model call of the evaluation; reaching it is not a run outcome but stops the evaluation
with an error, the usage line and a non-zero exit, and writes no report.
`--trace` turns on hosted Agents SDK tracing, and `results.json` (`usage`) and
`report.md` (Usage) record calls, tokens and the estimated cost per operation.

An eval pack is a JSON file validated by `EvalPack` in `evals/schema.py`
(`format` `arkive-eval-pack/2`). It holds `retrieval_cases`, `qa_cases` and an
optional `holdout_memo` task. A retrieval case may set `filters`, a mapping
from facet name to value or alias that layer 1 passes to `rank_channels`;
facets are induced per snapshot, so `arkive-rag inspect` lists the names and
values a pack can use. Expected facts are *groups* of alternative
substrings, matched NFKC-, case-, dash-, whitespace- and
thousands-separator-insensitively. A `RubricCheck` passes when the memo
satisfies every `require_groups` group and contains no `forbid_any` phrase.
An anti-contamination check requires the held-out project's own values and
forbids only claims that a faithful memo cannot make, so a memo that contrasts
its values with an earlier project's still passes. `evals/example_pack.eval.json`
is a fictional pack; real packs stay in the private store.

| Layer | Module | Measures |
| --- | --- | --- |
| 1 | `ablation.py` | Completion@1/3/5 and CompletionMRR@5 of BM25-only, dense-only and hybrid RRF ranking over the pack's retrieval cases, per category. A case completes at rank r when the top-r chunks jointly cover every required group and expected document. `TopicSnapshot.rank_channels` ranks all three arms from one query embedding per case. |
| 2 | `agents.py` | An agent conversation answers each QA case with raw-file tools (text grep, page text, page images over the files `discover_corpus` accepts) and with the nine Arkive MCP tools. Each completed answer gets deterministic phrase coverage (`required_groups`) and a blinded judge score; the judge grades citations against the case's `source_anchors`. Raw-file extraction runs once before timing. |
| 3 | `agents.py` | One deliverable is held out. Before any file is copied or any model is called, the reference document (`reference_doc_id`) must be a delivered document, not a curator note, and carry a `held_out_markers` entry in its path, and every file under `included_paths` (selected through `discover_corpus`, so symlinks are refused) must have no marker in its path, not share the reference's top-level folder (or, for a reference at the corpus root, not sit at the root) and not have the reference's bytes. Markers match case- and dash-insensitively (`fold_text`). Pack validation refuses an included path that is not written plainly (`./`, an empty part or a trailing `/`). The holdout snapshot is compiled from those bytes alone, in a worker thread (its blueprint agent runs start their own event loop) with the full snapshot's embedding model and dimensions (reusing `--cache`), so its facets, glossary, blueprints and guide are induced without the held-out material; it is checked again and only then published as `<out>/holdout.arkive.db`. The cache cannot serve the full corpus's induction results to the holdout build, because each induction request covers the documents it was built from. Three arms draft the deliverable from the task's `run_notes` and `memo_instructions`: no tools, a five-chunk search tool, and the full MCP tools. Completed drafts get the task's rubric checks and a blinded side-by-side judge against the reference. |

The evaluation harness compiles the holdout snapshot through `compile_corpus`
itself, the one real OpenAI build outside `arkive-rag build`. Blinded drafts and
`blind_key.json` are written to `<out>/blind-drafts`. `tests/test_evals.py`
covers pack validation, scoring, the rubric, the holdout guards, the run-status
contract, the layer 3 flow, the runner's input checks and the tool adapters
offline, using the example pack and a scripted OpenAI HTTP transport.

### Example corpora

`evals/example_corpora/` holds two small fictional corpora, so anyone can build
real snapshots without private documents. They are organised in different
ways, and induction gives them different facet schemas:

- `coastal-survey-projects`: one folder per survey project, each holding a
  proposal, a technical memorandum and reviewer feedback, with a curator note
  that records a missing review. `q102-technical-memorandum.pdf` has a bar
  chart whose per-bench volumes appear only in the image.
- `creator-campaigns`: one folder per creator, then one per campaign, holding
  campaign briefs and post-campaign reviews, with a casual root `AGENTS.md`.
  `harbourline-launch-deck.pptx`, the results deck sent after the campaign's
  flight, has a chart whose weekly reach values appear only in the image.

Both repeat deliverable types, reviewer feedback, defined and undefined
acronyms, one acronym with two meanings, tool names, stated attributes,
exact numbers and filenames. They are built with the ordinary `arkive-rag
build` command; the snapshots are not tracked.

### OpenCode check

`python -m evals.opencode_check SNAPSHOT... --out DIR [--model
provider/model] [--scenarios FILE] [--only ID] [--timeout S]` checks that a real
MCP client, OpenCode, can discover and use the Arkive tools with a real model,
non-interactively, on several topics at once. It needs Docker and
`OPENAI_API_KEY`, and it is not part of `pytest`.

- **Image.** `evals/opencode_check/Dockerfile` starts from the official
  `ghcr.io/anomalyco/opencode:2.3.0` image, pinned by digest. Its `opencode`
  binary is linked against musl, so the image stays on that Alpine base and
  adds Alpine's Python 3.14. The runner builds the context in a temporary
  directory: the arkive-rag wheel built from the checkout (`uv build`), the
  runtime requirements exported from `uv.lock` with hashes (`uv export`), and
  the files beside the Dockerfile. The image is tagged
  `arkive-opencode-check:<content hash>` and built only when that tag is
  missing. It holds no key, snapshot or corpus.
- **Client configuration.** `opencode.json` becomes the system configuration
  `/etc/opencode/opencode.json`, which outranks global and project files. It
  declares one local stdio MCP server, `arkive`, that runs `arkive-rag serve`
  on every snapshot under `/snapshots`, with `codemode: false` so the tools sit
  on the model's native tool list as `arkive_<tool>`. Its permissions deny
  every action and then allow `arkive_*`.
- **Startup.** `opencode run --standalone` sends the first model request
  without waiting for local MCP servers, so `arkive`, which needs several
  seconds to open its snapshots, is missing from the tool list.
  `run-scenario.sh` therefore starts a private `opencode serve`, waits until
  `opencode mcp list` reports `arkive` connected (120 s at most, each listing
  bounded to 10 s, exit 3 otherwise) and then runs the prompt with `opencode
  run --server ... --format json --auto -- PROMPT`, so a prompt that starts
  with a dash is still the message.
- **Isolation.** Each scenario runs in a fresh container, as the invoking
  user's UID and GID (`--user`), so the owner-only snapshot files stay
  readable for any host UID; the runner refuses to run as root.
  `run-scenario.sh` keeps HOME and the working directory under `/tmp`, so
  neither depends on an image account. The snapshots are mounted read-only and
  no other host path is mounted; all capabilities are dropped, with
  `no-new-privileges`. OpenCode's permission list denies every tool outside
  the `arkive` server. The containers use Docker's default network, so network
  egress is not restricted. The check has been run with a rootful Docker
  daemon; rootless Docker is untested. `OPENAI_API_KEY` is passed by name
  (`--env OPENAI_API_KEY`), so its value never appears in a command line, and
  reaches both the model requests and the `search` tool's query embeddings.
  The runner refuses an empty key, and replaces the key's value in the event
  stream and stderr before parsing them, so no file it writes holds the key.
  Each container is named `arkive-opencode-check-<run id>-<scenario id>`, with
  a random run ID, so concurrent runs do not collide. It is force-removed
  (`docker rm --force`) when its scenario ends, times out (`--timeout`, 600 s
  by default), fails or is interrupted; a removal that fails is printed and
  makes the exit status 1.
- **Scenarios.** `scenarios.json` (`format` `arkive-opencode-scenarios/1`,
  validated by `ScenarioFile` in `scenarios.py`) is tied to the example
  corpora. A scenario has a prompt, the Arkive tools that must complete at
  least once, argument expectations (`non_empty`: some completed call passed
  a non-empty argument such as `filters`; `invalid_filter_recovery`: a call
  whose `filters` set a facet to the scenario's invalid value ended in a tool
  error, and a later call to the same tool on the same topic, setting that
  facet to another value, completed), the topics its calls must query, answer
  phrase groups matched as in eval packs, an optional requirement that the
  answer names every facet of its topics (read from the served snapshots,
  because facets are induced per build), and an optional page (`page_image`:
  topic, `doc_id`, `page_number`) whose `view_page` call must return an image.
  Document IDs follow sorted path order, so they are stable across builds of
  one corpus. Every scenario also requires that the run ended with an answer
  and no error event, and that no tool outside the `arkive` server was called. The shipped scenarios cover
  topic and facet discovery, three glossary lookups (a defined term, a term
  the corpus never defines and a term with two meanings), a filtered listing
  with a facet the agent finds itself, recovery from an invalid filter value,
  an exact number found with `grep`, a chart value read through `view_page`,
  a draft that starts from a blueprint, and a question that needs both topics.
- **Results.** The runner parses the newline-delimited JSON events
  (`tool_use`, `text`, `step_finish`, `error`); the final answer is the text
  written after the last tool call. It writes `results.json` (per scenario:
  tool calls with arguments, status and output, the final answer, each check's
  outcome, the tokens and OpenCode's cost estimate from `step_finish` events,
  the exit code), `report.md` and the raw event stream of each scenario under
  `events/`, all owner-only (`write_private_text` in `evals/run.py`, which
  refuses a symlink at the destination; `events/` must be a real directory).
  OpenCode can omit the last step's `step_finish`, so token and cost totals
  are lower bounds, and the `search` tool's query embeddings are not in them. The exit status is 1 when any scenario fails or
  cannot run. Before building an image or starting a container, the runner
  refuses an invalid scenario file, an `--out` that overlaps the scenario file
  or a snapshot (the rule `evals.run` applies to its inputs), a snapshot that
  does not open, two snapshots of one topic, a scenario topic no snapshot
  serves, an unknown `--only` ID, a missing or empty key, a root caller, and a
  symlink where an output file or `events/` would be written.

OpenCode 2.3.0 receives `view_page`'s image as a file part of the tool result
(`image/webp`), and the scenario checks that the `view_page` call for the chart's
page returned it. The chart value is also in
the page's visual summary, which `view_page` returns as text, so a correct
answer alone does not prove that the model read the image.

The README states what a pass proves. `tests/test_opencode_check.py` covers the
scenario file, event parsing, every kind of check failure, the container
command, the runner's input refusals, output links, key redaction and container
removal (including an interrupt) offline, with a fake Docker and no key.

## Verification

All commands run from a checkout root. Keep the environment and tool caches
outside the checkout, for example:

```sh
export UV_PROJECT_ENVIRONMENT=/tmp/arkive-venv MYPY_CACHE_DIR=/tmp/arkive-mypy \
  RUFF_CACHE_DIR=/tmp/arkive-ruff PYTHONDONTWRITEBYTECODE=1
uv sync --locked
uv run --locked ruff check src tests evals
uv run --locked ruff format --check src tests evals
uv run --locked mypy
uv run --locked pytest -o cache_dir=/tmp/arkive-pytest --basetemp=/tmp/arkive-pytest-tmp
uv build --out-dir /tmp/arkive-dist
```

The test suite makes no network calls. It needs LibreOffice (`soffice`) for the
Word and PowerPoint conversion test; that test is skipped when `soffice` is
absent.
