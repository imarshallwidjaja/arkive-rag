# Architecture

Arkive RAG turns a folder of consulting documents into one SQLite file, the
*snapshot*, and serves one or more snapshots to AI agents over MCP. Building a
snapshot calls OpenAI models. Reading one needs no network, except that the
`search` tool embeds each query with the OpenAI embedding model.

## Modules

| Module | Owns |
| --- | --- |
| `models.py` | Domain types, OpenAI structured-output schemas, and the `ArkiveError` hierarchy. |
| `llm.py` | The request protocols (`StructuredRequest`, `AgentRequest`), the provider-neutral `Tool`, the `LanguageGateway` protocol, `OpenAIGateway` (every provider call: structured Responses calls, Agents SDK runs, Decisions, embeddings, the retry policy), `BuildCache`, `UsageMeter` with the spend limit, and `.env` loading. |
| `compiler.py` | Corpus discovery, rendering, the transcription fidelity gate, section stitching, chunking, cited synthesis, and embedding. Its request types (`PageTranscription`, `DocumentCatalogue`, `CorpusSynthesis`) own every prompt, output schema and reasoning effort it sends. Produces a `CompiledCorpus`. |
| `snapshot.py` | The SQLite schema, the atomic writer, and `TopicSnapshot`, the read-only query surface (hybrid search, grep, sections, pages). |
| `server.py` | `TopicCatalog` (open snapshots keyed by topic) and the eight MCP tools. |
| `cli.py` | The `arkive-rag` command. It is the only package module that reads `.env` and constructs `OpenAIGateway`; the evaluation harness in `evals/` does the same for its own runs. |

Dependencies point one way: `cli` → `server`, `compiler`, `snapshot` → `llm`,
`models`. The compiler receives a `LanguageGateway`, so tests drive it with a
deterministic fake and no network. Only `llm.py` imports the OpenAI SDK and the
OpenAI Agents SDK (`openai-agents`); `compiler.py` and `evals/` do not.

## Corpus layout

Every non-hidden file under the corpus directory must be `.pdf`, `.docx`,
`.pptx` or `.md`. Any other file stops the build with `UnsupportedFormatError`
before the first model call. Paths starting with `.` are skipped. A symlink
anywhere in the corpus stops the build with `UnsafePathError`, and each source
is read once without following symlinks; Word and PowerPoint files are
converted from a private copy of those bytes. The snapshot and the build cache
must be different files outside the corpus directory.

- The first folder under the corpus root is the document's `project_id`
  (`p102-estuary-multibeam/P102 summary.docx` belongs to `p102-estuary-multibeam`). Files directly
  in the root belong to project `root`.
- Every `AGENTS.md` is a curator note. Its text is stored verbatim in the
  topic guide, and it is also compiled as a `guidance` document (client
  `curator`, chunk kind `guidance`, `is_curator_note`), so status questions
  such as "where is the reef case study?" match the note in both
  `search` and `grep`. Other documents the model classes as `guidance` are
  catalogued and cited like any other document.
- Documents are numbered in sorted path order: `doc_001`, sections
  `doc_001_sec_001`, chunks `doc_001_chk_0001`. The same corpus produces the
  same IDs.

## Build pipeline

`compile_corpus` compiles documents in a thread pool (`--max-concurrency`,
default 4) and keeps the output in path order. Each PDF, Word and PowerPoint
file follows one path:

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
   their visible section numbers.
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
   values. Because each request carries the previous page's transcript tail, a
   changed page also changes the cache keys of the later pages of its document.
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
5. **Catalogue.** `gpt-6.1-sol` at low effort returns the document's title,
   client, lifecycle stage, date and summary. Curator notes skip this call.
6. **Chunk.** `build_chunks` packs paragraphs and callouts into prose chunks of
   up to 1,800 characters, splitting an oversized paragraph at sentence ends. A
   table up to 2,400 characters is one `table` chunk; a larger table is split on
   row boundaries into pieces of up to 2,000 characters, each repeating the
   header and separator rows. Each visual transcript is its own `visual` chunk.
   No chunk exceeds 6,000 characters: an oversized sentence, table row or
   visual is wrapped at line breaks or spaces, and table pieces keep the
   header. Every chunk carries a breadcrumb naming project, client, stage,
   document, page range and section path, and its own page span.

Markdown files skip steps 1-3: `parse_markdown_blocks` splits them
deterministically, keeping a fenced code block as one literal paragraph, and
they have no page images for `view_page`.

After all documents compile, `gpt-6.1-sol` at medium effort writes the topic
guide from the catalogue and curator notes. Every bracketed `[doc...]` token
must have the exact form `[doc_XXX:doc_XXX_sec_YYY]` and resolve to a section
of a catalogued (non-curator) document, and at least one valid citation is
required when such a section exists. One
retry lists the invalid and valid IDs; a second failure raises
`InvalidCitationError`. Finally every chunk is embedded as
`breadcrumb + "\n" + content` with `text-embedding-3-large` at 1,536 dimensions.

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
  source set never answers a request over another;
- `judge`: operation, decision model, evidence and predicates;
- embeddings: model, dimensions and the batch's texts.

A rebuild of an unchanged corpus with the same cache makes no provider
requests. The first valid entry stored under a key is canonical: when two build
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

Tables (schema version `2.0`): `corpus_meta`, `curator_notes`, `documents`
(including the SHA-256 of each source file), `sections`, `pages` (native text,
visual summary and WebP per page), `chunks` (content, breadcrumb and float32
embedding), and the FTS5 table `chunks_fts` with NFKC-normalised content and
breadcrumb (`porter unicode61 remove_diacritics 2`).

`corpus_meta` records `corpus_fingerprint` and `arkive_version` (the package
version that wrote the file). The corpus fingerprint is the SHA-256 hex digest
of the lines `rel_path NUL sha256`, one per document sorted by `rel_path` and
joined by newlines. It depends only on source paths and bytes, not on models,
prompts or build time, so it is the content address of the corpus a snapshot
was built from (`CompiledCorpus.corpus_fingerprint`).

`TopicSnapshot.open` uses a read-only SQLite URI with the path
percent-encoded, so names containing `%`, `?` or `#` open the right file. It
checks the schema version, metadata and embedding sizes, and loads chunk
metadata and the embedding matrix into memory. It recomputes the fingerprint
from the `documents` table and raises `SnapshotIntegrityError` when a recorded
fingerprint differs. A `2.0` snapshot written before these keys existed opens
with the recomputed fingerprint and an empty `arkive_version`. `list_topics`
and `arkive-rag inspect` report both values; `inspect` also lists each
document's `sha256`.

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
  score and both ranks. A blank query or an invalid filter or `top_k` raises
  `InvalidFilterError` before the query is embedded.
- **grep** is a literal, case-insensitive line search over section Markdown.
  Pattern and text are NFKC-normalised and Unicode dashes fold to `-`, so
  `2016-2017` finds `2016–2017` and `km2` finds `km²`. A snippet is at most
  400 characters of the line, centred on the first match.
- `project` and `stage` filters must name values present in the snapshot;
  anything else raises `InvalidFilterError` listing the valid values.

## MCP server

`arkive-rag serve a.arkive.db b.arkive.db` opens every snapshot into a
`TopicCatalog` and refuses two snapshots with the same topic. The tools are
`list_topics`, `get_topic_guide`, `list_documents`, `get_document_outline`,
`search`, `grep`, `read_section` and `view_page`. `view_page` returns the
caption as text content and the page as `image/webp` image content. Domain
errors become tool errors whose message names the valid topics, projects,
stages or pages.

Over `streamable-http` the server has no authentication, so `serve` refuses a
host other than `127.0.0.1`, `localhost` or `::1` unless `--allow-nonlocal` is
given. The server reports the package version to clients.

`search` needs `OPENAI_API_KEY` in the server environment and all served
snapshots must share one embedding model and dimension. Without a key the
server still starts; `search` returns an error that points to the offline
tools.

## Evaluation

`evals/` is a harness outside the `arkive_rag` package; it is not in the wheel.
`python -m evals.run --pack <file> --snapshot <db> --out <dir> [--corpus <dir>]`
runs an eval pack and writes `results.json`, `report.md` and, for layer 3,
`blind-drafts/` to `--out`. `--out` is required and should be a private
directory outside the repository; the runner creates it with mode 0700 and
writes 0600 files. Every layer calls OpenAI, so it needs `OPENAI_API_KEY`.
Before creating any output or calling any model the runner checks that the pack
defines the requested layers, that layers 2 and 3 have `--corpus`, that
`--out` and `--cache` overlap none of the pack, snapshot and corpus paths, that
the snapshot's `corpus_fingerprint` matches the pack's when the pack sets one,
and that the fingerprint of the `--corpus` files (found by `discover_corpus`)
equals the snapshot's.

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
(`search_embed_calls`). `--max-cost` sets one spend limit over every model call
of the evaluation; reaching it is not a run outcome but stops the evaluation
with an error, the usage line and a non-zero exit, and writes no report.
`--trace` turns on hosted Agents SDK tracing, and `results.json` (`usage`) and
`report.md` (Usage) record calls, tokens and the estimated cost per operation.

An eval pack is a JSON file validated by `EvalPack` in `evals/schema.py`
(`format` `arkive-eval-pack/1`). It holds `retrieval_cases`, `qa_cases` and an
optional `holdout_memo` task. Expected facts are *groups* of alternative
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
| 2 | `agents.py` | An agent conversation answers each QA case with raw-file tools (text grep, page text, page images over the files `discover_corpus` accepts) and with the eight Arkive MCP tools. Each completed answer gets deterministic phrase coverage (`required_groups`) and a blinded judge score; the judge grades citations against the case's `source_anchors`. Raw-file extraction runs once before timing. |
| 3 | `agents.py` | One deliverable is held out. Before any file is copied or any model is called, the reference document (`reference_doc_id`) must carry a `held_out_markers` entry in its project or path, and every file under `included_paths` (selected through `discover_corpus`, so symlinks are refused) must have no marker, not belong to the reference's project and not have the reference's bytes. The holdout snapshot is compiled from those bytes in a worker thread with the full snapshot's embedding model and dimensions (reusing `--cache`), checked again, and only then published as `<out>/holdout.arkive.db`. Three arms draft the deliverable from the task's `run_notes` and `memo_instructions`: no tools, a five-chunk search tool, and the full MCP tools. Completed drafts get the task's rubric checks and a blinded side-by-side judge against the reference. |

The evaluation harness compiles the holdout snapshot through `compile_corpus`
itself, the one real OpenAI build outside `arkive-rag build`. Blinded drafts and
`blind_key.json` are written to `<out>/blind-drafts`. `tests/test_evals.py`
covers pack validation, scoring, the rubric, the holdout guards, the run-status
contract, the layer 3 flow, the runner's input checks and the tool adapters
offline, using the example pack and a scripted OpenAI HTTP transport.

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
