# Arkive RAG

Arkive RAG compiles a human-curated folder of PDF, Word, PowerPoint and
Markdown reference documents into a single read-only SQLite snapshot
(`*.arkive.db`), and serves one or more snapshots to AI agents over MCP.

A domain lead, engineer, or marketing manager can drop their strongest
reference reports, slide decks, proposals, and feedback notes into a folder,
add optional plain-language context in `AGENTS.md`, and build a portable
snapshot that helps agents both answer pinpoint factual/visual questions and
draft new deliverables aligned to that team's craft.

Building a snapshot needs Python 3.14, [uv](https://docs.astral.sh/uv/),
LibreOffice (`soffice`, for `.docx` and `.pptx`) and an `OPENAI_API_KEY`. The
command reads `.env` from the working directory when the variable is not
already set.

```sh
uv sync --locked
uv run --locked arkive-rag build path/to/corpus -o bathymetry.arkive.db --cache build-cache.db
uv run --locked arkive-rag inspect bathymetry.arkive.db
uv run --locked arkive-rag grep bathymetry.arkive.db "harbour_dry_depth_thresh.tif"
uv run --locked arkive-rag search bathymetry.arkive.db "nine-year median rainfall" \
  --filter project=p101-harbour-lidar
uv run --locked arkive-rag serve bathymetry.arkive.db other.arkive.db
```

`serve` uses stdio by default; `--transport streamable-http` listens on
`127.0.0.1:8000` unless `--host` and `--port` say otherwise. The HTTP server
has no authentication, so a host other than `127.0.0.1`, `localhost` or `::1`
also needs `--allow-nonlocal`. Without `OPENAI_API_KEY` the server still starts
and every tool except `search` works.

The snapshot and `--cache` files must be different files outside the corpus
directory. A corpus may not contain symlinks.

`build` uses `gpt-6.1-sol`, `gpt-6-luna` on the Decisions API and
`text-embedding-3-large`; the models are fixed in code, and the snapshot
records them with the reasoning effort of each operation. It prints the
provider calls, tokens and estimated cost of the build, and a build report
of what it dropped or pruned without failing.
With `--max-cost 5` (a positive finite number of USD), the next provider attempt
is blocked once estimated spend reaches USD 5. The build then stops with an
error; if its last request takes the estimate past the limit, it finishes normally.
Calls already in flight can take the total past the limit, and the estimate ignores the long-context
surcharge.

The build reads the corpus and produces filter facets, a canonical
deliverable type for every document (with the types a feedback document
reviews), version families of drafts and copies, reviewer lessons from
feedback documents, the text units that recur across documents, and a cited
glossary. Every induction prompt stays within about 30k tokens however large
the corpus, and no step caps the number of deliverable types. `inspect` prints
the facets, glossary, deliverable types, version families, feedback lessons
and build report. See the
[build pipeline](docs/architecture.md#build-pipeline) for the induction and
validation rules. `search` and `grep` take repeated `--filter FACET=VALUE`
(a canonical value or an alias; several facets combine with AND), and the MCP
tools `list_documents`, `search` and `grep` take the same `filters` mapping.
An `AGENTS.md` anywhere in the corpus is a curator note: it guides induction,
appears in the topic guide and is searchable. The guide, glossary and lessons
never cite it; an agent answering a question from a note cites it
like any other document. With `--cache`, rebuilding an unchanged corpus makes
no provider calls and renders no page again. A document's ID is a hash of its
path, so adding or moving another file leaves it unchanged. Snapshots use
schema `6.0`. Rebuild older snapshots from their corpus.

The MCP server has ten tools. `get_topic_guide` returns the guide, curator
notes, facets, glossary terms and the deliverable types with their counts.
`prepare_draft(topic, project_inputs, deliverable_type, filters)` returns a
drafting packet of at most 45,000 bytes: up to three prior exemplars ranked
against the user's inputs, each with its complete outline and as much whole
section text as fits, other candidates, reviewer lessons, curator notes and
drafting guidance. Each section reports, as provenance, how many independent
version families share its recurring text, so the agent can tell established
house text from a clause one project used. It ranks with BM25 alone when the
server has no `OPENAI_API_KEY`. `list_documents` returns pages of at most 100
documents, and `search` shows at most three chunks of one version family
before it fills the list with more. Every tool result is one compact JSON
text block of at most 45,000 bytes and 1,600 lines, below the size at which a
client such as OpenCode cuts a tool result; a long outline or section comes in
pages through `get_document_outline` and `read_section` with `offset`.

## Private corpora, snapshots and eval packs

This repository holds code, the eval pack format and the eval runners. It holds
no client documents, client facts or personal data. Keep these outside it, in a
private store such as a private Git LFS or DVC repository, or a private bucket:

- the corpus folders;
- eval packs (`*.eval.json`) that state facts from a corpus;
- snapshots, named `<topic>-<corpus_fingerprint[:12]>.arkive.db`.

`arkive-rag inspect` prints the snapshot's `corpus_fingerprint`, the
`arkive_version` that wrote it, its facets, glossary, deliverable types,
version families, feedback lessons, the number of recurring units, build
settings and build report, and the `sha256` of every source document. The
fingerprint depends only on source paths and bytes, so you can check at any
time that a snapshot, a corpus folder and an eval pack belong together.
`.gitignore` excludes `*.arkive.db` and `*.eval.local.json`.

## Evaluation

The harness in `evals/` runs an eval pack against a snapshot and its corpus. It
compares retrieval arms, raw-file and Arkive agents, and holdout drafts,
and calls OpenAI models:

```sh
uv run --locked python -m evals.run --pack path/to/bathymetry.eval.json \
  --snapshot path/to/bathymetry-1a2b3c4d5e6f.arkive.db --corpus path/to/corpus \
  --out /tmp/arkive-eval-outputs
```

`--layer` is `all` (every layer the pack defines), `1`, `2` or `3`. Layer 3
runs every holdout in the pack, or the one `--holdout ID` names, for
`--drafts N` rounds (default 1), and writes each holdout's snapshot and each
round's blind drafts to `<out>/holdouts/<holdout_id>/`; layer 3 refuses an
`--out` that already holds a `holdouts` directory. Four drafting arms always
run in every round (no tools, chunk search, raw-file tools over the included
files, the Arkive MCP tools), and a fifth, with the holdout's `exemplar_paths`
documents in the prompt, runs when `exemplar_paths` is not empty. With
`exemplar_paths`, the report also says how many of those documents
`prepare_draft` surfaces for the holdout's inputs. Each completed draft is
measured with two lexical-overlap measures over its prose: house-text recall
and copied-text contamination. Layer 1 compares BM25, dense and hybrid
ranking with the search agents get.
Layers 2 and 3 need `--corpus`, whose files must still match the snapshot's
`corpus_fingerprint`; `--cache` reuses a build cache for the holdout snapshots
and must not overlap `--out`.
`--max-cost` works as in `build`; `--trace` sends traces of the agent runs to
the OpenAI dashboard (off by default), and fails if
`OPENAI_AGENTS_DISABLE_TRACING` is set to `1` or `true` (in any letter case).
The usage and estimated cost
of the run are written to `results.json` and `report.md`. Before it creates
any output or build cache or calls any model, the runner checks the pack, the
snapshot, the retrieval filters and expected documents, the Q&A source anchors
and their evidence size, the holdout guards of every layer it will run, and
that LibreOffice is on the path when a selected layer converts a Word or
PowerPoint file. It also checks the output and cache paths: an existing
`--out` must be a directory, not a symlink, that you can write and search,
with no directory named `results.json` or `report.md` in it; an absent `--out`
must not lie below an existing file, and its nearest existing directory must
be writable and searchable. When layer 3 runs, `--cache` must be either an
absent path that passes the same test or a regular file, not a symlink, that you can read and write, in a
directory you can write and search (SQLite writes `-wal` and `-shm` files
beside it). Each run records its tool calls
(name, arguments, size and start of the returned text), and the Q&A judge
grades each answer against the reference facts and the snapshot text of the
anchored pages, listing its claims as supported, contradicted or
unverifiable. A judge call that fails (invalid or refused output, a provider
failure) is recorded on its Q&A row or draft round as not judged, with the
reason, and the evaluation continues.
An agent run with status `provider_error`, `incomplete`, `empty_answer` or
`max_turns` is recorded without credit and evaluation continues. A failure
that stops the evaluation, such as a spend limit blocking the next attempt or
an unexpected tool failure, prints usage, exits non-zero and writes no report;
a report that an earlier run left in `--out` stays as it was.
Invalid input is rejected before model calls and output creation.
`--out` is required. Outputs quote corpus facts, so choose a private directory
outside this repository; the runner refuses an `--out` or `--cache` that
overlaps the pack, snapshot or corpus, and writes owner-only files. It
publishes each output file by writing a temporary file in the same directory
and renaming it over the old one, so a failed write leaves an earlier
`results.json` or `report.md` as it was.
`evals/example_pack.eval.json` is a fictional pack that shows the format
(`arkive-eval-pack/3`; the runner reads no older format) and is the offline
tests' fixture; a retrieval case's `filters` use the facet names and values
`inspect` prints for the snapshot. `evals/example_packs/` holds a pack for each
example corpus below.

## Example corpora and the OpenCode check

`evals/example_corpora/` holds three small fictional corpora organised in
different ways: `coastal-survey-projects` (survey projects with proposals,
technical memoranda and reviewer feedback), `creator-campaigns` (creators
and their campaigns with briefs, reviews and a launch deck) and
`acoustic-consultancy-files` (one flat folder of Word and PDF files with no
curator note, dated only by embedded file metadata: noise reports and
quotations, which recur, and one single-document type; the older and newer
reports differ in layout, share house paragraphs and a caveat that varies by
one instrument name, only the newest report has a data-handling section, and
one report has a chart whose values exist only in the image). Build them like
any other corpus, with the snapshots outside the repository; the OpenCode
check below uses the first two:

```sh
for c in coastal-survey-projects creator-campaigns; do
  uv run --locked arkive-rag build evals/example_corpora/$c \
    -o /tmp/arkive-examples/$c.arkive.db --cache /tmp/arkive-examples/$c.cache.db --max-cost 1.5
done
```

`evals/example_packs/<collection>.eval.json` is a fictional eval pack for each
of the three collections, built for real paid evaluation runs against its
snapshot. Each pack has four Q&A cases (a lookup, a multi-document question, an abstention and
a vocabulary question) and holdouts: the Fennick Canal proposal and technical
memorandum in `coastal-survey-projects`, the Kettlecraft winter campaign brief
in `creator-campaigns`, and two noise reports, Pellow Mill and Corvane depot,
in `acoustic-consultancy-files`. Layers 2 and 3 need the collection folder as
`--corpus`:

```sh
uv run --locked python -m evals.run \
  --pack evals/example_packs/coastal-survey-projects.eval.json \
  --snapshot /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  --corpus evals/example_corpora/coastal-survey-projects \
  --out /tmp/arkive-eval-outputs/coastal-survey-projects --max-cost 10
```

The OpenCode check runs the scenarios in `evals/opencode_check/scenarios.json`
against those snapshots inside OpenCode 2.3.0, one fresh Docker container per
scenario, with only the Arkive MCP tools allowed. It needs Docker and
`OPENAI_API_KEY`, which the containers receive through their environment. Each
container runs as your UID and GID with the snapshots mounted read-only and no
other host path. Network egress from the containers is not restricted. It has
been run with a rootful Docker daemon; rootless Docker is untested.

```sh
uv run --locked python -m evals.opencode_check \
  /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  /tmp/arkive-examples/creator-campaigns.arkive.db --out /tmp/arkive-opencode-check
```

`--model` picks the OpenCode model (default `openai/gpt-6.1-sol`) and `--only`
runs one scenario. The runner builds the image when needed and writes
`results.json`, `report.md` and each scenario's event stream to `--out`, with
the key's value redacted. `--out` must not overlap a snapshot or the scenario
file; the results quote corpus text, so keep them out of the repository. It
exits with status 1 when any scenario fails. A pass shows that this OpenCode
version with this model can discover and use the Arkive tools on two topics at
once; it is not a conformance proof for every MCP client.

- [Product goal and end-to-end system walkthrough](docs/walkthrough.md)
- [Architecture, pipeline, evaluation and verification](docs/architecture.md)

Agent working rules live in `AGENTS.md`.
