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
With `--max-cost 5` (a positive finite number of USD), no provider call starts once the estimated spend has
reached USD 5, and the build stops with an error. Calls already in flight can
take the total past the limit, and the estimate ignores the long-context
surcharge. `--trace` sends traces of agent runs to the OpenAI dashboard (off by
default); it fails if `OPENAI_AGENTS_DISABLE_TRACING` is set to `1` or `true`
(in any letter case).

The build reads every document, then induces from the whole corpus up to six
facets to filter by (path-bound when the folders already encode them, such as
projects or campaigns, otherwise judged per document, such as client or
author), a cited glossary, and a blueprint for each recurring deliverable type
(at most eight). A blueprint holds a section outline, conventions and reviewer
pitfalls stated generically, and standing passages: house text such as a
method background, a reference entry or a disclaimer. Code finds the
candidates: a paragraph, list item or table header row of one exemplar section
that recurs in another exemplar document (at least 0.8 of its word 4-grams, or
the same table header row), under a different top-level folder when the
exemplars span several folders. The blueprint agent only selects and labels
candidates, so a stored passage is exemplar text exactly as written. Code
cannot know that two folders are different projects, or that a recurring
sentence holds no project facts; the agent is told to leave out candidates
specific to one project, client, place, period or result.
`inspect` prints them. `search` and `grep` take repeated `--filter FACET=VALUE`
(a canonical value or an alias; several facets combine with AND), and the MCP
tools `list_documents`, `search` and `grep` take the same `filters` mapping.
An `AGENTS.md` anywhere in the corpus is a curator note: it guides induction,
appears in the topic guide and is searchable, but is never cited. With
`--cache`, rebuilding an unchanged corpus makes no provider calls. Snapshots
use schema `3.0`; a snapshot from an earlier version is rebuilt from its
corpus.

The MCP server has ten tools. `get_topic_guide` returns the guide, curator
notes, facets, glossary terms and a summary of each blueprint;
`get_blueprint(topic, deliverable_type)` returns one blueprint in full. Its
instructions tell a drafting agent to follow the blueprint, take every
project-specific fact from the user's inputs, reproduce the standing passages
word for word without commentary on their origin (adapting or leaving out a
passage that holds a project fact the user did not give, since the user's
inputs take precedence), and reuse corpus names and references only from
those passages.

## Private corpora, snapshots and eval packs

This repository holds code, the eval pack format and the eval runners. It holds
no client documents, client facts or personal data. Keep these outside it, in a
private store such as a private Git LFS or DVC repository, or a private bucket:

- the corpus folders;
- eval packs (`*.eval.json`) that state facts from a corpus;
- snapshots, named `<topic>-<corpus_fingerprint[:12]>.arkive.db`.

`arkive-rag inspect` prints the snapshot's `corpus_fingerprint`, the
`arkive_version` that wrote it, its facets, glossary, blueprints, build
settings and build report, and the `sha256` of every source document. The
fingerprint depends only on source paths and bytes, so you can check at any
time that a snapshot, a corpus folder and an eval pack belong together.
`.gitignore` excludes `*.arkive.db` and `*.eval.local.json`.

## Evaluation

The harness in `evals/` runs an eval pack against a snapshot and its corpus. It
compares retrieval arms, raw-file and Arkive agents, and holdout memo drafts,
and calls OpenAI models:

```sh
uv run --locked python -m evals.run --pack path/to/bathymetry.eval.json \
  --snapshot path/to/bathymetry-1a2b3c4d5e6f.arkive.db --corpus path/to/corpus \
  --out /tmp/arkive-eval-outputs
```

`--layer` is `all` (every layer the pack defines), `1`, `2` or `3`. Layers 2
and 3 need `--corpus`, whose files must still match the snapshot's
`corpus_fingerprint`; `--cache` reuses a build cache for the holdout snapshot.
`--max-cost` and `--trace` work as in `build`, and the usage and estimated cost
of the run are written to `results.json` and `report.md`. Before it creates
any output or calls any model, the runner checks the pack, the snapshot, the
retrieval filters and the holdout guards of every layer it will run. An
evaluation that reaches the spend limit, or fails after calling models, prints
its usage, stops with an error and writes no report.
`--out` is required. Outputs quote corpus facts, so choose a private directory
outside this repository; the runner refuses an `--out` or `--cache` that
overlaps the pack, snapshot or corpus, and writes owner-only files.
`evals/example_pack.eval.json` is a fictional pack that shows the format
(`arkive-eval-pack/2`); a retrieval case's `filters` use the facet names and
values `inspect` prints for the snapshot.

## Example corpora and the OpenCode check

`evals/example_corpora/` holds two small fictional corpora organised in
different ways: `coastal-survey-projects` (survey projects with proposals,
technical memoranda and reviewer feedback) and `creator-campaigns` (creators
and their campaigns with briefs, reviews and a launch deck). Build them like
any other corpus, with the snapshots outside the repository:

```sh
for c in coastal-survey-projects creator-campaigns; do
  uv run --locked arkive-rag build evals/example_corpora/$c \
    -o /tmp/arkive-examples/$c.arkive.db --cache /tmp/arkive-examples/$c.cache.db --max-cost 1.5
done
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
