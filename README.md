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
uv run --locked arkive-rag search bathymetry.arkive.db "nine-year median rainfall"
uv run --locked arkive-rag serve bathymetry.arkive.db other.arkive.db
```

`serve` uses stdio by default; `--transport streamable-http` listens on
`127.0.0.1:8000` unless `--host` and `--port` say otherwise. The HTTP server
has no authentication, so a host other than `127.0.0.1`, `localhost` or `::1`
also needs `--allow-nonlocal`. Without `OPENAI_API_KEY` the server still starts
and every tool except `search` works.

The snapshot and `--cache` files must be different files outside the corpus
directory. A corpus may not contain symlinks.

Each top-level folder in the corpus is a project. An `AGENTS.md` anywhere in the
corpus is a curator note: it appears in the topic guide and is searchable.

## Private corpora, snapshots and eval packs

This repository holds code, the eval pack format and the eval runners. It holds
no client documents, client facts or personal data. Keep these outside it, in a
private store such as a private Git LFS or DVC repository, or a private bucket:

- the corpus folders;
- eval packs (`*.eval.json`) that state facts from a corpus;
- snapshots, named `<topic>-<corpus_fingerprint[:12]>.arkive.db`.

`arkive-rag inspect` prints the snapshot's `corpus_fingerprint`, the
`arkive_version` that wrote it, and the `sha256` of every source document. The
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
`--out` is required. Outputs quote corpus facts, so choose a private directory
outside this repository; the runner refuses an `--out` or `--cache` that
overlaps the pack, snapshot or corpus, and writes owner-only files.
`evals/example_pack.eval.json` is a fictional pack that shows the format.

- [Product goal and end-to-end system walkthrough](docs/walkthrough.md)
- [Architecture, pipeline, evaluation and verification](docs/architecture.md)

Agent working rules live in `AGENTS.md`.
