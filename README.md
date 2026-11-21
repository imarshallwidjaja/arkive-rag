# Arkive RAG

![Raw documents compiled into a portable SQLite snapshot serving AI agents](docs/assets/arkive-concept-hero.png)

Arkive RAG compiles a curated folder of `.pdf`, `.docx`, `.pptx`, and `.md` reference documents into a single read-only SQLite snapshot (`*.arkive.db`, schema `6.0`), and serves one or more snapshots to AI agents over MCP.

Domain leads, engineers, and project teams place strong reference reports, proposals, slide decks, reviewer feedback, and optional `AGENTS.md` curator notes into any folder layout. `arkive-rag build` parses text, tables, and visual charts, induces corpus-specific facets, deliverable types, version families, verified reviewer lessons, recurring passages, and a cited glossary, and writes one portable database file. `arkive-rag serve` then gives any MCP-capable agent ten bounded tools for pinpoint Q&A and grounded deliverable drafting.

## Quickstart

Building a snapshot requires Python `3.14`, [uv](https://docs.astral.sh/uv/), LibreOffice (`soffice`, for `.docx` and `.pptx`), and an `OPENAI_API_KEY` (loaded automatically from `.env` in the working directory when unset).

```sh
uv sync --locked

# Compile a folder into a portable snapshot (keep outputs outside the corpus)
uv run --locked arkive-rag build path/to/corpus \
  -o /tmp/bathymetry.arkive.db \
  --cache /tmp/bathymetry-cache.sqlite \
  --max-cost 5

# Inspect facets, deliverable types, version families, lessons, and build report
uv run --locked arkive-rag inspect /tmp/bathymetry.arkive.db

# Run exact lookup or hybrid filtered search from the CLI
uv run --locked arkive-rag grep /tmp/bathymetry.arkive.db "harbour_dry_depth_thresh.tif"
uv run --locked arkive-rag search /tmp/bathymetry.arkive.db "median rainfall" \
  --filter project=p101-harbour-lidar

# Serve one or more snapshots to an MCP client (stdio default, or --transport streamable-http)
uv run --locked arkive-rag serve /tmp/bathymetry.arkive.db
```

Key operational facts:

- **Offline serving:** Without `OPENAI_API_KEY`, `serve` still starts and nine of the ten tools work (`prepare_draft` ranks with BM25 alone; only `search` needs query embeddings).
- **Local-only HTTP by default:** `--transport streamable-http` binds to `127.0.0.1:8000` with no authentication; non-loopback hosts require `--allow-nonlocal`.
- **Private corpora stay out of Git:** Store client documents, snapshots (`*.arkive.db`), and private eval packs (`*.eval.local.json`) in a private store outside this repository.

## Documentation

Full guides, diagrams, worked examples from public fictional collections, and verified benchmark results live in [`docs/`](docs/README.md):

- [Documentation index](docs/README.md) — Reading guide and visual diagram index.
- [System walkthrough](docs/walkthrough.md) — Product goal and end-to-end trace from folder to MCP response.
- [Parsing, induction, and build pipeline](docs/parsing-and-build.md) — Document conversion, visual transcription, caching, bounded induction, and worked public-corpus examples.
- [MCP serving, `prepare_draft`, and cost scaling](docs/mcp-serving.md) — The ten MCP tools, 45 KB drafting packet anatomy, provenance rules, hosted models, and scaling costs.
- [Evaluation harness and benchmark results](docs/evaluation-and-benchmarks.md) — Layer 1-3 harness, live OpenCode 2.3.0 container check, and verified results across four fictional corpora (up to 126 files).
- [Architecture and verification reference](docs/architecture.md) — Full module, schema, and offline verification specification.

Agent contributor instructions use progressive discovery starting at [`AGENTS.md`](AGENTS.md), refined by directory-level `AGENTS.md` files in `src/arkive_rag/`, `evals/`, `tests/`, and `docs/`.
