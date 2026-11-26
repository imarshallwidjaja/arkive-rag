# `src/arkive_rag/` instructions

Refines root `AGENTS.md` for the installable `arkive_rag` Python package (Python `3.14`, strict `mypy`, `ruff` line length `100`).

## Module ownership

| Module | Owns |
| --- | --- |
| `models.py` | Domain models, `ArkiveError`, NFC path-hash `document_id(rel_path)`, `DocumentDate`, `corpus_fingerprint`, and quote-folding helpers. |
| `passages.py` | Pure Markdown/table/prose unit parsing and 4-gram recurrence matching; zero I/O or model calls. |
| `llm.py` | Fixed model IDs (`gpt-6.1-sol`, `gpt-6-luna`, `text-embedding-3-large`), `PRICES`, `LanguageGateway` protocol, `OpenAIGateway`, `BuildCache`, spend guard (`--max-cost`), and `.env` loading. |
| `compiler.py` | Pass 1 reading: corpus discovery (no symlinks), LibreOffice `.docx`/`.pptx` conversion, `pypdfium2` 2x WebP rendering + render cache, vision transcription with fidelity retries, cross-page section/table stitching, `MAX_SECTION_CHARS = 32_000` splitting, and per-document observation cards. |
| `induction.py` | Pass 2 corpus induction: batched organisation (`0-6` facets, uncapped canonical deliverable types), content-bound facet judging, version-family detection/ordering, per-feedback-doc verified lessons, cited glossary, recurring unit clusters with holder samples, and bounded topic guide (`<= 16_000` chars / `<= 24_000` JSON bytes). |
| `snapshot.py` | Schema `7.0` SQLite writer (`write_snapshot`, atomic `0600` rename), read-only `TopicSnapshot.open` integrity verifier, FTS5 + L2 cosine RRF search with version-family cap, and `prepare_draft` packet builder. |
| `server.py` | FastMCP server exposing the ten tools over `stdio` or `streamable-http` (`--allow-nonlocal` guard for non-loopback hosts) and `server_instructions` (including sole-topic summary). |
| `cli.py` | Typer CLI (`arkive-rag build | inspect | grep | search | serve`). |

## Non-negotiable package invariants

1. **Schema hard cuts:** Whenever a change alters SQLite tables, columns, or stored semantics, bump `SCHEMA_VERSION` in `snapshot.py` (currently `"7.0"`). Do not add read adapters for older schema versions; old snapshots are rebuilt from their corpus.
2. **Stable IDs and deterministic ordering:** Document IDs come strictly from `document_id(rel_path)` (`doc_<10hex>` over NFC-normalised POSIX relative path) so inserting or moving another file never renumbers existing IDs. Keep documents ordered by sorted relative path (`ordinal`) across compilation, induction, and SQLite queries.
3. **Corpus generality:** Never hardcode domain nouns, client names, folder conventions, or heading keywords into `compiler.py`, `passages.py`, `induction.py`, `snapshot.py`, or `server.py`. Every rule must hold across flat folders, nested project trees, and mixed `.pdf`/`.docx`/`.pptx`/`.md` collections with or without `AGENTS.md`.
4. **Prompt and tool-response ceilings:**
   - Every induction and judge prompt stays at or below `MAX_PROMPT_CHARS = 90_000` (roughly 30k tokens), regardless of corpus size.
   - Every MCP tool result must serialize as one compact JSON text block within `45_000` bytes and `1_600` lines (`snapshot.RESPONSE_MAX_BYTES`, `RESPONSE_MAX_LINES`) so clients like OpenCode never truncate tool outputs. Use pagination (`offset`/`limit`) or explicit `omitted_*` counts rather than unbounded dumps.
   - Internal eval callers that need full outlines or full section Markdown use `TopicSnapshot.full_outline` / `section_markdown`, not the paginated MCP surface.
5. **`prepare_draft` packet rules:**
   - Keep `prepare_draft` deterministic with zero generative LLM calls inside the MCP server (it calls query embeddings only when `OPENAI_API_KEY` is set, otherwise ranks with BM25 alone).
   - Preserve the packing fairness invariants: `EXEMPLAR_OUTLINE_MAX_BYTES = 8_000` prevents a 50+ page exemplar from evicting compact siblings; `PACKET_MIN_INLINE_BYTES = 16_000` reserves inline prose space before keeping a third exemplar's outline; cross-type backfill when `>= 2` same-type exemplars exist requires a shared content word in `deliverable_type` and `CROSS_TYPE_RELEVANCE = 0.6`.
   - **Recurrence is evidence, not permission:** Expose version-family counts (`recurring_unit_families`), dates, and version status as provenance. Never reintroduce a hard `N >= 2` reuse gate or static `Blueprint` whitelist.
6. **Curator note handling:** Any `AGENTS.md` inside a corpus is ingested with `is_curator_note = True`. It guides induction and is returned in `get_topic_guide` and `prepare_draft`, is searchable, and must never be cited by the induced guide, glossary, or feedback lessons.
7. **Cache-before-work:** `BuildCache` keys rendered pages and LLM responses by content/input hashes. Keep reading dependent on file bytes and format (with `rel_path` only for diagnostics) so unchanged rebuilds make zero provider calls and zero page renders.

## Focused verification

```sh
uv run --locked ruff check src/arkive_rag && uv run --locked mypy
uv run --locked pytest -o cache_dir="$SCRATCH/pytest-cache" --basetemp="$SCRATCH/pytest-tmp" \
  tests/test_compiler.py tests/test_passages.py tests/test_induction.py \
  tests/test_fts_and_snapshot.py tests/test_mcp_and_cli.py tests/test_llm.py
```
