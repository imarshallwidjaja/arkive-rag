# `tests/` instructions

Refines root `AGENTS.md` for the offline `pytest` suite (`pythonpath = ["."]`, `--strict-markers`, `--import-mode=importlib`).

## Suite-wide invariants

1. **Strictly offline:** Every test must pass without network access or a real `OPENAI_API_KEY`. Drive builds, induction, and evaluation runners through the fake `LanguageGateway` or fake OpenAI client helpers in `tests/support.py`.
2. **Isolate CLI tests from root `.env`:** `arkive-rag` and `evals.run` call `load_dotenv_if_present()`, which reads `.env` from the current working directory when `OPENAI_API_KEY` is unset. Any test asserting missing-credential failure MUST call both `monkeypatch.delenv("OPENAI_API_KEY", raising=False)` and `monkeypatch.chdir(tmp_path)`; otherwise an untracked developer `.env` at the repo root will leak a live key into `pytest`.
3. **External caches and temp dirs:** Always pass `-o cache_dir="$SCRATCH/pytest-cache" --basetemp="$SCRATCH/pytest-tmp"` and `PYTHONDONTWRITEBYTECODE=1` so tests leave the checkout clean.
4. **LibreOffice dependency:** Office conversion tests (`.docx`/`.pptx`) require `soffice` on `PATH` and skip cleanly when absent. Do not add live Docker or live OpenAI dependencies to `tests/`.
5. **Ignore untracked legacy folders:** Only the top-level `test_*.py` files, `support.py`, and `acoustic-noise-report-sections.json` are tracked in Git. Do not add tests under untracked legacy subdirectories (`unit/`, `integration/`, `end_to_end/`).

## Owning test-file map

Place new assertions in the file that owns the invariant rather than creating standalone regression files:

| File | Owns |
| --- | --- |
| `support.py` | Shared fake `LanguageGateway`, synthetic corpora, deterministic embeddings, and snapshot builders. |
| `test_passages.py` | Pure Markdown/table/prose unit parsing, fence rules, and 4-gram recurrence pinning tests (uses `acoustic-noise-report-sections.json`). |
| `test_compiler.py` | Corpus discovery, symlink/path rejection, `DocumentDate` parsing, `BuildCache` render+LLM caching, fidelity retries, cross-page table stitching, and `MAX_SECTION_CHARS` splitting. |
| `test_induction.py` | Bounded batching (`<= 90_000` chars), facet induction/pruning, uncapped canonical types, version families, feedback lessons, glossary support checks, and `<= 16_000`-char guide verification. |
| `test_fts_and_snapshot.py` | Schema `6.0` SQLite integrity, read-only open checks, FTS5/RRF search, version-family deduplication, `prepare_draft` packing/outline caps/inline reserve/related-type backfill, and 45 KB / 1,600-line bounds. |
| `test_llm.py` | `OpenAIGateway`, `BuildCache`, structured outputs, Decisions API batching, retry/refusal handling, and `--max-cost` spend guard. |
| `test_mcp_and_cli.py` | All 10 MCP tools, compact JSON bounds, `server_instructions`, `streamable-http` `--allow-nonlocal` guard, and Typer CLI subcommands. |
| `test_evals.py` | `arkive-eval-pack/3` validation, example pack integrity, Layer 1-3 harnesses, holdout leak guards, `OverlapSources`, frozen oracle rules, and atomic output publication. |
| `test_opencode_check.py` | `scenarios.json` validation, event parsing, truncation/allowlist failure checks, key redaction, and Docker command construction (fake Docker). |

## Running a focused test

```sh
uv run --locked pytest -o cache_dir="$SCRATCH/pytest-cache" --basetemp="$SCRATCH/pytest-tmp" \
  tests/test_fts_and_snapshot.py -k "prepare_draft"
```
