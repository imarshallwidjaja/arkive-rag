# `evals/` instructions

Refines root `AGENTS.md` for the evaluation harness (`evals/*.py`), public example fixtures (`example_corpora/`, `example_packs/`), and containerised OpenCode acceptance runner (`opencode_check/`).

## Module and directory ownership

| Path | Owns |
| --- | --- |
| `schema.py` | Format `arkive-eval-pack/3` Pydantic schema (only format accepted), emphasis-safe `normalize_for_match`, `satisfies_groups`, and rubric/anchor models. |
| `ablation.py` | Layer 1 retrieval ablation (`bm25_only`, `dense_only`, `hybrid_rrf`, `served_search`) sharing one query embedding per case. |
| `agents.py` | Layer 2 grounded Q&A (`raw_files` vs `arkive_mcp`) via `OpenAIGateway.converse` and claim-level anchored judge (`supported` / `contradicted` / `unverifiable`). |
| `drafting.py` | Layer 3 holdout builder with leak guards, five drafting arms (`zero_shot`, `chunk_search`, `raw_files`, `arkive_mcp`, `exemplars_in_context`), per-document `OverlapSources` lexical measures, `prepare_draft` selection diagnostics, and blind comparison judge. |
| `run.py` | CLI entrypoint (`python -m evals.run`), full preflight validation, spend limit (`--max-cost`), and atomic owner-only (`0600`) output publication. |
| `example_corpora/` | Four strictly fictional tracked corpora (`coastal-survey-projects`, `creator-campaigns`, `acoustic-consultancy-files`, `structural-monitoring-consultancy`). |
| `example_packs/` | Matching `arkive-eval-pack/3` files for the four public corpora (`example_pack.eval.json` is the smaller offline test fixture). |
| `opencode_check/` | Live Docker OpenCode `2.3.0` runner (`python -m evals.opencode_check`) executing `scenarios.json` with only Arkive MCP tools enabled. |

## Hard rules for `evals/`

1. **Never place a developer `AGENTS.md` inside `evals/example_corpora/`.** `discover_corpus` treats every `AGENTS.md` inside a corpus folder as a curator note data file, which changes corpus fingerprints and breaks packs.
2. **Zero PII or client data in tracked fixtures.** All files under `example_corpora/` and `example_packs/` must remain strictly fictional. Always write evaluation outputs (`--out`) and private packs outside tracked directories (e.g., `/tmp/...` or gitignored `.tmp/...`); `run.py` refuses overlapping `--out`/`--cache` paths with pack, snapshot, or corpus.
3. **Preflight before spend or disk writes.** Every new validation on packs, snapshots, filters, anchor evidence sizes, holdout leak guards, LibreOffice availability, or output/cache paths must execute before creating `--out`, opening `--cache`, or making any OpenAI call.
4. **Evaluation fairness invariants (do not silently weaken):**
   - `EXEMPLAR_DRAFTING_RULES` in `drafting.py` is frozen independently of `src/arkive_rag/server.py` so tuning Arkive server instructions never shifts the `exemplars_in_context` control arm.
   - `DraftComparison` (the blind drafting judge) must always receive `holdout.instructions` alongside `run_notes` and the reference.
   - `OverlapSources` tests unit holding per individual holdout document (preventing cross-document 4-gram stitching) and attributes contamination over the union of qualifying holders (never charge a copied unit to a minimum-stray single holder, which hides borrowed instrument/site terms).
   - Never tune rubrics, metrics, or example corpora to rescue a failing product change.
5. **Paid run discipline:** Paid runs require an explicit `--max-cost`. Layer 3 refuses an `--out` directory that already contains `holdouts/` so prior blind drafts cannot be overwritten or mixed.
6. **OpenCode check (`opencode_check/`):**
   - Scenarios (`arkive-opencode-scenarios/1`) never hardcode induced facet names, deliverable type labels, or standing passages; they assert runtime discovery and packet reuse instead.
   - Every scenario fails automatically if OpenCode truncates any Arkive MCP tool result or calls a non-Arkive tool.
   - Keep `OPENAI_API_KEY` redacted in every written event stream and report.

## Focused verification

Offline harness tests (zero API calls, fake Docker for `opencode_check`):

```sh
uv run --locked pytest -o cache_dir="$SCRATCH/pytest-cache" --basetemp="$SCRATCH/pytest-tmp" \
  tests/test_evals.py tests/test_opencode_check.py
```
