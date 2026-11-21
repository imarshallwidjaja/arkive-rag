# `docs/` instructions

Refines root `AGENTS.md` for human-facing documentation and visual assets under `docs/`.

## Documentation map

- [`docs/README.md`](README.md) — Documentation index and reading order.
- [`docs/walkthrough.md`](walkthrough.md) — Product goal and concise end-to-end trace.
- [`docs/parsing-and-build.md`](parsing-and-build.md) — Document parsing, visual transcription, render/LLM caching, bounded induction, and worked public-corpus examples.
- [`docs/mcp-serving.md`](mcp-serving.md) — MCP server architecture, all ten tools, `prepare_draft` 45 KB packet mechanics, hosted models, and cost scaling.
- [`docs/evaluation-and-benchmarks.md`](evaluation-and-benchmarks.md) — Three-layer evaluation harness, live OpenCode container check, and verified public-corpus benchmark results.
- [`docs/architecture.md`](architecture.md) — Authoritative engineering reference, schema/invariant spec, and verification commands (referenced by `evals/schema.py`, `evals/opencode_check/__init__.py`, and root `AGENTS.md`; preserve existing section anchors).

## Authoring and maintenance rules

1. **Strict public-corpora-only PII boundary:** Never include client names, real project names, private documents, private facts, or private benchmark metrics in `docs/` or `README.md`. Every example, file trace, ID, and evaluation table must come exclusively from the four tracked fictional collections under `evals/example_corpora/` (`coastal-survey-projects`, `creator-campaigns`, `acoustic-consultancy-files`, `structural-monitoring-consultancy`).
2. **Verify against executable source:** Whenever code changes `SCHEMA_VERSION`, hosted model IDs (`src/arkive_rag/llm.py`), MCP tool signatures or byte/line ceilings (`src/arkive_rag/server.py`, `src/arkive_rag/snapshot.py`), or induction bounds (`src/arkive_rag/induction.py`), update the corresponding docs in the same change. If prose conflicts with code, code wins.
3. **Keep `README.md` as a landing page:** Root `README.md` holds only the product summary, prerequisites, quickstart commands, core safety/privacy rules, and links into `docs/`. Do not copy full pipeline specs or evaluation tables back into `README.md`.
4. **Human-legible prose:** Open each page and section with the concrete answer or mechanism, name real symbols/commands/numbers, keep one topic per paragraph, and omit filler or rejected-alternative narration unless explaining a live operational trade-off.
5. **Visual assets (`docs/assets/`):**
   - Technical diagrams use the Archival Folio palette (`#F4F1EA` paper, `#1D232A` ink, `#47525E` muted, `#1B5E63` link, `#B03F1B` focal accent). Edit the self-contained `.html` source first, validate it with `diagram-design`'s `self_check.py`, and re-export the companion standalone `.svg` via `export_svg.py` so Markdown embeds stay in sync.
   - Editorial concept illustrations (`*.png`) carry zero embedded text and stay consistent with `docs/assets/arkive-concept-hero.png` via reference-guided edits.
