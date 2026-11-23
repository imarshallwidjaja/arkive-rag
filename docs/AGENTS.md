# `docs/` instructions

Refines root `AGENTS.md` for human-facing documentation and visual assets under `docs/`.

## Documentation map

[`docs/README.md`](README.md) is the hub and owns the map of guides and appendices. Update it when a page is added, renamed, or removed. Guides use plain language with no module names, schema tables, formulas, or constants, and engineering detail goes in `docs/appendices/`. [`docs/architecture.md`](architecture.md) is the authoritative engineering reference, schema and invariant spec, and verification command list. `evals/schema.py`, `evals/opencode_check/__init__.py`, and root `AGENTS.md` link to its sections, so preserve its existing section headings and anchors.

## Authoring and maintenance rules

1. **Strict public-corpora-only PII boundary:** Never include client names, real project names, private documents, private facts, or private benchmark metrics in `docs/` or `README.md`. Every example, file trace, ID, and evaluation table must come exclusively from the four tracked fictional collections under `evals/example_corpora/` (`coastal-survey-projects`, `creator-campaigns`, `acoustic-consultancy-files`, `structural-monitoring-consultancy`).
2. **Verify against executable source:** Whenever code changes `SCHEMA_VERSION`, hosted model IDs (`src/arkive_rag/llm.py`), MCP tool signatures or byte/line ceilings (`src/arkive_rag/server.py`, `src/arkive_rag/snapshot.py`), or induction bounds (`src/arkive_rag/induction.py`), update the corresponding appendix and any affected guide in the same change. If prose conflicts with code, code wins.
3. **Keep `README.md` as a landing page:** Root `README.md` holds only the product summary, prerequisites, quickstart commands, core safety/privacy rules, and links into `docs/`. Do not copy full pipeline specs or evaluation tables back into `README.md`.
4. **Two tiers:** Guides explain behaviour, workflows, data boundaries, costs, and evidence in plain words and link to the matching appendix for mechanics. New constants, symbols, tables, and full benchmark data go in an appendix or `architecture.md`.
5. **Human-legible prose:** Open each page and section with the concrete answer or mechanism, explain unfamiliar terms (MCP, snapshot, embedding) at first use in a guide, keep one topic per paragraph, and omit filler or rejected-alternative narration unless explaining a live operational trade-off.
6. **Visual assets (`docs/assets/`):**
   - Technical diagrams (`system-overview`, `team-workflow`, `data-and-trust-boundaries`, `benchmark-highlights`, `build-pipeline`, `mcp-serving-and-drafting`, `evaluation-harness`) ship as a self-contained `.html` source and a companion `.svg`. They use the Archival Folio palette (`#F4F1EA` paper, `#1D232A` ink, `#47525E` muted, `#1B5E63` link, `#B03F1B` focal accent). Edit the `.html` first, validate it with `diagram-design`'s `self_check.py`, and re-export the `.svg` via `export_svg.py` so Markdown embeds stay in sync.
   - Editorial concept illustrations (`arkive-concept-hero.png`, `mcp-workflows-concept.png`, `privacy-vault-concept.png`) carry zero embedded text and stay consistent with `arkive-concept-hero.png` via reference-guided edits.
