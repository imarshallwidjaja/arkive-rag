# Arkive RAG documentation

Arkive RAG compiles a curated folder of `.pdf`, `.docx`, `.pptx`, and `.md` reference documents into one portable, read-only SQLite snapshot (`*.arkive.db`), and serves one or more snapshots to AI agents over MCP.

![Arkive RAG system overview](assets/system-overview.svg)

[Open interactive HTML diagram](assets/system-overview.html)

## Reading guide

Choose the document that matches what you need to do or understand:

| Document | Audience | Covers |
| --- | --- | --- |
| [System walkthrough](walkthrough.md) | First-time readers | Product goal, curation workflow, end-to-end trace of one PDF from `build` to `serve`, and SQLite table overview. |
| [Parsing, induction, and build pipeline](parsing-and-build.md) | Engineers and curators | How Arkive converts Office/PDF files, transcribes visual pages, stitches tables, caches work, induces facets/types/version families/lessons, and stores schema `6.0` snapshots, with real worked examples from public fictional collections. |
| [MCP serving, `prepare_draft`, and cost scaling](mcp-serving.md) | Agent builders and operators | All ten MCP tools, hybrid RRF search with version-family caps, the 45,000-byte `prepare_draft` packet anatomy, provenance rules, hosted OpenAI models, and cost/scaling profile. |
| [Evaluation harness and benchmark results](evaluation-and-benchmarks.md) | Evaluators and reviewers | The three-layer evaluation harness, live OpenCode 2.3.0 Docker acceptance check, the four public fictional collections, verified benchmark results, and known limits. |
| [Architecture and verification reference](architecture.md) | Contributors | Full technical specification of modules, schema invariants, validation checks, error handling, and offline repository verification gates. |

## Visual assets

Technical diagrams under `docs/assets/` ship as both standalone `.svg` figures (embedded inline in Markdown) and self-contained `.html` pages:

- [System overview](assets/system-overview.html) ([SVG](assets/system-overview.svg))
- [Build and parsing pipeline](assets/build-pipeline.html) ([SVG](assets/build-pipeline.svg))
- [MCP serving and `prepare_draft` packet](assets/mcp-serving-and-drafting.html) ([SVG](assets/mcp-serving-and-drafting.svg))
- [Evaluation harness](assets/evaluation-harness.html) ([SVG](assets/evaluation-harness.svg))

Conceptual editorial illustrations (`docs/assets/arkive-concept-hero.png` and `docs/assets/mcp-workflows-concept.png`) illustrate the compilation and dual-workflow model.

## Privacy boundary

Every example, document ID, excerpt, and evaluation table in this documentation suite comes exclusively from the four fictional collections tracked in `evals/example_corpora/`. No client documents, personal data, or private benchmark results appear in this repository.
