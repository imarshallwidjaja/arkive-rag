# Arkive RAG documentation

Arkive RAG turns a curated folder of `.pdf`, `.docx`, `.pptx`, and `.md` reference documents into one portable, read-only knowledge file (`*.arkive.db`). AI assistants read that file over MCP to answer questions with citations and to draft new work from past examples.

![Arkive RAG system overview](assets/system-overview.svg)

[Open the interactive diagram](assets/system-overview.html)

## Guides (start here)

Short plain-language reads for product, technical, and team leads.

| Guide | Covers |
| --- | --- |
| [What Arkive does and how a team uses it](overview.md) | The problem with raw folders, how a team curates a collection, the two everyday workflows, and a worked example. |
| [How folders become searchable knowledge](how-it-works.md) | What the build does to each document and how the server gives assistants bounded, read-only access. |
| [Data boundaries, security, and cost](security-privacy-and-cost.md) | What is sent to OpenAI and when, file and network safeguards, spending limits, and measured costs. |
| [Evidence, benchmarks, and known limits](results-and-limits.md) | How Arkive was tested, the main results, and where it does not help. |

## Technical appendices and reference

Engineering detail, full benchmark tables, and the contributor specification.

| Document | Covers |
| --- | --- |
| [Appendix A: Parsing, induction, and build pipeline](appendices/build-and-parsing.md) | The three build passes, limits, caching, snapshot contents, build model calls, and a worked example. |
| [Appendix B: MCP tools, drafting packet, and cost scaling](appendices/mcp-tools-and-drafting.md) | All eleven tools and their bounds, hybrid search, `prepare_draft` packing, model prices, and scaling. |
| [Appendix C: Evaluation harness and full benchmark tables](appendices/evaluation-benchmarks.md) | The evaluation layers, the four collections, every result table, and the commands to run it. |
| [Architecture and verification reference](architecture.md) | Modules, schema invariants, validation, error handling, and repository verification gates. |

## Visual assets

Diagrams in `assets/` ship as an `.svg` for embedding and a self-contained interactive `.html` page:

- [System overview](assets/system-overview.html) ([SVG](assets/system-overview.svg))
- [Team workflow](assets/team-workflow.html) ([SVG](assets/team-workflow.svg))
- [Data and trust boundaries](assets/data-and-trust-boundaries.html) ([SVG](assets/data-and-trust-boundaries.svg))
- [Benchmark highlights](assets/benchmark-highlights.html) ([SVG](assets/benchmark-highlights.svg))
- [Build and parsing pipeline](assets/build-pipeline.html) ([SVG](assets/build-pipeline.svg))
- [MCP serving and drafting packet](assets/mcp-serving-and-drafting.html) ([SVG](assets/mcp-serving-and-drafting.svg))
- [Evaluation harness](assets/evaluation-harness.html) ([SVG](assets/evaluation-harness.svg))

Three illustrations show the concepts: [the folder-to-snapshot model](assets/arkive-concept-hero.png), [the two workflows](assets/mcp-workflows-concept.png), and [the data boundary](assets/privacy-vault-concept.png).

## Privacy boundary

Every example, document ID, excerpt, and result table in these docs comes from the four fictional collections in `evals/example_corpora/`. No client documents, personal data, or private benchmark results appear in this repository.
