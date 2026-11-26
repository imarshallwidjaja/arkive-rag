# Arkive RAG

![A folder of team documents compiled into one portable knowledge file that AI assistants query](docs/assets/arkive-concept-hero.png)

Arkive RAG turns a folder of a team's reference documents into one portable, read-only knowledge file, called a snapshot. AI assistants use that file to answer questions with citations and to draft new deliverables that follow the team's past work. It reads PDF (`.pdf`), Word (`.docx`), PowerPoint (`.pptx`), Excel (`.xlsx`), HTML (`.html`, `.htm`) and Markdown (`.md`) files.

A team uses one knowledge file for two jobs:

- **Answering questions with citations.** The assistant finds exact numbers, filenames, table rows and chart values, and cites the document and section they came from, with page numbers when the document has pages. It answers from the current version of each document unless the question is about earlier drafts.
- **Drafting from past work.** The assistant receives relevant past examples and reviewer lessons. The drafting packet flags sections with recurring wording and includes a few samples. This helps the assistant judge which wording may be a team standard.

## At a glance

- **No data preparation.** Any folder layout works, and nobody has to tag, rename or reformat files in the supported formats. Convert legacy `.doc`, `.ppt` and `.xls` files first, because the build stops when it finds any other file type. Hidden files and folders are skipped. You can add plain-English notes in files named `AGENTS.md` to tell Arkive how the collection fits together.
- **AI calls happen while building.** Building the knowledge file sends document content to OpenAI. Serving it reads the file on your machine and makes no generative AI calls.
- **Serving works mostly offline.** Without an OpenAI key, every tool except meaning-based search works, and drafting ranks past work by keywords alone. With a key, the server sends each search query, and the start of the project notes given for a draft, to OpenAI to compute embeddings.
- **Build spend has a cap.** The `--max-cost` option stops new paid calls once the estimated spend reaches your limit.
- **Private data stays out of Git.** Keep client documents and snapshots in a private store outside this repository. `.gitignore` includes rules for snapshots and local evaluation packs. It does not automatically exclude client documents.

## Quickstart

You need Python 3.14, [uv](https://docs.astral.sh/uv/), LibreOffice (the `soffice` command, used to render Word and PowerPoint files) and an `OPENAI_API_KEY`. Excel and HTML files need no extra software. If the key is not set in your environment, Arkive reads it from a `.env` file in the working directory.

The commands below build `evals/example_corpora/coastal-survey-projects`, a fictional collection of nine files from three survey projects. Write build outputs outside the folder you build.

```sh
uv sync --locked

# Build the knowledge file (stops new paid calls at about USD 2)
uv run --locked arkive-rag build evals/example_corpora/coastal-survey-projects \
  -o /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  --cache /tmp/arkive-examples/coastal-survey-projects.cache.db \
  --max-cost 2

# Show the filters, document types, version groups and reviewer lessons Arkive found
uv run --locked arkive-rag inspect /tmp/arkive-examples/coastal-survey-projects.arkive.db

# Exact text lookup (offline)
uv run --locked arkive-rag grep /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  "oskary_pit_volume_2026.csv"

# Meaning-based search (needs OPENAI_API_KEY)
uv run --locked arkive-rag search /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  "minimum depth in the dredged channel"

# Serve the knowledge file to an MCP-capable AI assistant (stdio by default)
uv run --locked arkive-rag serve /tmp/arkive-examples/coastal-survey-projects.arkive.db
```

`search` and `grep` also accept `--filter FACET=VALUE`, `--type` for a document type, and `--versions all` to include superseded drafts. Arkive works out the filter names and document types for each collection, so read them from the `inspect` output first.

## Documentation

Guides:

- [Arkive overview and team workflows](docs/overview.md)
- [Building and serving a snapshot](docs/how-it-works.md)
- [Data boundaries, security, and cost](docs/security-privacy-and-cost.md)
- [Test results and known limits](docs/results-and-limits.md)

Reference:

- [Appendix A: Build reference](docs/appendices/build-and-parsing.md)
- [Appendix B: Server tools and drafting reference](docs/appendices/mcp-tools-and-drafting.md)
- [Appendix C: Evaluation method and full results](docs/appendices/evaluation-benchmarks.md)
- [Architecture and verification reference](docs/architecture.md)

The [documentation hub](docs/README.md) lists every diagram. Contributors start at [`AGENTS.md`](AGENTS.md).
