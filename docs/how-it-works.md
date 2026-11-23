# How folders become searchable knowledge

Arkive works in two steps. A one-time **build** reads every document in a folder and saves what it learns into one snapshot file. A local **server** then lets AI assistants read that snapshot. Paid AI models run only during the build. Serving reads the file on your own machine.

![Arkive system overview: build a snapshot from a folder, then serve it to AI assistants](assets/system-overview.svg)

[Open the interactive diagram](assets/system-overview.html)

## Step 1: Building a snapshot

An engineer runs `arkive-rag build` on the curated folder. The build does the following work and saves the result as one `*.arkive.db` file.

**Reads every page accurately.** Arkive converts Word and PowerPoint files to PDF, renders each page as an image, and sends the image and the page's own text to a vision model. The model returns the headings, paragraphs, and tables in reading order, and writes out values that appear only in charts, maps, and diagrams. Arkive checks that the numbers and filenames in the original text survive, and re-reads a page with more effort when they do not. A table that runs over a page break becomes one table again.

**Finds the filters and document types.** Arkive reads the folder names, the curator notes, and a short summary of every document. From these it proposes the filters that suit this collection, such as project, client, or campaign, and labels each document with a deliverable type, such as proposal, technical memorandum, or installation report. Nobody defines these in advance.

**Separates drafts from final versions.** When two documents share most of their text or title, Arkive asks a model whether they are versions of the same deliverable. It then uses dates to mark the newest as current and the rest as superseded. Search shows the current version first.

**Extracts reviewer lessons.** For each review or QA document, Arkive extracts lessons that apply to future work, each with a verbatim quote. It keeps a lesson only when it can find the quote in the source and a second model confirms the quote supports the lesson.

**Flags standard wording.** Arkive finds paragraphs, list items, and table headers that appear in more than one document, and records every document that holds them. This tells an assistant later which text is the team's standard wording.

**Writes a glossary and a guide.** Arkive collects the collection's own terms and acronyms with cited definitions. A definition that its evidence does not support is left empty instead of guessed. It also writes a short cited guide to the whole collection.

**Saves one portable file.** Everything goes into a single SQLite database file. A team can copy it, keep it beside other snapshots, and rebuild it when the folder changes. With a build cache, an unchanged rebuild costs nothing and an added file costs only that file plus a fresh pass over the collection structure.

> **For engineers:** [Appendix A: Parsing, induction, and build pipeline](appendices/build-and-parsing.md) covers the three build passes, limits, caching, and a worked example.

## Step 2: Serving to AI assistants

`arkive-rag serve` opens one or more snapshots and offers them over MCP. MCP (Model Context Protocol) is a standard way for an AI assistant to call tools that another program provides. Any MCP-capable assistant can connect, such as a coding agent or a chat client.

**Read-only and local.** The server opens each snapshot read-only, checks its integrity on startup, and answers from the file. It makes no generative AI calls. When an OpenAI key is present, the server sends two kinds of text to OpenAI to turn them into embeddings (numeric vectors used for meaning-based matching): each `search` query, and the first 8,000 characters of the project inputs given to `prepare_draft`. Without a key, nine of the ten tools still work. `prepare_draft` then ranks past work by keywords alone, and `search` reports that it needs a key.

**Bounded answers.** Every tool result stays below the size at which MCP clients such as OpenCode cut off tool results, so an assistant never silently loses the end of a result. Long outlines and sections are split into pages that the assistant requests in turn. Appendix B gives the exact byte and line limits.

**The drafting packet.** When asked to draft, the assistant sends the new project's facts to the `prepare_draft` tool, which returns the closest past deliverables of the requested type with their most relevant sections, reviewer lessons, and curator notes. Each section shows how many separate projects share its wording, so the assistant can reuse standard company text and treat a one-project clause with care. [Two everyday workflows](overview.md#two-everyday-workflows) describes what the packet holds.

> **For engineers:** [Appendix B: MCP tools, drafting packet, and cost scaling](appendices/mcp-tools-and-drafting.md) lists every tool, its limits, and how the packet is assembled.
