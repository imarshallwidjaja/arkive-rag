# Building and serving a snapshot

Arkive works in two steps. A one-time build reads every document in a folder and saves what it learns into one knowledge file, called a snapshot. A local server then lets AI assistants read that snapshot. Paid AI models run only during the build, and the server reads the file on your own machine.

![Arkive system overview: build a snapshot from a folder, then serve it to AI assistants](assets/system-overview.svg)

[Open the interactive diagram](assets/system-overview.html)

## Building a snapshot

An engineer runs `arkive-rag build` on the prepared folder. The build does the work below and saves the result as one file.

**Reads every page accurately.** Arkive converts Word and PowerPoint files to PDF and renders each page as an image. A vision model receives the image and the page's own text, and returns the headings, paragraphs and tables in reading order. It also writes out values that appear only in charts, maps and diagrams. Arkive checks that the numbers and filenames in the original text survive, and re-reads a page with more effort when they do not. A table that runs over a page break becomes one table again.

**Reads what the file itself holds.** Some content never appears on a printed page, so Arkive also reads it from the file:

- A Word file is read with its tracked changes accepted, so a deleted figure cannot run into its replacement. Its reviewer comments and tracked changes are kept as separate notes.
- A PowerPoint file adds each slide's speaker notes, comments and the numbers behind its native charts. Hidden slides are skipped.
- Excel workbooks and HTML pages have no pages to render, so Arkive converts them straight into headings and tables. A workbook keeps each sheet's tables, its cell comments, and any formula that was saved without a computed value. An HTML page keeps its text and tables, and Arkive never fetches its images.

**Records stated facts and tables.** For each document, the model lists the facts the document states about itself, such as its reference, client, site, status and instruments. Arkive keeps a fact only when it finds the model's quote word for word in the document, and links the fact to that section. Every table is also stored row by row, so an assistant can later find matching rows across the whole collection.

**Works out the filters and document types.** Arkive reads the folder names, the `AGENTS.md` notes and a short summary of every document. From these it proposes the filters that suit this collection, such as project, client or campaign. It also labels each document with a kind of deliverable, such as proposal, technical memorandum or installation report. Nobody defines these in advance.

**Separates drafts from final versions.** When documents share most of their text or title, Arkive gathers them into a small group. A model then decides which of them are versions of the same deliverable and which one, if any, clearly replaces the others. An addendum, a second project phase or the next issue of a monthly bulletin stays separate. A file's saved date alone never makes it the latest version. Afterwards, every tool that lists or searches documents shows only current versions unless the assistant asks for all of them, and says how many superseded drafts it left out.

**Extracts reviewer lessons.** For each review or QA document, Arkive extracts lessons that apply to future work, each with a quote. It keeps a lesson only when it finds the quote in the source and a second model confirms that the quote supports the lesson.

**Flags standard wording.** Arkive finds paragraphs, list items and table headers that appear in more than one document, and records every document that holds them. An assistant can then tell which text is the team's standard wording.

**Writes a glossary and a guide.** Arkive collects the collection's own terms and acronyms with cited definitions. When the evidence does not support a definition, Arkive leaves it empty instead of guessing. It also writes a short guide to the whole collection, with citations.

**Saves one portable file.** Everything goes into a single SQLite database file. With a build cache, rebuilding an unchanged folder costs nothing. Adding a file costs only the reading of that file plus a fresh pass over the collection as a whole.

[Appendix A: Build reference](appendices/build-and-parsing.md) gives the exact steps, limits and caching rules, with a worked example.

## Serving a snapshot to AI assistants

`arkive-rag serve` opens one or more snapshots and offers them to AI assistants over MCP. MCP (Model Context Protocol) is a standard way for an AI assistant to call tools that another program provides. Any MCP-capable assistant can connect, such as a coding agent or a chat client.

**Read-only and local.** The server opens each snapshot read-only, checks its integrity on startup, and answers from the file. It makes no generative AI calls. When an OpenAI key is set, the server sends two kinds of text to OpenAI to turn them into embeddings, the numeric vectors used for meaning-based matching: each search query, and the first 8,000 characters of the project facts given for a draft. Without a key, every tool except meaning-based search still works, and drafting ranks past work by keywords alone.

**Eleven tools.** The assistant gets tools to:

- discover the collection: its topics, guide, glossary and document list;
- find text, by meaning-based search or exact text lookup;
- read matching rows from the stored tables;
- read a whole section or view a rendered page;
- fetch a drafting packet.

The listing and search tools can narrow results by the filters Arkive found, such as project or client, and by kind of deliverable. They return current versions only unless the assistant asks for all versions. Each document listing and outline also shows the facts the document states about itself.

**Results that always arrive whole.** Every tool result stays below the size at which MCP clients such as OpenCode cut off long results, so an assistant never silently loses the end of a result. Long outlines and sections are split into pages that the assistant requests in turn.

**The drafting packet.** To draft, the assistant sends the new project's facts to the drafting tool. The tool returns relevant past examples with key sections, reviewer lessons and the team's collection notes. It flags sections with recurring wording and includes a few samples to help the assistant judge what may carry over. [Two everyday workflows](overview.md#two-everyday-workflows) describes what the packet holds.

[Appendix B: Server tools and drafting reference](appendices/mcp-tools-and-drafting.md) lists every tool with its options and size limits, and explains how the packet is assembled.
