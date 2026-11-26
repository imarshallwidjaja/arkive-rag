# Data boundaries, security, and cost

Arkive sends document content to OpenAI only while it builds a snapshot. After the build, the snapshot is a local file that the server reads without any generative AI calls. Every paid command accepts a spending limit.

![A private vault: documents and the snapshot stay with the team, and only build requests and optional query embeddings go out to OpenAI](assets/privacy-vault-concept.png)

## Data sent to OpenAI, by situation

![Data and trust boundaries between the team folder, the build, OpenAI, the snapshot, and AI assistants](assets/data-and-trust-boundaries.svg)

[Open the interactive diagram](assets/data-and-trust-boundaries.html)

| Situation | Sent to OpenAI | Stays local |
| --- | --- | --- |
| **First build** | Page images and page text, to transcribe them. Document sections, including the text of Excel and HTML files, to summarise them. Folder names, `AGENTS.md` notes and document summaries, to organise the collection. Text passages, to compute embeddings (numeric vectors used for meaning-based search). | Source files, rendered pages, and the finished snapshot. |
| **Rebuild of an unchanged folder with a build cache** | Nothing. The cache answers every request. | Everything. |
| **Serving with an API key** | The text of each search query, and the first 8,000 characters of the project facts given for a draft, to compute embeddings. No document content is sent. | The snapshot and every tool result. |
| **Serving without an API key** | Nothing. Every tool except meaning-based search works. | Everything. |

The AI assistant that calls Arkive is a separate program. It sends the tool results it receives to whatever model it uses, under that assistant's own data terms.

## Safeguards

- **Read-only serving.** The server opens each snapshot read-only and checks its structure and citations on startup. No tool can change a snapshot.
- **Owner-only files.** Arkive creates a new snapshot or build cache so that only the user who built it can read it. Replacing an existing snapshot keeps that file's permissions.
- **No symbolic links.** The build stops if the folder contains a symbolic link, before any model call.
- **Size limits on Office files.** Word, PowerPoint and Excel files are compressed archives. Before opening one, Arkive unpacks it in small steps and refuses it if it has too many entries, an encrypted or damaged entry, or would expand past a fixed size. A small crafted file therefore cannot expand into an unbounded amount of data. Arkive also refuses a workbook with more populated cells than a fixed limit, so a large workbook is never silently cut short. Appendix A gives the [archive limits](appendices/build-and-parsing.md#word-powerpoint-and-pdf-files) and the [workbook cell limit](appendices/build-and-parsing.md#excel-workbooks-and-html-pages).
- **No fetches from HTML.** Arkive reads an HTML page from its saved bytes only, and fetches none of its images from the network or from disk.
- **Outputs kept out of the source folder.** The build refuses to write a snapshot or cache inside the folder it reads.
- **Local-only HTTP by default.** When served over HTTP (`--transport streamable-http`), the server listens only on this machine (`127.0.0.1`). The HTTP mode has no built-in authentication, so serving to other hosts requires the explicit `--allow-nonlocal` option.
- **Private data outside Git.** Keep client documents and snapshots in a private store outside this repository. `.gitignore` includes rules for snapshots (`*.arkive.db` and `*.arkive.db.*`) and local evaluation packs (`*.eval.local.json`). It does not automatically exclude client documents.

## Cost and spending limits

Arkive uses three fixed OpenAI models:

- `gpt-6.1-sol` reads pages, describes each document, decides which documents are versions of one deliverable, and writes summaries. Version decisions use this larger model because a wrong latest version would hide the issued document from later answers.
- `gpt-6-luna` answers yes/no checks, such as whether a quote supports a reviewer lesson.
- `text-embedding-3-large` computes the vectors for meaning-based search.

[Appendix B](appendices/mcp-tools-and-drafting.md#models-prices-and-cost-growth) lists the price per million tokens of each model.

**Spending limit.** `arkive-rag build` and the evaluation runner accept `--max-cost <USD>`. Once the estimated spend reaches the limit, no new paid call starts. Calls already in progress can finish and take the total past the limit. The estimate leaves out OpenAI's surcharge for very long prompts. Each build prints its calls, tokens and estimated cost.

**Measured build costs.** These come from metered builds of the fictional collections tracked in this repository:

| Build | Paid calls | Cost |
| --- | --- | --- |
| First build of the 126-file `structural-monitoring-consultancy` archive (110 rendered pages, including a 52-page Word report) | 490 | $4.67 |
| Rebuild of the same archive with no changes, using the build cache | 0 | $0.00 |
| First build of a 9-file collection (`creator-campaigns`, `coastal-survey-projects` or `acoustic-consultancy-files`) | 31 to 42 | $0.18 to $0.37 |

Reading pages and describing each document make up most of the cost of the large archive, so build cost grows mainly with the number of rendered pages and the amount of document text. Adding one file to a cached build reads only that file and repeats the collection-wide pass. On the previous version, adding a 127th Markdown file this way cost $0.28.

**Effect on the assistant's own model bill.** The server costs nothing to run beyond the optional query embeddings. Its effect on the assistant's bill comes from keeping results small. On the 126-file archive:

- A question about a current value used about 5,000 input tokens with Arkive, about half of what the raw-file tools used.
- A question about four projects' interrogators and alarm levels used 40,600 input tokens with Arkive against 90,200 with raw-file tools.
- Across all seven questions in that test set, Arkive and the raw-file tools used about the same total input tokens, so the savings depend on the question.
- Long drafting costs more. To match the house wording of a 52-page report, the assistant read many sections and used more input tokens than with raw-file tools.

[Test results and known limits](results-and-limits.md) gives the details.
