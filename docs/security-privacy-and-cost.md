# Data boundaries, security, and cost

Arkive sends document content to OpenAI only while it builds a snapshot. After the build, the snapshot is a local file that the server reads without any generative AI calls. Every paid command takes a spending limit.

![A private vault: documents and the snapshot stay with the team, and only build requests and optional query embeddings go out to OpenAI](assets/privacy-vault-concept.png)

## What leaves your environment and when

![Data and trust boundaries between the team folder, the build, OpenAI, the snapshot, and AI assistants](assets/data-and-trust-boundaries.svg)

[Open the interactive diagram](assets/data-and-trust-boundaries.html)

| Situation | Sent to OpenAI | Stays local |
| --- | --- | --- |
| **First build** | Page images and page text for transcription, document sections for summaries, folder names, curator notes, and document summaries for organising the collection, and chunk text for embeddings (numeric vectors used for meaning-based search). | Source files, rendered pages, and the finished snapshot. |
| **Unchanged rebuild with `--cache`** | Nothing. The cache answers every request, so the build makes zero provider calls. | Everything. |
| **Serving with an API key** | The text of each `search` query, and the project inputs given to `prepare_draft`, to turn them into embeddings. No document content is sent. | The snapshot and every tool result. |
| **Serving without an API key** | Nothing. Nine of the ten tools work, and `search` reports that it needs a key. | Everything. |

The AI assistant that calls Arkive is a separate program. It sends the tool results it receives to whatever model it uses, under that assistant's own data terms.

## Operational safeguards

- **Read-only serving.** The server opens each snapshot in read-only mode and checks its structure and citations on startup. No tool can change a snapshot.
- **Owner-only files.** A new snapshot and a new build cache are created with file mode `0600`, so only the owning user can read them. Replacing an existing snapshot keeps that file's mode.
- **Symlinks rejected.** The build refuses symbolic links and files that resolve outside the corpus folder, before any model call.
- **Outputs kept out of the corpus.** The build refuses to write a snapshot or cache inside the folder it reads.
- **Local-only HTTP by default.** Over HTTP (`--transport streamable-http`), the server binds to `127.0.0.1`. The HTTP transport has no built-in authentication, so serving on another host requires the explicit flag `--allow-nonlocal`.
- **Private data outside Git.** Keep client documents and snapshots in a private store. The repository's `.gitignore` excludes `*.arkive.db` snapshots and `*.eval.local.json` private evaluation packs.

## Cost profile and spend controls

Arkive uses three fixed OpenAI models: `gpt-6.1-sol` for reading pages and writing summaries, `gpt-6-luna` for yes/no checks, and `text-embedding-3-large` for search vectors. [Appendix B](appendices/mcp-tools-and-drafting.md#hosted-models-costs-and-scaling) lists the price per million tokens of each.

**Spending limit.** `arkive-rag build` and `python -m evals.run` take `--max-cost <USD>`. Once the estimated spend reaches the limit, no new paid call starts. Calls already in flight can finish past the limit, and the estimate does not include the long-context surcharge. Each build prints its calls, tokens, and estimated cost.

**Measured build costs** on the tracked fictional 126-file `structural-monitoring-consultancy` archive (110 rendered pages, including a 52-page Word report):

| Build | Cost |
| --- | --- |
| Cold build of all 126 files | about $1.22 |
| Rebuild with no changes, using `--cache` | $0.0000 (0 provider calls) |
| Adding a 127th file, using `--cache` | $0.28 (reads the new file and reruns the collection-level pass) |

Build cost grows mainly with the number of rendered pages.

**Runtime savings.** The server itself costs nothing to run beyond the optional query embeddings. Its effect on the assistant's own model bill comes from bounded results. On the 126-file archive's Q&A test, assistants using Arkive read 42% fewer input tokens than assistants given the raw files (77,215 against 134,043). In the drafting test for a new installation report modelled on the archive's 52-page report, Arkive used 179,051 input tokens, 38% fewer than raw files (287,143) and 67% fewer than pasted full examples (540,990). [Evidence, benchmarks, and known limits](results-and-limits.md) gives the details.
