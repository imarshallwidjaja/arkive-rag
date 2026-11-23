# Evidence, benchmarks, and known limits

On the largest test collection, a 126-file engineering archive, assistants using Arkive covered the expected facts in more questions, and scored higher for accuracy, with 42% fewer input tokens than the same model reading the raw files. Arkive also stopped an assistant from copying another project's data policy into a new report. On small collections and very long single reports, simpler approaches did as well or better.

## Test method

All tests used four fictional collections tracked in this repository under `evals/example_corpora/`. They range from 9 files to a 126-file archive that mixes Markdown, Word, and PDF files, drafts, duplicates, and a 52-page report.

Each test gave the same AI model (`gpt-6.1-sol`) the same task with different tools:

- **Arkive:** the ten Arkive tools over a snapshot of the collection. A snapshot is the single SQLite file that `arkive-rag build` makes from a document folder. The assistant calls the tools over MCP (Model Context Protocol), a standard way for an AI assistant to use tools that another program provides.
- **Raw files:** tools to list, search, read, and view the original files directly.
- **Pasted examples:** hand-picked complete reference documents placed straight into the prompt. This is the control the others are compared against.

Answers were scored by phrase checks and by a separate blinded model judge that did not know which approach produced each answer. Drafting tests held one real document out of the collection, asked each approach to write it again from project notes, and compared the drafts with the held-out original over five rounds.

A final check ran 10 scenarios in a real MCP client, OpenCode 2.3.0, inside fresh Docker containers.

![Benchmark highlights: Q&A accuracy and tokens, and policy-leak avoidance](assets/benchmark-highlights.svg)

[Open the interactive diagram](assets/benchmark-highlights.html)

## Main results

**1. Better answers with fewer tokens on a large archive.** On the 126-file archive, Arkive covered the expected facts in 8 of 8 questions, and raw files in 6 of 8. On the two questions where an outdated draft disagrees with the issued document, Arkive covered the expected facts in 2 of 2 and raw files in 0 of 2. Arkive used 77,215 input tokens against 134,043 for raw files, 42% fewer.

**2. Avoided copying an unrelated project's policy.** For a night-noise report in the acoustic consultancy collection, pasting complete past reports led the model to copy another project's six-year data-retention policy into the new report. The judge scored that approach 4 out of 10 for discipline. Arkive's drafting packet showed the dates and how many projects share each passage, and the assistant avoided the borrowed policy. Arkive scored 10 out of 10 for discipline, won all 5 blind-judge rounds, and used 57% fewer input tokens than raw files.

**3. Works in a real MCP client.** All 10 live OpenCode scenarios passed. No Arkive result was truncated, and the assistant called no tool outside Arkive.

## When Arkive helps, and current limits

Arkive helps most when a collection is large, mixes formats, holds drafts beside final versions, or carries one-off clauses that must not spread to new work.

These limits were measured:

- **Small collections.** Under about 10 short Markdown files, pasting two or three files into the prompt uses fewer tokens than the overhead of MCP tools.
- **Very long reports.** For a new installation report modelled on the 52-page Askern Viaduct report, pasting the full report won all 5 blind-judge rounds against Arkive's compact drafting packet. The packet holds only part of a long report, so the assistant must read the omitted sections with the `read_section` tool to close the gap.
- **Overlapping filter names.** On the 126-file archive, Arkive organised the documents in several batches and produced two overlapping filters, `sensing_technology` and `monitoring_technology`. Both work as filters.

## Full data

[Appendix C: Evaluation harness and full benchmark tables](appendices/evaluation-benchmarks.md) has every collection and holdout result, the retrieval tests, and the commands to run the evaluation yourself.
