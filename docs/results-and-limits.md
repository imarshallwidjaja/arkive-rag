# Evidence, benchmarks, and known limits

On a 126-file engineering archive, the current version of Arkive picked the right latest version of every multi-version document in three separate builds, answered every current-value and version-history question correctly, and answered both questions that span many reports correctly in every run. For a long installation report, its drafts now pass every rubric check and beat the raw files and plain chunk search in all 5 blind-judge rounds. Pasting the complete past reports into the prompt still produced the best long-report draft, at a similar token cost.

## Test method

All tests used fictional documents. The main tests used the four collections tracked in this repository under `evals/example_corpora/`, which range from 9 files to a 126-file archive of Markdown, Word, and PDF files with drafts, duplicates, and a 52-page report. Three small synthetic stress collections tested hard cases: version traps, a 60-document table rollup, and deliberately messy Office, HTML, and PDF files. These three collections, and the extra question and drafting sets used for the current version, are fictional test material kept outside the repository, so only the four tracked collections and their question sets can be rerun from it.

Each test gave the same AI model (`gpt-6.1-sol`) the same task with different tools:

- **Arkive:** the eleven Arkive tools over a snapshot of the collection. A snapshot is the single SQLite file that `arkive-rag build` makes from a document folder. The assistant calls the tools over MCP (Model Context Protocol), a standard way for an AI assistant to use tools that another program provides.
- **Raw files:** tools to list, search, read, and view the original files directly.
- **Chunk search:** a plain search over text fragments, used in drafting tests.
- **Pasted examples:** hand-picked complete reference documents placed straight into the prompt. This is the control the others are compared against.

Answers were scored by phrase checks and by a separate blinded model judge that did not know which approach produced each answer. Most questions ran 3 times; the chart questions on the three small collections ran once. Drafting tests held one real document out of the collection, asked each approach to write it again from project notes, and compared the drafts with the held-out original over 5 rounds. A final check ran 13 scenarios in a real MCP client, OpenCode 2.3.0, inside fresh Docker containers.

The results below are for schema `7.0`, the current snapshot format. [Appendix C](appendices/evaluation-benchmarks.md) keeps the earlier schema `6.0` results as the baseline and gives every number.

## Main results

**1. Drafts are kept apart from issued versions.** In three independent builds of the 126-file archive, Arkive grouped all 5 multi-version documents exactly and named the right latest version in all 4 cases where one exists, with no wrong groupings. In the version-trap collection it also kept an addendum, two project phases, and two monthly bulletins apart from each other, and chose the issued final over a draft whose file was saved later.

**2. Current values and version history both answered correctly.** On questions about current values (such as alarm levels and a fee) and about how those values changed between drafts, Arkive passed 12 of 12 runs, never quoted a superseded value as current, and scored 5.00 out of 5 for accuracy with no contradicted claims. Current-value answers used about half the input tokens of the raw-file approach.

**3. Questions across many reports.** Two questions each needed facts from many reports. The assistant answered them by listing the documents of one deliverable type, then reading their stored attributes and sections, and in some runs by querying stored tables.

- **All 18 issued installation reports.** Counting and summarising them used a median of 97,600 input tokens over 3 runs, down 73.6% from 369,800 in one run on the previous version, and slightly below the raw-file median of 102,700. The averages point the other way: 177,000 for Arkive against 98,400 for raw files, because one Arkive run used 358,800.
- **Four projects' interrogators and alarm levels.** Listing them used 40,600 input tokens on average against 90,200 for raw files, with higher accuracy (5.00 against 4.00).

Across all 7 questions in this set, total input tokens were close: 793,000 for Arkive and 774,000 for raw files. On a 60-document synthetic collection whose values sit only in tables, Arkive used its table query across the collection and answered all 6 table rollup runs correctly.

**4. Long-report drafting now applies the house method.** For a new installation report modelled on the archive's 52-page report, Arkive's drafts passed 60 of 60 rubric checks, up from 55 of 60. The check for the house temperature-compensation step went from 0 of 5 rounds to 5 of 5. The share of standard house wording reused rose from 0.656 to 0.800, almost no text was copied from other projects' documents (a contamination score of 0.0011), and Arkive ranked second behind the pasted full examples in every round, ahead of raw files and chunk search 5 to 0. In a variant where the project notes gave a batch calibration value that replaces the house standard, a reading of all 5 Arkive drafts confirmed that each used the project value.

**5. Standard wording without borrowed site values.** For a night-noise report in the acoustic consultancy collection, house wording reuse rose from 0.343 to 0.759, Arkive won 4 of 5 blind-judge rounds, and no draft borrowed another site's values.

**6. Charts, new formats, and a real client.** Arkive found and answered all 4 chart-only values across the four collections. On five deliberately messy files (a workbook, an HTML page, a Word file with tracked changes and comments, a deck with notes and a native chart, and a PDF with compound units), every stored-content check passed, and Arkive answered 14 of 15 question runs correctly against 9 of 15 for raw files. All 13 live OpenCode scenarios passed, with no truncated result and no call to a tool outside Arkive.

## Earlier baseline

The previous snapshot format, schema `6.0`, is the baseline for the results above. The diagram below shows schema `6.0` results only: input tokens for Q&A and long-report drafting on the 126-file archive, and the discipline score of a drafting task where pasted examples borrowed another project's retention policy.

![Schema 6.0 baseline benchmark highlights: Q&A accuracy and tokens, and policy-leak avoidance](assets/benchmark-highlights.svg)

[Open the interactive baseline diagram](assets/benchmark-highlights.html)

## When Arkive helps, and current limits

Arkive helps most when a collection is large, mixes formats, holds drafts beside final versions, keeps values in tables across many documents, or carries one-off clauses that must not spread to new work.

These limits were measured:

- **Small collections.** Under about 10 short Markdown files, pasting two or three files into the prompt uses fewer tokens than the overhead of MCP tools. On the 60-document table collection, raw files also answered every question, with fewer input tokens, because the matching rows were few enough to find with one text search.
- **Very long reports.** Pasting the full 52-page report still won every blind-judge round for the long installation report. To reach its rubric score, the Arkive assistant read many extra sections and used about 495,000 input tokens over 5 rounds, close to the 541,000 of the pasted examples and more than the 298,000 of raw files.
- **Filters can differ between builds.** Arkive organises a large collection in batches. In two of three builds of the 126-file archive, the `project` filter listed only the jobs from one batch, covering 23 of 124 documents; the third build covered 110. An assistant that asks for a missing project gets the list of valid values and can retry.
- **Broad table queries.** A table query that mixes a column name with a value word, such as "Zone C Fail", also matches every table whose header contains "Zone". In two runs this returned 49 tables and raised input tokens about tenfold.
- **Reproduced method text.** In the night-noise report, Arkive reproduced the house survey-method text that the project notes did not mention. The judge marked this down for discipline (7 out of 10, against 10 for the previous version) while confirming that no other site's details leaked.
- **Chart slides without a drawn chart.** In one of three runs on the messy deck, the assistant viewed a slide image that showed no drawn chart and did not read the stored chart table in the same section.

## Full data

[Appendix C: Evaluation harness and full benchmark tables](appendices/evaluation-benchmarks.md) has every collection and holdout result, the before-and-after tables, the retrieval tests, and the commands to run the evaluation yourself on the four tracked collections. The synthetic stress collections and extra question sets are not in the repository, so their results cannot be rerun from it.
