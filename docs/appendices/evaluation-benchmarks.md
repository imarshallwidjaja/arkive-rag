# Appendix C: Evaluation method and full results

This appendix explains how Arkive was evaluated and gives every result, including the unfavourable ones, with the commands to rerun the tests that use tracked data. [Test results and known limits](../results-and-limits.md) is the plain-language summary. Maintainers will find the authoritative specification of the evaluation harness under [Evaluation](../architecture.md#evaluation), [Example corpora](../architecture.md#example-corpora) and [OpenCode check](../architecture.md#opencode-check).

The evaluation asks whether an assistant does better with Arkive than with direct file tools over the same documents. It also compares Arkive with a control that pastes complete, hand-picked example documents into the prompt. Every document involved is fictional.

![The evaluation harness: inputs are checked before any paid call, then four kinds of test write private reports](../assets/evaluation-harness.svg)

[Open the interactive diagram](../assets/evaluation-harness.html)

## Test collections

Four collections are tracked in `evals/example_corpora/`, each with a matching question and drafting set (an eval pack) in `evals/example_packs/`. Each stresses a different kind of collection.

| Collection | Files and formats | What it tests |
| --- | --- | --- |
| `coastal-survey-projects` | 9 files (Markdown, PDF, `AGENTS.md`) in 3 project folders | Proposals against technical memoranda, reviewer lessons, domain acronyms such as `MBES`, and wording that bleeds between proposals and memos. |
| `creator-campaigns` | 9 files (Markdown, PowerPoint, `AGENTS.md`) for 2 creators and 4 campaigns | Campaign briefs, review notes, a PowerPoint launch deck, and policy clauses borrowed between creators. |
| `acoustic-consultancy-files` | 9 files (Word, PDF) in one flat folder, no `AGENTS.md` | Dates only from file metadata, an older letter layout beside a newer numbered report layout, shared caveats that differ by one instrument name, values only in charts, and a data-retention policy that belongs to one project. |
| `structural-monitoring-consultancy` | 126 files (110 Markdown including 2 `AGENTS.md`, 10 Word, 6 PDF; 110 rendered pages) | Scale: 17 document types as written (28 found by the build), a 52-page Word report, series of superseded drafts, a byte-identical duplicate, unlabelled QA registers, look-alike methods (vibrating-wire against fibre Bragg grating), and a chart value (`-142`) drawn only as vector graphics in a PDF. |

Three synthetic stress collections tested hard cases for the current version: version traps (`version-traps`, 8 files), a 60-document table rollup (`table-scale-60`: 58 Markdown, 1 Excel and 1 HTML file), and five messy files in five formats (`formats-stress`). The current version was also tested with extra question and drafting sets on the tracked collections: current-value and version-history questions, questions across many reports, one chart question per collection, and a drafting test with a changed house value. The synthetic collections and the extra sets are fictional and kept outside Git, so their results cannot be rerun from the repository.

### Versions compared

- **Current version:** snapshots in the current format (schema `7.0`), served with eleven tools.
- **Previous version:** snapshots in the previous format (schema `6.0`), served with ten tools. Its results are the baseline.

Both used the same models: `gpt-6.1-sol`, `gpt-6-luna` and `text-embedding-3-large`.

## Kinds of test

`python -m evals.run` runs the first three kinds, selected with `--layer 1`, `2`, `3` or `all`. Before it creates the output folder, opens a cache or calls OpenAI, it checks its inputs: the eval pack's format, that the snapshot was built from the pack's collection, the filters, the size of each question's evidence, the guards that keep a held-out document out of a drafting test, that LibreOffice is available, and that output paths do not overlap inputs. Outputs are written in one step and readable only by their owner.

1. **Search tests (layer 1).** These compare four rankings on the same queries, with one shared query embedding per case: keyword only, meaning only, both combined, and the search that assistants actually get. The last hides superseded drafts and returns at most 3 passages per version group. Cases cover exact identifiers, paraphrases, values visible only in charts and maps, repaired font ligatures, and telling documents apart with filters. *Completion@k* is the share of cases whose top k passages together hold every required piece of evidence. *MRR@5* is the average of 1 divided by the rank at which a case completes, counting cases that do not complete within 5 results as 0.
2. **Question-answering tests (layer 2).** The same `gpt-6.1-sol` assistant answers each question twice: once with raw file tools (list files, search raw text, read pages, view page images) and once with the eleven Arkive tools. Questions cover lookups, facts spread over several documents, counting across documents, conflicts between drafts and issued versions, questions the collection cannot answer, and vocabulary. Each answer gets a phrase check and a blind judge. The judge, also `gpt-6.1-sol`, reads the reference facts and the stored text of the cited pages, scores accuracy, citations and completeness from 1 to 5, and labels every material claim as supported, contradicted or unverifiable. Tokens, tool calls and time are recorded.
3. **Drafting tests (layer 3).** The harness builds a separate snapshot without the target document. Its guards exclude files whose path marks them as held out and files byte-identical to the target. The guards read neither folder meaning nor content, so the pack author must exclude edited copies and documents that restate the held-out facts. Each round asks up to five approaches to write the target again from project notes. A rubric checks each draft for required content, and two measures compare its wording with the collection by overlap in four-word sequences:
   - *House-text recall* starts with target passages that sufficiently overlap text in at least one included document. It takes all distinct four-word sequences from those passages, including wording that the other documents may not contain. It excludes sequences in the project notes or task instructions, then measures the share of the remaining sequences that appear in the draft's prose.
   - *Copied-text contamination* measures how much draft wording matches included documents without support from the target report, project notes or drafting instructions. It counts only matches in sufficiently similar prose passages and divides that count by all distinct four-word sequences in the draft's prose.

   Both scores range from 0 to 1 when defined. They measure wording overlap and cannot establish reuse, copying or agreement in meaning. [Evaluation](../architecture.md#evaluation) lists their limits, including how reworded house text can affect the contamination score.

   A blind judge, `gpt-6.1-sol`, then ranks the drafts of each round against the target and scores each from 1 to 10 for structure, rigour, limitations, discipline and effort saved.
4. **Live OpenCode check.** `python -m evals.opencode_check` runs a real MCP client, OpenCode 2.3.0 with `openai/gpt-6.1-sol`, in fresh Docker containers, with the snapshots mounted read-only and only the Arkive tools enabled. It runs the ten scenarios in `scenarios.json`, or the thirteen in `scenarios-uplift.json`, which add a document-type listing, a version-history question and a table query on `structural-monitoring-consultancy`. A scenario fails if OpenCode calls a tool outside Arkive or cuts off an Arkive result.

The tables below use these names for the approaches compared:

| Name in tables | What the assistant had | ID in `results.json` |
| --- | --- | --- |
| Arkive | The eleven Arkive tools (ten on the previous version), centred on the drafting packet when drafting. | `arkive_mcp` |
| Raw files | File tools over the same documents. | `raw_files` |
| Chunk search | A plain search over text passages (drafting tests only). | `chunk_search` |
| Pasted examples | The drafting test's named example documents pasted into the prompt under fixed drafting rules. This is the control, and it runs only when the test names examples. | `exemplars_in_context` |
| No tools | The project notes only (drafting tests only). | `zero_shot` |

## Current-version results

Repeats for these runs:

- The question tests on `structural-monitoring-consultancy` and on the synthetic collections ran 3 times per approach.
- The chart questions on the three 9-file collections ran once per approach.
- Every drafting test ran 5 rounds per approach.

The completed metered runs cost $18.06: $16.36 for the tests on the tracked collections and $1.69 for the synthetic collections. One more drafting run cost about $2.05 before a host restart stopped it part-way. Its held-out snapshot was cached, and the clean rerun reused it at no cost.

### Metered build cost

| Collection | Documents | Chunks | Paid calls | Input tokens | Output tokens | Time | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `structural-monitoring-consultancy` (first build) | 126 | 1,086 | 490 | 1,567,475 | 285,214 | 2,540.7 s | $4.6673 |
| `structural-monitoring-consultancy` (unchanged rebuild with `--cache`) | 126 | 1,086 | 0 | 0 | 0 | — | $0.0000 |
| `acoustic-consultancy-files` (first build) | 9 | 71 | 42 | 87,387 | 23,053 | 207.6 s | $0.3667 |
| `coastal-survey-projects` (first build) | 9 | 54 | 42 | 49,542 | 14,443 | 148.1 s | $0.1986 |
| `creator-campaigns` (first build) | 9 | 47 | 31 | 38,520 | 11,833 | 141.7 s | $0.1754 |

The 126-file first build by step, as named in the build's cost line:

| Step | Cost | Calls |
| --- | ---: | ---: |
| Page transcription (`transcribe_page`) | $2.4820 | 110 |
| Document records (`observe`) | $1.3761 | 127 |
| Glossary (`write_glossary`) | $0.3670 | 10 |
| Filters and types (`organise_corpus`) | $0.1198 | 2 |
| Version groups (`partition_versions`) | $0.0830 | 13 |
| Collection guide (`write_guide`) | $0.0597 | 1 |
| Reviewer lessons (`extract_feedback_lessons`) | $0.0574 | 15 |
| Chunk vectors (`embed`) | $0.0340 | 10 |
| Text-based filter values (`bind_facets`) | $0.0273 | 124 |
| Filter merging (`canonicalise_facets`) | $0.0267 | 1 |
| Definition checks (`check_definitions`) | $0.0155 | 58 |
| Type merging (`canonicalise_types`) | $0.0146 | 1 |
| Merging a long document's record (`consolidate_card`) | $0.0034 | 1 |
| Lesson checks (`check_feedback_lessons`) | $0.0008 | 17 |

An earlier estimate of about $1.22 for this collection predates the current build steps and their per-step metering.

### Version groups and filter coverage

Three independent runs of the collection-organising step on `structural-monitoring-consultancy` were compared with the 5 multi-version documents verified in the source files. The first run was part of a full first build. The other two reran organisation over the document readings that the first build had cached. The 5 documents are three draft series (of a method statement, an installation report and a scoping note), a proposal `v0.9` and `v1.0` pair, and a byte-identical calibration sheet in two job folders, which has no defined latest version.

| Organising run | Exact groups | Latest version correct (of 4 with a defined latest) | Missed / false groups | Filters found (documents covered, of 124) |
| --- | --- | --- | --- | --- |
| Full first build | 5/5 | 4/4 | 0 / 0 | `client` (91), `job_cohort` (110), `sensing_technology` (122), `issue_status` (124), `project` (23) |
| Second run, cached readings | 5/5 | 4/4 | 0 / 0 | `client` (91), `project` (110), `archive_section` (110), `sensing_technology` (119), `issue_status` (124) |
| Third run, cached readings | 5/5 | 4/4 | 0 / 0 | `client` (91), `job_status` (110), `monitoring_technology` (124), `issue_status` (124), `project` (23) |

No run produced a filter that duplicates the document type, and every run typed the installation reports identically. Coverage of `project` varied. In two runs, the organising step worked in two batches and only the second batch's 6 archive jobs became `project` values (23 of 124 documents). The other run listed all 29 jobs (110 of 124). In one repeat of the question tests, the assistant filtered on a project outside those values, received the error that lists the valid values, and recovered on its next call. The evaluation report flags pairs of filters that cover almost the same documents (a document-set overlap, or Jaccard index, of at least 0.85). Here the flagged pairs ranged from 0.87 to 1.00. These pairs were independent filters that each cover nearly every document, such as `issue_status` and `sensing_technology`.

The synthetic version-trap collection (8 files) produced 1 of 1 exact group with the correct latest version and no false supersessions:

- **Draft with a later file date.** A `v0.9` Word draft with no printed date and a file metadata date of 2026-06-15 was grouped with the `v1.0` PDF final printed 2026-05-01. The final is the latest version and the draft is superseded.
- **Addendum.** A base report and its Addendum 1 stayed separate documents.
- **Project phases.** Phase 1 and phase 2 studies stayed separate.
- **Monthly series.** Two consecutive monthly bulletins stayed separate.

On its questions, Arkive reported the current crack width in 3 of 3 runs and never gave the superseded value. The default of current versions only kept both the base report and the addendum (3 of 3).

### Search tests on the structural collection

`evals/example_packs/structural-monitoring-consultancy.eval.json`, 10 cases, top 5 results:

| Ranking | Previous version (C@1 / C@3 / C@5 / MRR@5) | Current version (C@1 / C@3 / C@5 / MRR@5) |
| --- | --- | --- |
| Keyword only (`bm25_only`) | 20% / 40% / 50% / 0.325 | 20% / 50% / 50% / 0.333 |
| Meaning only (`dense_only`) | 20% / 40% / 40% / 0.283 | 20% / 40% / 40% / 0.283 |
| Combined (`hybrid_rrf`) | 10% / 30% / 60% / 0.248 | 20% / 50% / 50% / 0.317 |
| Served search (`served_search`) | 10% / 30% / 50% / 0.245 | 20% / 50% / 50% / 0.350 |

C@k is Completion@k. The combined ranking lost 10 points of Completion@5 while gaining at 1 and 3 results.

### Current values, version history and questions across many reports

`structural-monitoring-consultancy`, 7 questions × 3 repeats = 21 runs per approach:

| Approach | Phrase pass | Accuracy | Citations | Completeness | Contradicted | Unverifiable | Input tokens | Tool calls | Time |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Arkive | 21/21 | 5.00 | 5.00 | 4.95 | 0 | 3 | 793,249 | 122 | 404 s |
| Raw files | 21/21 | 4.81 | 4.76 | 4.86 | 0 | 7 | 774,296 | 154 | 441 s |

Per question (mean input tokens per run):

| Question | Kind | Approach | Phrase | Accuracy / Citations / Completeness | Contradicted / Unverifiable | Mean input tokens | Tool calls |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| Current Greyscar alarm levels | Current value | Arkive | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 4,710 | 3 |
| | | Raw files | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 11,253 | 6 |
| Current Mill Lane fee | Current value | Arkive | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 5,511 | 4 |
| | | Raw files | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 9,673 | 6 |
| How the Greyscar alarm levels changed | Version history | Arkive | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 17,674 | 20 |
| | | Raw files | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 16,681 | 15 |
| How the Mill Lane fee changed | Version history | Arkive | 3/3 | 5.00 / 5.00 / 4.67 | 0 / 0 | 8,272 | 9 |
| | | Raw files | 3/3 | 4.67 / 4.67 / 4.00 | 0 / 1 | 10,646 | 12 |
| All issued installation reports | Across many reports | Arkive | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 2 | 177,029 (runs of 358.8k, 97.6k and 74.7k) | 55 |
| | | Raw files | 3/3 | 5.00 / 4.67 / 5.00 | 0 / 1 | 98,412 | 62 |
| Projects using the Prisma FI-4 interrogator, with alarm levels | Across many reports | Arkive | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 1 | 40,623 | 25 |
| | | Raw files | 3/3 | 4.00 / 4.00 / 5.00 | 0 / 5 | 90,196 | 44 |
| Largest peak change at Ellery | Lookup | Arkive | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 10,598 | 6 |
| | | Raw files | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 21,238 | 9 |

On the four current-value and version-history questions, Arkive had 12 of 12 phrase passes, never gave a superseded value as current, made no contradicted claims, and scored 5.00 for accuracy.

On the question about all issued installation reports, the median Arkive run used 97.6k input tokens and the mean 177.0k. A single run of the same question on the previous version (schema `6.0`), in an earlier experiment, used 369,752. The current median is 73.6% lower than that one run and the current mean is 52.1% lower. One current run still used 358.8k, so the reduction does not hold on every run. Against raw files on the same question, the Arkive median was 97.6k against 102.7k, and the mean 177.0k against 98.4k.

Arkive answered both questions across many reports by listing documents filtered by type and issue status, then reading their facts and sections with `search`, `grep` and `read_section`. The two cheaper runs on the installation-report question made no `query_tables` call. The 358.8k run made 2 `query_tables` calls, and one was refused because it asked for `limit=100`. On the interrogator question, 1 of 3 runs called `query_tables`.

### Table rollup at 60 documents

The synthetic 60-document collection holds 203 anchor proof-load rows across 58 Markdown reports whose prose never mentions the tables, a master asset schedule in a workbook, and a regional summary HTML page. The build stored tables from all 60 documents (60 tables, 227 rows). Two questions ran 3 times per approach: the failing `Zone C` rows (6 rows across 6 documents, totalling 2,520 kN), and a join of one region's assets across the workbook and the HTML page.

| Approach | Phrase pass | Accuracy / Citations / Completeness | Contradicted | Turns (mean) | Tool calls | Input tokens (6 runs) |
| --- | --- | --- | --- | --- | --- | ---: |
| Arkive | 6/6 | 5.00 / 5.00 / 5.00 | 0 | 2–6 (3.33) | 14 (7 `query_tables`, 5 `search`, 2 `grep`) | 166,220 |
| Raw files | 6/6 | 5.00 / 5.00 / 5.00 | 0 | 3–4 (3.50) | 33 | 33,279 |

Two tool behaviours explain the token gap:

1. **Query words match separately.** `query_tables` matches any of the query's words. One `Zone C Fail` repeat called `query_tables(query="Fail")` and finished in 3 turns with 7,238 input tokens. The two repeats that called `query_tables(query="Zone C Fail")` matched every anchor table through the header word `Zone`, returned 49 tables (44,521 characters) plus a further page, and used 68,922 and 76,615 input tokens.
2. **Limit errors.** Twice the assistant asked for `limit=100` and spent a turn on the `limit must be between 1 and 50` error.

Raw files passed because the collection has only 15 `Fail` rows, under the raw search tool's limit of 30 matches, so one text search found every failing row.

### Charts across all four collections

| Collection and chart value | Served search (C@1 / C@3 / MRR@5) | Arkive: phrase / accuracy / citations / completeness / contradicted | Raw files: phrase / accuracy / citations / completeness / contradicted |
| --- | --- | --- | --- |
| Structural (`EW-05 = -142`, 3 repeats) | 0% / 100% / 0.500 | 3/3 / 5.00 / 5.00 / 5.00 / 0 | 3/3 / 5.00 / 5.00 / 5.00 / 0 |
| Acoustic (`05-06 = 14`, 1 repeat) | 0% / 100% / 0.500 | 1/1 / 5.00 / 5.00 / 5.00 / 0 | 1/1 / 5.00 / 5.00 / 5.00 / 0 |
| Coastal (`Bench 3 = 41,700`, 1 repeat) | 100% / 100% / 1.000 | 1/1 / 5.00 / 5.00 / 5.00 / 0 | 1/1 / 5.00 / 5.00 / 5.00 / 0 |
| Creator (`Week 3 = 48,200`, 1 repeat) | 100% / 100% / 1.000 | 1/1 / 5.00 / 5.00 / 5.00 / 0 | 1/1 / 5.00 / 5.00 / 5.00 / 0 |

Every chart value was transcribed exactly in a `[Visual — Page N: ...]` block.

### New formats and messy Office files

The synthetic collection has one file per format:

- a Word file with a tracked fee change and a reviewer comment;
- a deck with a hidden slide, speaker notes, a comment thread and a native bar chart;
- a two-sheet workbook with a merged header, formulas saved without values, a legacy note, a separate data block and a threaded comment;
- an HTML page with text before the first heading, a table header spanning columns, and an image pointing at this machine that must not be fetched;
- a 3-page PDF with 4-line running headers and compound units.

Every check of the stored content passed on all 5 files:

- The deleted fee appears only in an annotation and never runs into its replacement.
- The hidden slide is absent, and the notes, comments and chart values are stored.
- The formulas saved without values are kept.
- The HTML page kept its text before the first heading and its spanning-header table, and the stored text leaves out the image and its alt text.
- Running headers were removed while compound units survived.

These checks read stored content only. The offline test `test_an_html_page_reads_in_order_without_fetching_anything` in `tests/test_native_formats.py` checks that converting a page with an image on this machine and an image on local disk opens no network connection, requests no URL and does not open the local file.

| Question | Arkive | Raw files |
| --- | --- | --- |
| Workbook formula and comments | 3/3, 5.00 accuracy | 3/3, 5.00 |
| HTML text before the first heading, and table | 3/3, 5.00 | 3/3, 5.00 |
| Word tracked change and comment | 3/3, 5.00 | 0/3, 2.00 |
| Deck chart, notes and comments, without the hidden slide | 2/3, 4.33 | 0/3, 1.00 |
| PDF compound units | 3/3, 5.00 | 3/3, 5.00 |
| **Overall** | **14/15, 4.87 accuracy, 5.00 citations, 1 contradicted claim** | **9/15, 3.60 accuracy, 3.80 citations, 14 contradicted claims** |

The raw-file tools see the same text for workbooks and HTML pages, but cannot read Word comments, PowerPoint notes and comments, or chart data. Arkive's one miss came from the test file itself: it attaches the chart data without a drawn chart, so the rendered slide shows no bars. In 2 of 3 runs the assistant followed `view_page` with `read_section` and found the stored chart table. In the third it trusted the blank image.

### Drafting tests

**52-page installation report** (`tallow-cross-fbg-installation-report`, modelled on the Askern Viaduct report), Arkive:

| Measure | Previous version | Current version | Other approaches (current version) |
| --- | --- | --- | --- |
| Rubric checks passed | 55/60 (temperature-compensation check 0/5) | 60/60 (temperature-compensation check 5/5) | Raw files 55/60 (0/5), chunk search 56/60 (1/5), pasted examples 60/60 |
| House-text recall, median [min, max] | 0.656 [0.633, 0.679] | 0.800 [0.753, 0.819] | Chunk search 0.428, raw files 0.335, pasted examples 0.977 |
| Copied-text contamination, median | 0.0000 (0/4,336) | 0.0011 (4/4,521) | Pasted examples 0.0135 (79/3,543) |
| Blind-judge rank | 2nd ×1, 3rd ×2, 4th ×2 | 2nd in 5/5 (ahead of raw files and chunk search in all 5) | Pasted examples 1st in 5/5 |
| Judge medians (structure / rigour / limitations / discipline / effort saved) | 7 / 8 / 8 / 9 / 7 | 7 / 8 / 8 / 8 / 8 | Raw files 8 / 7 / 8 / 9 / 7, chunk search 7 / 7 / 8 / 9 / 6 |
| Tool calls / input tokens (5 rounds) / cost | 26 / 179k / $0.3982 | 103 / 494,829 (350k cached) / $0.5898 | Input tokens: raw files 298,184, chunk search 134,324, pasted examples 540,990 |

The drafting packet included 1 of the 3 named example documents, and listed all 3 among its candidates. The assistant made 103 tool calls, against 26 on the previous version, to read further sections, and its input tokens rose with them.

**Changed house value** (`tallow-cross-counterfactual-sfr`). The project notes say a batch calibration certificate's `9.4 pm/°C` replaces the house standard of `9.7 pm/°C`. Arkive ranked 1st in round 5, ahead of pasted examples, and 2nd in rounds 1 to 4. It was ahead of raw files and chunk search in all 5 rounds, with house-text recall 0.827 [0.822, 0.879], discipline 9, effort saved 8 and contamination 0.0040. The automated check for the replaced value scored 3/5, and all 11 other checks scored 5/5. A reading of all 5 drafts shows that every one used `9.4 pm/°C` and none used `9.7 pm/°C`. The 2 failures are rounds whose sentence about the replacement ("replaces the standard SFR temperature sensitivity of 9.7 pm/°C") contains the forbidden text `sensitivity of 9.7 pm`. The check matches that text literally, so it cannot tell naming a replaced value from using it.

**Night-noise report** (`corvane-depot-night-noise-report`), Arkive:

| Measure | Previous version | Current version |
| --- | --- | --- |
| House-text recall, median [min, max] | 0.343 [0.343, 0.350] | 0.759 [0.759, 1.000] (12 of 15 house passages reproduced: 3 passages × 5 rounds) |
| Copied-text contamination, median | 0.0069 (25/3,581) | 0.0071 (25/3,512) |
| No values borrowed from another site | 5/5 | 5/5 (rubric total 24/25; mitigation kept 4/5) |
| Blind-judge 1st place | 5/5 (discipline 10) | 4/5 (round 2 ranked 4th; structure 9, effort saved 9, discipline 7 [6, 8]) |
| Input tokens (5 rounds) | 47,880 | 53,670 (raw files 119,433) |

Discipline fell because the drafts reproduced the house survey-method text (compliance procedure, microphone height and clearance, weather screening) that the project notes did not state. The judge noted that this text matches the target report and found no wording from other sites.

### Live OpenCode check

The thirteen scenarios of `scenarios-uplift.json` ran against snapshots of `coastal-survey-projects`, `creator-campaigns` and `structural-monitoring-consultancy`. The command is under [Running an evaluation](#running-an-evaluation). They ran on OpenCode 2.3.0 with `openai/gpt-6.1-sol`, in the Alpine Linux image `arkive-opencode-check:9f88b15ebbec` (Python 3.14), built from hash-pinned prebuilt packages with the Excel and HTML readers installed.

- **Result:** 13 of 13 scenarios passed, with no calls to tools outside Arkive and no cut-off results, in 306 s at a cost of at least $0.2644.
- **New scenarios:** listing documents of one type (`type-filtered-list`, 2 calls; returned all 14 `Installation report` documents), finding version history (`version-history-discovery`, 5 calls; asked for all versions to compare `v0.8` and `v0.9` with `v1.0`), and a table rollup (`table-rollup`, 6 `query_tables` calls; returned the expected calibration months and the `1.21 pm/µε` array sensitivity).

OpenCode's cost figures are lower bounds, because it can leave out the last step's usage and does not count the server's query embeddings.

## Previous-version results

These baseline runs used snapshots in the previous format (schema `6.0`), served with ten tools, on the tracked collections. Drafting tests ran 5 rounds per approach.

### Questions at 126-file scale

`structural-monitoring-consultancy`, 8 questions covering lookups, facts across several documents, counting across documents, conflicts, unanswerable questions and vocabulary:

| Approach | Phrase coverage | Accuracy (1-5) | Citations (1-5) | Completeness (1-5) | Contradicted claims | Unverifiable claims | Input tokens | Tool calls | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw files | 6/8 | 4.75 | 4.62 | 4.62 | 0 | 6 | 134,043 | 36 | 100 s |
| Arkive | **8/8** | **4.88** | **5.00** | **4.75** | **0** | **1** | **77,215** (-42%) | **21** | **85 s** |

Raw files also needed 29.2 s of text extraction before the first question. On the two conflict questions, where superseded drafts disagree with issued documents, raw files reached 0/2 phrase coverage and Arkive 2/2.

Search tests on the same snapshot (10 cases, top 5): combined ranking reached 60% Completion@5, served search 50%, keyword only 50%, and meaning only 40%. All four reached 100% Completion@3 on the case whose answer is a chart value drawn only as vector graphics in a PDF.

### Drafting across all four collections

Medians across 5 completed rounds. The judge scores structure, rigour, limitations, discipline and effort saved from 1 to 10. "Examples included" gives how many of the drafting test's named examples the packet included, then how many it listed among its candidates. "fb" gives how many of the test's named feedback documents supplied at least one lesson in the packet. Token counts are totals over 5 rounds.

| Collection and drafting test | Approach | Examples included (packet / candidates) | House recall | Contamination | Judge medians (structure / rigour / limitations / discipline / effort) | 1st-place wins | Input tokens (5 rounds) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Structural (126 docs)** `brackenford-footbridge-proposal` | Arkive | 2/5 · 5/5 | **0.984** | **0.0000** | **9 / 9 / 9 / 10 / 9** | **5/5** | 221,802 |
| | Raw files | — | 0.810 | 0.0000 | 8 / 9 / 9 / 8 / 8 | 0/5 | 212,379 |
| | Pasted examples | control | 0.971 | 0.0000 | 9 / 8 / 8 / 10 / 8 | 0/5 | 30,175 |
| **Structural (126 docs)** `tallow-cross-fbg-installation-report` (52-page example) | Arkive | 2/3 · 3/3 (fb 1/1) | **0.656** | **0.0000** | 7 / 8 / 8 / 9 / 7 | 0/5 | **179,051** |
| | Raw files | — | 0.377 | 0.0000 | 8 / 8 / 8 / 9 / 7 | 0/5 | 287,143 |
| | Pasted examples | control | 0.977 | 0.0226 | **9 / 9 / 9 / 8 / 9** | **5/5** | 540,990 |
| **Acoustic (flat folder)** `corvane-depot-night-noise-report` | Arkive | 3/3 · 3/3 | 0.343 | **0.0069** | **9 / 8 / 9 / 10 / 9** | **5/5** | **47,880** |
| | Raw files | — | 0.292 | 0.0000 | 6 / 8 / 9 / 10 / 7 | 0/5 | 112,328 |
| | Pasted examples | control | 1.000 | 0.0757 | 8 / 6 / 5 / 4 / 6 | 0/5 | 17,340 |
| **Acoustic (flat folder)** `pellow-mill-facade-report` | Arkive | 3/3 · 3/3 | 0.346 | **0.0000** | **8 / 9 / 7 / 10 / 8** | **3/5** | 48,405 |
| | Raw files | — | 0.309 | 0.0000 | 8 / 8 / 7 / 10 / 8 | 1/5 | 28,760 |
| | Pasted examples | control | 0.346 | 0.0000 | 8 / 8 / 7 / 10 / 8 | 1/5 | 16,980 |
| **Coastal (9 docs)** `fennick-canal-proposal` | Arkive | 2/2 · 2/2 | **1.000** | **0.0000** | **9 / 10 / 10 / 10 / 9** | **2/5** | 38,815 |
| | Raw files | — | 1.000 | 0.0000 | 9 / 10 / 10 / 10 / 8 | 0/5 | 26,000 |
| | Pasted examples | control | 1.000 | 0.0000 | 9 / 10 / 9 / 10 / 9 | 2/5 | 8,305 |
| **Coastal (9 docs)** `fennick-canal-technical-memorandum` | Arkive | 2/2 · 2/2 (fb 2/2) | n/a (no house passages) | **0.0000** | **9 / 9 / 10 / 10 / 9** | 1/5 | 32,890 |
| | Raw files | — | n/a | 0.0000 | 9 / 10 / 10 / 10 / 9 | **4/5** | 19,640 |
| | Pasted examples | control | n/a | 0.0000 | 9 / 8 / 8 / 8 / 8 | 0/5 | 10,905 |
| **Creator (9 docs)** `kettlecraft-winter-campaign-brief` | Arkive | 3/3 · 3/3 (fb 2/2) | **1.000** | 0.0526 | **10 / 9 / 9 / 9 / 9** | 0/5 | 36,040 |
| | Raw files | — | 1.000 | 0.0470 | 10 / 9 / 9 / 9 / 9 | 1/5 | 20,020 |
| | Pasted examples | control | 1.000 | 0.0493 | 10 / 9 / 9 / 9 / 9 | 1/5 | 11,285 |

On `brackenford-footbridge-proposal`, median contamination was 0.0000, with one outlier round at 0.1118. On `kettlecraft-winter-campaign-brief`, chunk search won 3 of 5 rounds in the 9-document run, while Arkive matched the control on all five judge medians (10 / 9 / 9 / 9 / 9).

### Live OpenCode check on the previous version

Three snapshots were mounted at once (`coastal-survey-projects`, `creator-campaigns` and the 126-file `structural-monitoring-consultancy`) in OpenCode 2.3.0 with `openai/gpt-6.1-sol`:

- **Result:** 10 of 10 scenarios passed: topic and filter discovery, three glossary lookups (a defined term, an undefined term and a term with two meanings), a listing filtered on a filter the assistant found itself, recovery from an invalid filter, an exact number found with `grep`, a chart value read with `view_page`, a draft from the drafting packet, and a question across two topics.
- **Tool discipline:** no calls to tools outside Arkive and no cut-off results.
- **Cost reported by OpenCode:** at least $0.1134 across all 10 scenarios, 12 to 16 seconds each.

## Previous-version findings

1. **Provenance prevented a policy leak.** On `corvane-depot-night-noise-report`, the pasted examples led the drafts to copy another project's six-year audio retention policy, which dropped their discipline median to 4 (contamination 0.0757). The drafting packet reports how many version groups share each section and their dates, and Arkive avoided the borrowed policy (discipline 10, contamination 0.0069). It won 5 of 5 rounds and used 57% fewer input tokens than raw files.
2. **The advantage appeared at scale.** On 9-file collections, the MCP tool descriptions add a fixed cost to every turn, so pasting two or three short Markdown files uses fewer tokens. On the 126-file archive, Arkive beat raw files on question accuracy with 42% fewer tokens. On the 52-page report drafting test it used 38% fewer input tokens than raw files and 67% fewer than pasted examples.
3. **Very long reports remained a gap.** On `tallow-cross-fbg-installation-report`, pasted examples won 5 of 5 rounds (judge medians 9 / 9 / 9 / 8 / 9 against Arkive's 7 / 8 / 8 / 9 / 7) and passed the house temperature-compensation check 5 of 5 times against Arkive's 0 of 5. On the current version, Arkive passes that check 5 of 5 times and all 60 rubric checks, while pasted examples still win every round (see [Drafting tests](#drafting-tests)).
4. **Filter names overlapped.** On the 126-file archive, organising in batches produced two overlapping technology filters, `sensing_technology` and `monitoring_technology`. Both remained valid filters and did not block questions or drafting. The current version merges filters across batches, and no build produced both names, but `project` coverage still varies between builds (see [Version groups and filter coverage](#version-groups-and-filter-coverage)).

## Running an evaluation

The commands below rerun the tests that use tracked data: the four collections in `evals/example_corpora/` with their eval packs in `evals/example_packs/`, and the OpenCode scenarios in `scenarios.json` and `scenarios-uplift.json`. The extra current-version question and drafting sets and the three synthetic collections are kept outside Git, so the repository cannot rerun those results. Every command below makes paid OpenAI calls, and the OpenCode check also needs Docker.

```sh
# 1. Build a public example snapshot outside the repository
#    (repeat for creator-campaigns, and for structural-monitoring-consultancy
#    with a higher --max-cost, since its first build cost $4.67)
uv run --locked arkive-rag build evals/example_corpora/coastal-survey-projects \
  -o /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  --cache /tmp/arkive-examples/coastal-survey-projects.cache.db --max-cost 2

# 2. Run the search, question and drafting tests against it,
#    with each question repeated 3 times per approach
uv run --locked python -m evals.run \
  --pack evals/example_packs/coastal-survey-projects.eval.json \
  --snapshot /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  --corpus evals/example_corpora/coastal-survey-projects \
  --out /tmp/arkive-eval-outputs/coastal-survey-projects \
  --layer all --qa-repeats 3 --drafts 3 --max-cost 6

# 3. Run the live Docker OpenCode check: the ten default scenarios on two snapshots
uv run --locked python -m evals.opencode_check \
  /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  /tmp/arkive-examples/creator-campaigns.arkive.db \
  --out /tmp/arkive-opencode-check

# 4. Run the thirteen extended scenarios, which need all three snapshots
uv run --locked python -m evals.opencode_check \
  /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  /tmp/arkive-examples/creator-campaigns.arkive.db \
  /tmp/arkive-examples/structural-monitoring-consultancy.arkive.db \
  --scenarios evals/opencode_check/scenarios-uplift.json \
  --out /tmp/arkive-opencode-check-uplift
```
