# Appendix C: Evaluation harness and full benchmark tables

> Technical appendix. For the plain-language summary, read [Evidence, benchmarks, and known limits](../results-and-limits.md). The authoritative specification is in [Evaluation](../architecture.md#evaluation), [Example corpora](../architecture.md#example-corpora), and [OpenCode check](../architecture.md#opencode-check).

The evaluation harness in `evals/` tests whether a compiled snapshot beats giving the same model direct file tools over the raw corpus, and compares Arkive with a control that pastes human-selected full exemplars into the prompt. Every result below comes from the four public fictional collections in `evals/example_corpora/`, except three synthetic stress collections in the schema `7.0` section, which are also fictional and are described there.

![Arkive preflight-gated evaluation harness](../assets/evaluation-harness.svg)

[Open the interactive diagram](../assets/evaluation-harness.html)

## The four public fictional collections

Each collection has a matching `arkive-eval-pack/3` pack in `evals/example_packs/`, and each stresses a different corpus shape.

| Collection | Files and formats | Structural traps tested |
| --- | --- | --- |
| `coastal-survey-projects` | 9 files (`.md`, `.pdf`, `AGENTS.md`) across 3 project folders | Proposals against technical memoranda, reviewer feedback lessons, domain acronyms (`MBES`), cross-type bleed between proposals and memos. |
| `creator-campaigns` | 9 files (`.md`, `.pptx`, `AGENTS.md`) across 2 creators and 4 campaigns | Campaign briefs, review notes, a PowerPoint launch deck, policy clauses borrowed across creators. |
| `acoustic-consultancy-files` | 9 files (`.docx`, `.pdf`) in one flat folder, no `AGENTS.md` | Dates only from embedded file metadata, old letter and newer numbered report layouts, shared caveats that differ by one instrument name, chart-only values, a one-project data-retention policy. |
| `structural-monitoring-consultancy` | 126 files (110 `.md` including 2 `AGENTS.md`, 10 `.docx`, 6 `.pdf`; 110 rendered pages) | Scale: 17 authored types (28 induced), a 52-page Word baseline report, superseded draft families, a byte-identical duplicate, unlabelled QA disposition registers, method distractors (vibrating-wire against fibre Bragg grating), and a vector-only PDF chart value (`-142`). |

## Evaluation layers

Before it creates `--out`, opens `--cache`, or calls OpenAI, `python -m evals.run` runs a preflight that checks the pack schema, the snapshot `corpus_fingerprint`, facet filters, Q&A anchor evidence sizes, holdout leak guards, LibreOffice availability, and that output paths do not overlap inputs. Outputs are written atomically with owner-only (`0600`) permissions.

1. **Layer 1, retrieval ablation (`evals/ablation.py`).** Compares `bm25_only`, `dense_only`, `hybrid_rrf`, and `served_search` (the search agents get, hiding superseded drafts and capping each version family at 3 chunks) with one shared query embedding per case. Cases cover exact identifiers, paraphrases, visual-only chart and map questions, repaired font ligatures, and disambiguation with facet `filters`. Reports `Completion@1/3/5` and `CompletionMRR@5`.
2. **Layer 2, grounded Q&A (`evals/agents.py`).** Runs the same `gpt-6.1-sol` agent conversation (OpenAI Agents SDK through `OpenAIGateway.converse`) on lookup, multi-document, aggregation, conflict, abstention, and vocabulary cases under two tool sets: raw file tools (`list_corpus_files`, `grep_raw_text`, `read_raw_pages`, `view_raw_page_image`) and the eleven Arkive MCP tools. Completed answers get phrase coverage and a blinded `gpt-6.1-sol` judge that reads the reference facts and the snapshot text of the anchored pages and labels every material claim `supported`, `contradicted`, or `unverifiable`. Token usage, each tool call, and latency are recorded.
3. **Layer 3, holdout drafting (`evals/drafting.py`).** Builds an isolated `holdout.arkive.db` without the target. Guards exclude files whose path holds a held-out marker and files byte-identical to the reference. The guards infer nothing from folders and do not scan content, so the pack author must exclude changed copies and documents that restate held-out facts. When the holdout names exemplar documents, one `prepare_draft` call first records how many of them the packet surfaces. Each of `--drafts` rounds runs up to five arms, all drafting from project run notes:
   - `zero_shot`: no tools;
   - `chunk_search`: plain chunk search only;
   - `raw_files`: file tools over the included files;
   - `arkive_mcp`: the eleven Arkive tools, centred on `prepare_draft`;
   - `exemplars_in_context`: the holdout's named exemplars pasted into the prompt under the frozen `EXEMPLAR_DRAFTING_RULES`. This is the control, and it runs only when the holdout names exemplars.

   Each completed draft gets deterministic rubric checks, two per-document lexical 4-gram measures (house-text recall, and copied-text contamination from included documents the reference does not carry), and one blinded `gpt-6.1-sol` comparison per round that receives `holdout.instructions`, `run_notes`, and the reference.
4. **Live OpenCode check (`evals/opencode_check/`).** Runs the ten scenarios of `scenarios.json`, or the thirteen of `scenarios-uplift.json` (which adds a deliverable-type listing, a version-history question, and a `query_tables` read on `structural-monitoring-consultancy`), in fresh Docker containers with OpenCode 2.3.0 (`openai/gpt-6.1-sol`), snapshots mounted read-only, and only the Arkive MCP tools enabled. A scenario fails if OpenCode calls a non-Arkive tool or truncates an Arkive result.

## Schema `7.0` uplift benchmarks

These runs measured the current schema `7.0` snapshots against the schema `6.0` baseline below, on `gpt-6.1-sol`, `gpt-6-luna`, and `text-embedding-3-large`. Repeats:

- The Layer 2 cases on `structural-monitoring-consultancy` and on the synthetic stress collections ran 3 times per arm.
- The chart cases on the three 9-file collections ran once per arm.
- Every Layer 3 holdout ran 5 rounds per arm.

Completed metered runs cost $18.06 in total: $16.36 for the core suite on the tracked collections and $1.69 for the synthetic stress suite. One further drafting run cost about $2.05 before a host restart killed it mid-way. Its holdout build was cached, and the clean rerun reused it at no cost.

The tracked collections were evaluated with additional proof packs (current-value and version-history questions, cross-report aggregation, one chart case per collection, and a counterfactual drafting holdout) that are not tracked in `evals/example_packs/`. Three further synthetic stress collections tested version traps (`version-traps`, 8 files), a 60-document table rollup (`table-scale-60`: 58 `.md`, 1 `.xlsx`, 1 `.html`), and five messy files in five formats (`formats-stress`). These collections and the extra proof packs are kept outside Git, and every document in them is fictional.

### Metered build cost

| Collection | Docs | Chunks | Provider calls | Input tokens | Output tokens | Elapsed | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `structural-monitoring-consultancy` (cold) | 126 | 1,086 | 490 | 1,567,475 | 285,214 | 2,540.7 s | $4.6673 |
| `structural-monitoring-consultancy` (unchanged rebuild with `--cache`) | 126 | 1,086 | 0 | 0 | 0 | — | $0.0000 |
| `acoustic-consultancy-files` (cold) | 9 | 71 | 42 | 87,387 | 23,053 | 207.6 s | $0.3667 |
| `coastal-survey-projects` (cold) | 9 | 54 | 42 | 49,542 | 14,443 | 148.1 s | $0.1986 |
| `creator-campaigns` (cold) | 9 | 47 | 31 | 38,520 | 11,833 | 141.7 s | $0.1754 |

The 126-file cold build by operation: `transcribe_page` $2.4820 (110 calls), `observe` $1.3761 (127), `write_glossary` $0.3670 (10), `organise_corpus` $0.1198 (2), `partition_versions` $0.0830 (13), `write_guide` $0.0597 (1), `extract_feedback_lessons` $0.0574 (15), `embed` $0.0340 (10), `bind_facets` $0.0273 (124), `canonicalise_facets` $0.0267 (1), `check_definitions` $0.0155 (58), `canonicalise_types` $0.0146 (1), `consolidate_card` $0.0034 (1), `check_feedback_lessons` $0.0008 (17). The earlier estimate of about $1.22 for this collection predates the schema `7.0` pipeline and its per-operation metering.

### Version families and facet coverage

Three independent schema `7.0` inductions of `structural-monitoring-consultancy` were compared with the 5 source-verified multi-version families (three draft series of a method statement, an installation report, and a scoping note; a proposal `v0.9`/`v1.0` pair; and a byte-identical calibration sheet in two job folders, which has no defined latest member):

| Induction | Exact families | `latest` correct | Missed / false families | Facets induced (documents covered of 124 citable) |
| --- | --- | --- | --- | --- |
| Full build | 5/5 | 4/4 | 0 / 0 | `client` (91), `job_cohort` (110), `sensing_technology` (122), `issue_status` (124), `project` (23) |
| Fresh induction 2 | 5/5 | 4/4 | 0 / 0 | `client` (91), `project` (110), `archive_section` (110), `sensing_technology` (119), `issue_status` (124) |
| Fresh induction 3 | 5/5 | 4/4 | 0 / 0 | `client` (91), `job_status` (110), `monitoring_technology` (124), `issue_status` (124), `project` (23) |

No induction produced a facet that duplicates `deliverable_type`, and installation reports were typed identically in every run. Coverage of `project` varied: in two runs `organise_corpus` ran in two batches and only the second batch's 6 archive jobs became `project` values (23 of 124 documents), while the other run listed all 29 jobs (110 of 124). In one Layer 2 repeat the agent filtered on a project outside those values, received the valid-values error, and recovered on its next call. `facet_overlap` flags pairs with a document-set Jaccard of at least 0.85 (0.87 to 1.00 here); these were orthogonal facets that each cover nearly every document, such as `issue_status` and `sensing_technology`.

The synthetic version-trap collection (8 files) produced 1 of 1 exact family with the correct latest member and 0 false supersessions:

- **Date-inverted draft.** A `v0.9` Word draft with no printed date and a file metadata date of 2026-06-15 was grouped with the `v1.0` PDF final printed 2026-05-01. The final is `latest` and the draft `superseded`.
- **Addendum.** A base report and its Addendum 1 stayed separate `sole` documents.
- **Project phases.** Phase 1 and phase 2 studies stayed separate.
- **Monthly series.** Two consecutive monthly bulletins stayed separate.

On its Layer 2 questions, `arkive_mcp` reported the current crack width in 3 of 3 runs with no forbidden superseded value, and the default `versions="current"` kept both the base report and the addendum (3 of 3).

### Retrieval on the frozen structural pack (Layer 1)

`evals/example_packs/structural-monitoring-consultancy.eval.json`, 10 cases, top 5:

| Arm | Schema `6.0` (C@1 / C@3 / C@5 / MRR@5) | Schema `7.0` (C@1 / C@3 / C@5 / MRR@5) |
| --- | --- | --- |
| `bm25_only` | 20% / 40% / 50% / 0.325 | 20% / 50% / 50% / 0.333 |
| `dense_only` | 20% / 40% / 40% / 0.283 | 20% / 40% / 40% / 0.283 |
| `hybrid_rrf` | 10% / 30% / 60% / 0.248 | 20% / 50% / 50% / 0.317 |
| `served_search` | 10% / 30% / 50% / 0.245 | 20% / 50% / 50% / 0.350 |

`hybrid_rrf` lost 10 points of `Completion@5` while gaining at `@1` and `@3`.

### Current values, version history, and aggregation (Layer 2)

`structural-monitoring-consultancy`, 7 cases × 3 repeats = 21 runs per arm:

| Arm | Phrase pass | Accuracy | Citations | Completeness | Contradicted | Unverifiable | Input tokens | Tool calls | Elapsed |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `arkive_mcp` | 21/21 | 5.00 | 5.00 | 4.95 | 0 | 3 | 793,249 | 122 | 404 s |
| `raw_files` | 21/21 | 4.81 | 4.76 | 4.86 | 0 | 7 | 774,296 | 154 | 441 s |

Per case (mean input tokens per run):

| Case | Kind | Arm | Phrase | Acc / Cit / Cmp | Contradicted / Unverifiable | Mean input tokens | Tool calls |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| `greyscar_current_alarm_levels` | `current_value` | `arkive_mcp` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 4,710 | 3 |
| | | `raw_files` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 11,253 | 6 |
| `mill_lane_current_fee` | `current_value` | `arkive_mcp` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 5,511 | 4 |
| | | `raw_files` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 9,673 | 6 |
| `greyscar_alarm_version_history` | `version_history` | `arkive_mcp` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 17,674 | 20 |
| | | `raw_files` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 16,681 | 15 |
| `mill_lane_fee_version_history` | `version_history` | `arkive_mcp` | 3/3 | 5.00 / 5.00 / 4.67 | 0 / 0 | 8,272 | 9 |
| | | `raw_files` | 3/3 | 4.67 / 4.67 / 4.00 | 0 / 1 | 10,646 | 12 |
| `q5_all_issued_installation_reports` | `aggregation` | `arkive_mcp` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 2 | 177,029 (runs 358.8k, 97.6k, 74.7k) | 55 |
| | | `raw_files` | 3/3 | 5.00 / 4.67 / 5.00 | 0 / 1 | 98,412 | 62 |
| `q6_prisma_fi4_projects_alarms` | `aggregation` | `arkive_mcp` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 1 | 40,623 | 25 |
| | | `raw_files` | 3/3 | 4.00 / 4.00 / 5.00 | 0 / 5 | 90,196 | 44 |
| `ellery_largest_peak_change` | `lookup` | `arkive_mcp` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 10,598 | 6 |
| | | `raw_files` | 3/3 | 5.00 / 5.00 / 5.00 | 0 / 0 | 21,238 | 9 |

The four current-value and version-history cases give `arkive_mcp` 12 of 12 phrase passes, 0 forbidden superseded values, 0 contradicted claims, and 5.00 accuracy. On `q5_all_issued_installation_reports` the median `arkive_mcp` run used 97.6k input tokens and the mean 177.0k, against 369,752 in a single schema `6.0` run of the same question in an earlier experiment (−73.6% median, −52.1% mean). One schema `7.0` run still used 358.8k, so the reduction is not reliable on every run. Against `raw_files` on the same case, the `arkive_mcp` median was 97.6k against 102.7k and the mean 177.0k against 98.4k.

`arkive_mcp` answered the two aggregation cases by listing documents filtered on `deliverable_type` and `issue_status`, then reading their attributes and sections with `search`, `grep`, and `read_section`. The two cheaper `q5` runs made no `query_tables` call. The 358.8k run made 2 `query_tables` calls, one of them refused for `limit=100`. On `q6_prisma_fi4_projects_alarms`, 1 of 3 runs called `query_tables`.

### Table rollup at 60 documents

The synthetic 60-document collection holds 203 anchor proof-load rows across 58 Markdown reports whose prose never mentions the tables, a master asset schedule workbook, and a regional summary HTML page. `document_tables` indexed all 60 documents (60 tables, 227 rows). Two questions ran 3 times per arm: the failing `Zone C` rows (6 rows across 6 documents, totalling 2,520 kN) and a join of one region's assets across the workbook and the HTML page.

| Arm | Phrase pass | Accuracy / Citations / Completeness | Contradicted | Turns (mean) | Tool calls | Input tokens (6 runs) |
| --- | --- | --- | --- | --- | --- | ---: |
| `arkive_mcp` | 6/6 | 5.00 / 5.00 / 5.00 | 0 | 2–6 (3.33) | 14 (7 `query_tables`, 5 `search`, 2 `grep`) | 166,220 |
| `raw_files` | 6/6 | 5.00 / 5.00 / 5.00 | 0 | 3–4 (3.50) | 33 | 33,279 |

Two tool findings explain the token gap:

1. **OR expansion in `query_tables`.** `build_fts5_query` joins query words with OR. One `Zone C Fail` repeat that called `query_tables(query="Fail")` finished in 3 turns with 7,238 input tokens. The two repeats that called `query_tables(query="Zone C Fail")` matched every anchor table through the header word `Zone`, returned 49 tables (44,521 characters) plus an offset page, and used 68,922 and 76,615 input tokens.
2. **Limit errors.** Twice the agent asked for `limit=100` and spent a turn on the `limit must be between 1 and 50` error.

`raw_files` passed because the collection has only 15 `Fail` rows, under the raw `grep_raw_text` cap of 30 matches, so one text search found every failing row.

### Charts across all four collections

| Collection and chart value | Layer 1 `served_search` (C@1 / C@3 / MRR@5) | `arkive_mcp` phrase / Acc / Cit / Cmp / contradicted | `raw_files` phrase / Acc / Cit / Cmp / contradicted |
| --- | --- | --- | --- |
| `structural` (`EW-05 = -142`, 3 repeats) | 0% / 100% / 0.500 | 3/3 / 5.00 / 5.00 / 5.00 / 0 | 3/3 / 5.00 / 5.00 / 5.00 / 0 |
| `acoustic` (`05-06 = 14`, 1 repeat) | 0% / 100% / 0.500 | 1/1 / 5.00 / 5.00 / 5.00 / 0 | 1/1 / 5.00 / 5.00 / 5.00 / 0 |
| `coastal` (`Bench 3 = 41,700`, 1 repeat) | 100% / 100% / 1.000 | 1/1 / 5.00 / 5.00 / 5.00 / 0 | 1/1 / 5.00 / 5.00 / 5.00 / 0 |
| `creator` (`Week 3 = 48,200`, 1 repeat) | 100% / 100% / 1.000 | 1/1 / 5.00 / 5.00 / 5.00 / 0 | 1/1 / 5.00 / 5.00 / 5.00 / 0 |

Every chart value was transcribed verbatim in a `[Visual — Page N: ...]` block.

### New formats and messy Office files

The synthetic collection has one file per format: a Word file with a tracked fee change and a reviewer comment, a deck with a hidden slide, speaker notes, a comment thread, and a native bar chart, a two-sheet workbook with a merged header, uncached formulas, a legacy note, a disconnected data region, and a threaded comment, an HTML page with text before the first heading, a spanning table header, and a blocked loopback image, and a 3-page PDF with 4-line running headers and compound units. All programmatic snapshot checks passed on 5 of 5 files:

- The deleted fee appears only in an annotation and never fused with its replacement.
- The hidden slide is absent, and notes, comments, and chart values are stored.
- Uncached formulas are kept.
- The HTML page kept its text before the first heading and its spanning-header table, and the stored text omits the image and its alt text.
- Running headers were removed while compound units survived.

The snapshot check reads stored content only. The offline test `test_an_html_page_reads_in_order_without_fetching_anything` in `tests/test_native_formats.py` verifies that converting a page with a loopback image and a local-file image opens no socket, makes no URL request, and does not open the local file.

| Case | `arkive_mcp` | `raw_files` |
| --- | --- | --- |
| Workbook formula and comments | 3/3, 5.00 accuracy | 3/3, 5.00 |
| HTML pre-heading text and table | 3/3, 5.00 | 3/3, 5.00 |
| Word tracked change and comment | 3/3, 5.00 | 0/3, 2.00 |
| Deck chart, notes, comments, no hidden slide | 2/3, 4.33 | 0/3, 1.00 |
| PDF compound units | 3/3, 5.00 | 3/3, 5.00 |
| **Overall** | **14/15, 4.87 accuracy, 5.00 citations, 1 contradicted claim** | **9/15, 3.60 accuracy, 3.80 citations, 14 contradicted claims** |

The raw-file tools see the same Markdown for workbooks and HTML pages, but cannot read Word comments, PowerPoint notes and comments, or chart XML. The one `arkive_mcp` miss: the fixture attaches the chart without a drawn chart shape, so the rendered slide shows no bars. In 2 of 3 runs the agent followed `view_page` with `read_section` and found the stored chart table. In the third it trusted the blank image.

### Holdout drafting (Layer 3)

`tallow-cross-fbg-installation-report` (52-page Askern Viaduct exemplar), `arkive_mcp`:

| Measure | Schema `6.0` | Schema `7.0` | Other arms (schema `7.0`) |
| --- | --- | --- | --- |
| Rubric checks passed | 55/60 (`house_temperature_compensation` 0/5) | 60/60 (`house_temperature_compensation` 5/5) | `raw_files` 55/60 (0/5), `chunk_search` 56/60 (1/5), `exemplars_in_context` 60/60 |
| House-text recall, median [min, max] | 0.656 [0.633, 0.679] | 0.800 [0.753, 0.819] | `chunk_search` 0.428, `raw_files` 0.335, `exemplars_in_context` 0.977 |
| Copied-text contamination, median | 0.0000 (0/4,336) | 0.0011 (4/4,521) | `exemplars_in_context` 0.0135 (79/3,543) |
| Blind-judge rank | 2nd ×1, 3rd ×2, 4th ×2 | 2nd in 5/5 (5–0 against `raw_files` and `chunk_search`) | `exemplars_in_context` 1st in 5/5 |
| Judge medians (`S/R/L/D/E`) | 7 / 8 / 8 / 9 / 7 | 7 / 8 / 8 / 8 / 8 | `raw_files` 8 / 7 / 8 / 9 / 7, `chunk_search` 7 / 7 / 8 / 9 / 6 |
| Tool calls / input tokens (5 rounds) / arm cost | 26 / 179k / $0.3982 | 103 / 494,829 (350k cached) / $0.5898 | `raw_files` 298,184, `chunk_search` 134,324, `exemplars_in_context` 540,990 input tokens |

The packet surfaced 1 of the 3 named exemplars (3 of 3 among its candidates). The agent made 103 tool calls against 26 on schema `6.0`, reading further sections, and its input tokens rose accordingly.

`tallow-cross-counterfactual-sfr` changes the run notes so a batch calibration certificate's `9.4 pm/°C` replaces the house standard `9.7 pm/°C`. `arkive_mcp` ranked 1st in round 5 (ahead of `exemplars_in_context`) and 2nd in rounds 1 to 4, ahead of `raw_files` and `chunk_search` in all 5, with house-text recall 0.827 [0.822, 0.879], Discipline 9, Effort saved 8, and contamination 0.0040. The automated `counterfactual_sfr_override` check scored 3/5 (all 11 other checks 5/5). A reading of all 5 drafts shows that every one applied `9.4 pm/°C` and none applied `9.7 pm/°C`. The 2 failures are rounds whose supersession sentence ("replaces the standard SFR temperature sensitivity of 9.7 pm/°C") contains the forbidden substring `sensitivity of 9.7 pm`, so the check's literal match cannot distinguish naming a replaced value from using it.

`corvane-depot-night-noise-report`, `arkive_mcp`:

| Measure | Schema `6.0` | Schema `7.0` |
| --- | --- | --- |
| House-text recall, median [min, max] | 0.343 [0.343, 0.350] | 0.759 [0.759, 1.000] (12 of 15 house-unit hits: 3 house units × 5 rounds) |
| Copied-text contamination, median | 0.0069 (25/3,581) | 0.0071 (25/3,512) |
| `no_borrowed_site_values` | 5/5 | 5/5 (rubric total 24/25, `mitigation_kept` 4/5) |
| Blind-judge 1st place | 5/5 (Discipline 10) | 4/5 (round 2 ranked 4th; Structure 9, Effort saved 9, Discipline 7 [6, 8]) |
| Input tokens (5 rounds) | 47,880 | 53,670 (`raw_files` 119,433) |

Discipline fell because the drafts reproduced the house survey-method text (compliance procedure, microphone height and clearance, weather screening) that the run notes did not state. The judge noted that this text matches the reference and found no cross-site contamination.

### Live OpenCode check with the uplift scenarios

The thirteen scenarios of `scenarios-uplift.json` ran against the `coastal-survey-projects`, `creator-campaigns`, and `structural-monitoring-consultancy` snapshots (the command is under [Running an evaluation](#running-an-evaluation)). They ran in the Alpine (musl, Python 3.14) image `arkive-opencode-check:9f88b15ebbec`, built with `--require-hashes --only-binary :all:` and `docling-slim[format-xlsx,format-html]`, on OpenCode 2.3.0 with `openai/gpt-6.1-sol`:

- **Scenarios passed:** 13 of 13, with 0 non-Arkive tool calls and 0 truncated MCP results, in 306 s at a lower-bound cost of $0.2644.
- **New scenarios:** `type-filtered-list` (2 calls; returned all 14 `Installation report` documents), `version-history-discovery` (5 calls; used `versions="all"` to compare `v0.8` and `v0.9` with `v1.0`), and `table-rollup` (6 `query_tables` calls; returned the expected calibration months and the `1.21 pm/µε` array sensitivity).

## Schema `6.0` baseline benchmarks

These are the pre-uplift results. All runs used `gpt-6.1-sol`, `gpt-6-luna`, and `text-embedding-3-large` on schema `6.0` snapshots of the tracked collections, with the ten-tool server of that version. Layer 3 runs used 5 rounds per arm.

### Grounded Q&A at 126-file scale (Layer 2)

`structural-monitoring-consultancy`, 8 Q&A cases across lookup, multi-document, aggregation, conflict, abstention, and vocabulary:

| Arm | Phrase coverage | Accuracy (1-5) | Citations (1-5) | Completeness (1-5) | Contradicted claims | Unverifiable claims | Input tokens | Tool calls | Elapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `raw_files` | 6/8 | 4.75 | 4.62 | 4.62 | 0 | 6 | 134,043 | 36 | 100 s |
| `arkive_mcp` | **8/8** | **4.88** | **5.00** | **4.75** | **0** | **1** | **77,215** (-42%) | **21** | **85 s** |

Raw-file extraction took a further 29.2 s up front. On the two conflict cases, where superseded drafts disagree with issued documents, `raw_files` reached 0/2 phrase coverage and `arkive_mcp` 2/2.

Layer 1 on the same snapshot (10 cases, top 5): `hybrid_rrf` reached 60% `Completion@5`, `served_search` 50%, `bm25_only` 50%, and `dense_only` 40%. All four arms reached 100% `Completion@3` on the vector-only PDF chart case (`visual_only`).

### Holdout drafting across all four collections (Layer 3)

Medians across 5 completed rounds. The blind judge scores Structure, Rigor, Limitations, Discipline, and Effort saved (`S/R/L/D/E`) from 1 to 10. Token counts are 5-round totals.

| Collection and holdout | Arm | Selection (packet / cand.) | House recall | Contamination | Judge medians (`S/R/L/D/E`) | 1st-place wins | Input tokens (5 rounds) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Structural (126 docs)** `brackenford-footbridge-proposal` | `arkive_mcp` | 2/5 · 5/5 | **0.984** | **0.0000** | **9 / 9 / 9 / 10 / 9** | **5/5** | 221,802 |
| | `raw_files` | — | 0.810 | 0.0000 | 8 / 9 / 9 / 8 / 8 | 0/5 | 212,379 |
| | `exemplars_in_context` | control | 0.971 | 0.0000 | 9 / 8 / 8 / 10 / 8 | 0/5 | 30,175 |
| **Structural (126 docs)** `tallow-cross-fbg-installation-report` (52-page exemplar) | `arkive_mcp` | 2/3 · 3/3 (fb 1/1) | **0.656** | **0.0000** | 7 / 8 / 8 / 9 / 7 | 0/5 | **179,051** |
| | `raw_files` | — | 0.377 | 0.0000 | 8 / 8 / 8 / 9 / 7 | 0/5 | 287,143 |
| | `exemplars_in_context` | control | 0.977 | 0.0226 | **9 / 9 / 9 / 8 / 9** | **5/5** | 540,990 |
| **Acoustic (flat folder)** `corvane-depot-night-noise-report` | `arkive_mcp` | 3/3 · 3/3 | 0.343 | **0.0069** | **9 / 8 / 9 / 10 / 9** | **5/5** | **47,880** |
| | `raw_files` | — | 0.292 | 0.0000 | 6 / 8 / 9 / 10 / 7 | 0/5 | 112,328 |
| | `exemplars_in_context` | control | 1.000 | 0.0757 | 8 / 6 / 5 / 4 / 6 | 0/5 | 17,340 |
| **Acoustic (flat folder)** `pellow-mill-facade-report` | `arkive_mcp` | 3/3 · 3/3 | 0.346 | **0.0000** | **8 / 9 / 7 / 10 / 8** | **3/5** | 48,405 |
| | `raw_files` | — | 0.309 | 0.0000 | 8 / 8 / 7 / 10 / 8 | 1/5 | 28,760 |
| | `exemplars_in_context` | control | 0.346 | 0.0000 | 8 / 8 / 7 / 10 / 8 | 1/5 | 16,980 |
| **Coastal (9 docs)** `fennick-canal-proposal` | `arkive_mcp` | 2/2 · 2/2 | **1.000** | **0.0000** | **9 / 10 / 10 / 10 / 9** | **2/5** | 38,815 |
| | `raw_files` | — | 1.000 | 0.0000 | 9 / 10 / 10 / 10 / 8 | 0/5 | 26,000 |
| | `exemplars_in_context` | control | 1.000 | 0.0000 | 9 / 10 / 9 / 10 / 9 | 2/5 | 8,305 |
| **Coastal (9 docs)** `fennick-canal-technical-memorandum` | `arkive_mcp` | 2/2 · 2/2 (fb 2/2) | n/a (0 house units) | **0.0000** | **9 / 9 / 10 / 10 / 9** | 1/5 | 32,890 |
| | `raw_files` | — | n/a | 0.0000 | 9 / 10 / 10 / 10 / 9 | **4/5** | 19,640 |
| | `exemplars_in_context` | control | n/a | 0.0000 | 9 / 8 / 8 / 8 / 8 | 0/5 | 10,905 |
| **Creator (9 docs)** `kettlecraft-winter-campaign-brief` | `arkive_mcp` | 3/3 · 3/3 (fb 2/2) | **1.000** | 0.0526 | **10 / 9 / 9 / 9 / 9** | 0/5 | 36,040 |
| | `raw_files` | — | 1.000 | 0.0470 | 10 / 9 / 9 / 9 / 9 | 1/5 | 20,020 |
| | `exemplars_in_context` | control | 1.000 | 0.0493 | 10 / 9 / 9 / 9 / 9 | 1/5 | 11,285 |

On `brackenford-footbridge-proposal`, median contamination was `0.0000` with one outlier round at `0.1118`. On `kettlecraft-winter-campaign-brief`, `chunk_search` won 3 of 5 rounds in the 9-document run, while `arkive_mcp` matched the control on all five judge medians (`10/9/9/9/9`).

### Live OpenCode 2.3.0 container check

Three snapshots were mounted at once (`coastal-survey-projects`, `creator-campaigns`, and the 126-file `structural-monitoring-consultancy`) in OpenCode 2.3.0 (`openai/gpt-6.1-sol`):

- **Scenarios passed:** 10 of 10 (`topic-and-facet-discovery`, `glossary-defined-term`, `glossary-undefined-term`, `glossary-two-senses`, `filtered-listing-discovered-facet`, `invalid-filter-recovery`, `grep-exact-number`, `chart-value-view-page`, `draft-from-packet`, `cross-topic-question`).
- **Tool discipline:** no non-Arkive tool calls and no truncated MCP results.
- **Cost reported by OpenCode:** at least $0.1134 across all 10 scenarios, 12 to 16 seconds each.

## Schema `6.0` baseline findings

1. **Provenance prevents policy leaks.** On `corvane-depot-night-noise-report`, `exemplars_in_context` copied another project's six-year audio retention policy, which dropped its Discipline median to 4 (contamination `0.0757`). `prepare_draft` reports family counts and dates for each section, and `arkive_mcp` avoided the borrowed policy (Discipline 10, contamination `0.0069`), won 5 of 5 rounds, and used 57% fewer input tokens than `raw_files`.
2. **The advantage appears at scale.** On 9-file collections, MCP tool schemas add a fixed per-turn overhead, so pasting two or three short Markdown files uses fewer tokens. On the 126-file archive, Arkive beat `raw_files` on Q&A accuracy with 42% fewer tokens, and cut input tokens on the 52-page report holdout by 38% against `raw_files` and 67% against `exemplars_in_context`.
3. **Gap on very long multi-section reports.** On `tallow-cross-fbg-installation-report`, `exemplars_in_context` won 5 of 5 rounds (`9/9/9/8/9` against Arkive's `7/8/8/9/7`) and passed the `house_temperature_compensation` rubric check 5 of 5 times against Arkive's 0 of 5. On schema `7.0`, `arkive_mcp` passes that check 5 of 5 times and all 60 rubric checks, while `exemplars_in_context` still wins every round (see [Holdout drafting](#holdout-drafting-layer-3)).
4. **Overlapping facet names.** On the 126-file archive, batched organisation produced two overlapping technology facets, `sensing_technology` and `monitoring_technology`. Both remain valid filters and did not block Q&A or drafting. Schema `7.0` canonicalises facets across batches and no run produced both names, but `project` coverage still varies between builds (see [Version families and facet coverage](#version-families-and-facet-coverage)).

## Running an evaluation

The commands below reproduce the benchmarks that use tracked inputs: the four collections in `evals/example_corpora/` with their packs in `evals/example_packs/`, and the OpenCode scenarios in `scenarios.json` and `scenarios-uplift.json`. The additional proof packs and the three synthetic stress collections are kept outside Git, so the repository cannot rerun those results.

```sh
# 1. Build a public example snapshot outside the repository
#    (repeat for creator-campaigns, and for structural-monitoring-consultancy
#    with a higher --max-cost, since its cold build cost $4.67)
uv run --locked arkive-rag build evals/example_corpora/coastal-survey-projects \
  -o /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  --cache /tmp/arkive-examples/coastal-survey-projects.cache.db --max-cost 2

# 2. Run layers 1-3 against it, with each Q&A case repeated 3 times per arm
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

# 4. Run the thirteen uplift scenarios, which need all three snapshots
uv run --locked python -m evals.opencode_check \
  /tmp/arkive-examples/coastal-survey-projects.arkive.db \
  /tmp/arkive-examples/creator-campaigns.arkive.db \
  /tmp/arkive-examples/structural-monitoring-consultancy.arkive.db \
  --scenarios evals/opencode_check/scenarios-uplift.json \
  --out /tmp/arkive-opencode-check-uplift
```
