# What Arkive does and how a team uses it

Arkive RAG turns a folder of a team's best reference documents into one knowledge file that AI assistants can search, quote, and draft from. The file holds the text, tables, chart values, and page images of every document, plus a map of how the documents relate: which are drafts and which are final, which are reviewer feedback, and which wording the team reuses on every job.

![Team curates a folder, Arkive builds a snapshot, assistants answer and draft from it](assets/team-workflow.svg)

[Open the interactive diagram](assets/team-workflow.html)

## Problems with raw folders and generic AI chat

A shared drive of past reports is hard for an AI assistant to use well:

- **Documents are large.** A 50-page Word report or a slide deck does not fit comfortably in one prompt, so the assistant reads only part of it or reads it many times.
- **Values hide in images.** A chart legend or a map label often exists only as pixels. Plain text extraction misses it.
- **Old drafts sit beside final versions.** An assistant that reads a v0.8 draft can quote a number the issued v1.0 report corrected.
- **Past projects leak into new work.** When an assistant copies from a complete old report, it can carry over that project's client name, site details, or a one-off contract clause.

## Curation with folders and optional notes

A team lead collects the documents that show good work on one topic, typically 10 to 200 files. Arkive reads PDF files, Word documents (`.docx`), PowerPoint decks (`.pptx`), Excel workbooks (`.xlsx`), saved web pages (`.html` or `.htm`), and Markdown (`.md`). From Word files it keeps the accepted text of tracked changes and the reviewers' comments. From PowerPoint files it keeps speaker notes, comments, and the numbers behind native charts. Any folder layout works: one folder per project, per client, per author, or one flat folder. Nobody has to tag, rename, or reformat a file.

The lead can add plain-English notes in files named `AGENTS.md` anywhere in the folder. A note can say how the collection is organised, which documents are missing, or how to use the examples. Arkive reads these notes as guidance and passes them to the assistant.

An engineer then runs one build command. The result is a single snapshot file (`*.arkive.db`): a portable, read-only copy of everything Arkive learned from the folder. The team can copy it, keep several snapshots for different topics, and rebuild it when the folder changes.

## Two everyday workflows

![Two workflows from one snapshot: pinpoint answers and grounded drafting](assets/mcp-workflows-concept.png)

**1. Pinpoint questions with citations.** Someone asks a factual question, such as the calibration certificate of an instrument or the minimum depth of a survey. The assistant searches the snapshot, narrows by project, client, or kind of deliverable when the question names one, and answers with the document, section, and page numbers. When a value sits in a chart, the assistant can open the rendered page image and read it. When the answer is spread over table rows in many documents, such as every alarm level in every installation report, the assistant reads the matching rows directly instead of opening each document.

The assistant sees the current version of each document unless it asks for more. A question about the current fee gets the issued figure. A question about how the fee changed between drafts makes the assistant ask for every version and compare them.

**2. Drafting from past work.** Someone asks for a new deliverable, such as a proposal or an installation report, and supplies the new project's facts. Arkive returns a drafting packet:

- up to three past deliverables of the same kind that best match the new project, with their outlines and key sections;
- the reviewer lessons that apply to that kind of deliverable;
- the curator notes;
- for each passage, how many separate projects use the same wording, so the assistant can tell standard company text from a clause one project needed.

The assistant takes every fact of the new work from the supplied inputs. It follows the structure and house wording of the past examples.

## Example: a structural monitoring archive

The tracked fictional collection `structural-monitoring-consultancy` holds 126 files from an instrumentation consultancy: proposals, installation reports, drafts, QA checklists, calibration sheets, and rate cards for five clients from 2023 to 2026.

One file is the 52-page Word installation and baseline report for the Askern Viaduct job. Arkive reads all 52 pages, splits the report into 86 sections, and labels it an installation and baseline report dated 14 October 2024 for the archived Askern Viaduct project. It also records 17 facts the report states about itself, such as its reference `MI-IR-2409` and the interrogator serial `PF4-1876`, each linked to the section that states it, and stores its 61 tables so their rows can be read directly.

The same build finds that a calibration sheet for one interrogator was copied into two job folders. Arkive groups the two copies as one document family and marks one copy as superseded, so search shows the other.

From an unlabelled QA register in another job folder, Arkive extracts a reviewer lesson for installation reports: apply temperature compensation with the strain-free reference gratings before reporting wavelength shifts as strain, and state this step in the method. When an assistant later drafts an installation report, that lesson arrives in its drafting packet.

## Further reading

- [How folders become searchable knowledge](how-it-works.md) describes the build and the serving steps.
- [Data boundaries, security, and cost](security-privacy-and-cost.md) says what leaves your environment and what a build costs.
- [Evidence, benchmarks, and known limits](results-and-limits.md) gives the test results and where Arkive does not help.
