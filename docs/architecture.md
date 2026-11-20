# Architecture

Arkive RAG turns a folder of reference documents into one SQLite file, the
*snapshot*, and serves one or more snapshots to AI agents over MCP. Building a
snapshot calls OpenAI models. Reading one needs no network, except that the
`search` tool embeds each query with the OpenAI embedding model.

The build does not assume how a collection is organised. It reads every
document, induces from the whole corpus the facets an agent can filter by,
each document's canonical deliverable type and feedback role, version families,
reviewer lessons, recurring text units and a cited glossary, and stores them in
the snapshot next to the sections, pages and chunks. An agent drafting a new
deliverable asks the server for a drafting packet: ranked prior exemplars with
their text and the factual provenance of that text.

## Modules

| Module | Owns |
| --- | --- |
| `models.py` | Domain types (readings, observation cards, facets, document roles, feedback lessons, recurring units, glossary entries, the build report, read results and the drafting packet), the provider output schemas, the shared text folding (`fold_text`, `fold_for_quote`, `heading_context`), and the `ArkiveError` hierarchy. |
| `llm.py` | The `StructuredRequest` protocol, the provider-neutral `Tool`, the `LanguageGateway` protocol, `OpenAIGateway` (every provider call: structured Responses calls, Decisions, embeddings, the evaluation harness's agent conversations, the retry policy), `BuildCache`, `UsageMeter` with the spend limit, and `.env` loading. |
| `compiler.py` | Reading (discovery, rendering, the transcription fidelity gate, section stitching, chunk pieces, observation cards with quote location) and binding (IDs, breadcrumbs, embeddings). `compile_corpus` composes reading, induction and binding into a `CompiledCorpus`. Its request types (`PageTranscription`, `ObserveSections`, `ConsolidateCard`) own their prompts, output schemas and reasoning efforts. |
| `induction.py` | Induction: corpus organisation in bounded batches, type canonicalisation, facet binding, version families, feedback lessons, the glossary and the topic guide; the validate-and-retry helper `generate_checked`; support checks. Its request types (`OrganiseCorpus`, `CanonicaliseTypes`, `ExtractFeedbackLessons`, `WriteGlossary`, `WriteGuide`) own their prompts, schemas and efforts. |
| `passages.py` | Text passages, with no I/O: what a unit of section Markdown is (`standing_units`), what a visual transcript is and its removal (`without_visual_transcripts`), word 4-gram `shingles` and `containment` with their thresholds, and which units of a corpus recur (`recurring_units`, over plain `TextDocument` records). Induction, the drafting packet and the evaluation harness all measure text with it. |
| `snapshot.py` | The SQLite schema, the atomic writer, and `TopicSnapshot`, the read-only query surface (facet filters, hybrid search, grep, sections, pages, glossary lookup, document pages and the drafting packet). |
| `server.py` | `TopicCatalog` (open snapshots keyed by topic), the ten MCP tools and the server instructions. |
| `cli.py` | The `arkive-rag` command. It is the only package module that reads `.env` and constructs `OpenAIGateway`; the evaluation harness in `evals/` does the same for its own runs. |

Dependencies point one way: `cli` → `server` → `snapshot`, and `cli` →
`compiler` → `induction` → `llm`, with `induction` → `passages` and `snapshot` →
`passages`; every module uses `models`, and `passages` imports nothing else of
the package. `snapshot` and `server` import neither `compiler`, `induction` nor
`llm`, and make no model call apart from the query embedder the CLI passes in. The compiler and induction receive a `LanguageGateway`, so
tests drive them with a deterministic fake and no network. Only `llm.py`
imports the OpenAI SDK and the OpenAI Agents SDK (`openai-agents`).

## Corpus layout

Every non-hidden file under the corpus directory must be `.pdf`, `.docx`,
`.pptx` or `.md`. Any other file stops the build with `UnsupportedFormatError`
before the first model call. Paths starting with `.` are skipped. A symlink
anywhere in the corpus stops the build with `UnsafePathError`, and each source
is read once without following symlinks; Word and PowerPoint files are
converted from a private copy of those bytes. The snapshot and the build cache
must be different files outside the corpus directory.

- Folders carry no fixed meaning. Induction decides whether the folder
  structure encodes a facet (projects, campaigns, authors) and binds that
  facet to the folders it names.
- Every `AGENTS.md` is a curator note. Its text is stored verbatim in the
  topic guide and given to induction as the curators' stated intent, which
  outranks the model's inference. It is also compiled as a document
  (`is_curator_note`, chunk kind `guidance`), so status questions such as
  "where is the reef case study?" match the note in both `search` and `grep`.
  Induction cannot cite a curator note: no glossary entry, feedback lesson,
  recurring unit or guide citation may point at it, and it has no deliverable
  type of its own beyond its card's. An agent
  answering a question from a note cites its `doc_id` and `section_id` like
  any other document's. A
  path-bound facet value still covers a note in its folder, so a project
  filter also finds that project's note.
- A document's ID is `document_id(rel_path)`: `doc_` and the first 10 hex
  digits of the SHA-256 of its NFC-normalised corpus-relative path, such as
  `doc_a1b2c3d4e5`. Sections are `doc_a1b2c3d4e5_sec_001` and chunks
  `doc_a1b2c3d4e5_chk_0001`. An ID depends on the path only, so adding,
  removing or editing another file leaves every other ID unchanged. Two paths
  with one ID (such as the NFC and NFD spellings of one name) stop the build
  with `ArkiveError`, naming both, before any file is read.
- Documents are ordered by path, compared component by component as discovery
  sorts them. That order, never the hex order of IDs, orders everything a
  prompt, a cache key or a read returns: `list_documents`, `grep`, search
  ties, organisation batches, version candidate pairs, glossary quotes,
  feedback lessons and recurring units.

## Build pipeline

`compile_corpus` runs three passes in order. Reading runs per document in a
thread pool (`--max-concurrency`, default 4): PDF, Word and PowerPoint files
start first, largest first, then Markdown, and readings are gathered in path
order. The first failed read cancels the reads not yet started. Induction is corpus-wide and one executor of the same size bounds all
its provider calls in flight; binding needs both.

### Reading

`read_document(source, data, gateway)` returns a `DocumentReading`: the file's
SHA-256, its sections and chunk pieces, its rendered pages and its observation
card. A reading holds no corpus IDs and no path, so the provider responses it
is built from stay valid in the build cache when files are added, removed or
moved. Each PDF, Word and PowerPoint file follows one path:

1. **Render.** Word and PowerPoint files are converted to PDF by LibreOffice
   (`soffice --headless`) with a private temporary profile, a minimal
   environment and a 120 s timeout. Each PDF page is rendered by pypdfium2 at
   scale 2.0 (`PDF_RENDER_SCALE`) to WebP (quality 85, `WEBP_QUALITY`), and its
   full Unicode native text layer is extracted. A page whose longest side would
   pass 4,096 px (`MAX_RENDER_SIDE_PX`) at that scale, such as an A0 sheet,
   renders at the scale that fits it, so libwebp's 16,383 px limit and a huge
   bitmap are never reached; a Pillow or pdfium failure is `ConversionError`
   naming the file. libpdfium is not thread-safe, so every pdfium call runs
   under one process-wide lock; WebP encoding runs outside it. Before any
   rendering, the file's metadata date (step 7) is read from its original
   bytes, so a file whose metadata cannot be read fails before LibreOffice,
   pdfium or any model call runs. With a build cache, the rendered pages and
   that metadata date are stored per source (see Provider calls) and a later
   build of the same bytes runs neither LibreOffice nor pdfium.
2. **Transcribe.** `gpt-6.1-sol` at low reasoning effort receives the page
   image, the native text, and the tail of the previous page's transcript. It
   returns blocks (`heading`, `paragraph`, `table`, `callout`) and visual
   transcripts for charts, maps and figures. The image decides what the page
   says, which repairs broken glyph encodings in the text layer; headings keep
   their visible section numbers. The previous page's tail lets the model set
   the continuation flags below. Page N receives the last 1,200 characters of
   page N-1's transcribed blocks, excluding visual transcripts. A changed page
   causes the following page to be re-transcribed only if that tail changes;
   propagation stops when a page's tail is unchanged.
3. **Check fidelity.** Running headers, footers and page numbers are removed
   before anchors are collected, and only from a page's edge lines (its first
   two and last two non-blank lines): a top line repeated as a top line, or a
   bottom line as a bottom line, exactly on three or more pages, or repeated
   apart from a leading or trailing number that advances with the page; on a
   two-page document, such a line repeated at the same edge of both pages when
   both pages have at least five non-blank lines, the line is at most 120
   characters and holds no part of a data filename (a fragment of one wrapped
   over lines included), and any such number is the page number itself; a labelled page number (`Page 3`, `3 of 12`) on an edge line; and a
   bare number on a page's first or last line. A body line that repeats an
   edge line stays. The anchors are GIS/data
   filenames (`.tif`, `.tiff`, `.shp`, `.nc`, `.gpkg`, `.csv`), signed
   formatted numbers such as `-330.9` or `59,494`, and integers of three or
   more digits. A filename that a narrow table cell wraps over up to six lines
   of native text also accepts its rejoined form; each occurrence is its own
   anchor, so two wrapped names that end in the same fragment each need their
   own transcription, and a number inside a wrapped name (`dry_` / `2026` /
   `final.tif`, also when the tail starts with `-` or `.`) is part of that
   filename rather than a number anchor. They are checked against the exact
   text the page contributes to its sections, where each filename occurrence
   satisfies one anchor at most: a filename transcribed once cannot stand in
   for two anchors that both accept it. A missing filename is named whole, as
   native text wraps it. `FIDELITY_ATTEMPTS` holds the two attempts. If
   a filename is missing from the first (low effort) transcript, or recall is
   below 0.95 (1.0 when the page has one or two anchors), the page is
   transcribed once more at high effort and the missing values are named in
   the request. After that retry, a missing filename or recall below 0.85
   raises `TranscriptionFidelityError` naming the file, page and missing
   values.
4. **Stitch sections.** `assemble_sections` keeps a heading stack (levels 1-4;
   a heading with level 0 counts as level 2) and starts a section at every
   heading; text before the first heading is `Overview`. A table block keeps
   empty cells and accepts GFM tables without outer pipes; caption or footnote
   lines around the table stay as paragraphs. A block flagged
   `continues_from_previous_page` joins the previous paragraph or table, and the
   joined part spans both pages. A continued table loses its repeated header
   rows, and when `merges_into_previous_row` is set, its first row is merged
   cell by cell into the row cut by the page break; each table row keeps the
   pages it came from. Visual transcripts become quoted
   `[Visual — Page N: label]` blocks in the section. A section over 32,000
   characters (`MAX_SECTION_CHARS`), such as a 20-page continued table or an
   unheaded transcript, becomes consecutive sections of that heading path
   within the limit, cut between blocks. The first section holds the heading
   line and the start of the body, unless fewer than 256 characters
   (`MIN_FIRST_SLICE_CHARS`) are left beside the heading; then the body starts
   in the next section, and a heading longer than a section is cut as text. A
   larger block is sliced: a table on row boundaries with its header and
   separator rows repeated, each slice spanning only its own rows' pages, and a
   row too long for a slice wrapped cell by cell into rows of the same columns
   (a header that would leave a row less than a quarter of a slice is cut like
   rows instead); a visual transcript at line breaks, every slice under its
   `[Visual ...]` title (a title longer than a quarter of a slice is quoted once
   under `[Visual — Page N]`); other text at line breaks or spaces, a fenced
   code block closed at the end of one slice and reopened at the start of the
   next with its delimiter and an info word of at most 32 characters. A line
   that starts with an inline code span (`` ```cli``` runs it ``) is text, not
   a fence. The visual transcripts of a continued table's pages are sliced the
   same way and follow the slice that holds their page.
5. **Chunk pieces.** `section_pieces` packs paragraphs and callouts into prose
   pieces of up to 1,800 characters, splitting an oversized paragraph at
   sentence ends. A table up to 2,400 characters is one `table` piece; a larger
   table is split on row boundaries with a normal packing target of 2,000
   characters, repeating the header and separator rows, each piece spanning
   the pages of its own rows; a header too wide to leave a row a quarter of
   the target is cut like rows instead of repeated. Each visual transcript is
   its own
   `visual` piece. No piece exceeds 6,000 characters: a final split wraps
   oversized content at line breaks or spaces and does not repeat a table
   header on later fragments. A piece has its own page span and no ID or breadcrumb yet.
6. **Observe.** The document's sections are packed in order into batches of
   at most 24,000 characters (a larger section is split at line breaks or
   spaces into pieces within that size, each keeping the section's ordinal), and
   `gpt-6.1-sol` at low effort reads each batch (`observe`). It returns the
   title, date, deliverable type (free text, such as "technical memorandum"
   or "campaign brief"), summary, the attributes the document states about
   itself (client, author, audience, channel and similar), the terms it uses
   (with kind, aliases and whether the quote defines the term), and craft
   observations (structure, convention, caveat, disclaimer, reviewer
   feedback, tone; no induction step reads craft observations now, and they
   stay in the observation schema so that cached observations stay valid).
   Every term and craft observation carries a verbatim
   quote, never a section number. Code locates each quote in the batch's
   sections with `fold_for_quote` (NFKC, case, Unicode dashes, whitespace,
   `*` and `_` emphasis, backticks, table pipes, block-quote markers and
   line-break hyphenation ignored; a `*` with spaces on both sides and an
   underscore inside a word stay text) and records the first containing
   section's ordinal. Unlocated quotes get one rerun that names them. After
   the rerun, observations whose quote is still unlocated are discarded and
   counted in the build report; a batch that quoted something and had no
   quote located raises `RejectedOutputError`, also when its rerun returns
   no observations at all. A placeholder date answer such as `none` or
   `not stated` counts as no printed date. A document with several batches
   has its terms, craft and attributes unioned in code, and one
   `consolidate_card` request (low effort) writes its single title, date,
   deliverable type and summary from the batch headers. The date it keeps is
   always one a batch reported as printed, as that batch wrote it: the
   reported date its answer names (the same text, the same text once folded,
   or another form of the same unambiguous date, such as `19th May 2025` for
   `19ᵗʰ May 2025`), else the first reported. It is none when no batch
   reported one, and a placeholder is not a reported date, so consolidation
   neither invents a date nor drops one. A curator note gets a card from its own text and
   no model call: the first line with text is its title and the first line
   that is not a heading its summary.
7. **Date.** `document_date` gives the reading its calendar date
   (`YYYY-MM-DD`, or `YYYY-MM`) and basis. A printed date (the card's
   `doc_date`) parses when it has one reading: ISO, a day (ordinal suffixes
   such as `19ᵗʰ` included), English month name and year in either order, a
   month name and year, a numeric `D/M/YYYY` where exactly one of the first
   two numbers is above 12 or both are equal, or a year-first `YYYY/M/D` or
   `YYYY.M.D` (years 1900 to 2100). Its basis is `printed`. A printed date with no
   single reading, such as `05/04/2026` or a range, gives no calendar date and
   never falls back to the file's metadata. Only a document that prints no
   date takes the date in its original bytes, with basis `metadata`: a PDF's
   `CreationDate`, else `ModDate`; a Word or PowerPoint file's
   `dcterms:modified`, else `dcterms:created`, in `docProps/core.xml`. Each
   value must be a whole PDF date string or W3CDTF value (`YYYY-MM` precision
   included) whose hour, minute, second and offset are in range, with an
   offset only after an hour; trailing text, a range or an impossible time
   counts as absent, and the next field is read. A `core.xml` over
   1,000,000 bytes, or one that cannot be read, is `ConversionError`. The
   LibreOffice conversion and file system times are never read. Such a
   document's `doc_date` is the metadata date's ISO form, and induction
   prompts show it as `<date> (file metadata)`.

Markdown files skip steps 1-3: `parse_markdown_blocks` splits them
deterministically, keeping a fenced code block as one literal paragraph, and
they have no page images for `view_page`. A document with no text raises
`ConversionError`.

### Induction

`induce(documents, notes, gateway)` receives the readings with their IDs and
runs its steps in order: organise, bind, then version families, feedback
lessons and the glossary on one executor, then the guide. Recurring text units
are computed in code. One executor with `--max-concurrency` workers runs the
organisation batches, the binding questions, the version judgements, the
lesson extractions, the glossary batches and the definition checks; each task
makes one provider call at a time, so the limit bounds them together. Results
are gathered in submission order, and the first failure of any task cancels
the tasks not yet started. Every generating step goes through
`generate_checked`: run, validate, rerun once with the rejections appended to
the request, then raise `RejectedOutputError`. Only three outcomes reduce the
output without failing, and the build report counts each: an observation or
feedback lesson whose quote is still unlocated after its rerun, a glossary
definition the support check does not confirm, and a feedback lesson it does
not confirm.

Every induction prompt, instructions and request text together, stays within
90,000 characters (`MAX_PROMPT_CHARS`, about 30k tokens) on a corpus of any
size, on the first run and on the rerun alike: the limits below are chosen so
that each request's parts add up to less. A rerun names the rejections one per
line, each cut to 4,000 characters, as many as fit 6,000 characters
(`MAX_REJECTION_FEEDBACK_CHARS`), with a count of the rest. Heading paths are
cut to 500 characters (`heading_context`), and curator notes are shown in path
order up to 24,000 characters (`MAX_CURATOR_NOTES_PROMPT_CHARS`); a note that
does not fit is cut with a marker, and `truncated_curator_notes` in the build
report counts the notes no prompt shows in full. The snapshot stores every
note verbatim. No step caps how many deliverable types a corpus has.

1. **Organise** (`organise_corpus`, medium effort, then `canonicalise_types`,
   medium effort). Every citable document becomes a one-line digest:
   `doc_id | path | format | date | observed type | title | summary |
   stated attributes`, each field cut (path 160, type 80, title 120, summary
   180 and attributes 120 characters). Digests are packed in path order into
   batches of at most 44,000 characters (`MAX_ORGANISE_BATCH_CHARS`), so a
   corpus of about 150 documents is one batch and a larger one several. Each
   batch request also holds the curator notes and the folders that hold a
   document of the batch or no citable document at all, shallowest first, up
   to 12,000 characters. It returns up to six facets and one entry for every
   document of the batch: its deliverable type, whether it is feedback, and,
   for a feedback document, the types of deliverable it reviews. A feedback
   document keeps its own deliverable type, so it can be drafted as that type.
   Validation rejects an entry for a document outside the batch, a document
   given twice or not at all, a blank type, feedback without reviewed types,
   and reviewed types on a document that is not feedback.

   A facet has a snake_case name, a description and a closed list of values,
   each with a description and aliases. A facet is *path-bound* when every
   value names the corpus folders it stands for, or *content-bound* when no
   value names a folder. Values and aliases are compared by
   `fold_facet_label`, the folded form the snapshot's alias table and the
   filter resolver use. Repetition within one value is normalised: values and
   aliases are stripped, blank aliases dropped and one alias kept per folded
   form. Validation (`check_facets`) rejects a facet that mixes the two
   bindings, an unknown folder, a duplicate facet name, a blank value, a value
   given twice, a folded form that names two values of one facet, and more
   than six facets. `merge_facets` then joins the batches' facets by name in
   code: a name proposed path-bound by some batches and content-bound by
   others keeps the binding more batches proposed (path-bound on a tie) and
   drops the other proposals; values merge by folded label, keeping the first
   spelling and description and uniting folders and aliases, with one spelling
   kept per folded alias (`NM` from one batch and `nm` from another are one
   alias), and an alias that folds like another value of the facet is
   dropped. When more than six names
   remain, those proposed by the most batches are kept. With one batch the
   result is that batch's facets. A path-bound value also answers to the
   paths of its folders, unless a path already names a value of the facet or
   belongs to several values.

   One canonicalisation request then lists every distinct type label the
   batches used, as a document type or a reviewed type, with its document
   count, up to three example titles and up to three titles of feedback
   documents reviewing it (so a label named only as a reviewed type shows what
   it stands for), while the list fits 60,000 characters
   (`MAX_TYPE_LABELS_CHARS`); a corpus whose labels alone do not
   fit is refused with `ArkiveError`. It returns canonical types, each with a
   name, a description and the labels it covers; every label must belong to
   exactly one type, and two names may not fold alike. A type held by a single
   document stays a type of its own. A corpus of one batch is canonicalised
   all the same, so one code path serves every size.
2. **Bind.** A path-bound value is assigned to every document under its
   folders, curator notes included. Content-bound values are judged per
   citable document: one `bind_facets` call to the Decisions API whose
   evidence is the document's path, title, date, observed type, summary,
   attributes and outline (title, summary and each attribute cut, the
   attributes at 4,000 characters, headings cut by `heading_context`, the
   outline cut at 12,000 characters, each with a count of what was left out),
   with one predicate per value, which quotes the facet's and the value's
   descriptions cut to 300 characters each. A probability of 0.5 or more assigns the value; a
   refused question assigns nothing and is counted. A document may hold
   several values of a facet or none. Values no document holds are then
   pruned, and so are facets with fewer than two populated values; the build
   report names both.
3. **Version families** (`confirm_versions`, `order_versions`). A version
   family groups drafts, revisions and format exports of one deliverable.
   Byte-identical citable documents (the same `sha256`) are one family
   without a question, however short they are; only the first of them in
   path order takes part in the candidate search. Code proposes candidate
   pairs of citable documents through two inverted
   indexes: one of word 4-grams sampled by checksum (one in four), where a
   pair qualifies when one document holds at least 0.35 of the other's
   sampled 4-grams, and one of title and file-name words, where two documents
   of the same canonical type qualify when they share at least 0.6 of those
   words (two words or more each). A 4-gram or word held by more than 50
   documents proposes nothing, and a feedback document is never paired with a
   document that is not feedback: a review quoting what it reviews is another
   artifact. Each document keeps its six best partners, so
   pairs grow linearly with the corpus. The Decisions API then judges each
   pair against compact cards of both documents (path, format, page count,
   date, canonical type, title, summary and first twelve headings), many
   pairs per request within one 48,000-character evidence window, with the
   predicate that the two are "drafts, revisions, or format exports of the
   exact same deliverable artifact for the exact same engagement and
   reporting period, not a follow-on phase, a study with a different sensor or
   method, a periodic installment for another period, or a summary deck
   paired with a full report". Confirmed pairs (0.5 or more) join into
   families; a family's ID is its first member in path order.

   In a family of one the document is `sole`. In a larger family, among the
   distinct members (the first of byte-identical copies in path order), dates
   decide when every member has a date of the same precision and one is
   strictly latest; otherwise the decision model is asked, for each ordered
   pair, whether one is "a later, more final version superseding" the other,
   and a member that supersedes every other (each forward answer at least 0.5,
   each reverse answer below it) is `latest`. Every other member, copies
   included, is `superseded` with `superseded_by` naming the latest member, in
   one step. Without such a member, or with more than twelve distinct members,
   every distinct member is `unordered` (the build does not guess an order),
   and a later byte-identical copy is `superseded` by its first copy.
4. **Feedback lessons** (`extract_feedback_lessons`, low effort). They run
   after version families: a superseded feedback document (an earlier draft
   or a copy of a review) gives no lessons. Each other feedback document's
   sections are packed into batches whose rendered sections, each labelled
   with its section ID and heading, take at most 24,000 characters; a larger
   section is cut at line breaks with its lines in order, and a piece that
   starts inside a table (with or without outer pipes) repeats the table's
   header and separator rows when they leave room for a row. Each batch is one
   structured request that returns transferable lessons: a statement and a
   verbatim quote of an accepted or resolved reviewer criticism, without
   project-specific values. Quotes are located with `fold_for_quote`; a lesson
   with a blank statement or an unlocated quote gets one rerun that names it,
   and is then discarded and counted in `discarded_unlocated_quotes`. As for
   observations, a batch that proposed lessons and located none is a broken
   reading, and so is its rerun when it proposes none. The support check
   (`check_feedback_lessons`) asks whether the section records a reviewer
   point the authors accepted or resolved, not one they rejected, left
   unchanged or ruled out of scope; it judges all lessons of one section
   together, one request per evidence window, and drops and counts the
   unsupported ones in `dropped_feedback_lessons`. A stored lesson names its
   source document, its section and the document's reviewed types.
5. **Glossary** (`write_glossary`, medium effort, structured requests).
   Code merges term observations by folded term and alias, keeps a term seen
   in two or more sections or quoted where a document defines it, and scans
   the other citable sections for further occurrences (up to six snippets,
   case-folded, each at most 360 characters from 120 before the match,
   however long the matched alias). A term keeps at most twelve quotes
   (`MAX_QUOTES_PER_TERM`), defining quotes first, each group in path order;
   each quote is shown cut to 600 characters, its kind to 80, and at most
   twelve aliases of up to 120 characters each in an alias list cut to 600,
   so a term found in hundreds of documents stays a bounded entry. A term's name, aliases and quotes are
   chosen in a total order (most frequent name, ties broken by folded then
   literal text), so the same observations in any order give the same
   request. Terms are sent
   in batches with their quotes and snippets, each tagged with its section
   ID. Each entry has a term, kind, aliases and
   one or more senses; a sense has a definition, or `null` when the cited
   sections use the term without establishing its meaning, and section
   citations. Conflicting meanings are separate senses. Validation requires
   one entry per term given, at least one sense and one citation per sense,
   citations that resolve to citable sections, and the entry's own term or
   one of the aliases it publishes in every cited section; an entry's aliases
   are stripped, one per folded form and sorted. Then `check_definitions`
   asks the Decisions API whether the cited sections establish each
   definition; below 0.5 or refused, the definition becomes `null` and is
   counted. `unsupported_definitions` counts proposed definitions demoted by
   this check, not senses the generator already left null.
6. **Recurring units** (`recurring_units` in `passages.py`, no provider
   call), over the deliverables only: a feedback document quoting a flawed
   paragraph does not make it house text. A *unit* of section Markdown (`standing_units`) is a paragraph, one
   item of a list (a list marker, `LIST_MARKER`, starts an item: `-`, `*`,
   `+`, a number followed by `.` or `)`, or one of the bullet glyphs PDF text
   layers carry, U+2022, U+25E6, U+25AA, U+2023, U+2043 and U+F0B7, the
   Private Use Area bullet of Word's Symbol font; an unmarked line right after
   it continues it), or a table's header row with its separator row. A table
   starts at a line opening with `|`, or at a line holding `|` right above a
   separator row (a table without outer pipes); such a starting line is never
   a list item. A table whose first line opens with `|` runs through the
   following lines that hold `|` and are not list items; a table without outer
   pipes runs through every following line that holds `|`. A block quote that
   is not a visual transcript is a paragraph. Headings, visual transcripts
   (`> **[Visual` blocks), fenced blocks and table body rows are not units. A
   prose unit is folded without its leading list marker (`fold_unit`), so
   items numbered `1.` and `2.` in two documents compare alike. A unit is
   eligible with at least 8 words (`STANDING_MIN_WORDS`, counted after
   `fold_unit`), or as a table header row of at least two columns
   (`STANDING_MIN_TABLE_COLUMNS`). A section *holds* a unit when it holds at
   least 0.8 (`STANDING_RECURRENCE_CONTAINMENT`) of its word 4-grams after
   folding, counting the 4-grams of the section's whole text without visual
   transcripts together with those of its own units (folding the whole text
   joins a line-end hyphen with the next unit's first word, so neither form
   alone holds every 4-gram the other does), or, for a table header, when it
   has a table header of the same folded row. A visual transcript describes a
   figure, so it never makes a section hold prose. Where documents sit plays no part. Units that say the same
   thing (the same folded text, or containment of at least 0.8 both ways) are
   linked, and a cluster is everything linked directly or through other units,
   so a chain of light edits across several documents is one cluster even when
   its ends differ more than the threshold allows. A cluster's sections are
   every section of its units and every section holding one of them, its own
   documents' other sections included, each once; it is kept when they lie in
   two documents or more, and it is represented by its first unit in canonical
   order (document path, section ordinal, position). An
   inverted index of 4-grams finds the sections that could hold a unit by
   looking up only its rarest 4-grams, enough of them that any section
   holding 0.8 of the unit holds one; so the search stays near linear and
   still finds every holder. The snapshot stores each cluster's text and
   every section that holds it.
7. **Guide** (`write_guide`, medium effort). A cited Markdown overview from
   the bounded curator notes, the bound facets, each deliverable type with its
   document and feedback counts, the glossary term names (up to 4,000
   characters) and a catalog with one entry per citable document:
   `[doc_id:first_section_id] title (date, type): summary`, followed by up to
   six of its top-level sections as citation tokens. The catalog fits 60,000
   characters (`MAX_GUIDE_CATALOG_CHARS`) and the whole request text 72,000
   (`MAX_GUIDE_TEXT_CHARS`), rerun feedback included: as the corpus grows, entries drop headings, then
   shorten their summaries and titles, and past that the catalog lists
   entries in path order and counts the rest. Every bracketed `[doc_...]`
   token must have the exact form `[doc_a1b2c3d4e5:doc_a1b2c3d4e5_sec_001]`,
   the section belonging to the document named before it, and resolve to a
   section of a citable document, and at least one valid citation is required
   when such a section exists. A guide of more than 16,000 characters
   (`MAX_GUIDE_MARKDOWN_CHARS`) or 24,000 bytes as a UTF-8 JSON string
   (`MAX_GUIDE_MARKDOWN_BYTES`, so a control character counts as its
   six-byte escape) is rejected too, so `get_topic_guide` can return it
   whole. The rerun's citation rejection, at most 4,000 characters,
   opens with the repair instruction, then, when the guide named an unknown
   document or none, fifteen sample catalog tokens, then the valid tokens of
   the documents it tried to cite, then the invalid tokens it wrote (each cut
   to 80 characters); a part that does not fit is left out, and never every
   section of the corpus is listed. The request shows the topic cut to 200
   characters. A corpus with no citable section (curator notes only) gets a guide
   without citation tokens.

A support check judges the complete cited sections. Evidence longer than
48,000 characters is sent in windows (a longer section in parts that overlap
by 2,000 characters), and each predicate takes its highest probability over
the windows; it counts as refused only when every window refused it.

### Binding

`bind_document` gives each section and chunk its ID and each chunk the
breadcrumb `[<facet>: v1, v2 | ... | Doc: <title> (pp. X-Y) | Section: <heading
path>]`, with facets in definition order, values sorted and facets the
document holds no value of left out. Every chunk is then embedded as
`breadcrumb + "\n" + content` with `text-embedding-3-large` at 1,536
dimensions.

`bind_document` also stores each document's role, which its caller passes:
the induced canonical deliverable type, feedback role and version family of a
citable document, or `curator_note_role` for a curator note, which keeps its
card's type, is no feedback and is a family of its own.

`build` prints the usage line and the build report. The build report has
`discarded_unlocated_quotes`, `unsupported_definitions`,
`dropped_feedback_lessons`, `judge_refusals` (per Decisions operation),
`pruned_facet_values`, `pruned_facets` and `truncated_curator_notes`.

## Provider calls

`OpenAIGateway` is the only provider caller for builds and the evaluation
harness, and the single owner of retry policy; the SDK clients run with
`max_retries=0`. Domain code describes each call with a request object that
owns its prompt, output schema and reasoning effort; the gateway owns
transport. Model IDs are constants in `llm.py`: `DEFAULT_MODEL`
(`gpt-6.1-sol`), `DECISION_MODEL` (`gpt-6-luna`) and `text-embedding-3-large`
at 1,536 dimensions. The CLI has no model options.

- `run` sends one structured `responses.parse` call with the request's
  `reasoning.effort`.
- `judge` asks the Decisions API (`POST /v1/decisions`) one `predicate`
  question per statement over shared evidence and returns one probability
  per predicate in order, `None` where the model refused that question. No
  predicates means no request. Predicates go in order in requests of at most
  400 questions whose evidence and questions together hold at most 90,000
  characters (`MAX_JUDGE_REQUEST_CHARS`); evidence that leaves no room for a
  question is a `ValueError` before any request.
- `converse` is the free-text OpenAI Agents SDK conversation the evaluation
  harness uses (see Evaluation). It runs on the caller's event loop with its
  own async client and is never cached. Each model turn is one provider
  operation under the retry policy below; the Agents SDK's own retries stay
  off. Invalid tool input (`NotFoundError`, `InvalidFilterError`, `ValueError`
  or `KeyError`) and a call to a tool name the agent does not have go back to
  the model as text. Not finishing within `max_turns`, a response that is not
  `completed` or a provider error ends the conversation with that status; the
  spend limit and any other tool exception propagate. Tools use non-strict
  JSON schemas.
  Its `Conversation.tool_calls` lists every call to one of its tools in
  invocation order as a `ToolCall`: the tool name, the argument text as the
  model sent it (cut at 2,000 characters with a marker giving the full
  length), the length and first 200 characters of the text returned to the
  model, and the number of images. A call is recorded before its tool runs,
  so a call whose tool raised is listed with no result. A call to a tool name
  the conversation does not have is answered by the Agents SDK and is not
  listed.
- Hosted Agents SDK tracing is off unless the caller passes `trace=True`
  (`--trace`). The trace export key is passed with each run, not set on the
  SDK's process-wide exporter. `trace=True` while
  `OPENAI_AGENTS_DISABLE_TRACING` disables tracing fails when the gateway is
  constructed.
- Rate-limit, connection, timeout and server errors are retried, up to five
  attempts per operation, waiting 2, 5, 12 and 25 s or the provider's
  `Retry-After` (a value that is not a finite number is ignored). All attempts
  share one deadline (600 s for builds, 60 s for query embedding in `search`
  and `serve`), and each request's timeout (180 s, or 30 s for queries) is
  capped by the time left. A wait that would pass the deadline raises
  `ProviderError` at once, and no attempt starts after it. An agent turn's
  whole attempt, not only each socket read, ends at the deadline. The spend
  limit is checked before every attempt.
- A structured response that stops at `max_output_tokens` (starting at
  16,384) is retried once with double the budget. Agent turns allow 16,384
  output tokens.
- A refusal in a structured response raises `ProviderError`; a
  Decisions refusal returns `None` for that question. Other failures, a body
  that is not JSON, missing or malformed usage, or output that fails schema
  validation raise `ProviderError`. A `judge` probability that is not a
  finite number within 0..1 is malformed too.
- Embeddings are requested in batches of up to 128 inputs and 100,000
  characters. Response rows must carry each index exactly once and are
  reordered by it; every row is L2-normalised. A wrong shape or a zero or
  non-finite vector raises `ProviderError`.

`--cache <file>` enables `BuildCache`, a SQLite table of validated responses
keyed by SHA-256 of the call's inputs:

- `run`: operation, model, effort, instructions, text, output schema and the
  image hash;
- `judge`: operation, decision model, evidence and predicates;
- embeddings: model, dimensions and the batch's texts.

A rebuild of an unchanged corpus with the same cache makes no provider
requests: every collection that reaches a prompt or a cache key is in a
canonical sorted order, and concurrent results are gathered in submission
order. Adding a file requests the reading of that file only; induction and
the embeddings are requested again because the corpus changed. A request with
identical inputs can use a cached response, including in a subset build; its
inputs cover everything the request can see. The
first valid entry stored under a key is canonical: when two build
threads miss the same key at once, the second writer returns the first
writer's value. An entry that no longer decodes or validates is a miss and is
replaced. A new cache file is created with mode `0600`.

The same file is the compiler's render cache (`RenderCache`): the table
`page_render_cache` holds each rendered source's pages (native text and WebP
bytes, numbered 1..N), and `page_render_meta` holds one row per source with its
page count and its file's metadata date, written in the same transaction. Both
are keyed by SHA-256 of `RENDER_CACHE_REVISION`, the source format, the source
bytes' SHA-256, the render scale, the WebP quality and the side limit. An
entry is a miss, and is replaced, when its metadata row is missing, its pages
are not exactly 1..page count, or a value has the wrong type, an image is not
a WebP image that decodes in full or the date is not a calendar date; the first complete entry is
canonical. A cache hit opens neither LibreOffice nor pdfium, also for the
metadata date. `build` passes the cache to `compile_corpus`, and the
evaluation harness passes it to the holdout build. The render settings are
recorded in `build_settings` (`rendering`). Closing the cache closes its
connection; SQLite checkpoints the write-ahead log when the last connection
closes.

Every provider response adds its operation, model, input, cached-input and
output tokens to a `UsageMeter`, which estimates USD from Standard-tier list
prices in `llm.PRICES` (`gpt-6.1-sol`, `gpt-6-luna` on Decisions, and
`text-embedding-3-large`; the long-context surcharge is not modelled).
`build` prints the totals. With `--max-cost`, no provider call starts once the
estimate has reached the limit: it raises `BudgetExceededError`, a
`ProviderError`, and `build` exits with an error after printing the totals.
Calls already in flight can take the total past the limit. If the last request
takes the estimate past the limit, the build finishes normally. A spend limit with a
model that has no price is refused when the gateway is constructed.

## Snapshot file

`write_snapshot` writes a uniquely named sibling temporary file (mode
`0600`), inserts everything in one transaction, runs `PRAGMA
foreign_key_check` and `PRAGMA integrity_check`, fsyncs, and renames it over
the output, then fsyncs the directory. A replaced snapshot keeps its file
mode. A failed build leaves any previous snapshot untouched, and SQLite or
file-system failures raise `SnapshotIntegrityError`. The text tables are
written first and the page images last, so sections and chunks lie together
in the file rather than between page BLOBs.

Tables (schema version `6.0`):

| Table | Contents |
| --- | --- |
| `corpus_meta` | `schema_version`, `topic`, `title`, `created_at`, `corpus_fingerprint`, `arkive_version`, `embedding_model`, `embedding_dimensions`, `guide_markdown`, `build_settings` (JSON: the model IDs and the effort of each operation) and `build_report` (JSON, as printed by `build`). |
| `curator_notes` | Verbatim `AGENTS.md` files with their path. |
| `documents` | One row per source file: `doc_id`, `ordinal` (path order), `rel_path`, `sha256`, `format`, `title`, `doc_date` (as printed, or the metadata date), `doc_date_iso` and `doc_date_basis` (`printed` or `metadata`; both null or both set, and when set `doc_date` agrees: a metadata date's `doc_date` is `doc_date_iso`, and a printed one parses to it), `deliverable_type` (canonical; a curator note's card type), `summary`, `page_count`, `is_curator_note`, `is_feedback`, `feedback_for_json`, `family_id`, `version_status` (`sole`, `latest`, `superseded` or `unordered`) and `superseded_by` (set exactly when superseded). |
| `deliverable_types` | Each canonical type with its order and description. |
| `facets` | Name, order, description and binding (`path` or `content`). |
| `facet_values` | Each facet's values with their order, description and folders (JSON). |
| `facet_aliases` | Every folded form of a value, the canonical value included, keyed by facet and folded alias, so a form that would name two values cannot be stored. |
| `document_facets` | Document-to-value assignments, with foreign keys to `documents` and `facet_values`. |
| `glossary` | One validated `GlossaryEntry` JSON document per row. |
| `feedback_lessons` | Each lesson's statement, source document, section and reviewed types (JSON). |
| `unit_clusters`, `cluster_sections` | Each recurring unit's kind and text, and every section of a deliverable that holds it, in order, with that section's sample of a prose unit. |
| `sections`, `pages`, `chunks` | Section Markdown with heading path and pages; native text, visual summary and WebP per page; chunk content, breadcrumb and float32 embedding. |
| `chunks_fts` | FTS5 over NFKC-normalised chunk content and breadcrumb (`porter unicode61 remove_diacritics 2`). |

The corpus fingerprint is the SHA-256 hex digest of the lines `rel_path NUL
sha256`, one per document sorted by `rel_path` and joined by newlines. It
depends only on source paths and bytes, not on models, prompts or build time,
so it is the content address of the corpus a snapshot was built from
(`CompiledCorpus.corpus_fingerprint`).

`TopicSnapshot.open` uses a read-only SQLite URI with the path
percent-encoded, so names containing `%`, `?` or `#` open the right file. It
refuses any schema version other than `6.0`, naming the file's version and the
one it reads; an older snapshot (`5.0`, whose recurring units and version
families follow earlier rules, or `3.0` and `4.0`) is rebuilt from its corpus. Every
`corpus_meta` key and every table is required, and `open` runs `PRAGMA
foreign_key_check`, so a dangling facet assignment or alias is refused. Each
facet value must have its canonical alias row. A snapshot is a build output
that may be truncated or damaged: any decoding or cross-table failure while
it opens, and any SQLite error from a later read, is `SnapshotIntegrityError`
naming the file. `open` recomputes the fingerprint from the
`documents` table and raises `SnapshotIntegrityError` when the recorded one
differs. It validates every glossary row through its model and checks that
every cited section belongs to a citable document of the snapshot; that no two
deliverable types have the same `fold_facet_label` form (`prepare_draft`
resolves a type by that form); that every citable document's type and reviewed
types are listed and that it is feedback exactly when it names reviewed types;
that each version family's ID is its first member in path order, a family of
one and only a family of one is `sole`, at most one member is `latest`, and a
superseded member names another member that is not superseded itself, the
latest member when the family has one (so supersession takes one step and has
no cycle); that each feedback lesson cites a section of its feedback source and
names listed types; and that every recurring unit is held by citable sections
of two documents, none of them feedback. It loads the
facets, aliases, document-to-facets map, sections, families and recurring units
once, with chunk metadata and the embedding matrix. `list_topics` and
`arkive-rag inspect` report the fingerprint and version; `inspect` also prints
the facets, glossary, deliverable types, version families of two or more
documents, feedback lessons, the number of recurring units, build settings,
build report and each document's `sha256` and section count.

## Private corpus and eval versioning

Code, the eval pack schema and the eval runners live in this repository, which
holds no client documents, client facts or personal data. Corpora, eval packs
with corpus facts, and snapshots live in a private store outside it (a private
Git LFS or DVC repository, or a private bucket):

```text
private-store/
  bathymetry/corpus/...                         source documents
  bathymetry/bathymetry.eval.json               eval pack, pinned to a corpus_fingerprint
  bathymetry/bathymetry-1a2b3c4d5e6f.arkive.db  snapshot, named by corpus_fingerprint[:12]
```

A snapshot file is immutable once published: a changed corpus gets a new
fingerprint and so a new file name. To check that a snapshot matches a corpus
folder, compare `inspect`'s `corpus_fingerprint` and per-document `sha256`
values with the folder's files. An eval pack that sets `corpus_fingerprint`
refuses any snapshot built from a different corpus. `.gitignore` excludes
`*.arkive.db`, `*.arkive.db.*` and `*.eval.local.json` so that snapshots and
local packs are not staged by accident.

## Retrieval

- **Filters.** `list_documents`, `search`, `rank_channels`, `grep` and
  `prepare_draft` take `filters`, a mapping from facet name to one value. A
  value matches a canonical facet value or one of its aliases after
  `fold_text`. Several facets combine with AND; an empty mapping means no
  filter. `TopicSnapshot.resolve_filters` turns the mapping into the set of
  matching document IDs once, and every read path uses that set. An unknown
  facet or value raises `InvalidFilterError` naming the valid facets or values.
- **Ranking.** Two channels are fused with Reciprocal Rank Fusion (each
  ranking adds `1 / (60 + rank)` to a chunk's score), each taking its top 300
  chunks (`CHANNEL_DEPTH`):
  - BM25 over `chunks_fts`, weighting content 10 and breadcrumb 2.
    `build_fts5_query` quotes every letter-and-digit token (`_` separates
    tokens, as in the `unicode61` tokenizer) and adds adjacent-token phrases,
    so user text such as `Sentinel-2`, `C12-SUM(C13:C15)` or `NOT` can never be
    an FTS5 syntax error. Filler words (`what`, `the`, `of` and similar) are
    dropped when at least two other tokens remain, and a phrase of two filler
    words is never added.
  - Exact cosine similarity between the query vector and every chunk that
    passes the filters.

  Ties break on cosine similarity, then document path order and chunk ordinal.
  `rank_channels` returns the top `top_k` of each channel and of the fusion
  with no diversity rule or size limit; the ablation layer reads it.
- **One tool result.** Every read result that can grow with the corpus fits
  one tool result: at most 45,000 UTF-8 bytes (`RESPONSE_MAX_BYTES`) and
  1,600 lines (`RESPONSE_MAX_LINES`) of `response_text`, the one compact JSON
  text block the MCP server sends, below the 51,200 bytes and 2,000 lines at
  which a client such as OpenCode cuts a tool result. Every result is measured
  as served (`fits_response`) before it is returned; the drafting packet
  prices each addition by its exact bytes and measures the finished packet.
- **Active pool.** A superseded document is left out wherever the document
  that supersedes it passes the filters (`_active_pool`), so a copy or draft
  stays reachable when filters leave out the version that replaces it.
- **search** returns the fused ranking diversified by version family. Only
  the active pool is ranked, so a long superseded draft cannot take the
  channels' 300 places from its latest version. At most three chunks of one family
  (`SEARCH_FAMILY_CAP`) are taken first; when that leaves fewer than `top_k`
  hits, as when the filters name one document, the held-back chunks fill the
  list. Hits keep fused-rank order. Each reports its fused score, both ranks,
  its document's facets and `more_from_document`, the further ranked chunks of
  its document the list does not show, counted after the list is cut to one
  tool result. A blank query or an invalid filter or `top_k` raises
  `InvalidFilterError` before the query is embedded.
- **grep** is a literal, case-insensitive line search over section Markdown.
  Pattern and text are compared by `fold_text` (NFKC, Unicode dashes as `-`,
  case-folded), so `2016-2017` finds `2016–2017` and `km2` finds `km²`. A
  snippet is at most 400 characters of the line, centred on the first match.
  `max_matches` is 1 to 50 (`MAX_GREP_MATCHES`); the matches that fit one
  tool result are returned.
- **Glossary lookup.** Without a query, `lookup_glossary` lists terms with
  their kind, aliases and whether any sense has a definition. With a query, it
  returns the full entries whose folded term or alias contains the folded
  query; no match is an empty list, and a query of more than 1,000 characters
  is refused. Either list is one page from `offset` that fits one tool
  result, with `total` and `next_offset`; an entry too large for a page of
  its own keeps as many citations per sense as fit, counting the rest in
  `omitted_citations`.
- **Document pages.** `TopicSnapshot.list_documents` returns every document
  passing the filters, with its source hash, for Python callers such as
  `inspect` and the evaluation harness. `list_documents_page` returns one page
  of them without hashes: `limit` from 1 to 100 (default 50) from `offset`,
  fewer when more would not fit one tool result, with `total` and
  `next_offset`. A summary in a listing is cut to 240 characters
  (`LIST_SUMMARY_CHARS`).
- **Outlines and sections.** `outline(doc_id, offset)` returns the sections
  from `offset` that fit one tool result, each heading path cut by
  `heading_context`, with `total_sections` and `next_offset`.
  `read_section(section_id, offset)` returns the section's Markdown from
  character `offset`: the rest when it fits, else up to the last line break
  that fits, with `total_chars` and `next_offset`. A section holds at most
  32,000 characters, which in a script of two- or three-byte characters is
  more than one tool result. Python callers, such as `inspect` and the
  evaluation harness, read a whole outline with `full_outline(doc_id)` and a
  whole section with `section_markdown(section_id)`, never through these
  pages.
- **Display fields.** Every result that shows a document title or heading
  path beside other data (search hits, grep matches, document pages,
  outlines, section parts, exemplars, candidates and supplemental sections)
  cuts the title to 500 characters (`TITLE_CHARS`) and the heading path by
  `heading_context`, so one long heading cannot push a match out of its
  result. A result whose smallest form still does not fit is a
  `SnapshotIntegrityError`, never an empty list.
- **Topic summaries.** `summary()` lists each facet's values and the
  deliverable types up to 50 each (`SUMMARY_MAX_NAMES`), fewer when more
  would not fit, counting the rest in `omitted_values` and
  `omitted_deliverable_types`.
- **Topic guide.** `guide()` keeps the guide text and the facet names and
  descriptions (cut to 300 characters, `GUIDE_DESCRIPTION_CHARS`, as are the
  deliverable type descriptions), then as many deliverable types as fit, then
  curator notes in path order while they fit (a note that does not fit is
  left out, and its sections stay readable with `read_section`), then as many
  values per facet as fit, then glossary terms in order;
  `omitted_deliverable_types`, `omitted_curator_notes`,
  `omitted_facet_values` and `omitted_glossary_terms` count what was left
  out. The build keeps the guide text within 24,000 bytes, so a valid
  snapshot's guide always fits.

### Drafting packet

`TopicSnapshot.prepare_draft(project_inputs, embed_query, deliverable_type,
filters)` returns a `DraftPacket` that fits one tool result. Its short fields
come first: the guidance, lessons, curator notes and `recurring_samples`, then
the exemplars' Markdown. It makes no model call besides embedding the start
of the inputs (8,000 characters) when an embedder is given, and works without
one. Blank inputs or an invalid filter raise `InvalidFilterError` and an
unknown type raises `NotFoundError` listing the topic's types with their
document counts, all before any embedding; a type matches after
`fold_facet_label`, and a blank type is no type.

- **Pool.** Every citable document of the active pool that is not feedback; a
  feedback document of the requested type stays, so a review can be drafted
  as a review.
- **Relevance.** The inputs are ranked with the hybrid ranking above, or BM25
  alone without an embedder. The lexical query ORs the 48 distinct input words
  held by the fewest chunks, since each word adds to a BM25 query's cost;
  filler and generic drafting words (`please`, `write`, `whether`, `report`,
  `project` and similar) are left out. With a requested type, its documents
  are also ranked among themselves, so hundreds of documents of another type
  cannot crowd them out of the channels; their relevance and their matched
  sections come from that ranking. A
  document's relevance sums `1 / (60 + rank)` over its best three chunks
  (`RELEVANCE_CHUNKS`), so a long document does not outrank a closer short one
  by its length alone, plus up to a quarter of a first-ranked chunk's score
  for input words in its title, type and summary.
- **Exemplars.** Up to three. With a requested type, its documents fill the
  slots first, best first; a free slot takes the best relevant document
  (relevance above zero) of another type when at most one exemplar is of the
  requested type, or when that document reaches 0.6 of the best exemplar's
  relevance (`CROSS_TYPE_RELEVANCE`). Without a type, the three best fill them. An
  unordered version family fills one slot, and the exemplar lists its other
  members in `family_doc_ids` (at most 24, the rest counted in
  `omitted_family_members`); its summary is cut to 2,000 characters. The next
  eight documents of other families are
  `other_candidates`, with type, date, page count, summary and relevance.
- **Sections.** Every exemplar carries its complete outline. Each entry has
  its heading path, level, pages, length, `visual_count` and
  `family_recurrence`; an included entry has its Markdown without visual
  transcripts. `family_recurrence` reports the section's units that another
  version family also holds (`recurring_unit_count`), the packet's exemplars
  holding them, the most families holding any one of them out of the topic's
  version families of deliverables (`corpus_family_count` of
  `corpus_family_total`) and the newest date among the holders. A superseded
  draft is not counted as a holder, so a clause its latest version dropped
  does not count as house text. `recurring_samples` hold one entry per
  recurring prose unit (two or more families; table header rows are not
  house text) that an included section is the first to show, at most twelve:
  that section's sample, which the build stores with each holder. A sample is
  at most 16 words folded by `fold_unit` (so a line-end hyphen is joined and a
  list marker dropped): the first words of the section's own unit of the
  cluster, or, when the section holds the unit inside other text such as one
  sentence of a longer paragraph, the longest run of words whose every 4-gram
  is the unit's, so it stops where the section's own words begin. An agent, and the OpenCode check, can tell house text from a
  project's own paragraph in a section that mixes both, in the exemplar's
  own words. The envelope fits first, without outlines, notes or lessons: the
  later exemplars move to the candidates, then candidates are dropped from
  the end, and at most eight candidates are kept, moved exemplars first. Then
  curator notes and lessons come in, each up to 6,000 bytes. Then, while the
  exemplars' complete outlines do not fit, the last exemplar moves to the
  candidates; a first exemplar left alone keeps as many outline entries as
  fit, counting the entries cut in `omitted_outline_sections`, so every
  exemplar after the first has its complete outline. Section text is then
  packed whole, never cut, each addition kept only while the packet fits:
  the first exemplar's input-matched sections (at most six), sections with
  recurring units, and first and last two sections; up to two supplemental
  sections; the rest of the first exemplar; then the other exemplars'
  input-matched sections, recurring sections that show a unit no included
  section shows yet, and first and last two sections; then the rest of them.
  A section that does not fit is skipped, its entry kept, and counted in
  `omitted_sections`; later, smaller ones may still fit. The packet is
  serialized a handful of times, not once per section: the envelope and the
  outlines are measured as served, each later addition is priced by the
  bytes its JSON value adds within 512 bytes below the bound
  (`PACKET_MARGIN_BYTES`, for the counters' digits), and the finished packet
  is measured once, the last additions taken back if it still does not
  fit.
- **Supplemental sections.** Sections of deliverables of the active pool,
  outside the exemplars' families and of the requested type, that hold the
  most rare input words: words of four or more characters, not numbers, held
  by at most two version families of deliverables (reviews, curator notes and
  superseded drafts do not count). A named team
  member's biography in a seventh proposal is the kind of section this finds.
- **Guidance.** `lessons` are the feedback lessons reviewing the requested
  type, or the first exemplar's type when none is requested, from every
  feedback document of the topic whatever the filters (they are house rules
  for the type), up to 6,000 bytes; `curator_notes` the topic's notes up to
  6,000 bytes; each with a count of what was left out.
  `drafting_guidance` (`DRAFTING_GUIDANCE`) tells the agent to take every fact
  of the new work only from the user's inputs, reuse and adapt method text
  where the inputs use the same method, weigh dates, families and recurrence
  counts as provenance (asking the user when an institutional policy is
  genuinely uncertain in an interactive session), follow the closest
  exemplar's structure, apply the lessons and notes, read omitted sections
  with `read_section`, and write a standalone deliverable without Arkive IDs.

## MCP server

`arkive-rag serve a.arkive.db b.arkive.db` opens every snapshot into a
`TopicCatalog` and refuses two snapshots with the same topic. Every structured
tool result is one text block of `response_text` (compact JSON) beside its
structured content, so what the snapshot measured is what the client reads. It
has ten tools:

- `list_topics(offset=0)`: one page of the served topics (`TopicPage`), each
  with its title cut to 500 characters, its facets and allowed values and its
  deliverable types, up to 50 of each and fewer when a summary would not fit
  a page of its own. Only the topics on the page are summarised.
- `get_topic_guide(topic)`: the cited guide, curator notes, facet
  definitions, glossary terms and each deliverable type with its description,
  document count and feedback count.
- `prepare_draft(topic, project_inputs, deliverable_type=None, filters=None)`:
  the drafting packet above.
- `list_documents(topic, filters=None, limit=50, offset=0)`: one document
  page.
- `get_document_outline(topic, doc_id, offset=0)` and
  `read_section(topic, section_id, offset=0)`: one page of an outline, one part
  of a section.
- `search`, `grep`, `view_page` and `lookup_glossary(topic, query=None,
  offset=0)`.

`view_page` returns the caption as text content and the page as `image/webp`
image content. The caption (`page_caption`) holds the title cut to 500
characters, the visual summary and the native text within the byte bound and
1,500 lines; the visual summary takes at most half of the room, and a part
that does not fit is cut with a line counting the characters and lines left
out. Domain errors become tool errors whose message names the valid topics,
facets, values, deliverable types or pages: at most 40 of them with a count
of the rest, and at most 200 characters of the caller's input.

The server's instructions are short. They name `list_topics` and facet
filters; for questions, the read tools, citing `doc_id`, `section_id` and
pages, and saying that the corpus does not establish a fact when `search` and
`grep` find no evidence for it; and for drafting, `prepare_draft` with the
user's full inputs, following the packet's guidance and reading omitted
sections with `read_section`.

Over `streamable-http` the server has no authentication, so `serve` refuses a
host other than `127.0.0.1`, `localhost` or `::1` unless `--allow-nonlocal` is
given. The server reports the package version to clients.

`search` needs `OPENAI_API_KEY` in the server environment and all served
snapshots must share one embedding model and dimension. Without a key the
server still starts; `search` returns an error that points to the offline
tools, and `prepare_draft` ranks with BM25 alone.

The CLI's `search` and `grep` take repeated `--filter FACET=VALUE`; naming a
facet twice is a usage error, because each facet takes one value.

## Evaluation

`evals/` is a harness outside the `arkive_rag` package; it is not in the wheel.
`python -m evals.run --pack <file> --snapshot <db> --out <dir> [--corpus <dir>]`
runs an eval pack and writes `results.json`, `report.md` and, for layer 3,
one `holdouts/<holdout_id>/` folder per holdout to `--out`. `--out` is required and should be a private
directory outside the repository; the runner creates it with mode 0700,
refusing a symlink in its place, and writes 0600 files. It publishes every
output file by writing a temporary file in the same directory and renaming it
over the destination (`write_private_text`), so a failed write never truncates
an earlier result and a symlink at the destination is replaced, never written
through. Every layer calls OpenAI, so it needs `OPENAI_API_KEY`.
Before creating any output, gateway or build cache or calling any model the runner checks, for every
layer it will run (so `--layer all` checks layer 3 before layer 1 calls
anything), that the pack is a valid `arkive-eval-pack/3` file (a pack of
another format is refused with a message naming format 3; no older format is
read), that it defines the requested layers, that `--holdout` names one of
its holdouts and comes with layer 3, that `--drafts` (a positive number of
draft rounds, default 1) comes with layer 3, that layers 2 and 3 have `--corpus`,
that `--out` and `--cache` overlap none of the pack, snapshot and corpus
paths nor each other, that an existing `--out` is a directory, not a symlink,
that the runner can write and search, with no directory named `results.json`
or `report.md` in it, that an absent `--out` lies below no existing
non-directory and its nearest existing ancestor is a directory the runner can
write and search, that `--out` holds, when layer 3 runs, no `holdouts` directory from an earlier run (layer 3 needs a fresh
directory, so no stale draft can sit beside new ones), that a `--cache` given
with layer 3 is either absent and creatable by the same test as `--out` or a
regular file, not a symlink, that the runner can read and write in a directory
it can write and search (SQLite writes `-wal` and `-shm` files beside it),
that the snapshot opens, that its `corpus_fingerprint` matches the
pack's when the pack sets one, that every retrieval case's `filters` resolve
against it and its `expected_docs` name documents of it, that every Q&A source anchor names a document of the snapshot
and pages within it, that each anchor of a case other than `abstention`
overlaps some section, and that each case's evidence (below) is at most 48,000
characters, that the fingerprint of the `--corpus` files (found by
`discover_corpus`) equals the snapshot's, that each selected holdout's
reference, included files and exemplar paths pass the layer 3 guards below,
that its reference and every `exemplar_paths` document have at least one
section in the snapshot, and that LibreOffice (`soffice`) is on the path when
a selected layer will convert a Word or PowerPoint file (layer 2 reads the
whole corpus; layer 3 builds each selected holdout from its included files). A failure is a usage error
(exit 2) without a traceback. Then, still before `--out` exists and before any
model call, the gateways check the credential, `--trace` and model prices, and,
when layer 3 runs, the `--cache` file is opened; a failure there exits 1 with the usage line and
the error. A failure after model calls have been made also prints the usage
line before the error.

Eval agents and judges go through `OpenAIGateway` at medium reasoning effort
on `gpt-6.1-sol`: agents through `converse`, judges through `run`, under the
gateway's retry policy and metering. An agent run records its final status:
only `completed` counts as an answer; `incomplete` (or another response
status), `empty_answer`, `max_turns` and `provider_error` (including a
`ProviderError` raised by a tool) are reported and never credited. A query
embedding that fails behind the MCP `search` tool also ends its run as
`provider_error`: the MCP server hands the model a tool error, so the harness
reads the failure from its query embedder after the conversation. Invalid
input to a harness tool goes back to the model as an error message; any other
exception from a harness tool stops the evaluation. An unexpected exception
inside an MCP server tool reaches the model as a tool error, as any MCP host
would see it, and does not stop the evaluation. Each run reports its tokens,
turns, the query embeddings that returned a vector (`search_embed_calls`), and
its tool calls, each with the tool name, the argument text, the length and
first 200 characters of the returned text and the image count (see Provider
calls). The query embeddings are recorded per run, so a failure in one run
cannot mark the next. `report.md` totals the calls and the returned
characters per arm and tool name. `--max-cost` sets one spend limit, a positive finite
number of USD, over every model call of the evaluation. Once the estimate
reaches the limit, the next provider attempt stops the evaluation with an
error, the usage line and a non-zero exit, and writes no report; a report an
earlier run left in `--out` stays as it was. If the last
request takes the estimate past the limit, the evaluation finishes normally.
`--trace` turns on hosted Agents SDK tracing, and `results.json` (`usage`) and
`report.md` (Usage) record calls, tokens and the estimated cost per operation.

An eval pack is a JSON file validated by `EvalPack` in `evals/schema.py`
(`format` `arkive-eval-pack/3`). It holds `retrieval_cases`, `qa_cases` and a
list of `holdouts`. A retrieval case may set `filters`, a mapping
from facet name to value or alias that layer 1 passes to `rank_channels` and
`search`;
facets are induced per snapshot, so `arkive-rag inspect` lists the names and
values a pack can use. A Q&A case has a `kind`, one of `lookup`,
`multi_document`, `aggregation`, `abstention`, `conflict` and `vocabulary`,
and `source_anchors`, each `{"doc_id": ..., "pages": [first, last]}` or
`"pages": null` for the whole document (Markdown sources have no pages, so
their anchors use null; an anchor with page numbers on a Markdown document
fails the preflight). Page numbers must be JSON integers: a boolean, a
numeric string or a float such as `3.0` is refused. Every kind but
`abstention`, where the corpus does not hold the answer, needs at least one
anchor, and each of its anchors must overlap the text of some section of the
snapshot; an anchor on a page no section covers fails the preflight with the
case and the anchor named. A holdout has a unique
`holdout_id` (letters, digits, `.`, `_` and `-`), `included_paths`,
`held_out_markers`, `reference_doc_id`, `instructions`, `run_notes`, an
optional `rubric` and `exemplar_paths`: the corpus-relative files the pack
author names as this deliverable's exemplars and feedback documents, each of
which must be one of the holdout's included files and a delivered document,
not a curator note. The list is the same in
every run: it does not depend on what induction makes of the corpus. An empty
list means the exemplars-in-context arm does not run for that holdout. The
rubric may be empty, because the lexical-overlap measures below need none.
Expected facts are
*groups* of alternative substrings, matched NFKC-, case-, dash-, whitespace-
and thousands-separator-insensitively and with Markdown emphasis ignored, so
`does **not** hold` matches `does not hold`. Emphasis is backticks, `*` runs
that touch a non-space character (a spaced `2 * 3` keeps its sign) and `_`
runs with a letter or digit on one side only: `_italic_` and `__bold__` lose
their underscores, while `harbour_dry` and `survey__v2.tif` keep every one.
Table pipes stay, so a group such as `| values` can test a table row. Pack
validation refuses a phrase in `required_groups`, `require_groups` or
`forbid_any` that is empty once normalized, such as ```` ``` ```` or `**`,
because an empty phrase would match every text. A `RubricCheck` passes when the draft
satisfies every `require_groups` group and contains no `forbid_any` phrase.
An anti-contamination check requires the held-out project's own values and
forbids only claims that a faithful draft cannot make, so a draft that
contrasts its values with an earlier project's still passes. `evals/example_pack.eval.json`
is a fictional pack for no corpus in the repository (its `corpus_fingerprint`
is null); it is the offline tests' fixture. `evals/example_packs/` holds one
fictional pack per example collection, for real paid runs against snapshots
of that collection (see Example corpora). Real packs stay in the private store.

| Layer | Module | Measures |
| --- | --- | --- |
| 1 | `ablation.py` | Completion@1/3/5 and CompletionMRR@5 of four rankings over the pack's retrieval cases, per category. A case completes at rank r when the top-r chunks jointly cover every required group and expected document. `TopicSnapshot.rank_channels` ranks three arms over raw chunk channels: `bm25_only`, `dense_only` and `hybrid_rrf`. The fourth, `served_search`, is `TopicSnapshot.search` with the same top-k, the ranking agents get: it leaves out a superseded document whose replacement passes the case's filters and takes at most `SEARCH_FAMILY_CAP` (3) chunks of one version family before held-back chunks fill the list; it shows whether that collapse and cap keep or improve completion. All four use one query embedding per case, made only after the first ranking has validated the query, filters and top-k, so an invalid case fails with no embedding call. |
| 2 | `agents.py` | An agent conversation answers each QA case once with raw-file tools (text grep, page text, page images over the files `discover_corpus` accepts) and once with the ten Arkive MCP tools. Each completed answer gets deterministic phrase coverage (`required_groups`) and a blinded judge score with evidence, below. Raw-file extraction runs once before timing. The report groups results by `kind`. |
| 3 | `drafting.py` | For each holdout, or the one `--holdout` names, builds a holdout snapshot, records which documents `prepare_draft` selects, and runs `--drafts` rounds of four draft arms, and a fifth when the holdout's `exemplar_paths` is not empty, measured against a reference deliverable; the procedure, guards and measures are below. |

The Q&A judge sees the question, the reference facts, the anchors and their
evidence: the snapshot's Markdown of every section that overlaps an anchor's
pages (every section of the document for a null anchor), each labelled with
the document's path, `doc_id`, `section_id` and page span, so a citation by
file and page and a citation by ID both resolve. The evidence is the
snapshot's transcript and can itself be wrong, so the judge is told that the
reference facts decide when they and the evidence disagree. Evidence over
48,000 characters fails the preflight for that case; it is never cut. The
judge scores factual accuracy and completeness against the reference facts (a
fact the evidence supports is not an accuracy defect) and citation quality
against the anchors, and lists the answer's material claims, each
`supported` (by the reference facts or the evidence), `contradicted` (by the
reference facts, or by the evidence where the facts are silent) or
`unverifiable`. Each judged row of `results.json` carries the claims and
`claim_counts`; `report.md` gives the contradicted and unverifiable counts per
arm and per kind. An unverifiable claim is reported, never counted as a
contradiction. A judge call that fails with a `ProviderError` other than the
spend limit (output that fails schema validation, a refusal, output still
truncated after the retry, a provider failure) leaves that row with `judge`
null and the reason in `judge_missing`; the row keeps its phrase coverage.
`report.md` counts the judged rows per arm and kind, averages judge scores over
judged rows only, and lists each completed run that was not judged with the
reason. The spend limit still stops the evaluation.

For layer 3:

1. Select a delivered reference document, not a curator note, with a `held_out_markers` entry in its path. Select files through `included_paths`; pack validation requires plain paths and `discover_corpus` refuses symlinks. Every `exemplar_paths` entry must be one of the selected files and not a curator note.
2. Before copying files or calling a model, the guards reject a selected file that is held-out material: its path contains a marker, or its bytes match the reference. Markers match case- and dash-insensitively. The guards infer nothing from folders, so a holdout works the same in a folder per project, nested folders or a flat folder: the pack author chooses markers that cover the held-out work's files (a marker may be a folder name) and lists `included_paths` explicitly. The guards do not scan content: the pack author must leave out changed copies, curator notes, summaries and other documents that restate held-out facts.
3. Compile the selected bytes alone into a holdout snapshot, using the full snapshot's embedding model and dimensions and optionally `--cache`. Check the snapshot again before publishing it as `<out>/holdouts/<holdout_id>/holdout.arkive.db`. Cache reuse follows the request-input rules above.
4. When `exemplar_paths` is not empty, record the exemplar selection (below): one `prepare_draft` call on the holdout snapshot before the first round.
5. Run `--drafts` rounds. Four arms always run in every round from `run_notes` and `instructions`: `zero_shot` (no tools), `chunk_search` (a five-chunk search tool), `raw_files` (the four raw-file tools of layer 2 over the included files) and `arkive_mcp` (the MCP tools and the server instructions). The fifth, `exemplars_in_context`, runs when `exemplar_paths` is not empty; otherwise the report says it was not run. Every arm has the same 16-turn limit. Every completed draft gets the rubric checks and the two lexical-overlap measures below, and the round's completed drafts get one blinded side-by-side judgment against the reference. A document of the holdout snapshot without sections, such as an empty curator note, contributes no sections; a reference or exemplar document without sections stops the holdout.

The `raw_files` arm measures what an agent with file tools alone drafts
from the collection: `list_corpus_files`, `grep_raw_text`, `read_raw_pages`
and `view_raw_page_image`, as in layer 2. Its instructions are the holdout's
`instructions` followed by `RAW_FILES_DRAFTING_HINT` in `evals/drafting.py`,
which tells it to look for prior deliverables and reviewer feedback with the
tools, take every new-project fact from the instructions and the run notes
(the authorised inputs the blind judge accepts) and write a standalone
deliverable without file paths. Its files are exactly the included files the
holdout snapshot was compiled from, listed and searched in the snapshot's path
order whatever the order of `included_paths`. Before any round, the guard of
step 2 runs again over their paths and SHA-256 hashes, and each file must match
a holdout snapshot document byte for byte. Nothing is rendered again: a
Markdown file is read as its raw text, and any other file is the native text
layer and WebP of the pages the holdout build stored, the same pdfium output
layer 2 renders. Page images are read from the holdout snapshot when the
agent asks for one, so the arm starts no LibreOffice process.

The `exemplars_in_context` arm has no tools. Its prompt holds the section
Markdown of each `exemplar_paths` document of the holdout snapshot, labelled
with its path and without visual transcripts, then the run notes. Its
instructions are the holdout's `instructions` followed by
`EXEMPLAR_DRAFTING_RULES` in `evals/drafting.py`. The arm is the control: it
shows what full context of the documents a person chose achieves, and the
`arkive_mcp` arm is compared against it across server changes. For that
reason `EXEMPLAR_DRAFTING_RULES` is frozen on its own. It does not follow
`arkive_rag.server.INSTRUCTIONS` or the packet's `DRAFTING_GUIDANCE`: a
control that changed with the server would move with the treatment and
confound every before/after comparison. `tests/test_evals.py` pins
`EXEMPLAR_DRAFTING_RULES` word for word in one test, and pins the server's
drafting surface that the `arkive_mcp` arm measures in another: the whole of
`INSTRUCTIONS`, `DRAFTING_GUIDANCE`, the names of the server's tools (which
`ARKIVE_TOOLS` of the OpenCode check must equal) and the `prepare_draft` and
`read_section` descriptions as the server publishes them. A server change
fails only the second test. Change `EXEMPLAR_DRAFTING_RULES` only as a
deliberate change of baseline, and rerun the baseline when you do. The report
gives each arm's input tokens, so the cost of carrying the exemplars is
visible beside the other arms.

The exemplar selection measures, with no model call besides one query
embedding, whether `prepare_draft` surfaces the documents the pack author
named. It calls `prepare_draft` on the holdout snapshot with the holdout's
`instructions` and `run_notes` as project inputs and, as `deliverable_type`,
the reference's type when the holdout snapshot has that type (`requested_type`;
null for a type whose only document is the reference). An agent drafting that
deliverable through the MCP tool would request the same type, so an exemplar
of another type cannot decide which type's lessons the packet carries.
`results.json` (`selection`) records the named deliverables
(`oracle_exemplar_paths`), the named feedback documents
(`oracle_feedback_paths`), the packet's exemplars and other candidates by
path, and three shares. When the reference is a feedback document, or every
named document is one, the holdout drafts a review: every named document is
then a deliverable to surface, and none counts as feedback. Every share counts
version families once (`oracle_exemplar_families`,
`oracle_feedback_families`), so naming two versions of one deliverable counts
as one, and a superseded document counts through the member that replaced it.
A named deliverable family is surfaced when one of its documents is among the
packet's exemplars (`exemplar_recall_at_packet`) or among its exemplars and
other candidates (`exemplar_recall_at_candidates`). `feedback_attached` is the
share of named feedback families that some packet lesson comes from, so a
superseded review counts when its replacement supplied the lessons. A share is
null when the holdout names no document of its kind. `report.md` gives the
selection in one line per holdout.

The two measures compare wording, not meaning. They reuse the build's text
functions from `arkive_rag.passages`: `standing_units`, word 4-gram
`shingles` of `fold_for_quote` text, `containment`, `STANDING_MIN_WORDS` and
`STANDING_RECURRENCE_CONTAINMENT`, and the list-marker pattern `LIST_MARKER`.
A visual transcript is a block quote whose first line opens with
`> **[Visual`, the same rule `standing_units` applies, so a line with that
prefix inside an ordinary block quote or a fenced block starts nothing. The
sections of curator notes in the holdout snapshot are not holdout text.

- The *prose units* of a text are its `standing_units` that are not table
  headers, with visual transcripts removed by `standing_units` itself. Each
  unit loses its leading list marker before it is folded, so a `1.` item and
  a `-` item with the same words measure the same. A marker counts only when
  horizontal space and content follow it on the same line; a `1.` alone on its
  line stays a word of the paragraph it opens. A unit's shingles are those
  of its folded text.
- A draft's *prose shingles* are the union of its prose units' shingles, of
  units of any length.
- The *text shingles* of the reference, of authorised input (the holdout's
  `run_notes` and `instructions`) and of each holdout section are the union of
  the text's whole-text shingles (visual transcripts removed) and the shingles
  of each of its prose units. Folding a whole text joins a hyphen or dash at a
  line end with the next unit's first word, so the two forms differ; their
  union means a sentence present on both sides always matches on this side,
  and a shingle the reference carries anywhere, even in a table, is never
  stray.
- A *holdout document's text* is the union of the text shingles of its
  sections: one set per delivered document of the holdout snapshot.
- A prose unit is *held* when at least one holdout document's text contains
  at least `STANDING_RECURRENCE_CONTAINMENT` of its shingles; its *holders*
  are the documents that do. A paragraph that joins sentences from two
  sections of one document can therefore be held. Common 4-grams spread over
  several unrelated documents cannot add up to a held unit, however many
  documents the holdout has: with hundreds of documents, an ordinary sentence
  would otherwise find most of its 4-grams somewhere in the collection.
- A *house unit* is a prose unit of the reference, in the full snapshot, of at
  least `STANDING_MIN_WORDS` words that is held. The *house shingles* are the
  union of the house units' shingles minus authorised input's text shingles.
- *House-text recall* is the share of the house shingles that are among the
  draft's prose shingles. It is null when the holdout has no house shingles;
  the report gives the number of house units and house shingles.
- A *copied unit* is a prose unit of the draft, of at least
  `STANDING_MIN_WORDS` words, that is held. The *stray shingles* are the union,
  over the copied units, of each unit's shingles that its holders' text holds
  and neither the reference's nor authorised input's text shingles hold.
- *Copied-text contamination* is the number of stray shingles divided by the
  number of the draft's prose shingles. Words a drafter adds to a copied
  passage, such as the new project's client name, are not in the document it
  was copied from and do not count, even when another holdout document
  happens to hold them. It is 0 for a draft without a copied unit and null only
  for a draft without prose shingles. Both measures lie between 0 and 1.

`results.json` lists, per draft, each house unit with the share of it the draft
holds and each copied unit with its stray-shingle count, by the first 120
characters of the unit. The measures have limits. They compare wording, not
meaning. Tables, headings and fenced blocks are not prose: their text counts
neither as house text nor towards a draft. A unit under `STANDING_MIN_WORDS`
words is never a house or copied unit, though its shingles count in the
draft's total. A visual transcript contributes nothing and separates the text
around it, as blank lines would. A block quote is one unit however many
paragraphs it holds, so a copied paragraph inside a longer quote can go
unseen. The shared fold joins a hyphen at a line end with the next word. A
house paragraph that the reference's authors reworded counts partly as
contamination. A draft paragraph stitched together from several short copied
sentences, with words of the drafter's own between them, can hold under
`STANDING_RECURRENCE_CONTAINMENT` of its shingles in any one document and then
counts as no copied unit; so does a paragraph that joins sentences copied from
two different documents. A document's text includes each section's whole-text
shingles, which join the end of one list item to the start of the next, so a
draft sentence that happens to span two adjacent holdout list items can count
as copied. The zero-shot arm never sees the collection, so its contamination
shows the observed rate of such false matches. Repeated drafts from one
holdout build measure drafting variation, not build variation.

Rubric diagnostics label each required group of each rubric check by where one
of its alternatives first occurs: `run_notes`, `instructions`, `holdout` (any
section of the holdout snapshot) or `none`. The report lists every check with a
group labelled `none` as "required phrase found in no supplied input". The
label tells the pack author where a phrase can come from; it is not a verdict
on whether a draft can derive it.

The blind judge receives the holdout's `instructions` as task instructions,
the run notes, the reference without its visual transcripts and the round's
completed drafts under `Candidate_<n>` labels, and is told that figures are out
of scope. It is told that both the task instructions and the run notes are
authorised inputs, so a draft that names the author, role or company the
instructions give is not penalised for inventing them. The judge gets the
whole reference, however long; it is never cut. Its output counts only when its
assessments and its ranking each name exactly the labels sent, once each.
Otherwise, or when the judge call fails with a `ProviderError` other than the
spend limit (output that fails schema validation, a refusal, output still
truncated after the retry, a provider failure), the round keeps its drafts and
measures, `blind_comparison` is null and `blind_comparison_missing` gives the
reason, as it does for a round where no draft completed. The spend limit still
stops the evaluation.

`report.md` gives, per holdout and arm, the completed drafts, both measures and
each judge score as median [minimum, maximum] over the rounds with n, the
number of observations behind it, the rounds that judged the arm, the stray
and draft shingle totals, rubric pass counts per check, input and output
tokens and tool-call totals, with each round's ranking or the reason it was
not judged. Drafts that did not complete are left out of the measures, rubric
counts and judge scores; their tokens and tool calls are counted. In the
report, `n/a` means no observation: no completed draft with a value, no judged
round or, for recall, no house shingles. In `results.json`, a null recall means the holdout
has no house shingles, a null contamination means the draft has no prose
shingles, and null measures and rubric for an arm mean its draft did not
complete; zero is a measured value. No decision rule is applied: comparing
arms from these medians and ranges is the reader's step.

The evaluation harness compiles the holdout snapshot through `compile_corpus`
itself, the one real OpenAI build outside `arkive-rag build`. Each round's
blinded drafts and `blind_key.json` are written to
`<out>/holdouts/<holdout_id>/blind-drafts/round-<n>/`. `results.json` keys
layer 3 results by `holdout_id`; each holdout gives its `selection` and lists
its `rounds`, each with the arms' runs, rubric results and measures.
`tests/test_evals.py` covers pack validation, scoring, the rubric, the
lexical-overlap measures, the rubric diagnostics, the exemplars-in-context
prompt and its frozen rules, the server drafting surface pins, the holdout
guards, the raw-file arm's files, the exemplar selection, the run-status
contract, per-run embedding records, the judge's evidence and claim
counts, the layer 3 rounds, the runner's input checks, a run of every layer of
the example pack and the tool adapters offline, using the example pack and a
scripted OpenAI HTTP transport; `tests/test_llm.py` covers the tool-call record.

### Example corpora

`evals/example_corpora/` holds three small fictional corpora, so anyone can
build real snapshots without private documents. They are organised in
different ways, and induction gives them different facet schemas:

- `coastal-survey-projects`: one folder per survey project, each holding a
  proposal, a technical memorandum and reviewer feedback, with a curator note
  that records a missing review. `q102-technical-memorandum.pdf` has a bar
  chart whose per-bench volumes appear only in the image.
- `creator-campaigns`: one folder per creator, then one per campaign, holding
  campaign briefs and post-campaign reviews, with a casual root `AGENTS.md`.
  `harbourline-launch-deck.pptx`, the results deck sent after the campaign's
  flight, has a chart whose weekly reach values appear only in the image.
- `acoustic-consultancy-files`: one flat folder of Word and PDF files with no
  curator note, dated only by the files' embedded metadata. It holds two
  recurring deliverable types, noise reports and quotations, and one type with
  a single document, a response to planning comments. The older and newer
  noise reports use different layouts; they share recurring house paragraphs,
  a caveat that varies by one instrument name, and a section only the newest
  report holds. `corvane-depot-night-noise-report.pdf` has a chart whose
  values appear only in the image.

The first two repeat deliverable types, reviewer feedback, defined and undefined
acronyms, one acronym with two meanings, tool names, stated attributes,
exact numbers and filenames. Each also has one standing section that recurs
across projects: every coastal proposal ends with the same standard terms,
and every campaign brief ends with the same disclosure, so their drafting
packets report recurring units. They are built with the ordinary
`arkive-rag build` command; the snapshots are not tracked.

`evals/example_packs/<collection>.eval.json` is the eval pack of each
collection, built for real paid evaluation runs against a snapshot of that
collection; `evals/example_pack.eval.json` stays the offline test fixture.
Each pack sets the collection's `corpus_fingerprint`, has no retrieval cases,
and has four Q&A cases, one each of `lookup`, `multi_document`, `abstention`
and `vocabulary`, whose anchors cover whole documents. The holdouts are:

- `coastal-survey-projects`: the proposal and the technical memorandum of the
  Fennick Canal project (`q103`), each built from the `q101` and `q102`
  folders.
- `creator-campaigns`: the Kettlecraft winter campaign brief, built from the
  root `AGENTS.md`, the other creator's campaigns and the same creator's
  Corriemoor campaign.
- `acoustic-consultancy-files`: two noise reports, the Pellow Mill facade
  report and the Corvane depot night noise report, each built from the files
  of the other three projects.

`test_example_collection_packs_match_their_collections_and_hold_out_their_references`
in `tests/test_evals.py` checks, without a snapshot or a model call, that every
pack in `evals/example_packs/` loads, that its fingerprint matches its
collection's files, that every anchor and `reference_doc_id` names a document
of the collection (its ID is `document_id` of the file's path), and that every
holdout passes `check_reference_held_out` and `select_holdout_sources`.

### OpenCode check

`python -m evals.opencode_check SNAPSHOT... --out DIR [--model
provider/model] [--scenarios FILE] [--only ID] [--timeout S]` checks that a real
MCP client, OpenCode, can discover and use the Arkive tools with a real model,
non-interactively, on several topics at once. It needs Docker and
`OPENAI_API_KEY`, and it is not part of `pytest`.

- **Image.** `evals/opencode_check/Dockerfile` starts from the official
  `ghcr.io/anomalyco/opencode:2.3.0` image, pinned by digest. Its `opencode`
  binary is linked against musl, so the image stays on that Alpine base and
  adds Alpine's Python 3.14. The runner builds the context in a temporary
  directory: the arkive-rag wheel built from the checkout (`uv build`), the
  runtime requirements exported from `uv.lock` with hashes (`uv export`), and
  the files beside the Dockerfile. The image is tagged
  `arkive-opencode-check:<content hash>` and built only when that tag is
  missing. It holds no key, snapshot or corpus.
- **Client configuration.** `opencode.json` becomes the system configuration
  `/etc/opencode/opencode.json`, which outranks global and project files. It
  declares one local stdio MCP server, `arkive`, that runs `arkive-rag serve`
  on every snapshot under `/snapshots`, with `codemode: false` so the tools sit
  on the model's native tool list as `arkive_<tool>`, and a start-up timeout of
  180 s (`timeout.startup`): OpenCode marks a server that misses it failed and
  never retries it, and on a loaded machine `arkive-rag serve` has needed 10
  to 20 s to answer `initialize`. Its permissions deny every action and then
  allow `arkive_*`.
- **Startup.** `opencode run --standalone` sends the first model request
  without waiting for local MCP servers, so `arkive`, which needs several
  seconds to open its snapshots, is missing from the tool list.
  `run-scenario.sh` therefore starts a private `opencode serve`, waits until
  `opencode mcp list` reports `arkive` connected (a 240 s polling deadline,
  with each listing bounded to 10 s; one listing and a diagnostic listing can
  extend beyond the deadline; exit 3 when it passes or the server is reported
  failed) and then runs the prompt with `opencode
  run --server ... --format json --auto -- PROMPT`, so a prompt that starts
  with a dash is still the message.
- **Isolation.** Each scenario runs in a fresh container, as the invoking
  user's UID and GID (`--user`), so the owner-only snapshot files stay
  readable for any host UID; the runner refuses to run as root.
  `run-scenario.sh` keeps HOME and the working directory under `/tmp`, so
  neither depends on an image account. The snapshots are mounted read-only and
  no other host path is mounted; all capabilities are dropped, with
  `no-new-privileges`. OpenCode's permission list denies every tool outside
  the `arkive` server. The containers use Docker's default network, so network
  egress is not restricted. The check has been run with a rootful Docker
  daemon; rootless Docker is untested. `OPENAI_API_KEY` is passed by name
  (`--env OPENAI_API_KEY`), so its value never appears in a command line, and
  reaches both the model requests and the `search` tool's query embeddings.
  The runner refuses an empty key, and replaces the key's value in the event
  stream and stderr before parsing them, so no file it writes holds the key.
  Each container is named `arkive-opencode-check-<run id>-<scenario id>`, with
  a random run ID, so concurrent runs do not collide. It is force-removed
  (`docker rm --force`) when its scenario ends, times out (`--timeout`, 600 s
  by default), fails or is interrupted; a removal that fails is printed and
  makes the exit status 1.
- **Scenarios.** `scenarios.json` (`format` `arkive-opencode-scenarios/1`,
  validated by `ScenarioFile` in `scenarios.py`) is tied to the example
  corpora. A scenario has a prompt, the Arkive tools that must complete at
  least once, argument expectations (`non_empty`: some completed call passed
  a non-empty argument such as `filters`; `invalid_filter_recovery`: a call
  whose `filters` set a facet to the scenario's invalid value ended in a tool
  error, and a later call to the same tool on the same topic, setting that
  facet to another value, completed), the topics its calls must query, answer
  phrase groups matched as in eval packs, an optional requirement that the
  answer names every facet of its topics (read from the served snapshots,
  because facets are induced per build), an optional requirement that the
  answer reproduces a recurring passage
  (`answer_reproduces_recurring_passage_of`, a topic: the passages are the
  `recurring_samples` of the packet that the last completed `prepare_draft`
  call on that topic returned, because packets are built per request, so a
  project paragraph copied from a section that also holds house text does not
  count; a passage counts when ten of its words in a row, or all of a shorter
  one, appear in the answer as words after `fold_for_quote`, so an agent may
  adapt its opening words to the project; when that packet includes no such passage the
  scenario fails a separate check, `precondition: packet has a recurring
  passage`, which marks the snapshot rather than the client), and an optional
  page (`page_image`: topic, `doc_id`, `page_number`) whose `view_page` call
  must return an image.
  Document IDs follow sorted path order, so they are stable across builds of
  one corpus. Every scenario also requires that the run ended with an answer
  and no error event, that no tool outside the `arkive` server was called, and
  that OpenCode cut no Arkive tool result (`state.metadata.truncated`). The shipped scenarios cover
  topic and facet discovery, three glossary lookups (a defined term, a term
  the corpus never defines and a term with two meanings), a filtered listing
  with a facet the agent finds itself, recovery from an invalid filter value,
  an exact number found with `grep`, a chart value read through `view_page`,
  a draft that must fetch the drafting packet with `prepare_draft` and
  reproduce one of its recurring passages, and a question that needs both
  topics.
- **Results.** The runner parses the newline-delimited JSON events
  (`tool_use`, `text`, `step_finish`, `error`); the final answer is the text
  written after the last tool call. It writes `results.json` (per scenario:
  tool calls with arguments, status and output, the final answer, each check's
  outcome, the tokens and OpenCode's cost estimate from `step_finish` events,
  the exit code), `report.md` and the raw event stream of each scenario under
  `events/`, all owner-only (`write_private_text` in `evals/run.py`, which
  publishes by replacement, so a symlink at the destination is replaced, never
  written through; `events/` must be a real directory).
  OpenCode can omit the last step's `step_finish`, so token and cost totals
  are lower bounds, and the `search` tool's query embeddings are not in them. The exit status is 1 when any scenario fails or
  cannot run. Before building an image or starting a container, the runner
  refuses an invalid scenario file, an `--out` that overlaps the scenario file
  or a snapshot (the rule `evals.run` applies to its inputs), a snapshot that
  does not open, two snapshots of one topic, a scenario topic no snapshot
  serves, an unknown `--only` ID, a missing or empty key, a root caller, and a
  symlink where an output file or `events/` would be written.

OpenCode 2.3.0 receives `view_page`'s image as a file part of the tool result
(`image/webp`), and the scenario checks that the `view_page` call for the chart's
page returned it. The chart value is also in
the page's visual summary, which `view_page` returns as text, so a correct
answer alone does not prove that the model read the image.

The README states what a pass proves. `tests/test_opencode_check.py` covers the
scenario file, event parsing, every kind of check failure, the container
command, the runner's input refusals, output links, key redaction and container
removal (including an interrupt) offline, with a fake Docker and no key.

## Verification

All commands run from a checkout root. Keep the environment and tool caches
outside the checkout, for example:

```sh
export SCRATCH="$(mktemp -d /tmp/arkive-checkout.XXXXXX)"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv" MYPY_CACHE_DIR="$SCRATCH/mypy" \
  RUFF_CACHE_DIR="$SCRATCH/ruff" PYTHONDONTWRITEBYTECODE=1
uv sync --locked
uv run --locked ruff check src tests evals
uv run --locked ruff format --check src tests evals
uv run --locked mypy
uv run --locked pytest -o cache_dir="$SCRATCH/pytest-cache" --basetemp="$SCRATCH/pytest-tmp"
uv build --out-dir "$SCRATCH/dist"
```

The test suite makes no network calls. It needs LibreOffice (`soffice`) for the
Word and PowerPoint conversion test; that test is skipped when `soffice` is
absent.
