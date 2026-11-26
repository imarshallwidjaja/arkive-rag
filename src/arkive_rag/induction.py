"""Induction: how a whole corpus is organised, induced from the readings.

`induce` runs its steps in a fixed order: organise (facets, and every citable document's
canonical deliverable type and feedback role), bind (facet values to documents), then version
families, feedback lessons and the glossary, and last the topic guide. Recurring text units
are computed in code (`arkive_rag.passages`). Every generating step goes through
`generate_checked`: run, validate, rerun once with the rejections named, then raise. Only
three outcomes reduce output without failing, and the build report counts each: an
observation or feedback lesson whose quote is still unlocated after the rerun, a glossary
definition the support check does not confirm (kept with `definition=None`), and a feedback
lesson it does not confirm (dropped).

Every prompt is bounded whatever the corpus size: documents reach the organisation step as
one-line digests in batches, heading paths are cut by `heading_context`, curator notes by
`MAX_CURATOR_NOTES_PROMPT_CHARS`, and the guide's catalog shrinks per document as the corpus
grows. No step caps how many deliverable types a corpus has.

Everything that reaches a prompt or a cache key is in a canonical sorted order, and
concurrent results are gathered in submission order, so an unchanged corpus with the same
build cache rebuilds without provider requests.
"""

from __future__ import annotations

import json
import re
import threading
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Self

from arkive_rag import passages as _passages
from arkive_rag.llm import Effort, LanguageGateway
from arkive_rag.models import (
    ArkiveError,
    BuildReport,
    CorpusDocument,
    CuratorNote,
    DeliverableType,
    DocumentReading,
    DocumentRole,
    Facet,
    FacetCanonicalisationDraft,
    FacetDraft,
    FacetValue,
    FeedbackLesson,
    FeedbackLessonsDraft,
    GlossaryDraft,
    GlossaryEntry,
    GlossaryEntryDraft,
    GlossarySense,
    GuideDraft,
    LocatedAttribute,
    NotFoundError,
    OrganisationDraft,
    ReadSection,
    RejectedOutputError,
    TermObservation,
    TypeCanonicalisationDraft,
    UnitCluster,
    VersionPartitionDraft,
    VersionStatus,
    fold_facet_label,
    fold_for_quote,
    fold_text,
    heading_context,
)

MAX_FACETS = 6
# A content-bound value is assigned, a definition or feedback lesson kept, when the decision
# model's probability reaches this threshold. A refusal never confirms.
SUPPORT_THRESHOLD = 0.5
GLOSSARY_BATCH_CHARS = 24_000
MAX_OCCURRENCES_PER_TERM = 6
# Quotes shown per term, defining quotes first, and the characters shown of each quote and of
# a term's alias list, so one term found in hundreds of documents stays a bounded entry.
MAX_QUOTES_PER_TERM = 12
MAX_TERM_QUOTE_CHARS = 600
MAX_TERM_ALIASES_CHARS = 600
MAX_TERM_ALIASES_SHOWN = 12
MAX_TERM_ALIAS_CHARS = 120
MAX_TERM_KIND_CHARS = 80
# An occurrence snippet starts this many characters before the match and is at most
# `OCCURRENCE_CHARS` long, however long the matched alias is.
OCCURRENCE_RADIUS = 120
OCCURRENCE_CHARS = 360
# A support check sends at most this many characters of evidence per request. Longer evidence
# is judged in windows, a section too long for one window is split into overlapping parts, and
# each predicate takes its highest probability over the windows: no cited text is cut.
EVIDENCE_WINDOW_CHARS = 48_000
EVIDENCE_OVERLAP_CHARS = 2_000
# A window labels each section with its heading path. The section's Markdown already holds its
# own heading line, so a longer label is shortened; this keeps the parts of a long section
# close to a full window whatever its heading, so their number stays bounded by its length.
EVIDENCE_LABEL_CHARS = 1_000

# Every induction prompt (instructions and request text) stays within this many characters,
# about 30k tokens, on a corpus of any size, on the first run and on the rerun alike. The
# limits below are chosen so that their sum per request stays within it.
MAX_PROMPT_CHARS = 90_000
# The rejections a rerun names, each cut to `MAX_REJECTION_CHARS`, as many as fit
# `MAX_REJECTION_FEEDBACK_CHARS`. A guide's citation rejection is the longest a check writes
# (`MAX_CITATION_FEEDBACK_CHARS`), so it reaches the rerun whole.
MAX_REJECTION_FEEDBACK_CHARS = 6_000
MAX_REJECTION_CHARS = 4_000
# Curator notes shown in any one induction prompt, in path order; the snapshot keeps them all.
MAX_CURATOR_NOTES_PROMPT_CHARS = 24_000
# Document digests per organisation request; a larger corpus is organised in several batches.
MAX_ORGANISE_BATCH_CHARS = 44_000
DIGEST_SUMMARY_CHARS = 180
DIGEST_TITLE_CHARS = 120
DIGEST_PATH_CHARS = 160
DIGEST_TYPE_CHARS = 80
DIGEST_ATTRIBUTE_CHARS = 120
# The folders an organisation batch lists, shallowest first.
MAX_FOLDER_LIST_CHARS = 12_000
# The type labels of one canonicalisation request, with example titles while they fit.
MAX_TYPE_LABELS_CHARS = 60_000
TYPE_LABEL_EXAMPLES = 3
# The outline lines of one binding question's evidence, its stated attributes, and each
# description a binding predicate quotes.
MAX_BINDING_OUTLINE_CHARS = 12_000
MAX_BINDING_ATTRIBUTES_CHARS = 4_000
MAX_BINDING_DESCRIPTION_CHARS = 300

# The facet proposals of one canonicalisation request: every value, alias and folder, with
# value descriptions cut to the first of these lengths that fits (0: no value descriptions).
MAX_FACET_PROPOSALS_CHARS = 54_000
FACET_DESCRIPTION_STEPS = (200, 60, 0)

# Version families. Code proposes scored candidate pairs and groups them into bounded
# components; one structured request per component partitions it into families.
# Share of one document's distinctive sampled word 4-grams the other must hold.
VERSION_CANDIDATE_CONTAINMENT = 0.35
# Distinctive sampled 4-grams each document of a 4-gram candidate pair must have; a document
# with fewer is paired by its title and file-name words alone.
VERSION_MIN_GRAMS = 8
# Share of distinctive title and file-name words two documents of one deliverable type share.
VERSION_TITLE_OVERLAP = 0.6
# Partners kept per document, best first, so the number of pairs grows linearly.
MAX_VERSION_PARTNERS = 6
# One 4-gram in this many is sampled (by checksum) to estimate containment.
VERSION_SHINGLE_SAMPLE = 4
# A sampled 4-gram or title word held by more documents than this is boilerplate shared
# across deliverables, not a sign of one deliverable: it counts in neither part of a score,
# while a family of up to this many versions keeps every 4-gram its members share.
VERSION_COMMON_POSTING = 12
# Documents of one partition request, so that its cards and pairs fit well within the
# prompt bound; weaker pairs that would join a larger component are dropped and counted.
MAX_VERSION_COMPONENT_DOCS = 16
VERSION_EVIDENCE_HEADINGS = 12
VERSION_HEADING_CHARS = 120
VERSION_SUMMARY_CHARS = 600
VERSION_ATTRIBUTE_CHARS = 600

# A feedback document is read in batches of at most this many characters of its sections.
FEEDBACK_BATCH_CHARS = 24_000

# The guide request. Its catalog shrinks per document as the corpus grows.
MAX_GUIDE_TEXT_CHARS = 72_000
# The guide an agent reads whole in `get_topic_guide`, so it stays well within one tool result:
# at most this many characters, and this many bytes as a JSON string in UTF-8, the form a tool
# result carries it in (a guide in a script of three-byte characters reaches the byte limit
# first, and a control character costs six bytes as a `\uXXXX` escape).
MAX_GUIDE_MARKDOWN_CHARS = 16_000
MAX_GUIDE_MARKDOWN_BYTES = 24_000
MAX_GUIDE_TOPIC_CHARS = 200
MAX_GUIDE_CATALOG_CHARS = 60_000
GUIDE_HEADINGS_PER_DOCUMENT = 6
MAX_GUIDE_GLOSSARY_CHARS = 4_000
MAX_GUIDE_TYPES_CHARS = 8_000
MAX_GUIDE_FACETS_CHARS = 6_000
MAX_CITATION_FEEDBACK_TOKENS = 20
CITATION_FEEDBACK_SAMPLES = 15
MAX_CITATION_FEEDBACK_CHARS = 4_000
# An invalid token is quoted back cut to this length; a model can write any bracketed text.
CITATION_FEEDBACK_TOKEN_CHARS = 80
# Stated target types shown in one feedback request.
MAX_LESSON_TYPES_CHARS = 2_000

_FACET_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")
# A bracketed token that starts like a document ID; prose such as `[docx]` is not one.
_CITATION_TOKEN = re.compile(r"\[doc_[^\]]*\]")
# `[doc_id:section_id]`, the section ID belonging to the document ID before it.
_CITATION = re.compile(r"\[(doc_[0-9a-f]{10}):(\1_sec_\d+)\]")
_DOCUMENT_ID = re.compile(r"doc_[0-9a-f]{10}")
_WHITESPACE = re.compile(r"\s+")
_NAME_WORD = re.compile(r"[^\W\d_]{2,}")


# Validate and retry.


@dataclass(frozen=True)
class Checked[R]:
    """The verdict on one generated result.

    No rejections: `result` is complete. Rejections with a `result`: an accepted reduction
    (the caller has already counted what it dropped), used only after the rerun. Rejections
    without a `result`: structurally invalid.
    """

    rejections: tuple[str, ...]
    result: R | None

    @classmethod
    def accept(cls, result: R) -> Self:
        return cls((), result)

    @classmethod
    def reduce(cls, rejections: Sequence[str], result: R) -> Self:
        return cls(tuple(rejections), result)

    @classmethod
    def reject(cls, rejections: Sequence[str]) -> Self:
        if not rejections:
            raise ValueError("a rejection needs at least one reason")
        return cls(tuple(rejections), None)


def rejection_feedback(rejections: Sequence[str]) -> str:
    """The rerun's feedback: each rejection on one line cut to `MAX_REJECTION_CHARS`, as many
    as fit `MAX_REJECTION_FEEDBACK_CHARS`, then a count of the rest."""
    lines = [f"- {one_line(r, MAX_REJECTION_CHARS)}" for r in rejections]
    head, tail = "YOUR PREVIOUS ANSWER WAS REJECTED:", "Answer again and fix every point."
    listed = bounded_lines(
        lines, MAX_REJECTION_FEEDBACK_CHARS - len(head) - len(tail) - 2, "rejections"
    )
    return f"{head}\n{listed}\n{tail}"


def generate_checked[T, R](
    what: str, produce: Callable[[str | None], T], check: Callable[[T], Checked[R]]
) -> R:
    """Run `produce`, validate its output with `check`, and rerun once naming the rejections.

    `produce` receives the feedback text for the rerun (None on the first run) and must make
    it part of its request. After the rerun a reduced result is returned; a result that is
    still invalid raises `RejectedOutputError`.
    """
    checked = check(produce(None))
    if not checked.rejections:
        return _complete(checked)
    checked = check(produce(rejection_feedback(checked.rejections)))
    if not checked.rejections:
        return _complete(checked)
    if checked.result is not None:
        return checked.result
    raise RejectedOutputError(
        f"{what} still rejected after one rerun: {'; '.join(checked.rejections)}"
    )


def _complete[R](checked: Checked[R]) -> R:
    if checked.result is None:
        raise AssertionError("an accepted result must be present")
    return checked.result


class _NotStarted(Exception):
    """A queued task skipped because an earlier task had already failed."""


class InductionExecutor:
    """The one owner of induction concurrency: at most `workers` tasks run at once, and each
    task makes at most one provider call at a time, so `workers` bounds the provider calls in
    flight across binding, blueprints, glossary batches and definition checks.

    Only the thread that created the executor submits work and waits for it; a task never
    waits on the executor (a waiting task could hold the last worker). The first failure of
    any submitted task is recorded before its worker takes more work, so no task starts after
    it; waiting then raises that first failure.
    """

    def __init__(self, workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._submitted: list[Future[Any]] = []
        self._failure_lock = threading.Lock()
        self._first_failure: BaseException | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)

    def submit[R](self, fn: Callable[..., R], *args: Any) -> Future[R]:
        def guarded() -> R:
            with self._failure_lock:
                if self._first_failure is not None:
                    raise _NotStarted
            try:
                return fn(*args)
            except BaseException as error:
                with self._failure_lock:
                    if self._first_failure is None:
                        self._first_failure = error
                raise

        future = self._pool.submit(guarded)
        self._submitted.append(future)
        return future

    def results[R](self, futures: Sequence[Future[R]]) -> list[R]:
        """The futures' results in submission order, or the first failure of any task."""
        while True:
            with self._failure_lock:
                failure = self._first_failure
            if failure is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                raise failure
            if all(f.done() for f in futures):
                return [f.result() for f in futures]
            wait([f for f in self._submitted if not f.done()], return_when=FIRST_COMPLETED)

    def map[A, R](self, fn: Callable[[A], R], items: Sequence[A]) -> list[R]:
        """`fn` over `items`; results in submission order."""
        return self.results([self.submit(fn, item) for item in items])


# The corpus as the induction steps see it.


@dataclass(frozen=True)
class CorpusView:
    """Citable documents and their sections, keyed by ID. Curator notes are not citable.

    `documents` are in path order. Document IDs are path hashes, so every collection that
    reaches a prompt is ordered by `citable_order` or `section_order`, never by ID text.
    """

    documents: tuple[CorpusDocument, ...]
    citable: tuple[CorpusDocument, ...] = field(init=False)
    # Citable document ID to its position in path order.
    citable_order: Mapping[str, int] = field(init=False)
    sections: Mapping[str, tuple[CorpusDocument, ReadSection]] = field(init=False)
    folded_sections: Mapping[str, str] = field(init=False)

    def __post_init__(self) -> None:
        citable = tuple(d for d in self.documents if d.is_citable)
        sections = {d.section_id(s.ordinal): (d, s) for d in citable for s in d.reading.sections}
        object.__setattr__(self, "citable", citable)
        object.__setattr__(self, "citable_order", {d.doc_id: i for i, d in enumerate(citable)})
        object.__setattr__(self, "sections", sections)
        folded = {sid: fold_for_quote(self.markdown(sid)) for sid in sections}
        object.__setattr__(self, "folded_sections", folded)

    def section_order(self, section_id: str) -> tuple[int, int]:
        """A citable section's position: its document's path order, then its ordinal."""
        doc, section = self.sections[section_id]
        return self.citable_order[doc.doc_id], section.ordinal

    def document(self, doc_id: str) -> CorpusDocument:
        if doc_id not in self.citable_order:
            raise NotFoundError(f"unknown or non-citable document {doc_id!r}")
        return self.citable[self.citable_order[doc_id]]

    def markdown(self, section_id: str) -> str:
        if section_id not in self.sections:
            raise NotFoundError(f"unknown or non-citable section {section_id!r}")
        return self.sections[section_id][1].markdown

    def heading(self, section_id: str) -> str:
        return self.sections[section_id][1].heading_path

    def folders(self) -> tuple[str, ...]:
        """Every folder that holds a document, curator notes included, sorted."""
        found: set[str] = set()
        for doc in self.documents:
            parts = doc.rel_path.split("/")[:-1]
            found.update("/".join(parts[: i + 1]) for i in range(len(parts)))
        return tuple(sorted(found))

    def evidence(self, section_ids: Sequence[str]) -> tuple[str, ...]:
        """The complete cited sections as support-check evidence, in windows of at most
        `EVIDENCE_WINDOW_CHARS`; evidence that fits is one window."""
        blocks: list[str] = []
        for sid in section_ids:
            label = self.heading(sid)
            if len(label) > EVIDENCE_LABEL_CHARS:
                label = f"{label[:EVIDENCE_LABEL_CHARS]}..."
            text, header = self.markdown(sid), f"=== {sid}: {label}"
            if len(header) + len(text) + 5 <= EVIDENCE_WINDOW_CHARS:
                blocks.append(f"{header} ===\n{text}")
                continue
            # Room for the part header; consecutive parts overlap, so a statement cut at one
            # part's end is whole at the start of the next.
            size = EVIDENCE_WINDOW_CHARS - len(header) - 40
            starts = range(0, len(text), size - EVIDENCE_OVERLAP_CHARS)
            parts = [
                text[i : i + size]
                for i in starts
                if i == 0 or i + EVIDENCE_OVERLAP_CHARS < len(text)
            ]
            blocks += [
                f"{header} (part {n} of {len(parts)}) ===\n{part}"
                for n, part in enumerate(parts, start=1)
            ]
        windows: list[str] = []
        for block in blocks:
            if windows and len(windows[-1]) + 2 + len(block) <= EVIDENCE_WINDOW_CHARS:
                windows[-1] += f"\n\n{block}"
            else:
                windows.append(block)
        return tuple(windows)


def judge_support(
    gateway: LanguageGateway, operation: str, evidence: Sequence[str], predicates: Sequence[str]
) -> list[float | None]:
    """Each predicate's highest probability over the evidence windows; None when every window
    refused it."""
    best: list[float | None] = [None] * len(predicates)
    for window in evidence:
        answers = gateway.judge(operation, window, predicates)
        for index, probability in enumerate(answers):
            current = best[index]
            if probability is not None and (current is None or probability > current):
                best[index] = probability
    return best


# Bounded prompt material.


def one_line(text: str, limit: int) -> str:
    """`text` on one line, whitespace runs collapsed, cut to `limit` characters with `...`."""
    flat = _WHITESPACE.sub(" ", text).strip()
    return flat if len(flat) <= limit else f"{flat[: max(0, limit - 3)]}..."


@dataclass(frozen=True)
class NotesText:
    """Curator notes as one prompt shows them: in path order within
    `MAX_CURATOR_NOTES_PROMPT_CHARS`. `truncated` counts the notes not shown in full."""

    text: str
    truncated: int


def notes_text(notes: Sequence[CuratorNote]) -> NotesText:
    if not notes:
        return NotesText("(none)", 0)
    blocks: list[str] = []
    used = 0
    for index, note in enumerate(notes):
        block = f"### {note.scope_path}\n{note.content}"
        room = MAX_CURATOR_NOTES_PROMPT_CHARS - used
        if len(block) <= room:
            blocks.append(block)
            used += len(block) + 2
            continue
        left = len(notes) - index
        marker = f"\n[... cut; {left} curator note(s) not shown in full]"
        if room > len(marker) + 200:
            blocks.append(block[: room - len(marker)] + marker)
        else:
            blocks.append(marker.strip())
        return NotesText("\n\n".join(blocks), left)
    return NotesText("\n\n".join(blocks), 0)


def bounded_lines(lines: Sequence[str], limit: int, what: str) -> str:
    """`lines` joined while they fit `limit` characters, then a line counting the rest."""
    kept: list[str] = []
    used = 0
    for index, line in enumerate(lines):
        if used + len(line) + 1 > limit - 60:
            kept.append(f"({len(lines) - index} more {what} not listed)")
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept)


# Step 1: organise.

ORGANISE_INSTRUCTIONS = f"""\
You organise one curated document collection so that agents can filter it and
find the right prior deliverables. You receive the folders, the curators' notes
(their stated intent outranks your inference) and one digest line per document
of a batch: doc_id | path | format | date | observed type | title | summary |
stated attributes.

facets: 0 to {MAX_FACETS} useful ways to filter the collection, such as client, project,
author, campaign, audience, channel, region or document role. Fewer is fine,
including none. name is snake_case. Each facet has a closed list of canonical
values, each with a description and aliases (other spellings or abbreviations;
an alias may be used once per facet). A facet is either path-bound, when the
folder structure already encodes it: then every value lists the folders it stands
for, exactly as given; or content-bound: then no value lists folders and each
document is classified by reading it. Never mix the two in one facet. Prefer
facets that split the collection into groups of several documents.

documents: one entry for every doc_id of the batch, never more or fewer.
- deliverable_type: a generic, reusable name for the kind of deliverable the
  document is (for example "technical memorandum", "survey proposal", "campaign
  brief", "consultation response letter"). Name a kind even when the batch has
  only one document of it; never fold a distinct kind into another to save names.
- is_feedback: true when the document critiques, reviews, responds to comments on
  or records post-project lessons about other deliverables.
- feedback_for_types: when is_feedback, the deliverable_type names of the
  deliverables it reviews (at least one); otherwise empty. A feedback document
  still has its own deliverable_type.
"""


@dataclass(frozen=True)
class DocumentDigest:
    """One citable document as the organisation step sees it."""

    doc_id: str
    rel_path: str
    format: str
    # `DocumentReading.date_label`.
    date_label: str | None
    observed_type: str
    title: str
    summary: str
    # What the document states about itself, such as client or author.
    attributes: tuple[LocatedAttribute, ...]

    def line(self) -> str:
        stated = "; ".join(f"{a.name}: {a.value}" for a in self.attributes)
        return " | ".join(
            (
                self.doc_id,
                one_line(self.rel_path, DIGEST_PATH_CHARS),
                self.format,
                one_line(self.date_label or "no date", 40),
                one_line(self.observed_type, DIGEST_TYPE_CHARS),
                one_line(self.title, DIGEST_TITLE_CHARS),
                one_line(self.summary, DIGEST_SUMMARY_CHARS),
                one_line(stated or "-", DIGEST_ATTRIBUTE_CHARS),
            )
        )


def digest_batches(digests: Sequence[DocumentDigest]) -> list[tuple[DocumentDigest, ...]]:
    """Digests packed in path order into batches of at most `MAX_ORGANISE_BATCH_CHARS`."""
    batches: list[list[DocumentDigest]] = []
    size = 0
    for digest in digests:
        length = len(digest.line()) + 1
        if batches and size + length <= MAX_ORGANISE_BATCH_CHARS:
            batches[-1].append(digest)
            size += length
        else:
            batches.append([digest])
            size = length
    return [tuple(b) for b in batches]


def _folders_of(rel_paths: Iterable[str]) -> set[str]:
    return {
        "/".join(parts[: i + 1])
        for rel_path in rel_paths
        for parts in [rel_path.split("/")[:-1]]
        for i in range(len(parts))
    }


def batch_folders(
    folders: Sequence[str], digests: Sequence[DocumentDigest], citable_paths: Sequence[str]
) -> tuple[str, ...]:
    """The folders a batch lists, shallowest first, then in path order: those holding a
    document of the batch, and those holding no citable document at all (only curator notes),
    which every batch lists."""
    held = _folders_of(d.rel_path for d in digests)
    without_citable = set(folders) - _folders_of(citable_paths)
    return tuple(
        sorted(
            (f for f in folders if f in held or f in without_citable),
            key=lambda f: (f.count("/"), f),
        )
    )


@dataclass(frozen=True)
class OrganiseCorpus:
    operation: ClassVar[str] = "organise_corpus"
    output: ClassVar[type[OrganisationDraft]] = OrganisationDraft
    effort: ClassVar[Effort] = "medium"
    instructions: ClassVar[str] = ORGANISE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    folders: tuple[str, ...]
    notes: str
    digests: tuple[DocumentDigest, ...]
    # "batch 2 of 3"; the batches share the folders, notes and instructions.
    batch: str
    feedback: str | None = None

    def text(self) -> str:
        folders = (
            bounded_lines([f"- {f}/" for f in self.folders], MAX_FOLDER_LIST_CHARS, "folders")
            or "(all documents of this batch are in the root)"
        )
        digests = "\n".join(d.line() for d in self.digests)
        text = (
            f"FOLDERS:\n{folders}\n\nCURATOR NOTES:\n{self.notes}\n\n"
            f"DOCUMENTS ({self.batch}):\n{digests}"
        )
        return f"{text}\n\n{self.feedback}" if self.feedback else text


@dataclass(frozen=True)
class ProposedRole:
    """A batch's reading of one document, with type labels not yet canonical."""

    deliverable_type: str
    is_feedback: bool
    feedback_for_types: tuple[str, ...]


@dataclass(frozen=True)
class BatchOrganisation:
    facets: tuple[Facet, ...]
    roles: dict[str, ProposedRole]


def check_facets(
    drafts: Sequence[FacetDraft], folders: frozenset[str]
) -> tuple[list[Facet], list[str]]:
    """Facets validated by the folded identity facet filters resolve (`fold_facet_label`).

    Repetition within one value is normalised away: values and aliases are stripped, blank
    aliases dropped, and one alias kept per folded form. Ambiguity is a rejection: a value
    given twice, or one folded form naming two values of a facet.
    """
    rejections: list[str] = []
    if len(drafts) > MAX_FACETS:
        rejections.append(f"{len(drafts)} facets; give at most {MAX_FACETS}")
    facets: list[Facet] = []
    names: set[str] = set()
    for facet in drafts:
        if not _FACET_NAME.fullmatch(facet.name):
            rejections.append(f"facet name {facet.name!r} is not snake_case")
        if facet.name in names:
            rejections.append(f"facet {facet.name!r} is given twice")
        names.add(facet.name)
        if not facet.values:
            rejections.append(f"facet {facet.name!r} has no values")
        bound = [bool(v.folders) for v in facet.values]
        if any(bound) and not all(bound):
            rejections.append(
                f"facet {facet.name!r} mixes values with folders and values without;"
                " a facet is either path-bound or content-bound"
            )
        owners: dict[str, str] = {}
        values: list[FacetValue] = []
        for value in facet.values:
            canonical = value.value.strip()
            if not canonical:
                rejections.append(f"facet {facet.name!r} has a blank value")
                continue
            key = fold_facet_label(canonical)
            if owners.get(key) is not None and fold_facet_label(owners[key]) == key:
                rejections.append(f"facet {facet.name!r} gives the value {canonical!r} twice")
                continue
            aliases: dict[str, str] = {}
            stripped = {a.strip() for a in value.aliases if a.strip()}
            for alias in sorted(stripped, key=lambda a: (fold_facet_label(a), a)):
                if fold_facet_label(alias) != key:
                    aliases.setdefault(fold_facet_label(alias), alias)
            for folded in (key, *aliases):
                owner = owners.setdefault(folded, canonical)
                if owner != canonical:
                    rejections.append(
                        f"facet {facet.name!r}: {folded!r} names both {owner!r} and {canonical!r}"
                    )
            clean = tuple(sorted({f.strip("/") for f in value.folders}))
            for folder in clean:
                if folder not in folders:
                    rejections.append(
                        f"facet {facet.name!r} value {canonical!r}: unknown folder {folder!r}"
                    )
            values.append(
                FacetValue(
                    value=canonical,
                    description=value.description,
                    aliases=tuple(sorted(aliases.values())),
                    folders=clean,
                )
            )
        facets.append(
            Facet(
                name=facet.name,
                description=facet.description,
                binding="path" if bound and all(bound) else "content",
                values=tuple(values),
            )
        )
    return facets, rejections


def check_batch(
    draft: OrganisationDraft, batch_doc_ids: Sequence[str], folders: frozenset[str]
) -> Checked[BatchOrganisation]:
    """A batch's facets, and exactly one role for every document of the batch."""
    facets, rejections = check_facets(draft.facets, folders)
    wanted = set(batch_doc_ids)
    roles: dict[str, ProposedRole] = {}
    for entry in draft.documents:
        if entry.doc_id not in wanted:
            rejections.append(f"{entry.doc_id!r} is not a document of this batch")
            continue
        if entry.doc_id in roles:
            rejections.append(f"{entry.doc_id!r} is classified twice")
            continue
        label = entry.deliverable_type.strip()
        reviewed = tuple(dict.fromkeys(t.strip() for t in entry.feedback_for_types))
        if not label:
            rejections.append(f"{entry.doc_id!r} has a blank deliverable_type")
        if any(not t for t in reviewed):
            rejections.append(f"{entry.doc_id!r} names a blank feedback_for_types entry")
        if entry.is_feedback and not reviewed:
            rejections.append(
                f"{entry.doc_id!r} is feedback but names no feedback_for_types; name the types"
                " of deliverable it reviews"
            )
        if not entry.is_feedback and reviewed:
            rejections.append(f"{entry.doc_id!r} is not feedback but names feedback_for_types")
        roles[entry.doc_id] = ProposedRole(label, entry.is_feedback, reviewed)
    missing = [d for d in batch_doc_ids if d not in roles]
    if missing:
        rejections.append(f"no entry for the documents {', '.join(missing)}")
    if rejections:
        return Checked.reject(rejections)
    return Checked.accept(BatchOrganisation(tuple(facets), roles))


@dataclass(frozen=True)
class FacetProposal:
    """One distinct facet the organisation batches proposed, exactly as checked, and how many
    batches proposed it."""

    facet: Facet
    batches: int

    def lines(self, description_chars: int) -> list[str]:
        binding = "path-bound" if self.facet.binding == "path" else "content-bound"
        head = (
            f"- facet {self.facet.name} ({binding}; proposed by {self.batches} batch(es)):"
            f" {one_line(self.facet.description, max(description_chars, 60))}"
        )
        lines = [head]
        for value in self.facet.values:
            parts = [f"  - value {value.value!r}"]
            if description_chars:
                parts.append(one_line(value.description, description_chars))
            if value.aliases:
                parts.append(f"aliases: {'; '.join(map(repr, value.aliases))}")
            if value.folders:
                parts.append(f"folders: {json.dumps(list(value.folders), ensure_ascii=False)}")
            lines.append(" | ".join(parts))
        return lines


def facet_proposals(batches: Sequence[Sequence[Facet]]) -> tuple[FacetProposal, ...]:
    """Every distinct facet of the batches, in first-proposal order, with nothing dropped: the
    canonicalisation request sees each batch's names, bindings, values, aliases and folders."""
    counts: dict[Facet, int] = {}
    for facets in batches:
        for facet in dict.fromkeys(facets):
            counts[facet] = counts.get(facet, 0) + 1
    return tuple(FacetProposal(facet, n) for facet, n in counts.items())


def facet_proposal_text(proposals: Sequence[FacetProposal]) -> str:
    """Every proposal with every value, alias and folder, with descriptions as long as fit
    `MAX_FACET_PROPOSALS_CHARS`. Values are never left out: proposals whose values alone do not
    fit are refused."""
    for description_chars in FACET_DESCRIPTION_STEPS:
        text = "\n".join(line for p in proposals for line in p.lines(description_chars))
        if len(text) <= MAX_FACET_PROPOSALS_CHARS:
            return text
    raise ArkiveError(
        f"the organisation batches proposed {len(proposals)} distinct facets whose values do not"
        f" fit one canonicalisation request of {MAX_FACET_PROPOSALS_CHARS} characters"
    )


CANONICALISE_FACETS_INSTRUCTIONS = f"""\
You canonicalise the facets that readers of one document collection proposed in
separate batches, so that agents filter the whole collection with one set of at
most {MAX_FACETS} facets. You receive the curators' notes (their stated intent
outranks your inference) and every distinct facet a batch proposed, with its
binding, values, aliases, folders and how many batches proposed it.

Merge facets that filter by the same property under different names into one
facet with one snake_case name and description. Within a facet, merge values
that name the same thing (spelling, abbreviation or wording differences) into
one canonical value, and keep the other spellings as its aliases; keep values
apart when they name different things. An alias names only one value of a
facet. Keep every facet a curator note asks for. When more than {MAX_FACETS}
distinct facets remain, keep those that best split the collection into groups
of several documents, preferring those more batches proposed. Give no facets
when none is useful.

A facet is either path-bound, when every value lists the folders it stands for,
or content-bound, when no value lists a folder; never mix the two in one facet.
A path-bound value keeps every folder of the values merged into it, written
exactly as given. Use only folders the proposals list.
"""


@dataclass(frozen=True)
class CanonicaliseFacets:
    operation: ClassVar[str] = "canonicalise_facets"
    output: ClassVar[type[FacetCanonicalisationDraft]] = FacetCanonicalisationDraft
    effort: ClassVar[Effort] = "medium"
    instructions: ClassVar[str] = CANONICALISE_FACETS_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    notes: str
    proposals: tuple[FacetProposal, ...]
    feedback: str | None = None

    def text(self) -> str:
        text = (
            f"CURATOR NOTES:\n{self.notes}\n\n"
            f"PROPOSED FACETS:\n{facet_proposal_text(self.proposals)}"
        )
        return f"{text}\n\n{self.feedback}" if self.feedback else text


def check_canonical_facets(
    draft: FacetCanonicalisationDraft, folders: frozenset[str]
) -> Checked[tuple[Facet, ...]]:
    facets, rejections = check_facets(draft.facets, folders)
    if rejections:
        return Checked.reject(rejections)
    return Checked.accept(tuple(facets))


def canonicalise_facets(
    gateway: LanguageGateway, notes: str, batches: Sequence[Sequence[Facet]]
) -> tuple[Facet, ...]:
    """One facet set from every batch's proposals: one request whenever any batch proposed a
    facet, a single batch included, so synonyms within one batch merge too. A canonical value
    may list only folders some proposal lists."""
    proposals = facet_proposals(batches)
    if not proposals:
        return ()
    proposed_folders = frozenset(f for p in proposals for v in p.facet.values for f in v.folders)
    facets = generate_checked(
        "facet canonicalisation",
        lambda feedback: gateway.run(CanonicaliseFacets(notes, proposals, feedback)),
        lambda draft: check_canonical_facets(draft, proposed_folders),
    )
    return tuple(map(with_folder_aliases, facets))


CANONICALISE_INSTRUCTIONS = """\
You canonicalise the deliverable type labels that readers of one document
collection proposed in separate batches. Give one canonical type for each kind
of deliverable: a generic name, a one-sentence description, and every given
label that names that kind. Merge labels only when they name the same kind of
deliverable (spelling, wording or plural differences); keep kinds apart when
their purpose differs, and keep a kind with a single document as its own type.
Every given label must appear in exactly one type's labels, written exactly as
given.
"""


@dataclass(frozen=True)
class TypeLabel:
    label: str
    # Documents a batch gave this label as their deliverable type.
    documents: int
    examples: tuple[str, ...]
    # Titles of feedback documents a batch said review deliverables of this label, so a label
    # named only as a reviewed type still shows what it stands for.
    reviewed_in: tuple[str, ...] = ()

    def line(self, examples: int) -> str:
        shown = "; ".join(one_line(e, 60) for e in self.examples[:examples])
        reviews = "; ".join(one_line(e, 60) for e in self.reviewed_in[:examples])
        count = (
            f"{self.documents} document(s)" if self.documents else "named only as a reviewed type"
        )
        return (
            f"- {self.label} ({count}{'; e.g. ' + shown if shown else ''}"
            f"{'; reviewed in ' + reviews if reviews else ''})"
        )


def type_label_text(labels: Sequence[TypeLabel]) -> str:
    """Every label, with as many example titles as fit `MAX_TYPE_LABELS_CHARS`. Labels are never
    left out: a collection whose labels alone do not fit is refused."""
    for examples in range(TYPE_LABEL_EXAMPLES, -1, -1):
        text = "\n".join(label.line(examples) for label in labels)
        if len(text) <= MAX_TYPE_LABELS_CHARS:
            return text
    raise ArkiveError(
        f"the organisation batches proposed {len(labels)} distinct deliverable type labels,"
        f" more than one canonicalisation request of {MAX_TYPE_LABELS_CHARS} characters can list"
    )


@dataclass(frozen=True)
class CanonicaliseTypes:
    operation: ClassVar[str] = "canonicalise_types"
    output: ClassVar[type[TypeCanonicalisationDraft]] = TypeCanonicalisationDraft
    effort: ClassVar[Effort] = "medium"
    instructions: ClassVar[str] = CANONICALISE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    labels: tuple[TypeLabel, ...]
    feedback: str | None = None

    def text(self) -> str:
        text = f"TYPE LABELS:\n{type_label_text(self.labels)}"
        return f"{text}\n\n{self.feedback}" if self.feedback else text


@dataclass(frozen=True)
class TypeCanon:
    types: tuple[DeliverableType, ...]
    # Folded label to its canonical type name.
    canonical: dict[str, str]


def check_types(
    draft: TypeCanonicalisationDraft, labels: Sequence[TypeLabel]
) -> Checked[TypeCanon]:
    given = {fold_facet_label(t.label): t.label for t in labels}
    rejections: list[str] = []
    canonical: dict[str, str] = {}
    names: dict[str, str] = {}
    types: list[DeliverableType] = []
    for dtype in draft.types:
        name = dtype.name.strip()
        if not name:
            rejections.append("a type has a blank name")
            continue
        if fold_facet_label(name) in names:
            rejections.append(f"type {name!r} is given twice")
            continue
        names[fold_facet_label(name)] = name
        if not dtype.labels:
            rejections.append(f"type {name!r} lists no labels")
        for label in dtype.labels:
            key = fold_facet_label(label)
            if key not in given:
                rejections.append(f"type {name!r}: {label!r} is not a given label")
            elif key in canonical:
                rejections.append(f"label {given[key]!r} is in two types")
            else:
                canonical[key] = name
        types.append(DeliverableType(name=name, description=dtype.description.strip()))
    missing = [label for key, label in given.items() if key not in canonical]
    if missing:
        rejections.append(f"no type lists the labels {', '.join(map(repr, missing))}")
    if rejections:
        return Checked.reject(rejections)
    return Checked.accept(
        TypeCanon(tuple(sorted(types, key=lambda t: (fold_text(t.name), t.name))), canonical)
    )


@dataclass(frozen=True)
class Organisation:
    facets: tuple[Facet, ...]
    deliverable_types: tuple[DeliverableType, ...]
    # Citable document ID to its canonical role, without version fields.
    roles: dict[str, ProposedRole]
    truncated_notes: int


def organise(
    gateway: LanguageGateway,
    view: CorpusView,
    notes: Sequence[CuratorNote],
    pool: InductionExecutor,
) -> Organisation:
    """Organise the citable documents in digest batches, canonicalise the batches' facets into
    one set, and map every proposed type label to one canonical type. One path for every
    corpus size: a small corpus is one batch, and its facets and labels are canonicalised all
    the same."""
    folders = view.folders()
    known = frozenset(folders)
    shown_notes = notes_text(notes)
    digests = [
        DocumentDigest(
            d.doc_id,
            d.rel_path,
            d.reading.format,
            d.reading.date_label,
            d.reading.card.deliverable_type,
            d.reading.card.title,
            d.reading.card.summary,
            d.reading.card.attributes,
        )
        for d in view.citable
    ]
    batches = digest_batches(digests)

    def organise_batch(numbered: tuple[int, tuple[DocumentDigest, ...]]) -> BatchOrganisation:
        number, batch = numbered
        request = OrganiseCorpus(
            batch_folders(folders, batch, [d.rel_path for d in view.citable]),
            shown_notes.text,
            batch,
            f"batch {number} of {len(batches)}",
        )
        doc_ids = [d.doc_id for d in batch]
        return generate_checked(
            f"corpus organisation, batch {number} of {len(batches)}",
            lambda feedback: gateway.run(replace(request, feedback=feedback)),
            lambda draft: check_batch(draft, doc_ids, known),
        )

    proposals = pool.map(organise_batch, list(enumerate(batches, start=1)))
    roles = {doc_id: role for p in proposals for doc_id, role in p.roles.items()}
    labels: dict[str, TypeLabel] = {}
    for doc in view.citable:
        role = roles[doc.doc_id]
        for label in (role.deliverable_type, *role.feedback_for_types):
            labels.setdefault(fold_facet_label(label), TypeLabel(label, 0, ()))
        for label in role.feedback_for_types:
            reviewed = labels[fold_facet_label(label)]
            labels[fold_facet_label(label)] = replace(
                reviewed,
                reviewed_in=(*reviewed.reviewed_in, doc.reading.card.title)[:TYPE_LABEL_EXAMPLES],
            )
        key = fold_facet_label(role.deliverable_type)
        prior = labels[key]
        labels[key] = replace(
            prior,
            documents=prior.documents + 1,
            examples=(*prior.examples, doc.reading.card.title)[:TYPE_LABEL_EXAMPLES],
        )
    ordered = tuple(sorted(labels.values(), key=lambda t: (fold_text(t.label), t.label)))
    canon = (
        generate_checked(
            "deliverable type canonicalisation",
            lambda feedback: gateway.run(CanonicaliseTypes(ordered, feedback)),
            lambda draft: check_types(draft, ordered),
        )
        if ordered
        else TypeCanon((), {})
    )

    def canonical(label: str) -> str:
        return canon.canonical[fold_facet_label(label)]

    canonical_roles = {
        doc_id: ProposedRole(
            canonical(role.deliverable_type),
            role.is_feedback,
            tuple(sorted({canonical(t) for t in role.feedback_for_types}, key=fold_text)),
        )
        for doc_id, role in roles.items()
    }
    facets = canonicalise_facets(gateway, shown_notes.text, [p.facets for p in proposals])
    return Organisation(
        facets,
        canon.types,
        canonical_roles,
        shown_notes.truncated,
    )


def with_folder_aliases(facet: Facet) -> Facet:
    """A path-bound value also answers to the paths of its folders, unless a path already
    names a value of the facet or belongs to several values."""
    if facet.binding != "path":
        return facet
    claimed = {fold_facet_label(f) for v in facet.values for f in (v.value, *v.aliases)}
    owners = Counter(fold_facet_label(f) for v in facet.values for f in v.folders)
    values = []
    for value in facet.values:
        added = {
            f
            for f in value.folders
            if fold_facet_label(f) not in claimed and owners[fold_facet_label(f)] == 1
        }
        values.append(value.model_copy(update={"aliases": tuple(sorted({*value.aliases, *added}))}))
    return facet.model_copy(update={"values": tuple(values)})


# Step 2: bind.

BIND_OPERATION = "bind_facets"


def binding_evidence(doc: CorpusDocument) -> str:
    """What one binding question reads about a document: its card and a bounded outline."""
    card = doc.reading.card
    outline = bounded_lines(
        [
            f"- {heading_context(s.heading_path)} (pp. {s.page_start}-{s.page_end})"
            for s in doc.reading.sections
        ],
        MAX_BINDING_OUTLINE_CHARS,
        "sections",
    )
    attributes = bounded_lines(
        [f"- {one_line(f'{a.name}: {a.value}', 600)}" for a in card.attributes],
        MAX_BINDING_ATTRIBUTES_CHARS,
        "attributes",
    )
    date = doc.reading.date_label or "not stated"
    return (
        f"PATH: {one_line(doc.rel_path, 500)}\nTITLE: {one_line(card.title, 300)}\n"
        f"DATE: {one_line(date, 40)}\nDELIVERABLE TYPE: {one_line(card.deliverable_type, 200)}\n"
        f"SUMMARY: {one_line(card.summary, 2_000)}\n"
        f"ATTRIBUTES:\n{attributes or '(none)'}\nOUTLINE:\n{outline}"
    )


def binding_predicate(facet: Facet, value: FacetValue) -> str:
    return (
        f'This document has {facet.name} "{value.value}".'
        f" {facet.name}: {one_line(facet.description, MAX_BINDING_DESCRIPTION_CHARS)}"
        f" {value.value}: {one_line(value.description, MAX_BINDING_DESCRIPTION_CHARS)}"
    )


@dataclass(frozen=True)
class Binding:
    facets: tuple[Facet, ...]
    assignments: dict[str, dict[str, tuple[str, ...]]]
    refusals: int
    pruned_values: tuple[str, ...]
    pruned_facets: tuple[str, ...]


def bind_facets(
    gateway: LanguageGateway, view: CorpusView, facets: Sequence[Facet], pool: InductionExecutor
) -> Binding:
    """Assign facet values to documents, then prune what no document holds.

    Path-bound values go to every document under their folders, curator notes included.
    Content-bound values are judged per citable document; a refused question assigns nothing.
    A value no document holds is pruned, then a facet with fewer than two populated values.
    """
    held: dict[str, dict[str, set[str]]] = {
        d.doc_id: {f.name: set() for f in facets} for d in view.documents
    }
    for facet in facets:
        if facet.binding != "path":
            continue
        for value in facet.values:
            for doc in view.documents:
                if any(doc.rel_path.startswith(f"{folder}/") for folder in value.folders):
                    held[doc.doc_id][facet.name].add(value.value)
    questions = [(f, v) for f in facets if f.binding == "content" for v in f.values]
    predicates = [binding_predicate(f, v) for f, v in questions]

    def judge(doc: CorpusDocument) -> list[float | None]:
        return gateway.judge(BIND_OPERATION, binding_evidence(doc), predicates)

    answers = pool.map(judge, view.citable if questions else ())
    refusals = 0
    for doc, probabilities in zip(view.citable if questions else (), answers, strict=True):
        for (facet, value), probability in zip(questions, probabilities, strict=True):
            if probability is None:
                refusals += 1
            elif probability >= SUPPORT_THRESHOLD:
                held[doc.doc_id][facet.name].add(value.value)
    kept: list[Facet] = []
    pruned_values: list[str] = []
    pruned_facets: list[str] = []
    for facet in facets:
        populated = {v for doc in held.values() for v in doc[facet.name]}
        values = tuple(v for v in facet.values if v.value in populated)
        pruned_values += [
            f"{facet.name}={v.value}" for v in facet.values if v.value not in populated
        ]
        if len(values) < 2:
            pruned_facets.append(facet.name)
            continue
        kept.append(facet.model_copy(update={"values": values}))
    assignments = {
        doc_id: {f.name: tuple(sorted(values[f.name])) for f in kept}
        for doc_id, values in held.items()
    }
    return Binding(tuple(kept), assignments, refusals, tuple(pruned_values), tuple(pruned_facets))


# Step 3: version families.


def _sampled_grams(doc: CorpusDocument) -> frozenset[int]:
    text = "\n".join(_passages.without_asides(s.markdown) for s in doc.reading.sections)
    # A checksum, not `hash`: the sample decides which pairs reach a prompt and a cache key.
    checksums = (zlib.crc32(" ".join(g).encode()) for g in _passages.shingles(fold_for_quote(text)))
    return frozenset(c for c in checksums if c % VERSION_SHINGLE_SAMPLE == 0)


def _name_words(doc: CorpusDocument) -> frozenset[str]:
    stem = doc.rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return frozenset(_NAME_WORD.findall(fold_text(f"{stem} {doc.reading.card.title}")))


def _distinctive[F](features: Sequence[frozenset[F]]) -> list[frozenset[F]]:
    """Each document's features without those held by more than `VERSION_COMMON_POSTING`
    documents: house text shared across many deliverables says nothing about versions."""
    holders = Counter(f for held in features for f in held)
    return [frozenset(f for f in held if holders[f] <= VERSION_COMMON_POSTING) for held in features]


def _overlaps[F](features: Sequence[frozenset[F]]) -> list[Counter[int]]:
    """For each document, how many features it shares with each other document, through an
    inverted index."""
    postings: dict[F, list[int]] = {}
    for index, held in enumerate(features):
        for feature in held:
            postings.setdefault(feature, []).append(index)
    shared: list[Counter[int]] = [Counter() for _ in features]
    for holders in postings.values():
        for i in holders:
            shared[i].update(j for j in holders if j != i)
    return shared


def version_candidates(
    view: CorpusView, roles: Mapping[str, ProposedRole]
) -> list[tuple[str, str, float]]:
    """Scored pairs of citable documents that may be versions of one deliverable, as
    `(doc_a, doc_b, score)` with `doc_a` first in path order, sorted by path order.

    Only the first of byte-identical documents takes part. Features held by more than
    `VERSION_COMMON_POSTING` documents count in neither the shared part nor the whole of a
    score. A pair is proposed when one document holds at least `VERSION_CANDIDATE_CONTAINMENT`
    of the other's distinctive sampled word 4-grams (each having `VERSION_MIN_GRAMS` or more),
    or when two documents of one deliverable type share at least `VERSION_TITLE_OVERLAP` of
    their distinctive title and file-name words (two words or more each); a document with too
    few distinctive 4-grams is paired by its words alone. A feedback document and a document
    that is not feedback are never paired: a review quoting what it reviews is another
    artifact. Each document keeps its `MAX_VERSION_PARTNERS` best partners, so the pairs grow
    linearly.
    """
    firsts: dict[str, CorpusDocument] = {}
    for d in view.citable:
        firsts.setdefault(d.reading.sha256, d)
    docs = list(firsts.values())
    grams = _distinctive([_sampled_grams(d) for d in docs])
    words = _distinctive([_name_words(d) for d in docs])
    shared_grams, shared_words = _overlaps(grams), _overlaps(words)
    kept: dict[tuple[int, int], float] = {}
    for i, doc in enumerate(docs):
        scores: dict[int, float] = {}
        feedback = roles[doc.doc_id].is_feedback
        for j, count in shared_grams[i].items():
            if roles[docs[j].doc_id].is_feedback != feedback:
                continue
            if min(len(grams[i]), len(grams[j])) < VERSION_MIN_GRAMS:
                continue
            score = max(count / len(grams[i]), count / len(grams[j]))
            if score >= VERSION_CANDIDATE_CONTAINMENT:
                scores[j] = score
        for j, count in shared_words[i].items():
            if roles[docs[j].doc_id].is_feedback != feedback:
                continue
            same_type = roles[doc.doc_id].deliverable_type == roles[docs[j].doc_id].deliverable_type
            if same_type and min(len(words[i]), len(words[j])) >= 2:
                score = count / len(words[i] | words[j])
                if score >= VERSION_TITLE_OVERLAP:
                    scores[j] = max(scores.get(j, 0.0), score)
        for j in sorted(scores, key=lambda j: (-scores[j], j))[:MAX_VERSION_PARTNERS]:
            kept[(min(i, j), max(i, j))] = scores[j]
    return [(docs[i].doc_id, docs[j].doc_id, kept[(i, j)]) for i, j in sorted(kept)]


@dataclass(frozen=True)
class VersionComponents:
    # Each component's documents in path order, components in the path order of their first.
    components: tuple[tuple[str, ...], ...]
    # Each component's kept candidate pairs, in the order `version_candidates` gives them.
    edges: tuple[tuple[tuple[str, str, float], ...], ...]
    dropped_edges: int


def version_components(
    order: Mapping[str, int], edges: Sequence[tuple[str, str, float]]
) -> VersionComponents:
    """The connected components of two or more documents of the candidate pairs, none larger
    than `MAX_VERSION_COMPONENT_DOCS`. Pairs join documents strongest first (ties in path order
    of the pair); a pair that would join two components into one over the bound is dropped and
    counted, so only the weakest links of a large cluster are cut. A document every pair of
    which was dropped is in no component."""
    parent: dict[str, str] = {}
    size: dict[str, int] = {}

    def root(doc_id: str) -> str:
        parent.setdefault(doc_id, doc_id)
        size.setdefault(doc_id, 1)
        while parent[doc_id] != doc_id:
            doc_id = parent[doc_id]
        return doc_id

    kept: list[tuple[str, str, float]] = []
    dropped = 0
    for a, b, score in sorted(edges, key=lambda e: (-e[2], order[e[0]], order[e[1]])):
        ra, rb = root(a), root(b)
        if ra != rb:
            if size[ra] + size[rb] > MAX_VERSION_COMPONENT_DOCS:
                dropped += 1
                continue
            first, second = sorted((ra, rb), key=order.__getitem__)
            parent[second] = first
            size[first] += size[second]
        kept.append((a, b, score))
    members: dict[str, list[str]] = {}
    for doc_id in sorted(parent, key=order.__getitem__):
        members.setdefault(root(doc_id), []).append(doc_id)
    components = sorted((m for m in members.values() if len(m) > 1), key=lambda m: order[m[0]])
    position = {doc_id: n for n, m in enumerate(components) for doc_id in m}
    kept_by: list[list[tuple[str, str, float]]] = [[] for _ in components]
    for edge in sorted(kept, key=lambda e: (order[e[0]], order[e[1]])):
        kept_by[position[edge[0]]].append(edge)
    return VersionComponents(
        tuple(tuple(m) for m in components), tuple(tuple(e) for e in kept_by), dropped
    )


def version_date(reading: DocumentReading) -> str:
    """The date a version card shows, with where it came from."""
    if reading.date is not None and reading.date.basis == "printed":
        return f"{reading.date.iso} (printed)"
    if reading.card.doc_date:
        return f"{one_line(reading.card.doc_date, 40)} (printed, not parsed)"
    if reading.date is not None:
        return f"{reading.date.iso} (file metadata)"
    return "not stated"


@dataclass(frozen=True)
class VersionCard:
    """One document as a version partition request shows it, every field bounded."""

    doc_id: str
    rel_path: str
    format: str
    pages: int
    date: str
    observed_type: str
    attributes: tuple[LocatedAttribute, ...]
    title: str
    summary: str
    headings: tuple[str, ...]

    @classmethod
    def of(cls, doc: CorpusDocument) -> VersionCard:
        card = doc.reading.card
        return cls(
            doc.doc_id,
            doc.rel_path,
            doc.reading.format,
            len(doc.reading.pages),
            version_date(doc.reading),
            card.deliverable_type,
            card.attributes,
            card.title,
            card.summary,
            tuple(s.heading_path for s in doc.reading.sections[:VERSION_EVIDENCE_HEADINGS]),
        )

    def render(self) -> str:
        stated = "; ".join(f"{a.name}: {a.value}" for a in self.attributes)
        headings = "\n".join(
            f"- {one_line(heading_context(h), VERSION_HEADING_CHARS)}" for h in self.headings
        )
        return (
            f"=== {self.doc_id} ===\nPath: {one_line(self.rel_path, 300)}\n"
            f"Format: {self.format} | Pages: {self.pages} | Date: {self.date}\n"
            f"Observed type: {one_line(self.observed_type, 200)}\n"
            f"Attributes: {one_line(stated or '-', VERSION_ATTRIBUTE_CHARS)}\n"
            f"Title: {one_line(self.title, 300)}\n"
            f"Summary: {one_line(self.summary, VERSION_SUMMARY_CHARS)}\n"
            f"First headings:\n{headings or '- (none)'}"
        )


PARTITION_VERSIONS_INSTRUCTIONS = """\
You decide which documents of one document collection are versions of one
another. You receive compact cards of a few documents that share text or title
words, and the candidate pairs code found with their overlap score (the share
of distinctive word 4-grams or title words they share). Overlap only proposes a
pair; shared templates and boilerplate also overlap.

Two documents belong in one family only when they are drafts, revisions or
format exports of the exact same deliverable artifact, for the exact same
engagement and reporting period. Never join distinct projects, sites or clients;
distinct phases or stages of one engagement (phase 1 and phase 2); periodic
installments for different periods (one month's or quarter's issue and the
next); an addendum or supplement and the base document it adds to; or a summary
deck and the full report it summarises.

families: every doc_id of the cards in exactly one family, written exactly as
given; a document with no other version is a family of its own.
latest_doc_id: the member that clearly supersedes every other member of its
family, for example a final over its drafts, a higher revision number, or an
unambiguously later printed issue date of the same deliverable. A date marked
"(file metadata)" records when a file was saved or exported, not when the
deliverable was issued: it never proves that one version supersedes another,
and a newer file date never makes a draft supersede a final. When the order is
ambiguous, give null; never guess. For a family of one, give null or its doc_id.
"""


@dataclass(frozen=True)
class PartitionVersions:
    operation: ClassVar[str] = "partition_versions"
    output: ClassVar[type[VersionPartitionDraft]] = VersionPartitionDraft
    effort: ClassVar[Effort] = "medium"
    instructions: ClassVar[str] = PARTITION_VERSIONS_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    # The component's documents in path order.
    cards: tuple[VersionCard, ...]
    overlaps: tuple[tuple[str, str, float], ...]
    feedback: str | None = None

    def text(self) -> str:
        cards = "\n\n".join(c.render() for c in self.cards)
        pairs = "\n".join(f"- {a} ~ {b}: {score:.2f}" for a, b, score in self.overlaps)
        text = f"DOCUMENTS:\n{cards}\n\nCANDIDATE PAIRS:\n{pairs}"
        return f"{text}\n\n{self.feedback}" if self.feedback else text


@dataclass(frozen=True)
class PartitionedFamily:
    # Members in path order.
    doc_ids: tuple[str, ...]
    latest: str | None


def check_partition(
    draft: VersionPartitionDraft, component: Sequence[str]
) -> Checked[tuple[PartitionedFamily, ...]]:
    """Every document of the component in exactly one family, and a latest member only from
    its own family."""
    order = {doc_id: n for n, doc_id in enumerate(component)}
    rejections: list[str] = []
    placed: set[str] = set()
    families: list[PartitionedFamily] = []
    for family in draft.families:
        if not family.doc_ids:
            rejections.append("a family lists no documents")
            continue
        members: list[str] = []
        for doc_id in family.doc_ids:
            if doc_id not in order:
                rejections.append(f"{doc_id!r} is not a document of these cards")
            elif doc_id in placed:
                rejections.append(f"{doc_id!r} is placed in more than one family")
            else:
                placed.add(doc_id)
                members.append(doc_id)
        latest = family.latest_doc_id
        if latest is not None and latest not in family.doc_ids:
            rejections.append(f"latest_doc_id {latest!r} is not a member of its family")
        if members:
            families.append(
                PartitionedFamily(tuple(sorted(members, key=order.__getitem__)), latest)
            )
    missing = [d for d in component if d not in placed]
    if missing:
        rejections.append(f"no family lists the documents {', '.join(missing)}")
    if rejections:
        return Checked.reject(rejections)
    return Checked.accept(tuple(sorted(families, key=lambda f: order[f.doc_ids[0]])))


@dataclass(frozen=True)
class VersionFamilies:
    # Citable document ID to (family ID, status, superseded by).
    members: dict[str, tuple[str, VersionStatus, str | None]]
    dropped_edges: int


def version_families(
    gateway: LanguageGateway,
    view: CorpusView,
    roles: Mapping[str, ProposedRole],
    pool: InductionExecutor,
) -> VersionFamilies:
    """Partition the citable documents into version families and order each one.

    The first of byte-identical documents in path order represents them. Candidate pairs of
    representatives form bounded components (`version_components`); each component of two or
    more is partitioned by one `PartitionVersions` request. A family's ID is its first member
    in path order, copies included. A family of one document is `sole`. When the request
    names a latest member, or a family's only representative has copies, that member is
    `latest` and every other member, copies included, is `superseded` by it in one step;
    otherwise every representative is `unordered` and a copy is `superseded` by its first
    copy.
    """
    copies: dict[str, list[str]] = {}
    representative: dict[str, str] = {}
    for doc in view.citable:
        first = representative.setdefault(doc.reading.sha256, doc.doc_id)
        copies.setdefault(first, []).append(doc.doc_id)
    found = version_components(view.citable_order, version_candidates(view, roles))
    by_component = dict(zip(found.components, found.edges, strict=True))

    def partition(component: tuple[str, ...]) -> tuple[PartitionedFamily, ...]:
        request = PartitionVersions(
            tuple(VersionCard.of(view.document(d)) for d in component), by_component[component]
        )
        return generate_checked(
            f"version partition of {', '.join(component)}",
            lambda feedback: gateway.run(replace(request, feedback=feedback)),
            lambda draft: check_partition(draft, component),
        )

    partitions = pool.map(partition, found.components)
    families = [f for p in partitions for f in p]
    joined = {d for f in families for d in f.doc_ids}
    families += [PartitionedFamily((d,), None) for d in copies if d not in joined]
    result: dict[str, tuple[str, VersionStatus, str | None]] = {}
    for family in families:
        members = sorted(
            (m for d in family.doc_ids for m in copies[d]), key=view.citable_order.__getitem__
        )
        family_id = members[0]
        if len(members) == 1:
            result[family_id] = (family_id, "sole", None)
            continue
        latest = family.doc_ids[0] if len(family.doc_ids) == 1 else family.latest
        for rep in family.doc_ids:
            for doc_id in copies[rep]:
                if latest is not None:
                    status: VersionStatus = "latest" if doc_id == latest else "superseded"
                    result[doc_id] = (family_id, status, None if doc_id == latest else latest)
                elif doc_id == rep:
                    result[doc_id] = (family_id, "unordered", None)
                else:
                    result[doc_id] = (family_id, "superseded", rep)
    return VersionFamilies(result, found.dropped_edges)


# Step 4: feedback lessons.

LESSON_INSTRUCTIONS = """\
You read one feedback document of a collection: a review, a comment response or a
post-project record about other deliverables. Extract the transferable writing
and review lessons it records: criticisms a reviewer raised that an author
accepted or resolved, and the resolution, stated generically so that they apply
to a future deliverable of the reviewed types. Leave out items noted with no
change, rejected or out of scope. Leave out every project-specific value: client,
site and person names, places, fees, coordinates and project dates. Keep a
house-wide constant when the reviewer establishes or enforces it as a rule that
holds beyond this project: a standard reference value or equipment setting, a
standard period or duration, a standard threshold or tolerance, or a regulatory
or methodology clause and its required value. For each lesson give a short
statement and a quote: verbatim text of the sections that shows it. Give an
empty list when the sections hold none.
"""


@dataclass(frozen=True)
class LessonSection:
    section_id: str
    heading: str
    markdown: str

    def render(self) -> str:
        return f"{_lesson_label(self.section_id, self.heading)}{self.markdown}"


def _lesson_label(section_id: str, heading: str) -> str:
    return f"=== {section_id}: {heading_context(heading)} ===\n"


@dataclass(frozen=True)
class ExtractFeedbackLessons:
    operation: ClassVar[str] = "extract_feedback_lessons"
    output: ClassVar[type[FeedbackLessonsDraft]] = FeedbackLessonsDraft
    effort: ClassVar[Effort] = "low"
    instructions: ClassVar[str] = LESSON_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    doc_title: str
    target_types: tuple[str, ...]
    sections: tuple[LessonSection, ...]
    feedback: str | None = None

    def text(self) -> str:
        sections = "\n\n".join(s.render() for s in self.sections)
        text = (
            f"DOCUMENT: {one_line(self.doc_title, 300)}\n"
            f"REVIEWS DELIVERABLES OF TYPE:"
            f" {one_line(', '.join(self.target_types), MAX_LESSON_TYPES_CHARS)}\n\n"
            f"SECTIONS:\n{sections}"
        )
        return f"{text}\n\n{self.feedback}" if self.feedback else text


_TABLE_SEPARATOR = re.compile(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?")


def _split_markdown(markdown: str, limit: int) -> list[str]:
    """`markdown` cut at line breaks into pieces of at most `limit` characters, its lines in
    order. A piece that starts inside a table (with or without outer pipes) repeats the table's
    header and separator rows when they leave room for the next row, so every row stays
    readable under its columns. A single line longer than a piece is cut as text into pieces
    of its own."""
    pieces: list[str] = []
    lines: list[str] = []
    size = 0
    header: list[str] = []
    body_from = 0
    all_lines = markdown.split("\n")

    def flush() -> None:
        nonlocal size
        if lines:
            pieces.append("\n".join(lines))
        lines.clear()
        size = 0

    for index, line in enumerate(all_lines):
        if "|" not in line:
            header = []
        elif (
            not header
            and index + 1 < len(all_lines)
            and _TABLE_SEPARATOR.fullmatch(all_lines[index + 1].strip())
        ):
            header, body_from = [line, all_lines[index + 1]], index + 2
        if len(line) > limit:
            flush()
            pieces += [line[i : i + limit] for i in range(0, len(line), limit)]
            continue
        if lines and size + 1 + len(line) > limit:
            flush()
        # Every piece that starts on a table row, after a full piece or after a row cut as
        # text, starts with the table's header.
        repeat = "\n".join(header) if header and index >= body_from else ""
        if not lines and repeat and len(repeat) + 1 + len(line) <= limit:
            lines.append(repeat)
            size = len(repeat)
        size += len(line) + (1 if lines else 0)
        lines.append(line)
    flush()
    return pieces


def feedback_batches(doc: CorpusDocument) -> list[tuple[LessonSection, ...]]:
    """A feedback document's sections packed in order into batches whose rendered sections
    (`LessonSection.render`, joined by blank lines) are at most `FEEDBACK_BATCH_CHARS`; a larger
    section is split by `_split_markdown`, leaving room for each piece's label."""
    batches: list[list[LessonSection]] = []
    size = 0
    for section in doc.reading.sections:
        sid = doc.section_id(section.ordinal)
        label = _lesson_label(sid, section.heading_path)
        for piece in _split_markdown(section.markdown, FEEDBACK_BATCH_CHARS - len(label)):
            item = LessonSection(sid, section.heading_path, piece)
            cost = len(item.render())
            if batches and size + 2 + cost <= FEEDBACK_BATCH_CHARS:
                batches[-1].append(item)
                size += 2 + cost
            else:
                batches.append([item])
                size = cost
    return [tuple(b) for b in batches]


@dataclass(frozen=True)
class LocatedLessons:
    lessons: tuple[tuple[str, str], ...]  # (statement, section ID)
    discarded: int


def locate_lessons(
    draft: FeedbackLessonsDraft, sections: Sequence[LessonSection], *, rerun_of_unlocated: bool
) -> Checked[LocatedLessons]:
    """Locate each lesson's quote in the batch. A lesson with a blank statement or a quote not
    found is a rejection; after the rerun it is discarded and counted. As in
    `compiler.locate_batch`, a batch that proposed lessons and located none is a broken reading,
    and so is a rerun of one (`rerun_of_unlocated`) that proposes no lesson."""
    if rerun_of_unlocated and not draft.lessons:
        return Checked.reject(
            [
                "the earlier answer gave lessons but none of them could be located, and this"
                " answer gives none"
            ]
        )
    located: list[tuple[str, str]] = []
    rejections: list[str] = []
    for lesson in draft.lessons:
        if not lesson.statement.strip():
            rejections.append(f"the lesson quoting {lesson.quote!r} has a blank statement")
            continue
        needle = fold_for_quote(lesson.quote)
        holder = next(
            (s.section_id for s in sections if needle and needle in fold_for_quote(s.markdown)),
            None,
        )
        if holder is None:
            rejections.append(f"quote not found verbatim in the sections: {lesson.quote!r}")
        else:
            located.append((lesson.statement.strip(), holder))
    result = LocatedLessons(tuple(dict.fromkeys(located)), len(rejections))
    if not rejections:
        return Checked.accept(result)
    if located:
        return Checked.reduce(rejections, result)
    return Checked.reject(rejections)


LESSON_SUPPORT_OPERATION = "check_feedback_lessons"


def lesson_predicate(statement: str) -> str:
    return (
        "The text records a reviewer point that the authors accepted or resolved, not one they"
        " rejected, left unchanged or ruled out of scope, and this lesson states what was"
        f' accepted: "{statement}"'
    )


@dataclass(frozen=True)
class LessonResult:
    lessons: tuple[FeedbackLesson, ...]
    discarded: int
    dropped: int
    refusals: int


def extract_feedback_lessons(
    gateway: LanguageGateway, view: CorpusView, doc: CorpusDocument, role: ProposedRole
) -> LessonResult:
    """The confirmed lessons of one feedback document. Each batch is one structured request;
    the support check judges a section's lessons together, one request per evidence window."""
    title = doc.reading.card.title
    located: list[tuple[str, str]] = []
    discarded = 0

    def read(batch: tuple[LessonSection, ...]) -> LocatedLessons:
        nothing_located = False

        def check(draft: FeedbackLessonsDraft) -> Checked[LocatedLessons]:
            nonlocal nothing_located
            checked = locate_lessons(draft, batch, rerun_of_unlocated=nothing_located)
            nothing_located = checked.result is None
            return checked

        return generate_checked(
            f"{doc.rel_path}: feedback lessons",
            lambda feedback: gateway.run(
                ExtractFeedbackLessons(title, role.feedback_for_types, batch, feedback)
            ),
            check,
        )

    for batch in feedback_batches(doc):
        found = read(batch)
        located += found.lessons
        discarded += found.discarded
    by_section: dict[str, list[str]] = {}
    for statement, section_id in dict.fromkeys(located):
        by_section.setdefault(section_id, []).append(statement)
    lessons: list[FeedbackLesson] = []
    dropped = refusals = 0
    for section_id in sorted(by_section, key=view.section_order):
        statements = by_section[section_id]
        verdicts = judge_support(
            gateway,
            LESSON_SUPPORT_OPERATION,
            view.evidence([section_id]),
            [lesson_predicate(s) for s in statements],
        )
        for statement, probability in zip(statements, verdicts, strict=True):
            refusals += probability is None
            if probability is not None and probability >= SUPPORT_THRESHOLD:
                lessons.append(
                    FeedbackLesson(
                        statement=statement,
                        source_doc_id=doc.doc_id,
                        section_id=section_id,
                        doc_title=title,
                        target_types=role.feedback_for_types,
                    )
                )
            else:
                dropped += 1
    return LessonResult(tuple(lessons), discarded, dropped, refusals)


# Step 5: glossary.

GLOSSARY_INSTRUCTIONS = """\
You write glossary entries for terms a document collection uses. Each term comes
with the quotes where documents mention it, tagged with section IDs, and further
occurrence snippets (shown case-folded). Write one entry per given term: the
term, its kind, its aliases, and its senses. A sense has a definition only when
the cited sections establish what the term means; otherwise definition is null,
which records that the corpus uses the term without defining it. When documents
use the term with conflicting meanings, give one sense per meaning. Cite every
sense with the section IDs (for example doc_a1b2c3d4e5_sec_002) where the term occurs
with that meaning; use only the IDs shown. Never add knowledge the sections do
not contain.
"""


@dataclass(frozen=True)
class TermQuote:
    section_id: str
    text: str
    defines: bool


@dataclass(frozen=True)
class TermOccurrence:
    section_id: str
    snippet: str


@dataclass(frozen=True)
class TermCluster:
    """Observations of one term, merged by folded term and alias."""

    term: str
    kind: str
    aliases: tuple[str, ...]
    quotes: tuple[TermQuote, ...]
    occurrences: tuple[TermOccurrence, ...]

    @property
    def forms(self) -> frozenset[str]:
        return frozenset(f for f in map(fold_for_quote, (self.term, *self.aliases)) if f)

    def render(self) -> str:
        shown = [one_line(a, MAX_TERM_ALIAS_CHARS) for a in self.aliases[:MAX_TERM_ALIASES_SHOWN]]
        more = len(self.aliases) - len(shown)
        aliases = one_line(
            ", ".join(shown) + (f" and {more} more" if more > 0 else ""), MAX_TERM_ALIASES_CHARS
        )
        kind = one_line(self.kind, MAX_TERM_KIND_CHARS)
        lines = [f"TERM: {one_line(self.term, 200)} (kind: {kind}; aliases: {aliases})"]
        lines += [
            f"- quote [{q.section_id}]{' (defining)' if q.defines else ''}:"
            f" {one_line(q.text, MAX_TERM_QUOTE_CHARS)}"
            for q in self.quotes
        ]
        lines += [f"- occurrence [{o.section_id}]: {o.snippet}" for o in self.occurrences]
        return "\n".join(lines)


@dataclass(frozen=True)
class WriteGlossary:
    operation: ClassVar[str] = "write_glossary"
    output: ClassVar[type[GlossaryDraft]] = GlossaryDraft
    effort: ClassVar[Effort] = "medium"
    instructions: ClassVar[str] = GLOSSARY_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    clusters: tuple[TermCluster, ...]
    feedback: str | None = None

    def text(self) -> str:
        text = "TERMS:\n\n" + "\n\n".join(c.render() for c in self.clusters)
        return f"{text}\n\n{self.feedback}" if self.feedback else text


def _find_form(folded_text: str, form: str) -> re.Match[str] | None:
    """The first whole-word occurrence of `form` in `folded_text`; both are folded."""
    return re.search(rf"(?<!\w){re.escape(form)}(?!\w)", folded_text)


def canonical_aliases(term: str, forms: Sequence[str]) -> tuple[str, ...]:
    """`forms` as aliases of `term`: stripped, blanks dropped, one per folded form, none folding
    like the term, in (`fold_text`, text) order, a total order over strings."""
    chosen: dict[str, str] = {}
    term_form = fold_for_quote(term)
    stripped = {f.strip() for f in forms if f.strip()}
    for alias in sorted(stripped, key=lambda a: (fold_text(a), a)):
        folded = fold_for_quote(alias)
        if folded and folded != term_form:
            chosen.setdefault(folded, alias)
    return tuple(chosen.values())


def term_clusters(view: CorpusView) -> list[TermCluster]:
    """Term observations merged by folded term and alias, kept when seen in two or more
    sections or quoted where a document defines them, with occurrence snippets from other
    citable sections. A term keeps `MAX_QUOTES_PER_TERM` quotes, defining quotes first, each
    group in path order, shown in path order. Sorted by folded term."""
    observations = [
        (doc.section_id(t.section_ordinal), t)
        for doc in view.citable
        for t in doc.reading.card.terms
    ]
    parent: dict[str, str] = {}

    def root(form: str) -> str:
        while parent.setdefault(form, form) != form:
            form = parent[form]
        return form

    for _, obs in observations:
        forms = [f for f in map(fold_for_quote, (obs.term, *obs.aliases)) if f]
        for form in forms[1:]:
            a, b = root(forms[0]), root(form)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: dict[str, list[tuple[str, TermObservation]]] = {}
    for section_id, obs in observations:
        key = fold_for_quote(obs.term)
        if key:
            groups.setdefault(root(key), []).append((section_id, obs))
    clusters: list[TermCluster] = []
    for members in groups.values():
        sections = {sid for sid, _ in members}
        if len(sections) < 2 and not any(obs.defines for _, obs in members):
            continue
        # Every choice below is a total order over the members, never their observation
        # order, so the same observations in any order give the same cluster and prompt.
        terms = Counter(obs.term for _, obs in members)
        term = min(terms, key=lambda t: (-terms[t], fold_text(t), t))
        kinds = Counter(obs.kind for _, obs in members)
        kind = min(kinds, key=lambda k: (-kinds[k], k))
        aliases = canonical_aliases(
            term, [f for _, obs in members for f in (obs.term, *obs.aliases)]
        )

        def in_path_order(q: TermQuote) -> tuple[tuple[int, int], str, bool]:
            return view.section_order(q.section_id), q.text, q.defines

        every = {TermQuote(sid, obs.quote, obs.defines) for sid, obs in members}
        kept = sorted(every, key=lambda q: (not q.defines, in_path_order(q)))
        quotes = tuple(sorted(kept[:MAX_QUOTES_PER_TERM], key=in_path_order))
        forms = [f for f in map(fold_for_quote, (term, *aliases)) if f]
        occurrences: list[TermOccurrence] = []
        for sid, folded in view.folded_sections.items():
            if len(occurrences) >= MAX_OCCURRENCES_PER_TERM:
                break
            if sid in sections:
                continue
            for form in forms:
                match = _find_form(folded, form)
                if match:
                    start = max(0, match.start() - OCCURRENCE_RADIUS)
                    snippet = folded[start : start + OCCURRENCE_CHARS]
                    occurrences.append(TermOccurrence(sid, snippet))
                    break
        clusters.append(TermCluster(term, kind, aliases, quotes, tuple(occurrences)))
    return sorted(clusters, key=lambda c: (fold_for_quote(c.term), c.term))


def glossary_batches(clusters: Sequence[TermCluster]) -> list[tuple[TermCluster, ...]]:
    batches: list[list[TermCluster]] = []
    size = 0
    for cluster in clusters:
        length = len(cluster.render())
        if batches and size + length <= GLOSSARY_BATCH_CHARS:
            batches[-1].append(cluster)
            size += length
        else:
            batches.append([cluster])
            size = length
    return [tuple(b) for b in batches]


def check_glossary(
    draft: GlossaryDraft, clusters: Sequence[TermCluster], view: CorpusView
) -> Checked[list[GlossaryEntry]]:
    """Every given term gets one entry, and every cited section contains the entry's term or
    one of the aliases it publishes. Structure is checked before any entry is built."""
    rejections: list[str] = []
    accepted: list[tuple[GlossaryEntryDraft, tuple[str, ...]]] = []
    answered: set[int] = set()
    for entry in draft.entries:
        aliases = canonical_aliases(entry.term, entry.aliases)
        forms = {f for f in map(fold_for_quote, (entry.term, *aliases)) if f}
        owners = [i for i, c in enumerate(clusters) if c.forms & forms]
        if len(owners) != 1:
            rejections.append(f"entry {entry.term!r} does not name exactly one of the given terms")
            continue
        if owners[0] in answered:
            rejections.append(
                f"term {clusters[owners[0]].term!r} has two entries; give one entry with"
                " several senses"
            )
        answered.add(owners[0])
        problems = len(rejections)
        if not entry.senses:
            rejections.append(f"entry {entry.term!r} has no senses")
        for sense in entry.senses:
            if not sense.citations:
                rejections.append(f"a sense of {entry.term!r} has no citations")
            for citation in sense.citations:
                if citation not in view.sections:
                    rejections.append(f"{entry.term!r}: {citation!r} is not a citable section ID")
                elif not any(_find_form(view.folded_sections[citation], f) for f in forms):
                    rejections.append(
                        f"{citation} contains neither {entry.term!r} nor an alias the entry"
                        " gives; cite sections where the term occurs, or add the form used"
                        " there as an alias"
                    )
        if len(rejections) == problems:
            accepted.append((entry, aliases))
    for index, cluster in enumerate(clusters):
        if index not in answered:
            rejections.append(f"no entry for the term {cluster.term!r}")
    if rejections:
        return Checked.reject(rejections)
    return Checked.accept(
        [
            GlossaryEntry(
                term=entry.term,
                kind=entry.kind,
                aliases=aliases,
                senses=tuple(
                    GlossarySense(
                        definition=s.definition, citations=tuple(dict.fromkeys(s.citations))
                    )
                    for s in entry.senses
                ),
            )
            for entry, aliases in accepted
        ]
    )


DEFINITION_OPERATION = "check_definitions"


def definition_predicate(term: str, definition: str) -> str:
    return f'The text establishes that "{term}" means: "{definition}"'


@dataclass(frozen=True)
class GlossaryResult:
    entries: tuple[GlossaryEntry, ...]
    unsupported: int
    refusals: int


def write_glossary(
    gateway: LanguageGateway, view: CorpusView, pool: InductionExecutor
) -> GlossaryResult:
    batches = glossary_batches(term_clusters(view))

    def write(batch: tuple[TermCluster, ...]) -> list[GlossaryEntry]:
        return generate_checked(
            "glossary",
            lambda feedback: gateway.run(WriteGlossary(batch, feedback)),
            lambda draft: check_glossary(draft, batch, view),
        )

    entries = [e for batch in pool.map(write, batches) for e in batch]
    checks = [
        (e, i, entry.term, sense.definition, sense.citations)
        for e, entry in enumerate(entries)
        for i, sense in enumerate(entry.senses)
        if sense.definition is not None
    ]

    def support(item: tuple[int, int, str, str, tuple[str, ...]]) -> float | None:
        _, _, term, definition, citations = item
        predicate = definition_predicate(term, definition)
        (probability,) = judge_support(
            gateway, DEFINITION_OPERATION, view.evidence(citations), [predicate]
        )
        return probability

    verdicts = pool.map(support, checks)
    demoted = {
        (e, i)
        for (e, i, *_), p in zip(checks, verdicts, strict=True)
        if p is None or p < SUPPORT_THRESHOLD
    }
    result = tuple(
        entry.model_copy(
            update={
                "senses": tuple(
                    s.model_copy(update={"definition": None}) if (e, i) in demoted else s
                    for i, s in enumerate(entry.senses)
                )
            }
        )
        for e, entry in enumerate(entries)
    )
    ordered = tuple(sorted(result, key=lambda e: (fold_text(e.term), e.term)))
    return GlossaryResult(ordered, len(demoted), sum(p is None for p in verdicts))


# Step 6: guide.

GUIDE_INSTRUCTIONS = """\
You write a concise high-level orientation guide (target 4,000 to 8,000 characters,
about 600 to 1,200 words, and strictly under 12,000 characters) for a knowledge base
that AI agents will search. Synthesise themes and patterns across the collection rather
than cataloguing individual documents or projects one by one (agents use list_documents
and search for individual items). Using only the material given, describe in Markdown:
what the collection holds and how it is organised (its facets), what each deliverable
type covers, recurring methods and domain conventions, and the caveats or gaps the
curator notes record. Cite representative factual claims with exact section citation
tokens from the catalog, written as [doc_a1b2c3d4e5:doc_a1b2c3d4e5_sec_001] (1 to 3
representative citations per point, never exhaustive lists). Use no other citation form.
"""


@dataclass(frozen=True)
class GuideEntry:
    """One citable document as the guide request shows it: its first section, then up to
    `GUIDE_HEADINGS_PER_DOCUMENT` of its top-level sections."""

    doc_id: str
    title: str
    # `DocumentReading.date_label`.
    date_label: str | None
    deliverable_type: str
    summary: str
    first_section_id: str
    headings: tuple[tuple[str, str], ...]

    def render(self, *, headings: int, summary_chars: int, title_chars: int) -> str:
        date = self.date_label or "no date"
        line = (
            f"[{self.doc_id}:{self.first_section_id}] {one_line(self.title, title_chars)}"
            f" ({one_line(date, 40)}, {one_line(self.deliverable_type, 60)})"
        )
        if summary_chars:
            line += f": {one_line(self.summary, summary_chars)}"
        lines = [line]
        lines += [
            f"  - [{self.doc_id}:{sid}] {one_line(heading, 120)}"
            for sid, heading in self.headings[:headings]
        ]
        return "\n".join(lines)


# Catalog settings from richest to leanest: headings per document, summary and title length.
_CATALOG_SETTINGS = (
    *((h, 300, 150) for h in range(GUIDE_HEADINGS_PER_DOCUMENT, -1, -1)),
    (0, 200, 150),
    (0, 120, 120),
    (0, 60, 100),
    (0, 0, 80),
)


def render_catalog(entries: Sequence[GuideEntry], budget: int) -> str:
    """The richest catalog within `budget` characters. When even the leanest entries do not
    all fit, entries are listed in path order up to the budget and the rest counted."""
    for headings, summary_chars, title_chars in _CATALOG_SETTINGS:
        text = "\n".join(
            e.render(headings=headings, summary_chars=summary_chars, title_chars=title_chars)
            for e in entries
        )
        if len(text) <= budget:
            return text
    lean = [e.render(headings=0, summary_chars=0, title_chars=80) for e in entries]
    return bounded_lines(lean, budget, "documents")


@dataclass(frozen=True)
class TypeSummaryLine:
    name: str
    description: str
    documents: int
    feedback: int


@dataclass(frozen=True)
class WriteGuide:
    operation: ClassVar[str] = "write_guide"
    output: ClassVar[type[GuideDraft]] = GuideDraft
    effort: ClassVar[Effort] = "medium"
    instructions: ClassVar[str] = GUIDE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    topic: str
    notes: str
    catalog: tuple[GuideEntry, ...]
    facets: tuple[Facet, ...]
    glossary_terms: tuple[str, ...]
    deliverable_types: tuple[TypeSummaryLine, ...]
    feedback: str | None = None

    def text(self) -> str:
        facets = bounded_lines(
            [
                f"- {f.name} ({f.binding}-bound): {one_line(f.description, 200)}. Values:"
                f" {one_line(', '.join(v.value for v in f.values), 600)}"
                for f in self.facets
            ],
            MAX_GUIDE_FACETS_CHARS,
            "facets",
        )
        types = bounded_lines(
            [
                f"- {t.name}: {one_line(t.description, 200)} ({t.documents} documents,"
                f" {t.feedback} feedback documents)"
                for t in self.deliverable_types
            ],
            MAX_GUIDE_TYPES_CHARS,
            "deliverable types",
        )
        terms = one_line(", ".join(self.glossary_terms), MAX_GUIDE_GLOSSARY_CHARS)
        head = (
            f"TOPIC: {one_line(self.topic, MAX_GUIDE_TOPIC_CHARS)}\n\n"
            f"CURATOR NOTES:\n{self.notes}\n\n"
            f"FACETS:\n{facets or '(none)'}\n\nDELIVERABLE TYPES:\n{types or '(none)'}\n\n"
            f"GLOSSARY TERMS: {terms or '(none)'}\n\nCATALOG:\n"
        )
        tail = f"\n\n{self.feedback}" if self.feedback else ""
        budget = max(0, min(MAX_GUIDE_CATALOG_CHARS, MAX_GUIDE_TEXT_CHARS - len(head) - len(tail)))
        return f"{head}{render_catalog(self.catalog, budget)}{tail}"


def invalid_citations(markdown: str, view: CorpusView) -> list[str]:
    """Bracketed `[doc...]` tokens that do not resolve, or a marker when one is needed."""
    bad: list[str] = []
    resolved = 0
    for token in _CITATION_TOKEN.findall(markdown):
        match = _CITATION.fullmatch(token)
        if (
            match is None
            or match[2] not in view.sections
            or view.sections[match[2]][0].doc_id != match[1]
        ):
            bad.append(token)
        else:
            resolved += 1
    if view.sections and resolved == 0:
        bad.append("(no valid citation)")
    return bad


def citation_feedback(bad: Sequence[str], view: CorpusView) -> str:
    """The rejection of a guide's citations, within `MAX_CITATION_FEEDBACK_CHARS`: the repair
    instruction first, then, when it named an unknown document or none, a sample of valid
    tokens, then the valid tokens of the documents it tried to cite, then the invalid tokens
    it wrote. A part that does not fit is left out, so the instruction always arrives whole;
    never the whole corpus's section list."""
    tried = list(dict.fromkeys(m for token in bad for m in _DOCUMENT_ID.findall(token)))
    known = [d for d in tried if d in view.citable_order]
    parts = [
        "fix the citations: cite only catalog tokens, copied exactly as"
        " [doc_id:doc_id_sec_NNN] where the section belongs to that document"
    ]
    if len(known) < len(tried) or not tried:
        samples = [
            f"[{d.doc_id}:{d.section_id(d.reading.sections[0].ordinal)}]"
            for d in view.citable[:CITATION_FEEDBACK_SAMPLES]
            if d.reading.sections
        ]
        parts.append("unknown documents; valid tokens include " + ", ".join(samples))
    for doc_id in known[:5]:
        doc = view.document(doc_id)
        tokens = [f"[{doc_id}:{doc.section_id(s.ordinal)}]" for s in doc.reading.sections]
        listed = ", ".join(tokens[:MAX_CITATION_FEEDBACK_TOKENS])
        extra = len(tokens) - MAX_CITATION_FEEDBACK_TOKENS
        listed += f" and {extra} more" if extra > 0 else ""
        parts.append(f"valid citations of {doc_id}: {listed}")
    shown = [one_line(t, CITATION_FEEDBACK_TOKEN_CHARS) for t in bad[:MAX_CITATION_FEEDBACK_TOKENS]]
    more = len(bad) - len(shown)
    parts.append(f"invalid citations: {', '.join(shown)}" + (f" and {more} more" if more else ""))
    text = parts[0]
    for part in parts[1:]:
        if len(text) + 2 + len(part) <= MAX_CITATION_FEEDBACK_CHARS:
            text += f"; {part}"
    return text


def guide_catalog(view: CorpusView, roles: Mapping[str, DocumentRole]) -> tuple[GuideEntry, ...]:
    entries: list[GuideEntry] = []
    for doc in view.citable:
        sections = doc.reading.sections
        top = min(s.level for s in sections)
        entries.append(
            GuideEntry(
                doc_id=doc.doc_id,
                title=doc.reading.card.title,
                date_label=doc.reading.date_label,
                deliverable_type=roles[doc.doc_id].deliverable_type,
                summary=doc.reading.card.summary,
                first_section_id=doc.section_id(sections[0].ordinal),
                headings=tuple(
                    (doc.section_id(s.ordinal), heading_context(s.heading_path))
                    for s in sections[1:]
                    if s.level == top
                )[:GUIDE_HEADINGS_PER_DOCUMENT],
            )
        )
    return tuple(entries)


def write_guide(gateway: LanguageGateway, request: WriteGuide, view: CorpusView) -> str:
    def check(draft: GuideDraft) -> Checked[str]:
        length = len(draft.guide_markdown)
        size = len(json.dumps(draft.guide_markdown, ensure_ascii=False).encode())
        if length > MAX_GUIDE_MARKDOWN_CHARS or size > MAX_GUIDE_MARKDOWN_BYTES:
            return Checked.reject(
                [
                    f"the guide has {length} characters ({size} bytes as a JSON string), which"
                    f" exceeds the hard limit of {MAX_GUIDE_MARKDOWN_CHARS} characters and"
                    f" {MAX_GUIDE_MARKDOWN_BYTES} bytes; shorten it substantially to under"
                    " 8,000 characters (~1,000 words) by synthesising high-level patterns with"
                    " 1-2 representative citations per point instead of listing individual"
                    " documents or projects"
                ]
            )
        bad = invalid_citations(draft.guide_markdown, view)
        if not bad:
            return Checked.accept(draft.guide_markdown)
        if not view.sections:
            return Checked.reject(
                [
                    "invalid citations: "
                    + ", ".join(
                        one_line(t, CITATION_FEEDBACK_TOKEN_CHARS)
                        for t in bad[:MAX_CITATION_FEEDBACK_TOKENS]
                    )
                    + "; this"
                    " collection has no citable sections, so the guide must contain no citation"
                    " tokens"
                ]
            )
        return Checked.reject([citation_feedback(bad, view)])

    return generate_checked(
        "topic guide",
        lambda feedback: gateway.run(replace(request, feedback=feedback)),
        check,
    )


# Composition.

INDUCTION_EFFORTS: dict[str, str] = {
    OrganiseCorpus.operation: OrganiseCorpus.effort,
    CanonicaliseTypes.operation: CanonicaliseTypes.effort,
    CanonicaliseFacets.operation: CanonicaliseFacets.effort,
    PartitionVersions.operation: PartitionVersions.effort,
    ExtractFeedbackLessons.operation: ExtractFeedbackLessons.effort,
    WriteGlossary.operation: WriteGlossary.effort,
    WriteGuide.operation: WriteGuide.effort,
}


@dataclass(frozen=True)
class CorpusKnowledge:
    facets: tuple[Facet, ...]
    # doc_id -> facet name -> the document's sorted values; every kept facet has a key.
    assignments: dict[str, dict[str, tuple[str, ...]]]
    # Citable document ID to its role.
    roles: dict[str, DocumentRole]
    deliverable_types: tuple[DeliverableType, ...]
    glossary: tuple[GlossaryEntry, ...]
    feedback_lessons: tuple[FeedbackLesson, ...]
    unit_clusters: tuple[UnitCluster, ...]
    guide_markdown: str
    report: BuildReport


def unit_clusters(view: CorpusView, roles: Mapping[str, ProposedRole]) -> tuple[UnitCluster, ...]:
    """Recurring units of the deliverables. Feedback documents take no part: a reviewer quoting
    a flawed paragraph does not make it house text."""
    return _passages.recurring_units(
        [
            _passages.TextDocument(
                d.doc_id,
                d.rel_path,
                tuple(
                    _passages.TextSection(d.section_id(s.ordinal), s.markdown)
                    for s in d.reading.sections
                ),
            )
            for d in view.citable
            if not roles[d.doc_id].is_feedback
        ]
    )


def induce(
    documents: Sequence[CorpusDocument],
    notes: Sequence[CuratorNote],
    gateway: LanguageGateway,
    *,
    topic: str,
    max_concurrency: int = 4,
) -> CorpusKnowledge:
    """Organise, bind, version families, then feedback lessons and the glossary, then the
    guide."""
    view = CorpusView(tuple(documents))
    # Created first, so a non-positive limit fails before any provider call.
    with InductionExecutor(max_concurrency) as pool:
        organisation = organise(gateway, view, notes, pool)
        proposed = organisation.roles
        binding = bind_facets(gateway, view, organisation.facets, pool)
        versions = version_families(gateway, view, proposed, pool)
        # A superseded feedback document (an earlier draft or a copy of a review) gives no
        # lessons: its latest version gives the current ones. Lesson extraction is queued
        # before the glossary, which runs on the same workers behind it.
        lesson_futures = [
            pool.submit(extract_feedback_lessons, gateway, view, doc, proposed[doc.doc_id])
            for doc in view.citable
            if proposed[doc.doc_id].is_feedback and versions.members[doc.doc_id][1] != "superseded"
        ]
        glossary = write_glossary(gateway, view, pool)
        lessons = pool.results(lesson_futures)
    roles = {
        doc_id: DocumentRole(
            deliverable_type=role.deliverable_type,
            is_feedback=role.is_feedback,
            feedback_for_types=role.feedback_for_types,
            family_id=versions.members[doc_id][0],
            version_status=versions.members[doc_id][1],
            superseded_by=versions.members[doc_id][2],
        )
        for doc_id, role in proposed.items()
    }
    type_lines = tuple(
        TypeSummaryLine(
            t.name,
            t.description,
            sum(r.deliverable_type == t.name for r in roles.values()),
            sum(t.name in r.feedback_for_types for r in roles.values()),
        )
        for t in organisation.deliverable_types
    )
    request = WriteGuide(
        topic=topic,
        notes=notes_text(notes).text,
        catalog=guide_catalog(view, roles),
        facets=binding.facets,
        glossary_terms=tuple(e.term for e in glossary.entries),
        deliverable_types=type_lines,
    )
    guide = write_guide(gateway, request, view)
    refusals = {
        BIND_OPERATION: binding.refusals,
        DEFINITION_OPERATION: glossary.refusals,
        LESSON_SUPPORT_OPERATION: sum(r.refusals for r in lessons),
    }
    report = BuildReport(
        discarded_unlocated_quotes=sum(d.reading.discarded_quotes for d in documents)
        + sum(r.discarded for r in lessons),
        unsupported_definitions=glossary.unsupported,
        dropped_feedback_lessons=sum(r.dropped for r in lessons),
        judge_refusals={k: v for k, v in refusals.items() if v},
        pruned_facet_values=list(binding.pruned_values),
        pruned_facets=list(binding.pruned_facets),
        truncated_curator_notes=organisation.truncated_notes,
        dropped_version_edges=versions.dropped_edges,
    )
    return CorpusKnowledge(
        facets=binding.facets,
        assignments=binding.assignments,
        roles=roles,
        deliverable_types=organisation.deliverable_types,
        glossary=glossary.entries,
        feedback_lessons=tuple(lesson for r in lessons for lesson in r.lessons),
        unit_clusters=unit_clusters(view, proposed),
        guide_markdown=guide,
        report=report,
    )
