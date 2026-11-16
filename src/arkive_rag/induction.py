"""Induction: what a whole corpus is organised by and how it writes, induced from the readings.

`induce` runs five steps in a fixed order: organise (facets and recurring deliverable types),
bind (facet values to documents), then the glossary and the blueprints concurrently, then the
topic guide. Every generating step goes through `generate_checked`: run, validate, rerun once
with the rejections named, then raise. Only three outcomes reduce output without failing, and
the build report counts each: an observation whose quote is still unlocated after the rerun
(in the reading pass), a glossary definition the support check does not confirm (kept with
`definition=None`), and a blueprint convention or pitfall it does not confirm (dropped).

Everything that reaches a prompt or a cache key is in a canonical sorted order, and
concurrent results are gathered in submission order, so an unchanged corpus with the same
build cache rebuilds without provider requests.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Self

from arkive_rag.llm import Effort, LanguageGateway, Tool
from arkive_rag.models import (
    Blueprint,
    BlueprintDraft,
    BlueprintGuidance,
    BlueprintSection,
    BuildReport,
    CorpusDocument,
    CuratorNote,
    Facet,
    FacetValue,
    GlossaryDraft,
    GlossaryEntry,
    GlossaryEntryDraft,
    GlossarySense,
    GuidanceDraft,
    GuideDraft,
    NotFoundError,
    ObservationCard,
    OrganisationDraft,
    ReadSection,
    RejectedOutputError,
    StandingPassage,
    TermObservation,
    fold_facet_label,
    fold_for_quote,
    fold_text,
)

MAX_FACETS = 6
# Every recurring deliverable type costs one blueprint agent run.
MAX_DELIVERABLE_TYPES = 8
# A content-bound value is assigned, and a definition or blueprint statement kept, when the
# decision model's probability reaches this threshold. A refusal never assigns or confirms.
SUPPORT_THRESHOLD = 0.5
BLUEPRINT_MAX_TURNS = 12
GLOSSARY_BATCH_CHARS = 24_000
MAX_OCCURRENCES_PER_TERM = 6
OCCURRENCE_RADIUS = 160
# A support check sends at most this many characters of evidence per request. Longer evidence
# is judged in windows, a section too long for one window is split into overlapping parts, and
# each predicate takes its highest probability over the windows: no cited text is cut.
EVIDENCE_WINDOW_CHARS = 48_000
EVIDENCE_OVERLAP_CHARS = 2_000
# A window labels each section with its heading path. The section's Markdown already holds its
# own heading line, so a longer label is shortened; this keeps the parts of a long section
# close to a full window whatever its heading, so their number stays bounded by its length.
EVIDENCE_LABEL_CHARS = 1_000
MAX_TOOL_GREP_MATCHES = 20
# Names the behaviour of the blueprint tools. It is part of every blueprint request's source
# digest, so a change in what the tools return can never be answered from a cached run.
SOURCE_TOOLS_REVISION = "arkive-source-tools/1"

_FACET_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")
_CITATION_TOKEN = re.compile(r"\[doc[^\]]*\]")
_CITATION = re.compile(r"\[(doc_\d+):(doc_\d+_sec_\d+)\]")


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
    listed = "\n".join(f"- {r}" for r in rejections)
    return f"YOUR PREVIOUS ANSWER WAS REJECTED:\n{listed}\nAnswer again and fix every point."


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
    """Citable documents and their sections, keyed by ID. Curator notes are not citable."""

    documents: tuple[CorpusDocument, ...]
    citable: tuple[CorpusDocument, ...] = field(init=False)
    sections: Mapping[str, tuple[CorpusDocument, ReadSection]] = field(init=False)
    folded_sections: Mapping[str, str] = field(init=False)

    def __post_init__(self) -> None:
        citable = tuple(d for d in self.documents if d.is_citable)
        sections = {d.section_id(s.ordinal): (d, s) for d in citable for s in d.reading.sections}
        object.__setattr__(self, "citable", citable)
        object.__setattr__(self, "sections", sections)
        folded = {sid: fold_for_quote(self.markdown(sid)) for sid in sections}
        object.__setattr__(self, "folded_sections", folded)

    @property
    def citable_ids(self) -> frozenset[str]:
        return frozenset(d.doc_id for d in self.citable)

    def document(self, doc_id: str) -> CorpusDocument:
        for doc in self.citable:
            if doc.doc_id == doc_id:
                return doc
        raise NotFoundError(f"unknown or non-citable document {doc_id!r}")

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


def _card_lines(doc_id: str, rel_path: str, card: ObservationCard) -> list[str]:
    lines = [
        f"## {doc_id}: {card.title}",
        f"Path: {rel_path}",
        f"Date: {card.doc_date or 'not stated'} | Deliverable type: {card.deliverable_type}",
        f"Summary: {card.summary}",
    ]
    lines += [f"- {a.name}: {a.value}" for a in card.attributes]
    return lines


# Step 1: organise.

ORGANISE_INSTRUCTIONS = f"""\
You organise one curated document collection so that agents can filter it and
learn how its recurring deliverables are written. You receive the folder tree,
the curators' notes (their stated intent outranks your inference) and a card for
every citable document.

facets: 0 to {MAX_FACETS} useful ways to filter the collection, such as client, project,
author, campaign, audience, channel, region or document role. Fewer is fine,
including none. name is snake_case. Each facet has a closed list of canonical
values, each with a description and aliases (other spellings or abbreviations;
an alias may be used once per facet). A facet is either path-bound, when the
folder structure already encodes it: then every value lists the folders it stands
for, exactly as given in the folder tree; or content-bound: then no value lists
folders and each document is classified by reading it. Never mix the two in one
facet. Prefer facets that split the collection into groups of several documents.

deliverable_types: at most {MAX_DELIVERABLE_TYPES} kinds of deliverable the collection holds
more than one example of (for example technical memorandum, proposal, campaign
brief, slide deck); keep the ones most worth a writing blueprint. For each, give
a generic name and description, the document IDs of its exemplars, and the IDs
of documents that review or comment on such deliverables (feedback), which may
be empty. Use only document IDs from the cards.
"""


@dataclass(frozen=True)
class CardDigest:
    doc_id: str
    rel_path: str
    card: ObservationCard


@dataclass(frozen=True)
class OrganiseCorpus:
    operation: ClassVar[str] = "organise_corpus"
    output: ClassVar[type[OrganisationDraft]] = OrganisationDraft
    effort: ClassVar[Effort] = "medium"
    instructions: ClassVar[str] = ORGANISE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    folders: tuple[str, ...]
    notes: tuple[CuratorNote, ...]
    cards: tuple[CardDigest, ...]
    feedback: str | None = None

    def text(self) -> str:
        folders = "\n".join(f"- {f}/" for f in self.folders) or "(all documents are in the root)"
        notes = "\n\n".join(f"### {n.scope_path}\n{n.content}" for n in self.notes) or "(none)"
        cards = "\n\n".join(
            "\n".join(_card_lines(c.doc_id, c.rel_path, c.card)) for c in self.cards
        )
        text = f"FOLDERS:\n{folders}\n\nCURATOR NOTES:\n{notes}\n\nDOCUMENT CARDS:\n{cards}"
        return f"{text}\n\n{self.feedback}" if self.feedback else text


@dataclass(frozen=True)
class DeliverableType:
    name: str
    description: str
    exemplar_doc_ids: tuple[str, ...]
    feedback_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class Organisation:
    facets: tuple[Facet, ...]
    deliverable_types: tuple[DeliverableType, ...]


def check_organisation(
    draft: OrganisationDraft, folders: frozenset[str], citable_ids: frozenset[str]
) -> Checked[Organisation]:
    """Validate an organisation by the folded identity facet filters resolve (`fold_facet_label`).

    Repetition within one value is normalised away: values and aliases are stripped, blank
    aliases dropped, and one alias kept per folded form. Ambiguity is a rejection: a value given
    twice, or one folded form naming two values of a facet.
    """
    rejections: list[str] = []
    if len(draft.facets) > MAX_FACETS:
        rejections.append(f"{len(draft.facets)} facets; give at most {MAX_FACETS}")
    if len(draft.deliverable_types) > MAX_DELIVERABLE_TYPES:
        rejections.append(
            f"{len(draft.deliverable_types)} deliverable types; give at most"
            f" {MAX_DELIVERABLE_TYPES}, the recurring types most worth a blueprint"
        )
    facets: list[Facet] = []
    names: set[str] = set()
    for facet in draft.facets:
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
    types: list[DeliverableType] = []
    type_names: set[str] = set()
    for dtype in draft.deliverable_types:
        folded = fold_text(dtype.name).strip()
        if folded in type_names:
            rejections.append(f"deliverable type {dtype.name!r} is given twice")
        type_names.add(folded)
        if not dtype.exemplar_doc_ids:
            rejections.append(f"deliverable type {dtype.name!r} has no exemplars")
        for doc_id in (*dtype.exemplar_doc_ids, *dtype.feedback_doc_ids):
            if doc_id not in citable_ids:
                rejections.append(
                    f"deliverable type {dtype.name!r}: unknown document ID {doc_id!r}"
                )
        # A feedback document must not supply exemplar text such as a standing passage.
        for doc_id in sorted(set(dtype.exemplar_doc_ids) & set(dtype.feedback_doc_ids)):
            rejections.append(
                f"deliverable type {dtype.name!r}: {doc_id!r} is both an exemplar and a"
                " feedback document; give each document one role"
            )
        types.append(
            DeliverableType(
                name=dtype.name,
                description=dtype.description,
                exemplar_doc_ids=tuple(sorted(set(dtype.exemplar_doc_ids))),
                feedback_doc_ids=tuple(sorted(set(dtype.feedback_doc_ids))),
            )
        )
    if rejections:
        return Checked.reject(rejections)
    ordered_types = tuple(sorted(types, key=lambda t: fold_text(t.name)))
    return Checked.accept(Organisation(tuple(map(with_folder_aliases, facets)), ordered_types))


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


def organise(
    gateway: LanguageGateway, view: CorpusView, notes: Sequence[CuratorNote]
) -> Organisation:
    folders = view.folders()
    cards = tuple(CardDigest(d.doc_id, d.rel_path, d.reading.card) for d in view.citable)
    return generate_checked(
        "corpus organisation",
        lambda feedback: gateway.run(OrganiseCorpus(folders, tuple(notes), cards, feedback)),
        lambda draft: check_organisation(draft, frozenset(folders), view.citable_ids),
    )


# Step 2: bind.

BIND_OPERATION = "bind_facets"


def binding_evidence(doc: CorpusDocument) -> str:
    card = doc.reading.card
    outline = "\n".join(
        f"- {s.heading_path} (pp. {s.page_start}-{s.page_end})" for s in doc.reading.sections
    )
    attributes = "\n".join(f"- {a.name}: {a.value}" for a in card.attributes) or "(none)"
    return (
        f"PATH: {doc.rel_path}\nTITLE: {card.title}\nDATE: {card.doc_date or 'not stated'}\n"
        f"DELIVERABLE TYPE: {card.deliverable_type}\nSUMMARY: {card.summary}\n"
        f"ATTRIBUTES:\n{attributes}\nOUTLINE:\n{outline}"
    )


def binding_predicate(facet: Facet, value: FacetValue) -> str:
    return (
        f'This document has {facet.name} "{value.value}".'
        f" {facet.name}: {facet.description} {value.value}: {value.description}"
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


# Step 3: glossary.

GLOSSARY_INSTRUCTIONS = """\
You write glossary entries for terms a document collection uses. Each term comes
with the quotes where documents mention it, tagged with section IDs, and further
occurrence snippets (shown case-folded). Write one entry per given term: the
term, its kind, its aliases, and its senses. A sense has a definition only when
the cited sections establish what the term means; otherwise definition is null,
which records that the corpus uses the term without defining it. When documents
use the term with conflicting meanings, give one sense per meaning. Cite every
sense with the section IDs (for example doc_003_sec_002) where the term occurs
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
        lines = [f"TERM: {self.term} (kind: {self.kind}; aliases: {', '.join(self.aliases)})"]
        lines += [
            f"- quote [{q.section_id}]{' (defining)' if q.defines else ''}: {q.text}"
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
    citable sections. Sorted by folded term."""
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
        quotes = tuple(
            sorted(
                {TermQuote(sid, obs.quote, obs.defines) for sid, obs in members},
                key=lambda q: (q.section_id, q.text, q.defines),
            )
        )
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
                    end = min(len(folded), match.end() + OCCURRENCE_RADIUS)
                    occurrences.append(TermOccurrence(sid, folded[start:end]))
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


# Step 4: blueprints.

BLUEPRINT_INSTRUCTIONS = """\
You write the blueprint for one recurring deliverable type of a document
collection: how the collection's exemplars of that type are structured and which
conventions and reviewer pitfalls a new one must respect. Use the tools to read
the exemplars' sections and the feedback documents before you answer. The
request may list craft leads: observations noted while each document was read.
They are leads, not evidence: read the section a lead names and use it only if
the source shows it, citing the sections that do.

- description: what this deliverable is for and who reads it.
- outline: the sections a new deliverable of this type should have, in order,
  each with its purpose and the section IDs of exemplar sections that show it.
- conventions: house rules for structure, tables, units, naming, caveats,
  disclaimers and tone, each citing the section IDs that show it.
- pitfalls: what reviewers asked authors to fix, each citing the feedback or
  exemplar section IDs that show it.
- standing_passages: the request lists standing passage candidates: blocks of
  exemplar text (paragraphs, list items, table header rows) that recur in
  another exemplar document. Select each candidate that is standing house
  content: organisation or tool background, method references, reference
  entries, acknowledgements, disclaimers, disclosures, standard table headers
  and standing caveats. Leave out every candidate that is specific to one
  project, client, place, period or result. For each one you select, give its
  candidate_id exactly as listed and a short label. Never write passage text.
  Give an empty list when no candidate is standing house content or when the
  request lists none.

State the outline, conventions and pitfalls generically: no project-specific
values, names, places or dates. Cite only section IDs the tools returned (for
example doc_002_sec_004).
"""


def _string_argument(args: Mapping[str, Any], name: str) -> str:
    """A tool argument the model must give as a string; anything else is invalid input."""
    value = args.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    return value


class SourceTools:
    """Read-only `outline`, `read_section` and `grep` tools over the citable documents.

    Curator notes are not readable. `digest` covers everything the tools can return and
    `SOURCE_TOOLS_REVISION`.
    """

    def __init__(self, view: CorpusView) -> None:
        self._view = view
        content = [
            [
                d.doc_id,
                d.reading.card.title,
                d.reading.card.deliverable_type,
                [
                    [d.section_id(s.ordinal), s.heading_path, s.page_start, s.page_end, s.markdown]
                    for s in d.reading.sections
                ],
            ]
            for d in view.citable
        ]
        self.digest = hashlib.sha256(
            json.dumps([SOURCE_TOOLS_REVISION, content]).encode()
        ).hexdigest()

    def outline(self, args: dict[str, Any]) -> str:
        doc = self._view.document(_string_argument(args, "doc_id"))
        return json.dumps(
            {
                "doc_id": doc.doc_id,
                "title": doc.reading.card.title,
                "deliverable_type": doc.reading.card.deliverable_type,
                "sections": [
                    {
                        "section_id": doc.section_id(s.ordinal),
                        "heading_path": s.heading_path,
                        "pages": f"{s.page_start}-{s.page_end}",
                    }
                    for s in doc.reading.sections
                ],
            },
            ensure_ascii=False,
        )

    def read_section(self, args: dict[str, Any]) -> str:
        section_id = _string_argument(args, "section_id")
        markdown = self._view.markdown(section_id)
        return json.dumps(
            {
                "section_id": section_id,
                "heading_path": self._view.heading(section_id),
                "markdown": markdown,
            },
            ensure_ascii=False,
        )

    def grep(self, args: dict[str, Any]) -> str:
        pattern = _string_argument(args, "pattern")
        if not pattern.strip():
            raise ValueError("grep pattern must not be blank")
        needle = fold_text(pattern)
        matches: list[dict[str, str]] = []
        for section_id, (_, section) in self._view.sections.items():
            for line in section.markdown.split("\n"):
                if needle in fold_text(line):
                    matches.append({"section_id": section_id, "line": line.strip()[:300]})
                    if len(matches) >= MAX_TOOL_GREP_MATCHES:
                        return json.dumps(matches, ensure_ascii=False)
        return json.dumps(matches, ensure_ascii=False)

    @property
    def tools(self) -> tuple[Tool, ...]:
        def schema(name: str) -> dict[str, Any]:
            return {"type": "object", "properties": {name: {"type": "string"}}, "required": [name]}

        return (
            Tool(
                "outline",
                "Title, deliverable type and section outline (section IDs, headings, pages)"
                " of one document.",
                schema("doc_id"),
                self.outline,
            ),
            Tool(
                "read_section",
                "Full Markdown of one section by section ID.",
                schema("section_id"),
                self.read_section,
            ),
            Tool(
                "grep",
                f"Case-insensitive literal search of every section; returns up to"
                f" {MAX_TOOL_GREP_MATCHES} matching lines with their section IDs.",
                schema("pattern"),
                self.grep,
            ),
        )


@dataclass(frozen=True, order=True)
class CraftLead:
    """A craft observation of an exemplar or feedback document, offered to the blueprint agent
    as a lead to verify in the source."""

    section_id: str
    aspect: str
    statement: str


def craft_leads(view: CorpusView, doc_ids: Sequence[str]) -> tuple[CraftLead, ...]:
    """The located craft observations of `doc_ids`, deduplicated, in canonical sorted order."""
    leads = {
        CraftLead(doc.section_id(c.section_ordinal), c.aspect, c.statement)
        for doc in map(view.document, doc_ids)
        for c in doc.reading.card.craft
    }
    return tuple(sorted(leads))


# Standing passage candidates. Code finds the blocks of a type's exemplar text that recur in
# another exemplar document; the blueprint agent only selects and labels them, so a stored
# passage is exemplar text by construction.

# Shorter prose is a phrase, not a passage worth reproducing.
STANDING_MIN_WORDS = 8
# A table header row is a unit once it fixes this many columns.
STANDING_MIN_TABLE_COLUMNS = 2
# Recurrence compares word 4-grams: long enough that shared vocabulary alone does not match.
STANDING_SHINGLE_WORDS = 4
# Share of a unit's 4-grams a section of another document must hold. Standing blocks measured
# 0.8 to 1.0 between projects, so light edits such as a changed year still count as recurrence.
STANDING_RECURRENCE_CONTAINMENT = 0.8

WORD = re.compile(r"\w+")
_MARKDOWN_HEADING_LINE = re.compile(r"#{1,6}\s")
# A list item's leading marker and the horizontal space after it, followed by content on the
# same line. Besides Markdown markers, the bullet glyphs PDF text layers carry into
# transcripts; U+F0B7 is the Private Use Area bullet that Word's Symbol font writes into PDF
# text layers.
LIST_MARKER = re.compile(
    r"(?:[-*+\u2022\u25e6\u25aa\u2023\u2043\uf0b7]|\d{1,3}[.)])[^\S\r\n]+(?=\S)"
)
_LIST_ITEM = re.compile(rf"\s*{LIST_MARKER.pattern}")
_FENCE_LINE = re.compile(r"\s*(`{3,}|~{3,})")
_TABLE_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
# How the compiler opens a visual transcript; a transcript is a description, not house text.
VISUAL_TRANSCRIPT_PREFIX = "> **[Visual"
_TABLE_KINDS = ("outer_pipe_table", "pipeless_table")


@dataclass(frozen=True)
class StandingUnit:
    """One block of a section's Markdown: a paragraph or list item (`header` false), or a
    table's header row with its separator row (`header` true). `text` is exactly as in the
    section, without surrounding whitespace."""

    text: str
    header: bool


def _table_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _is_table_separator(row: str) -> bool:
    return "|" in row and all(
        _TABLE_SEPARATOR_CELL.fullmatch(c.replace(" ", "")) for c in _table_cells(row)
    )


def standing_units(markdown: str) -> list[StandingUnit]:
    """The units of one section in order. A blank line, heading or fenced block ends a unit.
    A table starts at a line opening with `|` (outer pipes), or at a line holding `|` right
    above a separator row, whatever its first character. A table with outer pipes continues
    through lines that hold `|` and do not open with a list marker (`LIST_MARKER`), so a list
    item written right after it is a list item; a table without outer pipes continues
    through every line holding `|`, so its rows stay in it whatever their first character.
    Any other line that opens with a list marker starts a unit; an unmarked line right after
    a list item continues it. Headings, fenced blocks, visual transcripts and table body rows
    are not units; a table without a separator row has no header unit. A block quote that is
    not a visual transcript is a paragraph. A unit's text keeps its inner bytes, such as the
    trailing spaces of a hard line break."""
    units: list[StandingUnit] = []
    kind: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal kind
        text = "\n".join(lines).strip()
        if kind in ("paragraph", "item") or (
            kind == "quote" and not text.startswith(VISUAL_TRANSCRIPT_PREFIX)
        ):
            units.append(StandingUnit(text, header=False))
        elif kind in _TABLE_KINDS and len(lines) >= 2 and _is_table_separator(lines[1]):
            units.append(StandingUnit("\n".join(lines[:2]).strip(), header=True))
        kind = None
        lines.clear()

    fence: str | None = None
    all_lines = markdown.splitlines()
    for i, line in enumerate(all_lines):
        stripped = line.strip()
        if fence is not None:
            if stripped.startswith(fence) and not stripped.strip(fence[0]):
                fence = None
            continue
        if opening := _FENCE_LINE.match(line):
            flush()
            fence = opening[1]
            continue
        if not stripped or _MARKDOWN_HEADING_LINE.match(stripped):
            flush()
            continue
        if "|" in stripped and (
            kind == "pipeless_table" or (kind == "outer_pipe_table" and not _LIST_ITEM.match(line))
        ):
            pass
        elif stripped.startswith("|") or (
            "|" in stripped and i + 1 < len(all_lines) and _is_table_separator(all_lines[i + 1])
        ):
            flush()
            kind = "outer_pipe_table" if stripped.startswith("|") else "pipeless_table"
        elif _LIST_ITEM.match(line):
            flush()
            kind = "item"
        elif stripped.startswith(">"):
            if kind != "quote":
                flush()
                kind = "quote"
        elif kind not in ("paragraph", "item"):
            flush()
            kind = "paragraph"
        lines.append(line)
    flush()
    return units


def shingles(folded: str) -> frozenset[tuple[str, ...]]:
    """The word `STANDING_SHINGLE_WORDS`-grams of text already in `fold_for_quote` form."""
    words = WORD.findall(folded)
    n = STANDING_SHINGLE_WORDS
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def containment(part: frozenset[tuple[str, ...]], whole: frozenset[tuple[str, ...]]) -> float:
    """Share of `part`'s 4-grams that `whole` holds; `part` is never empty for a unit."""
    return len(part & whole) / len(part)


def _top_folder(rel_path: str) -> str:
    """The first path component, or "" for a document at the corpus root."""
    return rel_path.split("/")[0] if "/" in rel_path else ""


@dataclass(frozen=True)
class _PlacedUnit:
    doc: CorpusDocument
    section_id: str
    unit: StandingUnit
    # `fold_for_quote` of the text, or of the header row alone for a table header.
    folded: str
    grams: frozenset[tuple[str, ...]]

    def says_the_same_as(self, other: _PlacedUnit) -> bool:
        if self.unit.header or other.unit.header:
            return self.unit.header == other.unit.header and self.folded == other.folded
        return self.folded == other.folded or (
            containment(self.grams, other.grams) >= STANDING_RECURRENCE_CONTAINMENT
            and containment(other.grams, self.grams) >= STANDING_RECURRENCE_CONTAINMENT
        )


@dataclass(frozen=True)
class StandingCandidate:
    """A recurring block of exemplar text offered to the blueprint agent. `text` is exactly as
    in `citations[0]`, the first holder in canonical order; the other citations are the
    sections of other exemplar documents where it recurs, in canonical order."""

    candidate_id: str
    text: str
    citations: tuple[str, ...]


def standing_candidates(
    view: CorpusView, exemplar_doc_ids: Sequence[str]
) -> tuple[StandingCandidate, ...]:
    """The standing passage candidates of a deliverable type's exemplars.

    A unit (`standing_units`) is eligible with at least `STANDING_MIN_WORDS` words, or as a
    table header row of at least `STANDING_MIN_TABLE_COLUMNS` columns. It recurs in a section
    of another exemplar document that holds at least `STANDING_RECURRENCE_CONTAINMENT` of its
    word 4-grams, or, for a table header, that has a table header with the same folded row.
    When the exemplars span more than one top-level folder, that document must sit under
    another top-level folder than the unit's own: the folder is the only sign of a different
    project the code has. Recurring units that say the same thing (the same folded text, or
    containment at the threshold both ways) are one candidate, represented by the first in
    canonical order (document, section ordinal, position in the section).
    """
    docs = [view.document(d) for d in sorted(set(exemplar_doc_ids))]
    spans_folders = len({_top_folder(d.rel_path) for d in docs}) > 1
    sections = [
        (doc, doc.section_id(s.ordinal))
        for doc in docs
        for s in sorted(doc.reading.sections, key=lambda s: s.ordinal)
    ]
    section_grams = {sid: shingles(view.folded_sections[sid]) for _, sid in sections}
    placed: list[_PlacedUnit] = []
    for doc, sid in sections:
        for unit in standing_units(view.markdown(sid)):
            if unit.header:
                header = unit.text.splitlines()[0]
                if len(_table_cells(header)) < STANDING_MIN_TABLE_COLUMNS:
                    continue
                folded = fold_for_quote(header)
                if not folded:
                    continue
            else:
                folded = fold_for_quote(unit.text)
                if len(WORD.findall(folded)) < STANDING_MIN_WORDS:
                    continue
            placed.append(_PlacedUnit(doc, sid, unit, folded, shingles(folded)))
    headers: dict[str, set[str]] = {}
    for p in placed:
        if p.unit.header:
            headers.setdefault(p.section_id, set()).add(p.folded)

    def recurrences(p: _PlacedUnit) -> tuple[str, ...]:
        found: list[str] = []
        for doc, sid in sections:
            if doc.doc_id == p.doc.doc_id or (
                spans_folders and _top_folder(doc.rel_path) == _top_folder(p.doc.rel_path)
            ):
                continue
            if p.unit.header:
                held = p.folded in headers.get(sid, ())
            else:
                held = containment(p.grams, section_grams[sid]) >= STANDING_RECURRENCE_CONTAINMENT
            if held:
                found.append(sid)
        return tuple(found)

    groups: list[tuple[_PlacedUnit, tuple[str, ...]]] = []
    for p in placed:
        where = recurrences(p)
        if where and not any(p.says_the_same_as(holder) for holder, _ in groups):
            groups.append((p, where))
    return tuple(
        StandingCandidate(f"candidate_{n:02d}", holder.unit.text, (holder.section_id, *where))
        for n, (holder, where) in enumerate(groups, start=1)
    )


# The blueprint request carries every offered candidate in full, so the list is capped to keep
# the request bounded however many exemplars a type has.
MAX_STANDING_CANDIDATES = 60


def offered_candidates(
    view: CorpusView, candidates: Sequence[StandingCandidate]
) -> tuple[StandingCandidate, ...]:
    """At most `MAX_STANDING_CANDIDATES` of `candidates` (in canonical order): those that recur
    in the most exemplar documents, ties in canonical order. The kept candidates stay in
    canonical order and keep their IDs."""

    def documents(c: StandingCandidate) -> int:
        return len({view.sections[sid][0].doc_id for sid in c.citations})

    ranked = sorted(range(len(candidates)), key=lambda i: (-documents(candidates[i]), i))
    return tuple(candidates[i] for i in sorted(ranked[:MAX_STANDING_CANDIDATES]))


@dataclass(frozen=True)
class WriteBlueprint:
    operation: ClassVar[str] = "write_blueprint"
    output: ClassVar[type[BlueprintDraft]] = BlueprintDraft
    effort: ClassVar[Effort] = "medium"
    max_turns: ClassVar[int] = BLUEPRINT_MAX_TURNS
    instructions: ClassVar[str] = BLUEPRINT_INSTRUCTIONS
    deliverable_type: DeliverableType
    exemplar_outlines: tuple[str, ...]
    feedback_documents: tuple[str, ...]
    craft_leads: tuple[CraftLead, ...]
    standing_candidates: tuple[StandingCandidate, ...]
    # How many candidates `standing_candidates` found before `offered_candidates` cut them.
    standing_candidates_found: int
    tools: tuple[Tool, ...]
    source_digest: str
    feedback: str | None = None

    def text(self) -> str:
        dtype = self.deliverable_type
        exemplars = "\n\n".join(self.exemplar_outlines)
        feedback_docs = "\n".join(self.feedback_documents) or "(none)"
        leads = "\n".join(
            f"- [{lead.section_id}] {lead.aspect}: {lead.statement}" for lead in self.craft_leads
        )
        if self.standing_candidates:
            candidates = "\n\n".join(
                f"[{c.candidate_id}] found in {', '.join(c.citations)}\n{c.text}"
                for c in self.standing_candidates
            )
            offered = len(self.standing_candidates)
            if offered < self.standing_candidates_found:
                candidates = (
                    f"(this list was cut from {self.standing_candidates_found} candidates to"
                    f" the {offered} that recur in the most exemplar documents)\n\n{candidates}"
                )
        elif len(dtype.exemplar_doc_ids) == 1:
            candidates = (
                "(none: this type has one exemplar document, so no text can recur in another;"
                " return an empty standing_passages list)"
            )
        else:
            candidates = (
                "(none: no block of this type's exemplars recurs in another exemplar document;"
                " return an empty standing_passages list)"
            )
        text = (
            f"DELIVERABLE TYPE: {dtype.name}\nDESCRIPTION: {dtype.description}\n\n"
            f"EXEMPLAR OUTLINES:\n{exemplars}\n\nFEEDBACK DOCUMENTS:\n{feedback_docs}\n\n"
            f"CRAFT LEADS (verify each in its section before using it):\n{leads or '(none)'}\n\n"
            f"STANDING PASSAGE CANDIDATES (select by candidate_id; the first section listed"
            f" holds the text as shown):\n{candidates}"
        )
        return f"{text}\n\n{self.feedback}" if self.feedback else text


def check_blueprint(
    draft: BlueprintDraft,
    dtype: DeliverableType,
    candidates: Sequence[StandingCandidate],
    view: CorpusView,
) -> Checked[Blueprint]:
    """Structure, citations and the standing passage selection. Selected passages take their
    text and citations from the candidates, in candidate order."""
    rejections: list[str] = []
    if not draft.outline:
        rejections.append("the outline is empty")
    cited = [("outline", s.heading, s.citations) for s in draft.outline]
    cited += [("convention", g.statement, g.citations) for g in draft.conventions]
    cited += [("pitfall", g.statement, g.citations) for g in draft.pitfalls]
    for kind, label, citations in cited:
        if not citations:
            rejections.append(f"{kind} {label!r} has no citations")
        for citation in citations:
            if citation not in view.sections:
                rejections.append(f"{kind} {label!r}: {citation!r} is not a citable section ID")
    by_id = {c.candidate_id: c for c in candidates}
    labels: dict[str, str] = {}
    for selection in draft.standing_passages:
        if selection.candidate_id not in by_id:
            rejections.append(
                f"standing passage {selection.candidate_id!r} is not a listed candidate_id"
            )
        elif selection.candidate_id in labels:
            rejections.append(f"standing passage {selection.candidate_id!r} is selected twice")
        elif not selection.label.strip():
            rejections.append(f"standing passage {selection.candidate_id!r} has a blank label")
        labels.setdefault(selection.candidate_id, selection.label.strip())
    if rejections:
        return Checked.reject(rejections)

    def guidance(items: list[GuidanceDraft]) -> tuple[BlueprintGuidance, ...]:
        return tuple(
            BlueprintGuidance(statement=g.statement, citations=tuple(dict.fromkeys(g.citations)))
            for g in items
        )

    return Checked.accept(
        Blueprint(
            deliverable_type=dtype.name,
            description=draft.description,
            exemplar_doc_ids=dtype.exemplar_doc_ids,
            feedback_doc_ids=dtype.feedback_doc_ids,
            outline=tuple(
                BlueprintSection(
                    heading=s.heading,
                    purpose=s.purpose,
                    citations=tuple(dict.fromkeys(s.citations)),
                )
                for s in draft.outline
            ),
            conventions=guidance(draft.conventions),
            pitfalls=guidance(draft.pitfalls),
            standing_passages=tuple(
                StandingPassage(label=labels[c.candidate_id], text=c.text, citations=c.citations)
                for c in candidates
                if c.candidate_id in labels
            ),
        )
    )


BLUEPRINT_SUPPORT_OPERATION = "check_blueprint"


def guidance_predicate(statement: str) -> str:
    return f'The cited text shows or supports this writing guidance: "{statement}"'


@dataclass(frozen=True)
class BlueprintResult:
    blueprint: Blueprint
    dropped: int
    refusals: int


def _outline_text(doc: CorpusDocument) -> str:
    lines = [f"{doc.doc_id}: {doc.reading.card.title}"]
    lines += [f"- {doc.section_id(s.ordinal)}: {s.heading_path}" for s in doc.reading.sections]
    return "\n".join(lines)


def write_blueprint(
    gateway: LanguageGateway, view: CorpusView, tools: SourceTools, dtype: DeliverableType
) -> BlueprintResult:
    outlines = tuple(_outline_text(view.document(d)) for d in dtype.exemplar_doc_ids)
    feedback_docs = tuple(
        f"{d}: {view.document(d).reading.card.title}" for d in dtype.feedback_doc_ids
    )
    leads = craft_leads(view, (*dtype.exemplar_doc_ids, *dtype.feedback_doc_ids))
    found = standing_candidates(view, dtype.exemplar_doc_ids)
    candidates = offered_candidates(view, found)
    blueprint = generate_checked(
        f"blueprint for {dtype.name!r}",
        lambda feedback: gateway.run_agent(
            WriteBlueprint(
                dtype,
                outlines,
                feedback_docs,
                leads,
                candidates,
                len(found),
                tools.tools,
                tools.digest,
                feedback,
            )
        ),
        lambda draft: check_blueprint(draft, dtype, candidates, view),
    )
    dropped = refusals = 0

    def supported(items: tuple[BlueprintGuidance, ...]) -> tuple[BlueprintGuidance, ...]:
        nonlocal dropped, refusals
        kept: list[BlueprintGuidance] = []
        for item in items:
            (probability,) = judge_support(
                gateway,
                BLUEPRINT_SUPPORT_OPERATION,
                view.evidence(item.citations),
                [guidance_predicate(item.statement)],
            )
            if probability is None:
                refusals += 1
            if probability is not None and probability >= SUPPORT_THRESHOLD:
                kept.append(item)
            else:
                dropped += 1
        return tuple(kept)

    checked = blueprint.model_copy(
        update={
            "conventions": supported(blueprint.conventions),
            "pitfalls": supported(blueprint.pitfalls),
        }
    )
    return BlueprintResult(checked, dropped, refusals)


# Step 5: guide.

GUIDE_INSTRUCTIONS = """\
You write the orientation guide for a knowledge base that AI agents will search.
Using only the material given, describe in Markdown: what the collection holds and
how it is organised (its facets), what each deliverable type covers and which
blueprints exist, recurring methods, terms and results, and the caveats or gaps
the curator notes record. Cite every factual claim with the exact section citation
tokens from the catalog, written as [doc_XXX:doc_XXX_sec_YYY]. Use no other
citation form.
"""


@dataclass(frozen=True)
class GuideSection:
    section_id: str
    heading_path: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class GuideEntry:
    """One citable document as the guide request shows it."""

    doc_id: str
    rel_path: str
    card: ObservationCard
    facets: tuple[tuple[str, tuple[str, ...]], ...]
    sections: tuple[GuideSection, ...]

    def render(self) -> str:
        lines = _card_lines(self.doc_id, self.rel_path, self.card)[:4]
        if self.facets:
            facets = "; ".join(f"{name}: {', '.join(values)}" for name, values in self.facets)
            lines.append(f"Facets: {facets}")
        lines += [
            f"- [{self.doc_id}:{s.section_id}] {s.heading_path} (pp. {s.page_start}-{s.page_end})"
            for s in self.sections
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class WriteGuide:
    operation: ClassVar[str] = "write_guide"
    output: ClassVar[type[GuideDraft]] = GuideDraft
    effort: ClassVar[Effort] = "medium"
    instructions: ClassVar[str] = GUIDE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    topic: str
    notes: tuple[CuratorNote, ...]
    catalog: tuple[GuideEntry, ...]
    facets: tuple[Facet, ...]
    glossary_terms: tuple[str, ...]
    blueprints: tuple[tuple[str, str], ...]
    feedback: str | None = None

    def text(self) -> str:
        notes = "\n\n".join(f"### {n.scope_path}\n{n.content}" for n in self.notes) or "(none)"
        facets = "\n".join(
            f"- {f.name} ({f.binding}-bound): {f.description}. Values:"
            f" {', '.join(v.value for v in f.values)}"
            for f in self.facets
        )
        blueprints = "\n".join(f"- {name}: {description}" for name, description in self.blueprints)
        text = (
            f"TOPIC: {self.topic}\n\nCURATOR NOTES:\n{notes}\n\nFACETS:\n{facets or '(none)'}\n\n"
            f"BLUEPRINTS:\n{blueprints or '(none)'}\n\n"
            f"GLOSSARY TERMS: {', '.join(self.glossary_terms) or '(none)'}\n\n"
            f"CATALOG:\n" + "\n\n".join(e.render() for e in self.catalog)
        )
        return f"{text}\n\n{self.feedback}" if self.feedback else text


def invalid_citations(markdown: str, view: CorpusView) -> list[str]:
    """Bracketed `[doc...]` tokens that do not resolve, or a marker when one is needed."""
    valid = {(view.sections[sid][0].doc_id, sid) for sid in view.sections}
    bad: list[str] = []
    resolved = 0
    for token in _CITATION_TOKEN.findall(markdown):
        match = _CITATION.fullmatch(token)
        if match is None or (match[1], match[2]) not in valid:
            bad.append(token)
        else:
            resolved += 1
    if valid and resolved == 0:
        bad.append("(no valid citation)")
    return bad


def _guide_catalog(
    view: CorpusView, assignments: Mapping[str, Mapping[str, Sequence[str]]]
) -> tuple[GuideEntry, ...]:
    return tuple(
        GuideEntry(
            doc_id=doc.doc_id,
            rel_path=doc.rel_path,
            card=doc.reading.card,
            facets=tuple(
                (name, tuple(values)) for name, values in assignments[doc.doc_id].items() if values
            ),
            sections=tuple(
                GuideSection(doc.section_id(s.ordinal), s.heading_path, s.page_start, s.page_end)
                for s in doc.reading.sections
            ),
        )
        for doc in view.citable
    )


def write_guide(gateway: LanguageGateway, request: WriteGuide, view: CorpusView) -> str:
    def check(draft: GuideDraft) -> Checked[str]:
        bad = invalid_citations(draft.guide_markdown, view)
        if not bad:
            return Checked.accept(draft.guide_markdown)
        if not view.sections:
            return Checked.reject(
                [
                    f"invalid citations: {', '.join(bad)}; this collection has no citable"
                    " sections, so the guide must contain no citation tokens"
                ]
            )
        tokens = ", ".join(f"[{view.sections[s][0].doc_id}:{s}]" for s in view.sections)
        return Checked.reject([f"invalid citations: {', '.join(bad)}; cite only {tokens}"])

    return generate_checked(
        "topic guide",
        lambda feedback: gateway.run(replace(request, feedback=feedback)),
        check,
    )


# Composition.

INDUCTION_EFFORTS: dict[str, str] = {
    OrganiseCorpus.operation: OrganiseCorpus.effort,
    WriteGlossary.operation: WriteGlossary.effort,
    WriteBlueprint.operation: WriteBlueprint.effort,
    WriteGuide.operation: WriteGuide.effort,
}


@dataclass(frozen=True)
class CorpusKnowledge:
    facets: tuple[Facet, ...]
    # doc_id -> facet name -> the document's sorted values; every kept facet has a key.
    assignments: dict[str, dict[str, tuple[str, ...]]]
    glossary: tuple[GlossaryEntry, ...]
    blueprints: tuple[Blueprint, ...]
    guide_markdown: str
    report: BuildReport


def induce(
    documents: Sequence[CorpusDocument],
    notes: Sequence[CuratorNote],
    gateway: LanguageGateway,
    *,
    topic: str,
    max_concurrency: int = 4,
) -> CorpusKnowledge:
    """Organise, bind, then the glossary and blueprints concurrently, then the guide."""
    view = CorpusView(tuple(documents))
    # Created first, so a non-positive limit fails before any provider call.
    with InductionExecutor(max_concurrency) as pool:
        organisation = organise(gateway, view, notes)
        binding = bind_facets(gateway, view, organisation.facets, pool)
        tools = SourceTools(view)
        # Blueprints are queued first; the glossary batches and then the definition checks
        # queue behind them on the same workers.
        blueprint_futures = [
            pool.submit(write_blueprint, gateway, view, tools, dtype)
            for dtype in organisation.deliverable_types
        ]
        glossary = write_glossary(gateway, view, pool)
        blueprints = pool.results(blueprint_futures)
    request = WriteGuide(
        topic=topic,
        notes=tuple(notes),
        catalog=_guide_catalog(view, binding.assignments),
        facets=binding.facets,
        glossary_terms=tuple(e.term for e in glossary.entries),
        blueprints=tuple(
            (b.blueprint.deliverable_type, b.blueprint.description) for b in blueprints
        ),
    )
    guide = write_guide(gateway, request, view)
    refusals = {
        BIND_OPERATION: binding.refusals,
        DEFINITION_OPERATION: glossary.refusals,
        BLUEPRINT_SUPPORT_OPERATION: sum(b.refusals for b in blueprints),
    }
    report = BuildReport(
        discarded_unlocated_quotes=sum(d.reading.discarded_quotes for d in documents),
        unsupported_definitions=glossary.unsupported,
        dropped_blueprint_statements=sum(b.dropped for b in blueprints),
        judge_refusals={k: v for k, v in refusals.items() if v},
        pruned_facet_values=list(binding.pruned_values),
        pruned_facets=list(binding.pruned_facets),
    )
    return CorpusKnowledge(
        facets=binding.facets,
        assignments=binding.assignments,
        glossary=glossary.entries,
        blueprints=tuple(b.blueprint for b in blueprints),
        guide_markdown=guide,
        report=report,
    )
