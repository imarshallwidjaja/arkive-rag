"""Layer 3: draft held-out deliverables, measure the drafts and judge them blind.

For each holdout the harness builds a holdout snapshot without the held-out material, then runs
draft rounds. Before the first round it records which documents `prepare_draft` selects for
the holdout's inputs against the pack author's `exemplar_paths`. Every round runs the
holdout's arms from its run notes, scores each completed draft with the rubric and two
lexical-overlap measures, and sends the round's completed drafts to one blind judge call.
Agent runs go through `evals.agents.run_agent`; the judge goes through `OpenAIGateway.run`.

The two measures compare wording, not meaning. They share with induction's recurring units the
unit, folding, shingle and containment functions of `arkive_rag.passages` (`standing_units`,
`fold_unit`, word shingles, `containment`, `without_asides`) and its thresholds
(`STANDING_MIN_WORDS`, `STANDING_RECURRENCE_CONTAINMENT`). They differ from recurrence
discovery (`recurring_units`) in one way: a unit is held when one holdout document, all its
sections together, contains enough of its shingles, where discovery tests each section of
another document on its own. A paragraph that joins two sections of one document is held;
common 4-grams scattered over several unrelated documents never add up to a held unit.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import anyio
from pydantic import BaseModel

from arkive_rag import passages as _passages
from arkive_rag.compiler import RenderCache, compile_corpus, discover_corpus, read_source
from arkive_rag.llm import BudgetExceededError, Effort, LanguageGateway, OpenAIGateway, Tool
from arkive_rag.models import DocumentInfo, ProviderError, fold_facet_label, fold_for_quote
from arkive_rag.snapshot import TopicSnapshot, write_snapshot
from evals.agents import (
    EVAL_EFFORT,
    AgentRun,
    CountingEmbedder,
    RawCorpus,
    Score10,
    arkive_mcp_tools,
    object_schema,
    run_agent,
    with_server_instructions,
)
from evals.schema import Holdout, RubricCheck, held_out_markers_in, satisfies_groups

DRAFT_MAX_TURNS = 16
BLIND_SEED = 42

DraftArm = Literal["zero_shot", "chunk_search", "raw_files", "arkive_mcp", "exemplars_in_context"]


# Holdout guards and the holdout snapshot.


def check_reference_held_out(holdout: Holdout, reference: DocumentInfo) -> None:
    """The reference must be a delivered document, not a curator note, and held-out material,
    so no included path can hold it."""
    if reference.is_curator_note:
        raise ValueError(
            f"reference document {reference.doc_id} ({reference.rel_path}) is a curator note,"
            " not a delivered document"
        )
    if not held_out_markers_in(reference.rel_path, holdout.held_out_markers):
        raise ValueError(
            f"reference document {reference.doc_id} ({reference.rel_path}) contains no held-out"
            f" marker {list(holdout.held_out_markers)}"
        )


def _leak_reasons(
    holdout: Holdout, reference: DocumentInfo, rel_path: str, sha256: str
) -> list[str]:
    """Why a file is held-out material: a marker in its path, or the reference's bytes."""
    reasons = [
        f"held-out marker {m!r}" for m in held_out_markers_in(rel_path, holdout.held_out_markers)
    ]
    if sha256 == reference.sha256:
        reasons.append("same bytes as the reference document")
    return reasons


def select_holdout_sources(
    corpus_dir: Path, holdout: Holdout, reference: DocumentInfo
) -> dict[str, bytes]:
    """Read the corpus files under `holdout.included_paths`, refusing any held-out material
    and any `exemplar_paths` entry that is not one of the selected files or is a curator note.

    Selection goes through `discover_corpus`, so symlinks are refused before anything is read,
    and each file is read once: the bytes checked here are the bytes the holdout is built from.
    """
    sources = discover_corpus(corpus_dir)
    selected: dict[str, bytes] = {}
    curator_notes: set[str] = set()
    for part in holdout.included_paths:
        matched = [s for s in sources if s.rel_path == part or s.rel_path.startswith(f"{part}/")]
        if not matched:
            raise ValueError(f"included path {part!r} matches no corpus file")
        for source in matched:
            data = read_source(source)
            sha256 = hashlib.sha256(data).hexdigest()
            reasons = _leak_reasons(holdout, reference, source.rel_path, sha256)
            if reasons:
                raise ValueError(
                    f"included file {source.rel_path} is held-out material: {', '.join(reasons)}"
                )
            selected[source.rel_path] = data
            if source.is_curator_note:
                curator_notes.add(source.rel_path)
    for rel_path in holdout.exemplar_paths:
        if rel_path not in selected:
            raise ValueError(f"exemplar path {rel_path!r} is not an included corpus file")
        if rel_path in curator_notes:
            raise ValueError(
                f"exemplar path {rel_path!r} is a curator note, not a delivered document"
            )
    return selected


def assert_holdout_clean(
    holdout: Holdout, reference: DocumentInfo, files: Iterable[tuple[str, str]]
) -> None:
    """Fail before any drafting arm runs if held-out material reached what an arm can read:
    `files` are `(rel_path, sha256)` pairs, the holdout snapshot's documents or the raw files
    the raw-file arm exposes."""
    leaked = {
        rel_path: reasons
        for rel_path, sha256 in files
        if (reasons := _leak_reasons(holdout, reference, rel_path, sha256))
    }
    if leaked:
        raise ValueError(f"holdout snapshot leaks held-out material: {leaked}")


def compile_holdout_snapshot(
    corpus_dir: Path,
    output_path: Path,
    gateway: LanguageGateway,
    holdout: Holdout,
    reference: DocumentInfo,
    render_cache: RenderCache | None = None,
) -> dict[str, bytes]:
    """Build the holdout snapshot and publish it at `output_path` only after its guard passes;
    return the included files it was built from (`select_holdout_sources`).

    `render_cache`, normally the build cache `gateway` uses, lets a holdout skip rendering
    files an earlier build rendered."""
    sources = select_holdout_sources(corpus_dir, holdout, reference)
    unverified = output_path.with_name(f".{output_path.name}.unverified")
    with tempfile.TemporaryDirectory(prefix="arkive-holdout-") as scratch:
        holdout_dir = Path(scratch, "corpus")
        for rel_path, data in sources.items():
            (holdout_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (holdout_dir / rel_path).write_bytes(data)
        corpus = compile_corpus(
            holdout_dir,
            gateway,
            topic="holdout",
            title="Corpus without the held-out deliverable",
            render_cache=render_cache,
        )
        try:
            write_snapshot(corpus, unverified)
            with TopicSnapshot.open(unverified) as built:
                assert_holdout_clean(
                    holdout, reference, ((d.rel_path, d.sha256) for d in built.list_documents())
                )
            os.replace(unverified, output_path)
        except BaseException:
            unverified.unlink(missing_ok=True)
            raise
    return sources


def holdout_raw_corpus(
    holdout: Holdout, reference: DocumentInfo, files: Mapping[str, bytes], built: TopicSnapshot
) -> RawCorpus:
    """The raw-file arm's corpus: the included `files` the holdout snapshot `built` was
    compiled from, read from its stored pages (`RawCorpus.from_snapshot`), so no file is
    rendered again. Refuses held-out material among `files` before reading anything; each
    file is hashed once, for both that guard and the snapshot match."""
    digests = {rel_path: hashlib.sha256(data).hexdigest() for rel_path, data in files.items()}
    assert_holdout_clean(holdout, reference, digests.items())
    return RawCorpus.from_snapshot(files, built, digests=digests)


def section_markdowns(snapshot: TopicSnapshot, doc_id: str) -> list[str]:
    """One document's section Markdown in reading order; empty for a document without
    sections, such as an empty curator note."""
    return [snapshot.section_markdown(s.section_id) for s in snapshot.full_outline(doc_id)]


# Lexical-overlap measures.

Shingles = frozenset[tuple[str, ...]]
# Results quote each house or copied unit by its opening characters only.
UNIT_HEAD_CHARS = 120


@dataclass(frozen=True)
class _ProseUnit:
    text: str
    words: int
    grams: Shingles

    @property
    def long(self) -> bool:
        """Long enough to be a house or copied unit."""
        return self.words >= _passages.STANDING_MIN_WORDS


def _prose_units(markdown: str) -> list[_ProseUnit]:
    """Every paragraph and list item (`standing_units` that are not table headers), each folded
    on its own after losing its leading list marker, so `1.` and `-` items measure the same.
    Headings, tables, fenced blocks and visual transcripts are not units. `text` stays as
    written."""
    units: list[_ProseUnit] = []
    for unit in _passages.standing_units(markdown):
        if unit.header:
            continue
        folded = _passages.fold_unit(unit.text)
        units.append(
            _ProseUnit(unit.text, len(_passages.WORD.findall(folded)), _passages.shingles(folded))
        )
    return units


def _text_shingles(text: str) -> Shingles:
    """What a text carries anywhere: its whole-text shingles, tables and headings included and
    visual transcripts removed, together with its prose units' shingles. Folding the whole text
    joins a line-end hyphen or dash with the next unit's first word, and a unit keeps no list
    marker, so neither form alone holds every shingle the other does."""
    whole = _passages.shingles(fold_for_quote(_passages.without_asides(text)))
    return whole.union(*(u.grams for u in _prose_units(text)))


@dataclass(frozen=True)
class _HoldoutText:
    """The holdout text as the documents that hold each shingle, so testing a unit reads only
    its own shingles, however many documents the holdout has."""

    # Each shingle's documents, in index order, each once (a document's text is a set).
    holders: dict[tuple[str, ...], list[int]]

    @classmethod
    def of(cls, documents: Sequence[Shingles]) -> _HoldoutText:
        """`documents` are the holdout documents' texts, one shingle set each."""
        holders: dict[tuple[str, ...], list[int]] = {}
        for index, grams in enumerate(documents):
            for gram in grams:
                holders.setdefault(gram, []).append(index)
        return cls(holders)

    def held_part(self, grams: Shingles) -> Shingles:
        """The shingles of a unit's `grams` that its holders hold, the documents that each
        hold at least `STANDING_RECURRENCE_CONTAINMENT` of `grams`; empty exactly when no
        document holds the unit."""
        counts = Counter(i for gram in grams for i in self.holders.get(gram, ()))
        need = _passages.STANDING_RECURRENCE_CONTAINMENT * len(grams)
        unit_holders = {i for i, n in counts.items() if n >= need}
        return frozenset(g for g in grams if not unit_holders.isdisjoint(self.holders.get(g, ())))


@dataclass(frozen=True)
class HouseUnitRecall:
    """One house unit, by its first `UNIT_HEAD_CHARS` characters, and the share of its shingles
    a draft holds."""

    text_head: str
    share: float


@dataclass(frozen=True)
class CopiedUnit:
    """One copied unit of a draft, by its first `UNIT_HEAD_CHARS` characters, and how many of
    its shingles are stray: in the holdout documents that hold the unit, and in neither the
    reference nor the authorised input."""

    text_head: str
    stray_shingles: int


@dataclass(frozen=True)
class LexicalOverlap:
    """The two lexical-overlap measures of one draft (`OverlapSources.measure`).

    `house_text_recall` is None when the holdout has no house shingles;
    `copied_text_contamination` is None only when the draft has no prose shingles, and is 0 for
    a draft without a copied unit. `draft_shingles` counts the draft's prose shingles.
    """

    house_text_recall: float | None
    copied_text_contamination: float | None
    draft_shingles: int
    stray_shingles: int
    house_units: tuple[HouseUnitRecall, ...]
    copied_units: tuple[CopiedUnit, ...]


@dataclass(frozen=True)
class OverlapSources:
    """What a holdout's drafts are measured against, in shingle form.

    A text's *prose units* are its paragraphs and list items (`_prose_units`); a draft's *prose
    shingles* are the union of its prose units' shingles. The *text shingles* of the reference,
    of authorised input (the run notes and instructions) and of each holdout section are
    `_text_shingles`: a superset of both the whole-text and the per-unit forms, so a sentence
    present on both sides always matches on this side. A *holdout document's text* is the union
    of its sections' text shingles. A prose unit is *held* when at least one holdout document's
    text contains at least `STANDING_RECURRENCE_CONTAINMENT` of its shingles; its *holders* are
    the documents that do. A *house unit* is a held prose unit of the reference of at least
    `STANDING_MIN_WORDS` words; the *house shingles* are the union of the house units' shingles
    minus authorised input. A draft's *copied unit* is a held prose unit of at least
    `STANDING_MIN_WORDS` words.
    """

    holdout: _HoldoutText
    reference: Shingles
    authorised: Shingles
    house_units: tuple[_ProseUnit, ...]
    house_shingles: Shingles

    @classmethod
    def from_texts(
        cls,
        *,
        reference_sections: Sequence[str],
        holdout_documents: Sequence[Sequence[str]],
        run_notes: str,
        instructions: str,
    ) -> OverlapSources:
        """`holdout_documents` hold each delivered document of the holdout as its section
        texts; curator notes are not holdout text, so the caller leaves them out."""
        holdout = _HoldoutText.of(
            [frozenset().union(*map(_text_shingles, sections)) for sections in holdout_documents]
        )
        authorised = _text_shingles(run_notes) | _text_shingles(instructions)
        house = tuple(
            unit
            for section in reference_sections
            for unit in _prose_units(section)
            if unit.long and holdout.held_part(unit.grams)
        )
        return cls(
            holdout=holdout,
            reference=frozenset().union(*map(_text_shingles, reference_sections)),
            authorised=authorised,
            house_units=house,
            house_shingles=frozenset().union(*(u.grams for u in house)) - authorised,
        )

    def measure(self, draft: str) -> LexicalOverlap:
        """*House-text recall*: the share of the house shingles among the draft's prose
        shingles. *Copied-text contamination*: the *stray shingles*, the union over the copied
        units of each unit's shingles that its holders' text holds and neither the reference's
        nor authorised input's text shingles hold, as a share of the draft's prose shingles.
        Words a drafter added to a copied passage are stray only if the document it was copied
        from holds them. Both stray shingles and house shingles are subsets of what they are
        divided by, so both measures lie in 0..1."""
        units = _prose_units(draft)
        draft_grams: Shingles = frozenset().union(*(u.grams for u in units))
        recall = (
            len(self.house_shingles & draft_grams) / len(self.house_shingles)
            if self.house_shingles
            else None
        )
        known = self.reference | self.authorised
        copied: list[CopiedUnit] = []
        stray: set[tuple[str, ...]] = set()
        for unit in units:
            part = self.holdout.held_part(unit.grams) if unit.long else frozenset()
            if part:
                unit_stray = part - known
                stray |= unit_stray
                copied.append(CopiedUnit(unit.text[:UNIT_HEAD_CHARS], len(unit_stray)))
        contamination = len(stray) / len(draft_grams) if draft_grams else None
        return LexicalOverlap(
            house_text_recall=recall,
            copied_text_contamination=contamination,
            draft_shingles=len(draft_grams),
            stray_shingles=len(stray),
            house_units=tuple(
                HouseUnitRecall(
                    u.text[:UNIT_HEAD_CHARS], _passages.containment(u.grams, draft_grams)
                )
                for u in self.house_units
            ),
            copied_units=tuple(copied),
        )


# Rubric.


def score_rubric(draft: str, rubric: Sequence[RubricCheck]) -> dict[str, bool]:
    return {check.check_id: check.passes(draft) for check in rubric}


InputSource = Literal["run_notes", "instructions", "holdout", "none"]


@dataclass(frozen=True)
class RequiredGroupSource:
    """Where one of a required group's alternatives first occurs, in the order run notes,
    instructions, holdout (any section of the holdout snapshot); `none` when in no input."""

    alternatives: tuple[str, ...]
    found_in: InputSource


@dataclass(frozen=True)
class RubricDiagnostic:
    """The input sources of one rubric check's required groups. A group found in no input is a
    label for the pack author, not a verdict that a draft cannot derive the phrase."""

    check_id: str
    groups: tuple[RequiredGroupSource, ...]


def rubric_diagnostics(
    rubric: Sequence[RubricCheck],
    *,
    run_notes: str,
    instructions: str,
    holdout_sections: Sequence[str],
) -> tuple[RubricDiagnostic, ...]:
    def source(group: tuple[str, ...]) -> InputSource:
        if satisfies_groups(run_notes, (group,)):
            return "run_notes"
        if satisfies_groups(instructions, (group,)):
            return "instructions"
        if any(satisfies_groups(s, (group,)) for s in holdout_sections):
            return "holdout"
        return "none"

    return tuple(
        RubricDiagnostic(
            check.check_id,
            tuple(RequiredGroupSource(group, source(group)) for group in check.require_groups),
        )
        for check in rubric
    )


# Exemplars in context.

# The rules of the exemplars-in-context arm: the control that drafts from the documents the pack
# author chose, whole and in the prompt. It measures what full context of human-selected
# exemplars achieves, so it is the fixed baseline the arkive_mcp arm is compared against across
# changes to the server. This text is therefore frozen on its own: it does not follow
# `arkive_rag.server.INSTRUCTIONS` or `arkive_rag.snapshot.DRAFTING_GUIDANCE`, and a test pins it
# word for word apart from the server's drafting surface. Moving it with the server would change
# the control together with the treatment and confound every before/after comparison. Change it
# only as a deliberate change of baseline, and rerun the baseline when you do.
EXEMPLAR_DRAFTING_RULES = """\
The prompt holds prior deliverables of the same collection, each labelled with its file path,
followed by the run notes of the new project.
- Follow the structure and conventions the documents share, and avoid what any reviewer
  feedback among them criticises.
- Every fact about the new project (its client and other parties, people, places, dates,
  quantities, budgets, targets, findings, identifiers, file names) comes only from the run
  notes. This rule takes precedence over the next ones and over anything in the documents.
- Mirror the documents' heading levels and generic table column headers. Do not reuse their
  project-specific headings or facts.
- The documents' house wording (organisation or tool background, method references,
  acknowledgements, disclaimers, standing caveats) is the collection's own text. Reproduce it
  where it applies, with its wording, and keep its organisation names, method references and
  attributions as written. Adapt or leave out only the part that belongs to one particular
  project (its client, site, dates, quantities or results), using the run notes. Reuse other
  wording, names, references and attributions from the documents only within that house
  wording.
- Where the run notes enumerate items (files, deliverables, parameters, results), list every
  item in full, never as patterns or "etc.".
- Write a standalone deliverable: no file paths of the documents, and no mention of the
  documents in the prompt."""


def exemplars_prompt(documents: Sequence[tuple[str, str]], run_notes: str) -> str:
    """The exemplars-in-context prompt: each `(path, markdown)` document, labelled with its path
    and without visual transcripts, then the run notes."""
    blocks = [
        f"===== DOCUMENT {path} =====\n{_passages.without_asides(markdown).strip()}"
        for path, markdown in documents
    ]
    return "\n\n".join([*blocks, f"RUN NOTES:\n{run_notes}"])


# Raw files.

# Added to the holdout's instructions for the raw_files arm, which otherwise has only the
# tools' own descriptions to tell it what the files are for. Its authorised inputs are the
# ones the blind judge accepts (`DRAFT_JUDGE_INSTRUCTIONS`): the task instructions and the run
# notes, so a role or company the instructions give is a project fact, not an invention.
RAW_FILES_DRAFTING_HINT = """\
Use the file tools to find relevant prior deliverables and reviewer feedback in the
collection. Take every fact about the new project only from these instructions and the run
notes, and write a standalone deliverable without file paths."""


# Exemplar selection.


@dataclass(frozen=True)
class ExemplarSelection:
    """How `prepare_draft`, given the holdout's instructions and run notes and the reference's
    deliverable type, surfaces the documents the pack author named in `exemplar_paths`.

    `oracle_exemplar_paths` are the named deliverables and `oracle_feedback_paths` the named
    feedback documents. When the reference is a feedback document, or every named document is
    one, the holdout drafts a review, so every named document is a deliverable to surface and
    none is a feedback document. `requested_type` is the deliverable type passed to
    `prepare_draft`: the reference's, when the holdout snapshot has that type, else None.

    Every share counts version families once, so naming two versions of one deliverable
    counts as one, and a superseded document counts through the member that replaced it.
    `oracle_exemplar_families` and `oracle_feedback_families` count the named families. A named
    deliverable family is surfaced when one of its documents is among the packet's exemplars
    (`exemplar_recall_at_packet`) or among its exemplars and other candidates
    (`exemplar_recall_at_candidates`); each recall is None without a named deliverable.
    `feedback_attached` is the share of named feedback families that a packet lesson comes
    from, None without one. `query_embeddings` counts the query embeddings the call made."""

    oracle_exemplar_paths: tuple[str, ...]
    oracle_feedback_paths: tuple[str, ...]
    oracle_exemplar_families: int
    oracle_feedback_families: int
    requested_type: str | None
    packet_exemplar_paths: tuple[str, ...]
    packet_candidate_paths: tuple[str, ...]
    exemplar_recall_at_packet: float | None
    exemplar_recall_at_candidates: float | None
    feedback_attached: float | None
    query_embeddings: int


def exemplar_selection(
    built: TopicSnapshot, holdout: Holdout, reference: DocumentInfo, embedder: CountingEmbedder
) -> ExemplarSelection:
    """One deterministic `prepare_draft` call on the holdout snapshot `built`, compared with
    `holdout.exemplar_paths`; no model writes anything. `reference` is the held-out
    deliverable in the full snapshot: its type is requested the way an agent drafting that
    deliverable would request it, so an exemplar of another type cannot set the lessons'
    type."""
    by_path = {d.rel_path: d for d in built.list_documents()}
    named = [by_path[p] for p in holdout.exemplar_paths]
    types = {
        fold_facet_label(t.deliverable_type): t.deliverable_type for t in built.type_summaries()
    }
    requested = types.get(fold_facet_label(reference.deliverable_type))
    with embedder.recording_run() as embeddings:
        packet = built.prepare_draft(
            f"{holdout.instructions}\n\n{holdout.run_notes}",
            embedder,
            deliverable_type=requested,
        )
    drafts_review = reference.is_feedback or all(d.is_feedback for d in named)
    oracle = named if drafts_review else [d for d in named if not d.is_feedback]
    feedback = [] if drafts_review else [d for d in named if d.is_feedback]
    oracle_families = {d.family_id for d in oracle}
    feedback_families = {d.family_id for d in feedback}
    candidates = [built.document(c.doc_id) for c in packet.other_candidates]
    exemplar_families = {built.document(e.doc_id).family_id for e in packet.exemplars}
    candidate_families = exemplar_families | {c.family_id for c in candidates}
    lesson_families = {built.document(lesson.source_doc_id).family_id for lesson in packet.lessons}

    def share(surfaced: set[str], named_families: set[str]) -> float | None:
        if not named_families:
            return None
        return len(named_families & surfaced) / len(named_families)

    return ExemplarSelection(
        oracle_exemplar_paths=tuple(d.rel_path for d in oracle),
        oracle_feedback_paths=tuple(d.rel_path for d in feedback),
        oracle_exemplar_families=len(oracle_families),
        oracle_feedback_families=len(feedback_families),
        requested_type=requested,
        packet_exemplar_paths=tuple(e.rel_path for e in packet.exemplars),
        packet_candidate_paths=tuple(c.rel_path for c in candidates),
        exemplar_recall_at_packet=share(exemplar_families, oracle_families),
        exemplar_recall_at_candidates=share(candidate_families, oracle_families),
        feedback_attached=share(lesson_families, feedback_families),
        query_embeddings=embeddings.calls,
    )


# Blind judge.


def blind_assignment(arms: Sequence[str], seed: int = BLIND_SEED) -> dict[str, str]:
    """Map Candidate_N labels to arm names in a seeded random order."""
    shuffled = list(arms)
    random.Random(seed).shuffle(shuffled)
    return {f"Candidate_{i}": arm for i, arm in enumerate(shuffled, start=1)}


class CandidateAssessment(BaseModel):
    candidate: str
    structure_and_house_style: Score10
    technical_rigor: Score10
    limitations_and_review_readiness: Score10
    factual_discipline_no_contamination: Score10
    editing_effort_saved: Score10
    strengths: list[str]
    defects: list[str]


class DraftBlindComparison(BaseModel):
    candidates: list[CandidateAssessment]
    ranking: list[str]

    def label_problems(self, labels: Sequence[str]) -> list[str]:
        """Why this comparison does not cover the round: its assessments and its ranking must
        each name exactly `labels`, the labels sent to the judge, once each."""
        problems = []
        for part, named in (
            ("candidates", [c.candidate for c in self.candidates]),
            ("ranking", self.ranking),
        ):
            missing = [label for label in labels if label not in named]
            unknown = sorted({n for n in named if n not in labels})
            repeated = sorted({n for n in named if n in labels and named.count(n) > 1})
            for what, names in (("missing", missing), ("unknown", unknown), ("repeated", repeated)):
                if names:
                    problems.append(f"{part} {what} {names}")
        return problems


DRAFT_JUDGE_INSTRUCTIONS = """\
You review anonymous drafts of one deliverable, written from the same task instructions and
run notes, against the reference deliverable the project team actually delivered. The task
instructions and the run notes are the authorised inputs: a fact, person, role or organisation
that either of them gives, such as the author's name, role or company, is not invented and
not contamination. The reference is given as text with its figures removed; figures, charts
and maps are out of scope. Score each candidate from 1 to 10 on structure_and_house_style,
technical_rigor, limitations_and_review_readiness, factual_discipline_no_contamination
(penalise any value that contradicts the authorised inputs or belongs to another project) and
editing_effort_saved (how much of the reference a reviewer would not need to rewrite). Give
concrete strengths and defects, then rank the candidates best first by their labels."""


@dataclass(frozen=True)
class DraftComparison:
    """Blind side-by-side review of one round's completed drafts against the reference
    deliverable, whose visual transcripts the caller has removed. `task_instructions` are the
    holdout's instructions, which every arm drafted under."""

    operation: ClassVar[str] = "judge_drafts"
    output: ClassVar[type[DraftBlindComparison]] = DraftBlindComparison
    effort: ClassVar[Effort] = EVAL_EFFORT
    instructions: ClassVar[str] = DRAFT_JUDGE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    task_instructions: str
    run_notes: str
    reference_markdown: str
    drafts: tuple[tuple[str, str], ...]

    def text(self) -> str:
        candidates = "\n\n".join(f"===== {label} =====\n{text}" for label, text in self.drafts)
        return (
            f"TASK INSTRUCTIONS:\n{self.task_instructions}\n\nRUN NOTES:\n{self.run_notes}"
            f"\n\nREFERENCE DELIVERABLE:\n{self.reference_markdown}\n\nCANDIDATES:\n{candidates}"
        )


# The holdout run.


async def run_layer3_holdout(
    gateway: OpenAIGateway,
    *,
    holdout_gateway: LanguageGateway,
    full_snapshot_path: Path,
    corpus_dir: Path,
    embedder: CountingEmbedder,
    output_dir: Path,
    holdout: Holdout,
    drafts: int,
    render_cache: RenderCache | None = None,
) -> dict[str, object]:
    """Draft one held-out deliverable in `drafts` rounds; only completed drafts are scored.

    The holdout snapshot is built once and written to `output_dir/holdout.arkive.db`.
    `holdout_gateway` compiles it, with `render_cache` when one is given; it must embed in the
    full snapshot's embedding space, because `embedder` serves both. The reference and the
    included files are checked before any file is copied or any model is called. Every round
    runs `zero_shot`, `chunk_search`, `raw_files` and `arkive_mcp`, and `exemplars_in_context`
    unless the holdout lists no `exemplar_paths` (it is then reported in `skipped_arms`), and
    makes one blind judge call when any draft completed. `raw_files` reads the included files
    through the raw-file tools of layer 2, served from the pages the holdout build stored
    (`holdout_raw_corpus`). With `exemplar_paths`, `selection` is the `exemplar_selection` made
    before the first round; without, it is None. Each round's `blind_drafts` maps Candidate
    labels to draft text; the caller writes them out.
    """
    if drafts < 1:
        raise ValueError(f"drafts must be at least 1, got {drafts}")
    with TopicSnapshot.open(full_snapshot_path) as full:
        reference = full.document(holdout.reference_doc_id)
        check_reference_held_out(holdout, reference)
        reference_sections = section_markdowns(full, reference.doc_id)
    if not reference_sections:
        raise ValueError(f"reference document {reference.doc_id} has no sections")
    reference_markdown = _passages.without_asides("\n\n".join(reference_sections))
    holdout_path = output_dir / "holdout.arkive.db"
    # A worker thread: compiling may start its own event loop for agent runs, and this
    # coroutine's loop is already running.
    included = await anyio.to_thread.run_sync(
        compile_holdout_snapshot,
        corpus_dir,
        holdout_path,
        holdout_gateway,
        holdout,
        reference,
        render_cache,
    )

    with TopicSnapshot.open(holdout_path) as built:
        documents = built.list_documents()
        raw = holdout_raw_corpus(holdout, reference, included, built)
        # Release the included files' bytes before drafting; `raw` keeps Markdown text only.
        del included
        # Each document's sections, read once. A document without sections, such as an empty
        # curator note, contributes none.
        sections = {d.rel_path: section_markdowns(built, d.doc_id) for d in documents}
        exemplars = []
        for path in holdout.exemplar_paths:
            if not sections[path]:
                raise ValueError(f"exemplar document {path} has no sections")
            exemplars.append((path, "\n\n".join(sections[path])))
        selection = exemplar_selection(built, holdout, reference, embedder) if exemplars else None
        sources = OverlapSources.from_texts(
            reference_sections=reference_sections,
            holdout_documents=[sections[d.rel_path] for d in documents if not d.is_curator_note],
            run_notes=holdout.run_notes,
            instructions=holdout.instructions,
        )
        diagnostics = rubric_diagnostics(
            holdout.rubric,
            run_notes=holdout.run_notes,
            instructions=holdout.instructions,
            holdout_sections=[m for markdowns in sections.values() for m in markdowns],
        )

        def search_chunks(args: dict[str, Any]) -> str:
            hits = built.search(str(args["query"]), embedder, top_k=5).hits
            return json.dumps(
                [
                    [h.chunk_id, h.doc_title, f"{h.page_start}-{h.page_end}", h.content]
                    for h in hits
                ],
                ensure_ascii=False,
            )

        chunk_search = Tool(
            "search_chunks",
            "Search prior project documents; returns the top 5 matching text chunks.",
            object_schema({"query": {"type": "string"}}, ["query"]),
            search_chunks,
        )
        notes_prompt = f"RUN NOTES:\n{holdout.run_notes}"
        async with arkive_mcp_tools(holdout_path, embedder) as (mcp_tools, mcp_notes):
            # Arm: (tools, instructions, prompt).
            arms: dict[DraftArm, tuple[Sequence[Tool], str, str]] = {
                "zero_shot": ((), holdout.instructions, notes_prompt),
                "chunk_search": ((chunk_search,), holdout.instructions, notes_prompt),
                "raw_files": (
                    raw.tools,
                    f"{holdout.instructions}\n\n{RAW_FILES_DRAFTING_HINT}",
                    notes_prompt,
                ),
                "arkive_mcp": (
                    mcp_tools,
                    with_server_instructions(holdout.instructions, mcp_notes),
                    notes_prompt,
                ),
            }
            skipped: dict[str, str] = {}
            if exemplars:
                arms["exemplars_in_context"] = (
                    (),
                    f"{holdout.instructions}\n\n{EXEMPLAR_DRAFTING_RULES}",
                    exemplars_prompt(exemplars, holdout.run_notes),
                )
            else:
                skipped["exemplars_in_context"] = "the holdout lists no exemplar_paths"
            rounds = []
            for number in range(1, drafts + 1):
                runs: dict[str, AgentRun] = {}
                for arm, (tools, instructions, prompt) in arms.items():
                    runs[arm] = await run_agent(
                        gateway,
                        operation=f"draft_{arm}",
                        instructions=instructions,
                        prompt=prompt,
                        tools=tools,
                        max_turns=DRAFT_MAX_TURNS,
                        embedder=embedder,
                    )
                rounds.append(
                    await _finish_round(gateway, holdout, reference_markdown, sources, number, runs)
                )
    return {
        "holdout_id": holdout.holdout_id,
        "holdout_snapshot": str(holdout_path),
        "holdout_documents": [d.rel_path for d in documents],
        "exemplar_paths": list(holdout.exemplar_paths),
        "selection": selection,
        "skipped_arms": skipped,
        "house_units": len(sources.house_units),
        "house_shingles": len(sources.house_shingles),
        "rubric_diagnostics": diagnostics,
        "rounds": rounds,
    }


async def _finish_round(
    gateway: OpenAIGateway,
    holdout: Holdout,
    reference_markdown: str,
    sources: OverlapSources,
    number: int,
    runs: dict[str, AgentRun],
) -> dict[str, object]:
    """Score one round's completed drafts and judge them in one blind call; the blind order
    is seeded by the round number.

    `blind_comparison` is the judge's output only when it covers the round; otherwise it is
    None and `blind_comparison_missing` says why: no draft completed, the judge call failed
    with a `ProviderError` (invalid or refused output, output still truncated after the
    retry, a provider failure), or the judge's output named the wrong labels. The round keeps
    its drafts and measures either way. The spend limit stops the evaluation."""
    completed = [arm for arm, run in runs.items() if run.status == "completed"]
    key = blind_assignment(completed, seed=BLIND_SEED + number)
    blind_drafts = {label: runs[arm].answer for label, arm in key.items()}
    comparison: DraftBlindComparison | None = None
    missing: str | None = "no arm completed, so no blind comparison ran"
    if blind_drafts:
        try:
            judged = await anyio.to_thread.run_sync(
                gateway.run,
                DraftComparison(
                    holdout.instructions,
                    holdout.run_notes,
                    reference_markdown,
                    tuple(blind_drafts.items()),
                ),
            )
        except BudgetExceededError:
            raise
        except ProviderError as exc:
            missing = f"the judge call failed: {exc}"
        else:
            problems = judged.label_problems(list(key))
            if problems:
                missing = f"the judge output does not cover the round: {'; '.join(problems)}"
            else:
                comparison, missing = judged, None
    return {
        "round": number,
        "arms": {
            arm: {
                "run": run,
                "rubric": score_rubric(run.answer, holdout.rubric) if arm in completed else None,
                "overlap": sources.measure(run.answer) if arm in completed else None,
            }
            for arm, run in runs.items()
        },
        "blind_key": key,
        "blind_drafts": blind_drafts,
        "blind_comparison": comparison,
        "blind_comparison_missing": missing,
    }
