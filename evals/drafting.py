"""Layer 3: draft held-out deliverables, measure the drafts and judge them blind.

For each holdout the harness builds a holdout snapshot without the held-out material, then runs
draft rounds. Every round runs the holdout's arms from its run notes, scores each completed
draft with the rubric and two lexical-overlap measures, and sends the round's completed drafts
to one blind judge call. Agent runs go through `evals.agents.run_agent`; the judge goes through
`OpenAIGateway.run`.

The two measures compare wording, not meaning. They share with induction's standing passages
the unit, shingle and containment functions of `arkive_rag.passages` (`standing_units`, word
shingles of `fold_for_quote` text, `containment`, `without_visual_transcripts`) and its
thresholds (`STANDING_MIN_WORDS`, `STANDING_RECURRENCE_CONTAINMENT`). They differ from
recurring-candidate discovery (`standing_candidates`) in three ways: a unit loses its leading
list marker (`LIST_MARKER`) before it is folded; a unit is held when the union of every holdout
section's shingles contains enough of its shingles, where discovery tests each section of
another document on its own; and no folder rule applies, where discovery asks for recurrence in
another top-level folder when the exemplars span folders.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import anyio
from pydantic import BaseModel

from arkive_rag import passages as _passages
from arkive_rag.compiler import RenderCache, compile_corpus, discover_corpus, read_source
from arkive_rag.llm import BudgetExceededError, Effort, LanguageGateway, OpenAIGateway, Tool
from arkive_rag.models import DocumentInfo, ProviderError, fold_for_quote
from arkive_rag.snapshot import TopicSnapshot, write_snapshot
from evals.agents import (
    EVAL_EFFORT,
    AgentRun,
    CountingEmbedder,
    Score10,
    arkive_mcp_tools,
    object_schema,
    run_agent,
    with_server_instructions,
)
from evals.schema import Holdout, RubricCheck, held_out_markers_in, satisfies_groups

DRAFT_MAX_TURNS = 16
BLIND_SEED = 42

DraftArm = Literal["zero_shot", "chunk_search", "arkive_mcp", "exemplars_in_context"]


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
    holdout: Holdout, reference: DocumentInfo, documents: Sequence[DocumentInfo]
) -> None:
    """Fail before any drafting arm runs if held-out material reached the holdout snapshot."""
    leaked = {
        d.rel_path: reasons
        for d in documents
        if (reasons := _leak_reasons(holdout, reference, d.rel_path, d.sha256))
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
) -> None:
    """Build the holdout snapshot and publish it at `output_path` only after its guard passes.

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
                assert_holdout_clean(holdout, reference, built.list_documents())
            os.replace(unverified, output_path)
        except BaseException:
            unverified.unlink(missing_ok=True)
            raise


def section_markdowns(snapshot: TopicSnapshot, doc_id: str) -> list[str]:
    """One document's section Markdown in reading order; empty for a document without
    sections, such as an empty curator note."""
    return [snapshot.read_section(s.section_id).markdown for s in snapshot.outline(doc_id).sections]


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
        marker = _passages.LIST_MARKER.match(unit.text)
        folded = fold_for_quote(unit.text[marker.end() :] if marker else unit.text)
        units.append(
            _ProseUnit(unit.text, len(_passages.WORD.findall(folded)), _passages.shingles(folded))
        )
    return units


def _text_shingles(text: str) -> Shingles:
    """What a text carries anywhere: its whole-text shingles, tables and headings included and
    visual transcripts removed, together with its prose units' shingles. Folding the whole text
    joins a line-end hyphen or dash with the next unit's first word, and a unit keeps no list
    marker, so neither form alone holds every shingle the other does."""
    whole = _passages.shingles(fold_for_quote(_passages.without_visual_transcripts(text)))
    return whole.union(*(u.grams for u in _prose_units(text)))


def _held(grams: Shingles, holdout: Shingles) -> bool:
    return _passages.containment(grams, holdout) >= _passages.STANDING_RECURRENCE_CONTAINMENT


@dataclass(frozen=True)
class HouseUnitRecall:
    """One house unit, by its first `UNIT_HEAD_CHARS` characters, and the share of its shingles
    a draft holds."""

    text_head: str
    share: float


@dataclass(frozen=True)
class CopiedUnit:
    """One copied unit of a draft, by its first `UNIT_HEAD_CHARS` characters, and how many of
    its shingles are stray: in the holdout text, and in neither the reference nor the
    authorised input."""

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
    present on both sides always matches on this side. The *holdout text* is the union of the
    holdout sections' text shingles. A prose unit is *held* when the holdout text contains at
    least `STANDING_RECURRENCE_CONTAINMENT` of its shingles. A *house unit* is a held prose unit
    of the reference of at least `STANDING_MIN_WORDS` words; the *house shingles* are the union
    of the house units' shingles minus authorised input. A draft's *copied unit* is a held prose
    unit of at least `STANDING_MIN_WORDS` words.
    """

    holdout: Shingles
    reference: Shingles
    authorised: Shingles
    house_units: tuple[_ProseUnit, ...]
    house_shingles: Shingles

    @classmethod
    def from_texts(
        cls,
        *,
        reference_sections: Sequence[str],
        holdout_sections: Sequence[str],
        run_notes: str,
        instructions: str,
    ) -> OverlapSources:
        """`holdout_sections` are the section texts of the holdout's delivered documents;
        curator notes are not holdout text, so the caller leaves them out."""
        holdout: Shingles = frozenset().union(*map(_text_shingles, holdout_sections))
        authorised = _text_shingles(run_notes) | _text_shingles(instructions)
        house = tuple(
            unit
            for section in reference_sections
            for unit in _prose_units(section)
            if unit.long and _held(unit.grams, holdout)
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
        shingles. *Copied-text contamination*: the *stray shingles*, the union of the copied
        units' shingles that occur in the holdout text and in neither the reference's nor
        authorised input's text shingles, as a share of the draft's prose shingles. Both stray
        shingles and house shingles are subsets of what they are divided by, so both measures
        lie in 0..1."""
        units = _prose_units(draft)
        draft_grams: Shingles = frozenset().union(*(u.grams for u in units))
        recall = (
            len(self.house_shingles & draft_grams) / len(self.house_shingles)
            if self.house_shingles
            else None
        )
        copied = [u for u in units if u.long and _held(u.grams, self.holdout)]
        strays = [(u.grams & self.holdout) - self.reference - self.authorised for u in copied]
        stray: Shingles = frozenset().union(*strays)
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
            copied_units=tuple(
                CopiedUnit(u.text[:UNIT_HEAD_CHARS], len(s))
                for u, s in zip(copied, strays, strict=True)
            ),
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

# The drafting obligations of the MCP server's instructions (`arkive_rag.server.INSTRUCTIONS`),
# reworded for documents in the prompt instead of tools. A test pins word for word the drafting
# part of those instructions, the `get_blueprint` and `read_section` tool descriptions as the
# server publishes them, and this text, beside a map of the obligations, so an edit to any of
# them fails until the pins are updated together.
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
        f"===== DOCUMENT {path} =====\n{_passages.without_visual_transcripts(markdown).strip()}"
        for path, markdown in documents
    ]
    return "\n\n".join([*blocks, f"RUN NOTES:\n{run_notes}"])


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
You review anonymous drafts of one deliverable, written from the same run notes, against the
reference deliverable the project team actually delivered. The reference is given as text with
its figures removed; figures, charts and maps are out of scope. Score each candidate from 1 to
10 on structure_and_house_style, technical_rigor, limitations_and_review_readiness,
factual_discipline_no_contamination (penalise any value that contradicts the run notes or
belongs to another project) and editing_effort_saved (how much of the reference a reviewer
would not need to rewrite). Give concrete strengths and defects, then rank the candidates
best first by their labels."""


@dataclass(frozen=True)
class DraftComparison:
    """Blind side-by-side review of one round's completed drafts against the reference
    deliverable, whose visual transcripts the caller has removed."""

    operation: ClassVar[str] = "judge_drafts"
    output: ClassVar[type[DraftBlindComparison]] = DraftBlindComparison
    effort: ClassVar[Effort] = EVAL_EFFORT
    instructions: ClassVar[str] = DRAFT_JUDGE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    run_notes: str
    reference_markdown: str
    drafts: tuple[tuple[str, str], ...]

    def text(self) -> str:
        candidates = "\n\n".join(f"===== {label} =====\n{text}" for label, text in self.drafts)
        return (
            f"RUN NOTES:\n{self.run_notes}\n\nREFERENCE DELIVERABLE:\n{self.reference_markdown}"
            f"\n\nCANDIDATES:\n{candidates}"
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
    runs `zero_shot`, `chunk_search` and
    `arkive_mcp`, and `exemplars_in_context` unless the holdout lists no `exemplar_paths` (it is
    then reported in `skipped_arms`), and makes one blind judge call when any draft completed.
    Each round's `blind_drafts` maps Candidate labels to draft text; the caller writes them out.
    """
    if drafts < 1:
        raise ValueError(f"drafts must be at least 1, got {drafts}")
    with TopicSnapshot.open(full_snapshot_path) as full:
        reference = full.document(holdout.reference_doc_id)
        check_reference_held_out(holdout, reference)
        reference_sections = section_markdowns(full, reference.doc_id)
    if not reference_sections:
        raise ValueError(f"reference document {reference.doc_id} has no sections")
    reference_markdown = _passages.without_visual_transcripts("\n\n".join(reference_sections))
    holdout_path = output_dir / "holdout.arkive.db"
    # A worker thread: compiling may start its own event loop for agent runs, and this
    # coroutine's loop is already running.
    await anyio.to_thread.run_sync(
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
        # Each document's sections, read once. A document without sections, such as an empty
        # curator note, contributes none.
        sections = {d.rel_path: section_markdowns(built, d.doc_id) for d in documents}
        exemplars = []
        for path in holdout.exemplar_paths:
            if not sections[path]:
                raise ValueError(f"exemplar document {path} has no sections")
            exemplars.append((path, "\n\n".join(sections[path])))
        sources = OverlapSources.from_texts(
            reference_sections=reference_sections,
            holdout_sections=[
                m for d in documents if not d.is_curator_note for m in sections[d.rel_path]
            ],
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
            hits = built.search(str(args["query"]), embedder, top_k=5)
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
                DraftComparison(holdout.run_notes, reference_markdown, tuple(blind_drafts.items())),
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
