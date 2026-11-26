"""The eval pack: a JSON file of benchmark cases for one corpus, and the text matching it uses.

Packs hold corpus-specific facts, so private packs live outside this repository next to their
corpus (see docs/architecture.md). A *group* is a list of alternative substrings; text satisfies
a group when it contains any alternative, and satisfies a list of groups when it satisfies every
group. Matching is NFKC, case-, dash-, whitespace- and thousands-separator-insensitive, ignores
Markdown emphasis, and folds an x or multiplication sign (U+00D7) between digits to " x ", so
"59,494 km²" matches "59494 km2", "does **not** hold" matches "does not hold" and "1800 x 1800"
matches "1800x1800". Emphasis is backticks, `*` runs that touch a non-space character (a spaced
`2 * 3` keeps its sign) and `_` runs with a letter or digit on one side only, so `_italic_` and
`__bold__` lose their underscores while `harbour_dry` and `survey__v2.tif` keep every one.
Table pipes stay, so a group can test a table row. Every phrase must keep some text once
normalized: a markup-only phrase such as "**" would otherwise match every text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from arkive_rag.models import fold_text

_DASHES = re.compile(r"[\u2010-\u2015\u2212]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")
_SPACES = re.compile(r"\s+")
_DIMENSION_SIGN = re.compile(r"(?<=\d)\s*[x\u00d7]\s*(?=\d)")
# `[^\W_]` is a letter or digit. An `_` run is matched whole: its outer lookaround refuses a
# word character, `_` included, so backtracking cannot strip part of the run in `survey__v2`.
_MATCH_EMPHASIS = re.compile(r"`|\*+(?=\S)|(?<=\S)\*+|(?<!\w)_+(?=[^\W_])|(?<=[^\W_])_+(?!\w)")


def normalize_for_match(text: str) -> str:
    plain = _MATCH_EMPHASIS.sub("", unicodedata.normalize("NFKC", text))
    folded = _DASHES.sub("-", plain).casefold()
    folded = _DIMENSION_SIGN.sub(" x ", folded)
    return _SPACES.sub(" ", _THOUSANDS.sub("", folded)).strip()


NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Group = Annotated[tuple[NonBlank, ...], Field(min_length=1)]
Groups = tuple[Group, ...]


def _require_matchable(owner: str, phrases: Iterable[str]) -> None:
    """Refuse a phrase that normalizes to nothing: `"" in text` holds for every text, so it
    would make a required group always pass and a forbidden phrase always fail."""
    for phrase in phrases:
        if not normalize_for_match(phrase):
            raise ValueError(f"{owner} phrase {phrase!r} is empty once markup is ignored")


def _flat(groups: Groups) -> list[str]:
    return [alt for group in groups for alt in group]


def satisfies_groups(text: str, groups: Groups) -> bool:
    haystack = normalize_for_match(text)
    return all(any(normalize_for_match(alt) in haystack for alt in group) for group in groups)


def held_out_markers_in(rel_path: str, markers: Sequence[str]) -> list[str]:
    """The markers `rel_path` contains, compared in the shared `fold_text` form (NFKC, Unicode
    dashes as `-`, case-folded), so `P103` and `p103` are the same marker."""
    folded = fold_text(rel_path)
    return [m for m in markers if fold_text(m) in folded]


class _PackModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievalCase(_PackModel):
    """Layer 1: the top-k chunks must jointly satisfy every group and include every expected doc."""

    case_id: NonBlank
    category: NonBlank
    query: NonBlank
    required_groups: Annotated[Groups, Field(min_length=1)]
    expected_docs: tuple[NonBlank, ...] = ()
    # Facet name to one value or alias, as `TopicSnapshot.resolve_filters` takes them. The
    # snapshot's facets are induced from its corpus: `arkive-rag inspect` lists them.
    filters: dict[NonBlank, NonBlank] = {}

    @model_validator(mode="after")
    def _phrases_matchable(self) -> Self:
        _require_matchable(f"retrieval case {self.case_id!r}", _flat(self.required_groups))
        return self


QAKind = Literal[
    "lookup",
    "multi_document",
    "aggregation",
    "abstention",
    "conflict",
    "vocabulary",
    # The value the latest issued version states, where earlier drafts state another.
    "current_value",
    # The final value together with the draft or review history that preceded it.
    "version_history",
]


# A JSON integer: no boolean, numeric string or float such as 3.0.
PageNumber = Annotated[int, Field(gt=0, strict=True)]


class SourceAnchor(_PackModel):
    """Where a Q&A case's facts are held: pages `first..last` of one document of the snapshot
    (1-based, inclusive), or the whole document when `pages` is null (Markdown sources have no
    pages). The runner checks before any model call that the document and pages exist and, for
    every kind but `abstention`, that some section overlaps them."""

    doc_id: NonBlank
    pages: tuple[PageNumber, PageNumber] | None

    @model_validator(mode="after")
    def _pages_ascend(self) -> Self:
        if self.pages is not None and self.pages[0] > self.pages[1]:
            raise ValueError(f"source anchor {self.doc_id} pages {list(self.pages)} run backwards")
        return self


class QACase(_PackModel):
    """Layer 2: `required_groups` and `forbid_any` score phrase coverage (`passes_qa_phrases`);
    the judge sees `ground_truth_facts`, the snapshot's text of the `source_anchors`, and
    grades citations against the anchors. Every `kind` but `abstention` (the corpus does not
    hold the answer) needs an anchor.

    `forbid_any` lists phrases a correct answer never states, such as a superseded draft
    value for a `current_value` case. As for a rubric check, a value a correct answer may
    mention in contrast belongs in a required group, not here."""

    case_id: NonBlank
    kind: QAKind
    question: NonBlank
    ground_truth_facts: Annotated[tuple[NonBlank, ...], Field(min_length=1)]
    required_groups: Annotated[Groups, Field(min_length=1)]
    source_anchors: tuple[SourceAnchor, ...] = ()
    forbid_any: tuple[NonBlank, ...] = ()

    @model_validator(mode="after")
    def _anchored_unless_abstention(self) -> Self:
        if self.kind != "abstention" and not self.source_anchors:
            raise ValueError(
                f"qa case {self.case_id!r} of kind {self.kind!r} has no source_anchors"
            )
        _require_matchable(
            f"qa case {self.case_id!r}", [*_flat(self.required_groups), *self.forbid_any]
        )
        return self


def passes_qa_phrases(answer: str, case: QACase) -> tuple[bool, tuple[str, ...]]:
    """Whether `answer` covers the case's phrases, and the `forbid_any` phrases it states.

    It passes when it satisfies every required group and states no forbidden phrase."""
    text = normalize_for_match(answer)
    hits = tuple(p for p in case.forbid_any if normalize_for_match(p) in text)
    return satisfies_groups(answer, case.required_groups) and not hits, hits


class RubricCheck(_PackModel):
    """Passes when the draft satisfies every `require_groups` group and contains no phrase in
    `forbid_any`.

    Forbid only claims that cannot appear in a faithful draft. A value that a faithful draft may
    legitimately mention, for example when contrasting with an earlier project, belongs in a
    required group of the right value, not in `forbid_any`.
    """

    check_id: NonBlank
    category: NonBlank
    description: NonBlank
    require_groups: Groups = ()
    forbid_any: tuple[NonBlank, ...] = ()

    @model_validator(mode="after")
    def _tests_something(self) -> Self:
        if not self.require_groups and not self.forbid_any:
            raise ValueError(f"rubric check {self.check_id!r} has no require_groups or forbid_any")
        _require_matchable(
            f"rubric check {self.check_id!r}", [*_flat(self.require_groups), *self.forbid_any]
        )
        return self

    def passes(self, draft: str) -> bool:
        if not satisfies_groups(draft, self.require_groups):
            return False
        text = normalize_for_match(draft)
        return not any(normalize_for_match(phrase) in text for phrase in self.forbid_any)


PACK_FORMAT = "arkive-eval-pack/3"


def _check_plain_relative_path(what: str, rel: str) -> None:
    path = PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts or "\\" in rel:
        raise ValueError(f"{what} {rel!r} must be relative to the corpus root")
    if rel != path.as_posix() or rel == ".":
        raise ValueError(
            f"{what} {rel!r} must be a plain corpus-relative path such as"
            f" {path.as_posix()!r}: no './', empty parts or trailing '/'"
        )


class Holdout(_PackModel):
    """Layer 3: draft a held-out deliverable from run notes using only the included corpus paths.

    `holdout_id` names the holdout's output folder `<out>/holdouts/<holdout_id>/`.
    `included_paths` are corpus-relative files or folders copied into the holdout corpus,
    written plainly (`p101` or `p101/memo.pdf`: no `./`, empty part or trailing `/`). A file is
    held-out material when its path contains a `held_out_markers` entry (compared case- and
    dash-insensitively) or its bytes equal the reference's; no included file may be. The guards
    infer nothing from folders and do not scan content: choose markers that cover every file of
    the held-out work (a marker may be a folder name), and leave out any curator note, summary
    or other document that restates held-out facts. `reference_doc_id` names the delivered
    document, in the full snapshot, that the blind judge compares drafts against; it must not
    be a curator note, and its path must contain a held-out marker. `exemplar_paths` are the
    included files the pack author names as this deliverable's exemplars and feedback
    documents; each must be a delivered document, not a curator note. `rubric` may be empty.
    """

    holdout_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    included_paths: Annotated[tuple[NonBlank, ...], Field(min_length=1)]
    held_out_markers: Annotated[tuple[NonBlank, ...], Field(min_length=1)]
    reference_doc_id: NonBlank
    instructions: NonBlank
    run_notes: NonBlank
    rubric: tuple[RubricCheck, ...] = ()
    exemplar_paths: tuple[NonBlank, ...] = ()

    @model_validator(mode="after")
    def _paths_are_safe_and_exclude_held_out_material(self) -> Self:
        for rel in self.included_paths:
            _check_plain_relative_path("included path", rel)
            if held_out_markers_in(rel, self.held_out_markers):
                raise ValueError(f"included path {rel!r} contains a held-out marker")
        for rel in self.exemplar_paths:
            _check_plain_relative_path("exemplar path", rel)
            if not any(rel == part or rel.startswith(f"{part}/") for part in self.included_paths):
                raise ValueError(f"exemplar path {rel!r} is not under any included path")
        _require_unique("exemplar path", list(self.exemplar_paths))
        _require_unique("rubric check_id", [c.check_id for c in self.rubric])
        return self


class ExpectedFamily(_PackModel):
    """One version family the snapshot should induce: its `members` as plain corpus-relative
    paths, and optionally the member that should be its latest version."""

    members: Annotated[tuple[NonBlank, ...], Field(min_length=2)]
    latest: NonBlank | None = None

    @model_validator(mode="after")
    def _members_are_plain_and_hold_latest(self) -> Self:
        for rel in self.members:
            _check_plain_relative_path("expected family member", rel)
        _require_unique("expected family member", list(self.members))
        if self.latest is not None and self.latest not in self.members:
            raise ValueError(f"expected family latest {self.latest!r} is not one of its members")
        return self


class EvalPack(_PackModel):
    format: Literal["arkive-eval-pack/3"]
    name: NonBlank
    description: NonBlank
    # When set, the runner refuses a snapshot built from any other corpus.
    corpus_fingerprint: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    retrieval_cases: tuple[RetrievalCase, ...] = ()
    qa_cases: tuple[QACase, ...] = ()
    holdouts: tuple[Holdout, ...] = ()
    # The non-singleton version families the snapshot should induce; the runner compares
    # them with the snapshot's families before any layer runs.
    expected_families: tuple[ExpectedFamily, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _format_first(cls, data: Any) -> Any:
        # Checked before the fields, so an older pack gets one message naming the format the
        # runner reads instead of an error for every field that changed.
        if isinstance(data, dict) and data.get("format") != PACK_FORMAT:
            raise ValueError(
                f"format {data.get('format')!r} is not supported; this runner reads"
                f" {PACK_FORMAT!r} packs only"
            )
        return data

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        _require_unique("retrieval case_id", [c.case_id for c in self.retrieval_cases])
        _require_unique("qa case_id", [c.case_id for c in self.qa_cases])
        _require_unique("holdout_id", [h.holdout_id for h in self.holdouts])
        _require_unique(
            "expected family member across families",
            [m for f in self.expected_families for m in f.members],
        )
        return self

    @classmethod
    def load(cls, path: Path) -> EvalPack:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def _require_unique(what: str, values: list[str]) -> None:
    duplicates = sorted({v for v in values if values.count(v) > 1})
    if duplicates:
        raise ValueError(f"duplicate {what}: {duplicates}")
