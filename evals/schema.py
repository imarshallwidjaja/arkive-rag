"""The eval pack: a JSON file of benchmark cases for one corpus, and the text matching it uses.

Packs hold corpus-specific facts, so private packs live outside this repository next to their
corpus (see docs/architecture.md). A *group* is a list of alternative substrings; text satisfies
a group when it contains any alternative, and satisfies a list of groups when it satisfies every
group. Matching is NFKC, case-, dash-, whitespace- and thousands-separator-insensitive, and
folds an x or multiplication sign (U+00D7) between digits to " x ", so "59,494 km²" matches
"59494 km2" and "1800 x 1800" matches "1800x1800".
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

_DASHES = re.compile(r"[\u2010-\u2015\u2212]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")
_SPACES = re.compile(r"\s+")
_DIMENSION_SIGN = re.compile(r"(?<=\d)\s*[x\u00d7]\s*(?=\d)")


def normalize_for_match(text: str) -> str:
    folded = _DASHES.sub("-", unicodedata.normalize("NFKC", text)).casefold()
    folded = _DIMENSION_SIGN.sub(" x ", folded)
    return _SPACES.sub(" ", _THOUSANDS.sub("", folded)).strip()


NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Group = Annotated[tuple[NonBlank, ...], Field(min_length=1)]
Groups = tuple[Group, ...]


def satisfies_groups(text: str, groups: Groups) -> bool:
    haystack = normalize_for_match(text)
    return all(any(normalize_for_match(alt) in haystack for alt in group) for group in groups)


class _PackModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievalCase(_PackModel):
    """Layer 1: the top-k chunks must jointly satisfy every group and include every expected doc."""

    case_id: NonBlank
    category: NonBlank
    query: NonBlank
    required_groups: Annotated[Groups, Field(min_length=1)]
    expected_docs: tuple[NonBlank, ...] = ()
    project: NonBlank | None = None
    stage: NonBlank | None = None


class QACase(_PackModel):
    """Layer 2: `required_groups` scores phrase coverage; the judge sees `ground_truth_facts`
    and grades citations against `source_anchors`, the documents and pages that hold the facts
    (for example "p101/memo.pdf (doc_001) page 5")."""

    case_id: NonBlank
    question: NonBlank
    ground_truth_facts: Annotated[tuple[NonBlank, ...], Field(min_length=1)]
    required_groups: Annotated[Groups, Field(min_length=1)]
    source_anchors: tuple[NonBlank, ...] = ()


class RubricCheck(_PackModel):
    """Passes when the memo satisfies every `require_groups` group and contains no phrase in
    `forbid_any`.

    Forbid only claims that cannot appear in a faithful memo. A value that a faithful memo may
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
        return self

    def passes(self, memo: str) -> bool:
        if not satisfies_groups(memo, self.require_groups):
            return False
        text = normalize_for_match(memo)
        return not any(normalize_for_match(phrase) in text for phrase in self.forbid_any)


class HoldoutMemoTask(_PackModel):
    """Layer 3: draft a held-out deliverable from run notes using only the included corpus paths.

    `included_paths` are corpus-relative files or folders copied into the holdout corpus. The
    holdout corpus must contain no project or path containing a `held_out_markers` entry.
    `reference_doc_id` names the delivered document, in the full snapshot, that the blind judge
    compares drafts against; its project or path must contain a held-out marker.
    """

    included_paths: Annotated[tuple[NonBlank, ...], Field(min_length=1)]
    held_out_markers: Annotated[tuple[NonBlank, ...], Field(min_length=1)]
    reference_doc_id: NonBlank
    memo_instructions: NonBlank
    run_notes: NonBlank
    rubric: Annotated[tuple[RubricCheck, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _paths_are_safe_and_exclude_held_out_material(self) -> Self:
        for rel in self.included_paths:
            path = PurePosixPath(rel)
            if path.is_absolute() or ".." in path.parts or "\\" in rel:
                raise ValueError(f"included path {rel!r} must be relative to the corpus root")
            if any(marker in rel for marker in self.held_out_markers):
                raise ValueError(f"included path {rel!r} contains a held-out marker")
        _require_unique("rubric check_id", [c.check_id for c in self.rubric])
        return self


class EvalPack(_PackModel):
    format: Literal["arkive-eval-pack/1"]
    name: NonBlank
    description: NonBlank
    # When set, the runner refuses a snapshot built from any other corpus.
    corpus_fingerprint: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    retrieval_cases: tuple[RetrievalCase, ...] = ()
    qa_cases: tuple[QACase, ...] = ()
    holdout_memo: HoldoutMemoTask | None = None

    @model_validator(mode="after")
    def _unique_case_ids(self) -> Self:
        _require_unique("retrieval case_id", [c.case_id for c in self.retrieval_cases])
        _require_unique("qa case_id", [c.case_id for c in self.qa_cases])
        return self

    @classmethod
    def load(cls, path: Path) -> EvalPack:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def _require_unique(what: str, values: list[str]) -> None:
    duplicates = sorted({v for v in values if values.count(v) > 1})
    if duplicates:
        raise ValueError(f"duplicate {what}: {duplicates}")
