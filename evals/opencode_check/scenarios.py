"""Scenario files, OpenCode event streams, and the checks that compare one with the other.

A scenario file (`format` `arkive-opencode-scenarios/1`) lists prompts for an agent that has
only the Arkive MCP tools. Each scenario names the Arkive tools that must complete at least
once, optional expectations about tool arguments, the topics its calls must query, an
optional page whose image `view_page` must return, and answer phrase groups matched as in
eval packs (`evals.schema.satisfies_groups`). Every scenario also
requires that the agent called no tool outside the Arkive server and that the run ended with
an answer. Facet names are induced per build, so a scenario never names one: it can instead
require the answer to name every facet of its topics, read from the served snapshots. For the
same reason a drafting scenario never names a standing passage or a deliverable type: it names
a topic, and can require the answer to reproduce one of the passages of the blueprint its last
`get_blueprint` call on that topic returned.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arkive_rag.models import fold_for_quote
from evals.schema import Groups, NonBlank, normalize_for_match, satisfies_groups

ARKIVE_SERVER = "arkive"
ArkiveTool = Literal[
    "list_topics",
    "get_topic_guide",
    "get_blueprint",
    "list_documents",
    "get_document_outline",
    "search",
    "grep",
    "read_section",
    "view_page",
    "lookup_glossary",
]
ARKIVE_TOOLS: frozenset[str] = frozenset(get_args(ArkiveTool))


class _ScenarioModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NonEmptyArgument(_ScenarioModel):
    """Some completed call to one of `tools` passed a non-empty `argument`."""

    condition: Literal["non_empty"]
    tools: Annotated[tuple[ArkiveTool, ...], Field(min_length=1)]
    argument: NonBlank

    def label(self) -> str:
        return f"arguments: {'/'.join(self.tools)}.{self.argument} non_empty"


class InvalidFilterRecovery(_ScenarioModel):
    """A call to one of `tools` whose `filters` set a facet to `invalid_value` ended in a tool
    error, and a later call to the same tool on the same topic, setting that facet to another
    value, completed: the agent read the error and recovered."""

    condition: Literal["invalid_filter_recovery"]
    tools: Annotated[tuple[ArkiveTool, ...], Field(min_length=1)]
    invalid_value: NonBlank

    def label(self) -> str:
        return f"filters: {'/'.join(self.tools)} recovers from {self.invalid_value!r}"


ArgumentExpectation = Annotated[
    NonEmptyArgument | InvalidFilterRecovery, Field(discriminator="condition")
]


class PageImage(_ScenarioModel):
    """The `view_page` call for this page of this document completed and returned an image."""

    topic: NonBlank
    doc_id: NonBlank
    page_number: Annotated[int, Field(ge=1)]

    def label(self) -> str:
        return f"image of {self.topic} {self.doc_id} page {self.page_number} from view_page"


class Scenario(_ScenarioModel):
    scenario_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$")]
    description: NonBlank
    prompt: NonBlank
    required_tools: tuple[ArkiveTool, ...] = ()
    argument_expectations: tuple[ArgumentExpectation, ...] = ()
    # Every topic here must be the `topic` argument of at least one completed call.
    topics_queried: tuple[NonBlank, ...] = ()
    answer_groups: Groups = ()
    # The answer names every facet of each topic in `topics_queried` (or of every served
    # topic when that is empty), compared with `_` and `-` read as spaces.
    answer_names_facets: bool = False
    # A topic: the answer reproduces at least one standing passage of the blueprint that the
    # last completed `get_blueprint` call on this topic returned (`_reproduction_outcome`).
    answer_reproduces_standing_passage_of: NonBlank | None = None
    page_image: PageImage | None = None


class ScenarioFile(_ScenarioModel):
    format: Literal["arkive-opencode-scenarios/1"]
    description: NonBlank
    # The topics the served snapshots must provide.
    topics: Annotated[tuple[NonBlank, ...], Field(min_length=1)]
    scenarios: Annotated[tuple[Scenario, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        ids = [s.scenario_id for s in self.scenarios]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate scenario_id: {duplicates}")
        for s in self.scenarios:
            named = set(s.topics_queried)
            if s.page_image:
                named.add(s.page_image.topic)
            if s.answer_reproduces_standing_passage_of:
                named.add(s.answer_reproduces_standing_passage_of)
            unknown = sorted(named - set(self.topics))
            if unknown:
                raise ValueError(f"scenario {s.scenario_id!r} queries undeclared topics {unknown}")
        return self

    @classmethod
    def load(cls, path: Path) -> ScenarioFile:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ToolCall:
    """One finished tool call from the event stream; `name` is OpenCode's tool name, which
    for an MCP tool is `<server>_<tool>`."""

    name: str
    arguments: Mapping[str, Any]
    status: str
    output: str | None
    error: str | None
    # MIME types of the files (such as images) in the MCP result, from `state.metadata`.
    file_types: tuple[str, ...] = ()

    @property
    def arkive_tool(self) -> str | None:
        prefix = f"{ARKIVE_SERVER}_"
        tool = self.name.removeprefix(prefix)
        return tool if self.name.startswith(prefix) and tool in ARKIVE_TOOLS else None


@dataclass(frozen=True)
class Usage:
    """Totals over the `step_finish` events the stream carried. OpenCode can end a stream
    without the last step's `step_finish`, so these are lower bounds."""

    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class Transcript:
    tool_calls: tuple[ToolCall, ...]
    # The text the agent wrote after its last tool call.
    final_answer: str
    errors: tuple[str, ...]
    usage: Usage
    unreadable_lines: tuple[str, ...] = field(default=())


def parse_events(lines: Iterable[str]) -> Transcript:
    """Read `opencode run --format json` output: one JSON event per line.

    `tool_use` events carry `part.tool` and `part.state` (`status`, `input`, `output` or
    `error`); `text` events carry `part.text`; `step_finish` events carry `part.tokens` and
    `part.cost`; `error` events carry `error`. OpenCode keeps an MCP result's content items in
    `part.state.metadata.content`; files there (an image from `view_page`) carry a `mime`.
    Other event types are ignored. A line that is not a JSON object is kept in
    `unreadable_lines`, and the run then fails its checks.
    """
    calls: list[ToolCall] = []
    texts_after_last_call: list[str] = []
    errors: list[str] = []
    unreadable: list[str] = []
    usage: dict[str, Any] = {
        "steps": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.0,
    }
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            unreadable.append(line[:200])
            continue
        if not isinstance(event, dict):
            unreadable.append(line[:200])
            continue
        kind = event.get("type")
        part = _object(event.get("part"))
        if kind == "tool_use":
            state = _object(part.get("state"))
            content = _object(state.get("metadata")).get("content")
            file_types = tuple(
                str(item.get("mime"))
                for item in (content if isinstance(content, list) else [])
                if isinstance(item, dict) and item.get("type") == "file"
            )
            calls.append(
                ToolCall(
                    name=str(part.get("tool", "")),
                    arguments=_object(state.get("input")),
                    status=str(state.get("status", "")),
                    output=_as_text(state.get("output")),
                    error=_as_text(state.get("error")),
                    file_types=file_types,
                )
            )
            texts_after_last_call = []
        elif kind == "text" and isinstance(part.get("text"), str):
            texts_after_last_call.append(part["text"])
        elif kind == "step_finish":
            tokens = _object(part.get("tokens"))
            cache = _object(tokens.get("cache"))
            usage["steps"] += 1
            usage["input_tokens"] += _count(tokens.get("input"))
            usage["output_tokens"] += _count(tokens.get("output"))
            usage["reasoning_tokens"] += _count(tokens.get("reasoning"))
            usage["cache_read_tokens"] += _count(cache.get("read"))
            usage["cache_write_tokens"] += _count(cache.get("write"))
            cost = part.get("cost")
            if isinstance(cost, int | float) and not isinstance(cost, bool):
                usage["cost_usd"] += float(cost)
        elif kind == "error":
            errors.append(json.dumps(event.get("error"), ensure_ascii=False)[:2000])
    return Transcript(
        tool_calls=tuple(calls),
        final_answer="\n".join(texts_after_last_call).strip(),
        errors=tuple(errors),
        usage=Usage(**usage),
        unreadable_lines=tuple(unreadable),
    )


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


@dataclass(frozen=True)
class CheckOutcome:
    check: str
    passed: bool
    detail: str


def evaluate(
    scenario: Scenario,
    transcript: Transcript,
    exit_code: int | None,
    facets_by_topic: Mapping[str, Sequence[str]],
) -> list[CheckOutcome]:
    """Every check of `scenario` against one run. `exit_code` is None when the run did not
    finish (it timed out); `facets_by_topic` holds the facet names of each served topic."""
    outcomes = [_run_completed(transcript, exit_code), _only_arkive_tools(transcript)]
    completed = [c for c in transcript.tool_calls if c.arkive_tool and c.status == "completed"]
    for tool in scenario.required_tools:
        n = sum(c.arkive_tool == tool for c in completed)
        outcomes.append(CheckOutcome(f"called: {tool}", n > 0, f"{n} completed call(s)"))
    for expectation in scenario.argument_expectations:
        outcomes.append(_argument_outcome(expectation, transcript.tool_calls))
    if scenario.topics_queried:
        queried = {c.arguments.get("topic") for c in completed}
        missing = [t for t in scenario.topics_queried if t not in queried]
        outcomes.append(
            CheckOutcome(
                "topics queried",
                not missing,
                f"missing {missing}" if missing else "every topic queried",
            )
        )
    if scenario.page_image:
        outcomes.append(_page_image_outcome(scenario.page_image, completed))
    answer = transcript.final_answer
    for group in scenario.answer_groups:
        found = satisfies_groups(answer, (group,))
        outcomes.append(
            CheckOutcome(f"answer: {' | '.join(group)}", found, "found" if found else "absent")
        )
    if scenario.answer_names_facets:
        topics = scenario.topics_queried or tuple(facets_by_topic)
        names = [name for topic in topics for name in facets_by_topic[topic]]
        spoken = _spaced(answer)
        absent = [name for name in names if _spaced(name) not in spoken]
        outcomes.append(
            CheckOutcome(
                "answer names every facet",
                not absent,
                f"absent: {absent}" if absent else f"named {names}",
            )
        )
    if scenario.answer_reproduces_standing_passage_of:
        outcomes.append(
            _reproduction_outcome(completed, answer, scenario.answer_reproduces_standing_passage_of)
        )
    return outcomes


# A long passage counts as reproduced when its opening words are; the agent may adapt a
# project fact later in it, as the drafting instructions require.
REPRODUCED_PREFIX_WORDS = 12
_WORD = re.compile(r"\w+")


def _words(text: str) -> str:
    """Text as space-separated words after `fold_for_quote`, padded so a containment test
    matches whole words only."""
    return f" {' '.join(_WORD.findall(fold_for_quote(text)))} "


def _reproduction_outcome(completed: Sequence[ToolCall], answer: str, topic: str) -> CheckOutcome:
    """Whether the answer reproduces a standing passage (a passage of kind `standing`, which
    the drafting rules say to reproduce as written) of the blueprint that the last completed
    `get_blueprint` call on `topic` returned, the one the agent drafted from: its text, or
    its first `REPRODUCED_PREFIX_WORDS` words, compared after `fold_for_quote`. A blueprint
    without standing passages is a missing precondition of the snapshot, reported under its
    own check name so it is not read as an agent failure."""
    fetched = [
        c
        for c in completed
        if c.arkive_tool == "get_blueprint" and c.arguments.get("topic") == topic
    ]
    if not fetched:
        return CheckOutcome(
            "answer reproduces a standing passage",
            False,
            f"no completed get_blueprint call on {topic}",
        )
    blueprint = _json_object(fetched[-1].output)
    items = blueprint.get("passages")
    passages = [
        p["text"]
        for p in (items if isinstance(items, list) else [])
        if isinstance(p, dict) and p.get("kind") == "standing" and isinstance(p.get("text"), str)
    ]
    if not passages:
        return CheckOutcome(
            "precondition: blueprint has a standing passage",
            False,
            f"the {blueprint.get('deliverable_type')!r} blueprint of {topic} holds no standing"
            " passage, so reproduction cannot be checked; the served snapshot, not the agent,"
            " fails this scenario",
        )
    spoken = _words(answer)
    for text in passages:
        words = _words(text).split()
        needle = f" {' '.join(words[:REPRODUCED_PREFIX_WORDS])} "
        if words and needle in spoken:
            return CheckOutcome(
                "answer reproduces a standing passage", True, f"reproduced {text[:60]!r}"
            )
    return CheckOutcome(
        "answer reproduces a standing passage",
        False,
        f"none of {len(passages)} standing passage(s) appears in the answer",
    )


def _json_object(text: str | None) -> dict[str, Any]:
    try:
        return _object(json.loads(text or ""))
    except json.JSONDecodeError:
        return {}


def _spaced(text: str) -> str:
    return " ".join(re.sub(r"[_\-]+", " ", normalize_for_match(text)).split())


def _run_completed(transcript: Transcript, exit_code: int | None) -> CheckOutcome:
    problems = []
    if exit_code is None:
        problems.append("timed out")
    elif exit_code != 0:
        problems.append(f"exit code {exit_code}")
    if transcript.errors:
        problems.append(f"{len(transcript.errors)} error event(s)")
    if transcript.unreadable_lines:
        problems.append(f"{len(transcript.unreadable_lines)} line(s) that are not JSON events")
    if not transcript.final_answer:
        problems.append("no final answer")
    return CheckOutcome("run completed", not problems, "; ".join(problems) or "answered")


def _only_arkive_tools(transcript: Transcript) -> CheckOutcome:
    others = sorted({c.name for c in transcript.tool_calls if c.arkive_tool is None})
    return CheckOutcome(
        "only Arkive tools",
        not others,
        f"other tools called: {others}" if others else "no other tool called",
    )


def _non_empty(value: Any) -> bool:
    return isinstance(value, dict | list | str) and len(value) > 0


def _argument_outcome(expectation: ArgumentExpectation, calls: Sequence[ToolCall]) -> CheckOutcome:
    if isinstance(expectation, InvalidFilterRecovery):
        return _recovery_outcome(expectation, calls)
    n = sum(
        c.arkive_tool in expectation.tools
        and c.status == "completed"
        and _non_empty(c.arguments.get(expectation.argument))
        for c in calls
    )
    return CheckOutcome(expectation.label(), n > 0, f"{n} completed call(s) with it")


def _filters(call: ToolCall) -> dict[str, Any]:
    return _object(call.arguments.get("filters"))


def _recovery_outcome(
    expectation: InvalidFilterRecovery, calls: Sequence[ToolCall]
) -> CheckOutcome:
    invalid = expectation.invalid_value
    rejected = False
    for i, call in enumerate(calls):
        if call.arkive_tool not in expectation.tools or call.status != "error":
            continue
        facets = [name for name, value in _filters(call).items() if value == invalid]
        if not facets:
            continue
        rejected = True
        for later in calls[i + 1 :]:
            if (
                later.arkive_tool == call.arkive_tool
                and later.status == "completed"
                and later.arguments.get("topic") == call.arguments.get("topic")
            ):
                retried = _filters(later)
                for facet in facets:
                    if facet in retried and retried[facet] != invalid:
                        detail = f"{facet}={invalid!r} failed, {facet}={retried[facet]!r} completed"
                        return CheckOutcome(expectation.label(), True, detail)
    detail = (
        "no later completed call to the same tool and topic set that facet to another value"
        if rejected
        else f"no call setting a facet to {invalid!r} ended in a tool error"
    )
    return CheckOutcome(expectation.label(), False, detail)


def _page_image_outcome(expected: PageImage, completed: Sequence[ToolCall]) -> CheckOutcome:
    views = [
        c
        for c in completed
        if c.arkive_tool == "view_page"
        and c.arguments.get("topic") == expected.topic
        and c.arguments.get("doc_id") == expected.doc_id
        and _count(c.arguments.get("page_number")) == expected.page_number
    ]
    images = sum(any(t.startswith("image/") for t in c.file_types) for c in views)
    return CheckOutcome(
        expected.label(),
        images > 0,
        f"{len(views)} completed call(s) for that page, {images} returned an image",
    )
