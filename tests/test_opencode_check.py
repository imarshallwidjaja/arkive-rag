"""The OpenCode check's offline parts: the scenario file, event-stream parsing, the checks and
the runner's input refusals, output handling, key redaction and container cleanup. Nothing
here starts Docker or calls a model: `subprocess.run` is replaced by a fake Docker."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from arkive_rag import passages
from evals.opencode_check import runner
from evals.opencode_check.scenarios import (
    EqualArgument,
    NonEmptyArgument,
    Scenario,
    ScenarioFile,
    Transcript,
    evaluate,
    parse_events,
)
from evals.schema import normalize_for_match
from tests.support import BATHYMETRY_CORPUS, compile_snapshot


def _event(kind: str, **part: Any) -> str:
    return json.dumps({"type": kind, "timestamp": 1, "sessionID": "ses_1", "part": part})


def _tool(
    tool: str,
    arguments: dict[str, Any],
    status: str = "completed",
    files: tuple[str, ...] = (),
    output: str = "{}",
) -> str:
    state: dict[str, Any] = {"status": status, "input": arguments}
    if status == "completed":
        state["output"] = output
        content: list[dict[str, Any]] = [{"type": "text", "text": output}]
        content += [{"type": "file", "uri": "data:,", "mime": mime} for mime in files]
        state["metadata"] = {"content": content}
    else:
        state["error"] = "unknown project value 'zz'; valid values: Q101, Q102"
    return _event("tool_use", type="tool", tool=tool, state=state)


def _step_finish(cost: float, **tokens: Any) -> str:
    return _event("step_finish", type="step-finish", reason="stop", cost=cost, tokens=tokens)


FACETS = {"survey": ("project", "document_role"), "campaigns": ("creator",)}


def _scenario(**fields: Any) -> Scenario:
    return Scenario.model_validate(
        {"scenario_id": "case", "description": "d", "prompt": "p", **fields}
    )


def _failed(scenario: Scenario, transcript: Transcript, exit_code: int | None = 0) -> list[str]:
    return [c.check for c in evaluate(scenario, transcript, exit_code, FACETS) if not c.passed]


def test_shipped_scenario_topics_are_shipped_example_corpora_and_prompts_name_no_facet() -> None:
    scenario_file = ScenarioFile.load(runner.DEFAULT_SCENARIOS)
    assert scenario_file.topics == ("coastal-survey-projects", "creator-campaigns")
    # Other example corpora may exist for the evaluation harness without an OpenCode scenario.
    corpora = Path(__file__).resolve().parents[1] / "evals" / "example_corpora"
    assert set(scenario_file.topics) <= {p.name for p in corpora.iterdir() if p.is_dir()}
    every_prompt = " ".join(s.prompt for s in scenario_file.scenarios)
    assert "document_role" not in every_prompt and "creator=" not in every_prompt
    assert len(scenario_file.scenarios) == 10


def test_uplift_scenarios_extend_the_shipped_ones_with_structural_monitoring_scenarios() -> None:
    shipped = ScenarioFile.load(runner.DEFAULT_SCENARIOS)
    uplift = ScenarioFile.load(runner.CHECK_DIR / "scenarios-uplift.json")
    topic = "structural-monitoring-consultancy"
    assert uplift.topics == (*shipped.topics, topic)
    corpora = Path(__file__).resolve().parents[1] / "evals" / "example_corpora"
    assert set(uplift.topics) <= {p.name for p in corpora.iterdir() if p.is_dir()}
    # Every shipped scenario, unchanged, then the three Schema 7.0 scenarios.
    assert uplift.scenarios[:10] == shipped.scenarios
    added = {s.scenario_id: s for s in uplift.scenarios[10:]}
    assert list(added) == ["type-filtered-list", "version-history-discovery", "table-rollup"]
    assert all(s.topics_queried == (topic,) for s in added.values())
    corpus = corpora / topic
    # Answer values are values the corpus states, so a faithful answer can meet them.
    corpus_text = normalize_for_match(
        " ".join(p.read_text(encoding="utf-8") for p in corpus.rglob("*.md"))
    )
    for scenario_id in ("version-history-discovery", "table-rollup"):
        for group in added[scenario_id].answer_groups:
            assert any(normalize_for_match(a) in corpus_text for a in group), group
    # Deliverable types and facets are induced per build, so no prompt names a type label.
    with_type = added["type-filtered-list"]
    assert with_type.required_tools == ("list_topics", "list_documents")
    assert "deliverable_type=" not in with_type.prompt
    (history,) = added["version-history-discovery"].argument_expectations
    assert isinstance(history, EqualArgument)
    assert (history.argument, history.value, history.topic) == ("versions", "all", topic)
    rollup = added["table-rollup"]
    assert rollup.required_tools == ("query_tables",)
    (table_call,) = rollup.argument_expectations
    assert isinstance(table_call, NonEmptyArgument)
    assert (table_call.tools, table_call.topic) == (("query_tables",), topic)
    # The table-rollup values are table cells, each serial in the row of its calibration month,
    # so only a table read can supply them together.
    rows = [
        row
        for p in corpus.rglob("*.md")
        for table in passages.tables(p.read_text(encoding="utf-8"))
        for row in table.rows
    ]
    assert ("PF4-1876", "2022", "August each year") in rows
    assert ("PF4-2043", "2025", "March each year") in rows
    assert ("Strain sensitivity", "1.21 pm/\u00b5\u03b5") in rows
    cells = " ".join(cell for row in rows for cell in row)
    for group in rollup.answer_groups:
        assert any(a in cells for a in group), group


def test_partial_uplift_answers_fail_their_answer_groups() -> None:
    uplift = ScenarioFile.load(runner.CHECK_DIR / "scenarios-uplift.json")
    by_id = {s.scenario_id: s for s in uplift.scenarios}
    topic = "structural-monitoring-consultancy"

    def failed(scenario_id: str, call: str, answer: str) -> list[str]:
        transcript = parse_events([call, _event("text", type="text", text=answer)])
        return _failed(by_id[scenario_id], transcript)

    history = _tool("arkive_search", {"topic": topic, "query": "alarm", "versions": "all"})
    full_history = (
        "v1.0 (issued): amber 60% (120), red 85% (170). The v0.8 and v0.9 drafts set amber"
        " 75% (150) and red 100% (200)."
    )
    assert failed("version-history-discovery", history, full_history) == []
    # The draft values alone, without the issued percentages or the drafts' levels, fail.
    assert failed("version-history-discovery", history, "120, 170, v0.8") == [
        "answer: 60% | 60 %",
        "answer: 85% | 85 %",
        "answer: 75% | 75 % | 150",
        "answer: 100% | 100 % | 200",
    ]
    other_topic = _tool("arkive_search", {"topic": "survey", "query": "a", "versions": "all"})
    assert f"arguments: search/grep/list_documents/query_tables.versions == 'all' on {topic}" in (
        failed("version-history-discovery", other_topic, full_history)
    )
    tables = _tool("arkive_query_tables", {"topic": topic, "query": "PF4"})
    full_rollup = "PF4-1876: due August; PF4-2043: due March; LS-8 strain sensitivity 1.21."
    assert failed("table-rollup", tables, full_rollup) == []
    # The serials and the sensitivity without the calibration months fail.
    assert failed("table-rollup", tables, "PF4-1876, PF4-2043, 1.21") == [
        "answer: August",
        "answer: March",
    ]


def test_an_equal_argument_needs_a_completed_call_with_that_value_of_the_same_type() -> None:
    scenario = _scenario(
        argument_expectations=[
            {
                "condition": "equals",
                "tools": ["search", "grep"],
                "argument": "versions",
                "value": "all",
            }
        ]
    )
    label = "arguments: search/grep.versions == 'all'"
    answered = _event("text", type="text", text="ok")

    def failed(*calls: str) -> list[str]:
        return _failed(scenario, parse_events([*calls, answered]))

    assert failed(_tool("arkive_grep", {"topic": "survey", "versions": "all"})) == []
    assert failed(_tool("arkive_search", {"topic": "survey", "versions": "current"})) == [label]
    assert failed(_tool("arkive_search", {"topic": "survey"})) == [label]
    assert failed(_tool("arkive_search", {"topic": "survey", "versions": "all"}, "error")) == [
        label
    ]
    assert failed(_tool("arkive_list_documents", {"topic": "survey", "versions": "all"})) == [label]
    flagged = _scenario(
        argument_expectations=[
            {"condition": "equals", "tools": ["search"], "argument": "flag", "value": True}
        ]
    )
    one = _tool("arkive_search", {"topic": "survey", "flag": 1})
    assert _failed(flagged, parse_events([one, answered])) == ["arguments: search.flag == True"]
    true = _tool("arkive_search", {"topic": "survey", "flag": True})
    assert _failed(flagged, parse_events([true, answered])) == []
    on_survey = _scenario(
        argument_expectations=[
            {
                "condition": "equals",
                "tools": ["search"],
                "argument": "versions",
                "value": "all",
                "topic": "survey",
            }
        ]
    )
    elsewhere = _tool("arkive_search", {"topic": "campaigns", "versions": "all"})
    assert _failed(on_survey, parse_events([elsewhere, answered])) == [
        "arguments: search.versions == 'all' on survey"
    ]
    here = _tool("arkive_search", {"topic": "survey", "versions": "all"})
    assert _failed(on_survey, parse_events([elsewhere, here, answered])) == []


DISCLOSURE = (
    'Every creator post made under this brief is labelled "Paid partnership" in its first'
    " line or in its first three seconds of video."
)


PROJECT_PARAGRAPH = (
    "Pellow Bakery opens its third shop in Truro this spring with a sourdough class for the"
    " whole street."
)


def _packet_output(samples: Sequence[str], *sections: str) -> str:
    """A drafting packet whose one exemplar includes each section, with `samples` as its
    `recurring_samples`."""
    return json.dumps(
        {
            "deliverable_type": "Campaign brief",
            "recurring_samples": list(samples),
            "exemplars": [
                {
                    "doc_id": "doc_002",
                    "sections": [
                        {"section_id": f"doc_002_sec_{n:03d}", "included": True, "markdown": text}
                        for n, text in enumerate(sections, start=1)
                    ],
                }
            ],
        }
    )


def test_shipped_drafting_scenario_needs_the_packet_and_a_recurring_passage() -> None:
    (draft,) = [
        s
        for s in ScenarioFile.load(runner.DEFAULT_SCENARIOS).scenarios
        if s.scenario_id == "draft-from-packet"
    ]
    topic = {"topic": "creator-campaigns"}
    request = topic | {"project_inputs": "Pellow Bakery, spring 2027, GBP 9,000, email."}

    def brief(mandatories: str) -> str:
        return _event(
            "text",
            type="text",
            text="Campaign brief: Pellow Bakery, budget GBP 9,000. Objective: ... One CTA."
            f" Measures: ... Mandatories: {mandatories}",
        )

    # Ten words of a recurring passage in a row count, whatever its opening: here the agent
    # adapted the first three words to the newsletter. Emphasis, quotes and case do not
    # matter.
    reproduced = brief(
        "*Every newsletter issue made under this brief is labelled Paid Partnership in its"
        " first line* of the header."
    )
    unreproduced = brief("say the post is sponsored.")
    # The disclosure section also holds the exemplar's own project paragraph; copying that
    # paragraph is not house text.
    copied_project = brief(PROJECT_PARAGRAPH)
    mixed = f"## Disclosure\n\n{PROJECT_PARAGRAPH}\n\n{DISCLOSURE}"
    sample = " ".join(DISCLOSURE.split()[:16])
    guide = _tool("arkive_get_topic_guide", topic)
    packet = _tool(
        "arkive_prepare_draft",
        request,
        output=_packet_output([sample], "## Objective\n\nGrow.", mixed),
    )
    single_family = _tool("arkive_prepare_draft", request, output=_packet_output([], mixed))
    unknown = _tool("arkive_prepare_draft", topic | {"deliverable_type": "press kit"}, "error")

    assert _failed(draft, parse_events([guide, packet, reproduced])) == []
    for answer in (unreproduced, copied_project):
        assert _failed(draft, parse_events([packet, answer])) == [
            "answer reproduces a recurring passage"
        ]
    # A packet whose included sections recur in no other version family fails a precondition
    # of the snapshot, named as such, rather than the agent's reproduction check.
    outcomes = evaluate(draft, parse_events([single_family, reproduced]), 0, FACETS)
    (missing,) = [o for o in outcomes if not o.passed]
    assert missing.check == "precondition: packet has a recurring passage"
    assert "the served snapshot, not the agent, fails this scenario" in missing.detail
    # The guide holds no exemplar text, so a draft from it alone misses the checks.
    assert _failed(draft, parse_events([guide, reproduced])) == [
        "called: prepare_draft",
        "answer reproduces a recurring passage",
    ]
    assert _failed(draft, parse_events([unknown, reproduced])) == [
        "called: prepare_draft",
        "topics queried",
        "answer reproduces a recurring passage",
    ]
    # Only a packet of the scenario's topic counts, and of those the last one fetched.
    elsewhere = _tool(
        "arkive_prepare_draft",
        {"topic": "coastal-survey-projects", "project_inputs": "A proposal."},
        output=_packet_output([sample], mixed),
    )
    assert _failed(draft, parse_events([guide, elsewhere, reproduced])) == [
        "answer reproduces a recurring passage"
    ]
    assert _failed(draft, parse_events([packet, single_family, reproduced])) == [
        "precondition: packet has a recurring passage"
    ]
    assert _failed(draft, parse_events([single_family, packet, reproduced])) == []


def test_a_truncated_arkive_result_fails_every_scenario() -> None:
    truncated = json.loads(_tool("arkive_list_documents", {"topic": "survey"}))
    truncated["part"]["state"]["metadata"]["truncated"] = True
    answer = _event("text", type="text", text="Done.")

    assert _failed(_scenario(), parse_events([json.dumps(truncated), answer])) == [
        "no Arkive result truncated"
    ]
    whole = _tool("arkive_list_documents", {"topic": "survey"})
    assert _failed(_scenario(), parse_events([whole, answer])) == []


def test_scenario_file_refuses_duplicates_unknown_tools_and_undeclared_topics() -> None:
    def load(*scenarios: dict[str, Any]) -> ScenarioFile:
        return ScenarioFile.model_validate(
            {
                "format": "arkive-opencode-scenarios/1",
                "description": "d",
                "topics": ["survey"],
                "scenarios": list(scenarios),
            }
        )

    case = {"scenario_id": "a", "description": "d", "prompt": "p"}
    assert load(case).scenarios[0].scenario_id == "a"
    with pytest.raises(ValidationError, match="duplicate scenario_id"):
        load(case, case)
    with pytest.raises(ValidationError, match="undeclared topics"):
        load(case | {"topics_queried": ["campaigns"]})
    with pytest.raises(ValidationError, match="undeclared topics"):
        load(case | {"answer_reproduces_recurring_passage_of": "campaigns"})
    with pytest.raises(ValidationError, match="undeclared topics"):
        load(
            case
            | {
                "argument_expectations": [
                    {
                        "condition": "non_empty",
                        "tools": ["query_tables"],
                        "argument": "query",
                        "topic": "campaigns",
                    }
                ]
            }
        )
    with pytest.raises(ValidationError, match="required_tools"):
        load(case | {"required_tools": ["read"]})
    with pytest.raises(ValidationError, match="condition"):
        load(
            case
            | {
                "argument_expectations": [
                    {"tools": ["search"], "argument": "filters", "condition": "present"}
                ]
            }
        )
    with pytest.raises(ValidationError, match="scenario_id"):
        load(case | {"scenario_id": "Has Spaces"})


def test_event_stream_parsing_keeps_calls_files_usage_and_the_text_after_the_last_call() -> None:
    lines = [
        _event("step_start", type="step-start"),
        _event("text", type="text", text="Let me look."),
        _tool("arkive_list_documents", {"topic": "survey", "filters": {"project": "zz"}}, "error"),
        _tool("arkive_view_page", {"topic": "survey", "doc_id": "doc_006"}, files=("image/webp",)),
        _step_finish(0.25, input=3, output=40, reasoning=2, cache={"read": 100, "write": 50}),
        "",
        "this is not an event",
        _event("text", type="text", text="Bench 3: 41,700 m3"),
        _event("text", type="text", text="(doc_006, p. 1)"),
        _step_finish(0.5, input=1, output=10),
        json.dumps({"type": "error", "error": {"name": "ProviderError", "message": "429"}}),
    ]
    transcript = parse_events(lines)
    assert [c.name for c in transcript.tool_calls] == ["arkive_list_documents", "arkive_view_page"]
    failed, viewed = transcript.tool_calls
    assert failed.status == "error" and failed.error is not None and "valid values" in failed.error
    assert failed.arkive_tool == "list_documents" and failed.arguments["filters"] == {
        "project": "zz"
    }
    assert viewed.file_types == ("image/webp",) and viewed.output == "{}"
    assert transcript.final_answer == "Bench 3: 41,700 m3\n(doc_006, p. 1)"
    assert transcript.unreadable_lines == ("this is not an event",)
    assert len(transcript.errors) == 1 and "429" in transcript.errors[0]
    usage = transcript.usage
    assert (usage.steps, usage.input_tokens, usage.output_tokens) == (2, 4, 50)
    assert (usage.reasoning_tokens, usage.cache_read_tokens, usage.cache_write_tokens) == (
        2,
        100,
        50,
    )
    assert usage.cost_usd == pytest.approx(0.75)


def test_a_run_that_meets_every_expectation_passes_every_check() -> None:
    scenario = _scenario(
        required_tools=["list_topics", "list_documents"],
        argument_expectations=[
            {
                "condition": "invalid_filter_recovery",
                "tools": ["list_documents"],
                "invalid_value": "zz",
            },
            {
                "condition": "non_empty",
                "tools": ["search", "list_documents"],
                "argument": "filters",
            },
        ],
        topics_queried=["survey", "campaigns"],
        answer_groups=[["41,700", "41700"], ["doc_006"]],
        answer_names_facets=True,
        page_image={"topic": "survey", "doc_id": "doc_006", "page_number": 1},
    )
    transcript = parse_events(
        [
            _tool("arkive_list_topics", {}),
            _tool(
                "arkive_list_documents", {"topic": "survey", "filters": {"project": "zz"}}, "error"
            ),
            _tool("arkive_list_documents", {"topic": "survey", "filters": {"project": "Q102"}}),
            _tool(
                "arkive_view_page",
                {"topic": "survey", "doc_id": "doc_006", "page_number": 1},
                files=("image/webp",),
            ),
            _tool("arkive_search", {"topic": "campaigns", "query": "q", "filters": None}),
            _event(
                "text",
                type="text",
                text="Bench 3 holds 41700 m3 (doc_006). Facets: project, document role, creator.",
            ),
        ]
    )
    outcomes = evaluate(scenario, transcript, 0, FACETS)
    assert all(o.passed for o in outcomes), [o for o in outcomes if not o.passed]
    assert len(outcomes) == 12


def test_each_kind_of_failure_names_its_check() -> None:
    answered = _event("text", type="text", text="An answer.")
    plain = _scenario()
    assert _failed(plain, parse_events([answered])) == []
    # The run itself: timeout, exit code, error event, unreadable line, no answer.
    assert _failed(plain, parse_events([answered]), exit_code=None) == ["run completed"]
    assert _failed(plain, parse_events([answered]), exit_code=1) == ["run completed"]
    assert _failed(plain, parse_events([answered, '{"type":"error","error":{}}'])) == [
        "run completed"
    ]
    assert _failed(plain, parse_events([answered, "garbage"])) == ["run completed"]
    assert _failed(plain, parse_events([_tool("arkive_list_topics", {})])) == ["run completed"]
    # A tool outside the Arkive server, or a name that only looks like one.
    for name in ("read", "shell", "arkive_delete_everything"):
        assert _failed(plain, parse_events([_tool(name, {}), answered])) == ["only Arkive tools"]
    # A required tool that never completed.
    needs_grep = _scenario(required_tools=["grep"])
    errored_grep = _tool("arkive_grep", {"topic": "survey", "pattern": "x"}, "error")
    assert _failed(needs_grep, parse_events([errored_grep, answered])) == ["called: grep"]
    # Argument expectations.
    non_empty = _scenario(
        argument_expectations=[
            {"tools": ["search"], "argument": "filters", "condition": "non_empty"}
        ]
    )
    unfiltered = _tool("arkive_search", {"topic": "survey", "filters": {}})
    assert _failed(non_empty, parse_events([unfiltered, answered])) == [
        "arguments: search.filters non_empty"
    ]
    good = _tool("arkive_list_documents", {"topic": "survey", "filters": {"project": "Q101"}})
    # Topics, answer groups, facet names and images.
    both = _scenario(topics_queried=["survey", "campaigns"])
    assert _failed(both, parse_events([good, answered])) == ["topics queried"]
    grouped = _scenario(answer_groups=[["38,400"], ["18,500", "18500"]])
    assert _failed(grouped, parse_events([_event("text", type="text", text="GBP 38400")])) == [
        "answer: 18,500 | 18500"
    ]
    facets = _scenario(answer_names_facets=True, topics_queried=["survey"])
    named_one = _event("text", type="text", text="Filter by project.")
    assert _failed(facets, parse_events([good, named_one])) == ["answer names every facet"]


def test_invalid_filter_recovery_credits_only_the_retry_of_the_rejected_facet() -> None:
    recovery = _scenario(
        argument_expectations=[
            {
                "condition": "invalid_filter_recovery",
                "tools": ["list_documents"],
                "invalid_value": "zz",
            }
        ]
    )
    label = "filters: list_documents recovers from 'zz'"

    def listing(filters: dict[str, str], status: str = "completed", topic: str = "survey") -> str:
        return _tool("arkive_list_documents", {"topic": topic, "filters": filters}, status)

    def failed(*calls: str) -> list[str]:
        return _failed(recovery, parse_events([*calls, _event("text", type="text", text="ok")]))

    bad = listing({"project": "zz"}, "error")
    assert failed(bad, listing({"project": "Q101"})) == []
    assert failed(bad, listing({"project": "Q101", "document_role": "memo"})) == []
    # No rejected call, or the retry came first.
    assert failed(listing({"project": "Q101"})) == [label]
    assert failed(listing({"project": "Q101"}), bad) == [label]
    # The rejected call did not carry the scenario's invalid value.
    other_error = listing({"project": "Q999"}, "error")
    assert failed(other_error, listing({"project": "Q101"})) == [label]
    # The retry changed facet, kept the invalid value, used another topic or another tool,
    # or did not complete.
    assert failed(bad, listing({"document_role": "memo"})) == [label]
    assert failed(bad, listing({"project": "zz"})) == [label]
    assert failed(bad, listing({"project": "Q101"}, topic="campaigns")) == [label]
    searched = _tool(
        "arkive_search", {"topic": "survey", "query": "q", "filters": {"project": "Q101"}}
    )
    assert failed(bad, searched) == [label]
    assert failed(bad, listing({"project": "Q101"}, "error")) == [label]


def test_page_image_check_needs_the_image_of_the_named_page() -> None:
    expected = _scenario(page_image={"topic": "survey", "doc_id": "doc_006", "page_number": 1})
    label = "image of survey doc_006 page 1 from view_page"
    answered = _event("text", type="text", text="41,700")

    def view(
        files: tuple[str, ...] = ("image/webp",), status: str = "completed", **args: Any
    ) -> str:
        arguments = {"topic": "survey", "doc_id": "doc_006", "page_number": 1} | args
        return _tool("arkive_view_page", arguments, status, files)

    assert _failed(expected, parse_events([view(), answered])) == []
    for wrong in (
        view(files=()),
        view(doc_id="doc_005"),
        view(page_number=2),
        view(page_number=True),
        view(topic="campaigns"),
        _tool(
            "arkive_read_section", {"topic": "survey", "doc_id": "doc_006"}, files=("image/png",)
        ),
    ):
        assert _failed(expected, parse_events([wrong, answered])) == [label], wrong


def test_scenario_command_mounts_snapshots_read_only_and_never_carries_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    served = [
        runner.ServedSnapshot(Path("/data/a.arkive.db"), "survey", "0" * 64, ("project",)),
        runner.ServedSnapshot(Path("/data/b.arkive.db"), "campaigns", "1" * 64, ()),
    ]
    command = runner.scenario_command("img:tag", "box", "1234:5678", served, "openai/m", "a prompt")
    assert "sk-test-not-a-real-key" not in " ".join(command)
    assert command[command.index("OPENAI_API_KEY") - 1] == "--env"
    assert command[command.index("--user") + 1] == "1234:5678"
    volumes = [command[i + 1] for i, arg in enumerate(command) if arg == "--volume"]
    assert volumes == [
        "/data/a.arkive.db:/snapshots/0.arkive.db:ro",
        "/data/b.arkive.db:/snapshots/1.arkive.db:ro",
    ]
    # Default network; removed by the runner rather than by `--rm`.
    assert "--network" not in command and "--rm" not in command
    assert command[-3:] == ["img:tag", "openai/m", "a prompt"]


def test_runner_refuses_bad_inputs_before_building_an_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_image(context_dir: Path) -> str:
        raise AssertionError("the runner built an image after refusable input")

    monkeypatch.setattr(runner, "build_image", no_image)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    snapshot = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    scenarios = tmp_path / "scenarios.json"
    body = {
        "format": "arkive-opencode-scenarios/1",
        "description": "d",
        "topics": ["bathymetry"],
        "scenarios": [{"scenario_id": "a", "description": "d", "prompt": "p"}],
    }
    scenarios.write_text(json.dumps(body))
    other_topic = tmp_path / "other.json"
    other_topic.write_text(json.dumps(body | {"topics": ["marketing"]}))
    out = tmp_path / "results"

    def refusal(*args: str) -> str:
        result = CliRunner().invoke(runner.app, list(args))
        assert result.exit_code == 2, result.output
        assert not out.exists()
        return " ".join(result.output.replace("\u2502", " ").split())

    base = (str(snapshot), "--scenarios", str(scenarios))
    assert "overlaps snapshot" in refusal(*base, "--out", str(snapshot))
    assert "overlaps --scenarios" in refusal(*base, "--out", str(tmp_path))
    assert "overlaps --scenarios" in refusal(*base, "--out", str(scenarios))
    assert "the same topic" in refusal(*base, str(snapshot), "--out", str(out))
    assert "no snapshot serves the scenario topics ['marketing']" in refusal(
        str(snapshot), "--scenarios", str(other_topic), "--out", str(out)
    )
    assert "is not a valid scenario file" in refusal(
        str(snapshot), "--scenarios", str(snapshot), "--out", str(out)
    )
    garbage = tmp_path / "garbage.arkive.db"
    garbage.write_bytes(b"not a database")
    assert "file is not a database" in refusal(
        str(garbage), "--scenarios", str(scenarios), "--out", str(out)
    )
    assert "unknown scenario IDs ['nope']" in refusal(*base, "--out", str(out), "--only", "nope")
    monkeypatch.setenv("OPENAI_API_KEY", " ")
    assert "OPENAI_API_KEY is not set or is empty" in refusal(*base, "--out", str(out))
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.chdir(tmp_path)
    assert "OPENAI_API_KEY is not set" in refusal(*base, "--out", str(out))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(os, "getuid", lambda: 0)
    assert "non-root user" in refusal(*base, "--out", str(out))


KEY = "sk-test-SYNTHETIC-KEY-0123456789abcdef"


class FakeDocker:
    """Stands in for `subprocess.run`: answers `docker run` with a scripted result (or an
    interrupt) and `docker rm --force` with success or `rm_error`, recording every command."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        rm_error: str | None = None,
        interrupt: bool = False,
    ) -> None:
        self.stdout, self.stderr, self.exit_code = stdout, stderr, exit_code
        self.rm_error, self.interrupt = rm_error, interrupt
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        command = list(command)
        self.commands.append(command)
        if command[:2] == ["docker", "run"]:
            if self.interrupt:
                raise KeyboardInterrupt
            return subprocess.CompletedProcess(command, self.exit_code, self.stdout, self.stderr)
        if command[:3] == ["docker", "rm", "--force"]:
            failed = self.rm_error is not None
            return subprocess.CompletedProcess(command, int(failed), "", self.rm_error or "")
        raise AssertionError(f"unexpected command {command}")

    def started(self) -> list[str]:
        return [c[c.index("--name") + 1] for c in self.commands if c[:2] == ["docker", "run"]]

    def removed(self) -> list[str]:
        return [c[3] for c in self.commands if c[:3] == ["docker", "rm", "--force"]]


@pytest.fixture
def check_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], Path]:
    """CLI arguments for a two-scenario run on a compiled snapshot, with the image build and
    the Docker lookup stubbed; returns the arguments and the output directory."""
    monkeypatch.setattr(runner, "build_image", lambda context_dir: "img:tag")
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.chdir(tmp_path)
    snapshot = compile_snapshot(tmp_path, BATHYMETRY_CORPUS)
    scenarios = tmp_path / "scenarios.json"
    case = {"description": "d", "prompt": "p"}
    body = {
        "format": "arkive-opencode-scenarios/1",
        "description": "d",
        "topics": ["bathymetry"],
        "scenarios": [case | {"scenario_id": "one"}, case | {"scenario_id": "two"}],
    }
    scenarios.write_text(json.dumps(body))
    out = tmp_path / "results"
    return [str(snapshot), "--scenarios", str(scenarios), "--out", str(out)], out


def test_the_key_never_reaches_a_written_artifact(
    check_run: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    args, out = check_run
    leaked_error = {
        "status": "error",
        "input": {"topic": "bathymetry", "query": KEY},
        "error": f"401 Incorrect API key provided: {KEY}",
    }
    stdout = "\n".join(
        [
            _event("tool_use", type="tool", tool="arkive_search", state=leaked_error),
            json.dumps({"type": "error", "error": {"message": f"bad key {KEY}"}}),
            f"not an event {KEY}",
            _event("text", type="text", text=f"The key is {KEY}."),
        ]
    )
    fake = FakeDocker(stdout=stdout, stderr=f"provider said {KEY}\n", exit_code=1)
    monkeypatch.setattr(subprocess, "run", fake)
    result = CliRunner().invoke(runner.app, args)
    assert result.exit_code == 1, result.output
    written = sorted(p for p in out.rglob("*") if p.is_file())
    assert [p.relative_to(out).as_posix() for p in written] == [
        "events/one.jsonl",
        "events/two.jsonl",
        "report.md",
        "results.json",
    ]
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert KEY not in text, path
        assert runner.REDACTED_KEY in text, path
    assert KEY not in result.output


def test_every_container_is_removed_under_a_unique_name(
    check_run: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = check_run
    answered = _event("text", type="text", text="An answer.")
    names: list[str] = []
    for _ in range(2):
        fake = FakeDocker(stdout=answered)
        monkeypatch.setattr(subprocess, "run", fake)
        result = CliRunner().invoke(runner.app, args)
        assert result.exit_code == 0, result.output
        assert fake.removed() == fake.started() and len(fake.started()) == 2
        names += fake.started()
    # Two runs of two scenarios: four distinct names, each naming its scenario.
    assert len(set(names)) == 4
    assert [n.rsplit("-", 1)[1] for n in names] == ["one", "two", "one", "two"]


def test_a_container_that_cannot_be_removed_fails_the_run_and_is_named(
    check_run: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    args, out = check_run
    fake = FakeDocker(stdout=_event("text", type="text", text="ok"), rm_error="daemon gone")
    monkeypatch.setattr(subprocess, "run", fake)
    result = CliRunner().invoke(runner.app, args)
    assert result.exit_code == 1
    for name in fake.started():
        assert f"error: container {name} was not removed: daemon gone" in result.output
    assert "containers not removed" in result.output
    assert json.loads((out / "results.json").read_text())["passed"] == 2


def test_an_interrupted_scenario_still_removes_its_container(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(rm_error: str | None) -> FakeDocker:
        fake = FakeDocker(interrupt=True, rm_error=rm_error)
        monkeypatch.setattr(subprocess, "run", fake)
        with pytest.raises(KeyboardInterrupt):
            runner.run_scenario(
                _scenario(), ["docker", "run", "--name", "box"], "box", 60, FACETS, KEY
            )
        return fake

    assert interrupted(None).removed() == ["box"]
    # A failed removal is reported, and the interrupt still propagates.
    assert interrupted("daemon gone").removed() == ["box"]
    assert "container box was not removed: daemon gone" in capsys.readouterr().err


def test_outputs_are_never_written_through_links(
    check_run: tuple[list[str], Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, out = check_run
    fake = FakeDocker(stdout=_event("text", type="text", text="ok"))
    monkeypatch.setattr(subprocess, "run", fake)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out.mkdir()

    def refusal() -> str:
        result = CliRunner().invoke(runner.app, args)
        assert result.exit_code == 2, result.output
        assert fake.commands == [] and list(elsewhere.iterdir()) == []
        return " ".join(result.output.replace("\u2502", " ").split())

    (out / "events").symlink_to(elsewhere, target_is_directory=True)
    assert "is not a real directory" in refusal()
    (out / "events").unlink()
    (out / "results.json").symlink_to(elsewhere / "results.json")
    assert "refusing to write through symlinks" in refusal()
