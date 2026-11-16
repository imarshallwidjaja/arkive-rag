from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import stat
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx2
import numpy as np
import openai
import pytest
from agents import Runner
from agents.tracing.processors import default_exporter
from pydantic import BaseModel

from arkive_rag import llm
from arkive_rag.llm import (
    DECISION_MODEL,
    DEFAULT_MODEL,
    BudgetExceededError,
    BuildCache,
    Effort,
    OpenAIGateway,
    Tool,
    ToolImage,
    ToolReply,
    ToolResult,
    UsageMeter,
    embedding_batches,
    load_dotenv_if_present,
)
from arkive_rag.models import (
    NotFoundError,
    ProviderError,
    SnapshotIntegrityError,
)
from tests.support import (
    TEST_BASE_URL,
    Outcome,
    ScriptedOpenAI,
    decision_body,
    function_call,
    message,
    response_body,
    structured_body,
    usage,
)

JSON_HEADERS = {"content-type": "application/json"}
REQUEST = httpx2.Request("POST", "https://api.openai.test/v1/responses")


class ProbeOutput(BaseModel):
    """A structured output owned by this test suite."""

    title: str
    client: str
    stage: str
    doc_date: str | None
    summary: str


def probe_api() -> ScriptedOpenAI:
    """The scripted provider, answering structured requests with `META` when unscripted."""
    api = ScriptedOpenAI()
    api.defaults["ProbeOutput"] = META
    return api


META = ProbeOutput(
    title="Memo", client="Contoso Ports", stage="memorandum", doc_date=None, summary="Bathymetry."
)


@dataclass(frozen=True)
class Probe:
    """A structured request owned by this test suite."""

    operation: ClassVar[str] = "probe"
    output: ClassVar[type[ProbeOutput]] = ProbeOutput
    instructions: ClassVar[str] = "Catalogue the document."
    effort: Effort = "low"
    prompt: str = "outline"
    image_webp: bytes | None = None

    def text(self) -> str:
        return self.prompt


def lookup(args: dict[str, Any]) -> ToolResult:
    if args["key"] == "missing":
        raise NotFoundError("no section 'missing'")
    if args["key"] == "chart":
        return ToolReply("chart page", (ToolImage(b"RIFF-chart", "image/webp"),))
    return f"value of {args['key']}"


LOOKUP = Tool(
    "lookup",
    "Read one value.",
    {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    lookup,
)


@dataclass(frozen=True)
class AgentProbe:
    operation: ClassVar[str] = "agent_probe"
    output: ClassVar[type[ProbeOutput]] = ProbeOutput
    effort: Effort = "medium"
    max_turns: int = 3
    tools: Sequence[Tool] = (LOOKUP,)
    source_digest: str = "a" * 64
    prompt: str = "Catalogue it."
    instructions: ClassVar[str] = "Use the tools, then answer."

    def text(self) -> str:
        return self.prompt


def connect_error() -> httpx2.ConnectError:
    return httpx2.ConnectError("connection refused", request=REQUEST)


# 100k input tokens at $2.00/1M plus 10k output at $10.00/1M: $0.30.
SPEND_030: dict[str, int] = {
    "input_tokens": 100_000,
    "cached_input_tokens": 0,
    "output_tokens": 10_000,
}


def test_missing_api_key_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIGateway()


def test_dotenv_fills_unset_variables_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARKIVE_KEEP", "process")
    monkeypatch.delenv("ARKIVE_NEW", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nexport ARKIVE_NEW='from-file'\nARKIVE_KEEP=file\n")

    load_dotenv_if_present(env)
    load_dotenv_if_present(tmp_path / "absent.env")

    assert (os.environ["ARKIVE_NEW"], os.environ["ARKIVE_KEEP"]) == ("from-file", "process")


# Structured calls.


def test_structured_request_sends_model_effort_instructions_text_and_image() -> None:
    api = probe_api()

    result = api.gateway().run(Probe(effort="high", prompt="330.9", image_webp=b"RIFF-page"))

    assert isinstance(result, ProbeOutput)
    body = api.requests("/responses")[0]
    assert (body["model"], body["reasoning"], body["instructions"]) == (
        DEFAULT_MODEL,
        {"effort": "high"},
        "Catalogue the document.",
    )
    text, image = body["input"][0]["content"]
    assert text == {"type": "input_text", "text": "330.9"}
    assert image["image_url"].startswith("data:image/webp;base64,")
    assert body["text"]["format"]["name"] == "ProbeOutput"


def test_transient_errors_retry_within_five_attempts() -> None:
    api = probe_api()
    api.script("/responses", connect_error(), httpx2.ReadTimeout("slow", request=REQUEST))
    api.script("/responses", connect_error(), connect_error())

    assert isinstance(api.gateway().run(Probe()), ProbeOutput)
    assert len(api.requests("/responses")) == 5


def test_persistent_transient_errors_raise_provider_error() -> None:
    api = probe_api()
    api.script("/responses", *[connect_error() for _ in range(5)])

    with pytest.raises(ProviderError, match="after 5 attempts"):
        api.gateway().run(Probe())
    assert len(api.requests("/responses")) == 5


def test_non_transient_errors_are_not_retried() -> None:
    api = probe_api()
    api.script("/responses", httpx2.Response(400, json={"error": {"message": "bad schema"}}))

    with pytest.raises(ProviderError, match="bad schema"):
        api.gateway().run(Probe())
    assert len(api.requests("/responses")) == 1


def test_truncated_output_retries_once_with_doubled_token_budget_and_meters_both() -> None:
    api = probe_api()
    truncated = META.model_dump_json()[:20]
    api.script(
        "/responses",
        response_body([message(truncated)], "incomplete", "max_output_tokens"),
        structured_body(META),
    )
    gateway = api.gateway()

    assert gateway.run(Probe()) == META
    assert [b["max_output_tokens"] for b in api.requests("/responses")] == [16384, 32768]
    (row,) = gateway.meter.rows()
    assert (row.operation, row.model, row.calls, row.input_tokens) == (
        "probe",
        DEFAULT_MODEL,
        2,
        200,
    )


def test_other_incomplete_status_or_bad_output_raises_without_retry() -> None:
    api = probe_api()
    api.script(
        "/responses",
        response_body([message("{}")], "incomplete", "content_filter"),
        response_body([message('{"title": ')]),
        httpx2.Response(200, text="<html>bad gateway</html>"),
        response_body([message(META.model_dump_json())], tokens={}),
    )
    gateway = api.gateway()

    with pytest.raises(ProviderError, match="status 'incomplete'"):
        gateway.run(Probe())
    with pytest.raises(ProviderError, match="invalid output"):
        gateway.run(Probe())
    with pytest.raises(ProviderError, match="non-JSON body"):
        gateway.run(Probe())
    with pytest.raises(ProviderError, match="malformed usage"):
        gateway.run(Probe())
    assert len(api.requests("/responses")) == 4


def test_retries_honour_retry_after_and_share_one_deadline() -> None:
    def rate_limited(header: str, value: str) -> httpx2.Response:
        return httpx2.Response(429, headers={header: value}, json={"error": {"message": "slow"}})

    api = probe_api()
    api.script("/responses", rate_limited("retry-after-ms", "1"))
    api.gateway(request_timeout=30.0, operation_deadline=60.0, retry_delays=(9.0,) * 4).run(Probe())
    timeouts = [e.read_timeout for e in api.exchanges]
    assert len(timeouts) == 2 and all(t is not None and 0 < t <= 30.0 for t in timeouts)

    api.script("/responses", rate_limited("retry-after", "120"))
    short = api.gateway(request_timeout=30.0, operation_deadline=5.0)
    with pytest.raises(ProviderError, match="would pass the 5 s deadline"):
        short.run(Probe(prompt="other"))
    assert len(api.exchanges) == 3
    last = api.exchanges[2].read_timeout
    assert last is not None and last <= 5.0


def test_no_attempt_starts_once_the_deadline_has_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]

    def oversleep(seconds: float) -> None:
        # The backoff sleep returns well after the deadline.
        clock[0] += seconds + 10.0

    monkeypatch.setattr(llm, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=oversleep))
    api = probe_api()
    api.script("/responses", connect_error())

    with pytest.raises(ProviderError, match="deadline passed before attempt 2"):
        api.gateway(operation_deadline=5.0).run(Probe())
    assert len(api.requests("/responses")) == 1


def test_an_async_attempt_that_outlasts_the_deadline_ends_it_without_another_attempt() -> None:
    attempts: list[httpx2.Request] = []

    async def stall(request: httpx2.Request) -> httpx2.Response:
        # The transport ignores the request timeout, so only the whole-attempt deadline ends it.
        attempts.append(request)
        await asyncio.sleep(30)
        raise AssertionError("the attempt outlived the operation deadline")

    def stalling_client() -> openai.AsyncOpenAI:
        transport = httpx2.MockTransport(stall)
        return openai.AsyncOpenAI(
            api_key="test-key",
            base_url=TEST_BASE_URL,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=transport),
        )

    gateway = OpenAIGateway(
        client=probe_api().client(),
        async_client_factory=stalling_client,
        request_timeout=30.0,
        operation_deadline=0.2,
        retry_delays=(0.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ProviderError, match=r"attempt 1 did not finish within the 0\.2 s deadline"):
        gateway.run_agent(AgentProbe())
    assert len(attempts) == 1


@pytest.mark.parametrize("value", ["NaN", "inf", "soon"])
def test_retry_after_that_is_not_a_finite_number_is_ignored(value: str) -> None:
    api = probe_api()
    rate_limited = httpx2.Response(
        429, headers={"retry-after": value}, json={"error": {"message": "slow"}}
    )
    api.script("/responses", rate_limited, structured_body(META))

    assert api.gateway().run(Probe()) == META
    assert len(api.requests("/responses")) == 2


def test_spend_limit_is_checked_before_every_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    meter = UsageMeter(max_cost_usd=0.30)
    api = probe_api()
    api.script("/responses", connect_error())
    handle = api._handle

    def spend_elsewhere(request: httpx2.Request) -> httpx2.Response:
        # Another build thread's usage reaches the limit while this attempt is in flight.
        meter.record("other", DEFAULT_MODEL, **SPEND_030)
        return handle(request)

    monkeypatch.setattr(api, "_handle", spend_elsewhere)
    with pytest.raises(BudgetExceededError):
        api.gateway(meter=meter).run(Probe())
    assert len(api.requests("/responses")) == 1


def test_structured_call_fails_after_both_output_token_budgets_are_exhausted() -> None:
    api = probe_api()
    truncated = response_body([message('{"title": ')], "incomplete", "max_output_tokens")
    api.script("/responses", truncated, truncated)
    gateway = api.gateway()

    with pytest.raises(ProviderError, match="exceeded 32768 output tokens"):
        gateway.run(Probe())
    assert [b["max_output_tokens"] for b in api.requests("/responses")] == [16384, 32768]
    assert gateway.meter.rows()[0].calls == 2


def html_page() -> httpx2.Response:
    return httpx2.Response(
        200, text="<html>bad gateway</html>", headers={"content-type": "text/html"}
    )


def test_malformed_responses_raise_provider_error_on_every_call_type() -> None:
    api = probe_api()
    gateway = api.gateway()
    no_usage_decision = decision_body([0.5]) | {"usage": None}
    no_count_embedding = {
        "object": "list",
        "model": "m",
        "data": [{"object": "embedding", "index": 0, "embedding": [1.0] * 8}],
        "usage": {"total_tokens": 1},
    }
    vector = [1.0] * 8
    text_index_embedding = no_count_embedding | {
        "data": [
            {"object": "embedding", "index": "0", "embedding": vector},
            {"object": "embedding", "index": 1, "embedding": vector},
        ],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    no_output = response_body([message("{}")]) | {"output": None}
    api.script("/decisions", html_page(), no_usage_decision)
    api.script("/embeddings", html_page(), no_count_embedding, text_index_embedding)
    api.script(
        "/responses",
        html_page(),
        response_body([message("{}")], tokens={"input_tokens": 1}),
        no_output,
        no_output,
    )

    for _ in range(2):
        with pytest.raises(ProviderError):
            gateway.judge("check", "evidence", ["p"])
        with pytest.raises(ProviderError):
            gateway.embed_texts(["x"])
    with pytest.raises(ProviderError, match="indices"):
        gateway.embed_texts(["x", "y"])
    with pytest.raises(ProviderError, match="not a response"):
        gateway.run_agent(AgentProbe())
    with pytest.raises(ProviderError, match="malformed usage: output_tokens"):
        gateway.run_agent(AgentProbe(prompt="other"))
    with pytest.raises(ProviderError, match="unreadable response"):
        gateway.run_agent(AgentProbe(prompt="no output"))
    conversation = asyncio.run(
        gateway.converse(
            operation="probe",
            instructions="i",
            prompt="q",
            tools=(LOOKUP,),
            max_turns=2,
            effort="low",
        )
    )
    assert conversation.status == "provider_error"
    assert "unreadable response" in (conversation.error or "")

    # Usage without cached-token details is metered as uncached input.
    api.script(
        "/decisions", decision_body([0.5]) | {"usage": {"input_tokens": 7, "output_tokens": 1}}
    )
    assert gateway.judge("check", "fresh evidence", ["p"]) == [0.5]
    decision_rows = [r for r in gateway.meter.rows() if r.model == DECISION_MODEL]
    assert [(r.input_tokens, r.cached_input_tokens) for r in decision_rows] == [(7, 0)]


# Embeddings.


def test_embeddings_batch_map_rows_by_index_normalize_and_reject_bad_indices() -> None:
    api = probe_api()
    gateway = api.gateway()
    texts = [f"chunk {i}" for i in range(300)]

    matrix = gateway.embed_texts(texts)

    assert [len(b["input"]) for b in api.requests("/embeddings")] == [128, 128, 44]
    assert matrix.shape == (300, 8) and matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert gateway.meter.rows()[0].input_tokens == 300
    assert [len(b) for b in embedding_batches(["x" * 40_000] * 5)] == [2, 2, 1]
    assert embedding_batches(["x" * 150_000, "y"]) == [["x" * 150_000], ["y"]]

    vector = [0.0] * 8
    duplicate = {
        "object": "list",
        "model": "m",
        "data": [{"object": "embedding", "index": 0, "embedding": vector}] * 2,
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    api.script("/embeddings", duplicate)
    with pytest.raises(ProviderError, match="indices"):
        gateway.embed_texts(["duplicate", "x"])


def test_only_the_embedding_sdk_call_turns_type_errors_into_malformed_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = probe_api().gateway()

    def broken_meter() -> None:
        raise TypeError("meter bug")

    monkeypatch.setattr(gateway.meter, "check_budget", broken_meter)
    with pytest.raises(TypeError, match="meter bug"):
        gateway.embed_texts(["x"])


# Decisions.


def test_judge_returns_probabilities_in_order_with_refusals_as_none() -> None:
    api = probe_api()
    gateway = api.gateway()

    assert gateway.judge("bind_facet", "evidence", []) == []
    assert api.request_count == 0
    api.script("/decisions", decision_body([0.9, None, 0.1]))
    assert gateway.judge("bind_facet", "Doc: harbour memo", ["a", "b", "c"]) == [0.9, None, 0.1]
    (body,) = api.requests("/decisions")
    assert body == {
        "model": DECISION_MODEL,
        "input": "Doc: harbour memo",
        "questions": [{"type": "predicate", "instructions": p} for p in "abc"],
    }


def test_judge_splits_more_than_400_predicates_across_requests() -> None:
    api = probe_api()
    api.script("/decisions", decision_body([0.2] * 400), decision_body([0.8]))
    gateway = api.gateway()

    answers = gateway.judge("check_support", "evidence", [f"p{i}" for i in range(401)])

    assert answers == [0.2] * 400 + [0.8]
    assert [len(b["questions"]) for b in api.requests("/decisions")] == [400, 1]
    assert api.requests("/decisions")[1]["questions"][0]["instructions"] == "p400"
    (row,) = gateway.meter.rows()
    assert (row.model, row.calls, row.output_tokens) == (DECISION_MODEL, 2, 2)


@pytest.mark.parametrize("probability", [float("nan"), 1.5, -0.2])
def test_judge_rejects_probabilities_that_are_not_finite_or_outside_0_to_1(
    probability: float,
) -> None:
    api = probe_api()
    # httpx2 refuses to serialise NaN as JSON; json.dumps writes it as the provider would.
    body = json.dumps(decision_body([0.4, probability])).encode()
    api.script("/decisions", httpx2.Response(200, content=body, headers=JSON_HEADERS))

    with pytest.raises(ProviderError, match=r"not a number within 0\.\.1"):
        api.gateway().judge("check", "evidence", ["p", "q"])


@pytest.mark.parametrize("entry", ["[NaN]", "[1.5]", "[-0.2]", "[true]"])
def test_judge_cache_entries_that_are_not_numbers_within_0_to_1_are_misses(
    tmp_path: Path, entry: str
) -> None:
    api = probe_api()
    path = tmp_path / "cache.db"
    with BuildCache(path) as cache:
        gateway = api.gateway(cache=cache)
        gateway.judge("check", "evidence", ["p"])
        with sqlite3.connect(path) as db:
            db.execute("UPDATE llm_cache SET response_json = ?", (entry,))
        db.close()

        assert gateway.judge("check", "evidence", ["p"]) == [0.75]
        assert api.request_count == 2


# Agent runs.


def agent_turns(*bodies: Outcome) -> ScriptedOpenAI:
    api = probe_api()
    api.script("/responses", *bodies)
    return api


def test_agent_run_feeds_tool_input_errors_and_images_back_and_returns_structured_output() -> None:
    api = agent_turns(
        response_body([function_call("c1", "lookup", '{"key": "missing"}')]),
        response_body(
            [
                function_call("c2", "lookup", '{"key": "intro"}'),
                function_call("c3", "lookup", '{"key": "chart"}'),
            ]
        ),
        response_body([message(META.model_dump_json())]),
    )
    gateway = api.gateway()

    assert gateway.run_agent(AgentProbe()) == META

    first, _, third = api.requests("/responses")
    assert (first["reasoning"], first["model"]) == ({"effort": "medium"}, DEFAULT_MODEL)
    assert first["tools"][0]["strict"] is False and first["tools"][0]["name"] == "lookup"
    outputs = [i for i in third["input"] if i.get("type") == "function_call_output"]
    assert [o["output"] for o in outputs[:2]] == [
        "ERROR: NotFoundError: no section 'missing'",
        "value of intro",
    ]
    text, image = outputs[2]["output"]
    assert text == {"type": "input_text", "text": "chart page"}
    assert image["type"] == "input_image" and image["detail"] == "high"
    assert image["image_url"] == "data:image/webp;base64,UklGRi1jaGFydA=="
    assert gateway.meter.rows()[0].calls == 3


def test_agent_run_ends_on_provider_errors_from_tools_turns_or_the_turn_limit() -> None:
    def failing(args: dict[str, Any]) -> str:
        raise ProviderError("embedding service down")

    broken = Tool("lookup", "Read one value.", {"type": "object", "properties": {}}, failing)
    api = agent_turns(response_body([function_call("c1", "lookup", "{}")]))
    with pytest.raises(ProviderError, match="embedding service down"):
        api.gateway().run_agent(AgentProbe(tools=(broken,)))
    assert len(api.requests("/responses")) == 1

    looping = [response_body([function_call(f"c{i}", "lookup", '{"key": "k"}')]) for i in range(3)]
    api = agent_turns(*looping)
    with pytest.raises(ProviderError, match="did not finish within 2 turns"):
        api.gateway().run_agent(AgentProbe(max_turns=2))
    assert len(api.requests("/responses")) == 2

    api = agent_turns(response_body([message("{}")], "incomplete", "max_output_tokens"))
    with pytest.raises(ProviderError, match="status 'incomplete'"):
        api.gateway().run_agent(AgentProbe())


def test_agent_run_meters_completed_turns_when_a_later_turn_fails() -> None:
    api = agent_turns(
        response_body([function_call("c1", "lookup", '{"key": "k"}')], tokens=usage(40, 4)),
        httpx2.Response(400, json={"error": {"message": "context too long"}}),
    )
    gateway = api.gateway()

    with pytest.raises(ProviderError, match="context too long"):
        gateway.run_agent(AgentProbe())
    (row,) = gateway.meter.rows()
    assert (row.operation, row.calls, row.input_tokens, row.output_tokens) == (
        "agent_probe",
        1,
        40,
        4,
    )


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("tool bug"), SnapshotIntegrityError("database disk image is malformed")],
)
def test_agent_run_ends_on_tool_exceptions_that_are_not_invalid_input(failure: Exception) -> None:
    def broken(args: dict[str, Any]) -> str:
        raise failure

    api = agent_turns(
        response_body([function_call("c1", "lookup", "{}")]),
        response_body([message(META.model_dump_json())]),
    )
    tool = Tool("lookup", "Read.", {"type": "object", "properties": {}}, broken)
    with pytest.raises(type(failure), match=str(failure)):
        api.gateway().run_agent(AgentProbe(tools=(tool,)))
    assert len(api.requests("/responses")) == 1


def test_agent_turns_retry_transient_failures_under_the_gateway_policy() -> None:
    first_turn = response_body([function_call("c1", "lookup", '{"key": "k"}')])
    api = agent_turns(
        first_turn,
        connect_error(),
        connect_error(),
        response_body([message(META.model_dump_json())]),
    )
    gateway = api.gateway()

    assert gateway.run_agent(AgentProbe()) == META
    assert len(api.requests("/responses")) == 4
    assert gateway.meter.rows()[0].calls == 2

    api = agent_turns(first_turn, *[connect_error() for _ in range(5)])
    with pytest.raises(ProviderError, match="after 5 attempts"):
        api.gateway().run_agent(AgentProbe())
    assert len(api.requests("/responses")) == 6


def test_agent_run_returns_unknown_tool_names_to_the_model_and_fails_on_behaviour_errors() -> None:
    api = agent_turns(
        response_body([function_call("c1", "hallucinated_tool", "{}")]),
        response_body([message(META.model_dump_json())]),
    )
    assert api.gateway().run_agent(AgentProbe()) == META
    second = api.requests("/responses")[1]
    (output,) = [i for i in second["input"] if i.get("type") == "function_call_output"]
    assert output["call_id"] == "c1" and "hallucinated_tool" in str(output["output"])

    api = agent_turns(response_body([message("not a catalogue")]))
    with pytest.raises(ProviderError, match="failed"):
        api.gateway().run_agent(AgentProbe())


def test_conversation_records_each_tool_call_in_order_with_its_arguments_and_result() -> None:
    long_key = "k" * 3_000
    long_arguments = json.dumps({"key": long_key})
    api = agent_turns(
        response_body(
            [
                function_call("c1", "lookup", '{"key": "intro"}'),
                function_call("c2", "lookup", '{"key": "chart"}'),
            ]
        ),
        response_body([function_call("c3", "lookup", '{"key": "missing"}')]),
        response_body([function_call("c4", "lookup", '{"key": ')]),
        response_body([function_call("c5", "lookup", long_arguments)]),
        response_body([function_call("c6", "hallucinated_tool", "{}")]),
        response_body([message("done")]),
    )

    run = asyncio.run(
        api.gateway().converse(
            operation="probe",
            instructions="i",
            prompt="q",
            tools=(LOOKUP,),
            max_turns=8,
            effort="low",
        )
    )

    assert run.status == "completed"
    intro, chart, missing, malformed, long = run.tool_calls
    assert intro == llm.ToolCall("lookup", '{"key": "intro"}', 14, "value of intro", 0)
    assert (chart.result_chars, chart.result_head, chart.images) == (10, "chart page", 1)
    # Invalid input is text returned to the model, so its error message is the result.
    assert missing.result_head == "ERROR: NotFoundError: no section 'missing'"
    assert malformed.arguments == '{"key": '
    assert malformed.result_head is not None and malformed.result_head.startswith(
        "ERROR: JSONDecodeError"
    )
    assert long.arguments == (
        f"{long_arguments[:2_000]}[... cut, {len(long_arguments)} characters in all]"
    )
    assert long.result_chars == len(f"value of {long_key}")
    assert long.result_head == f"value of {long_key}"[:200]

    def failing(args: dict[str, Any]) -> str:
        raise ProviderError("embedding service down")

    broken = Tool("broken", "Fail.", {"type": "object", "properties": {}}, failing)
    api = agent_turns(
        response_body([function_call("c1", "lookup", '{"key": "a"}')]),
        response_body([function_call("c2", "broken", "{}")]),
    )
    ended = asyncio.run(
        api.gateway().converse(
            operation="probe",
            instructions="i",
            prompt="q",
            tools=(LOOKUP, broken),
            max_turns=4,
            effort="low",
        )
    )
    # A call whose tool raised is still listed, with no result.
    assert ended.status == "provider_error"
    assert [(c.name, c.result_chars, c.result_head) for c in ended.tool_calls] == [
        ("lookup", 10, "value of a"),
        ("broken", None, None),
    ]


def test_trace_passes_its_key_with_each_run_and_leaves_the_global_exporter_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(llm.TRACING_KILL_SWITCH, raising=False)
    exporter_key = default_exporter().api_key
    configs: list[Any] = []

    async def record_run(agent: Any, text: str, *, max_turns: int, run_config: Any) -> Any:
        # Nothing runs, so no trace is exported from the test.
        configs.append(run_config)
        return SimpleNamespace(final_output=META)

    monkeypatch.setattr(Runner, "run", record_run)
    probe_api().gateway(api_key="sk-trace-test", trace=True).run_agent(AgentProbe())
    probe_api().gateway(api_key="sk-other-test").run_agent(AgentProbe())

    traced, untraced = configs
    assert (traced.tracing_disabled, traced.tracing) == (False, {"api_key": "sk-trace-test"})
    assert (untraced.tracing_disabled, untraced.tracing) == (True, None)
    assert traced.tool_not_found_behavior == "return_error_to_model"
    assert default_exporter().api_key == exporter_key

    monkeypatch.setenv(llm.TRACING_KILL_SWITCH, "1")
    with pytest.raises(ProviderError, match=llm.TRACING_KILL_SWITCH):
        probe_api().gateway(api_key="sk-trace-test", trace=True)
    probe_api().gateway(api_key="sk-trace-test")


# Usage and the spend limit.


def test_spend_limit_stops_the_next_call_once_the_estimate_reaches_it() -> None:
    api = probe_api()
    # 100k input tokens at $2.00/1M plus 10k output at $10.00/1M: $0.30.
    api.script("/responses", structured_body(META, tokens=usage(100_000, 10_000)))
    gateway = api.gateway(meter=UsageMeter(max_cost_usd=0.30))

    gateway.run(Probe())
    assert gateway.meter.estimated_cost_usd == pytest.approx(0.30)
    for call in (
        lambda: gateway.run(Probe(prompt="next")),
        lambda: gateway.judge("check", "evidence", ["p"]),
        lambda: gateway.embed_texts(["x"]),
        lambda: gateway.run_agent(AgentProbe()),
    ):
        with pytest.raises(BudgetExceededError, match=r"reached the \$0.3 limit"):
            call()
    assert api.request_count == 1
    assert issubclass(BudgetExceededError, ProviderError)


def test_usage_meter_prices_cached_input_and_rejects_unpriced_models_under_a_limit() -> None:
    meter = UsageMeter()
    meter.record(
        "t", DEFAULT_MODEL, input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=0
    )
    meter.record(
        "j", DECISION_MODEL, input_tokens=1_000_000, cached_input_tokens=0, output_tokens=9
    )
    meter.record("x", "unlisted-model", input_tokens=5, cached_input_tokens=0, output_tokens=5)

    assert meter.estimated_cost_usd == pytest.approx(1.00 + 0.05 + 0.10)
    assert meter.summary()["unpriced_models"] == ["unlisted-model"]
    assert "excluding unpriced unlisted-model" in meter.summary_line()
    with pytest.raises(ProviderError, match="needs prices for other-model"):
        probe_api().gateway(model="other-model", meter=UsageMeter(max_cost_usd=1.0))
    with pytest.raises(ValueError, match="positive"):
        UsageMeter(max_cost_usd=0)


def test_usage_with_more_cached_than_input_tokens_is_rejected_before_it_is_metered() -> None:
    api = probe_api()
    # Metered, 10 of 20 cached tokens would be a negative cost under the spend limit.
    api.script("/responses", structured_body(META, tokens=usage(10, 1, cached=20)))
    gateway = api.gateway(meter=UsageMeter(max_cost_usd=0.30))

    with pytest.raises(ProviderError, match="20 cached of 10 input tokens"):
        gateway.run(Probe())
    assert gateway.meter.rows() == []


# Build cache.


def test_cache_hits_make_no_requests_and_every_key_input_is_a_miss(tmp_path: Path) -> None:
    api = probe_api()
    for _ in range(6):
        api.script("/responses", structured_body(META))
    with BuildCache(tmp_path / "cache.db") as cache:
        gateway = api.gateway(cache=cache)
        probe = Probe(image_webp=b"RIFF-a")
        agent = AgentProbe()
        first = gateway.run(probe)
        gateway.run_agent(agent)
        gateway.judge("check", "evidence", ["p", "q"])
        vector = gateway.embed_query("rainfall")
        requests = api.request_count

        assert gateway.run(probe) == first
        assert gateway.run_agent(agent) == META
        assert gateway.judge("check", "evidence", ["p", "q"]) == [0.75, 0.75]
        assert np.array_equal(gateway.embed_query("rainfall"), vector)
        assert api.request_count == requests

        gateway.run(replace(probe, image_webp=b"RIFF-b"))
        gateway.run(replace(probe, effort="high"))
        assert api.request_count == requests + 2
        changed_schema = Tool(LOOKUP.name, LOOKUP.description, {"type": "object"}, lookup)
        for variant in (
            replace(agent, effort="high"),
            replace(agent, tools=(changed_schema,)),
            replace(agent, max_turns=4),
            replace(agent, source_digest="b" * 64),
        ):
            api.script("/responses", structured_body(META))
            before = api.request_count
            gateway.run_agent(variant)
            assert api.request_count == before + 1, variant


def test_concurrent_misses_for_one_key_return_the_first_stored_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = META, META.model_copy(update={"title": "Other transcript"})
    api = probe_api()
    api.script("/responses", structured_body(first), structured_body(second))
    handle = api._handle
    started: list[bool] = []
    overtaken: list[ProbeOutput] = []

    def overtake(request: httpx2.Request) -> httpx2.Response:
        # While the first request is in flight, another thread misses the same key and finishes.
        if not started:
            started.append(True)
            overtaken.append(gateway.run(Probe()))
        return handle(request)

    monkeypatch.setattr(api, "_handle", overtake)
    with BuildCache(tmp_path / "cache.db") as cache:
        gateway = api.gateway(cache=cache)

        assert gateway.run(Probe()) == first
        assert overtaken == [first]
        assert gateway.run(Probe()) == first
        assert api.request_count == 2


def test_cache_instances_on_one_file_keep_the_first_stored_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cache.db"
    with BuildCache(path) as first, BuildCache(path) as second:
        real = first._db
        stored: dict[str, str] = {}
        racer = threading.Thread(target=lambda: stored.update(second=second.store("k", "B", str)))

        class StoreRacesAfterRead:
            """The other instance stores the same key between this one's read and its write."""

            def execute(self, sql: str, *params: Any) -> Any:
                cursor = real.execute(sql, *params)
                if not sql.startswith("SELECT") or racer.ident is not None:
                    return cursor
                rows = cursor.fetchall()
                racer.start()
                racer.join(timeout=0.5)
                return SimpleNamespace(fetchone=lambda: rows[0] if rows else None)

            def __getattr__(self, name: str) -> Any:
                return getattr(real, name)

        monkeypatch.setattr(first, "_db", StoreRacesAfterRead())
        stored["first"] = first.store("k", "A", str)
        racer.join()

        assert stored == {"first": "A", "second": "A"}
        assert second.get("k") == "A"


def test_undecodable_cache_entries_are_misses_that_get_replaced(tmp_path: Path) -> None:
    api = probe_api()
    path = tmp_path / "new-dir" / "cache.db"
    with BuildCache(path) as cache:
        gateway = api.gateway(cache=cache)
        meta = gateway.run(Probe())
        vector = gateway.embed_query("rainfall")
        answers = gateway.judge("check", "evidence", ["p"])
        with sqlite3.connect(path) as db:
            db.execute("UPDATE llm_cache SET response_json = 'not base64!'")
        db.close()

        assert gateway.run(Probe()) == meta
        assert np.array_equal(gateway.embed_query("rainfall"), vector)
        assert gateway.judge("check", "evidence", ["p"]) == answers
        assert api.request_count == 6
        gateway.run(Probe())
        gateway.embed_query("rainfall")
        gateway.judge("check", "evidence", ["p"])
        assert api.request_count == 6
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(json.dumps(gateway.describe())) == {
        "model": DEFAULT_MODEL,
        "decision_model": DECISION_MODEL,
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": "8",
    }
