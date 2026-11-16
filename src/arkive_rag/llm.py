"""Model gateway: the only code that talks to the model provider, with its build cache, usage
meter and spend limit.

Domain modules own prompts, output schemas and reasoning effort through request objects; the
gateway owns transport, retries, caching and metering. Single-shot structured calls (`run`) and
bounded agent runs (`run_agent`) are separate operations: a request never carries an optional
tool list.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import inspect
import json
import math
import os
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Literal, Protocol, Self, cast

import numpy as np
import openai
from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    RunConfig,
    Runner,
    ToolOutputImage,
    ToolOutputText,
)
from agents.exceptions import AgentsException, MaxTurnsExceeded
from agents.models.openai_responses import OpenAIResponsesModel
from agents.tool_context import ToolContext
from agents.tracing import TracingConfig
from numpy.typing import NDArray
from openai.types import CreateEmbeddingResponse
from openai.types.decision import (
    AnswerAnswerResourcePredicate,
    AnswerAnswerResourceRefusal,
    Decision,
)
from openai.types.responses import Response, ResponseInputParam
from openai.types.shared import Reasoning
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from arkive_rag.models import InvalidFilterError, NotFoundError, ProviderError

Effort = Literal["low", "medium", "high"]
# Tool exceptions that mean the model sent invalid input; they go back to it as text.
TOOL_INPUT_ERRORS = (NotFoundError, InvalidFilterError, ValueError, KeyError)

DEFAULT_MODEL = "gpt-6.1-sol"
DECISION_MODEL = "gpt-6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 128
# About 35k tokens per request, far below the provider's 300k-token batch ceiling.
EMBEDDING_BATCH_CHARS = 100_000
# The Decisions API accepts 1 to 400 questions per request.
MAX_JUDGE_PREDICATES = 400
# Waits between the five attempts of one provider operation.
RETRY_DELAYS_SECONDS = (2.0, 5.0, 12.0, 25.0)
BUILD_REQUEST_TIMEOUT_SECONDS = 180.0
BUILD_DEADLINE_SECONDS = 600.0
QUERY_REQUEST_TIMEOUT_SECONDS = 30.0
QUERY_DEADLINE_SECONDS = 60.0
INITIAL_MAX_OUTPUT_TOKENS = 16384
AGENT_MAX_OUTPUT_TOKENS = 16384
# When this is "1" or "true", the Agents SDK records no traces whatever a run's settings say.
TRACING_KILL_SWITCH = "OPENAI_AGENTS_DISABLE_TRACING"
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@dataclass(frozen=True)
class ModelPrice:
    """Standard-tier list price in USD per one million tokens."""

    input: float
    cached_input: float
    output: float


# The estimate ignores the long-context surcharge, so it understates requests whose input
# crosses that threshold. gpt-6-luna is priced as served by the Decisions API (no output charge).
PRICES: Mapping[str, ModelPrice] = {
    DEFAULT_MODEL: ModelPrice(input=2.00, cached_input=0.10, output=10.00),
    DECISION_MODEL: ModelPrice(input=0.10, cached_input=0.01, output=0.0),
    DEFAULT_EMBEDDING_MODEL: ModelPrice(input=0.13, cached_input=0.13, output=0.0),
}


class BudgetExceededError(ProviderError):
    """The estimated spend reached the configured limit before another provider call."""


def load_dotenv_if_present(path: Path) -> None:
    """Load KEY=VALUE lines from `path` without overriding variables already set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# Requests and tools.


class StructuredRequest[T: BaseModel](Protocol):
    """One single-shot structured call. Its owner chooses the prompt, schema and effort.

    `operation` is a stable name: it is part of the cache key and labels the usage row.
    """

    @property
    def operation(self) -> str: ...
    @property
    def output(self) -> type[T]: ...
    @property
    def effort(self) -> Effort: ...
    @property
    def instructions(self) -> str: ...
    @property
    def image_webp(self) -> bytes | None: ...
    def text(self) -> str: ...


@dataclass(frozen=True)
class ToolImage:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class ToolReply:
    """Tool output with images; a tool that returns only text returns a `str`."""

    text: str
    images: tuple[ToolImage, ...] = ()


type ToolResult = str | ToolReply


@dataclass(frozen=True)
class Tool:
    """A read-only, provider-neutral tool an agent may call.

    `parameters` is the JSON schema of the argument object. `invoke` receives the parsed
    arguments and may be a plain function or a coroutine function. Invalid input
    (`TOOL_INPUT_ERRORS`: `NotFoundError`, `InvalidFilterError`, `ValueError` or `KeyError`)
    goes back to the model as text; any other exception, such as `SnapshotIntegrityError` or a
    `ProviderError`, ends the run.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    invoke: Callable[[dict[str, Any]], ToolResult | Awaitable[ToolResult]]


class AgentRequest[T: BaseModel](Protocol):
    """One bounded agent run with read-only tools, ending in structured output.

    `source_digest` is a SHA-256 over everything the tools can read; it is part of the cache
    key, so a result computed while the tools saw one source set never answers another.
    """

    @property
    def operation(self) -> str: ...
    @property
    def output(self) -> type[T]: ...
    @property
    def effort(self) -> Effort: ...
    @property
    def max_turns(self) -> int: ...
    @property
    def tools(self) -> Sequence[Tool]: ...
    @property
    def source_digest(self) -> str: ...
    @property
    def instructions(self) -> str: ...
    def text(self) -> str: ...


TOOL_CALL_ARGUMENTS_CHARS = 2_000
TOOL_CALL_RESULT_HEAD_CHARS = 200


@dataclass(frozen=True)
class ToolCall:
    """One call the model made to one of the conversation's tools.

    `arguments` is the argument text exactly as the model sent it, so malformed JSON is kept;
    text over `TOOL_CALL_ARGUMENTS_CHARS` characters is cut there and ends with a marker giving
    the full length. `result_chars` is the length of the text returned to the model (an
    invalid-input error message included) and `result_head` its first
    `TOOL_CALL_RESULT_HEAD_CHARS` characters; both are None when the tool raised and so ended
    the run. `images` counts the images returned with the text.
    """

    name: str
    arguments: str
    result_chars: int | None
    result_head: str | None
    images: int


@dataclass(frozen=True)
class Conversation:
    """One free-text agent conversation. Only status "completed" is a usable answer.

    Other statuses: the final response's own status (for example "incomplete", with any
    partial `answer`), "empty_answer", "max_turns", or "provider_error" with `error` set.

    `tool_calls` lists the calls in the order they were invoked. A call is recorded before its
    tool runs, so a call whose tool raised is listed too. A call to a tool name the
    conversation does not have is answered by the Agents SDK and is not listed.
    """

    answer: str
    status: str
    error: str | None
    input_tokens: int
    output_tokens: int
    tool_calls: tuple[ToolCall, ...]
    turns: int
    elapsed_seconds: float


class LanguageGateway(Protocol):
    """Everything the compiler needs from language and embedding models."""

    @property
    def embedding_model(self) -> str: ...

    @property
    def embedding_dimensions(self) -> int: ...

    def describe(self) -> dict[str, str]: ...

    def run[T: BaseModel](self, request: StructuredRequest[T]) -> T: ...

    def run_agent[T: BaseModel](self, request: AgentRequest[T]) -> T: ...

    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        """One probability per predicate, in order; None where the provider refused it."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


# Build cache.


class BuildCache:
    """Content-addressed store of validated provider responses, shared by build threads."""

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # A new cache file is private (0600); SQLite gives its WAL files the same mode.
            with contextlib.suppress(FileExistsError):
                os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            self._db = sqlite3.connect(path, check_same_thread=False)
        except (sqlite3.Error, OSError) as exc:
            raise ProviderError(f"cannot open build cache {path}: {exc}") from exc
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (cache_key TEXT PRIMARY KEY,"
                " response_json TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            self._db.commit()
        except sqlite3.Error as exc:
            self._db.close()
            raise ProviderError(f"cannot open build cache {path}: {exc}") from exc

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256(json.dumps(list(parts)).encode()).hexdigest()

    def get(self, cache_key: str) -> str | None:
        try:
            with self._lock:
                row = self._db.execute(
                    "SELECT response_json FROM llm_cache WHERE cache_key = ?", (cache_key,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise ProviderError(f"build cache read failed: {exc}") from exc
        return None if row is None else str(row[0])

    def store[R](self, cache_key: str, response_json: str, decode: Callable[[str], R | None]) -> R:
        """Store `response_json` unless a valid entry exists, and return the stored entry decoded.

        The first valid entry for a key is canonical: when two build threads miss the same key
        concurrently, the later writer gets the earlier writer's value, so this build and the
        next one see the same response. An entry `decode` rejects is replaced.
        """
        existing: R | None = None
        try:
            with self._lock:
                # One write transaction from the read to the write, so the rule also holds
                # between processes and other `BuildCache` instances on this file.
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    row = self._db.execute(
                        "SELECT response_json FROM llm_cache WHERE cache_key = ?", (cache_key,)
                    ).fetchone()
                    if row is not None:
                        existing = decode(str(row[0]))
                    if existing is None:
                        self._db.execute(
                            "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)",
                            (cache_key, response_json, datetime.now(UTC).isoformat()),
                        )
                    self._db.commit()
                except BaseException:
                    self._db.rollback()
                    raise
        except sqlite3.Error as exc:
            raise ProviderError(f"build cache write failed: {exc}") from exc
        if existing is not None:
            return existing
        stored = decode(response_json)
        if stored is None:
            raise AssertionError(f"build cache entry for {cache_key} does not decode")
        return stored

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# Usage metering and the spend limit.


@dataclass(frozen=True)
class UsageRow:
    operation: str
    model: str
    calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    # None when the model has no entry in `PRICES`.
    estimated_usd: float | None


class UsageMeter:
    """Thread-safe token counts per (operation, model) and their estimated cost.

    With `max_cost_usd`, `check_budget` raises `BudgetExceededError` once the estimate has
    reached the limit, so the provider call that would follow is never made.
    """

    def __init__(self, max_cost_usd: float | None = None) -> None:
        if max_cost_usd is not None and not max_cost_usd > 0:
            raise ValueError(f"spend limit must be positive, got {max_cost_usd}")
        self.max_cost_usd = max_cost_usd
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], list[int]] = {}

    def require_priced(self, models: Sequence[str]) -> None:
        """A spend limit is only enforceable when every model it covers has a price."""
        unpriced = sorted({m for m in models if m not in PRICES})
        if self.max_cost_usd is not None and unpriced:
            raise ProviderError(
                f"a spend limit needs prices for {', '.join(unpriced)}; priced models are"
                f" {', '.join(sorted(PRICES))}"
            )

    def record(
        self,
        operation: str,
        model: str,
        *,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
    ) -> None:
        with self._lock:
            counts = self._counts.setdefault((operation, model), [0, 0, 0, 0])
            counts[0] += 1
            counts[1] += input_tokens
            counts[2] += cached_input_tokens
            counts[3] += output_tokens

    def rows(self) -> list[UsageRow]:
        with self._lock:
            snapshot = sorted((key, list(counts)) for key, counts in self._counts.items())
        rows: list[UsageRow] = []
        for (operation, model), (calls, tokens_in, cached, tokens_out) in snapshot:
            price = PRICES.get(model)
            cost = None
            if price is not None:
                cost = (
                    (tokens_in - cached) * price.input
                    + cached * price.cached_input
                    + tokens_out * price.output
                ) / 1_000_000
            rows.append(UsageRow(operation, model, calls, tokens_in, cached, tokens_out, cost))
        return rows

    @property
    def estimated_cost_usd(self) -> float:
        """The estimate over priced models; `rows` marks any unpriced one."""
        return sum(row.estimated_usd or 0.0 for row in self.rows())

    def check_budget(self) -> None:
        if self.max_cost_usd is None:
            return
        spent = self.estimated_cost_usd
        if spent >= self.max_cost_usd:
            raise BudgetExceededError(
                f"estimated spend ${spent:.4f} reached the ${self.max_cost_usd:g} limit;"
                " no further provider calls are made"
            )

    def summary(self) -> dict[str, Any]:
        rows = self.rows()
        return {
            "calls": sum(r.calls for r in rows),
            "input_tokens": sum(r.input_tokens for r in rows),
            "cached_input_tokens": sum(r.cached_input_tokens for r in rows),
            "output_tokens": sum(r.output_tokens for r in rows),
            "estimated_cost_usd": round(sum(r.estimated_usd or 0.0 for r in rows), 6),
            "unpriced_models": sorted({r.model for r in rows if r.estimated_usd is None}),
            "max_cost_usd": self.max_cost_usd,
            "rows": [vars(r) for r in rows],
        }

    def summary_line(self) -> str:
        s = self.summary()
        line = (
            f"usage: {s['calls']} provider calls, {s['input_tokens']} input tokens"
            f" ({s['cached_input_tokens']} cached), {s['output_tokens']} output tokens,"
            f" estimated ${s['estimated_cost_usd']:.4f}"
        )
        if s["unpriced_models"]:
            line += f" (excluding unpriced {', '.join(s['unpriced_models'])})"
        return line


# Retry policy.


def embedding_batches(texts: Sequence[str]) -> list[list[str]]:
    """Group texts into requests bounded by input count and total characters."""
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and (
            len(current) == EMBEDDING_BATCH_SIZE or size + len(text) > EMBEDDING_BATCH_CHARS
        ):
            batches.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    return [*batches, current] if current else batches


def _retry_after(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, when it sent a finite numeric Retry-After header."""
    if not isinstance(exc, openai.APIStatusError):
        return None
    for header, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = exc.response.headers.get(header)
        if value is not None:
            try:
                seconds = float(value) / divisor
            except ValueError:
                return None
            return max(seconds, 0.0) if math.isfinite(seconds) else None
    return None


def _retry_delay(
    exc: Exception,
    attempt: int,
    deadline: float,
    operation_deadline: float,
    retry_delays: Sequence[float],
) -> float:
    """The wait before the next attempt, or `ProviderError` when no attempt is left."""
    if attempt > len(retry_delays):
        raise ProviderError(f"provider unavailable after {attempt} attempts: {exc}") from exc
    delay = _retry_after(exc)
    if delay is None:
        delay = retry_delays[attempt - 1]
    if time.monotonic() + delay >= deadline:
        raise ProviderError(
            f"provider unavailable after {attempt} attempts; waiting {delay:g} s"
            f" would pass the {operation_deadline:g} s deadline: {exc}"
        ) from exc
    return delay


def _time_left(deadline: float, attempt: int, operation_deadline: float) -> float:
    """Seconds before `deadline`, or `ProviderError` when no attempt may start."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProviderError(
            f"the {operation_deadline:g} s provider deadline passed before attempt {attempt}"
        )
    return remaining


def retry_transient[R](
    call: Callable[..., R],
    *,
    check_budget: Callable[[], None],
    request_timeout: float,
    operation_deadline: float,
    retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
) -> R:
    """Run `call(timeout=...)`, retrying transient failures until one shared deadline.

    `check_budget` runs before every attempt, so a retry never starts once the spend limit is
    reached. The client behind `call` must be built with `max_retries=0` so retries never
    stack. Every failure, transient or not, surfaces as `ProviderError`.
    """
    deadline = time.monotonic() + operation_deadline
    attempt = 0
    while True:
        attempt += 1
        check_budget()
        remaining = _time_left(deadline, attempt, operation_deadline)
        try:
            return call(timeout=min(request_timeout, remaining))
        except _TRANSIENT_ERRORS as exc:
            time.sleep(_retry_delay(exc, attempt, deadline, operation_deadline, retry_delays))
        except (openai.OpenAIError, ValidationError) as exc:
            raise ProviderError(f"provider request failed: {exc}") from exc


async def retry_transient_async[R](
    call: Callable[..., Awaitable[R]],
    *,
    check_budget: Callable[[], None],
    request_timeout: float,
    operation_deadline: float,
    retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
) -> R:
    """`retry_transient` for an awaitable call; the same policy, waiting without blocking.

    Each attempt as a whole, not only each socket read, ends at the shared deadline.
    """
    deadline = time.monotonic() + operation_deadline
    attempt = 0
    while True:
        attempt += 1
        check_budget()
        remaining = _time_left(deadline, attempt, operation_deadline)
        try:
            async with asyncio.timeout(remaining):
                return await call(timeout=min(request_timeout, remaining))
        except TimeoutError as exc:
            raise ProviderError(
                f"provider attempt {attempt} did not finish within the {operation_deadline:g} s"
                " deadline"
            ) from exc
        except _TRANSIENT_ERRORS as exc:
            await asyncio.sleep(
                _retry_delay(exc, attempt, deadline, operation_deadline, retry_delays)
            )
        except (openai.OpenAIError, ValidationError) as exc:
            raise ProviderError(f"provider request failed: {exc}") from exc


# Agents SDK adapters.


class _IncompleteResponse(ProviderError):
    """An agent turn whose response finished with a status other than "completed"."""

    def __init__(self, message: str, status: str, partial_text: str) -> None:
        super().__init__(message)
        self.status = status
        self.partial_text = partial_text


class _ToolFailure(Exception):
    """Carries an exception a tool raised through the SDK, which wraps it in `UserError`."""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


@dataclass
class _RunTally:
    """Counts and tool calls for one agent run; only touched from that run's event loop."""

    turns: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def _recorded_arguments(arguments: str) -> str:
    if len(arguments) <= TOOL_CALL_ARGUMENTS_CHARS:
        return arguments
    return f"{arguments[:TOOL_CALL_ARGUMENTS_CHARS]}[... cut, {len(arguments)} characters in all]"


class _GatewayResponsesModel(OpenAIResponsesModel):
    """The SDK's Responses model with each turn's request routed through the gateway.

    One model turn is one provider operation: the gateway's retry policy, deadline, spend
    limit check and metering apply to it, and a non-completed response raises
    `_IncompleteResponse` after its usage is recorded.
    """

    def __init__(
        self,
        model: str,
        client: openai.AsyncOpenAI,
        turn: Callable[[dict[str, Any]], Awaitable[Response]],
        label: str,
    ) -> None:
        super().__init__(model, client)
        self._turn = turn
        self._label = label

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        # Tools never run inside this call, so a validation error here is the SDK failing to
        # build its model response from a malformed body (for example "output": null).
        try:
            return await super().get_response(*args, **kwargs)
        except ValidationError as exc:
            raise ProviderError(f"{self._label} returned an unreadable response: {exc}") from exc

    # The SDK overloads this private method on `stream`; the gateway never streams. Pinned
    # with openai-agents==0.24.0, whose request building `_build_response_create_kwargs` owns.
    async def _fetch_response(  # type: ignore[override]
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        stream: bool = False,
        prompt: Any = None,
    ) -> Response:
        if stream:
            raise RuntimeError("the gateway does not stream agent turns")
        create_kwargs = self._build_response_create_kwargs(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            stream=False,
            prompt=prompt,
        )
        return await self._turn(create_kwargs)


def _sdk_tool_output(result: ToolResult) -> str | list[ToolOutputText | ToolOutputImage]:
    if isinstance(result, str):
        return result
    images = [
        ToolOutputImage(
            image_url=f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode()}",
            detail="high",
        )
        for image in result.images
    ]
    return [ToolOutputText(text=result.text), *images]


def _sdk_tool(tool: Tool, tally: _RunTally) -> FunctionTool:
    async def invoke(_context: ToolContext[Any], arguments: str) -> Any:
        index = len(tally.tool_calls)
        tally.tool_calls.append(ToolCall(tool.name, _recorded_arguments(arguments), None, None, 0))
        try:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("tool arguments must be a JSON object")
            result = tool.invoke(args)
            if inspect.isawaitable(result):
                result = await result
        except TOOL_INPUT_ERRORS as exc:
            # Invalid input: recovering from it is part of the agent's task.
            result = f"ERROR: {type(exc).__name__}: {exc}"
        except Exception as exc:
            raise _ToolFailure(exc) from exc
        text, images = (result, ()) if isinstance(result, str) else (result.text, result.images)
        tally.tool_calls[index] = replace(
            tally.tool_calls[index],
            result_chars=len(text),
            result_head=text[:TOOL_CALL_RESULT_HEAD_CHARS],
            images=len(images),
        )
        return _sdk_tool_output(result)

    # Non-strict: tool schemas come from callers (and MCP servers) with optional properties.
    return FunctionTool(
        name=tool.name,
        description=tool.description,
        params_json_schema=dict(tool.parameters),
        on_invoke_tool=invoke,
        strict_json_schema=False,
    )


def _canonical_tools(tools: Sequence[Tool]) -> list[Tool]:
    ordered = sorted(tools, key=lambda t: t.name)
    names = [t.name for t in ordered]
    if len(set(names)) != len(names):
        raise ValueError(f"agent tool names must be unique: {names}")
    return ordered


# The gateway.


class OpenAIGateway:
    """OpenAI-backed gateway. It is the single owner of provider retry policy.

    Each operation gets up to five attempts that share one deadline; every request's timeout
    is capped by the time left before that deadline. Every provider response is metered, and
    with a spend limit no provider call starts once the estimate has reached it. Hosted
    Agents SDK tracing is off unless `trace` is set.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        decision_model: str = DECISION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        cache: BuildCache | None = None,
        meter: UsageMeter | None = None,
        trace: bool = False,
        client: openai.OpenAI | None = None,
        async_client_factory: Callable[[], openai.AsyncOpenAI] | None = None,
        request_timeout: float = BUILD_REQUEST_TIMEOUT_SECONDS,
        operation_deadline: float = BUILD_DEADLINE_SECONDS,
        retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
    ) -> None:
        self.meter = meter or UsageMeter()
        self.meter.require_priced([model, decision_model, embedding_model])
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if (client is None or async_client_factory is None) and not key:
            raise ProviderError("OPENAI_API_KEY is not set; pass it or add it to .env")
        # max_retries=0: the SDK must not retry underneath this gateway's own policy.
        self._client = client or openai.OpenAI(api_key=key, max_retries=0, timeout=request_timeout)
        # An AsyncOpenAI client must not be shared across event loops, so every agent run
        # builds its own on the loop that runs it and closes it afterwards.
        self._async_client_factory = async_client_factory or partial(
            openai.AsyncOpenAI, api_key=key, max_retries=0, timeout=request_timeout
        )
        if trace and os.environ.get(TRACING_KILL_SWITCH, "").lower() in ("true", "1"):
            raise ProviderError(
                f"tracing was requested, but {TRACING_KILL_SWITCH} disables Agents SDK tracing"
                " in this environment; unset it or drop --trace"
            )
        # Passed per run, so gateways with different keys never share exporter state.
        self._tracing: TracingConfig | None = TracingConfig(api_key=key) if trace and key else None
        self._trace = trace
        self._model = model
        self._decision_model = decision_model
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._cache = cache
        self._request_timeout = request_timeout
        self._operation_deadline = operation_deadline
        self._retry_delays = tuple(retry_delays)

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def embedding_dimensions(self) -> int:
        return self._embedding_dimensions

    def describe(self) -> dict[str, str]:
        return {
            "model": self._model,
            "decision_model": self._decision_model,
            "embedding_model": self._embedding_model,
            "embedding_dimensions": str(self._embedding_dimensions),
        }

    # Structured calls.

    def run[T: BaseModel](self, request: StructuredRequest[T]) -> T:
        """One `responses.parse` call; the cache key and the request come from the same inputs."""
        operation, schema, effort = request.operation, request.output, request.effort
        instructions, text, image = request.instructions, request.text(), request.image_webp
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        image_hash = ""
        if image is not None:
            image_url = "data:image/webp;base64," + base64.b64encode(image).decode()
            content.append({"type": "input_image", "image_url": image_url, "detail": "high"})
            image_hash = hashlib.sha256(image).hexdigest()
        schema_text = json.dumps(schema.model_json_schema(), sort_keys=True)
        cache_key = BuildCache.key(
            operation, self._model, effort, instructions, text, schema_text, image_hash
        )
        cached = self._cached_model(cache_key, schema)
        if cached is not None:
            return cached
        label = f"{operation} on {self._model}"
        max_tokens = INITIAL_MAX_OUTPUT_TOKENS
        for _ in range(2):
            # The raw response lets the status be read before the SDK parses the output:
            # output truncated at max_output_tokens is invalid JSON and fails parsing.
            raw = self._with_retry(
                partial(
                    self._client.responses.with_raw_response.parse,
                    model=self._model,
                    instructions=instructions,
                    input=cast("ResponseInputParam", [{"role": "user", "content": content}]),
                    text_format=schema,
                    reasoning={"effort": effort},
                    max_output_tokens=max_tokens,
                )
            )
            try:
                body = json.loads(raw.text)
            except ValueError as exc:
                raise ProviderError(f"{label} returned a non-JSON body") from exc
            if not isinstance(body, dict):
                raise ProviderError(f"{label} returned a non-object body")
            self._record_usage(operation, self._model, label, body.get("usage"))
            reason = (body.get("incomplete_details") or {}).get("reason")
            if body.get("status") == "incomplete" and reason == "max_output_tokens":
                max_tokens *= 2
                continue
            if body.get("status") != "completed":
                raise ProviderError(f"{label} returned status {body.get('status')!r}")
            try:
                parsed: T | None = raw.parse().output_parsed
            except (ValueError, openai.OpenAIError) as exc:
                raise ProviderError(f"{label} returned invalid output: {exc}") from exc
            if parsed is None:
                raise ProviderError(f"{label} returned no parsed output")
            return self._store_model(cache_key, parsed)
        raise ProviderError(f"{label} exceeded {max_tokens // 2} output tokens")

    # Decisions.

    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        """Decisions `predicate` questions over shared evidence, in order.

        Each answer is the probability that the predicate holds, or None where the provider
        refused that question. Empty predicates make no request; more than
        `MAX_JUDGE_PREDICATES` are split across requests.
        """
        results: list[float | None] = []
        for start in range(0, len(predicates), MAX_JUDGE_PREDICATES):
            batch = list(predicates[start : start + MAX_JUDGE_PREDICATES])
            results += self._judge_batch(operation, evidence, batch)
        return results

    def _judge_batch(
        self, operation: str, evidence: str, predicates: list[str]
    ) -> list[float | None]:
        cache_key = BuildCache.key(
            operation, self._decision_model, evidence, json.dumps(predicates)
        )

        def decode(entry: str) -> list[float | None] | None:
            # An entry that no longer validates is a miss; the fresh response replaces it.
            try:
                stored = _JUDGE_ANSWERS.validate_json(entry)
            except ValidationError:
                return None
            return stored if len(stored) == len(predicates) else None

        cached = self._cache.get(cache_key) if self._cache else None
        if cached is not None and (hit := decode(cached)) is not None:
            return hit
        label = f"{operation} on {self._decision_model}"
        decision = self._with_retry(
            partial(
                self._client.decisions.create,
                model=self._decision_model,
                input=evidence,
                questions=[{"type": "predicate", "instructions": p} for p in predicates],
            )
        )
        if not isinstance(decision, Decision):
            raise ProviderError(f"{label} returned a body that is not a decision")
        self._record_usage(operation, self._decision_model, label, decision.usage)
        if not isinstance(decision.answers, list) or len(decision.answers) != len(predicates):
            count = len(decision.answers) if isinstance(decision.answers, list) else "no"
            raise ProviderError(f"{label} returned {count} answers for {len(predicates)} questions")
        answers: list[float | None] = []
        for answer in decision.answers:
            if isinstance(answer, AnswerAnswerResourceRefusal):
                answers.append(None)
            elif isinstance(answer, AnswerAnswerResourcePredicate):
                try:
                    answers.append(_PROBABILITY.validate_python(answer.probability))
                except ValidationError as exc:
                    raise ProviderError(
                        f"{label} returned probability {answer.probability!r}, not a number"
                        " within 0..1"
                    ) from exc
            else:
                raise ProviderError(f"{label} returned a {type(answer).__name__} to a predicate")
        if self._cache:
            return self._cache.store(cache_key, json.dumps(answers), decode)
        return answers

    # Embeddings.

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self._embedding_dimensions), dtype=np.float32)
        return np.vstack([self._embed_batch("embed", batch) for batch in embedding_batches(texts)])

    def embed_query(self, query: str) -> NDArray[np.float32]:
        vector: NDArray[np.float32] = self._embed_batch("embed_query", [query])[0]
        return vector

    def _decode_matrix(self, entry: str, rows: int) -> NDArray[np.float32] | None:
        """A cached embedding matrix, or None for an entry that does not decode."""
        try:
            raw = np.frombuffer(base64.b64decode(entry, validate=True), dtype=np.float32)
            matrix = raw.reshape(rows, self._embedding_dimensions).copy()
        except ValueError:
            return None
        return matrix if bool(np.all(np.isfinite(matrix))) else None

    def _embed_batch(self, operation: str, texts: list[str]) -> NDArray[np.float32]:
        """`operation` labels the usage row only; the cache key depends on the texts."""
        cache_key = BuildCache.key(
            "embed", self._embedding_model, json.dumps(texts), str(self._embedding_dimensions), ""
        )
        decode = partial(self._decode_matrix, rows=len(texts))
        cached = self._cache.get(cache_key) if self._cache else None
        if cached is not None and (hit := decode(cached)) is not None:
            return hit
        label = f"{operation} on {self._embedding_model}"

        def create(*, timeout: float) -> CreateEmbeddingResponse:
            try:
                return self._client.embeddings.create(
                    model=self._embedding_model,
                    input=texts,
                    dimensions=self._embedding_dimensions,
                    timeout=timeout,
                )
            except (AttributeError, TypeError, ValueError) as exc:
                # The SDK decodes base64 vectors inside `create` and fails this way on a body
                # that is not an embedding list, such as an HTML error page sent with status 200.
                raise ProviderError(f"{label} returned a malformed body: {exc!r}") from exc

        response = self._with_retry(create)
        if not isinstance(response, CreateEmbeddingResponse):
            raise ProviderError(f"{label} returned a body that is not an embedding list")
        usage = _usage_fields(label, response.usage)
        self.meter.record(
            operation,
            self._embedding_model,
            input_tokens=_token_count(label, usage, "prompt_tokens"),
            cached_input_tokens=0,
            output_tokens=0,
        )
        if not isinstance(response.data, list):
            raise ProviderError(f"{label} returned no embedding rows")
        indices = [item.index for item in response.data]
        if any(isinstance(i, bool) or not isinstance(i, int) for i in indices) or sorted(
            indices
        ) != list(range(len(texts))):
            raise ProviderError(
                f"embedding response indices {indices} do not cover 0..{len(texts) - 1} once each"
            )
        rows = sorted(response.data, key=lambda item: item.index)
        try:
            matrix = np.asarray([row.embedding for row in rows], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"{label} returned malformed vectors: {exc}") from exc
        if matrix.shape != (len(texts), self._embedding_dimensions):
            raise ProviderError(
                f"embedding response shape {matrix.shape} does not match"
                f" ({len(texts)}, {self._embedding_dimensions})"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms) | (norms == 0)):
            raise ProviderError("embedding response contains a zero or non-finite vector")
        matrix = (matrix / norms).astype(np.float32)
        if self._cache:
            return self._cache.store(cache_key, base64.b64encode(matrix.tobytes()).decode(), decode)
        return matrix

    # Agent runs.

    def run_agent[T: BaseModel](self, request: AgentRequest[T]) -> T:
        """A bounded Agents SDK run ending in `request.output`, on its own event loop.

        Not finishing within `max_turns`, a non-completed response, a model behaviour error,
        or a provider or budget error raises `ProviderError`. A tool exception other than
        invalid input propagates unchanged; a `ProviderError` from a tool is one.
        """
        tools = _canonical_tools(request.tools)
        instructions, text, schema = request.instructions, request.text(), request.output
        tool_specs = json.dumps(
            [[t.name, t.description, dict(t.parameters)] for t in tools], sort_keys=True
        )
        cache_key = BuildCache.key(
            request.operation,
            self._model,
            request.effort,
            instructions,
            text,
            json.dumps(schema.model_json_schema(), sort_keys=True),
            str(request.max_turns),
            tool_specs,
            request.source_digest,
        )
        cached = self._cached_model(cache_key, schema)
        if cached is not None:
            return cached
        label = f"{request.operation} on {self._model}"

        async def run() -> T:
            async with self._async_client_factory() as client:
                try:
                    output = await self._agent_run(
                        client,
                        operation=request.operation,
                        effort=request.effort,
                        instructions=instructions,
                        text=text,
                        tools=tools,
                        max_turns=request.max_turns,
                        output_type=schema,
                        tally=_RunTally(),
                    )
                except MaxTurnsExceeded as exc:
                    raise ProviderError(
                        f"{label} did not finish within {request.max_turns} turns"
                    ) from exc
                except AgentsException as exc:
                    raise ProviderError(f"{label} failed: {exc}") from exc
            if not isinstance(output, schema):
                raise ProviderError(f"{label} returned no {schema.__name__}")
            return output

        return self._store_model(cache_key, asyncio.run(run()))

    async def converse(
        self,
        *,
        operation: str,
        instructions: str,
        prompt: str,
        tools: Sequence[Tool],
        max_turns: int,
        effort: Effort,
    ) -> Conversation:
        """A free-text agent conversation on the caller's event loop, never cached.

        Tools may wrap resources bound to that loop. A provider failure, including a
        `ProviderError` from a tool, ends the conversation with status "provider_error". The
        spend limit is not a conversation outcome: `BudgetExceededError` propagates, as does
        any tool exception other than invalid input or a `ProviderError`.
        """
        started = time.monotonic()
        tally = _RunTally()

        def finish(answer: str, status: str, error: str | None = None) -> Conversation:
            return Conversation(
                answer,
                status,
                error,
                tally.input_tokens,
                tally.output_tokens,
                tuple(tally.tool_calls),
                tally.turns,
                time.monotonic() - started,
            )

        async with self._async_client_factory() as client:
            try:
                output = await self._agent_run(
                    client,
                    operation=operation,
                    effort=effort,
                    instructions=instructions,
                    text=prompt,
                    tools=_canonical_tools(tools),
                    max_turns=max_turns,
                    output_type=None,
                    tally=tally,
                )
            except MaxTurnsExceeded:
                return finish("", "max_turns")
            except _IncompleteResponse as exc:
                return finish(exc.partial_text, exc.status)
            except BudgetExceededError:
                raise
            except (ProviderError, AgentsException) as exc:
                return finish("", "provider_error", str(exc))
        answer = str(output)
        return finish(answer, "completed" if answer.strip() else "empty_answer")

    async def _agent_run(
        self,
        client: openai.AsyncOpenAI,
        *,
        operation: str,
        effort: Effort,
        instructions: str,
        text: str,
        tools: Sequence[Tool],
        max_turns: int,
        output_type: type[BaseModel] | None,
        tally: _RunTally,
    ) -> object:
        label = f"{operation} on {self._model}"

        async def turn(create_kwargs: dict[str, Any]) -> Response:
            tally.turns += 1
            response = await retry_transient_async(
                partial(client.responses.create, **create_kwargs),
                check_budget=self.meter.check_budget,
                request_timeout=self._request_timeout,
                operation_deadline=self._operation_deadline,
                retry_delays=self._retry_delays,
            )
            if not isinstance(response, Response):
                raise ProviderError(f"{label} returned a body that is not a response")
            input_tokens, output_tokens = self._record_usage(
                operation, self._model, label, response.usage
            )
            tally.input_tokens += input_tokens
            tally.output_tokens += output_tokens
            if response.status != "completed":
                raise _IncompleteResponse(
                    f"{label} returned status {response.status!r}",
                    str(response.status),
                    response.output_text,
                )
            return response

        agent = Agent(
            name=operation,
            instructions=instructions,
            model=_GatewayResponsesModel(self._model, client, turn, label),
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=effort), max_tokens=AGENT_MAX_OUTPUT_TOKENS
            ),
            tools=[_sdk_tool(tool, tally) for tool in tools],
            output_type=output_type,
        )
        try:
            result = await Runner.run(
                agent,
                text,
                max_turns=max_turns,
                run_config=RunConfig(
                    tracing_disabled=not self._trace,
                    tracing=self._tracing,
                    workflow_name=operation,
                    # A hallucinated tool name is a model mistake it can recover from.
                    tool_not_found_behavior="return_error_to_model",
                ),
            )
        except AgentsException as exc:
            if isinstance(exc.__cause__, _ToolFailure):
                raise exc.__cause__.original from None
            raise
        return result.final_output

    # Shared plumbing.

    def _cached_model[T: BaseModel](self, cache_key: str, schema: type[T]) -> T | None:
        cached = self._cache.get(cache_key) if self._cache else None
        return None if cached is None else _decode_model(schema, cached)

    def _store_model[T: BaseModel](self, cache_key: str, result: T) -> T:
        """Cache a validated result; the first result stored under a key is the one returned."""
        if not self._cache:
            return result
        schema = type(result)
        return self._cache.store(
            cache_key, result.model_dump_json(), partial(_decode_model, schema)
        )

    def _record_usage(
        self, operation: str, model: str, label: str, usage: object
    ) -> tuple[int, int]:
        """Meter one Responses or Decisions usage object; returns input and output tokens."""
        fields = _usage_fields(label, usage)
        # Usage without cached-token details counts no cached input, so the estimate errs high.
        details = _usage_fields(label, fields.get("input_tokens_details") or {})
        cached = 0
        if details.get("cached_tokens") is not None:
            cached = _token_count(label, details, "cached_tokens")
        input_tokens = _token_count(label, fields, "input_tokens")
        output_tokens = _token_count(label, fields, "output_tokens")
        if cached > input_tokens:
            # Metered as is, it would make a negative cost and lower the spend estimate.
            raise ProviderError(
                f"{label} returned malformed usage: {cached} cached of {input_tokens} input tokens"
            )
        self.meter.record(
            operation,
            model,
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
        )
        return input_tokens, output_tokens

    def _with_retry[R](self, call: Callable[..., R]) -> R:
        return retry_transient(
            call,
            check_budget=self.meter.check_budget,
            request_timeout=self._request_timeout,
            operation_deadline=self._operation_deadline,
            retry_delays=self._retry_delays,
        )


def _decode_model[T: BaseModel](schema: type[T], entry: str) -> T | None:
    """A cached result, or None for an entry that no longer validates (a cache miss)."""
    try:
        return schema.model_validate_json(entry)
    except ValidationError:
        return None


def _usage_fields(label: str, usage: object) -> dict[str, Any]:
    if isinstance(usage, BaseModel):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        raise ProviderError(f"{label} returned a response without usage")
    return usage


def _token_count(label: str, usage: Mapping[str, Any], field: str) -> int:
    value = usage.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderError(f"{label} returned malformed usage: {field} is {value!r}")
    return value


# Strict: a JSON boolean is not a probability.
_Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)]
_PROBABILITY: TypeAdapter[float] = TypeAdapter(_Probability)
# A refused question is stored as null.
_JUDGE_ANSWERS: TypeAdapter[list[float | None]] = TypeAdapter(list[_Probability | None])
