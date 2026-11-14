from __future__ import annotations

import base64
import json
import os
import sqlite3
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx2
import numpy as np
import openai
import pytest

from arkive_rag.llm import BuildCache, OpenAIGateway, embedding_batches, load_dotenv_if_present
from arkive_rag.models import DocumentMetaExtraction, PageExtraction, ProviderError
from tests.support import FakeOpenAIClient, raw_response

REQUEST = httpx2.Request("POST", "https://api.openai.test/v1/responses")


def gateway_for(client: FakeOpenAIClient, cache: BuildCache | None = None) -> OpenAIGateway:
    return OpenAIGateway(
        client=cast(openai.OpenAI, client),
        cache=cache,
        embedding_dimensions=client.dimensions,
        retry_delays=(0.0, 0.0, 0.0, 0.0),
    )


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


def test_transient_errors_retry_within_five_attempts() -> None:
    client = FakeOpenAIClient(
        [openai.APIConnectionError(request=REQUEST), openai.APITimeoutError(request=REQUEST)] * 2
    )

    meta = gateway_for(client).extract_document_meta("p1/memo.pdf", "p1", "outline")

    assert isinstance(meta, DocumentMetaExtraction)
    assert len(client.parse_calls) == 5


def test_persistent_transient_errors_raise_provider_error() -> None:
    client = FakeOpenAIClient([openai.APIConnectionError(request=REQUEST)] * 5)

    with pytest.raises(ProviderError, match="after 5 attempts"):
        gateway_for(client).extract_document_meta("p1/memo.pdf", "p1", "outline")
    assert len(client.parse_calls) == 5


def test_non_transient_errors_are_not_retried() -> None:
    response = httpx2.Response(400, request=REQUEST)
    client = FakeOpenAIClient([openai.BadRequestError("bad schema", response=response, body=None)])

    with pytest.raises(ProviderError, match="bad schema"):
        gateway_for(client).extract_document_meta("p1/memo.pdf", "p1", "outline")
    assert len(client.parse_calls) == 1


def test_truncated_output_retries_once_with_doubled_token_budget() -> None:
    truncated = raw_response(None, status="incomplete", reason="max_output_tokens")
    client = FakeOpenAIClient([truncated])

    gateway_for(client).transcribe_page("text", b"RIFF", None)

    assert [c["max_output_tokens"] for c in client.parse_calls] == [16384, 32768]


def test_other_incomplete_status_or_invalid_output_raises_without_retry() -> None:
    filtered = raw_response(None, status="incomplete", reason="content_filter")
    invalid = raw_response(None)
    client = FakeOpenAIClient([filtered, invalid])
    gateway = gateway_for(client)

    with pytest.raises(ProviderError, match="status 'incomplete'"):
        gateway.transcribe_page("text", b"RIFF", None)
    with pytest.raises(ProviderError, match="invalid output"):
        gateway.transcribe_page("text", b"RIFF", None)
    assert len(client.parse_calls) == 2


def test_strong_model_and_required_tokens_reach_the_request() -> None:
    client = FakeOpenAIClient()

    gateway_for(client).transcribe_page("330.9", b"RIFF", "prev", ("330.9",), use_strong_model=True)

    (call,) = client.parse_calls
    assert call["model"] == "gpt-5.4"
    assert call["text_format"] is PageExtraction
    text = call["input"][0]["content"][0]["text"]
    assert "330.9" in text.split("dropped these values")[1] and "prev" in text
    assert call["input"][0]["content"][1]["image_url"].startswith("data:image/webp;base64,")


def test_embeddings_batch_order_and_normalize() -> None:
    client = FakeOpenAIClient()
    texts = [f"chunk {i}" for i in range(300)]

    matrix = gateway_for(client).embed_texts(texts)

    assert [len(batch) for batch in client.embedding_calls] == [128, 128, 44]
    assert matrix.shape == (300, 8) and matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert [len(b) for b in embedding_batches(["x" * 40_000] * 5)] == [2, 2, 1]
    assert embedding_batches(["x" * 150_000, "y"]) == [["x" * 150_000], ["y"]]


def test_cache_hit_returns_validated_output_without_requests(tmp_path: Path) -> None:
    client = FakeOpenAIClient()
    with BuildCache(tmp_path / "cache.db") as cache:
        gateway = gateway_for(client, cache)
        first = gateway.transcribe_page("text", b"RIFF-a", None)
        vector = gateway.embed_query("rainfall")
        calls = client.call_count

        assert gateway.transcribe_page("text", b"RIFF-a", None) == first
        assert np.array_equal(gateway.embed_query("rainfall"), vector)
        assert client.call_count == calls
        gateway.transcribe_page("text", b"RIFF-b", None)
        assert client.call_count == calls + 1


def test_retries_honour_retry_after_and_share_one_deadline() -> None:
    def rate_limited(header: str, value: str) -> openai.RateLimitError:
        response = httpx2.Response(429, request=REQUEST, headers={header: value})
        return openai.RateLimitError("slow down", response=response, body=None)

    client = FakeOpenAIClient([rate_limited("retry-after-ms", "1")])
    gateway = OpenAIGateway(
        client=cast(openai.OpenAI, client), request_timeout=30.0, operation_deadline=60.0
    )
    gateway.extract_document_meta("p1/memo.pdf", "p1", "outline")
    timeouts = [call["timeout"] for call in client.parse_calls]
    assert len(timeouts) == 2 and all(0 < t <= 30.0 for t in timeouts)

    short = OpenAIGateway(
        client=cast(openai.OpenAI, client), request_timeout=30.0, operation_deadline=5.0
    )
    client.script = [rate_limited("retry-after", "120")]
    with pytest.raises(ProviderError, match="would pass the 5 s deadline"):
        short.extract_document_meta("p1/other.pdf", "p1", "outline")
    assert len(client.parse_calls) == 3 and client.parse_calls[2]["timeout"] <= 5.0


def test_undecodable_cache_entries_are_misses_that_get_replaced(tmp_path: Path) -> None:
    client = FakeOpenAIClient()
    path = tmp_path / "new-dir" / "cache.db"
    with BuildCache(path) as cache:
        gateway = gateway_for(client, cache)
        page = gateway.transcribe_page("text", b"RIFF", None)
        vector = gateway.embed_query("rainfall")
        with sqlite3.connect(path) as db:
            db.execute("UPDATE llm_cache SET response_json = 'not base64!'")
        db.close()

        assert gateway.transcribe_page("text", b"RIFF", None) == page
        assert np.array_equal(gateway.embed_query("rainfall"), vector)
        assert client.call_count == 4
        gateway.transcribe_page("text", b"RIFF", None)
        gateway.embed_query("rainfall")
        assert client.call_count == 4
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _openai_client(handler: Callable[[httpx2.Request], httpx2.Response]) -> openai.OpenAI:
    """The pinned SDK client with its real request and response handling, served locally."""
    return openai.OpenAI(
        api_key="test-key",
        base_url="https://api.openai.test/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def _response_body(text: str, status: str = "completed", reason: str | None = None) -> Any:
    message = {"type": "output_text", "text": text, "annotations": []}
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.4-mini",
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [message],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def test_real_client_responses_complete_extend_truncation_and_reject_bad_bodies() -> None:
    meta = DocumentMetaExtraction(
        title="Memo",
        client="Contoso Ports",
        stage="memorandum",
        doc_date=None,
        summary="Bathymetry.",
    )
    replies = [
        httpx2.Response(
            200, json=_response_body(meta.model_dump_json()[:20], "incomplete", "max_output_tokens")
        ),
        httpx2.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "slow"}}),
        httpx2.Response(200, json=_response_body(meta.model_dump_json())),
        httpx2.Response(200, text="<html>bad gateway</html>"),
        httpx2.Response(200, json=_response_body('{"title": ')),
    ]
    seen: list[tuple[int, float]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append((body["max_output_tokens"], request.extensions["timeout"]["read"]))
        return replies.pop(0)

    gateway = OpenAIGateway(
        client=_openai_client(handler), request_timeout=30.0, operation_deadline=60.0
    )

    assert gateway.extract_document_meta("p1/memo.pdf", "p1", "outline") == meta
    assert [tokens for tokens, _ in seen] == [16384, 32768, 32768]
    assert all(0 < timeout <= 30.0 for _, timeout in seen)
    with pytest.raises(ProviderError, match="non-JSON body"):
        gateway.extract_document_meta("p1/memo.pdf", "p1", "outline")
    with pytest.raises(ProviderError, match="invalid output"):
        gateway.extract_document_meta("p1/memo.pdf", "p1", "outline")
    assert replies == []


def test_real_client_embeddings_map_rows_by_index_and_reject_bad_indices() -> None:
    directions = np.eye(4, dtype=np.float32)

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["encoding_format"] == "base64"
        rows = [
            {
                "object": "embedding",
                "index": i,
                "embedding": base64.b64encode((directions[i] * (i + 2)).tobytes()).decode(),
            }
            for i in range(len(body["input"]))
        ]
        if body["input"][0] == "duplicate":
            rows[1]["index"] = 0
        usage = {"prompt_tokens": 1, "total_tokens": 1}
        payload = {"object": "list", "data": rows[::-1], "model": body["model"], "usage": usage}
        return httpx2.Response(200, json=payload)

    gateway = OpenAIGateway(client=_openai_client(handler), embedding_dimensions=4)

    assert np.array_equal(gateway.embed_texts(["a", "b", "c"]), directions[:3])
    with pytest.raises(ProviderError, match="indices"):
        gateway.embed_texts(["duplicate", "x"])
