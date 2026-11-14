"""Model gateway: the only code that talks to the model provider, plus its build cache."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, cast

import numpy as np
import openai
from numpy.typing import NDArray
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, ValidationError

from arkive_rag.models import (
    CorpusSynthesisExtraction,
    DocumentMetaExtraction,
    PageExtraction,
    ProviderError,
)

DEFAULT_PAGE_MODEL = "gpt-5.4-mini"
DEFAULT_SYNTHESIS_MODEL = "gpt-5.4"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 128
# About 35k tokens per request, far below the provider's 300k-token batch ceiling.
EMBEDDING_BATCH_CHARS = 100_000
# Waits between the five attempts of one provider operation.
RETRY_DELAYS_SECONDS = (2.0, 5.0, 12.0, 25.0)
BUILD_REQUEST_TIMEOUT_SECONDS = 180.0
BUILD_DEADLINE_SECONDS = 600.0
QUERY_REQUEST_TIMEOUT_SECONDS = 30.0
QUERY_DEADLINE_SECONDS = 60.0
INITIAL_MAX_OUTPUT_TOKENS = 16384
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

PAGE_INSTRUCTIONS = """\
You transcribe one rendered document page into structured Markdown blocks.
The page image is the authority for layout and for the characters a reader sees.
Use the native text layer to verify numbers, units, filenames and identifiers.
Some PDFs encode font glyphs wrongly in the text layer, for example \u019f or =
in place of "ti" (ENVes\u019fgator, Sen=nel-2) or a dropped ligature (aper):
repair those from the image. Never otherwise change the author's wording,
spelling or typos.

Rules:
- Emit blocks in reading order. kind is heading, paragraph, table or callout.
- heading_level is 1-4 for headings (1 = document title or top-level section)
  and 0 for every other kind. Heading markdown is the heading text without
  leading # marks and keeps any visible section number, for example
  "3.3. Wet and Dry Years Analysis".
- Tables are GitHub Markdown tables with a header row and a | --- | separator row.
  Never summarise a table: transcribe every row and cell.
- Set continues_from_previous_page when the first block on this page continues
  a paragraph or table that the previous page ended mid-way. A continued table
  may omit or repeat its header row. Set merges_into_previous_row when the first
  continued table row is the remainder of a row cut by the page break.
- Omit running headers, running footers and page numbers.
- For every chart, map, diagram, photo or figure add a visual: label is its
  caption or a short title; transcript states what it shows, every axis label,
  legend entry, annotation and printed value.
- Copy every number, unit and filename exactly as printed. Never invent content.
"""

META_INSTRUCTIONS = """\
You catalogue one consulting document. From its path, project folder, outline
and opening text give: the document title, the client organisation, the
lifecycle stage (proposal, memorandum, feedback, summary, case_study, guidance
or reference), the document date as printed (null if absent) and a factual
two-to-four sentence summary naming its subject, methods and key results.
"""

SYNTHESIS_INSTRUCTIONS = """\
You write the orientation guide for a knowledge base that AI agents will search.
Using only the catalog and curator notes, describe in Markdown: the projects,
clients and how they relate; what each document type covers; recurring methods,
datasets and results; and caveats or gaps the curator notes record.
Cite every factual claim with the exact section citation tokens from the
catalog, written as [doc_XXX:doc_XXX_sec_YYY]. Use no other citation form.
"""


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
    def key(operation: str, model: str, prompt_text: str, schema: str, input_hash: str) -> str:
        parts = json.dumps([operation, model, prompt_text, schema, input_hash])
        return hashlib.sha256(parts.encode()).hexdigest()

    def get(self, cache_key: str) -> str | None:
        try:
            with self._lock:
                row = self._db.execute(
                    "SELECT response_json FROM llm_cache WHERE cache_key = ?", (cache_key,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise ProviderError(f"build cache read failed: {exc}") from exc
        return None if row is None else str(row[0])

    def put(self, cache_key: str, response_json: str) -> None:
        try:
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)",
                    (cache_key, response_json, datetime.now(UTC).isoformat()),
                )
                self._db.commit()
        except sqlite3.Error as exc:
            raise ProviderError(f"build cache write failed: {exc}") from exc

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


class LanguageGateway(Protocol):
    """Everything the compiler and server need from language and embedding models."""

    @property
    def embedding_model(self) -> str: ...

    @property
    def embedding_dimensions(self) -> int: ...

    def transcribe_page(
        self,
        native_text: str,
        webp_bytes: bytes,
        prev_context: str | None,
        required_tokens: tuple[str, ...] = (),
        *,
        use_strong_model: bool = False,
    ) -> PageExtraction: ...

    def extract_document_meta(
        self, rel_path: str, project_id: str, outline_and_excerpt: str
    ) -> DocumentMetaExtraction: ...

    def synthesize_corpus(
        self,
        topic: str,
        curator_notes_text: str,
        catalog_text: str,
        invalid_citations_feedback: str | None = None,
    ) -> str: ...

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


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
    """Seconds the provider asked us to wait, when it sent a numeric Retry-After header."""
    if not isinstance(exc, openai.APIStatusError):
        return None
    for header, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = exc.response.headers.get(header)
        if value is not None:
            try:
                return max(float(value) / divisor, 0.0)
            except ValueError:
                return None
    return None


class OpenAIGateway:
    """OpenAI-backed gateway. It is the single owner of provider retry policy.

    Each operation gets up to five attempts that share one deadline; every request's
    timeout is capped by the time left before that deadline.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        page_model: str = DEFAULT_PAGE_MODEL,
        synthesis_model: str = DEFAULT_SYNTHESIS_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        cache: BuildCache | None = None,
        client: openai.OpenAI | None = None,
        request_timeout: float = BUILD_REQUEST_TIMEOUT_SECONDS,
        operation_deadline: float = BUILD_DEADLINE_SECONDS,
        retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
    ) -> None:
        if client is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ProviderError("OPENAI_API_KEY is not set; pass it or add it to .env")
            # max_retries=0: the SDK must not retry underneath this gateway's own policy.
            client = openai.OpenAI(api_key=key, max_retries=0, timeout=request_timeout)
        self._client = client
        self._page_model = page_model
        self._synthesis_model = synthesis_model
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

    def transcribe_page(
        self,
        native_text: str,
        webp_bytes: bytes,
        prev_context: str | None,
        required_tokens: tuple[str, ...] = (),
        *,
        use_strong_model: bool = False,
    ) -> PageExtraction:
        text = f"NATIVE TEXT LAYER:\n{native_text or '(none: the page has no text layer)'}\n"
        if prev_context:
            text += f"\nEND OF PREVIOUS PAGE TRANSCRIPT:\n{prev_context}\n"
        if required_tokens:
            text += (
                "\nA previous transcription dropped these values. Each appears on the page;"
                f" transcribe them exactly where they occur: {', '.join(required_tokens)}\n"
            )
        model = self._synthesis_model if use_strong_model else self._page_model
        return self._parse(
            "transcribe_page", model, PAGE_INSTRUCTIONS, text, PageExtraction, webp_bytes
        )

    def extract_document_meta(
        self, rel_path: str, project_id: str, outline_and_excerpt: str
    ) -> DocumentMetaExtraction:
        text = f"PATH: {rel_path}\nPROJECT FOLDER: {project_id}\n\n{outline_and_excerpt}"
        return self._parse(
            "document_meta", self._page_model, META_INSTRUCTIONS, text, DocumentMetaExtraction
        )

    def synthesize_corpus(
        self,
        topic: str,
        curator_notes_text: str,
        catalog_text: str,
        invalid_citations_feedback: str | None = None,
    ) -> str:
        text = f"TOPIC: {topic}\n\nCURATOR NOTES:\n{curator_notes_text}\n\nCATALOG:\n{catalog_text}"
        if invalid_citations_feedback:
            text += f"\n\nYOUR PREVIOUS DRAFT HAD INVALID CITATIONS:\n{invalid_citations_feedback}"
        result = self._parse(
            "synthesize_corpus",
            self._synthesis_model,
            SYNTHESIS_INSTRUCTIONS,
            text,
            CorpusSynthesisExtraction,
        )
        return result.synthesis_markdown

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self._embedding_dimensions), dtype=np.float32)
        return np.vstack([self._embed_batch(batch) for batch in embedding_batches(texts)])

    def embed_query(self, query: str) -> NDArray[np.float32]:
        vector: NDArray[np.float32] = self._embed_batch([query])[0]
        return vector

    def _cached_matrix(self, cache_key: str, rows: int) -> NDArray[np.float32] | None:
        """A cached embedding matrix, or None on a miss or an entry that does not decode."""
        cached = self._cache.get(cache_key) if self._cache else None
        if cached is None:
            return None
        try:
            raw = np.frombuffer(base64.b64decode(cached, validate=True), dtype=np.float32)
            matrix = raw.reshape(rows, self._embedding_dimensions).copy()
        except ValueError:
            return None
        return matrix if bool(np.all(np.isfinite(matrix))) else None

    def _embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        cache_key = BuildCache.key(
            "embed",
            self._embedding_model,
            json.dumps(texts),
            str(self._embedding_dimensions),
            "",
        )
        cached = self._cached_matrix(cache_key, len(texts))
        if cached is not None:
            return cached
        response = self._with_retry(
            partial(
                self._client.embeddings.create,
                model=self._embedding_model,
                input=texts,
                dimensions=self._embedding_dimensions,
            )
        )
        indices = sorted(item.index for item in response.data)
        if indices != list(range(len(texts))):
            raise ProviderError(
                f"embedding response indices {indices} do not cover 0..{len(texts) - 1} once each"
            )
        rows = sorted(response.data, key=lambda item: item.index)
        matrix = np.asarray([row.embedding for row in rows], dtype=np.float32)
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
            self._cache.put(cache_key, base64.b64encode(matrix.tobytes()).decode())
        return matrix

    def _parse[T: BaseModel](
        self,
        operation: str,
        model: str,
        instructions: str,
        prompt_text: str,
        schema: type[T],
        webp_bytes: bytes | None = None,
    ) -> T:
        """Request `schema` output; the cache key and the request come from the same inputs."""
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]
        input_hash = ""
        if webp_bytes is not None:
            image_url = "data:image/webp;base64," + base64.b64encode(webp_bytes).decode()
            content.append({"type": "input_image", "image_url": image_url, "detail": "high"})
            input_hash = hashlib.sha256(webp_bytes).hexdigest()
        schema_text = json.dumps(schema.model_json_schema(), sort_keys=True)
        cache_key = BuildCache.key(
            operation, model, instructions + prompt_text, schema_text, input_hash
        )
        cached = self._cache.get(cache_key) if self._cache else None
        if cached is not None:
            # An entry that no longer validates is a miss; the fresh response replaces it.
            with contextlib.suppress(ValidationError):
                return schema.model_validate_json(cached)
        max_tokens = INITIAL_MAX_OUTPUT_TOKENS
        for _ in range(2):
            # The raw response lets the status be read before the SDK parses the output:
            # output truncated at max_output_tokens is invalid JSON and fails parsing.
            raw = self._with_retry(
                partial(
                    self._client.responses.with_raw_response.parse,
                    model=model,
                    instructions=instructions,
                    input=cast("ResponseInputParam", [{"role": "user", "content": content}]),
                    text_format=schema,
                    max_output_tokens=max_tokens,
                )
            )
            try:
                body = json.loads(raw.text)
            except ValueError as exc:
                raise ProviderError(f"{operation} on {model} returned a non-JSON body") from exc
            if not isinstance(body, dict):
                raise ProviderError(f"{operation} on {model} returned a non-object body")
            reason = (body.get("incomplete_details") or {}).get("reason")
            if body.get("status") == "incomplete" and reason == "max_output_tokens":
                max_tokens *= 2
                continue
            if body.get("status") != "completed":
                raise ProviderError(
                    f"{operation} on {model} returned status {body.get('status')!r}"
                )
            try:
                parsed = raw.parse().output_parsed
            except (ValueError, openai.OpenAIError) as exc:
                raise ProviderError(
                    f"{operation} on {model} returned invalid output: {exc}"
                ) from exc
            if parsed is None:
                raise ProviderError(f"{operation} on {model} returned no parsed output")
            if self._cache:
                self._cache.put(cache_key, parsed.model_dump_json())
            return parsed
        raise ProviderError(f"{operation} on {model} exceeded {max_tokens // 2} output tokens")

    def _with_retry[R](self, call: Callable[..., R]) -> R:
        return retry_transient(
            call,
            request_timeout=self._request_timeout,
            operation_deadline=self._operation_deadline,
            retry_delays=self._retry_delays,
        )


def retry_transient[R](
    call: Callable[..., R],
    *,
    request_timeout: float,
    operation_deadline: float,
    retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
) -> R:
    """Run `call(timeout=...)`, retrying transient failures until one shared deadline.

    The client behind `call` must be built with `max_retries=0` so retries never stack.
    Every failure, transient or not, surfaces as `ProviderError`.
    """
    deadline = time.monotonic() + operation_deadline
    attempt = 0
    while True:
        attempt += 1
        remaining = deadline - time.monotonic()
        try:
            return call(timeout=min(request_timeout, remaining))
        except _TRANSIENT_ERRORS as exc:
            if attempt > len(retry_delays):
                raise ProviderError(
                    f"provider unavailable after {attempt} attempts: {exc}"
                ) from exc
            delay = _retry_after(exc)
            if delay is None:
                delay = retry_delays[attempt - 1]
            if time.monotonic() + delay >= deadline:
                raise ProviderError(
                    f"provider unavailable after {attempt} attempts; waiting {delay:g} s"
                    f" would pass the {operation_deadline:g} s deadline: {exc}"
                ) from exc
            time.sleep(delay)
        except (openai.OpenAIError, ValidationError) as exc:
            raise ProviderError(f"provider request failed: {exc}") from exc
