"""Deterministic test doubles and synthetic source documents."""

from __future__ import annotations

import base64
import json
import re
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from xml.sax.saxutils import escape

import httpx2
import numpy as np
import openai
from numpy.typing import NDArray
from pydantic import BaseModel

from arkive_rag.compiler import (
    CorpusSynthesis,
    DocumentCatalogue,
    PageTranscription,
    compile_corpus,
    parse_markdown_blocks,
)
from arkive_rag.llm import AgentRequest, OpenAIGateway, StructuredRequest
from arkive_rag.models import (
    CorpusSynthesisExtraction,
    DocumentMetaExtraction,
    LifecycleStage,
    PageExtraction,
)
from arkive_rag.snapshot import write_snapshot

_SECTION_TOKEN = re.compile(r"\[doc_\d+:doc_\d+_sec_\d+\]")
TEST_BASE_URL = "https://api.openai.test/v1"
SCRIPTED_EMBEDDING_DIMENSIONS = 8


def hash_embedding(text: str, dimensions: int) -> NDArray[np.float32]:
    """Bag-of-words vector: texts sharing words have positive cosine similarity."""
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in re.findall(r"\w+", text.lower()):
        vector[zlib.crc32(token.encode()) % dimensions] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return vector / norm


class FakeGateway:
    """A `LanguageGateway` that answers each compiler request type deterministically.

    Transcription echoes the native text layer as Markdown blocks. `page_script` and
    `synthesis_script` queue scripted responses consumed in order. Every request is kept in
    `requests`, and its operation name in `calls`.
    """

    embedding_model = "fake-hash-embedding"
    embedding_dimensions = 64

    def __init__(
        self,
        *,
        page_script: Sequence[PageExtraction] = (),
        synthesis_script: Sequence[str] = (),
    ) -> None:
        self.page_script = list(page_script)
        self.synthesis_script = list(synthesis_script)
        self.requests: list[object] = []
        self.calls: list[str] = []

    @property
    def transcribe_calls(self) -> list[tuple[str, tuple[str, ...]]]:
        return [
            (r.effort, r.missing_values) for r in self.requests if isinstance(r, PageTranscription)
        ]

    @property
    def synthesis_feedback(self) -> list[str | None]:
        return [r.citation_feedback for r in self.requests if isinstance(r, CorpusSynthesis)]

    def describe(self) -> dict[str, str]:
        return {"embedding_model": self.embedding_model}

    def run[T: BaseModel](self, request: StructuredRequest[T]) -> T:
        self.requests.append(request)
        self.calls.append(request.operation)
        return cast(T, self.answer(request))

    def answer(self, request: object) -> BaseModel:
        match request:
            case PageTranscription(native_text=native_text, image_webp=webp):
                assert webp[:4] == b"RIFF", "pages must be rendered as WebP"
                if self.page_script:
                    return self.page_script.pop(0)
                return parse_markdown_blocks(native_text)
            case DocumentCatalogue(rel_path=rel_path, project_id=project_id):
                return self.catalogue(rel_path, project_id)
            case CorpusSynthesis(topic=topic, catalog_text=catalog):
                if self.synthesis_script:
                    markdown = self.synthesis_script.pop(0)
                else:
                    citations = _SECTION_TOKEN.findall(catalog)[:2]
                    markdown = (
                        f"# {topic}\n\nThe corpus covers several projects {' '.join(citations)}."
                    )
                return CorpusSynthesisExtraction(synthesis_markdown=markdown)
        raise AssertionError(f"unexpected request {request!r}")

    def catalogue(self, rel_path: str, project_id: str) -> DocumentMetaExtraction:
        name = rel_path.lower()
        stage: LifecycleStage = (
            "proposal" if "proposal" in name else "memorandum" if "memo" in name else "reference"
        )
        client = "Contoso Ports" if "estuary" in project_id else "Northwind Marine"
        title = Path(rel_path).stem.replace("-", " ").title()
        return DocumentMetaExtraction(
            title=title, client=client, stage=stage, doc_date=None, summary=f"About {title}."
        )

    def run_agent[T: BaseModel](self, request: AgentRequest[T]) -> T:
        raise AssertionError(f"the compiler makes no agent requests: {request.operation}")

    def judge(self, operation: str, evidence: str, predicates: Sequence[str]) -> list[float | None]:
        raise AssertionError(f"the compiler makes no judge requests: {operation}")

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append("embed_texts")
        rows = [hash_embedding(text, self.embedding_dimensions) for text in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.embedding_dimensions), np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return hash_embedding(query, self.embedding_dimensions)


# Scripted OpenAI HTTP API: the pinned SDK clients with their real request and response
# handling, served by a local transport.


def usage(input_tokens: int = 100, output_tokens: int = 10, cached: int = 0) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def response_body(
    output: Sequence[dict[str, Any]],
    status: str = "completed",
    reason: str | None = None,
    tokens: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A Responses API body; `tokens=None` gives the default usage of 100 in / 10 out."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6.1-sol",
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": list(output),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": tokens if tokens is not None else usage(),
    }


def structured_body(output: BaseModel, **kwargs: Any) -> dict[str, Any]:
    return response_body([message(output.model_dump_json())], **kwargs)


def decision_body(answers: Sequence[float | None], input_tokens: int = 50) -> dict[str, Any]:
    return {
        "answers": [
            {"type": "refusal", "name": None}
            if p is None
            else {"type": "predicate", "name": None, "probability": p}
            for p in answers
        ],
        "model": "gpt-6-luna",
        "usage": usage(input_tokens, 1),
    }


def _default_structured(schema_name: str) -> BaseModel:
    if schema_name == "DocumentMetaExtraction":
        return DocumentMetaExtraction(
            title="Memo",
            client="Northwind Marine",
            stage="memorandum",
            doc_date="2025-10",
            summary="Bathymetry memo.",
        )
    if schema_name == "CorpusSynthesisExtraction":
        return CorpusSynthesisExtraction(synthesis_markdown="Overview [doc_001:doc_001_sec_001].")
    raise AssertionError(f"no default response for schema {schema_name!r}; script one")


@dataclass
class Exchange:
    path: str
    body: dict[str, Any]
    read_timeout: float | None


Outcome = dict[str, Any] | httpx2.Response | Exception


class ScriptedOpenAI:
    """Serves `/responses`, `/decisions` and `/embeddings` to real SDK clients from scripts.

    A scripted outcome is a JSON body (status 200), an `httpx2.Response`, or a transport
    exception to raise (`httpx2.ConnectError` reaches the SDK as `APIConnectionError`). When
    a path's script is empty, catalogue and synthesis requests get a valid default, decisions
    answer 0.75 to every question and embeddings return positive vectors.
    """

    def __init__(self) -> None:
        self.scripts: dict[str, list[Outcome]] = {
            "/responses": [],
            "/decisions": [],
            "/embeddings": [],
        }
        self.exchanges: list[Exchange] = []

    def script(self, path: str, *outcomes: Outcome) -> None:
        self.scripts[path] += outcomes

    def requests(self, path: str) -> list[dict[str, Any]]:
        return [e.body for e in self.exchanges if e.path == path]

    @property
    def request_count(self) -> int:
        return len(self.exchanges)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path.removeprefix("/v1")
        body = json.loads(request.content)
        timeout = request.extensions.get("timeout", {}).get("read")
        self.exchanges.append(Exchange(path, body, timeout))
        script = self.scripts[path]
        outcome = script.pop(0) if script else self._default(path, body)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, httpx2.Response):
            return outcome
        return httpx2.Response(200, json=outcome)

    def _default(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if path == "/responses":
            return structured_body(_default_structured(body["text"]["format"]["name"]))
        if path == "/decisions":
            return decision_body([0.75] * len(body["questions"]))
        rows = [
            {
                "object": "embedding",
                "index": i,
                "embedding": base64.b64encode(
                    np.full(SCRIPTED_EMBEDDING_DIMENSIONS, i + 1, dtype=np.float32).tobytes()
                ).decode(),
            }
            for i in reversed(range(len(body["input"])))
        ]
        tokens = {"prompt_tokens": len(body["input"]), "total_tokens": len(body["input"])}
        return {"object": "list", "data": rows, "model": body["model"], "usage": tokens}

    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key="test-key",
            base_url=TEST_BASE_URL,
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self._handle)),
        )

    def async_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key="test-key",
            base_url=TEST_BASE_URL,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle)),
        )

    def gateway(self, **kwargs: Any) -> OpenAIGateway:
        options: dict[str, Any] = {
            "embedding_dimensions": SCRIPTED_EMBEDDING_DIMENSIONS,
            "retry_delays": (0.0, 0.0, 0.0, 0.0),
        }
        return OpenAIGateway(
            client=self.client(), async_client_factory=self.async_client, **(options | kwargs)
        )


def write_pdf(path: Path, pages: Sequence[Sequence[str]]) -> Path:
    """Write a minimal Helvetica PDF with one text line per entry."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    page_ids: list[int] = []
    font_id = 3
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for lines in pages:
        text = "".join(
            f"({line.replace(chr(92), r'\\').replace('(', r'\(').replace(')', r'\)')}) Tj T* "
            for line in lines
        )
        stream = f"BT /F1 11 Tf 14 TL 60 780 Td {text}ET".encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        content_id = len(objects)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents %d 0 R"
            b" /Resources << /Font << /F1 %d 0 R >> >> >>" % (content_id, font_id)
        )
        page_ids.append(len(objects))
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    body = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n%s\nendobj\n" % (number, obj)
    xref = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    body += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    body += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(body))
    return path


def write_docx(path: Path, paragraphs: Sequence[str]) -> Path:
    """Write a minimal WordprocessingML document with one paragraph per entry."""
    body = "".join(f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>" for text in paragraphs)
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns='
            '"http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType='
            '"application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns='
            '"http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{rel}" Target="word/document.xml"/></Relationships>',
        )
        archive.writestr(
            "word/document.xml",
            f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{ns}">'
            f"<w:body>{body}</w:body></w:document>",
        )
    return path


def write_flat_presentation(path: Path, slides: Sequence[Sequence[str]]) -> Path:
    """Write a flat OpenDocument presentation (.fodp) that LibreOffice can turn into .pptx."""
    pages = "".join(
        f'<draw:page draw:name="s{i}"><draw:frame svg:x="1cm" svg:y="1cm" svg:width="24cm"'
        ' svg:height="14cm"><draw:text-box>'
        + "".join(f"<text:p>{escape(line)}</text:p>" for line in lines)
        + "</draw:text-box></draw:frame></draw:page>"
        for i, lines in enumerate(slides, start=1)
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
        ' xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"'
        ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
        ' xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"'
        ' office:version="1.3" office:mimetype="application/vnd.oasis.opendocument.presentation">'
        f"<office:body><office:presentation>{pages}</office:presentation></office:body>"
        "</office:document>",
        encoding="utf-8",
    )
    return path


BATHYMETRY_CORPUS = {
    "p101-harbour-lidar/harbour-memo.md": (
        "# Rainfall\n\nThe chart shows the Nine Year Median = 330.9 mm reference line for"
        " 2016\u20132017. Sentinel-2 imagery was not used; airborne lidar was.\n\n"
        "## Statistics\n\nThe p-value: 0.05 threshold applied. Area covered 59,494 km\u00b2"
        " of the harbour approaches.\n\n| Layer | File |\n| --- | --- |\n| Dry season |"
        " harbour_dry_depth_thresh.tif |\n\n# Bathymetry\n\nSoundings were reduced with"
        " formula C12-SUM(C13:C15) in the workbook.\n"
    ),
    "p102-estuary-multibeam/estuary-proposal.md": (
        "# Scope\n\nEstuary multibeam bathymetry using Sentinel-2 and tide gauge depth.\n\n"
        "# Budget\n\nTotal fee 120,000 AUD.\n"
    ),
    "p103-reef-sonar/AGENTS.md": (
        "# Status\n\nThe reef Sentinel-2 case study has not been written yet.\n"
    ),
}


def compile_snapshot(
    root: Path,
    files: dict[str, str],
    *,
    topic: str = "bathymetry",
    pdfs: dict[str, Sequence[Sequence[str]]] | None = None,
    gateway: FakeGateway | None = None,
) -> Path:
    """Compile Markdown (and optional synthetic PDF) sources into `root/<topic>.arkive.db`."""
    corpus_dir = root / f"{topic}-corpus"
    for rel_path, text in files.items():
        (corpus_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (corpus_dir / rel_path).write_text(text, encoding="utf-8")
    for rel_path, pages in (pdfs or {}).items():
        (corpus_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
        write_pdf(corpus_dir / rel_path, pages)
    corpus = compile_corpus(corpus_dir, gateway or FakeGateway(), topic=topic, title=topic.upper())
    output = root / f"{topic}.arkive.db"
    write_snapshot(corpus, output)
    return output
