"""Deterministic test doubles and synthetic source documents."""

from __future__ import annotations

import json
import re
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from numpy.typing import NDArray

from arkive_rag.compiler import compile_corpus, parse_markdown_blocks
from arkive_rag.models import (
    CorpusSynthesisExtraction,
    DocumentMetaExtraction,
    LifecycleStage,
    PageBlock,
    PageExtraction,
)
from arkive_rag.snapshot import write_snapshot

_SECTION_TOKEN = re.compile(r"\[doc_\d+:doc_\d+_sec_\d+\]")


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
    """A `LanguageGateway` that echoes the native text layer as Markdown blocks.

    `page_script` and `synthesis_script` queue scripted responses consumed in order.
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
        self.calls: list[str] = []
        self.transcribe_calls: list[tuple[bool, tuple[str, ...]]] = []
        self.synthesis_feedback: list[str | None] = []

    def transcribe_page(
        self,
        native_text: str,
        webp_bytes: bytes,
        prev_context: str | None,
        required_tokens: tuple[str, ...] = (),
        *,
        use_strong_model: bool = False,
    ) -> PageExtraction:
        assert webp_bytes[:4] == b"RIFF", "pages must be rendered as WebP"
        self.calls.append("transcribe_page")
        self.transcribe_calls.append((use_strong_model, required_tokens))
        if self.page_script:
            return self.page_script.pop(0)
        return parse_markdown_blocks(native_text)

    def extract_document_meta(
        self, rel_path: str, project_id: str, outline_and_excerpt: str
    ) -> DocumentMetaExtraction:
        self.calls.append("extract_document_meta")
        name = rel_path.lower()
        stage: LifecycleStage = (
            "proposal" if "proposal" in name else "memorandum" if "memo" in name else "reference"
        )
        client = "Contoso Ports" if "estuary" in project_id else "Northwind Marine"
        title = Path(rel_path).stem.replace("-", " ").title()
        return DocumentMetaExtraction(
            title=title, client=client, stage=stage, doc_date=None, summary=f"About {title}."
        )

    def synthesize_corpus(
        self,
        topic: str,
        curator_notes_text: str,
        catalog_text: str,
        invalid_citations_feedback: str | None = None,
    ) -> str:
        self.calls.append("synthesize_corpus")
        self.synthesis_feedback.append(invalid_citations_feedback)
        if self.synthesis_script:
            return self.synthesis_script.pop(0)
        citations = _SECTION_TOKEN.findall(catalog_text)[:2]
        return f"# {topic}\n\nThe corpus covers several projects {' '.join(citations)}."

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append("embed_texts")
        rows = [hash_embedding(text, self.embedding_dimensions) for text in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.embedding_dimensions), np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return hash_embedding(query, self.embedding_dimensions)


def raw_response(
    output: object, status: str = "completed", reason: str | None = None
) -> SimpleNamespace:
    """Mimics `LegacyAPIResponse`: JSON `text` and a `parse()` that validates the output.

    `output=None` models output text that is not valid JSON (for example, truncated), so
    `parse()` raises the `ValidationError` the SDK would.
    """
    details = {"reason": reason} if reason else None
    text = json.dumps({"status": status, "incomplete_details": details})

    def parse() -> SimpleNamespace:
        if output is None:
            PageExtraction.model_validate_json('{"blocks": [')
        return SimpleNamespace(output_parsed=output)

    return SimpleNamespace(text=text, parse=parse)


def _default_output(schema: type) -> object:
    if schema is DocumentMetaExtraction:
        return DocumentMetaExtraction(
            title="Memo",
            client="Northwind Marine",
            stage="memorandum",
            doc_date="2025-10",
            summary="Bathymetry memo.",
        )
    if schema is CorpusSynthesisExtraction:
        return CorpusSynthesisExtraction(synthesis_markdown="Overview [doc_001:doc_001_sec_001].")
    paragraph = PageBlock(
        kind="paragraph",
        heading_level=0,
        continues_from_previous_page=False,
        merges_into_previous_row=False,
        markdown="Page text",
    )
    return PageExtraction(blocks=[paragraph], visuals=[])


class FakeOpenAIClient:
    """Stands in for `openai.OpenAI`: records requests and replays scripted outcomes.

    A scripted outcome is a response object to return or an exception to raise; once the
    script is exhausted every request succeeds with a fixed valid output.
    """

    def __init__(self, script: Sequence[object] = (), dimensions: int = 8) -> None:
        self.script = list(script)
        self.dimensions = dimensions
        self.parse_calls: list[dict[str, Any]] = []
        self.embedding_calls: list[list[str]] = []
        self.responses = SimpleNamespace(with_raw_response=SimpleNamespace(parse=self._parse))
        self.embeddings = SimpleNamespace(create=self._embed)

    @property
    def call_count(self) -> int:
        return len(self.parse_calls) + len(self.embedding_calls)

    def _parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        outcome = self.script.pop(0) if self.script else None
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome or raw_response(_default_output(kwargs["text_format"]))

    def _embed(self, *, model: str, input: list[str], dimensions: int, timeout: float) -> Any:
        self.embedding_calls.append(input)
        rows = [
            SimpleNamespace(index=i, embedding=[float(i + 1)] * dimensions)
            for i in reversed(range(len(input)))
        ]
        return SimpleNamespace(data=rows)


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
