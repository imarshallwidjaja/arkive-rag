"""Agent runs over raw files or Arkive, and layer 2: grounded Q&A with a judge.

Every model call goes through `OpenAIGateway`: agent conversations through `converse`, judges
through `run`. The gateway owns retries, metering and the spend limit.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import anyio
import numpy as np
from mcp import Client
from mcp.types import ImageContent, TextContent
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from arkive_rag.compiler import (
    convert_office_to_pdf,
    discover_corpus,
    read_source,
    render_pdf,
)
from arkive_rag.llm import (
    BudgetExceededError,
    Conversation,
    Effort,
    OpenAIGateway,
    Tool,
    ToolImage,
    ToolReply,
    ToolResult,
)
from arkive_rag.models import ProviderError
from arkive_rag.server import TopicCatalog, create_mcp_server
from arkive_rag.snapshot import QueryEmbedder, TopicSnapshot
from evals.schema import (
    QACase,
    satisfies_groups,
)

EVAL_EFFORT: Effort = "medium"
QA_MAX_TURNS = 12
MAX_RAW_PAGES_PER_READ = 10
MAX_GREP_MATCHES = 30


def _stops_evaluation(failure: Exception) -> bool:
    """The spend limit and failures other than `ProviderError` end the evaluation, not a run."""
    return isinstance(failure, BudgetExceededError) or not isinstance(failure, ProviderError)


@dataclass
class RunEmbeddings:
    """The query embeddings of one agent run: how many returned a vector, and one failure, the
    first that stops the evaluation, else the first `ProviderError`."""

    calls: int = 0
    failure: Exception | None = None


class CountingEmbedder:
    """The query embedder behind every agent tool, recording the embeddings of the current run.

    The MCP server turns any exception in its `search` tool into a tool error for the model, so
    `run_agent` reads a failure from the run's `RunEmbeddings` instead. A record exists only
    inside `recording_run`, so nothing one run recorded can reach the next; an embedding made
    outside any run, such as a layer 1 query, is passed through unrecorded.
    """

    def __init__(self, embed_query: QueryEmbedder) -> None:
        self._embed_query = embed_query
        self._run: RunEmbeddings | None = None

    def __call__(self, query: str) -> NDArray[np.float32]:
        record = self._run
        try:
            vector = self._embed_query(query)
        except Exception as exc:
            if record is not None:
                kept = record.failure
                if kept is None or (_stops_evaluation(exc) and not _stops_evaluation(kept)):
                    record.failure = exc
            raise
        if record is not None:
            record.calls += 1
        return vector

    @contextmanager
    def recording_run(self) -> Iterator[RunEmbeddings]:
        """Record the embeddings made until the block exits; agent runs are sequential."""
        if self._run is not None:
            raise RuntimeError("an agent run is already recording query embeddings")
        record = self._run = RunEmbeddings()
        try:
            yield record
        finally:
            self._run = None


@dataclass(frozen=True)
class AgentRun(Conversation):
    """One agent conversation and the query embeddings its tools made."""

    search_embed_calls: int


async def run_agent(
    gateway: OpenAIGateway,
    *,
    operation: str,
    instructions: str,
    prompt: str,
    tools: Sequence[Tool],
    max_turns: int,
    embedder: CountingEmbedder,
) -> AgentRun:
    """One free-text conversation, with the query embeddings its tools made.

    A query embedding that failed with a `ProviderError` ends the run as "provider_error",
    whatever the model did with the tool error it saw. The spend limit and any other embedder
    failure propagate and stop the evaluation.
    """
    with embedder.recording_run() as embeddings:
        run = await gateway.converse(
            operation=operation,
            instructions=instructions,
            prompt=prompt,
            tools=tools,
            max_turns=max_turns,
            effort=EVAL_EFFORT,
        )
    failure = embeddings.failure
    if failure is not None:
        if _stops_evaluation(failure):
            raise failure
        run = replace(run, status="provider_error", error=f"query embedding failed: {failure}")
    # Shallow: `asdict` would turn the `ToolCall` records into dictionaries.
    fields = {f.name: getattr(run, f.name) for f in dataclass_fields(run)}
    return AgentRun(**fields, search_embed_calls=embeddings.calls)


def object_schema(properties: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required)}


# Raw-file baseline.


def _page_argument(args: dict[str, Any], name: str) -> int:
    """A page-number argument; anything but an integer is invalid input (`ValueError`)."""
    value = args[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer page number, got {value!r}")
    return value


PageImage = Callable[[str, int], bytes | None]


class RawCorpus:
    """The native text layer of every page of some corpus files, and their page images: what
    the raw-file tools search and read. A Markdown file is one page of raw text without an
    image. `page_image(rel_path, page_number)` returns one page's WebP, or None when the page
    has none; `page_texts` and `page_image` must describe the same files and pages."""

    def __init__(self, page_texts: Mapping[str, Sequence[str]], page_image: PageImage) -> None:
        self.files = dict(page_texts)
        self._page_image = page_image

    @classmethod
    def extract(cls, corpus_dir: Path) -> RawCorpus:
        """Render every file `discover_corpus` accepts, the set a snapshot is compiled from:
        LibreOffice converts Word and PowerPoint files and pdfium renders each PDF page."""
        texts: dict[str, list[str]] = {}
        images: dict[tuple[str, int], bytes] = {}
        for source in discover_corpus(corpus_dir):
            data = read_source(source)
            if source.format == "md":
                texts[source.rel_path] = [data.decode("utf-8")]
                continue
            pdf = (
                data
                if source.format == "pdf"
                else convert_office_to_pdf(data, source.path.suffix.lower(), source.rel_path)
            )
            pages = render_pdf(pdf)
            texts[source.rel_path] = [text for text, _ in pages]
            for number, (_, webp) in enumerate(pages, start=1):
                images[source.rel_path, number] = webp
        return cls(texts, lambda rel_path, number: images.get((rel_path, number)))

    @classmethod
    def from_snapshot(
        cls, files: Mapping[str, bytes], snapshot: TopicSnapshot, *, digests: Mapping[str, str]
    ) -> RawCorpus:
        """`files`, the corpus files `snapshot` was compiled from, read without rendering: a
        Markdown file is its own bytes, and any other file is the native text and WebP of the
        pages the build rendered and stored, the same pdfium output `extract` produces. Images
        are read from `snapshot` when a tool asks for one, so `snapshot` must stay open while
        the tools run. `digests` maps each file to the SHA-256 hex digest of its bytes, which
        the caller computed once for its own checks, so no file is hashed twice.

        The files are listed and searched in `snapshot.list_documents()` path order, whatever
        the order of `files`. Refuses (ValueError) `digests` for other files than `files`, a
        set of files other than the snapshot's documents, and a file the snapshot holds with
        other bytes, since its stored pages would not be this file's."""
        documents = snapshot.list_documents()
        if digests.keys() != files.keys():
            raise ValueError("digests must name exactly the files")
        if files.keys() != {d.rel_path for d in documents}:
            missing = sorted({d.rel_path for d in documents} - files.keys())
            extra = sorted(files.keys() - {d.rel_path for d in documents})
            raise ValueError(
                f"{snapshot.path} was not compiled from exactly these files: documents without"
                f" a file {missing}, files without a document {extra}"
            )
        texts: dict[str, list[str]] = {}
        doc_ids: dict[str, str] = {}
        for document in documents:
            rel_path = document.rel_path
            if document.sha256 != digests[rel_path]:
                raise ValueError(
                    f"{snapshot.path} holds no document with the bytes of {rel_path}, so its"
                    " stored pages cannot stand for the file"
                )
            if document.format == "md":
                texts[rel_path] = [files[rel_path].decode("utf-8")]
                continue
            doc_ids[rel_path] = document.doc_id
            texts[rel_path] = [
                snapshot.view_page(document.doc_id, number).native_text
                for number in range(1, document.page_count + 1)
            ]

        def page_image(rel_path: str, number: int) -> bytes | None:
            doc_id = doc_ids.get(rel_path)
            return None if doc_id is None else snapshot.view_page(doc_id, number).webp_bytes

        return cls(texts, page_image)

    def _pages(self, rel_path: str) -> Sequence[str]:
        if rel_path not in self.files:
            raise KeyError(f"unknown file {rel_path!r}; call list_corpus_files")
        return self.files[rel_path]

    def list_corpus_files(self, args: dict[str, Any]) -> str:
        return json.dumps([{"rel_path": r, "pages": len(p)} for r, p in self.files.items()])

    def grep_raw_text(self, args: dict[str, Any]) -> str:
        needle = str(args["pattern"]).casefold()
        matches = [
            {"rel_path": rel, "page": number, "line": line.strip()[:300]}
            for rel, pages in self.files.items()
            for number, text in enumerate(pages, start=1)
            for line in text.splitlines()
            if needle and needle in line.casefold()
        ]
        return json.dumps(matches[:MAX_GREP_MATCHES], ensure_ascii=False)

    def read_raw_pages(self, args: dict[str, Any]) -> str:
        pages = self._pages(str(args["rel_path"]))
        start = max(1, _page_argument(args, "start_page"))
        end = min(len(pages), _page_argument(args, "end_page"), start + MAX_RAW_PAGES_PER_READ - 1)
        return "\n\n".join(f"--- page {n} ---\n{pages[n - 1]}" for n in range(start, end + 1))

    def view_raw_page_image(self, args: dict[str, Any]) -> ToolResult:
        rel, number = str(args["rel_path"]), _page_argument(args, "page_number")
        pages = self._pages(rel)
        image = self._page_image(rel, number) if 1 <= number <= len(pages) else None
        if image is None:
            return f"ERROR: {rel} has no rendered page {number}"
        return ToolReply(f"{rel} page {number}", (ToolImage(image, "image/webp"),))

    @property
    def tools(self) -> list[Tool]:
        return [
            Tool(
                "list_corpus_files",
                "List every corpus file with its page count.",
                object_schema({}),
                self.list_corpus_files,
            ),
            Tool(
                "grep_raw_text",
                "Case-insensitive literal search of the native text layer; returns file, page,"
                " line.",
                object_schema({"pattern": {"type": "string"}}, ["pattern"]),
                self.grep_raw_text,
            ),
            Tool(
                "read_raw_pages",
                f"Native text of pages start_page..end_page (1-based, at most"
                f" {MAX_RAW_PAGES_PER_READ} pages per call).",
                object_schema(
                    {
                        "rel_path": {"type": "string"},
                        "start_page": {"type": "integer"},
                        "end_page": {"type": "integer"},
                    },
                    ["rel_path", "start_page", "end_page"],
                ),
                self.read_raw_pages,
            ),
            Tool(
                "view_raw_page_image",
                "Rendered image of one page, for charts, maps, tables and checkboxes.",
                object_schema(
                    {"rel_path": {"type": "string"}, "page_number": {"type": "integer"}},
                    ["rel_path", "page_number"],
                ),
                self.view_raw_page_image,
            ),
        ]


# Arkive over MCP.


@asynccontextmanager
async def arkive_mcp_tools(
    snapshot_path: Path, embed_query: QueryEmbedder
) -> AsyncIterator[tuple[list[Tool], str]]:
    """Every tool the real Arkive MCP server exposes, called through an MCP client session.

    Also yields the server's initialize instructions, which real MCP hosts append to the
    model's system instructions; pass them through `with_server_instructions`. The tools are
    bound to the session, so conversations using them run on this event loop.
    """
    failure: Exception | None = None
    with TopicCatalog([snapshot_path]) as catalog:
        async with Client(create_mcp_server(catalog, embed_query)) as session:
            listed = await session.list_tools()

            def mcp_tool(name: str, description: str, schema: dict[str, Any]) -> Tool:
                async def invoke(args: dict[str, Any]) -> ToolResult:
                    result = await session.call_tool(name, args)
                    texts = [c.text for c in result.content if isinstance(c, TextContent)]
                    if result.is_error:
                        texts.insert(0, "ERROR:")
                    images = tuple(
                        ToolImage(base64.b64decode(c.data), c.mime_type)
                        for c in result.content
                        if isinstance(c, ImageContent)
                    )
                    text = "\n".join(texts)
                    return ToolReply(text, images) if images else text

                return Tool(
                    name,
                    description,
                    object_schema(schema.get("properties", {}), schema.get("required", [])),
                    invoke,
                )

            tools = [mcp_tool(t.name, t.description or "", t.input_schema) for t in listed.tools]
            try:
                yield tools, session.instructions or ""
            except Exception as exc:
                # Re-raised once the session has closed, so the caller gets it unchanged
                # instead of wrapped in the session's task-group exception group.
                failure = exc
    if failure is not None:
        raise failure


def with_server_instructions(instructions: str, server_instructions: str) -> str:
    """Append MCP server instructions the way MCP hosts do, when the server sent any."""
    if not server_instructions.strip():
        return instructions
    return f"{instructions}\n\nMCP SERVER INSTRUCTIONS:\n{server_instructions.strip()}"


# Layer 2.

QA_INSTRUCTIONS = """\
Answer the question from the document corpus your tools expose. Use the tools to find
evidence; check charts, tables and checkboxes on page images when text is ambiguous.
Cite the file name or doc_id and the page for every fact. If the corpus does not
contain the answer, say so instead of guessing."""

Score5 = Annotated[int, Field(ge=1, le=5)]
Score10 = Annotated[int, Field(ge=1, le=10)]


ClaimStatus = Literal["supported", "contradicted", "unverifiable"]
CLAIM_STATUSES: tuple[ClaimStatus, ...] = ("supported", "contradicted", "unverifiable")


class JudgedClaim(BaseModel):
    claim: str
    status: ClaimStatus


class QAJudgeScore(BaseModel):
    factual_accuracy_1_to_5: Score5
    citation_quality_1_to_5: Score5
    completeness_1_to_5: Score5
    claims: list[JudgedClaim]
    rationale: str

    def claim_counts(self) -> dict[ClaimStatus, int]:
        return {s: sum(c.status == s for c in self.claims) for s in CLAIM_STATUSES}


QA_JUDGE_INSTRUCTIONS = """\
You grade one answer to a question about a document corpus. You do not know which system
wrote it. You get the reference facts, the source anchors (the documents and pages that hold
the facts) and EVIDENCE: the corpus text of those anchors, each block labelled with its file
path, doc_id, section_id and pages. The evidence is a transcript and can itself be wrong;
when the reference facts and the evidence disagree, the reference facts decide.

Score from 1 (poor) to 5 (excellent):
- factual_accuracy_1_to_5: against the reference facts. A fact the evidence supports is not
  an accuracy defect, even when the reference facts do not mention it.
- completeness_1_to_5: how fully the answer gives the reference facts.
- citation_quality_1_to_5: whether the answer cites the anchored documents and pages for the
  facts it states. A citation by file name and page and a citation by doc_id or section_id
  are equally good. When no anchors are listed, judge whether citations are specific and
  plausible.

Then list in claims every material factual claim of the answer, each with its text and a
status: "supported" when the reference facts or the evidence support it; "contradicted"
when the reference facts contradict it, or the evidence does where the reference facts are
silent; "unverifiable" when neither supports nor contradicts it. An unverifiable claim is
not a contradiction."""

QA_EVIDENCE_MAX_CHARS = 48_000


def qa_evidence(snapshot: TopicSnapshot, case: QACase) -> str:
    """The snapshot's Markdown of the sections that overlap each source anchor's pages (the
    whole document for an anchor without pages), each labelled with its document's path,
    `doc_id`, `section_id` and page span; a section that two anchors share appears once.

    Raises `NotFoundError` for an unknown document and `ValueError` for pages outside the
    document, an anchor of a case other than `abstention` that no section overlaps, or evidence
    over `QA_EVIDENCE_MAX_CHARS` characters, which is refused rather than cut.
    """
    if not case.source_anchors:
        return "(no source anchors listed)"
    blocks: dict[str, str] = {}
    for anchor in case.source_anchors:
        document = snapshot.document(anchor.doc_id)
        if anchor.pages is not None and anchor.pages[1] > document.page_count:
            extent = (
                f"has {document.page_count} pages"
                if document.page_count
                else "has no pages; anchor the whole document with pages null"
            )
            raise ValueError(
                f"qa case {case.case_id!r}: source anchor {anchor.doc_id} pages"
                f" {list(anchor.pages)}: {document.rel_path} {extent}"
            )
        overlapping = [
            s
            for s in snapshot.full_outline(anchor.doc_id)
            if anchor.pages is None
            or not (s.page_end < anchor.pages[0] or s.page_start > anchor.pages[1])
        ]
        if not overlapping and case.kind != "abstention":
            pages = "the whole document" if anchor.pages is None else f"pages {list(anchor.pages)}"
            raise ValueError(
                f"qa case {case.case_id!r}: source anchor {anchor.doc_id} {pages}: no section of"
                f" {document.rel_path} overlaps it, so the judge would get no evidence for it"
            )
        for section in overlapping:
            if section.section_id in blocks:
                continue
            markdown = snapshot.section_markdown(section.section_id)
            blocks[section.section_id] = (
                f"--- {document.rel_path} ({document.doc_id}), section {section.section_id},"
                f" pages {section.page_start}-{section.page_end} ---\n{markdown}"
            )
    evidence = "\n\n".join(blocks.values()) or "(no section overlaps the source anchors)"
    if len(evidence) > QA_EVIDENCE_MAX_CHARS:
        raise ValueError(
            f"qa case {case.case_id!r}: the evidence of its source anchors is {len(evidence)}"
            f" characters, over the {QA_EVIDENCE_MAX_CHARS} characters the judge accepts; anchor"
            " fewer pages"
        )
    return evidence


@dataclass(frozen=True)
class QAJudgement:
    """Grade one completed answer against the case's reference facts, source anchors and the
    anchors' evidence (`qa_evidence`)."""

    operation: ClassVar[str] = "judge_qa_answer"
    output: ClassVar[type[QAJudgeScore]] = QAJudgeScore
    effort: ClassVar[Effort] = EVAL_EFFORT
    instructions: ClassVar[str] = QA_JUDGE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    case: QACase
    evidence: str
    answer: str

    def text(self) -> str:
        facts = "\n".join(f"- {fact}" for fact in self.case.ground_truth_facts)
        anchors = (
            "\n".join(
                f"- {a.doc_id}, "
                + ("whole document" if a.pages is None else f"pages {a.pages[0]}-{a.pages[1]}")
                for a in self.case.source_anchors
            )
            or "(none listed)"
        )
        return (
            f"QUESTION:\n{self.case.question}\n\nREFERENCE FACTS:\n{facts}\n\n"
            f"SOURCE ANCHORS:\n{anchors}\n\nEVIDENCE:\n{self.evidence}\n\nANSWER:\n{self.answer}"
        )


async def run_layer2_qa(
    gateway: OpenAIGateway,
    snapshot_path: Path,
    corpus_dir: Path,
    embedder: CountingEmbedder,
    cases: Sequence[QACase],
) -> dict[str, object]:
    """Only completed runs are judged and can cover phrases; other runs keep their status.

    Each case runs once per arm. A judged row carries the judge's claims and `claim_counts`;
    a row that was not judged has `judge` and `claim_counts` None and says why in
    `judge_missing`: the run did not complete, or the judge call failed with a `ProviderError`
    (invalid or refused output, output still truncated after the retry, a provider failure),
    which keeps the row's phrase coverage. The spend limit stops the evaluation.
    """
    # Extracted once before any timed query, so the baseline is not charged LibreOffice
    # start-up on each question.
    started = time.monotonic()
    raw = RawCorpus.extract(corpus_dir)
    raw_prep_seconds = time.monotonic() - started
    rows: list[dict[str, object]] = []
    with TopicSnapshot.open(snapshot_path) as snapshot:
        evidence = {case.case_id: qa_evidence(snapshot, case) for case in cases}
    async with arkive_mcp_tools(snapshot_path, embedder) as (mcp_tools, mcp_notes):
        arms: dict[str, tuple[Sequence[Tool], str]] = {
            "raw_files": (raw.tools, QA_INSTRUCTIONS),
            "arkive_mcp": (mcp_tools, with_server_instructions(QA_INSTRUCTIONS, mcp_notes)),
        }
        for case in cases:
            for arm, (tools, instructions) in arms.items():
                run = await run_agent(
                    gateway,
                    operation=f"qa_{arm}",
                    instructions=instructions,
                    prompt=case.question,
                    tools=tools,
                    max_turns=QA_MAX_TURNS,
                    embedder=embedder,
                )
                completed = run.status == "completed"
                judge: QAJudgeScore | None = None
                judge_missing: str | None = "the run did not complete, so it was not judged"
                if completed:
                    try:
                        judge = await anyio.to_thread.run_sync(
                            gateway.run, QAJudgement(case, evidence[case.case_id], run.answer)
                        )
                    except BudgetExceededError:
                        raise
                    except ProviderError as exc:
                        judge_missing = f"the judge call failed: {exc}"
                    else:
                        judge_missing = None
                rows.append(
                    {
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "arm": arm,
                        "status": run.status,
                        "phrase_coverage": completed
                        and satisfies_groups(run.answer, case.required_groups),
                        "groups_matched": sum(
                            completed and satisfies_groups(run.answer, (g,))
                            for g in case.required_groups
                        ),
                        "groups_total": len(case.required_groups),
                        "run": run,
                        "judge": judge,
                        "judge_missing": judge_missing,
                        "claim_counts": None if judge is None else judge.claim_counts(),
                    }
                )
    return {"raw_prep_seconds": raw_prep_seconds, "rows": rows}
