"""Layers 2 and 3: agents over raw files or Arkive, plus blinded judges.

Every model call goes through `OpenAIGateway`: agent conversations through `converse`, judges
through `run`. The gateway owns retries, metering and the spend limit.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated, Any, ClassVar

import anyio
import numpy as np
from mcp import Client
from mcp.types import ImageContent, TextContent
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from arkive_rag.compiler import (
    compile_corpus,
    convert_office_to_pdf,
    discover_corpus,
    read_source,
    render_pdf,
)
from arkive_rag.llm import (
    BudgetExceededError,
    Conversation,
    Effort,
    LanguageGateway,
    OpenAIGateway,
    Tool,
    ToolImage,
    ToolReply,
    ToolResult,
)
from arkive_rag.models import DocumentInfo, ProviderError
from arkive_rag.server import TopicCatalog, create_mcp_server
from arkive_rag.snapshot import QueryEmbedder, TopicSnapshot, write_snapshot
from evals.schema import HoldoutMemoTask, QACase, RubricCheck, satisfies_groups

EVAL_EFFORT: Effort = "medium"
QA_MAX_TURNS = 12
MEMO_MAX_TURNS = 16
MAX_RAW_PAGES_PER_READ = 10
MAX_GREP_MATCHES = 30
BLIND_SEED = 42


def _stops_evaluation(failure: Exception) -> bool:
    """The spend limit and failures other than `ProviderError` end the evaluation, not a run."""
    return isinstance(failure, BudgetExceededError) or not isinstance(failure, ProviderError)


class CountingEmbedder:
    """The query embedder behind every agent tool, counting the embeddings it returned.

    It also keeps one failure since `take_failure` was last called: the first one that stops
    the evaluation, else the first `ProviderError`. The MCP server turns any exception in its
    `search` tool into a tool error for the model, so `run_agent` reads the failure from here
    instead. Agent runs are sequential, so the change in `calls` and the failure recorded during
    one run belong to that run.
    """

    def __init__(self, embed_query: QueryEmbedder) -> None:
        self._embed_query = embed_query
        self._failure: Exception | None = None
        self.calls = 0

    def __call__(self, query: str) -> NDArray[np.float32]:
        try:
            vector = self._embed_query(query)
        except Exception as exc:
            kept = self._failure
            if kept is None or (_stops_evaluation(exc) and not _stops_evaluation(kept)):
                self._failure = exc
            raise
        self.calls += 1
        return vector

    def take_failure(self) -> Exception | None:
        failure, self._failure = self._failure, None
        return failure


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
    embed_calls_before = embedder.calls
    run = await gateway.converse(
        operation=operation,
        instructions=instructions,
        prompt=prompt,
        tools=tools,
        max_turns=max_turns,
        effort=EVAL_EFFORT,
    )
    failure = embedder.take_failure()
    if failure is not None:
        if _stops_evaluation(failure):
            raise failure
        run = replace(run, status="provider_error", error=f"query embedding failed: {failure}")
    return AgentRun(**asdict(run), search_embed_calls=embedder.calls - embed_calls_before)


def object_schema(properties: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required)}


# Raw-file baseline.


def _page_argument(args: dict[str, Any], name: str) -> int:
    """A page-number argument; anything but an integer is invalid input (`ValueError`)."""
    value = args[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer page number, got {value!r}")
    return value


@dataclass(frozen=True)
class RawPage:
    text: str
    webp: bytes | None


class RawCorpus:
    """Native text and page images of every corpus file, extracted once before any timed query
    so the baseline is not charged LibreOffice start-up on each question. The files are the ones
    `discover_corpus` accepts, the same set a snapshot is compiled from."""

    def __init__(self, corpus_dir: Path) -> None:
        started = time.monotonic()
        self.files: dict[str, list[RawPage]] = {}
        for source in discover_corpus(corpus_dir):
            data = read_source(source)
            if source.format == "md":
                self.files[source.rel_path] = [RawPage(data.decode("utf-8"), None)]
                continue
            pdf = (
                data
                if source.format == "pdf"
                else convert_office_to_pdf(data, source.path.suffix.lower(), source.rel_path)
            )
            self.files[source.rel_path] = [RawPage(text, webp) for text, webp in render_pdf(pdf)]
        self.prep_seconds = time.monotonic() - started

    def _pages(self, rel_path: str) -> list[RawPage]:
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
            for number, page in enumerate(pages, start=1)
            for line in page.text.splitlines()
            if needle and needle in line.casefold()
        ]
        return json.dumps(matches[:MAX_GREP_MATCHES], ensure_ascii=False)

    def read_raw_pages(self, args: dict[str, Any]) -> str:
        pages = self._pages(str(args["rel_path"]))
        start = max(1, _page_argument(args, "start_page"))
        end = min(len(pages), _page_argument(args, "end_page"), start + MAX_RAW_PAGES_PER_READ - 1)
        return "\n\n".join(f"--- page {n} ---\n{pages[n - 1].text}" for n in range(start, end + 1))

    def view_raw_page_image(self, args: dict[str, Any]) -> ToolResult:
        rel, number = str(args["rel_path"]), _page_argument(args, "page_number")
        pages = self._pages(rel)
        image = pages[number - 1].webp if 1 <= number <= len(pages) else None
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


class QAJudgeScore(BaseModel):
    factual_accuracy_1_to_5: Score5
    citation_quality_1_to_5: Score5
    completeness_1_to_5: Score5
    hallucination_free: bool
    rationale: str


QA_JUDGE_INSTRUCTIONS = """\
You grade one answer to a question about a document corpus against reference facts.
You do not know which system wrote it. Score factual accuracy, citation quality and
completeness from 1 (poor) to 5 (excellent). Citation quality measures whether the answer
cites the documents and pages listed under SOURCE ANCHORS for the facts it states; when no
anchors are listed, judge whether citations are specific and plausible. hallucination_free
is false if the answer states any material fact that contradicts or is absent from the
reference facts."""


@dataclass(frozen=True)
class QAJudgement:
    """Grade one completed answer against the case's reference facts and source anchors."""

    operation: ClassVar[str] = "judge_qa_answer"
    output: ClassVar[type[QAJudgeScore]] = QAJudgeScore
    effort: ClassVar[Effort] = EVAL_EFFORT
    instructions: ClassVar[str] = QA_JUDGE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    case: QACase
    answer: str

    def text(self) -> str:
        facts = "\n".join(f"- {fact}" for fact in self.case.ground_truth_facts)
        anchors = "\n".join(f"- {a}" for a in self.case.source_anchors) or "(none listed)"
        return (
            f"QUESTION:\n{self.case.question}\n\nREFERENCE FACTS:\n{facts}\n\n"
            f"SOURCE ANCHORS:\n{anchors}\n\nANSWER:\n{self.answer}"
        )


async def run_layer2_qa(
    gateway: OpenAIGateway,
    snapshot_path: Path,
    corpus_dir: Path,
    embedder: CountingEmbedder,
    cases: Sequence[QACase],
) -> dict[str, object]:
    """Only completed runs are judged and can cover phrases; other runs keep their status."""
    raw = RawCorpus(corpus_dir)
    rows: list[dict[str, object]] = []
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
                judge = (
                    await anyio.to_thread.run_sync(gateway.run, QAJudgement(case, run.answer))
                    if completed
                    else None
                )
                rows.append(
                    {
                        "case_id": case.case_id,
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
                    }
                )
    return {"raw_prep_seconds": raw.prep_seconds, "rows": rows}


# Layer 3.


def check_reference_held_out(task: HoldoutMemoTask, reference: DocumentInfo) -> None:
    """The reference deliverable must be held-out material, so no included path can hold it."""
    if not any(m in reference.project_id or m in reference.rel_path for m in task.held_out_markers):
        raise ValueError(
            f"reference document {reference.doc_id} ({reference.rel_path}, project"
            f" {reference.project_id}) contains no held-out marker {list(task.held_out_markers)}"
        )


def _leak_reasons(
    task: HoldoutMemoTask, reference: DocumentInfo, rel_path: str, project_id: str, sha256: str
) -> list[str]:
    reasons = [
        f"held-out marker {marker!r}"
        for marker in task.held_out_markers
        if marker in rel_path or marker in project_id
    ]
    if project_id == reference.project_id:
        reasons.append(f"reference project {project_id!r}")
    if sha256 == reference.sha256:
        reasons.append("same bytes as the reference document")
    return reasons


def select_holdout_sources(
    corpus_dir: Path, task: HoldoutMemoTask, reference: DocumentInfo
) -> dict[str, bytes]:
    """Read the corpus files under `task.included_paths`, refusing any held-out material.

    Selection goes through `discover_corpus`, so symlinks are refused before anything is read,
    and each file is read once: the bytes checked here are the bytes the holdout is built from.
    """
    sources = discover_corpus(corpus_dir)
    selected: dict[str, bytes] = {}
    for part in task.included_paths:
        matched = [s for s in sources if s.rel_path == part or s.rel_path.startswith(f"{part}/")]
        if not matched:
            raise ValueError(f"included path {part!r} matches no corpus file")
        for source in matched:
            data = read_source(source)
            sha256 = hashlib.sha256(data).hexdigest()
            reasons = _leak_reasons(task, reference, source.rel_path, source.project_id, sha256)
            if reasons:
                raise ValueError(
                    f"included file {source.rel_path} is held-out material: {', '.join(reasons)}"
                )
            selected[source.rel_path] = data
    return selected


def assert_holdout_clean(
    task: HoldoutMemoTask, reference: DocumentInfo, documents: Sequence[DocumentInfo]
) -> None:
    """Fail before any memo arm runs if held-out material reached the holdout snapshot."""
    leaked = {
        d.rel_path: reasons
        for d in documents
        if (reasons := _leak_reasons(task, reference, d.rel_path, d.project_id, d.sha256))
    }
    if leaked:
        raise ValueError(f"holdout snapshot leaks held-out material: {leaked}")


def compile_holdout_snapshot(
    corpus_dir: Path,
    output_path: Path,
    gateway: LanguageGateway,
    task: HoldoutMemoTask,
    reference: DocumentInfo,
) -> None:
    """Build the holdout snapshot and publish it at `output_path` only after its guard passes."""
    sources = select_holdout_sources(corpus_dir, task, reference)
    unverified = output_path.with_name(f".{output_path.name}.unverified")
    with tempfile.TemporaryDirectory(prefix="arkive-holdout-") as scratch:
        holdout_dir = Path(scratch, "corpus")
        for rel_path, data in sources.items():
            (holdout_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (holdout_dir / rel_path).write_bytes(data)
        corpus = compile_corpus(
            holdout_dir, gateway, topic="holdout", title="Corpus without the held-out deliverable"
        )
        try:
            write_snapshot(corpus, unverified)
            with TopicSnapshot.open(unverified) as holdout:
                assert_holdout_clean(task, reference, holdout.list_documents())
            os.replace(unverified, output_path)
        except BaseException:
            unverified.unlink(missing_ok=True)
            raise


def document_markdown(snapshot: TopicSnapshot, doc_id: str) -> str:
    """One document's section Markdown in reading order."""
    sections = snapshot.outline(doc_id).sections
    if not sections:
        raise ValueError(f"{snapshot.path} document {doc_id} has no sections")
    return "\n\n".join(snapshot.read_section(s.section_id).markdown for s in sections)


def blind_assignment(arms: Sequence[str], seed: int = BLIND_SEED) -> dict[str, str]:
    """Map Candidate_N labels to arm names in a seeded random order."""
    shuffled = list(arms)
    random.Random(seed).shuffle(shuffled)
    return {f"Candidate_{i}": arm for i, arm in enumerate(shuffled, start=1)}


def score_memo_rubric(memo: str, rubric: Sequence[RubricCheck]) -> dict[str, bool]:
    return {check.check_id: check.passes(memo) for check in rubric}


class CandidateAssessment(BaseModel):
    candidate: str
    structure_and_house_style: Score10
    technical_rigor: Score10
    limitations_and_review_readiness: Score10
    factual_discipline_no_contamination: Score10
    editing_effort_saved: Score10
    strengths: list[str]
    defects: list[str]


class MemoBlindComparison(BaseModel):
    candidates: list[CandidateAssessment]
    ranking: list[str]


MEMO_JUDGE_INSTRUCTIONS = """\
You review anonymous draft memoranda written from the same run notes, against the
reference memorandum the project team actually delivered. Score each candidate from 1 to 10
on structure_and_house_style, technical_rigor, limitations_and_review_readiness,
factual_discipline_no_contamination (penalise any value that contradicts the run notes or
belongs to another project) and editing_effort_saved (how much of the reference a reviewer
would not need to rewrite). Give concrete strengths and defects, then rank the candidates
best first by their labels."""


@dataclass(frozen=True)
class MemoComparison:
    """Blind side-by-side review of completed drafts against the reference memorandum."""

    operation: ClassVar[str] = "judge_memo_drafts"
    output: ClassVar[type[MemoBlindComparison]] = MemoBlindComparison
    effort: ClassVar[Effort] = EVAL_EFFORT
    instructions: ClassVar[str] = MEMO_JUDGE_INSTRUCTIONS
    image_webp: ClassVar[None] = None
    run_notes: str
    reference_markdown: str
    drafts: tuple[tuple[str, str], ...]

    def text(self) -> str:
        candidates = "\n\n".join(f"===== {label} =====\n{text}" for label, text in self.drafts)
        return (
            f"RUN NOTES:\n{self.run_notes}\n\nREFERENCE MEMORANDUM:\n{self.reference_markdown}"
            f"\n\nCANDIDATES:\n{candidates}"
        )


async def run_layer3_holdout_memo(
    gateway: OpenAIGateway,
    *,
    holdout_gateway: LanguageGateway,
    full_snapshot_path: Path,
    corpus_dir: Path,
    embedder: CountingEmbedder,
    output_dir: Path,
    task: HoldoutMemoTask,
) -> dict[str, object]:
    """Draft the held-out deliverable with three arms; only completed drafts are scored.

    `holdout_gateway` compiles the holdout snapshot; it must embed in the full snapshot's
    embedding space, because `embedder` serves both. The reference and the included files are
    checked before any file is copied or any model is called. The result's `blind_drafts` maps
    Candidate labels to draft text; the caller writes them out.
    """
    with TopicSnapshot.open(full_snapshot_path) as full:
        reference = full.document(task.reference_doc_id)
        check_reference_held_out(task, reference)
        reference_markdown = document_markdown(full, reference.doc_id)
    holdout_path = output_dir / "holdout.arkive.db"
    # A worker thread: compiling may start its own event loop for agent runs, and this
    # coroutine's loop is already running.
    await anyio.to_thread.run_sync(
        compile_holdout_snapshot, corpus_dir, holdout_path, holdout_gateway, task, reference
    )

    prompt = f"RUN NOTES:\n{task.run_notes}"
    runs: dict[str, AgentRun] = {}

    async def run_arm(arm: str, tools: Sequence[Tool], server_instructions: str = "") -> None:
        runs[arm] = await run_agent(
            gateway,
            operation=f"memo_{arm}",
            instructions=with_server_instructions(task.memo_instructions, server_instructions),
            prompt=prompt,
            tools=tools,
            max_turns=MEMO_MAX_TURNS,
            embedder=embedder,
        )

    with TopicSnapshot.open(holdout_path) as holdout:
        documents = holdout.list_documents()

        def search_chunks(args: dict[str, Any]) -> str:
            hits = holdout.search(str(args["query"]), embedder, top_k=5)
            return json.dumps(
                [
                    [h.chunk_id, h.doc_title, f"{h.page_start}-{h.page_end}", h.content]
                    for h in hits
                ],
                ensure_ascii=False,
            )

        chunk_search = Tool(
            "search_chunks",
            "Search prior project documents; returns the top 5 matching text chunks.",
            object_schema({"query": {"type": "string"}}, ["query"]),
            search_chunks,
        )
        await run_arm("arm_a_zero_shot", ())
        await run_arm("arm_b_chunk_only_rag", (chunk_search,))
    async with arkive_mcp_tools(holdout_path, embedder) as (mcp_tools, mcp_notes):
        await run_arm("arm_c_full_arkive_mcp", mcp_tools, mcp_notes)

    completed = [arm for arm, run in runs.items() if run.status == "completed"]
    key = blind_assignment(completed)
    drafts = {label: runs[arm].answer for label, arm in key.items()}
    comparison = None
    if drafts:
        comparison = await anyio.to_thread.run_sync(
            gateway.run,
            MemoComparison(task.run_notes, reference_markdown, tuple(drafts.items())),
        )
    return {
        "holdout_snapshot": str(holdout_path),
        "holdout_documents": [d.rel_path for d in documents],
        "arms": {
            arm: {
                "run": run,
                "rubric": score_memo_rubric(run.answer, task.rubric)
                if run.status == "completed"
                else None,
            }
            for arm, run in runs.items()
        },
        "blind_key": key,
        "blind_drafts": drafts,
        "blind_comparison": comparison,
    }
