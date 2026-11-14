"""MCP server over one or more read-only topic snapshots."""

from __future__ import annotations

import base64
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Self

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent

from arkive_rag import __version__
from arkive_rag.models import (
    ArkiveError,
    DocumentInfo,
    DocumentOutline,
    GrepMatch,
    NotFoundError,
    SearchHit,
    SectionReadView,
    TopicGuide,
    TopicSummary,
)
from arkive_rag.snapshot import QueryEmbedder, TopicSnapshot

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

INSTRUCTIONS = """\
Arkive serves curated reference collections ("topics"). Call list_topics first if the
topic slug is unknown, then follow the mode that matches the user's request:

1. Answering questions about the corpus:
   - Use search for semantic/keyword queries and grep for exact strings, numbers or
     filenames (read get_topic_guide when project scope or missing-deliverable status
     matters). Use read_section for full context and view_page only when inspecting a
     chart, map, table layout or visual detail is needed.
   - Cite every fact from the corpus by doc_id, section_id and page numbers.

2. Drafting a new standalone deliverable from current project inputs:
   - Call get_topic_guide and list_documents. Read documents in the feedback stage to
     internalise reviewer pitfalls, then call get_document_outline and read_section on
     the newest completed deliverables of the target stage to learn house structure,
     tool/methodology background, standard tables (expanding all deliverable filenames
     in full), methodological caveats, references, acknowledgements and disclaimers.
   - Output a clean, standalone deliverable grounded strictly in the current project's
     facts: NEVER include internal Arkive doc_id/section_id citations, meta-references
     to exemplar or feedback documents, or any date, recipient name, postal address or
     measurement that does not appear in the current project inputs.
"""


class TopicCatalog:
    """Owns the open snapshots served by one process, keyed by topic slug."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._topics: dict[str, TopicSnapshot] = {}
        try:
            for path in paths:
                snapshot = TopicSnapshot.open(path)
                if snapshot.topic in self._topics:
                    other = self._topics[snapshot.topic].path
                    snapshot.close()
                    raise ArkiveError(
                        f"topic {snapshot.topic!r} is served by both {other} and {path}"
                    )
                self._topics[snapshot.topic] = snapshot
        except BaseException:
            self.close()
            raise

    @property
    def snapshots(self) -> list[TopicSnapshot]:
        return list(self._topics.values())

    def get(self, topic: str) -> TopicSnapshot:
        if topic not in self._topics:
            raise NotFoundError(f"unknown topic {topic!r}; valid topics: {', '.join(self._topics)}")
        return self._topics[topic]

    def close(self) -> None:
        for snapshot in self._topics.values():
            snapshot.close()
        self._topics.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@contextmanager
def _as_tool_errors() -> Iterator[None]:
    try:
        yield
    except ArkiveError as exc:
        raise ToolError(str(exc)) from exc


def create_mcp_server(catalog: TopicCatalog, embed_query: QueryEmbedder | None = None) -> MCPServer:
    server = MCPServer(name="arkive-rag", instructions=INSTRUCTIONS, version=__version__)

    @server.tool()
    def list_topics() -> list[TopicSummary]:
        """List the topics this server serves, with their projects and lifecycle stages."""
        return [snapshot.summary() for snapshot in catalog.snapshots]

    @server.tool()
    def get_topic_guide(topic: str) -> TopicGuide:
        """Cited overview of a topic plus its curator notes. Read this before searching."""
        with _as_tool_errors():
            return catalog.get(topic).guide()

    @server.tool()
    def list_documents(
        topic: str, project: str | None = None, stage: str | None = None
    ) -> list[DocumentInfo]:
        """List documents in a topic, optionally filtered by project folder or lifecycle stage."""
        with _as_tool_errors():
            return catalog.get(topic).list_documents(project=project, stage=stage)

    @server.tool()
    def get_document_outline(topic: str, doc_id: str) -> DocumentOutline:
        """Document metadata and its section outline with page ranges."""
        with _as_tool_errors():
            return catalog.get(topic).outline(doc_id)

    @server.tool()
    def search(
        topic: str,
        query: str,
        project: str | None = None,
        stage: str | None = None,
        top_k: int = 8,
    ) -> list[SearchHit]:
        """Hybrid keyword and semantic search over chunks. Needs OPENAI_API_KEY on the server."""
        with _as_tool_errors():
            snapshot = catalog.get(topic)
            if embed_query is None:
                raise ToolError(
                    "semantic search needs OPENAI_API_KEY on the server. grep, list_documents,"
                    " get_document_outline, read_section and view_page work without it."
                )
            return snapshot.search(query, embed_query, project=project, stage=stage, top_k=top_k)

    @server.tool()
    def grep(
        topic: str,
        pattern: str,
        project: str | None = None,
        stage: str | None = None,
        max_matches: int = 20,
    ) -> list[GrepMatch]:
        """Case-insensitive literal search for exact strings, numbers and filenames."""
        with _as_tool_errors():
            return catalog.get(topic).grep(
                pattern, project=project, stage=stage, max_matches=max_matches
            )

    @server.tool()
    def read_section(topic: str, section_id: str) -> SectionReadView:
        """Full Markdown of one section, with the IDs of its neighbouring sections.

        When drafting a new deliverable, reuse structure, methodology framing, caveats and
        standard boilerplate; never copy another project's dates, recipients, addresses or
        measurements, and never cite internal doc_id/section_id in a standalone deliverable.
        """
        with _as_tool_errors():
            return catalog.get(topic).read_section(section_id)

    @server.tool()
    def view_page(topic: str, doc_id: str, page_number: int) -> list[TextContent | ImageContent]:
        """Rendered page image (WebP) with its native text and visual summary."""
        with _as_tool_errors():
            page = catalog.get(topic).view_page(doc_id, page_number)
        caption = (
            f"{page.doc_title} ({page.doc_id}) page {page.page_number} of {page.page_count}\n\n"
            f"Visual summary:\n{page.visual_summary or '(none)'}\n\n"
            f"Native text:\n{page.native_text}"
        )
        image = base64.b64encode(page.webp_bytes).decode()
        return [
            TextContent(type="text", text=caption),
            ImageContent(type="image", data=image, mime_type="image/webp"),
        ]

    return server
