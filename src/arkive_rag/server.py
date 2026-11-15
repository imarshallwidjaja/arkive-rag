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
    GlossaryLookup,
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
topic slug is unknown. Each topic has facets (ways to filter it, such as client, project,
author or campaign); list_documents, search and grep accept `filters`, a mapping from facet
name to one value or alias, and combine several facets with AND. Then follow the mode that
matches the user's request:

1. Answering questions about the corpus:
   - Use search for semantic or keyword queries and grep for exact strings, numbers or
     filenames. Read get_topic_guide when scope or missing material matters, and call
     lookup_glossary when a term, acronym or abbreviation is unclear. Use read_section for
     full context and view_page only when a chart, map, table layout or visual detail
     matters.
   - Cite every fact from the corpus by doc_id, section_id and page numbers.

2. Drafting a new deliverable from the user's own inputs:
   - Call get_topic_guide and start from the blueprint for the deliverable type: its
     outline, conventions and reviewer pitfalls. Read the blueprint's feedback documents,
     then call get_document_outline and read_section on its exemplars to learn structure,
     standard tables, caveats, references and disclaimers. Use lookup_glossary for the
     collection's terminology.
   - Output a clean, standalone deliverable grounded strictly in the user's inputs: NEVER
     include internal Arkive doc_id/section_id citations, references to exemplar or
     feedback documents, or any name, date, address or figure that does not appear in the
     user's inputs.
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
        """List the topics this server serves, with their facets and allowed facet values."""
        return [snapshot.summary() for snapshot in catalog.snapshots]

    @server.tool()
    def get_topic_guide(topic: str) -> TopicGuide:
        """Cited overview of a topic, its curator notes, facet definitions, deliverable
        blueprints and glossary terms. Read this before searching or drafting."""
        with _as_tool_errors():
            return catalog.get(topic).guide()

    @server.tool()
    def list_documents(topic: str, filters: dict[str, str] | None = None) -> list[DocumentInfo]:
        """List documents in a topic. `filters` maps facet names to one value or alias each;
        several facets combine with AND."""
        with _as_tool_errors():
            return catalog.get(topic).list_documents(filters=filters)

    @server.tool()
    def get_document_outline(topic: str, doc_id: str) -> DocumentOutline:
        """Document metadata and its section outline with page ranges."""
        with _as_tool_errors():
            return catalog.get(topic).outline(doc_id)

    @server.tool()
    def search(
        topic: str, query: str, filters: dict[str, str] | None = None, top_k: int = 8
    ) -> list[SearchHit]:
        """Hybrid keyword and semantic search over chunks, optionally restricted by facet
        `filters`. Needs OPENAI_API_KEY on the server."""
        with _as_tool_errors():
            snapshot = catalog.get(topic)
            if embed_query is None:
                raise ToolError(
                    "semantic search needs OPENAI_API_KEY on the server. grep, list_documents,"
                    " get_document_outline, read_section, view_page and lookup_glossary work"
                    " without it."
                )
            return snapshot.search(query, embed_query, filters=filters, top_k=top_k)

    @server.tool()
    def grep(
        topic: str, pattern: str, filters: dict[str, str] | None = None, max_matches: int = 20
    ) -> list[GrepMatch]:
        """Case-insensitive literal search for exact strings, numbers and filenames,
        optionally restricted by facet `filters`."""
        with _as_tool_errors():
            return catalog.get(topic).grep(pattern, filters=filters, max_matches=max_matches)

    @server.tool()
    def read_section(topic: str, section_id: str) -> SectionReadView:
        """Full Markdown of one section, with the IDs of its neighbouring sections.

        When drafting a new deliverable, reuse structure, framing, caveats and standard
        boilerplate; never copy another document's names, dates, addresses or figures, and
        never cite internal doc_id/section_id in a standalone deliverable.
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

    @server.tool()
    def lookup_glossary(topic: str, query: str | None = None) -> GlossaryLookup:
        """The topic's glossary. Without `query`, every term with whether the corpus defines
        it. With `query`, the full entries whose term or alias contains it: each sense has a
        definition and section citations, and a null definition means the corpus uses the
        term there without defining it."""
        with _as_tool_errors():
            return catalog.get(topic).lookup_glossary(query)

    return server
