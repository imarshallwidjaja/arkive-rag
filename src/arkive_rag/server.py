"""MCP server over one or more read-only topic snapshots."""

from __future__ import annotations

import base64
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Annotated, Self

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel

from arkive_rag import __version__
from arkive_rag.models import (
    ArkiveError,
    DocumentOutline,
    DocumentPage,
    DraftPacket,
    GlossaryLookup,
    GrepResult,
    NotFoundError,
    SearchResult,
    SectionReadView,
    TableQueryResult,
    TopicGuide,
    TopicPage,
    VersionScope,
    choices,
    echoed,
)
from arkive_rag.snapshot import (
    QueryEmbedder,
    TopicSnapshot,
    page_caption,
    response_text,
    topic_page,
)

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

INSTRUCTIONS = """\
Arkive serves reference collections ("topics").
1. Call list_topics only when more than one topic is served or a topic's facets are unknown.
   Facets filter list_documents, search, grep, query_tables and prepare_draft (`filters`
   combine with AND); deliverable_type narrows the first four to one document type.
2. Answering questions: start with targeted search or grep, or query_tables for values in
   tables (columns names the headers you need); use read_section only when a hit needs
   surrounding context, get_topic_guide for scope or curator notes, lookup_glossary for
   terms, and view_page for charts, maps or visual layout. Cite doc_id, section_id and pages,
   and answer as soon as the evidence is sufficient. Delivered final reports and memoranda
   outrank provisional proposals or earlier drafts; note both when they differ. Reads cover
   current versions; when excluded_superseded_count > 0 and the user asks about earlier
   drafts or version history, repeat the call with versions="all". When search and grep find
   no evidence, state that the corpus does not establish it and stop.
3. Drafting: call prepare_draft with the user's full project_inputs, follow its
   drafting_guidance, exemplars, lessons and curator_notes, and read_section for omitted
   sections.
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
            raise NotFoundError(
                f"unknown topic {echoed(topic)}; valid topics: {choices(self._topics)}"
            )
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


def _structured(result: BaseModel) -> CallToolResult:
    """A structured tool result whose text block is `response_text`, the compact JSON the
    snapshot measures its results by, so a result that fits its budget reaches the client
    whole."""
    return CallToolResult(
        content=[TextContent(type="text", text=response_text(result))],
        structured_content=result.model_dump(mode="json"),
    )


@contextmanager
def _as_tool_errors() -> Iterator[None]:
    try:
        yield
    except ArkiveError as exc:
        raise ToolError(str(exc)) from exc


def server_instructions(catalog: TopicCatalog) -> str:
    """Server initialize instructions; names the served topic inline when there is only one."""
    if len(catalog.snapshots) != 1:
        return INSTRUCTIONS
    sole = catalog.snapshots[0]
    facets = ", ".join(f.name for f in sole.facets) or "none"
    types = ", ".join(t.deliverable_type for t in sole.type_summaries()[:20]) or "none"
    return (
        f"{INSTRUCTIONS}Served topic: {sole.topic!r} (facets: {facets};"
        f" deliverable_types: {types}).\n"
    )


def create_mcp_server(catalog: TopicCatalog, embed_query: QueryEmbedder | None = None) -> MCPServer:
    server = MCPServer(
        name="arkive-rag", instructions=server_instructions(catalog), version=__version__
    )

    @server.tool()
    def list_topics(offset: int = 0) -> Annotated[CallToolResult, TopicPage]:
        """List served topics, their facets/values and deliverable types (paged by offset)."""
        with _as_tool_errors():
            return _structured(topic_page(catalog.snapshots, offset))

    @server.tool()
    def get_topic_guide(topic: str) -> Annotated[CallToolResult, TopicGuide]:
        """Topic overview, curator notes, facet definitions, glossary terms and type counts."""
        with _as_tool_errors():
            return _structured(catalog.get(topic).guide())

    @server.tool()
    def prepare_draft(
        topic: str,
        project_inputs: str,
        deliverable_type: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> Annotated[CallToolResult, DraftPacket]:
        """Assemble ranked exemplars, outlines, section text, reviewer lessons, curator notes
        and drafting guidance for a new deliverable. Pass the user's full inputs verbatim as
        project_inputs; optional deliverable_type matches ignoring case."""
        with _as_tool_errors():
            return _structured(
                catalog.get(topic).prepare_draft(
                    project_inputs, embed_query, deliverable_type=deliverable_type, filters=filters
                )
            )

    @server.tool()
    def list_documents(
        topic: str,
        filters: dict[str, str] | None = None,
        deliverable_type: str | None = None,
        versions: VersionScope = "current",
        limit: int = 50,
        offset: int = 0,
    ) -> Annotated[CallToolResult, DocumentPage]:
        """Paged documents in path order with type, stated attributes, feedback role and
        version family."""
        with _as_tool_errors():
            return _structured(
                catalog.get(topic).list_documents_page(
                    filters=filters,
                    deliverable_type=deliverable_type,
                    versions=versions,
                    limit=limit,
                    offset=offset,
                )
            )

    @server.tool()
    def get_document_outline(
        topic: str, doc_id: str, offset: int = 0
    ) -> Annotated[CallToolResult, DocumentOutline]:
        """Document metadata and section outline with page ranges (paged by offset)."""
        with _as_tool_errors():
            return _structured(catalog.get(topic).outline(doc_id, offset=offset))

    @server.tool()
    def search(
        topic: str,
        query: str,
        filters: dict[str, str] | None = None,
        deliverable_type: str | None = None,
        versions: VersionScope = "current",
        top_k: int = 5,
    ) -> Annotated[CallToolResult, SearchResult]:
        """Hybrid keyword and semantic chunk search, diversified by version family."""
        with _as_tool_errors():
            snapshot = catalog.get(topic)
            if embed_query is None:
                raise ToolError(
                    "semantic search needs OPENAI_API_KEY on the server. grep, list_documents,"
                    " get_document_outline, read_section, view_page, lookup_glossary,"
                    " query_tables and prepare_draft work without it."
                )
            return _structured(
                snapshot.search(
                    query,
                    embed_query,
                    filters=filters,
                    deliverable_type=deliverable_type,
                    versions=versions,
                    top_k=top_k,
                )
            )

    @server.tool()
    def grep(
        topic: str,
        pattern: str,
        filters: dict[str, str] | None = None,
        deliverable_type: str | None = None,
        versions: VersionScope = "current",
        max_matches: int = 12,
    ) -> Annotated[CallToolResult, GrepResult]:
        """Case-insensitive literal search for exact strings, numbers and filenames (max 50);
        a table row match shows its header row."""
        with _as_tool_errors():
            return _structured(
                catalog.get(topic).grep(
                    pattern,
                    filters=filters,
                    deliverable_type=deliverable_type,
                    versions=versions,
                    max_matches=max_matches,
                )
            )

    @server.tool()
    def query_tables(
        topic: str,
        query: str | None = None,
        columns: list[str] | None = None,
        filters: dict[str, str] | None = None,
        deliverable_type: str | None = None,
        versions: VersionScope = "current",
        limit: int = 20,
        offset: int = 0,
    ) -> Annotated[CallToolResult, TableQueryResult]:
        """Rows of Markdown tables: columns keeps tables whose headers contain every given
        name and shows those columns plus the row labels; query keeps rows matching its words
        (paged by offset)."""
        with _as_tool_errors():
            return _structured(
                catalog.get(topic).query_tables(
                    query=query,
                    columns=columns,
                    filters=filters,
                    deliverable_type=deliverable_type,
                    versions=versions,
                    limit=limit,
                    offset=offset,
                )
            )

    @server.tool()
    def read_section(
        topic: str, section_id: str, offset: int = 0
    ) -> Annotated[CallToolResult, SectionReadView]:
        """Markdown of one section with neighbour IDs (pass next_offset for further parts)."""
        with _as_tool_errors():
            return _structured(catalog.get(topic).read_section(section_id, offset=offset))

    @server.tool()
    def view_page(topic: str, doc_id: str, page_number: int) -> list[TextContent | ImageContent]:
        """Rendered page image (WebP) with visual summary and native text."""
        with _as_tool_errors():
            page = catalog.get(topic).view_page(doc_id, page_number)
        caption = page_caption(page)
        image = base64.b64encode(page.webp_bytes).decode()
        return [
            TextContent(type="text", text=caption),
            ImageContent(type="image", data=image, mime_type="image/webp"),
        ]

    @server.tool()
    def lookup_glossary(
        topic: str, query: str | None = None, offset: int = 0
    ) -> Annotated[CallToolResult, GlossaryLookup]:
        """Glossary terms (no query) or matching entries with section citations (null definition
        means unestablished from cited text, not absent from the corpus)."""
        with _as_tool_errors():
            return _structured(catalog.get(topic).lookup_glossary(query, offset=offset))

    return server
