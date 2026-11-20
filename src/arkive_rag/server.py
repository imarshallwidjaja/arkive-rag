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
    GrepMatch,
    NotFoundError,
    SearchHit,
    SectionReadView,
    TopicGuide,
    TopicPage,
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
Arkive serves curated reference collections ("topics").
1. Call list_topics when the topic slug or its facets are unknown. Facets filter a topic
   (client, project, author and the like): list_documents, search, grep and prepare_draft
   accept `filters`, a mapping from facet name to one value or alias; facets combine with AND.
2. Answering questions: prefer targeted search or grep (default limits) over broad listings,
   read_section for context, get_topic_guide for scope and curator notes, lookup_glossary for
   unclear terms, and view_page when a chart, map or layout matters. Cite doc_id, section_id
   and pages. Check whether a proposal or draft differs from a delivered report or final
   memorandum: treat the delivered final deliverable as authoritative while noting provisional
   proposal or earlier-draft values when relevant. When search and grep find no evidence for a
   requested fact or mechanism, say that the corpus does not establish it and stop searching
   rather than inferring an answer from adjacent workflows.
3. Drafting a deliverable: call prepare_draft with the user's full project_inputs, never
   summarised, and follow the drafting_guidance, exemplars, lessons and curator_notes it
   returns. Use read_section for an outline entry it did not include.
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


def _structured(result: BaseModel | Sequence[BaseModel]) -> CallToolResult:
    """A structured tool result whose text block is `response_text`, the compact JSON the
    snapshot measures its results by, so a result that fits its budget reaches the client
    whole. A list result is structured as `{"result": [...]}`, as the SDK wraps one."""
    if isinstance(result, BaseModel):
        structured = result.model_dump(mode="json")
    else:
        structured = {"result": [item.model_dump(mode="json") for item in result]}
    return CallToolResult(
        content=[TextContent(type="text", text=response_text(result))],
        structured_content=structured,
    )


@contextmanager
def _as_tool_errors() -> Iterator[None]:
    try:
        yield
    except ArkiveError as exc:
        raise ToolError(str(exc)) from exc


def create_mcp_server(catalog: TopicCatalog, embed_query: QueryEmbedder | None = None) -> MCPServer:
    server = MCPServer(name="arkive-rag", instructions=INSTRUCTIONS, version=__version__)

    @server.tool()
    def list_topics(offset: int = 0) -> Annotated[CallToolResult, TopicPage]:
        """List the topics this server serves, with their facets, allowed facet values and
        deliverable types (the first 50 of each; omitted_* counts the rest). Pass next_offset
        as offset for the next page of topics."""
        with _as_tool_errors():
            return _structured(topic_page(catalog.snapshots, offset))

    @server.tool()
    def get_topic_guide(topic: str) -> Annotated[CallToolResult, TopicGuide]:
        """Cited overview of a topic: its scope, curator notes, facet definitions, glossary
        terms and its deliverable types with document and feedback counts. Notes, facet values
        and terms that do not fit are counted in its omitted_* fields."""
        with _as_tool_errors():
            return _structured(catalog.get(topic).guide())

    @server.tool()
    def prepare_draft(
        topic: str,
        project_inputs: str,
        deliverable_type: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> Annotated[CallToolResult, DraftPacket]:
        """Everything needed to draft one new deliverable from the user's inputs: up to three
        prior exemplars ranked against them, each with its complete outline and as much whole
        section text as fits, other candidates, reviewer lessons, curator notes and drafting
        guidance, with provenance (dates, version families, recurrence counts). Pass the
        user's full inputs as project_inputs. deliverable_type is optional and matched
        ignoring case; an unknown type lists the valid ones. Works without OPENAI_API_KEY."""
        with _as_tool_errors():
            return _structured(
                catalog.get(topic).prepare_draft(
                    project_inputs, embed_query, deliverable_type=deliverable_type, filters=filters
                )
            )

    @server.tool()
    def list_documents(
        topic: str, filters: dict[str, str] | None = None, limit: int = 50, offset: int = 0
    ) -> Annotated[CallToolResult, DocumentPage]:
        """One page of a topic's documents in path order, with their deliverable type,
        feedback role and version family. `filters` maps facet names to one value or alias
        each; several facets combine with AND. Pass next_offset as offset for the next page."""
        with _as_tool_errors():
            return _structured(
                catalog.get(topic).list_documents_page(filters=filters, limit=limit, offset=offset)
            )

    @server.tool()
    def get_document_outline(
        topic: str, doc_id: str, offset: int = 0
    ) -> Annotated[CallToolResult, DocumentOutline]:
        """Document metadata and one page of its section outline with page ranges. Pass
        next_offset as offset for the next page of a long outline."""
        with _as_tool_errors():
            return _structured(catalog.get(topic).outline(doc_id, offset=offset))

    @server.tool()
    def search(
        topic: str, query: str, filters: dict[str, str] | None = None, top_k: int = 5
    ) -> Annotated[CallToolResult, list[SearchHit]]:
        """Hybrid keyword and semantic search over chunks, optionally restricted by facet
        `filters`, with at most three chunks per version family unless fewer hits remain.
        Needs OPENAI_API_KEY on the server."""
        with _as_tool_errors():
            snapshot = catalog.get(topic)
            if embed_query is None:
                raise ToolError(
                    "semantic search needs OPENAI_API_KEY on the server. grep, list_documents,"
                    " get_document_outline, read_section, view_page, lookup_glossary and"
                    " prepare_draft work without it."
                )
            return _structured(snapshot.search(query, embed_query, filters=filters, top_k=top_k))

    @server.tool()
    def grep(
        topic: str, pattern: str, filters: dict[str, str] | None = None, max_matches: int = 12
    ) -> Annotated[CallToolResult, list[GrepMatch]]:
        """Case-insensitive literal search for exact strings, numbers and filenames,
        optionally restricted by facet `filters`; at most 50 matches."""
        with _as_tool_errors():
            return _structured(
                catalog.get(topic).grep(pattern, filters=filters, max_matches=max_matches)
            )

    @server.tool()
    def read_section(
        topic: str, section_id: str, offset: int = 0
    ) -> Annotated[CallToolResult, SectionReadView]:
        """Markdown of one section, with the IDs of its neighbouring sections. A section too
        long for one result comes in parts: pass next_offset as offset for the next part."""
        with _as_tool_errors():
            return _structured(catalog.get(topic).read_section(section_id, offset=offset))

    @server.tool()
    def view_page(topic: str, doc_id: str, page_number: int) -> list[TextContent | ImageContent]:
        """Rendered page image (WebP) with its visual summary and native text; text too long
        for one result is cut and says how much it left out."""
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
        """The topic's glossary, one page from `offset`. Without `query`, each term with
        whether a sense has a definition. With `query`, the full entries whose term or alias
        contains it: each sense has section citations. A null definition means no definition
        was established from its cited evidence; it does not prove the corpus never defines
        the term."""
        with _as_tool_errors():
            return _structured(catalog.get(topic).lookup_glossary(query, offset=offset))

    return server
