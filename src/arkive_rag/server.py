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
    Blueprint,
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
topic slug is unknown; it lists each topic's facets and the deliverable types that have a
blueprint. Facets are ways to filter a topic, such as client, project, author or
campaign; list_documents, search and grep accept `filters`, a mapping from facet name to
one value or alias, and combine several facets with AND. Then follow the mode that
matches the user's request:

1. Answering questions about the corpus:
   - Use search for semantic or keyword queries and grep for exact strings, numbers or
     filenames. get_topic_guide helps when the collection's scope, its curator notes or
     missing material matter. Call lookup_glossary when a term, acronym or abbreviation
     is unclear. Use read_section for full context and view_page only when a chart, map,
     table layout or visual detail matters.
   - Cite every fact from the corpus by doc_id and section_id, with page numbers where
     the source has rendered pages. Markdown sources have no pages.

2. Drafting a new deliverable from the user's own inputs:
   - Call get_blueprint for the deliverable type and follow its outline, conventions
     and reviewer pitfalls.
   - For each section whose layout matters, read one of its cited exemplar sections
     with read_section and mirror its heading levels and generic table column headers.
     Do not reuse project-specific headings or facts. Reuse other wording only from
     blueprint passages.
   - Every fact about the user's project (its client and other parties, people, places,
     dates, quantities, budgets, targets, findings, identifiers, file names) comes only
     from the user's inputs, which outrank every passage. This rule takes precedence
     over the next ones.
   - The blueprint's passages are the collection's own house text; use them without
     commentary about where they come from, and keep their organisation names, method
     references and attributions as written. In a passage of any kind, adapt or leave
     out only the part that belongs to one particular project (its client, site, dates,
     quantities or results), using the user's inputs.
   - Reproduce a standing passage as written, apart from such a part.
   - Include a conditional passage only when the user's inputs establish its condition.
     Take the part that varies from the inputs, and leave out any sentence of it that
     states a result the inputs do not give. When the inputs do not settle the
     condition, leave the passage out, with no note about it in the deliverable.
   - Include a single_exemplar passage only where it applies to the new project and
     the deliverable has a section with the purpose of the section that holds it. Copy
     no heading text from its heading_path.
   - A passage's variants are provenance, not alternatives to combine.
   - Names, references and attributions from the corpus may be reused only inside a
     blueprint passage of any kind.
   - Where the user's inputs enumerate items (files, deliverables, parameters,
     results), list every item in full, never as patterns or "etc.".
   - Write a standalone deliverable: no Arkive doc_id or section_id, and no mention of
     the exemplar or feedback documents.
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
        """Cited overview of a topic: its scope, curator notes, facet definitions, glossary
        terms and a summary of each deliverable blueprint (type, description, number of
        exemplars). get_blueprint returns a blueprint in full."""
        with _as_tool_errors():
            return catalog.get(topic).guide()

    @server.tool()
    def get_blueprint(topic: str, deliverable_type: str) -> Blueprint:
        """The full writing blueprint for one deliverable type: section outline with cited
        sections, conventions, reviewer pitfalls, exemplar and feedback documents, and
        passages of the collection's house text, each exactly as in its source section.
        Every fact about the user's project comes only from the user's inputs, which
        outrank every passage: in any passage, adapt or leave out the part that belongs to
        one particular project. By kind: reproduce a standing passage as written; include
        a conditional passage only when the user's inputs establish its condition, taking
        the varying part from the inputs, and otherwise leave it out with no note in the
        deliverable; include a single_exemplar passage only where it applies to the new
        project and the deliverable has a section of the same purpose as its own, copying
        no heading text from its heading_path. Variants are provenance, not alternatives
        to combine. The type matches ignoring case and surrounding spaces."""
        with _as_tool_errors():
            return catalog.get(topic).blueprint(deliverable_type)

    @server.tool()
    def list_documents(topic: str, filters: dict[str, str] | None = None) -> list[DocumentInfo]:
        """List documents in a topic. `filters` maps facet names to one value or alias each;
        several facets combine with AND. Each document carries `doc_date` as printed and
        `doc_date_iso` (YYYY-MM-DD, or YYYY-MM when only the month is known) with its
        `doc_date_basis`: `printed` when parsed from the printed date, `embedded` when taken
        from the file's metadata because the document prints none; both are null when the
        document has no date or prints one that is not unambiguous."""
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

        When drafting, mirror an exemplar section's heading levels and generic table
        column headers, but not its project-specific headings or facts. Reuse other
        wording, names, references and attributions only inside the blueprint's passages:
        standing ones as written, conditional ones only when the user's inputs establish
        their condition, single_exemplar ones only where they apply to the new project and
        the deliverable has a section of the same purpose. Every fact about the user's
        project comes only from the user's inputs, which outrank every passage. Never cite
        doc_id or section_id in the deliverable.
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
        """The topic's glossary. Without `query`, every term with whether a sense has a
        definition. With `query`, the full entries whose term or alias contains it: each sense has
        section citations. A null definition means no definition was established from its
        cited evidence; it does not prove the corpus never defines the term."""
        with _as_tool_errors():
            return catalog.get(topic).lookup_glossary(query)

    return server
