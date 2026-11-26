"""`arkive-rag` command line."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

import typer

from arkive_rag.compiler import compile_corpus
from arkive_rag.llm import (
    QUERY_DEADLINE_SECONDS,
    QUERY_REQUEST_TIMEOUT_SECONDS,
    BuildCache,
    OpenAIGateway,
    UsageMeter,
    load_dotenv_if_present,
)
from arkive_rag.models import ArkiveError, VersionScope
from arkive_rag.server import LOOPBACK_HOSTS, TopicCatalog, create_mcp_server
from arkive_rag.snapshot import QueryEmbedder, TopicSnapshot, check_build_paths, write_snapshot

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)

FilterOption = Annotated[
    list[str] | None,
    typer.Option(
        "--filter",
        help="FACET=VALUE; only documents holding this facet value or alias. Repeat for"
        " several facets, which combine with AND.",
    ),
]
TypeOption = Annotated[
    str | None,
    typer.Option("--type", help="Only documents of this deliverable type (case-insensitive)."),
]
VersionsOption = Annotated[
    VersionScope,
    typer.Option(
        help="current leaves out documents superseded by another document in scope; all keeps"
        " every version."
    ),
]


@contextmanager
def _domain_errors() -> Iterator[None]:
    try:
        yield
    except ArkiveError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


def _echo_json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


def _parse_filters(values: list[str] | None) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in values or []:
        facet, separator, value = item.partition("=")
        facet, value = facet.strip(), value.strip()
        if not separator or not facet or not value:
            raise typer.BadParameter(f"{item!r} is not FACET=VALUE", param_hint="--filter")
        if facet in filters:
            raise typer.BadParameter(
                f"facet {facet!r} is given twice; each facet takes one value and facets"
                " combine with AND",
                param_hint="--filter",
            )
        filters[facet] = value
    return filters


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "topic"


@app.command()
def build(
    corpus_dir: Annotated[Path, typer.Argument(help="Directory of documents to compile.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Snapshot file to write.")],
    topic: Annotated[
        str | None, typer.Option(help="Topic slug; default from the directory.")
    ] = None,
    title: Annotated[str | None, typer.Option(help="Human title; default is the topic.")] = None,
    cache: Annotated[Path | None, typer.Option(help="SQLite cache of provider responses.")] = None,
    max_concurrency: Annotated[int, typer.Option(min=1)] = 4,
    max_cost: Annotated[
        float | None,
        typer.Option(
            help="Start no provider call once the estimated spend has reached this many USD;"
            " calls in flight can pass it, and the estimate ignores the long-context surcharge."
        ),
    ] = None,
) -> None:
    """Compile CORPUS_DIR into a snapshot file."""
    slug = topic or _slug(corpus_dir.resolve().name)
    if max_cost is not None and not (math.isfinite(max_cost) and max_cost > 0):
        raise typer.BadParameter("--max-cost must be a positive finite number of USD")
    meter = UsageMeter(max_cost)
    with _domain_errors():
        check_build_paths(corpus_dir, output, *([cache] if cache else []))
        build_cache = BuildCache(cache) if cache else None
        try:
            gateway = OpenAIGateway(cache=build_cache, meter=meter)
            corpus = compile_corpus(
                corpus_dir,
                gateway,
                topic=slug,
                title=title or slug,
                max_concurrency=max_concurrency,
                render_cache=build_cache,
            )
        finally:
            if build_cache:
                build_cache.close()
            # Printed on failure too: a stopped build has still spent what it spent.
            typer.echo(meter.summary_line())
        write_snapshot(corpus, output)
    chunk_count = len(corpus.all_chunks())
    typer.echo(f"build report: {corpus.build_report.model_dump_json()}")
    typer.echo(f"wrote {output}: {len(corpus.documents)} documents, {chunk_count} chunks")


@app.command()
def inspect(snapshot: Path) -> None:
    """Print a JSON summary of a snapshot."""
    with _domain_errors(), TopicSnapshot.open(snapshot) as snap:
        documents = snap.list_documents()
        families: dict[str, list[dict[str, str | None]]] = {}
        for d in documents:
            families.setdefault(d.family_id, []).append(
                {
                    "doc_id": d.doc_id,
                    "version_status": d.version_status,
                    "superseded_by": d.superseded_by,
                }
            )
        _echo_json(
            {
                "topic": snap.topic,
                "title": snap.title,
                "created_at": snap.created_at,
                "corpus_fingerprint": snap.corpus_fingerprint,
                "arkive_version": snap.arkive_version,
                "document_count": len(documents),
                "glossary_terms": len(snap.glossary),
                "embedding_model": snap.embedding_model,
                "embedding_dimensions": snap.embedding_dimensions,
                "build_settings": snap.build_settings,
                "build_report": snap.build_report.model_dump(),
                "facets": [f.model_dump() for f in snap.facets],
                "glossary": [e.model_dump() for e in snap.glossary],
                "deliverable_types": [t.model_dump() for t in snap.type_summaries()],
                "version_families": [
                    {"family_id": family_id, "members": members}
                    for family_id, members in families.items()
                    if len(members) > 1
                ],
                "feedback_lessons": [lesson.model_dump() for lesson in snap.feedback_lessons],
                "recurring_units": snap.recurring_unit_count,
                "documents": [
                    {
                        **d.model_dump(exclude={"summary"}),
                        "sections": len(snap.full_outline(d.doc_id)),
                    }
                    for d in documents
                ],
            }
        )


def _embedder_for(snapshots: list[TopicSnapshot]) -> QueryEmbedder | None:
    """Query embedder matching the snapshots' embedding space, or None without an API key."""
    if not snapshots or not os.environ.get("OPENAI_API_KEY"):
        return None
    spaces = {(s.embedding_model, s.embedding_dimensions) for s in snapshots}
    if len(spaces) != 1:
        raise ArkiveError(
            f"snapshots use different embedding spaces {sorted(spaces)}; serve them separately"
        )
    model, dimensions = spaces.pop()
    gateway = OpenAIGateway(
        embedding_model=model,
        embedding_dimensions=dimensions,
        request_timeout=QUERY_REQUEST_TIMEOUT_SECONDS,
        operation_deadline=QUERY_DEADLINE_SECONDS,
    )
    return gateway.embed_query


@app.command()
def search(
    snapshot: Path,
    query: str,
    filter_values: FilterOption = None,
    deliverable_type: TypeOption = None,
    versions: VersionsOption = "current",
    top_k: int = 8,
) -> None:
    """Hybrid search of a snapshot (needs OPENAI_API_KEY)."""
    filters = _parse_filters(filter_values)
    with _domain_errors(), TopicSnapshot.open(snapshot) as snap:
        embedder = _embedder_for([snap])
        if embedder is None:
            raise ArkiveError("search needs OPENAI_API_KEY; use grep for offline exact matching")
        result = snap.search(
            query,
            embedder,
            filters=filters,
            deliverable_type=deliverable_type,
            versions=versions,
            top_k=top_k,
        )
        _echo_json(result.model_dump())


@app.command()
def grep(
    snapshot: Path,
    pattern: str,
    filter_values: FilterOption = None,
    deliverable_type: TypeOption = None,
    versions: VersionsOption = "current",
    max_matches: int = 20,
) -> None:
    """Literal case-insensitive search of section text (offline)."""
    filters = _parse_filters(filter_values)
    with _domain_errors(), TopicSnapshot.open(snapshot) as snap:
        result = snap.grep(
            pattern,
            filters=filters,
            deliverable_type=deliverable_type,
            versions=versions,
            max_matches=max_matches,
        )
        _echo_json(result.model_dump())


@app.command()
def serve(
    snapshots: Annotated[list[Path], typer.Argument(help="Snapshot files, one per topic.")],
    transport: Literal["stdio", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    allow_nonlocal: Annotated[
        bool, typer.Option(help="Allow streamable-http on a non-loopback host.")
    ] = False,
) -> None:
    """Serve snapshots over MCP."""
    if transport == "streamable-http" and host not in LOOPBACK_HOSTS and not allow_nonlocal:
        typer.echo(
            f"error: --host {host} would expose the unauthenticated MCP server to the network;"
            " pass --allow-nonlocal to do it anyway",
            err=True,
        )
        raise typer.Exit(1)
    with _domain_errors(), TopicCatalog(snapshots) as catalog:
        server = create_mcp_server(catalog, _embedder_for(catalog.snapshots))
        if transport == "stdio":
            server.run("stdio")
        else:
            server.run("streamable-http", host=host, port=port)


def main() -> None:
    load_dotenv_if_present(Path.cwd() / ".env")
    app()
