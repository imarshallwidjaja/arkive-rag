"""`python -m evals.run --pack <file> ... --out <dir>`: run an eval pack's layers and write
results.json and report.md."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import anyio
import typer
from pydantic import BaseModel

from arkive_rag.compiler import discover_corpus, read_source
from arkive_rag.llm import (
    QUERY_DEADLINE_SECONDS,
    QUERY_REQUEST_TIMEOUT_SECONDS,
    BuildCache,
    OpenAIGateway,
    UsageMeter,
    load_dotenv_if_present,
)
from arkive_rag.models import ArkiveError, corpus_fingerprint
from arkive_rag.snapshot import TopicSnapshot
from evals.ablation import ARMS, RankSummary, run_layer1
from evals.agents import CountingEmbedder, run_layer2_qa, run_layer3_holdout_memo
from evals.schema import EvalPack, RubricCheck

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def write_private_text(path: Path, text: str) -> None:
    """Write `text` to `path` readable by the owner only, also when `path` already exists."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(text)


def _embedder(model: str, dimensions: int, meter: UsageMeter) -> CountingEmbedder:
    """Query embedder in the snapshot's embedding space."""
    gateway = OpenAIGateway(
        embedding_model=model,
        embedding_dimensions=dimensions,
        meter=meter,
        request_timeout=QUERY_REQUEST_TIMEOUT_SECONDS,
        operation_deadline=QUERY_DEADLINE_SECONDS,
    )
    return CountingEmbedder(gateway.embed_query)


def _usage_report(summary: dict[str, Any]) -> list[str]:
    limit = summary["max_cost_usd"]
    lines = [
        "## Usage",
        "",
        f"Estimated cost ${summary['estimated_cost_usd']:.4f} at Standard list prices"
        f" (spend limit: {'none' if limit is None else f'${limit:g}'}); the estimate ignores the"
        " long-context surcharge.",
        "",
        "| Operation | Model | Calls | Input tokens | Cached input | Output tokens | Est. USD |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        cost = "unpriced" if row["estimated_usd"] is None else f"{row['estimated_usd']:.4f}"
        lines.append(
            f"| {row['operation']} | {row['model']} | {row['calls']} | {row['input_tokens']}"
            f" | {row['cached_input_tokens']} | {row['output_tokens']} | {cost} |"
        )
    return [*lines, ""]


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _layer1_report(result: dict[str, Any]) -> list[str]:
    lines = [
        "## Layer 1: retrieval ablation",
        "",
        f"Top-k {result['top_k']}. Completion@k: the top-k prefix covers every required fact"
        " group and expected document of the case.",
        "",
    ]
    header = "| Scope | Arm | Completion@1 | Completion@3 | Completion@5 | CompletionMRR@5 |"
    lines += [header, "| --- | --- | --- | --- | --- | --- |"]
    scopes: list[tuple[str, dict[str, RankSummary]]] = [("overall", result["overall"])]
    scopes += list(result["by_category"].items())
    for scope, arms in scopes:
        for arm in ARMS:
            s = arms[arm]
            lines.append(
                f"| {scope} | {arm} | {_pct(s.completion_at_1)} | {_pct(s.completion_at_3)}"
                f" | {_pct(s.completion_at_5)} | {s.completion_mrr_at_5:.3f} |"
            )
    return [*lines, ""]


def _layer2_report(result: dict[str, Any]) -> list[str]:
    lines = [
        "## Layer 2: grounded QA",
        "",
        f"Raw-file extraction ran once before timing: {result['raw_prep_seconds']:.1f} s.",
        "Only completed runs are judged; phrase coverage and hallucination-free count other"
        " runs as misses, and judge averages cover judged runs only.",
        "",
        "| Arm | Completed | Phrase coverage | Accuracy | Citations | Completeness |"
        " Hallucination-free | Input tokens | Output tokens | Tool calls | Search embeddings |"
        " Seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ("raw_files", "arkive_mcp"):
        rows = [r for r in result["rows"] if r["arm"] == arm]
        judged = [r["judge"] for r in rows if r["judge"] is not None]
        n, k = len(rows), len(judged)

        def mean(field: str, judged: list[Any] = judged, k: int = k) -> str:
            return f"{sum(getattr(j, field) for j in judged) / k:.2f}" if k else "n/a"

        lines.append(
            f"| {arm} | {sum(r['status'] == 'completed' for r in rows)}/{n}"
            f" | {sum(r['phrase_coverage'] for r in rows)}/{n}"
            f" | {mean('factual_accuracy_1_to_5')} | {mean('citation_quality_1_to_5')}"
            f" | {mean('completeness_1_to_5')}"
            f" | {sum(j.hallucination_free for j in judged)}/{n}"
            f" | {sum(r['run'].input_tokens for r in rows)}"
            f" | {sum(r['run'].output_tokens for r in rows)}"
            f" | {sum(r['run'].tool_calls for r in rows)}"
            f" | {sum(r['run'].search_embed_calls for r in rows)}"
            f" | {sum(r['run'].elapsed_seconds for r in rows):.0f} |"
        )
    failed = [r for r in result["rows"] if r["status"] != "completed"]
    if failed:
        lines += ["", "Runs not completed:", ""]
        lines += [f"- {r['case_id']} / {r['arm']}: {r['status']}" for r in failed]
    return [*lines, ""]


def _layer3_report(result: dict[str, Any], rubric: tuple[RubricCheck, ...]) -> list[str]:
    arms: dict[str, Any] = result["arms"]
    lines = ["## Layer 3: holdout memo drafting", ""]
    lines += [f"- {arm}: status {a['run'].status}" for arm, a in arms.items()]
    lines += ["", "| Check | Category | " + " | ".join(arms) + " |"]
    lines.append("| --- | --- |" + " --- |" * len(arms))

    def mark(arm: str, check_id: str) -> str:
        scores = arms[arm]["rubric"]
        if scores is None:
            return "not completed"
        return "pass" if scores[check_id] else "FAIL"

    for check in rubric:
        marks = " | ".join(mark(a, check.check_id) for a in arms)
        lines.append(f"| {check.check_id} | {check.category} | {marks} |")
    totals = " | ".join(
        f"{sum(arms[a]['rubric'].values())}/{len(rubric)}" if arms[a]["rubric"] else "n/a"
        for a in arms
    )
    lines += [f"| total | | {totals} |", ""]
    key: dict[str, str] = result["blind_key"]
    comparison = result["blind_comparison"]
    if comparison is None:
        return [*lines, "No arm completed, so no blind comparison ran.", ""]
    lines += ["Blind judge (1-10) of completed drafts, unblinded with blind_key.json:", ""]
    lines += ["| Candidate | Arm | Structure | Rigor | Limitations | Discipline | Effort saved |"]
    lines += ["| --- | --- | --- | --- | --- | --- | --- |"]
    for c in comparison.candidates:
        lines.append(
            f"| {c.candidate} | {key.get(c.candidate, '?')} | {c.structure_and_house_style}"
            f" | {c.technical_rigor} | {c.limitations_and_review_readiness}"
            f" | {c.factual_discipline_no_contamination} | {c.editing_effort_saved} |"
        )
    ranking = ", ".join(f"{label} ({key.get(label, '?')})" for label in comparison.ranking)
    return [*lines, "", f"Ranking: {ranking}", "", "Blind drafts: `blind-drafts/`", ""]


@app.command()
def main(
    pack: Annotated[Path, typer.Option(help="Eval pack JSON (*.eval.json)")],
    snapshot: Annotated[Path, typer.Option(help="Snapshot compiled from the pack's corpus")],
    out: Annotated[Path, typer.Option(help="Private output directory, apart from every input")],
    corpus: Annotated[
        Path | None, typer.Option(help="Corpus directory; layers 2 and 3 need it")
    ] = None,
    layer: Annotated[Literal["all", "1", "2", "3"], typer.Option()] = "all",
    cache: Annotated[Path | None, typer.Option(help="Build cache reused by the holdout")] = None,
    max_cost: Annotated[
        float | None,
        typer.Option(
            help="Start no provider call once the estimated spend has reached this many USD;"
            " calls in flight can pass it, and the estimate ignores the long-context surcharge"
        ),
    ] = None,
    trace: Annotated[
        bool, typer.Option(help="Send Agents SDK traces of agent runs to the OpenAI dashboard")
    ] = False,
) -> None:
    """Run the eval pack's layers against a snapshot and its corpus (needs OPENAI_API_KEY)."""
    if max_cost is not None and not max_cost > 0:
        raise typer.BadParameter("--max-cost must be positive")
    eval_pack = EvalPack.load(pack)
    layers = _check_inputs(eval_pack, layer, pack, snapshot, corpus, out, cache)
    load_dotenv_if_present(Path.cwd() / ".env")
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    # One meter: the spend limit and the usage report cover every model call of the run.
    meter = UsageMeter(max_cost)
    with TopicSnapshot.open(snapshot) as snap:
        embedding_model, embedding_dimensions = snap.embedding_model, snap.embedding_dimensions
    results: dict[str, Any] = {}
    report = ["# Arkive evaluation report", "", f"Pack: `{eval_pack.name}`", ""]
    report += [f"Snapshot: `{snapshot}`", ""]
    try:
        embed = _embedder(embedding_model, embedding_dimensions, meter)
        gateway = OpenAIGateway(meter=meter, trace=trace)
        if "1" in layers:
            results["layer1"] = run_layer1(snapshot, embed, eval_pack.retrieval_cases)
            report += _layer1_report(results["layer1"])
        if "2" in layers:
            assert corpus is not None
            results["layer2"] = anyio.run(
                run_layer2_qa, gateway, snapshot, corpus, embed, eval_pack.qa_cases
            )
            report += _layer2_report(results["layer2"])
        if "3" in layers:
            task, corpus_dir = eval_pack.holdout_memo, corpus
            assert task is not None and corpus_dir is not None
            build_cache = BuildCache(cache) if cache else None

            async def layer3() -> dict[str, object]:
                return await run_layer3_holdout_memo(
                    gateway,
                    holdout_gateway=OpenAIGateway(
                        cache=build_cache,
                        meter=meter,
                        trace=trace,
                        embedding_model=embedding_model,
                        embedding_dimensions=embedding_dimensions,
                    ),
                    full_snapshot_path=snapshot,
                    corpus_dir=corpus_dir,
                    embedder=embed,
                    output_dir=out,
                    task=task,
                )

            try:
                results["layer3"] = layer3_result = anyio.run(layer3)
            finally:
                if build_cache:
                    build_cache.close()
            blind_dir = out / "blind-drafts"
            blind_dir.mkdir(mode=0o700, exist_ok=True)
            for label, text in cast(dict[str, str], layer3_result["blind_drafts"]).items():
                write_private_text(blind_dir / f"{label}.md", text)
            write_private_text(
                blind_dir / "blind_key.json", json.dumps(layer3_result["blind_key"], indent=2)
            )
            report += _layer3_report(layer3_result, task.rubric)
    except ArkiveError as exc:
        # A stopped evaluation (the spend limit, or a failure outside any agent run) writes
        # no report, but still states what it spent.
        typer.echo(meter.summary_line())
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(meter.summary_line())
    results["usage"] = meter.summary()
    report += _usage_report(results["usage"])
    write_private_text(
        out / "results.json", json.dumps(results, indent=2, default=_jsonable, ensure_ascii=False)
    )
    write_private_text(out / "report.md", "\n".join(report))
    typer.echo(f"wrote {out / 'results.json'} and {out / 'report.md'}")


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _live_corpus_fingerprint(corpus_dir: Path) -> str:
    sources = discover_corpus(corpus_dir)
    return corpus_fingerprint(
        (s.rel_path, hashlib.sha256(read_source(s)).hexdigest()) for s in sources
    )


def _check_inputs(
    pack: EvalPack,
    layer: str,
    pack_path: Path,
    snapshot: Path,
    corpus: Path | None,
    out: Path,
    cache: Path | None,
) -> set[str]:
    """The layers to run: `all` means every layer the pack defines.

    Refuses, before creating any output or calling any model: a layer the pack cannot run, a
    missing corpus, an output or cache path that overlaps an input, a snapshot of another corpus
    than the pack's, and a corpus folder whose files no longer match the snapshot. `main`
    relies on these checks.
    """
    available = {
        name
        for name, present in (
            ("1", bool(pack.retrieval_cases)),
            ("2", bool(pack.qa_cases)),
            ("3", pack.holdout_memo is not None),
        )
        if present
    }
    layers = available if layer == "all" else {layer}
    if not layers or not layers <= available:
        raise typer.BadParameter(f"pack {pack.name!r} defines no cases for layer {layer}")
    needs_corpus = bool(layers & {"2", "3"})
    if needs_corpus and corpus is None:
        raise typer.BadParameter("layers 2 and 3 need --corpus")
    inputs = {"--pack": pack_path, "--snapshot": snapshot}
    if corpus is not None:
        inputs["--corpus"] = corpus
    writable = {"--out": out} | ({"--cache": cache} if cache is not None else {})
    for written, written_path in writable.items():
        for name, input_path in inputs.items():
            if _overlaps(written_path.resolve(), input_path.resolve()):
                raise typer.BadParameter(f"{written} {written_path} overlaps {name} {input_path}")
    with TopicSnapshot.open(snapshot) as snap:
        fingerprint = snap.corpus_fingerprint
    if pack.corpus_fingerprint not in (None, fingerprint):
        raise typer.BadParameter(
            f"snapshot corpus_fingerprint {fingerprint} does not match pack {pack.name!r}"
            f" ({pack.corpus_fingerprint})"
        )
    if needs_corpus and corpus is not None:
        try:
            live = _live_corpus_fingerprint(corpus)
        except ArkiveError as exc:
            raise typer.BadParameter(f"--corpus {corpus}: {exc}") from exc
        if live != fingerprint:
            raise typer.BadParameter(
                f"--corpus {corpus} has corpus_fingerprint {live}, but the snapshot was built"
                f" from {fingerprint}; rebuild the snapshot or restore the corpus"
            )
    return layers


if __name__ == "__main__":
    app()
