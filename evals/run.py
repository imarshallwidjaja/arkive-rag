"""`python -m evals.run --pack <file> ... --out <dir>`: run an eval pack's layers and write
results.json and report.md.

Before it creates a gateway, the build cache or any output, the preflight (`_check_inputs`)
checks the output and cache paths:

- `--out` and `--cache` overlap no input and not each other.
- An existing `--out` is a directory, not a symlink, that the evaluator can write and search,
  and neither `results.json` nor `report.md` in it is a directory. An absent `--out` lies below
  no existing non-directory, and its nearest existing ancestor is a directory the evaluator
  can write and search.
- When layer 3 runs, `--out` holds no `holdouts` directory. An absent `--cache` passes the
  test of an absent `--out`; an existing one is a regular file, not a symlink, that the
  evaluator can read and write, in a directory it can write and search (SQLite writes `-wal`
  and `-shm` files beside it).

Every output file is published by replacement (`write_private_text`), so an earlier
`results.json` or `report.md`, whatever its owner or mode, stays as it was until a complete
new file replaces it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import shutil
import stat
import statistics
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import anyio
import typer
from pydantic import BaseModel, ValidationError

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
from arkive_rag.passages import STANDING_RECURRENCE_CONTAINMENT
from arkive_rag.snapshot import SEARCH_FAMILY_CAP, TopicSnapshot
from evals.ablation import ARMS, RankSummary, run_layer1
from evals.agents import AgentRun, CountingEmbedder, qa_evidence, run_layer2_qa
from evals.drafting import (
    ExemplarSelection,
    check_reference_held_out,
    run_layer3_holdout,
    select_holdout_sources,
)
from evals.schema import EvalPack, Holdout

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def write_private_text(path: Path, text: str) -> None:
    """Publish `text` at `path`, readable by the owner only: write it to a private temporary
    file in the same directory and rename that over `path`. A failure leaves whatever was at
    `path` unchanged and removes the temporary file; a symlink at `path` is replaced, never
    written through. An interrupt that arrives after the rename leaves the new file published."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        # mkstemp's mode is subject to the umask; the published file is always 0600.
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def make_private_directory(path: Path) -> None:
    """Create the directory `path`, owner-only, unless it exists. Refuses (ValueError)
    anything at `path` other than a real directory, such as a symlink to one."""
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{path} is not a real directory; refusing to write into it") from None


def refuse_overlapping_paths(written: dict[str, Path], inputs: dict[str, Path]) -> None:
    """Refuse a written path that is, contains or lies inside an input path."""
    for written_name, written_path in written.items():
        for name, input_path in inputs.items():
            if _overlaps(written_path.resolve(), input_path.resolve()):
                raise typer.BadParameter(
                    f"{written_name} {written_path} overlaps {name} {input_path}"
                )


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
        " group and expected document of the case. bm25_only, dense_only and hybrid_rrf rank"
        " raw chunk channels; served_search is the search agents get (`snapshot.search` with"
        f" top_k={result['top_k']}), which leaves out a superseded document whose replacement"
        f" passes the case's filters and takes at most {SEARCH_FAMILY_CAP} chunks of one version"
        " family before filling the list with held-back chunks. Every arm uses the case's one"
        " query embedding.",
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


def _tool_call_report(runs_by_arm: dict[str, list[AgentRun]]) -> list[str]:
    """Per arm and tool name: the calls and the characters of text they returned."""
    lines = ["Tool calls by tool:", "", "| Arm | Tool | Calls | Result characters |"]
    lines.append("| --- | --- | --- | --- |")
    for arm, runs in runs_by_arm.items():
        totals: dict[str, tuple[int, int]] = {}
        for call in (c for run in runs for c in run.tool_calls):
            calls, chars = totals.get(call.name, (0, 0))
            totals[call.name] = (calls + 1, chars + (call.result_chars or 0))
        for name, (calls, chars) in sorted(totals.items()):
            lines.append(f"| {arm} | {name} | {calls} | {chars} |")
        if not totals:
            lines.append(f"| {arm} | (none) | 0 | 0 |")
    return [*lines, ""]


def _qa_scores(rows: list[dict[str, Any]]) -> str:
    """Completed, judged, phrase coverage, judge means and claim counts of some Q&A rows."""
    judged = [r["judge"] for r in rows if r["judge"] is not None]
    n, k = len(rows), len(judged)

    def mean(field: str) -> str:
        return f"{sum(getattr(j, field) for j in judged) / k:.2f}" if k else "n/a"

    def claims(status: str) -> str:
        counts = [r["claim_counts"][status] for r in rows if r["claim_counts"] is not None]
        return str(sum(counts)) if k else "n/a"

    return (
        f"{sum(r['status'] == 'completed' for r in rows)}/{n} | {k}/{n}"
        f" | {sum(r['phrase_coverage'] for r in rows)}/{n}"
        f" | {mean('factual_accuracy_1_to_5')} | {mean('citation_quality_1_to_5')}"
        f" | {mean('completeness_1_to_5')} | {claims('contradicted')} | {claims('unverifiable')}"
    )


def _layer2_report(result: dict[str, Any]) -> list[str]:
    rows: list[dict[str, Any]] = result["rows"]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    lines = [
        "## Layer 2: grounded QA",
        "",
        f"Raw-file extraction ran once before timing: {result['raw_prep_seconds']:.1f} s.",
        "Each case runs once per arm. Only completed runs are judged; phrase coverage counts"
        " other runs as misses. A completed run whose judge call failed keeps its phrase"
        " coverage and is not judged. Judge averages and claim counts cover judged runs only."
        " The judge lists the answer's material claims as supported, contradicted (by the"
        " reference facts, or by the anchored evidence where the facts are silent) or"
        " unverifiable; an unverifiable claim is not a contradiction.",
        "",
        "| Arm | Completed | Judged | Phrase coverage | Accuracy | Citations | Completeness |"
        " Contradicted claims | Unverifiable claims | Input tokens | Output tokens | Tool calls |"
        " Search embeddings | Seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        lines.append(
            f"| {arm} | {_qa_scores(arm_rows)}"
            f" | {sum(r['run'].input_tokens for r in arm_rows)}"
            f" | {sum(r['run'].output_tokens for r in arm_rows)}"
            f" | {sum(len(r['run'].tool_calls) for r in arm_rows)}"
            f" | {sum(r['run'].search_embed_calls for r in arm_rows)}"
            f" | {sum(r['run'].elapsed_seconds for r in arm_rows):.0f} |"
        )
    lines += [
        "",
        "By kind:",
        "",
        "| Kind | Arm | Completed | Judged | Phrase coverage | Accuracy | Citations |"
        " Completeness | Contradicted claims | Unverifiable claims |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for kind in dict.fromkeys(r["kind"] for r in rows):
        for arm in arms:
            kind_rows = [r for r in rows if r["kind"] == kind and r["arm"] == arm]
            lines.append(f"| {kind} | {arm} | {_qa_scores(kind_rows)} |")
    lines += [
        "",
        *_tool_call_report({a: [r["run"] for r in rows if r["arm"] == a] for a in arms}),
    ]
    failed = [r for r in rows if r["status"] != "completed"]
    if failed:
        lines += ["Runs not completed:", ""]
        lines += [f"- {r['case_id']} / {r['arm']}: {r['status']}" for r in failed]
        lines.append("")
    unjudged = [r for r in rows if r["status"] == "completed" and r["judge"] is None]
    if unjudged:
        lines += ["Completed runs not judged:", ""]
        lines += [f"- {r['case_id']} / {r['arm']}: {r['judge_missing']}" for r in unjudged]
        lines.append("")
    return lines


def _median_range(values: list[float], digits: int) -> str:
    """Median [minimum, maximum] (n=observations), or n/a for no observation."""
    if not values:
        return "n/a"
    median = statistics.median(values)
    return (
        f"{median:.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"
        f" (n={len(values)})"
    )


def _selection_line(selection: ExemplarSelection) -> str:
    """The exemplar selection of one holdout in one line."""

    def share(value: float | None, total: int) -> str:
        return "n/a" if value is None else f"{round(value * total)}/{total}"

    exemplars = selection.oracle_exemplar_families
    feedback = selection.oracle_feedback_families
    requested = selection.requested_type
    return (
        "Exemplar selection (prepare_draft on the instructions and run notes, deliverable type"
        f" {'none' if requested is None else repr(requested)}, version families counted once):"
        " named exemplars among the packet's exemplars"
        f" {share(selection.exemplar_recall_at_packet, exemplars)}, among its exemplars and"
        f" other candidates {share(selection.exemplar_recall_at_candidates, exemplars)};"
        f" named feedback documents with a packet lesson"
        f" {share(selection.feedback_attached, feedback)}."
    )


JUDGE_SCORES = (
    ("Structure", "structure_and_house_style"),
    ("Rigor", "technical_rigor"),
    ("Limitations", "limitations_and_review_readiness"),
    ("Discipline", "factual_discipline_no_contamination"),
    ("Effort saved", "editing_effort_saved"),
)


def _layer3_report(result: dict[str, Any]) -> list[str]:
    """Per arm over the draft rounds: completed drafts, both lexical-overlap measures and the
    blind judge's scores as median [minimum, maximum] with their observation counts, judged
    rounds, rubric pass counts, tokens and tool calls."""
    holdout_id = result["holdout_id"]
    rounds: list[dict[str, Any]] = result["rounds"]
    arms = list(dict.fromkeys(arm for r in rounds for arm in r["arms"]))
    lines = [f"## Layer 3: holdout `{holdout_id}`", ""]
    lines.append(
        f"{len(rounds)} draft rounds; every arm below runs in every round, and each round's"
        " completed drafts get one blind judge call."
    )
    for arm, reason in result["skipped_arms"].items():
        lines.append(f"{arm} was not run: {reason}.")
    selection = result["selection"]
    if selection is not None:
        lines += ["", _selection_line(selection)]
    lines += [
        "",
        "Drafts that did not complete are left out of the recall, contamination, rubric and"
        " judge columns; their tokens and tool calls are counted. Each median comes with"
        " [minimum, maximum] and n, the number of observations behind it; n/a means no"
        " observation.",
        "",
        "Lexical-overlap measures (wording, not meaning), over each draft's prose shingles"
        " (headings, tables and fenced blocks do not count): house-text recall is the share of"
        " the house shingles a draft holds; copied-text contamination is the share of a draft's"
        " prose shingles that come from its copied units, occur in a holdout document that holds"
        " the unit and occur neither in the reference nor in the run notes and instructions."
        " A unit is held when one holdout document holds at least"
        f" {_pct(STANDING_RECURRENCE_CONTAINMENT)} of its shingles. The holdout has"
        f" {result['house_units']} house units and {result['house_shingles']} house shingles"
        + ("; recall is n/a without house shingles." if not result["house_shingles"] else "."),
        "",
        "| Arm | Completed drafts | House-text recall | Copied-text contamination |"
        " Stray / draft shingles | Input tokens | Output tokens | Tool calls |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        cells = [r["arms"][arm] for r in rounds]
        overlaps = [c["overlap"] for c in cells if c["overlap"] is not None]
        recall = [o.house_text_recall for o in overlaps if o.house_text_recall is not None]
        contamination = [
            o.copied_text_contamination for o in overlaps if o.copied_text_contamination is not None
        ]
        runs = [c["run"] for c in cells]
        lines.append(
            f"| {arm} | {len(overlaps)}/{len(rounds)} | {_median_range(recall, 3)}"
            f" | {_median_range(contamination, 4)}"
            f" | {sum(o.stray_shingles for o in overlaps)}"
            f"/{sum(o.draft_shingles for o in overlaps)}"
            f" | {sum(r.input_tokens for r in runs)} | {sum(r.output_tokens for r in runs)}"
            f" | {sum(len(r.tool_calls) for r in runs)} |"
        )
    lines.append("")
    diagnostics = result["rubric_diagnostics"]
    if diagnostics:
        lines += ["Rubric checks passed over completed drafts:", ""]
        lines += ["| Check | " + " | ".join(arms) + " |", "| --- |" + " --- |" * len(arms)]
        for diagnostic in diagnostics:
            check_id = diagnostic.check_id
            marks = []
            for arm in arms:
                rubrics = [r["arms"][arm]["rubric"] for r in rounds]
                done = [scored for scored in rubrics if scored is not None]
                marks.append(f"{sum(s[check_id] for s in done)}/{len(done)}")
            lines.append(f"| {check_id} | " + " | ".join(marks) + " |")
        lines.append("")
        unsupplied = [
            (d.check_id, g.alternatives)
            for d in diagnostics
            for g in d.groups
            if g.found_in == "none"
        ]
        lines += [
            f"- `{check_id}`: required phrase found in no supplied input: {list(alternatives)}"
            for check_id, alternatives in unsupplied
        ]
        if unsupplied:
            lines.append("")
    else:
        lines += ["The holdout has no rubric checks.", ""]
    lines += _tool_call_report({arm: [r["arms"][arm]["run"] for r in rounds] for arm in arms})
    judged: dict[str, dict[str, list[float]]] = {a: {f: [] for _, f in JUDGE_SCORES} for a in arms}
    judged_rounds = dict.fromkeys(arms, 0)
    rankings: list[str] = []
    for r in rounds:
        key: dict[str, str] = r["blind_key"]
        comparison = r["blind_comparison"]
        if comparison is None:
            rankings.append(f"- round {r['round']}: not judged: {r['blind_comparison_missing']}")
            continue
        # A stored comparison names exactly the round's labels, once each.
        for arm in key.values():
            judged_rounds[arm] += 1
        for c in comparison.candidates:
            for _, score in JUDGE_SCORES:
                judged[key[c.candidate]][score].append(getattr(c, score))
        ranked = ", ".join(f"{label} ({key[label]})" for label in comparison.ranking)
        rankings.append(f"- round {r['round']}: {ranked}")
    lines += [
        "Blind judge (1-10) over the rounds that judged the arm, unblinded with each round's"
        " blind_key.json:",
        "",
        "| Arm | Judged rounds | " + " | ".join(name for name, _ in JUDGE_SCORES) + " |",
        "| --- | --- |" + " --- |" * len(JUDGE_SCORES),
    ]
    for arm in arms:
        judge_cells = " | ".join(_median_range(judged[arm][f], 1) for _, f in JUDGE_SCORES)
        lines.append(f"| {arm} | {judged_rounds[arm]}/{len(rounds)} | {judge_cells} |")
    lines += ["", "Rankings:", "", *rankings, ""]
    failed = [
        f"- round {r['round']} / {arm}: {c['run'].status}"
        for r in rounds
        for arm, c in r["arms"].items()
        if c["run"].status != "completed"
    ]
    if failed:
        lines += ["Runs not completed:", "", *failed, ""]
    drafts = f"holdouts/{holdout_id}/blind-drafts/round-<n>/"
    return [*lines, f"Blind drafts: `{drafts}`", ""]


@app.command()
def main(
    pack: Annotated[Path, typer.Option(help="Eval pack JSON (*.eval.json)")],
    snapshot: Annotated[Path, typer.Option(help="Snapshot compiled from the pack's corpus")],
    out: Annotated[Path, typer.Option(help="Private output directory, apart from every input")],
    corpus: Annotated[
        Path | None, typer.Option(help="Corpus directory; layers 2 and 3 need it")
    ] = None,
    layer: Annotated[Literal["all", "1", "2", "3"], typer.Option()] = "all",
    holdout: Annotated[
        str | None, typer.Option(help="Run only this holdout_id in layer 3; default: every one")
    ] = None,
    drafts: Annotated[
        int | None,
        typer.Option(min=1, help="Draft rounds per holdout in layer 3; default: 1"),
    ] = None,
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
    if max_cost is not None and not (math.isfinite(max_cost) and max_cost > 0):
        raise typer.BadParameter("--max-cost must be a positive finite number of USD")
    try:
        eval_pack = EvalPack.load(pack)
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise typer.BadParameter(f"--pack {pack} is not a valid eval pack: {exc}") from exc
    layers, holdouts = _check_inputs(eval_pack, layer, holdout, pack, snapshot, corpus, out, cache)
    if drafts is not None and "3" not in layers:
        raise typer.BadParameter("--drafts sets layer 3 draft rounds; layer 3 is not run")
    load_dotenv_if_present(Path.cwd() / ".env")
    # One meter: the spend limit and the usage report cover every model call of the run.
    meter = UsageMeter(max_cost)
    with TopicSnapshot.open(snapshot) as snap:
        embedding_model, embedding_dimensions = snap.embedding_model, snap.embedding_dimensions
    results: dict[str, Any] = {}
    report = ["# Arkive evaluation report", "", f"Pack: `{eval_pack.name}`", ""]
    report += [f"Snapshot: `{snapshot}`", ""]
    build_cache: BuildCache | None = None
    try:
        # The gateways check the credential, tracing and prices, and the cache is opened,
        # before `--out` exists and before any model call.
        embed = _embedder(embedding_model, embedding_dimensions, meter)
        gateway = OpenAIGateway(meter=meter, trace=trace)
        if "3" in layers:
            build_cache = BuildCache(cache) if cache else None
            holdout_gateway = OpenAIGateway(
                cache=build_cache,
                meter=meter,
                trace=trace,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        make_private_directory(out)
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
            corpus_dir = corpus
            assert corpus_dir is not None
            holdouts_dir = out / "holdouts"
            make_private_directory(holdouts_dir)
            results["layer3"] = layer3_results = {}
            for task in holdouts:
                holdout_dir = holdouts_dir / task.holdout_id
                make_private_directory(holdout_dir)

                async def layer3(task: Holdout = task, holdout_dir: Path = holdout_dir) -> Any:
                    return await run_layer3_holdout(
                        gateway,
                        holdout_gateway=holdout_gateway,
                        full_snapshot_path=snapshot,
                        corpus_dir=corpus_dir,
                        embedder=embed,
                        output_dir=holdout_dir,
                        holdout=task,
                        drafts=drafts or 1,
                        render_cache=build_cache,
                    )

                layer3_results[task.holdout_id] = layer3_result = anyio.run(layer3)
                blind_dir = holdout_dir / "blind-drafts"
                make_private_directory(blind_dir)
                for draft_round in cast(list[dict[str, Any]], layer3_result["rounds"]):
                    round_dir = blind_dir / f"round-{draft_round['round']}"
                    make_private_directory(round_dir)
                    for label, text in draft_round["blind_drafts"].items():
                        write_private_text(round_dir / f"{label}.md", text)
                    write_private_text(
                        round_dir / "blind_key.json",
                        json.dumps(draft_round["blind_key"], indent=2),
                    )
                report += _layer3_report(layer3_result)
    except BaseException as exc:
        # A stopped evaluation (the spend limit, or a failure outside any agent run) writes
        # no report, but still states what it spent.
        typer.echo(meter.summary_line())
        if isinstance(exc, ArkiveError | ValueError):
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc
        raise
    finally:
        if build_cache:
            build_cache.close()
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


# The files `main` writes into `--out` itself.
OUTPUT_FILES = ("results.json", "report.md")


def refuse_uncreatable(option: str, path: Path) -> None:
    """Refuse an absent `path` the evaluator could not create: one below an existing
    non-directory, or whose nearest existing ancestor it cannot write and search."""
    for ancestor in path.parents:
        if os.path.lexists(ancestor):
            if not ancestor.is_dir():
                raise typer.BadParameter(
                    f"{option} {path}: {ancestor} exists and is not a directory"
                )
            if not os.access(ancestor, os.W_OK | os.X_OK):
                raise typer.BadParameter(
                    f"{option} {path} cannot be created: {ancestor} is not writable"
                )
            return


def refuse_unwritable_out(out: Path) -> None:
    """Refuse an `--out` that `main` could not create or publish into: an absent `--out` that
    `refuse_uncreatable` refuses, an existing `--out` that is a symlink, not a directory or
    not writable, or an output file name in it taken by a directory. `main` publishes its
    output files by replacement, so an existing output file of any owner or mode is fine."""
    if not os.path.lexists(out):
        refuse_uncreatable("--out", out)
        return
    if out.is_symlink():
        raise typer.BadParameter(f"--out {out} is a symlink; give a real directory")
    if not out.is_dir():
        raise typer.BadParameter(f"--out {out} exists and is not a directory")
    if not os.access(out, os.W_OK | os.X_OK):
        raise typer.BadParameter(f"--out {out} is not writable")
    for name in OUTPUT_FILES:
        path = out / name
        if os.path.lexists(path) and stat.S_ISDIR(os.lstat(path).st_mode):
            raise typer.BadParameter(
                f"--out {out}: {name} is a directory; give a fresh output directory"
            )


def refuse_unusable_cache(cache: Path) -> None:
    """Refuse a `--cache` the build cache could not open and write: an existing one that is a
    symlink or not a regular file, that the evaluator cannot read and write, or whose
    directory it cannot write and search (SQLite writes `-wal` and `-shm` files beside it),
    and an absent one that `refuse_uncreatable` refuses."""
    if not os.path.lexists(cache):
        refuse_uncreatable("--cache", cache)
        return
    if cache.is_symlink() or not cache.is_file():
        raise typer.BadParameter(f"--cache {cache} is not a regular file")
    if not os.access(cache, os.R_OK | os.W_OK):
        raise typer.BadParameter(f"--cache {cache} is not readable and writable")
    if not os.access(cache.parent, os.W_OK | os.X_OK):
        raise typer.BadParameter(
            f"--cache {cache}: its directory {cache.parent} is not writable, and SQLite writes"
            " -wal and -shm files beside the cache"
        )


def _check_inputs(
    pack: EvalPack,
    layer: str,
    holdout_id: str | None,
    pack_path: Path,
    snapshot: Path,
    corpus: Path | None,
    out: Path,
    cache: Path | None,
) -> tuple[set[str], tuple[Holdout, ...]]:
    """The layers to run, `all` meaning every layer the pack defines, and the holdouts layer 3
    runs: the one `holdout_id` names, else every holdout of the pack.

    Refuses, before creating any output, gateway or build cache and before calling any model:
    a layer the pack cannot run, a `holdout_id` without layer 3 or unknown to the pack, a
    missing corpus, an output or cache path that overlaps an input or the other, an `--out`
    that `refuse_unwritable_out` refuses or, when layer 3 runs, that already holds a
    `holdouts` directory, a `--cache` that `refuse_unusable_cache` refuses when layer 3 runs,
    a snapshot that cannot be opened or was built from another corpus than the pack's, a
    retrieval case filter the snapshot cannot resolve or expected document it does not hold,
    a Q&A source anchor the snapshot cannot serve or whose evidence is over the judge's
    limit, a corpus folder whose files no longer match the snapshot, a holdout whose
    reference, included files or exemplar paths break its guards or whose reference or
    exemplar document has no section in the snapshot, and a Word or PowerPoint file a
    selected layer would convert (layer 2 reads the whole corpus, layer 3 each selected
    holdout's included files) without LibreOffice `soffice` on the path. `main` relies on
    these checks.
    """
    available = {
        name
        for name, present in (
            ("1", bool(pack.retrieval_cases)),
            ("2", bool(pack.qa_cases)),
            ("3", bool(pack.holdouts)),
        )
        if present
    }
    layers = available if layer == "all" else {layer}
    if not layers or not layers <= available:
        raise typer.BadParameter(f"pack {pack.name!r} defines no cases for layer {layer}")
    holdouts = pack.holdouts if "3" in layers else ()
    if holdout_id is not None:
        if "3" not in layers:
            raise typer.BadParameter("--holdout selects a layer 3 holdout; layer 3 is not run")
        holdouts = tuple(h for h in pack.holdouts if h.holdout_id == holdout_id)
        if not holdouts:
            known = [h.holdout_id for h in pack.holdouts]
            raise typer.BadParameter(f"--holdout {holdout_id!r} is not in the pack: {known}")
    needs_corpus = bool(layers & {"2", "3"})
    if needs_corpus and corpus is None:
        raise typer.BadParameter("layers 2 and 3 need --corpus")
    inputs = {"--pack": pack_path, "--snapshot": snapshot}
    if corpus is not None:
        inputs["--corpus"] = corpus
    writable = {"--out": out} | ({"--cache": cache} if cache is not None else {})
    refuse_overlapping_paths(writable, inputs)
    if cache is not None:
        refuse_overlapping_paths({"--cache": cache}, {"--out": out})
    refuse_unwritable_out(out)
    if cache is not None and "3" in layers:
        refuse_unusable_cache(cache)
    if "3" in layers and os.path.lexists(out / "holdouts"):
        raise typer.BadParameter(
            f"--out {out} already holds a holdouts directory from an earlier run; give a fresh"
            " output directory for layer 3"
        )
    try:
        with TopicSnapshot.open(snapshot) as snap:
            fingerprint = snap.corpus_fingerprint
            for case in pack.retrieval_cases if "1" in layers else ():
                try:
                    snap.resolve_filters(case.filters)
                    for doc_id in case.expected_docs:
                        snap.document(doc_id)
                except ArkiveError as exc:
                    raise typer.BadParameter(f"retrieval case {case.case_id!r}: {exc}") from exc
            for qa_case in pack.qa_cases if "2" in layers else ():
                try:
                    qa_evidence(snap, qa_case)
                except ArkiveError as exc:
                    raise typer.BadParameter(f"qa case {qa_case.case_id!r}: {exc}") from exc
                except ValueError as exc:
                    raise typer.BadParameter(str(exc)) from exc
            references = {h.holdout_id: snap.document(h.reference_doc_id) for h in holdouts}
            section_counts = {
                d.rel_path: len(snap.full_outline(d.doc_id))
                for d in (snap.list_documents() if holdouts else ())
            }
    except ArkiveError as exc:
        raise typer.BadParameter(f"--snapshot {snapshot}: {exc}") from exc
    if pack.corpus_fingerprint not in (None, fingerprint):
        raise typer.BadParameter(
            f"snapshot corpus_fingerprint {fingerprint} does not match pack {pack.name!r}"
            f" ({pack.corpus_fingerprint})"
        )
    office_files: set[str] = set()
    if needs_corpus and corpus is not None:
        try:
            sources = discover_corpus(corpus)
            live = corpus_fingerprint(
                (s.rel_path, hashlib.sha256(read_source(s)).hexdigest()) for s in sources
            )
        except ArkiveError as exc:
            raise typer.BadParameter(f"--corpus {corpus}: {exc}") from exc
        if live != fingerprint:
            raise typer.BadParameter(
                f"--corpus {corpus} has corpus_fingerprint {live}, but the snapshot was built"
                f" from {fingerprint}; rebuild the snapshot or restore the corpus"
            )
        office_files = {s.rel_path for s in sources if s.format in ("docx", "pptx")}
    converted = set(office_files) if "2" in layers else set()
    for task in holdouts:
        assert corpus is not None
        reference = references[task.holdout_id]
        try:
            check_reference_held_out(task, reference)
            selected = select_holdout_sources(corpus, task, reference)
        except (ArkiveError, ValueError) as exc:
            raise typer.BadParameter(f"holdout {task.holdout_id!r}: {exc}") from exc
        for rel_path in (reference.rel_path, *task.exemplar_paths):
            if not section_counts[rel_path]:
                raise typer.BadParameter(
                    f"holdout {task.holdout_id!r}: {rel_path} has no sections in --snapshot"
                    f" {snapshot}, so it cannot be a reference or exemplar document"
                )
        converted |= office_files & selected.keys()
    if converted and shutil.which("soffice") is None:
        raise typer.BadParameter(
            f"LibreOffice `soffice` is not on the path, and the selected layers convert"
            f" {len(converted)} Word or PowerPoint files, such as {min(converted)}"
        )
    return layers, holdouts


if __name__ == "__main__":
    app()
