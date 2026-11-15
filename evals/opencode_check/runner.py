"""`python -m evals.opencode_check SNAPSHOT... --out DIR`: run the scenario file against the
Arkive MCP server inside OpenCode, one fresh container per scenario, and write results.json
and report.md.

Each scenario container runs as the invoking user's UID and GID with the snapshots mounted
read-only and no other host path, on Docker's default network: its network egress is not
restricted. `OPENAI_API_KEY` reaches the containers through their environment only, and its
value is redacted from everything the runner writes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from arkive_rag.llm import load_dotenv_if_present
from arkive_rag.models import ArkiveError
from arkive_rag.snapshot import TopicSnapshot
from evals.opencode_check.scenarios import (
    CheckOutcome,
    Scenario,
    ScenarioFile,
    Transcript,
    evaluate,
    parse_events,
)
from evals.run import make_private_directory, refuse_overlapping_paths, write_private_text

CHECK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CHECK_DIR.parents[1]
DEFAULT_SCENARIOS = CHECK_DIR / "scenarios.json"
DEFAULT_MODEL = "openai/gpt-6.1-sol"
# Pinned in the Dockerfile; reported with the results.
OPENCODE_VERSION = "2.3.0"
IMAGE_REPOSITORY = "arkive-opencode-check"
IMAGE_FILES = ("Dockerfile", "opencode.json", "run-scenario.sh")
REDACTED_KEY = "[OPENAI_API_KEY redacted]"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@dataclass(frozen=True)
class ServedSnapshot:
    path: Path
    topic: str
    corpus_fingerprint: str
    facets: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioRun:
    scenario: Scenario
    exit_code: int | None
    seconds: float
    transcript: Transcript
    stderr_tail: str
    checks: list[CheckOutcome]
    # Docker's error when the scenario container could not be removed.
    cleanup_error: str | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def check_inputs(
    scenarios_path: Path, snapshots: Sequence[Path], out: Path
) -> tuple[ScenarioFile, list[ServedSnapshot]]:
    """Refuse, before any container or model call: an invalid scenario file, an output path
    overlapping an input, a snapshot that cannot be opened, two snapshots of one topic, and a
    scenario topic no snapshot serves."""
    try:
        scenario_file = ScenarioFile.load(scenarios_path)
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise typer.BadParameter(f"{scenarios_path} is not a valid scenario file: {exc}") from exc
    if not snapshots:
        raise typer.BadParameter("name at least one snapshot")
    inputs = {"--scenarios": scenarios_path} | {f"snapshot {p}": p for p in snapshots}
    refuse_overlapping_paths({"--out": out}, inputs)
    served: list[ServedSnapshot] = []
    for path in snapshots:
        try:
            with TopicSnapshot.open(path) as snap:
                served.append(
                    ServedSnapshot(
                        path=path.resolve(),
                        topic=snap.topic,
                        corpus_fingerprint=snap.corpus_fingerprint,
                        facets=tuple(f.name for f in snap.facets),
                    )
                )
        except ArkiveError as exc:
            raise typer.BadParameter(f"snapshot {path}: {exc}") from exc
    topics = [s.topic for s in served]
    if len(set(topics)) != len(topics):
        raise typer.BadParameter(f"two snapshots serve the same topic: {sorted(topics)}")
    missing = sorted(set(scenario_file.topics) - set(topics))
    if missing:
        raise typer.BadParameter(f"no snapshot serves the scenario topics {missing}")
    return scenario_file, served


def _run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)


def build_image(context_dir: Path) -> str:
    """Assemble the build context (the files in IMAGE_FILES, the wheel built from this
    checkout and the locked runtime requirements with hashes), and build the image unless an
    image with the same content tag exists. Returns the tag."""
    for name in IMAGE_FILES:
        shutil.copy2(CHECK_DIR / name, context_dir / name)
    _run(["uv", "build", "--wheel", "--out-dir", str(context_dir)], cwd=REPOSITORY_ROOT)
    _run(
        [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-header",
            "--format",
            "requirements-txt",
            "--output-file",
            str(context_dir / "requirements.txt"),
        ],
        cwd=REPOSITORY_ROOT,
    )
    digest = hashlib.sha256()
    for path in sorted(p for p in context_dir.iterdir() if p.is_file()):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    tag = f"{IMAGE_REPOSITORY}:{digest.hexdigest()[:12]}"
    present = subprocess.run(
        ["docker", "image", "inspect", tag], capture_output=True, text=True, check=False
    )
    if present.returncode != 0:
        _run(["docker", "build", "--tag", tag, str(context_dir)], timeout=1800)
    return tag


def container_name(run_id: str, scenario_id: str) -> str:
    return f"{IMAGE_REPOSITORY}-{run_id}-{scenario_id}"


def scenario_command(
    image: str,
    container: str,
    user: str,
    snapshots: Sequence[ServedSnapshot],
    model: str,
    prompt: str,
) -> list[str]:
    """The `docker run` command for one scenario, as `user` (`UID:GID`, so the owner-only
    snapshot files stay readable). The key is named, not given a value, so Docker copies it
    from the runner's environment and it never appears in the command. The container is not
    started with `--rm`: `run_scenario` removes it on every exit path."""
    mounts = [
        arg
        for i, s in enumerate(snapshots)
        for arg in ("--volume", f"{s.path}:/snapshots/{i}.arkive.db:ro")
    ]
    return [
        "docker",
        "run",
        "--name",
        container,
        "--user",
        user,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--env",
        "OPENAI_API_KEY",
        *mounts,
        "--entrypoint",
        "run-scenario",
        image,
        model,
        prompt,
    ]


def remove_container(name: str) -> str | None:
    """Force-remove the container `name`. Returns None once it is gone (`docker rm --force`
    also succeeds for a container that never started), else Docker's error."""
    try:
        done = subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if done.returncode != 0:
        return done.stderr.strip() or f"docker rm exit code {done.returncode}"
    return None


def redact(text: str, key: str) -> str:
    return text.replace(key, REDACTED_KEY)


def run_scenario(
    scenario: Scenario,
    command: list[str],
    container: str,
    timeout: float,
    facets_by_topic: dict[str, tuple[str, ...]],
    key: str,
) -> tuple[ScenarioRun, str]:
    """Run one scenario container and remove it, also on a timeout, an error or an interrupt.
    Returns the run and its raw event stream, both with `key` redacted."""
    start = time.monotonic()
    exit_code: int | None
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        exit_code, stdout, stderr = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        stdout = _decoded(exc.stdout)
        stderr = _decoded(exc.stderr)
    finally:
        cleanup_error = remove_container(container)
        if cleanup_error is not None:
            typer.echo(f"error: container {container} was not removed: {cleanup_error}", err=True)
    seconds = time.monotonic() - start
    stdout, stderr = redact(stdout, key), redact(stderr, key)
    transcript = parse_events(stdout.splitlines())
    checks = evaluate(scenario, transcript, exit_code, facets_by_topic)
    stderr_tail = "\n".join(stderr.strip().splitlines()[-20:])
    run = ScenarioRun(scenario, exit_code, seconds, transcript, stderr_tail, checks, cleanup_error)
    return run, stdout


def _decoded(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def run_record(run: ScenarioRun) -> dict[str, Any]:
    return {
        "scenario_id": run.scenario.scenario_id,
        "description": run.scenario.description,
        "prompt": run.scenario.prompt,
        "passed": run.passed,
        "exit_code": run.exit_code,
        "timed_out": run.exit_code is None,
        "seconds": round(run.seconds, 1),
        "tool_calls": [dataclasses.asdict(c) for c in run.transcript.tool_calls],
        "final_answer": run.transcript.final_answer,
        "error_events": list(run.transcript.errors),
        "unreadable_lines": list(run.transcript.unreadable_lines),
        "checks": [dataclasses.asdict(c) for c in run.checks],
        "usage": dataclasses.asdict(run.transcript.usage),
        "stderr_tail": run.stderr_tail,
    }


def render_report(
    model: str, image: str, served: Sequence[ServedSnapshot], runs: Sequence[ScenarioRun]
) -> str:
    passed = sum(r.passed for r in runs)
    cost = sum(r.transcript.usage.cost_usd for r in runs)
    lines = [
        "# OpenCode check report",
        "",
        f"OpenCode {OPENCODE_VERSION}, model `{model}`, image `{image}`.",
        "",
        "Snapshots:",
        "",
        *[
            f"- `{s.topic}` ({s.corpus_fingerprint[:12]}), facets: {', '.join(s.facets)}"
            for s in served
        ],
        "",
        f"Scenarios passed: {passed}/{len(runs)}.",
        "",
        "| Scenario | Result | Tool calls | Steps | Input | Cache read | Cache write | Output"
        " | OpenCode cost (USD) | Seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs:
        u = r.transcript.usage
        lines.append(
            f"| {r.scenario.scenario_id} | {'pass' if r.passed else 'FAIL'}"
            f" | {len(r.transcript.tool_calls)} | {u.steps} | {u.input_tokens}"
            f" | {u.cache_read_tokens} | {u.cache_write_tokens} | {u.output_tokens}"
            f" | {u.cost_usd:.4f} | {r.seconds:.0f} |"
        )
    lines += [
        "",
        f"OpenCode's own cost estimate over all scenarios: ${cost:.4f}. It sums the"
        " `step_finish` events, and OpenCode can omit the final step's event, so it is a lower"
        " bound. Query embeddings made by the `search` tool are not included.",
        "",
    ]
    for r in runs:
        lines += [f"## {r.scenario.scenario_id}: {'pass' if r.passed else 'FAIL'}", ""]
        lines += [r.scenario.description, ""]
        calls = ", ".join(
            f"{c.name}{'' if c.status == 'completed' else f' ({c.status})'}"
            for c in r.transcript.tool_calls
        )
        lines += [f"Tool calls: {calls or 'none'}", ""]
        lines += [f"- {'pass' if c.passed else 'FAIL'}: {c.check} ({c.detail})" for c in r.checks]
        answer = r.transcript.final_answer or "(none)"
        lines += ["", "Final answer:", "", *[f"> {line}" for line in answer.splitlines()], ""]
        if r.stderr_tail and not r.passed:
            lines += ["stderr (tail):", "", "```text", r.stderr_tail, "```", ""]
    return "\n".join(lines)


@app.command()
def main(
    snapshots: Annotated[list[Path], typer.Argument(help="Snapshots to serve (*.arkive.db)")],
    out: Annotated[Path, typer.Option(help="Private output directory, apart from every input")],
    model: Annotated[str, typer.Option(help="OpenCode model as provider/model")] = DEFAULT_MODEL,
    scenarios: Annotated[Path, typer.Option(help="Scenario file")] = DEFAULT_SCENARIOS,
    only: Annotated[
        list[str] | None, typer.Option(help="Run only these scenario IDs (repeatable)")
    ] = None,
    timeout: Annotated[float, typer.Option(help="Seconds allowed per scenario")] = 600.0,
) -> None:
    """Run the OpenCode scenarios against the snapshots (needs Docker and OPENAI_API_KEY)."""
    scenario_file, served = check_inputs(scenarios, snapshots, out)
    selected = [s for s in scenario_file.scenarios if not only or s.scenario_id in only]
    unknown = sorted(set(only or ()) - {s.scenario_id for s in scenario_file.scenarios})
    if unknown or not selected:
        raise typer.BadParameter(f"unknown scenario IDs {unknown}")
    load_dotenv_if_present(Path.cwd() / ".env")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key.strip():
        raise typer.BadParameter("OPENAI_API_KEY is not set or is empty")
    if os.getuid() == 0:
        raise typer.BadParameter("run the check as a non-root user; containers run as the caller")
    if shutil.which("docker") is None:
        raise typer.BadParameter("docker is not on PATH")
    events_dir = out / "events"
    outputs = [
        out / "results.json",
        out / "report.md",
        *(events_dir / f"{s.scenario_id}.jsonl" for s in selected),
    ]
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        make_private_directory(events_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    # Refused again when written; checked here so a link costs no container or model call.
    links = [str(p) for p in outputs if p.is_symlink()]
    if links:
        raise typer.BadParameter(f"refusing to write through symlinks: {links}")
    facets_by_topic = {s.topic: s.facets for s in served}
    with tempfile.TemporaryDirectory(prefix="arkive-opencode-image-") as context:
        image = build_image(Path(context))
    typer.echo(f"image {image}")
    # Container names are unique per run and scenario, so concurrent runs cannot collide.
    run_id = secrets.token_hex(8)
    user = f"{os.getuid()}:{os.getgid()}"
    runs: list[ScenarioRun] = []
    for scenario in selected:
        container = container_name(run_id, scenario.scenario_id)
        command = scenario_command(image, container, user, served, model, scenario.prompt)
        run, stream = run_scenario(scenario, command, container, timeout, facets_by_topic, key)
        write_private_text(events_dir / f"{scenario.scenario_id}.jsonl", stream)
        runs.append(run)
        typer.echo(
            f"{scenario.scenario_id}: {'pass' if run.passed else 'FAIL'}"
            f" ({run.seconds:.0f} s, OpenCode cost ${run.transcript.usage.cost_usd:.4f})"
        )
    results = {
        "opencode_version": OPENCODE_VERSION,
        "model": model,
        "image": image,
        "snapshots": [
            {"path": str(s.path), "topic": s.topic, "corpus_fingerprint": s.corpus_fingerprint}
            for s in served
        ],
        "passed": sum(r.passed for r in runs),
        "total": len(runs),
        "opencode_cost_usd": sum(r.transcript.usage.cost_usd for r in runs),
        "scenarios": [run_record(r) for r in runs],
    }
    write_private_text(out / "results.json", json.dumps(results, indent=2, ensure_ascii=False))
    write_private_text(out / "report.md", render_report(model, image, served, runs))
    typer.echo(f"wrote {out / 'results.json'} and {out / 'report.md'}")
    left = [container_name(run_id, r.scenario.scenario_id) for r in runs if r.cleanup_error]
    if left:
        typer.echo(f"error: containers not removed: {', '.join(left)}", err=True)
    if left or not all(r.passed for r in runs):
        raise typer.Exit(1)
