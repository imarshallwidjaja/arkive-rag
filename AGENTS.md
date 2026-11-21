# Agent instructions

Read `AGENTS.md` from this root inward to the files you are editing. A closer `AGENTS.md` refines this file and supersedes a root rule when it names that override.

## Progressive discovery

Open the nearest file below when your work enters its directory:

| Directory | Read | Owns |
| --- | --- | --- |
| `src/arkive_rag/` | [`src/arkive_rag/AGENTS.md`](src/arkive_rag/AGENTS.md) | Package module boundaries, schema `6.0` hard-cut rules, path-hash IDs, fixed OpenAI models, prompt/MCP byte ceilings, and `prepare_draft` packing invariants. |
| `evals/` | [`evals/AGENTS.md`](evals/AGENTS.md) | `arkive-eval-pack/3` harness, preflight/leak guards, frozen oracle control rules, example corpora/packs, and Docker `opencode_check`. |
| `tests/` | [`tests/AGENTS.md`](tests/AGENTS.md) | Offline-only test invariant, `.env` isolation, owning test-file map, and targeted pytest invocations. |
| `docs/` | [`docs/AGENTS.md`](docs/AGENTS.md) | Human documentation suite, strict public-corpus-only PII rule, and visual assets (`docs/assets/`). |

Before changing dependencies, the build pipeline, the snapshot schema, or the MCP tools, also read [`docs/architecture.md`](docs/architecture.md).

## Commands and scratch isolation

Run every command from the checkout root. Keep the virtual environment, tool caches, pytest cache/temp files, and build outputs outside the checkout on a per-checkout scratch path so concurrent worktrees never collide:

```sh
export SCRATCH="$(mktemp -d /tmp/arkive-checkout.XXXXXX)"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv" MYPY_CACHE_DIR="$SCRATCH/mypy" \
  RUFF_CACHE_DIR="$SCRATCH/ruff" PYTHONDONTWRITEBYTECODE=1
uv sync --locked
uv run --locked ruff check src tests evals
uv run --locked ruff format --check src tests evals
uv run --locked mypy
uv run --locked pytest -o cache_dir="$SCRATCH/pytest-cache" --basetemp="$SCRATCH/pytest-tmp"
uv build --out-dir "$SCRATCH/dist"
```

Run checks in that order (`ruff check` -> `ruff format --check` -> `mypy` -> `pytest` -> `uv build`). `pytest` is strictly offline; run real OpenAI calls only via `arkive-rag build`, `python -m evals.run`, or `python -m evals.opencode_check` with an explicit `--max-cost`.

## Names and repo layout

- Name code, files, docs, and concepts by purpose, domain meaning, or outcome so each name makes sense in isolation. Never encode temporary planning labels (phases, options, workstreams, ticket IDs) into durable names.
- Use `kebab-case` for component directories, CLI entry points (`arkive-rag`), and example corpora; use standard `snake_case` inside `src/arkive_rag/`, `evals/`, and `tests/`.
- Keep shared agent config in vendor-agnostic files (`AGENTS.md`, and `.agents/` if present). Do not make a vendor directory the source of truth.
- Worktree compatibility is an invariant: each checkout must bootstrap, test, and build independently without touching another checkout's mutable state. Preserve unrelated ignored directories in the root checkout (such as legacy local folders or `.venv`).
- Add a nested `AGENTS.md` only for an existing directory whose rules differ from root, and never inside `evals/example_corpora/` (corpus folders treat any `AGENTS.md` as a curator note).

## Engineering judgement

- **Material boundary changes:** Before changing a public API, CLI/MCP contract, persisted schema, ownership of state/policy, or concurrency model, compare at least two genuinely different designs against misuse resistance, invariant ownership, dependency visibility, change locality, failure behaviour, and testability.
- **Narrow contracts:** Validate untrusted inputs at the constructing boundary; inside the boundary, trust the established contract. Prefer explicit domain types or separate operations over mode flags, correlated optional arguments, or magic values.
- **Single owner:** The component that owns state or an invariant owns its validation, ordering, defaults, and recovery. Dependencies that affect behaviour belong in explicit arguments or composition wiring, never hidden behind ambient globals or side-effect imports.
- **Abstractions:** Keep simple duplication while concepts can evolve independently; extract shared code only for one stable rule with real consumers. Avoid `Helper`, `Utils`, `Common`, `Base`, or `Manager` catch-all containers.
- **Comments:** Record invariants, non-obvious constraints, or external contracts that code cannot express. Do not narrate control flow or retain discarded alternatives.

## Failure semantics and anti-fallback policy

Classify every failure before adding a fallback, retry, default, or degraded mode. A fallback is valid only when its result is an explicit semantic outcome of the operation.

| Failure | Treatment |
| --- | --- |
| Valid domain outcome | Represent it in the type or result |
| Invalid external input | Reject at the constructing boundary |
| Impossible state | Strengthen the earlier invariant; do not add a local patch |
| Transient dependency failure | Bounded retry only at the owner when retry is safe |
| Expected degraded capability | Named fallback with observable degradation |
| Broken invariant or programmer error | Fail loudly |
| Corrupt or ambiguous state | Stop, quarantine, or escalate; do not guess |
| Optional non-critical capability | Omit only when omission is contracted success |

Never manufacture empty strings, empty objects, swallowed catches, or invented defaults to hide distinct failures.

## Operating and verification

- Reproduce a bug or failing check before patching. Leave unexpected edits you did not make untouched.
- Never claim a check passed unless it ran on the final tree and produced passing output; state any unrun check and why.
- Place each test in one owning layer (unit for pure single-module rules, integration for persistence/IPC/ordering/coordination, end-to-end only when lower layers cannot verify the user-visible contract). Prefer editing or extending the existing suite file over creating standalone regression files.
- Keep all client documents, private facts, snapshots (`*.arkive.db`), and local eval packs (`*.eval.local.json`) out of Git.
- Do not push unless explicitly asked.
