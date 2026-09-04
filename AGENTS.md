# Agent instructions

Add a root rule only when it will change later agent behaviour and the evidence is already in this tree.

## Nested instructions

Read `AGENTS.md` from this root inward to the files in scope. A closer file refines this one. It overrides a root rule only when it names that override.

Put a rule in the narrowest directory where it stays true. Add a nested `AGENTS.md` only in the same change as that directory, and only if the rule is not true at root. Do not add an empty or repeating nested file, or an instruction file for a path that does not exist.

## Agent config

Version shared agent configuration in vendor-agnostic files: `AGENTS.md`, and `.agents/` when that directory exists. Do not make a vendor tool directory the source of truth.

## Write what exists

Name the current choice using only files, packages, services, directories, commands, and product objects that exist in this repository. Do not invent a parallel product definition, architecture, or decision register here.

## Names

Name code, files, docs, components, and concepts by purpose, responsibility, domain meaning, or outcome so the name makes sense in isolation.

- Use kebab-case for repository component directories, configuration assets, and command-line entry points.
- The root Python package is `arkive_rag` located under `src/arkive_rag/`. Python modules and packages follow standard snake_case.
- Do not name things after phases, options, workstreams, ticket numbers, or other planning labels that only make sense in the current conversation or an external document. A planning identifier may appear only when it is a first-class concept defined alongside the artifact.

Use comments to preserve information that code cannot state clearly: an invariant, a non-obvious constraint, the reason for a surprising choice, or an external contract. Do not narrate control flow, retain discarded alternatives, or use comments to compensate for vague names or misplaced responsibility.

## Tooling

When a language first lands in this repository, choose that language's current conventional portable package manager and a modern test runner in the same change. Prefer tools that reproduce on a clean checkout without a preconfigured machine. Record the choice in the component manifest, not in this file. After that, follow the toolchain already in the tree.

## Repository structure

A component owns what is required to understand, test, build, and operate it, and should stay understandable without first learning a repository-wide abstraction. When more than one component exists, the root composes them and coordinates common workflows; it does not absorb domain behaviour.

Standardise the ideas of develop, test, validate, build, and operate without requiring the same technology for each; stay idiomatic inside a component's own stack.

Membership in this repository does not imply shared lifecycle, dependencies, configuration, or tooling. Keep deployment, source, dependency, and workspace boundaries explicit, and share contracts across languages without forcing a repository-wide implementation abstraction.

Worktree compatibility is an invariant. Run meaningful work from independent Git worktrees under `.worktrees/`. Each checkout must establish its own working state and bootstrap, develop, test, and build without another checkout's mutable state or a privileged working directory. Concurrent worktrees must not collide through generated artifacts, development infrastructure, caches, ports, or other mutable resources.

## Engineering judgement

Treat a change as a material boundary change when it creates or alters a public API, cross-component contract, ownership of state or policy, persisted representation, deployment interface, or concurrency model. Before implementing one, compare at least two genuinely different designs. Choose using misuse resistance, ownership of invariants, dependency visibility, change locality, failure behaviour, and testability. Skip this comparison for ordinary local edits.

At a material call-site boundary, prefer a narrow contract that makes valid operations obvious and invalid combinations hard to express. Use separate operations or meaningful domain types instead of mode flags, correlated optional arguments, magic values, or caller-specific branches. Validate untrusted input strictly at the boundary; inside that boundary, trust the established contract.

The component that owns state or an invariant also owns its policy, validation, ordering, defaults, and recovery. Callers express intent; they do not reproduce the owner's rules. When one policy change requires coordinated edits in several callers, first reconsider the ownership boundary instead of synchronising copied conditionals.

Dependencies that affect behaviour must be visible in a component's contract or composition wiring. Do not hide I/O, mutable state, ordering requirements, or failure modes behind ambient globals, service locators, side-effect imports, or neutral-looking helpers.

Optimise abstractions for change locality and clear ownership, not line-count reduction. Keep simple duplication while concepts can evolve independently; extract a shared capability only when it represents one stable rule or behaviour with identifiable consumers.

A forwarding layer must enforce a contract, adapt a real boundary, own stable policy, or provide isolation required for verification. Otherwise call the owner directly. Do not add interfaces, factories, registries, or configuration switches for hypothetical variants, or generic `Helper`, `Utils`, `Common`, `Base`, or `Manager` containers that hide responsibility.

Perform preparatory refactoring only when the current shape prevents the requested change from being expressed or verified safely. Bound it to the obstructing code, preserve behaviour, and stop when the target change is local and its contract is visible. Do not turn preparatory work into adjacent cleanup.

## Failure semantics and anti-fallback policy

When a possible local failure appears, classify it before adding a fallback, retry, default, optional chain, swallowed catch, compatibility path, or degraded mode: decide whether that state should be possible, who owns it, and whether continuing is semantically valid. A fallback is valid only when its result is an explicit semantic outcome of the operation. "Handle errors gracefully" does not justify continuing.

| Failure | Treatment |
| --- | --- |
| Valid domain outcome | Represent it in the type or result |
| Invalid external input | Reject at the constructing boundary |
| State that should be impossible here | Strengthen the earlier invariant; do not add a local check |
| Transient dependency failure | Bounded retry only at the owner, and only if retry is safe |
| Expected degraded capability | Named fallback; make degradation observable |
| Broken invariant or programmer error | Fail loudly |
| Corrupt or ambiguous state | Stop, quarantine, or escalate; do not guess |
| Optional non-critical capability | Omit only when omission is contracted success |
| Invalid configuration update | Keep last-known-good and alert when that is the policy |

Do not manufacture a successful-looking value to keep execution moving: empty objects, empty strings, swallowed catches, and invented defaults flatten distinct failures into one apparently successful result. Defining the operation so a former error is already success is different: `remove(key)` may mean "ensure the key is absent."

If continuing requires guessing about correctness, fail loudly rather than recovering locally from a state the component cannot interpret.

Every fallback, retry, compatibility path, or degraded mode is additional system behaviour with its own correctness, freshness, authorisation, and failure properties, and needs a contract, tests, and observability. The owner's public result must still distinguish success, expected domain failure, and unavailability. A retry also needs a deadline and must not stack on another layer that already retries. A fallback that takes over must still record that the primary path failed.

## Operating

| Situation | Do | Don't |
| --- | --- | --- |
| Implementation | Smallest correct change, then verify and report | Stop at a proposal unless a plan was asked for |
| Ambiguous requirement | One short question, and only when it affects correctness, safety, data scope, persistence, UX, or public contracts | Block on harmless ambiguity |
| Bug or failing check | Reproduce the failing behaviour first | Patch before the first wrong behaviour is known |
| Feature | Name observable acceptance criteria before editing | Use implementation steps as acceptance criteria |
| Unexpected edits in the tree | Leave them | Overwrite work you did not make |
| Review | Findings first, by severity, with file/line | Lead with summary or praise |

## Verify

Never claim a check passed unless it ran and produced passing output. If it cannot run, say why and name the command.

When adding, moving, or deleting a test, name the invariant first and place it in one owning test layer. Use unit tests when one module owns the rule and it reproduces without I/O, persistence, process lifecycle, or multi-component coordination; integration tests when the rule lives at a boundary or depends on serialisation, persistence, retries, ordering, IPC, lifecycle, or component coordination; end-to-end tests only when the user-visible contract cannot be trusted from lower layers. If torn between unit and integration, choose integration.

Prefer the existing suite for that owning layer: edit an existing test in an existing file first; add a new test to that file next; create a new file in that suite only when the case needs its own owner-shaped home; create a standalone regression file only when no existing suite can express the case cleanly and the case has durable contract value.

Do not duplicate the same invariant across layers unless each layer proves a distinct failure mode, and do not lock implementation structure unless that unit owns the invariant. Search for tests of the same rule; keep the strongest owned location; merge unique assertions there; delete or simplify weaker duplicates. Name tests by behaviour and owner, not by ticket.

A regression test should capture the first wrong behaviour, not only a downstream symptom. Do not test forwarding layers merely for coverage.

Scale test depth with risk, uncertainty, blast radius, and reversibility. Exercise invalid inputs and expected failures when a boundary changes. If changed behaviour cannot receive automated coverage, state why and record the concrete verification used instead.

## Git

Do not push unless asked.
