# whetkit

**Measure — then improve — how well LLM agents pick and use the tools your MCP
server exposes.**

`whetkit` is a local-first CLI that runs an agent against your
[MCP](https://modelcontextprotocol.io) server on a set of eval tasks, scores
its **tool-selection hit-rate**, and then *curates* the tool set — renaming
cryptic tools, rewriting vague descriptions, pruning noise, and merging
duplicates — via a **reversible overlay proxy** that never modifies your
server. It re-runs the eval through the overlay and hands you a before/after
report.

Unlike MCP inspectors and testing frameworks, whetkit **closes the
optimization loop**: it measures agent behavior, proposes a curated tool
surface, applies it through a reversible proxy, and re-runs the same evals to
quantify the improvement.

```text
whetkit doctor    ──►  ten-second lint of the tool surface        (no tasks, no API key)
whetkit inspect   ──►  what does the agent actually see?
whetkit generate  ──►  draft eval tasks from the inventory        (review, then run)
whetkit run       ──►  how often does it pick the right tools?   (hit-rate)
whetkit curate    ──►  fix the tool set, prove it helped          (before → after)
```

## Why tool curation matters

Agents don't read your code — they read your tool names, descriptions, and
schemas. A server that grew organically ends up with `data_query_1`,
`proc_ord`, and `do_thing`: every one of them costs the model a guess, and
every duplicate splits its attention. In practice a large share of agent
failures on MCP servers are *tool-selection* failures — the model calls the
wrong tool, loops through near-duplicates, or gives up — and they are fixable
without touching a line of server code, because the fix is metadata.
whetkit makes that loop measurable: score the failures, patch the metadata
through an overlay, and show the hit-rate delta.

## Install

```sh
uv tool install whetkit   # or: uvx whetkit / pipx install whetkit
```

(Released to PyPI from tags — see [RELEASING.md](RELEASING.md). PyPI can lag
behind main; the quickstart below runs from source.)

## Quickstart (5 minutes)

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key (or
OpenAI — see `--model`). Python is pinned via `.python-version`; uv fetches
it automatically.

```sh
git clone https://github.com/benlamlih/whetkit && cd whetkit
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
```

**1. Inspect the bundled sample server** — a deliberately messy e-commerce
server (14 tools: cryptic names, vague descriptions, duplicates, noise):

```sh
uv run whetkit inspect --server examples/sample-server
```

**2. Baseline eval** — run 5 tasks against it and score the hit-rate
(deterministic tool-matching + LLM-judge on the final answers):

```sh
uv run whetkit run --server examples/sample-server --tasks examples/tasks
```

**3. Curate and prove it** — analyze the failures, generate a curation
overlay, re-run the eval through it, and get the before/after:

```sh
uv run whetkit curate --server examples/sample-server --tasks examples/tasks
```

This writes:

- `.whetkit/curation-plan.yaml` — the reviewable, hand-editable overlay plan
- `.whetkit/report.html` — self-contained before/after report (open it in a browser)
- `.whetkit/report.json` — the same data, machine-readable
- `.whetkit/traces.sqlite3` — full reasoning-path traces of every run

The sample server's failures are tool-selection failures, so weaker models
flip several tasks from MISS to HIT through the overlay. Frontier models
often ace even the messy baseline on a 14-tool server — there the delta
shows up in the other columns instead: tools exposed, tokens per task, and
extra calls. On large real-world servers (dozens of tools, near-duplicates)
the hit-rate delta comes back.

Stdio servers' own logs are hidden so they can't garble the output; set
`WHETKIT_SERVER_LOGS=1` to see them when debugging a server that won't start.

The plan is yours to edit: tweak a rename the optimizer got wrong, un-hide a
tool, then re-score the curated view directly —

```sh
uv run whetkit run --server examples/sample-server --tasks examples/tasks \
  --plan .whetkit/curation-plan.yaml --group curated-v2
```

**4. Use the curated view for real** — serve it to any MCP client:

```sh
uv run whetkit overlay --server examples/sample-server --plan .whetkit/curation-plan.yaml
```

Nothing about your origin server changes, ever. Delete the plan and you are
back to the original world.

## Pointing it at your own server

- `--server` accepts a URL (streamable HTTP; `--http-mode stateless` for
  2026-07-28-spec servers), a directory containing `server.json` or
  `server.py`, or a `.py`/`.json` path directly.
- Write tasks in YAML — format reference in
  [docs/task-format.md](docs/task-format.md) — or draft them:
  `whetkit generate --server <your-server> --out tasks/generated.yaml`
  writes candidate tasks from the tool inventory (validated against the
  live tool list; review before trusting).
- `--model` / `--judge-model` / `--optimizer-model` take
  `provider:model_id`, e.g. `anthropic:claude-sonnet-5` or `openai:gpt-5.2`.
  Keys come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.

## Commands

| Command | What it does |
|---|---|
| `whetkit doctor` | Lint the tool surface: vague descriptions, cryptic names, near-duplicates, context bloat. `--json`; `--fail-on warn` for CI. |
| `whetkit inspect` | Tool inventory: names, params, description tokens, schema complexity. |
| `whetkit generate` | Draft eval tasks from the inventory (server-context aware, read-only unless `--allow-writes`); review-before-trust YAML. |
| `whetkit run` | Agentic eval with real tool execution. `--runs N` for mean±range and flaky-task detection, `--concurrency` for independent tasks, `--reset-cmd` for stateful fixtures, `--plan` to score a curated view, `--summary-json` for machine-readable results (with cost estimates). |
| `whetkit curate` | Baseline → LLM-proposed overlay plan → curated eval → before/after report. `--prune-unused` for the cost play. |
| `whetkit fix` | Self-correcting curation: propose → eval → feed regressions back → revise, up to `--max-iterations`; keeps the best plan by measured results. |
| `whetkit plan-init` | Scaffold a view plan: `--keep a,b`, `--from-tasks tasks/`, `--from-traces traces.sqlite3` — hide everything else. |
| `whetkit diff` | Compare two `--summary-json` files: metric deltas + per-task PASS/MISS/FLAKY transitions. |
| `whetkit export` | Share a plan: `--to markdown` (upstream-PR fix table) or `--to json` (gateway overrides). |
| `whetkit report` | Rebuild the HTML/JSON report from stored traces. |
| `whetkit overlay` | Serve the curated view as a stdio MCP server. |

Authenticated servers: any string in `server.json` may reference `${ENV_VARS}`
(e.g. an `Authorization` header) — credentials never live in the file.
Summaries flag probable task-spec gaps, errored runs, and failed tool calls
so a bad score always says why.

## How scoring works

- **Deterministic tool-match**: each task lists the expected tool calls
  (with acceptable alternatives, optionally ordered). Order-tolerant by
  default, `--match-mode exact` for strict grading. Reports
  precision/recall, missing and extra calls.
- **LLM-as-judge**: grades the agent's final answer against the task's
  natural-language `success_criteria` with a strict calibrated prompt;
  verdicts are cached in SQLite. `--judge auto|on|off`.
- **Hit** = right tools *and* (when judged) task success. The headline
  metric is the hit-rate across tasks.

More docs: [docs/task-format.md](docs/task-format.md) ·
[docs/curation.md](docs/curation.md) · [VERSIONS.md](VERSIONS.md) ·
[MIGRATION.md](MIGRATION.md)

## Development

```sh
uv sync
uv run pytest        # full suite, no API key needed (scripted fake provider)
uv run ruff check .
```

Dependency versions are pinned exactly and documented with sources and check
dates in [VERSIONS.md](VERSIONS.md). The MCP transport layer supports stdio,
legacy stateful streamable-HTTP (2025 spec), and stateless streamable-HTTP
(2026-07-28 spec); the SDK-facing code is isolated for the v1→v2 migration
([MIGRATION.md](MIGRATION.md)).

Scope note: whetkit is deliberately a local-first CLI. Hosting, dashboards,
multi-tenancy, and security tooling are out of scope for Stage 1 — the
architecture just leaves room for them.

## License

Apache-2.0 — see [LICENSE](LICENSE).
