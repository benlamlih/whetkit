"""Command-line entry point for whetkit."""

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer

from whetkit.mcp import HttpMode, ServerSpec, inspect_server, resolve_server_spec

app = typer.Typer(
    name="whetkit",
    help="Evaluate and improve LLM agent tool selection on MCP servers.",
    no_args_is_help=True,
)


def _resolve_server(value: str, http_mode: HttpMode) -> ServerSpec:
    """resolve_server_spec with a CLI-friendly error instead of a traceback."""
    try:
        return resolve_server_spec(value, http_mode=http_mode)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _judge_key_var(judge_model: str) -> str | None:
    """The env var that holds the judge provider's API key."""
    from whetkit.llm import parse_model

    provider_name, _ = parse_model(judge_model)
    return {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider_name)


def _judge_enabled(judge: str, judge_model: str) -> bool:
    """--judge auto: grade with the LLM judge only when its API key is set."""
    if judge not in ("auto", "on", "off"):
        # an unknown value used to silently mean 'auto' — refuse it instead
        raise typer.BadParameter("--judge must be 'auto', 'on', or 'off'")
    if judge in ("on", "off"):
        return judge == "on"
    key_var = _judge_key_var(judge_model)
    return bool(key_var and os.environ.get(key_var))


def _judge_skip_hint(judge_model: str) -> str:
    """The 'judge skipped' hint, naming the env var for THIS judge's provider."""
    key_var = _judge_key_var(judge_model) or "the judge provider's API key"
    return f"(LLM judge skipped: no API key found — set {key_var} or pass --judge on)"


def _match_mode(value: str):
    """Validate --match-mode where options are read: a typo must be a
    BadParameter, not a ValueError traceback from inside the async body."""
    from whetkit.scoring import MatchMode

    try:
        return MatchMode(value)
    except ValueError as exc:
        valid = ", ".join(f"'{m.value}'" for m in MatchMode)
        raise typer.BadParameter(f"--match-mode must be one of: {valid}") from exc


def _resolve_task_servers(
    tasks: list, server_override: str | None, http_mode: HttpMode
) -> dict[str, ServerSpec]:
    if server_override is not None:
        spec = _resolve_server(server_override, http_mode)
        return {task.server: spec for task in tasks}
    return {
        task.server: _resolve_server(task.server, http_mode)
        for task in {t.server: t for t in tasks}.values()
    }


def _single_server_spec(servers: dict[str, ServerSpec], command: str) -> ServerSpec:
    """The one server a curation command may target.

    curate/fix inspect ONE server and write ONE plan. Silently curating only
    ``next(iter(servers))`` while the tasks span several servers would grade
    the plan against runs it never touched — refuse instead, before any
    provider call spends money.
    """
    distinct = {spec.model_dump_json(): spec for spec in servers.values()}
    if len(distinct) > 1:
        labels = ", ".join(sorted(spec.label() for spec in distinct.values()))
        raise typer.BadParameter(
            f"tasks span {len(distinct)} different servers ({labels}) — curating "
            f"multiple servers in one plan is unsupported. Run 'whetkit {command}' "
            "once per server (split the task set, or pass --server to force one)."
        )
    return next(iter(servers.values()))


def _write_report(report, out_dir: str) -> tuple[str, str]:
    from whetkit.report import render_html

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "report.html"
    json_path = out / "report.json"
    html_path.write_text(render_html(report))
    json_path.write_text(report.model_dump_json(indent=2))
    return str(html_path), str(json_path)


def _summary_payload(group_name: str, summary, task_runs: list) -> dict:
    """One run's metrics as plain data — what --summary-json emits per run."""
    return {
        "group": group_name,
        "hit_rate": summary.hit_rate,
        "tool_hit_rate": summary.tool_hit_rate,
        "judge_pass_rate": summary.judge_pass_rate,
        "avg_precision": summary.avg_precision,
        "avg_recall": summary.avg_recall,
        "avg_extra_calls": summary.avg_extra_calls,
        "tokens_in": sum(r.total_usage.input_tokens for r in task_runs),
        "tokens_out": sum(r.total_usage.output_tokens for r in task_runs),
        "latency_ms": sum(r.total_latency_ms for r in task_runs),
        "tasks": [
            {
                "id": score.task_id,
                "hit": score.hit,
                "tool_hit": score.tool_hit,
                "judge_passed": score.judge.passed if score.judge else None,
                "spec_gap": score.spec_gap,
                "run_status": str(score.run_status),
                "tool_errors": score.tool_errors,
                "called": score.tool_match.called,
                "missing": score.tool_match.missing_slots,
                "extra_calls": score.tool_match.extra_calls,
            }
            for score in summary.scores
        ],
    }


async def _eval_repeated(
    task_list: list,
    servers: dict[str, ServerSpec],
    config,
    *,
    runs: int,
    base_group: str,
    store_path: str,
    client_factory,
    score_one,
) -> tuple[list[list], list]:
    """Run the whole task set ``runs`` times; persist and score each repetition.

    Group names get a ``-1..-N`` suffix when runs > 1, and the whole group
    family (the base name plus any suffixed variants from earlier
    invocations) is replaced on the first repetition. Each repetition is
    persisted before scoring so a failure later in the pipeline (judge,
    optimizer, curated eval) never throws away paid agent runs. Returns
    (per-repetition task runs, per-repetition summaries).
    """
    from whetkit.runner import run_task
    from whetkit.tracing import TraceStore

    all_runs: list[list] = []
    summaries: list = []
    for rep in range(1, runs + 1):
        group_name = base_group if runs == 1 else f"{base_group}-{rep}"
        if runs > 1:
            typer.echo(f"-- run {rep}/{runs} --", err=True)
        rep_runs = []
        for task in task_list:
            typer.echo(f"running {task.id} ...", err=True)
            rep_runs.append(
                await run_task(task, servers[task.server], config, client_factory=client_factory)
            )
        with TraceStore(store_path) as trace_store:
            if rep == 1:
                trace_store.delete_group_family(base_group)
            trace_store.save_runs(rep_runs, run_group=group_name)
        all_runs.append(rep_runs)
        summaries.append(await score_one(rep_runs))
    return all_runs, summaries


def _group_family_note(base_group: str, runs: int) -> str:
    return f"'{base_group}-1'..'-{runs}'" if runs > 1 else f"'{base_group}'"


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"whetkit {version('whetkit')}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the whetkit version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Evaluate and improve LLM agent tool selection on MCP servers."""


@app.command()
def inspect(
    server: Annotated[
        str,
        typer.Option("--server", help="MCP server: URL, directory, server.json, or server.py"),
    ],
    http_mode: Annotated[
        HttpMode,
        typer.Option(
            "--http-mode",
            help="HTTP session mode: 'stateful' (legacy 2025) or 'stateless' (2026-07-28 spec)",
        ),
    ] = HttpMode.STATEFUL,
) -> None:
    """Connect to an MCP server and print its tool inventory."""
    spec = _resolve_server(server, http_mode)
    inventory = asyncio.run(inspect_server(spec))

    for line in inventory.summary_lines():
        typer.echo(line)
    typer.echo()

    name_w = max((len(t.name) for t in inventory.tools), default=4)
    header = f"{'NAME':<{name_w}}  {'PARAMS':>6}  {'CPLX':>4}  {'TOKENS':>6}  DESCRIPTION"
    typer.echo(header)
    typer.echo("-" * len(header))
    for tool in inventory.tools:
        desc = " ".join(tool.description.split())
        if len(desc) > 70:
            desc = desc[:67] + "..."
        typer.echo(
            f"{tool.name:<{name_w}}  {tool.param_count:>6}  {tool.complexity:>4}  "
            f"{tool.description_tokens:>6}  {desc}"
        )


@app.command()
def doctor(
    server: Annotated[
        str,
        typer.Option("--server", help="MCP server: URL, directory, server.json, or server.py"),
    ],
    json_out: Annotated[
        bool, typer.Option("--json", help="Emit findings as JSON instead of text")
    ] = False,
    fail_on: Annotated[
        str,
        typer.Option(
            "--fail-on",
            help="Exit non-zero when findings at this severity exist: 'error', 'warn', or 'never'",
        ),
    ] = "never",
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Lint a server's tool surface: vague descriptions, cryptic names,
    near-duplicates, schema and context-budget problems. No tasks or API
    key needed."""
    from whetkit.doctor import Severity, diagnose

    if fail_on not in ("error", "warn", "never"):
        raise typer.BadParameter("--fail-on must be 'error', 'warn', or 'never'")

    spec = _resolve_server(server, http_mode)
    inventory = asyncio.run(inspect_server(spec))
    findings = diagnose(inventory)

    if json_out:
        import json as jsonlib

        typer.echo(jsonlib.dumps([f.model_dump() for f in findings], indent=2))
    else:
        for line in inventory.summary_lines():
            typer.echo(line)
        typer.echo()
        if not findings:
            typer.echo("No problems found — this tool surface reads clean.")
        for finding in findings:
            typer.echo(f"[{finding.severity.upper():<5}] {finding.check}: {finding.message}")
        counts = {s: sum(f.severity == s for f in findings) for s in Severity}
        if findings:
            typer.echo(
                f"\n{counts[Severity.ERROR]} error(s), {counts[Severity.WARN]} warning(s), "
                f"{counts[Severity.INFO]} info — 'whetkit curate' can propose and prove fixes."
            )

    worst_hit = {
        "error": any(f.severity == Severity.ERROR for f in findings),
        "warn": any(f.severity in (Severity.ERROR, Severity.WARN) for f in findings),
        "never": False,
    }[fail_on]
    if worst_hit:
        raise typer.Exit(code=1)


@app.command()
def generate(
    server: Annotated[
        str,
        typer.Option("--server", help="MCP server: URL, directory, server.json, or server.py"),
    ],
    out: Annotated[
        str, typer.Option("--out", help="Where to write the drafted task YAML")
    ] = "tasks/generated.yaml",
    count: Annotated[int, typer.Option("--count", help="How many tasks to draft")] = 5,
    model: Annotated[
        str, typer.Option("--model", help="Generator model as provider:model_id")
    ] = "anthropic:claude-sonnet-5",
    allow_writes: Annotated[
        bool,
        typer.Option(
            "--allow-writes",
            help="Let drafts include non-destructive write tasks (default: read-only only)",
        ),
    ] = False,
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Draft eval tasks from the server's tool inventory (review before
    trusting — the header in the output file says the same)."""
    from whetkit.generate import GeneratorConfig, generate_tasks, write_tasks_yaml

    spec = _resolve_server(server, http_mode)
    inventory = asyncio.run(inspect_server(spec))
    tasks, warnings = asyncio.run(
        generate_tasks(
            inventory,
            server,
            count=count,
            config=GeneratorConfig(model=model),
            server_context=spec.label(),
            allow_writes=allow_writes,
        )
    )
    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)
    if not tasks:
        typer.echo("no valid tasks were drafted — try again or write them by hand", err=True)
        raise typer.Exit(code=1)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_tasks_yaml(tasks, out)
    typer.echo(f"drafted {len(tasks)} task(s) to {out} — review them, then:")
    typer.echo(f"  whetkit run --server {server} --tasks {out}")


@app.command(name="plan-init")
def plan_init(
    server: Annotated[
        str,
        typer.Option("--server", help="MCP server: URL, directory, server.json, or server.py"),
    ],
    keep: Annotated[
        str, typer.Option("--keep", help="Comma-separated tool names to keep visible")
    ] = "",
    from_tasks: Annotated[
        str | None,
        typer.Option(
            "--from-tasks",
            help="Also keep every tool referenced by these tasks' expected_tools",
        ),
    ] = None,
    from_traces: Annotated[
        str | None,
        typer.Option(
            "--from-traces",
            help=(
                "Also keep every tool actually called in this trace store "
                "(what real runs used, including tools no spec lists)"
            ),
        ),
    ] = None,
    traces_group: Annotated[
        str | None,
        typer.Option("--traces-group", help="Restrict --from-traces to one run group"),
    ] = None,
    out: Annotated[
        str, typer.Option("--out", help="Where to write the plan YAML")
    ] = ".whetkit/curation-plan.yaml",
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Scaffold a view plan: keep the named tools, hide everything else.
    The fastest way to serve a lean read-only slice of a big server."""
    from whetkit.curation import CurationPlan, ToolOverride, save_plan
    from whetkit.datasets import load_tasks

    keep_set = {name.strip() for name in keep.split(",") if name.strip()}
    if from_tasks:
        for task in load_tasks(from_tasks):
            for slot in task.expected_tool_slots:
                keep_set.update(slot)
    if from_traces:
        from whetkit.tracing import TraceStore

        if not Path(from_traces).is_file():
            raise typer.BadParameter(f"no trace store at {from_traces}")
        with TraceStore(from_traces) as trace_store:
            for run in trace_store.load_runs(traces_group):
                keep_set.update(run.called_tool_names)
    if not keep_set:
        raise typer.BadParameter(
            "nothing to keep — pass --keep, --from-tasks, and/or --from-traces"
        )

    spec = _resolve_server(server, http_mode)
    inventory = asyncio.run(inspect_server(spec))
    names = {t.name for t in inventory.tools}
    if unknown := sorted(keep_set - names):
        typer.echo(f"warning: not on the server, ignoring: {', '.join(unknown)}", err=True)

    hidden = [
        ToolOverride(original_name=t.name, hidden=True, reason="Not part of this view's workflows.")
        for t in inventory.tools
        if t.name not in keep_set
    ]
    plan = CurationPlan(
        server=spec.label(),
        notes=f"View plan: keep {len(names & keep_set)} tool(s), hide the rest.",
        overrides=hidden,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_plan(plan, out)
    typer.echo(f"kept {len(names & keep_set)}, hidden {len(hidden)} — wrote {out}")
    typer.echo(f"score it:  whetkit run --server {server} --tasks <tasks> --plan {out}")


@app.command()
def export(
    plan: Annotated[
        str, typer.Option("--plan", help="Curation plan YAML to export")
    ] = ".whetkit/curation-plan.yaml",
    to: Annotated[
        str,
        typer.Option(
            "--to",
            help=(
                "'markdown' — a fix report ready for an upstream issue/PR; "
                "'json' — a neutral override list for gateways and scripts"
            ),
        ),
    ] = "markdown",
    out: Annotated[str | None, typer.Option("--out", help="Write here instead of stdout")] = None,
) -> None:
    """Export a curation plan as a shareable fix report or neutral JSON."""
    from whetkit.curation import load_plan
    from whetkit.curation.export import plan_to_json, plan_to_markdown

    if to not in ("markdown", "json"):
        raise typer.BadParameter("--to must be 'markdown' or 'json'")
    if not Path(plan).is_file():
        raise typer.BadParameter(
            f"no curation plan at {plan} — run 'whetkit curate' first, or pass --plan"
        )
    curation_plan = load_plan(plan)
    if not curation_plan.overrides:
        typer.echo("plan has no overrides — nothing to export", err=True)
        raise typer.Exit(code=1)

    rendered = (plan_to_markdown if to == "markdown" else plan_to_json)(curation_plan)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(rendered)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(rendered, nl=False)


@app.command()
def diff(
    before: Annotated[str, typer.Argument(help="Baseline --summary-json file")],
    after: Annotated[str, typer.Argument(help="Comparison --summary-json file")],
) -> None:
    """Compare two --summary-json files: headline deltas plus per-task
    transitions. The before/after table without re-running anything."""
    import json as jsonlib

    docs = []
    for path in (before, after):
        if not Path(path).is_file():
            raise typer.BadParameter(f"no summary file at {path}")
        docs.append(jsonlib.loads(Path(path).read_text()))

    def mean(doc: dict, key: str) -> float | None:
        values = [r[key] for r in doc["runs"] if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    def cell(value: float | None, fmt: str) -> str:
        # a metric absent from one file (e.g. judging was off) must render
        # as "—", not fake a "0%" measurement that never happened
        return "—" if value is None else fmt.format(value)

    typer.echo(f"{'metric':<28} {'before':>10} {'after':>10}")
    for label, key, fmt in (
        ("Hit-rate", "hit_rate", "{:.0%}"),
        ("Tool-selection hit-rate", "tool_hit_rate", "{:.0%}"),
        ("Judge pass-rate", "judge_pass_rate", "{:.0%}"),
        ("Precision (avg)", "avg_precision", "{:.0%}"),
        ("Unnecessary calls/task", "avg_extra_calls", "{:.1f}"),
        ("Tokens in (per run)", "tokens_in", "{:.0f}"),
    ):
        b, a = mean(docs[0], key), mean(docs[1], key)
        typer.echo(f"{label:<28} {cell(b, fmt):>10} {cell(a, fmt):>10}")

    def outcomes(doc: dict) -> dict[str, list[bool]]:
        per_task: dict[str, list[bool]] = {}
        for run_doc in doc["runs"]:
            for task in run_doc["tasks"]:
                per_task.setdefault(task["id"], []).append(bool(task["hit"]))
        return per_task

    before_tasks, after_tasks = outcomes(docs[0]), outcomes(docs[1])
    typer.echo("")
    for task_id in sorted(set(before_tasks) | set(after_tasks)):

        def word(hits: list[bool] | None) -> str:
            if not hits:
                return "—"
            if all(hits):
                return "PASS"
            return "MISS" if not any(hits) else f"FLAKY {sum(hits)}/{len(hits)}"

        b_word, a_word = word(before_tasks.get(task_id)), word(after_tasks.get(task_id))
        marker = "" if b_word == a_word else ("  ↑" if a_word == "PASS" else "  ↓")
        typer.echo(f"  {task_id:<28} {b_word:>10} -> {a_word:<10}{marker}")


@app.command()
def run(
    tasks: Annotated[
        str, typer.Option("--tasks", help="Task YAML file or directory of task files")
    ],
    server: Annotated[
        str | None,
        typer.Option("--server", help="Override the server every task runs against"),
    ] = None,
    model: Annotated[
        str, typer.Option("--model", help="Agent model as provider:model_id")
    ] = "anthropic:claude-sonnet-5",
    judge: Annotated[
        str,
        typer.Option(
            "--judge",
            help="LLM-judge grading: 'auto' (on when the judge API key is set), 'on', or 'off'",
        ),
    ] = "auto",
    judge_model: Annotated[
        str, typer.Option("--judge-model", help="Judge model as provider:model_id")
    ] = "anthropic:claude-sonnet-5",
    group: Annotated[
        str, typer.Option("--group", help="Label for this batch of runs in the trace store")
    ] = "baseline",
    plan: Annotated[
        str | None,
        typer.Option(
            "--plan",
            help=(
                "Eval the curated view: apply this curation plan as an overlay "
                "and score against origin tool names (for hand-tuned plans)"
            ),
        ),
    ] = None,
    match_mode: Annotated[
        str, typer.Option("--match-mode", help="Tool matching: 'order_tolerant' or 'exact'")
    ] = "order_tolerant",
    max_turns: Annotated[int, typer.Option("--max-turns")] = 10,
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            help="Completion-token budget per model turn (raise for reasoning models)",
        ),
    ] = 1024,
    runs: Annotated[
        int,
        typer.Option(
            "--runs",
            help=(
                "Repeat the whole task set N times and report mean plus range "
                "— single runs are noise. Trace groups get a -1..-N suffix."
            ),
        ),
    ] = 1,
    task_timeout: Annotated[
        float,
        typer.Option(
            "--task-timeout",
            help=(
                "Per-task wall-clock budget in seconds (provider turns plus tool "
                "calls); an expired task is flagged as timed out, not hung"
            ),
        ),
    ] = 120.0,
    store: Annotated[
        str | None,
        typer.Option("--store", help="Trace SQLite path (default ./.whetkit/traces.sqlite3)"),
    ] = None,
    jsonl: Annotated[
        str | None,
        typer.Option(
            "--jsonl",
            help="Also write traces to this JSONL file (suffixed per run when --runs > 1)",
        ),
    ] = None,
    summary_json: Annotated[
        str | None,
        typer.Option(
            "--summary-json",
            help="Write a machine-readable summary (metrics + per-task outcomes) to this path",
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            help=(
                "Run up to N tasks of a repetition in parallel (each gets its own "
                "server connection). Only safe when tasks are independent — writes "
                "to shared state should stay at 1."
            ),
        ),
    ] = 1,
    reset_cmd: Annotated[
        str | None,
        typer.Option(
            "--reset-cmd",
            help=(
                "Shell command run before each repetition (and the first run) to "
                "reset server-side fixtures — makes --runs honest on stateful servers"
            ),
        ),
    ] = None,
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Run eval tasks against an MCP server and print scored results."""
    from whetkit.datasets import load_tasks
    from whetkit.runner import RunConfig, run_task
    from whetkit.scoring import JudgeCache, JudgeConfig, MultiRunSummary, score_runs
    from whetkit.tracing import TraceStore, default_store_path, write_jsonl

    if runs < 1:
        raise typer.BadParameter("--runs must be at least 1")
    if concurrency < 1:
        raise typer.BadParameter("--concurrency must be at least 1")
    if task_timeout <= 0:
        raise typer.BadParameter("--task-timeout must be positive")
    mode = _match_mode(match_mode)

    task_list = load_tasks(tasks)
    servers = _resolve_task_servers(task_list, server, http_mode)
    config = RunConfig(
        model=model, max_turns=max_turns, max_tokens=max_tokens, task_timeout_s=task_timeout
    )
    use_judge = _judge_enabled(judge, judge_model)
    store_path = store or str(default_store_path())

    from whetkit.mcp import MCPClient

    client_factory = MCPClient
    name_map: dict[str, str] | None = None
    if plan is not None:
        from functools import partial

        from whetkit.curation import CuratedMCPClient, load_plan

        if not Path(plan).is_file():
            raise typer.BadParameter(f"no curation plan at {plan}")
        curation_plan = load_plan(plan)
        client_factory = partial(CuratedMCPClient, plan=curation_plan)
        name_map = curation_plan.rename_map()
        # Validate the plan against every origin's live tool list before
        # spending on runs: unknown targets or presented-name collisions
        # would silently eval a broken curated view.
        distinct_specs = {spec.model_dump_json(): spec for spec in servers.values()}
        for origin_spec in distinct_specs.values():
            origin_names = {t.name for t in asyncio.run(inspect_server(origin_spec)).tools}
            if problems := curation_plan.validate_against(origin_names):
                raise typer.BadParameter(
                    f"curation plan {plan} is not valid for {origin_spec.label()}: "
                    + "; ".join(problems)
                )

    async def _run_once(group_name: str, cache: JudgeCache):
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(task):
            async with semaphore:
                typer.echo(f"running {task.id} ...", err=True)
                return await run_task(
                    task, servers[task.server], config, client_factory=client_factory
                )

        task_runs = list(await asyncio.gather(*(_one(task) for task in task_list)))

        with TraceStore(store_path) as trace_store:
            trace_store.save_runs(task_runs, run_group=group_name)

        summary = await score_runs(
            task_list,
            task_runs,
            mode=mode,
            judge_config=JudgeConfig(model=judge_model),
            judge_cache=cache,
            use_judge=use_judge,
            name_map=name_map,
        )

        typer.echo(f"\nResults (group '{group_name}', model {model}):")
        for score in summary.scores:
            mark = "PASS" if score.hit else "MISS"
            tools = " -> ".join(score.tool_match.called) or "(no tool calls)"
            line = f"  [{mark}] {score.task_id}: {tools}"
            if score.tool_match.missing_slots:
                line += f"  missing={score.tool_match.missing_slots}"
            if score.judge is not None:
                line += f"  judge={'pass' if score.judge.passed else 'FAIL'}"
            typer.echo(line)
        typer.echo("")
        for score in summary.scores:
            if score.spec_gap:
                called = ", ".join(score.tool_match.called) or "(none)"
                typer.echo(
                    f"⚠ possible task-spec gap: '{score.task_id}' reached a correct "
                    f"outcome (judge passed) via tools not listed in expected_tools "
                    f"— called: {called}"
                )
        for line in summary.summary_lines():
            typer.echo(line)
        tokens_in = sum(r.total_usage.input_tokens for r in task_runs)
        tokens_out = sum(r.total_usage.output_tokens for r in task_runs)
        from whetkit.llm import parse_model
        from whetkit.llm.pricing import estimate_cost_usd

        cost = estimate_cost_usd(parse_model(model)[1], tokens_in, tokens_out)
        cost_note = f"   ≈ ${cost:.4f} (est.)" if cost is not None else ""
        typer.echo(
            f"Tokens in/out: {tokens_in}/{tokens_out}   "
            f"Total latency: {sum(r.total_latency_ms for r in task_runs) / 1000:.1f}s"
            f"{cost_note}"
        )
        return task_runs, summary

    async def _run() -> None:
        summaries = []
        payloads = []
        cache = JudgeCache(default_store_path().parent / "judge_cache.sqlite3")
        try:
            for run_index in range(1, runs + 1):
                if reset_cmd:
                    import subprocess

                    typer.echo(f"reset: {reset_cmd}", err=True)
                    proc = subprocess.run(reset_cmd, shell=True)
                    if proc.returncode != 0:
                        typer.echo(
                            f"error: --reset-cmd failed with exit code {proc.returncode}",
                            err=True,
                        )
                        raise typer.Exit(code=1)
                group_name = group if runs == 1 else f"{group}-{run_index}"
                task_runs, summary = await _run_once(group_name, cache)
                summaries.append(summary)
                payloads.append(_summary_payload(group_name, summary, task_runs))
                if jsonl:
                    write_jsonl(task_runs, jsonl if runs == 1 else f"{jsonl}.{run_index}")
        finally:
            cache.close()

        multi = MultiRunSummary(summaries=summaries)
        if runs > 1:
            typer.echo(f"\n== across {runs} runs ==")
            for line in multi.summary_lines():
                typer.echo(line)
        if judge == "auto" and not use_judge:
            typer.echo(_judge_skip_hint(judge_model))
        suffix = f" (groups '{group}-1'..'-{runs}')" if runs > 1 else f" (group '{group}')"
        typer.echo(f"Traces saved to {store_path}{suffix}")

        if summary_json:
            import json as jsonlib

            document = {
                "model": model,
                "judge_model": judge_model if use_judge else None,
                "match_mode": match_mode,
                "plan": plan,
                "runs": payloads,
            }
            if runs > 1:
                document["aggregate"] = {
                    "n": multi.n,
                    "hit_rate_mean": multi.mean_hit_rate,
                    "hit_rate_min": min(s.hit_rate for s in summaries),
                    "hit_rate_max": max(s.hit_rate for s in summaries),
                    "flaky_tasks": multi.flaky_tasks(),
                }
            Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
            Path(summary_json).write_text(jsonlib.dumps(document, indent=2) + "\n")
            typer.echo(f"Summary JSON: {summary_json}")

    asyncio.run(_run())


@app.command()
def curate(
    tasks: Annotated[
        str, typer.Option("--tasks", help="Task YAML file or directory of task files")
    ],
    server: Annotated[
        str | None,
        typer.Option("--server", help="Override the server every task runs against"),
    ] = None,
    model: Annotated[
        str, typer.Option("--model", help="Agent model as provider:model_id")
    ] = "anthropic:claude-sonnet-5",
    optimizer_model: Annotated[
        str, typer.Option("--optimizer-model", help="Curation model as provider:model_id")
    ] = "anthropic:claude-sonnet-5",
    judge: Annotated[str, typer.Option("--judge", help="'auto', 'on', or 'off'")] = "auto",
    judge_model: Annotated[str, typer.Option("--judge-model")] = "anthropic:claude-sonnet-5",
    plan_path: Annotated[
        str, typer.Option("--plan", help="Where to write the curation plan YAML")
    ] = ".whetkit/curation-plan.yaml",
    report_dir: Annotated[
        str, typer.Option("--report-dir", help="Directory for report.html + report.json")
    ] = ".whetkit",
    match_mode: Annotated[str, typer.Option("--match-mode")] = "order_tolerant",
    max_turns: Annotated[int, typer.Option("--max-turns")] = 10,
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            help="Completion-token budget per model turn (raise for reasoning models)",
        ),
    ] = 1024,
    runs: Annotated[
        int,
        typer.Option(
            "--runs",
            help=(
                "Repeat the baseline and curated evals N times and report the "
                "mean hit-rate plus range — single runs are noise. The "
                "optimizer's proposal is built from the FIRST baseline run's "
                "traces; the report's per-task detail shows the last run. "
                "Trace groups get a -1..-N suffix."
            ),
        ),
    ] = 1,
    task_timeout: Annotated[
        float,
        typer.Option(
            "--task-timeout",
            help="Per-task wall-clock budget in seconds (provider turns plus tool calls)",
        ),
    ] = 120.0,
    prune_unused: Annotated[
        bool,
        typer.Option(
            "--prune-unused",
            help=(
                "Additionally hide every tool the eval never touched — the cost "
                "play for big servers. Only sound when the tasks cover all the "
                "workflows the curated view will serve."
            ),
        ),
    ] = False,
    store: Annotated[str | None, typer.Option("--store")] = None,
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Baseline-eval the server, propose a curation overlay, re-eval through
    it, and show the before/after hit-rate."""
    from whetkit.curation import CuratedMCPClient, propose_plan, save_plan
    from whetkit.curation.optimizer import OptimizerConfig
    from whetkit.curation.optimizer import prune_unused as apply_prune_unused
    from whetkit.datasets import load_tasks
    from whetkit.mcp import MCPClient, inspect_server
    from whetkit.runner import RunConfig
    from whetkit.scoring import JudgeCache, JudgeConfig, MultiRunSummary, score_runs
    from whetkit.tracing import default_store_path

    if runs < 1:
        raise typer.BadParameter("--runs must be at least 1")
    if task_timeout <= 0:
        raise typer.BadParameter("--task-timeout must be positive")
    mode = _match_mode(match_mode)
    task_list = load_tasks(tasks)
    servers = _resolve_task_servers(task_list, server, http_mode)
    curation_spec = _single_server_spec(servers, "curate")
    config = RunConfig(
        model=model, max_turns=max_turns, max_tokens=max_tokens, task_timeout_s=task_timeout
    )
    use_judge = _judge_enabled(judge, judge_model)
    judge_config = JudgeConfig(model=judge_model)
    store_path = store or str(default_store_path())

    async def _curate() -> None:
        cache = JudgeCache(default_store_path().parent / "judge_cache.sqlite3")

        async def _score(task_runs, name_map=None):
            return await score_runs(
                task_list,
                task_runs,
                mode=mode,
                judge_config=judge_config,
                judge_cache=cache,
                use_judge=use_judge,
                name_map=name_map,
            )

        try:
            typer.echo("== baseline eval ==", err=True)
            baseline_all, baseline_summaries = await _eval_repeated(
                task_list,
                servers,
                config,
                runs=runs,
                base_group="baseline",
                store_path=store_path,
                client_factory=MCPClient,
                score_one=_score,
            )

            typer.echo("== proposing curation plan ==", err=True)
            inventory = await inspect_server(curation_spec)
            # The proposal is built from the FIRST baseline run's traces (see
            # --runs help); further repetitions only measure variance.
            plan, warnings = await propose_plan(
                inventory,
                task_list,
                baseline_all[0],
                baseline_summaries[0].scores,
                OptimizerConfig(model=optimizer_model),
            )
            for warning in warnings:
                typer.echo(f"warning: {warning}", err=True)
            if prune_unused:
                every_baseline_run = [r for rep in baseline_all for r in rep]
                pruned = apply_prune_unused(plan, inventory, task_list, every_baseline_run)
                typer.echo(f"--prune-unused: hid {pruned} untouched tool(s)", err=True)
            save_plan(plan, plan_path)
            typer.echo(f"curation plan written to {plan_path}", err=True)

            typer.echo("== curated eval (through overlay) ==", err=True)
            curated_all, curated_summaries = await _eval_repeated(
                task_list,
                servers,
                config,
                runs=runs,
                base_group="curated",
                store_path=store_path,
                client_factory=lambda spec: CuratedMCPClient(spec, plan),
                score_one=lambda task_runs: _score(task_runs, name_map=plan.rename_map()),
            )
        finally:
            cache.close()

        from whetkit.report import build_report

        baseline_multi = MultiRunSummary(summaries=baseline_summaries)
        curated_multi = MultiRunSummary(summaries=curated_summaries)
        # Per-task detail (report + table) shows the LAST repetition; the
        # headline mean/range strings carry the cross-run picture.
        baseline_runs, baseline = baseline_all[-1], baseline_summaries[-1]
        curated_runs, curated = curated_all[-1], curated_summaries[-1]

        origin_names = {t.name for t in inventory.tools}
        report = build_report(
            task_list,
            baseline_runs,
            baseline,
            curated_runs,
            curated,
            plan,
            model=model,
            server=curation_spec.label(),
            tools_before=inventory.tool_count,
            tools_after=len(plan.presented_to_original(origin_names)),
            before_spread=baseline_multi.hit_rate_spread() if runs > 1 else None,
            after_spread=curated_multi.hit_rate_spread() if runs > 1 else None,
        )
        html_path, json_path = _write_report(report, report_dir)

        typer.echo("\n== before/after ==")
        if runs > 1:
            typer.echo(f"(per-task view shows the last of {runs} runs)")
        typer.echo(f"{'task':<28} {'before':<8} {'after':<8}")
        curated_by_id = {s.task_id: s for s in curated.scores}
        for score in baseline.scores:
            after = curated_by_id.get(score.task_id)
            typer.echo(
                f"{score.task_id:<28} "
                f"{'PASS' if score.hit else 'MISS':<8} "
                f"{('PASS' if after.hit else 'MISS') if after else '?':<8}"
            )
        typer.echo("")
        typer.echo(
            f"Hit-rate: {baseline_multi.hit_rate_spread()} -> "
            f"{curated_multi.hit_rate_spread()}   "
            f"Tool hit-rate: {baseline_multi.tool_hit_rate_spread()} -> "
            f"{curated_multi.tool_hit_rate_spread()}   "
            f"Precision: {baseline_multi.avg_precision_spread()} -> "
            f"{curated_multi.avg_precision_spread()}"
        )
        tok_before = (report.before.input_tokens + report.before.output_tokens) // len(task_list)
        tok_after = (report.after.input_tokens + report.after.output_tokens) // len(task_list)
        typer.echo(
            f"Tools: {report.tools_before} -> {report.tools_after}   "
            f"Tokens/task: {tok_before} -> {tok_after}"
        )
        typer.echo(
            f"Traces saved to {store_path} (groups {_group_family_note('baseline', runs)}, "
            f"{_group_family_note('curated', runs)})"
        )
        typer.echo(f"Report: {html_path} (machine-readable: {json_path})")

    asyncio.run(_curate())


@app.command()
def fix(
    tasks: Annotated[
        str, typer.Option("--tasks", help="Task YAML file or directory of task files")
    ],
    server: Annotated[
        str | None, typer.Option("--server", help="Override the server every task runs against")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", help="Agent model as provider:model_id")
    ] = "anthropic:claude-sonnet-5",
    optimizer_model: Annotated[
        str, typer.Option("--optimizer-model")
    ] = "anthropic:claude-sonnet-5",
    judge: Annotated[str, typer.Option("--judge", help="'auto', 'on', or 'off'")] = "auto",
    judge_model: Annotated[str, typer.Option("--judge-model")] = "anthropic:claude-sonnet-5",
    max_iterations: Annotated[
        int, typer.Option("--max-iterations", help="Propose→eval→revise rounds (≥1)")
    ] = 3,
    plan_path: Annotated[
        str, typer.Option("--plan", help="Where to write the best plan YAML")
    ] = ".whetkit/curation-plan.yaml",
    match_mode: Annotated[str, typer.Option("--match-mode")] = "order_tolerant",
    max_turns: Annotated[int, typer.Option("--max-turns")] = 10,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 1024,
    runs: Annotated[
        int,
        typer.Option(
            "--runs",
            help=(
                "Repeat the baseline and each iteration's curated eval N times; "
                "iteration decisions and the final report use the mean plus "
                "range. The optimizer sees the FIRST repetition's traces. "
                "Trace groups get a -1..-N suffix."
            ),
        ),
    ] = 1,
    task_timeout: Annotated[
        float,
        typer.Option(
            "--task-timeout",
            help="Per-task wall-clock budget in seconds (provider turns plus tool calls)",
        ),
    ] = 120.0,
    store: Annotated[str | None, typer.Option("--store")] = None,
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Self-correcting curation: propose a plan, eval through it, feed the
    regressions and remaining waste back to the optimizer, revise — up to
    --max-iterations — and keep the best plan by measured results."""
    from functools import partial

    from whetkit.curation import CuratedMCPClient, propose_plan, save_plan
    from whetkit.curation.optimizer import OptimizerConfig, propose_revision
    from whetkit.datasets import load_tasks
    from whetkit.runner import RunConfig
    from whetkit.scoring import JudgeCache, JudgeConfig, MultiRunSummary, score_runs
    from whetkit.tracing import default_store_path

    if max_iterations < 1:
        raise typer.BadParameter("--max-iterations must be at least 1")
    if runs < 1:
        raise typer.BadParameter("--runs must be at least 1")
    if task_timeout <= 0:
        raise typer.BadParameter("--task-timeout must be positive")
    mode = _match_mode(match_mode)
    task_list = load_tasks(tasks)
    servers = _resolve_task_servers(task_list, server, http_mode)
    curation_spec = _single_server_spec(servers, "fix")
    config = RunConfig(
        model=model, max_turns=max_turns, max_tokens=max_tokens, task_timeout_s=task_timeout
    )
    use_judge = _judge_enabled(judge, judge_model)
    judge_config = JudgeConfig(model=judge_model)
    store_path = store or str(default_store_path())
    optimizer_config = OptimizerConfig(model=optimizer_model)

    from whetkit.mcp import MCPClient

    async def _fix() -> None:
        from whetkit.mcp import inspect_server as _inspect

        cache = JudgeCache(default_store_path().parent / "judge_cache.sqlite3")

        async def _score(task_runs, name_map=None):
            return await score_runs(
                task_list,
                task_runs,
                mode=mode,
                judge_config=judge_config,
                judge_cache=cache,
                use_judge=use_judge,
                name_map=name_map,
            )

        async def _eval(group: str, client_factory=MCPClient, name_map=None):
            return await _eval_repeated(
                task_list,
                servers,
                config,
                runs=runs,
                base_group=group,
                store_path=store_path,
                client_factory=client_factory,
                score_one=lambda task_runs: _score(task_runs, name_map=name_map),
            )

        try:
            typer.echo("== baseline ==", err=True)
            baseline_all, baseline_summaries = await _eval("baseline")
            baseline_multi = MultiRunSummary(summaries=baseline_summaries)
            inventory = await _inspect(curation_spec)

            typer.echo("== proposing plan (iteration 1) ==", err=True)
            # The optimizer sees the FIRST repetition's traces (see --runs help).
            plan, warnings = await propose_plan(
                inventory,
                task_list,
                baseline_all[0],
                baseline_summaries[0].scores,
                optimizer_config,
            )
            for w in warnings:
                typer.echo(f"warning: {w}", err=True)

            def metric(multi):
                return (multi.mean_hit_rate, -multi.mean_avg_extra_calls, multi.mean_avg_precision)

            best_plan, best_multi = None, None
            for iteration in range(1, max_iterations + 1):
                typer.echo(f"== eval through plan (iteration {iteration}) ==", err=True)
                curated_all, curated_summaries = await _eval(
                    f"fix-{iteration}",
                    client_factory=partial(CuratedMCPClient, plan=plan),
                    name_map=plan.rename_map(),
                )
                curated_multi = MultiRunSummary(summaries=curated_summaries)
                typer.echo(
                    f"iteration {iteration}: hit {curated_multi.hit_rate_spread()} "
                    f"(baseline {baseline_multi.hit_rate_spread()}), "
                    f"extra calls {curated_multi.mean_avg_extra_calls:.1f}/task",
                    err=True,
                )
                if best_multi is None or metric(curated_multi) > metric(best_multi):
                    best_plan, best_multi = plan, curated_multi

                converged = (
                    curated_multi.mean_hit_rate >= baseline_multi.mean_hit_rate
                    and curated_multi.mean_avg_extra_calls <= 0.25
                )
                if converged or iteration == max_iterations:
                    if converged:
                        typer.echo("converged — no regressions, negligible waste", err=True)
                    break

                typer.echo("== revising plan ==", err=True)
                plan, warnings = await propose_revision(
                    plan,
                    inventory,
                    task_list,
                    baseline_all[0],
                    baseline_summaries[0].scores,
                    curated_all[0],
                    curated_summaries[0].scores,
                    optimizer_config,
                )
                for w in warnings:
                    typer.echo(f"warning: {w}", err=True)
        finally:
            cache.close()

        save_plan(best_plan, plan_path)
        typer.echo(f"\nbest plan (of {max_iterations} max iterations) -> {plan_path}")
        typer.echo(
            f"Hit-rate: {baseline_multi.hit_rate_spread()} -> {best_multi.hit_rate_spread()}   "
            f"Extra calls: {baseline_multi.mean_avg_extra_calls:.1f} -> "
            f"{best_multi.mean_avg_extra_calls:.1f}/task"
        )
        typer.echo(f"serve it:  whetkit overlay --server <origin> --plan {plan_path}")

    asyncio.run(_fix())


@app.command()
def report(
    tasks: Annotated[
        str, typer.Option("--tasks", help="Task YAML file or directory of task files")
    ],
    plan: Annotated[
        str, typer.Option("--plan", help="Curation plan YAML used for the 'after' runs")
    ] = ".whetkit/curation-plan.yaml",
    before: Annotated[str, typer.Option("--before", help="Trace group for baseline")] = "baseline",
    after: Annotated[str, typer.Option("--after", help="Trace group for curated")] = "curated",
    judge: Annotated[str, typer.Option("--judge", help="'auto', 'on', or 'off'")] = "auto",
    judge_model: Annotated[str, typer.Option("--judge-model")] = "anthropic:claude-sonnet-5",
    match_mode: Annotated[str, typer.Option("--match-mode")] = "order_tolerant",
    store: Annotated[str | None, typer.Option("--store")] = None,
    out: Annotated[str, typer.Option("--out", help="Output directory")] = ".whetkit",
) -> None:
    """Rebuild the before/after report from stored traces (judge verdicts
    come from the cache when available)."""
    from whetkit.curation import load_plan
    from whetkit.datasets import load_tasks
    from whetkit.report import build_report
    from whetkit.scoring import JudgeCache, JudgeConfig, score_runs
    from whetkit.tracing import TraceStore, default_store_path

    mode = _match_mode(match_mode)
    task_list = load_tasks(tasks)
    if not Path(plan).is_file():
        raise typer.BadParameter(
            f"no curation plan at {plan} — run 'whetkit curate' first, or pass --plan"
        )
    curation_plan = load_plan(plan)
    store_path = store or str(default_store_path())
    use_judge = _judge_enabled(judge, judge_model)

    with TraceStore(store_path) as trace_store:
        before_runs, before_dropped = trace_store.latest_runs_per_task(before)
        after_runs, after_dropped = trace_store.latest_runs_per_task(after)
    if not before_runs or not after_runs:
        raise typer.BadParameter(
            f"trace store {store_path} has no runs for groups "
            f"{before!r} and/or {after!r} — run 'whetkit curate' first"
        )
    for group_name, dropped in ((before, before_dropped), (after, after_dropped)):
        if dropped:
            typer.echo(
                f"warning: group '{group_name}' holds {dropped} older run(s) per task "
                "from previous invocations — using the most recent run per task",
                err=True,
            )

    async def _report() -> None:
        cache = JudgeCache(default_store_path().parent / "judge_cache.sqlite3")
        try:
            judge_config = JudgeConfig(model=judge_model)
            before_summary = await score_runs(
                task_list,
                before_runs,
                mode=mode,
                judge_config=judge_config,
                judge_cache=cache,
                use_judge=use_judge,
            )
            after_summary = await score_runs(
                task_list,
                after_runs,
                mode=mode,
                judge_config=judge_config,
                judge_cache=cache,
                use_judge=use_judge,
                name_map=curation_plan.rename_map(),
            )
        finally:
            cache.close()

        comparison = build_report(
            task_list,
            before_runs,
            before_summary,
            after_runs,
            after_summary,
            curation_plan,
            model=before_runs[0].model if before_runs else "",
            server=before_runs[0].server if before_runs else "",
        )
        html_path, json_path = _write_report(comparison, out)
        typer.echo(
            f"Hit-rate: {before_summary.hit_rate:.0%} -> {after_summary.hit_rate:.0%} "
            f"({len(comparison.improved)} improved, {len(comparison.regressed)} regressed)"
        )
        typer.echo(f"Report: {html_path} (machine-readable: {json_path})")

    asyncio.run(_report())


@app.command()
def overlay(
    server: Annotated[
        str, typer.Option("--server", help="Origin MCP server: URL, directory, or file path")
    ],
    plan: Annotated[str, typer.Option("--plan", help="Curation plan YAML to apply")],
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Serve the curated view of a server as a stdio MCP server (reversible:
    the origin is never modified)."""
    from whetkit.curation import load_plan, serve_overlay
    from whetkit.curation.overlay import InvalidPlanError

    origin = _resolve_server(server, http_mode)
    try:
        asyncio.run(serve_overlay(origin, load_plan(plan)))
    except InvalidPlanError as exc:
        typer.echo(f"error: curation plan is not valid for this origin: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
