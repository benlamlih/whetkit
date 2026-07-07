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


def _judge_enabled(judge: str, judge_model: str) -> bool:
    """--judge auto: grade with the LLM judge only when its API key is set."""
    if judge in ("on", "off"):
        return judge == "on"
    from whetkit.llm import parse_model

    provider_name, _ = parse_model(judge_model)
    key_var = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider_name)
    return bool(key_var and os.environ.get(key_var))


def _resolve_task_servers(
    tasks: list, server_override: str | None, http_mode: HttpMode
) -> dict[str, ServerSpec]:
    if server_override is not None:
        spec = resolve_server_spec(server_override, http_mode=http_mode)
        return {task.server: spec for task in tasks}
    return {
        task.server: resolve_server_spec(task.server, http_mode=http_mode)
        for task in {t.server: t for t in tasks}.values()
    }


def _write_report(report, out_dir: str) -> tuple[str, str]:
    from whetkit.report import render_html

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "report.html"
    json_path = out / "report.json"
    html_path.write_text(render_html(report))
    json_path.write_text(report.model_dump_json(indent=2))
    return str(html_path), str(json_path)


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
    spec = resolve_server_spec(server, http_mode=http_mode)
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

    spec = resolve_server_spec(server, http_mode=http_mode)
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
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Draft eval tasks from the server's tool inventory (review before
    trusting — the header in the output file says the same)."""
    from whetkit.generate import GeneratorConfig, generate_tasks, write_tasks_yaml

    spec = resolve_server_spec(server, http_mode=http_mode)
    inventory = asyncio.run(inspect_server(spec))
    tasks, warnings = asyncio.run(
        generate_tasks(inventory, server, count=count, config=GeneratorConfig(model=model))
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
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Run eval tasks against an MCP server and print scored results."""
    from whetkit.datasets import load_tasks
    from whetkit.runner import RunConfig, run_task
    from whetkit.scoring import JudgeCache, JudgeConfig, MatchMode, MultiRunSummary, score_runs
    from whetkit.tracing import TraceStore, default_store_path, write_jsonl

    if runs < 1:
        raise typer.BadParameter("--runs must be at least 1")

    task_list = load_tasks(tasks)
    servers = _resolve_task_servers(task_list, server, http_mode)
    config = RunConfig(model=model, max_turns=max_turns, max_tokens=max_tokens)
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

    async def _run_once(group_name: str, cache: JudgeCache):
        task_runs = []
        for task in task_list:
            typer.echo(f"running {task.id} ...", err=True)
            task_runs.append(
                await run_task(task, servers[task.server], config, client_factory=client_factory)
            )

        with TraceStore(store_path) as trace_store:
            trace_store.save_runs(task_runs, run_group=group_name)

        summary = await score_runs(
            task_list,
            task_runs,
            mode=MatchMode(match_mode),
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
        for line in summary.summary_lines():
            typer.echo(line)
        tokens_in = sum(r.total_usage.input_tokens for r in task_runs)
        tokens_out = sum(r.total_usage.output_tokens for r in task_runs)
        typer.echo(
            f"Tokens in/out: {tokens_in}/{tokens_out}   "
            f"Total latency: {sum(r.total_latency_ms for r in task_runs) / 1000:.1f}s"
        )
        return task_runs, summary

    async def _run() -> None:
        summaries = []
        cache = JudgeCache(default_store_path().parent / "judge_cache.sqlite3")
        try:
            for run_index in range(1, runs + 1):
                group_name = group if runs == 1 else f"{group}-{run_index}"
                task_runs, summary = await _run_once(group_name, cache)
                summaries.append(summary)
                if jsonl:
                    write_jsonl(task_runs, jsonl if runs == 1 else f"{jsonl}.{run_index}")
        finally:
            cache.close()

        if runs > 1:
            typer.echo(f"\n== across {runs} runs ==")
            for line in MultiRunSummary(summaries=summaries).summary_lines():
                typer.echo(line)
        if judge == "auto" and not use_judge:
            typer.echo(
                "(LLM judge skipped: no API key found — set ANTHROPIC_API_KEY or pass --judge on)"
            )
        suffix = f" (groups '{group}-1'..'-{runs}')" if runs > 1 else f" (group '{group}')"
        typer.echo(f"Traces saved to {store_path}{suffix}")

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
    store: Annotated[str | None, typer.Option("--store")] = None,
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Baseline-eval the server, propose a curation overlay, re-eval through
    it, and show the before/after hit-rate."""
    from whetkit.curation import CuratedMCPClient, propose_plan, save_plan
    from whetkit.curation.optimizer import OptimizerConfig
    from whetkit.datasets import load_tasks
    from whetkit.mcp import inspect_server
    from whetkit.runner import RunConfig, run_task
    from whetkit.scoring import JudgeCache, JudgeConfig, MatchMode, score_runs
    from whetkit.tracing import TraceStore, default_store_path

    task_list = load_tasks(tasks)
    servers = _resolve_task_servers(task_list, server, http_mode)
    config = RunConfig(model=model, max_turns=max_turns, max_tokens=max_tokens)
    use_judge = _judge_enabled(judge, judge_model)
    judge_config = JudgeConfig(model=judge_model)
    mode = MatchMode(match_mode)
    store_path = store or str(default_store_path())

    async def _curate() -> None:
        cache = JudgeCache(default_store_path().parent / "judge_cache.sqlite3")
        try:
            typer.echo("== baseline eval ==", err=True)
            baseline_runs = []
            for task in task_list:
                typer.echo(f"running {task.id} ...", err=True)
                baseline_runs.append(await run_task(task, servers[task.server], config))
            baseline = await score_runs(
                task_list,
                baseline_runs,
                mode=mode,
                judge_config=judge_config,
                judge_cache=cache,
                use_judge=use_judge,
            )

            typer.echo("== proposing curation plan ==", err=True)
            inventory = await inspect_server(next(iter(servers.values())))
            plan, warnings = await propose_plan(
                inventory,
                task_list,
                baseline_runs,
                baseline.scores,
                OptimizerConfig(model=optimizer_model),
            )
            for warning in warnings:
                typer.echo(f"warning: {warning}", err=True)
            save_plan(plan, plan_path)
            typer.echo(f"curation plan written to {plan_path}", err=True)

            typer.echo("== curated eval (through overlay) ==", err=True)
            curated_runs = []
            for task in task_list:
                typer.echo(f"running {task.id} ...", err=True)
                curated_runs.append(
                    await run_task(
                        task,
                        servers[task.server],
                        config,
                        client_factory=lambda spec: CuratedMCPClient(spec, plan),
                    )
                )
            curated = await score_runs(
                task_list,
                curated_runs,
                mode=mode,
                judge_config=judge_config,
                judge_cache=cache,
                use_judge=use_judge,
                name_map=plan.rename_map(),
            )
        finally:
            cache.close()

        with TraceStore(store_path) as trace_store:
            trace_store.save_runs(baseline_runs, run_group="baseline")
            trace_store.save_runs(curated_runs, run_group="curated")

        from whetkit.report import build_report

        origin_names = {t.name for t in inventory.tools}
        report = build_report(
            task_list,
            baseline_runs,
            baseline,
            curated_runs,
            curated,
            plan,
            model=model,
            server=next(iter(servers.values())).label(),
            tools_before=inventory.tool_count,
            tools_after=len(plan.presented_to_original(origin_names)),
        )
        html_path, json_path = _write_report(report, report_dir)

        typer.echo("\n== before/after ==")
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
            f"Hit-rate: {baseline.hit_rate:.0%} -> {curated.hit_rate:.0%}   "
            f"Tool hit-rate: {baseline.tool_hit_rate:.0%} -> {curated.tool_hit_rate:.0%}   "
            f"Precision: {baseline.avg_precision:.0%} -> {curated.avg_precision:.0%}"
        )
        tok_before = (report.before.input_tokens + report.before.output_tokens) // len(task_list)
        tok_after = (report.after.input_tokens + report.after.output_tokens) // len(task_list)
        typer.echo(
            f"Tools: {report.tools_before} -> {report.tools_after}   "
            f"Tokens/task: {tok_before} -> {tok_after}"
        )
        typer.echo(f"Traces saved to {store_path} (groups 'baseline', 'curated')")
        typer.echo(f"Report: {html_path} (machine-readable: {json_path})")

    asyncio.run(_curate())


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
    from whetkit.scoring import JudgeCache, JudgeConfig, MatchMode, score_runs
    from whetkit.tracing import TraceStore, default_store_path

    task_list = load_tasks(tasks)
    if not Path(plan).is_file():
        raise typer.BadParameter(
            f"no curation plan at {plan} — run 'whetkit curate' first, or pass --plan"
        )
    curation_plan = load_plan(plan)
    store_path = store or str(default_store_path())
    use_judge = _judge_enabled(judge, judge_model)

    with TraceStore(store_path) as trace_store:
        before_runs = trace_store.load_runs(before)
        after_runs = trace_store.load_runs(after)
    if not before_runs or not after_runs:
        raise typer.BadParameter(
            f"trace store {store_path} has no runs for groups "
            f"{before!r} and/or {after!r} — run 'whetkit curate' first"
        )

    async def _report() -> None:
        cache = JudgeCache(default_store_path().parent / "judge_cache.sqlite3")
        try:
            judge_config = JudgeConfig(model=judge_model)
            mode = MatchMode(match_mode)
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

    origin = resolve_server_spec(server, http_mode=http_mode)
    asyncio.run(serve_overlay(origin, load_plan(plan)))


if __name__ == "__main__":
    app()
