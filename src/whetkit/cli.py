"""Command-line entry point for whetkit."""

import asyncio
import os
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
    from pathlib import Path

    from whetkit.report import render_html

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "report.html"
    json_path = out / "report.json"
    html_path.write_text(render_html(report))
    json_path.write_text(report.model_dump_json(indent=2))
    return str(html_path), str(json_path)


@app.callback()
def main() -> None:
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
    match_mode: Annotated[
        str, typer.Option("--match-mode", help="Tool matching: 'order_tolerant' or 'exact'")
    ] = "order_tolerant",
    max_turns: Annotated[int, typer.Option("--max-turns")] = 10,
    store: Annotated[
        str | None,
        typer.Option("--store", help="Trace SQLite path (default ./.whetkit/traces.sqlite3)"),
    ] = None,
    jsonl: Annotated[
        str | None, typer.Option("--jsonl", help="Also write traces to this JSONL file")
    ] = None,
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Run eval tasks against an MCP server and print scored results."""
    from whetkit.datasets import load_tasks
    from whetkit.runner import RunConfig, run_task
    from whetkit.scoring import JudgeCache, JudgeConfig, MatchMode, score_runs
    from whetkit.tracing import TraceStore, default_store_path, write_jsonl

    task_list = load_tasks(tasks)
    servers = _resolve_task_servers(task_list, server, http_mode)
    config = RunConfig(model=model, max_turns=max_turns)
    use_judge = _judge_enabled(judge, judge_model)
    store_path = store or str(default_store_path())

    async def _run() -> None:
        runs = []
        for task in task_list:
            typer.echo(f"running {task.id} ...", err=True)
            runs.append(await run_task(task, servers[task.server], config))

        with TraceStore(store_path) as trace_store:
            trace_store.save_runs(runs, run_group=group)
        if jsonl:
            write_jsonl(runs, jsonl)

        cache = JudgeCache(default_store_path().parent / "judge_cache.sqlite3")
        try:
            summary = await score_runs(
                task_list,
                runs,
                mode=MatchMode(match_mode),
                judge_config=JudgeConfig(model=judge_model),
                judge_cache=cache,
                use_judge=use_judge,
            )
        finally:
            cache.close()

        typer.echo(f"\nResults (group '{group}', model {model}):")
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
        if not use_judge:
            typer.echo(
                "(LLM judge skipped: no API key found — set ANTHROPIC_API_KEY or pass --judge on)"
            )
        typer.echo(f"Traces saved to {store_path} (group '{group}')")

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
    config = RunConfig(model=model, max_turns=max_turns)
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
            )
        finally:
            cache.close()

        with TraceStore(store_path) as trace_store:
            trace_store.save_runs(baseline_runs, run_group="baseline")
            trace_store.save_runs(curated_runs, run_group="curated")

        from whetkit.report import build_report

        report = build_report(
            task_list,
            baseline_runs,
            baseline,
            curated_runs,
            curated,
            plan,
            model=model,
            server=next(iter(servers.values())).label(),
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
