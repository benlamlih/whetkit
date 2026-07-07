"""Command-line entry point for mcp-eval."""

import asyncio
import os
from typing import Annotated

import typer

from mcp_eval.mcp import HttpMode, ServerSpec, inspect_server, resolve_server_spec

app = typer.Typer(
    name="mcp-eval",
    help="Evaluate and improve LLM agent tool selection on MCP servers.",
    no_args_is_help=True,
)


def _judge_enabled(judge: str, judge_model: str) -> bool:
    """--judge auto: grade with the LLM judge only when its API key is set."""
    if judge in ("on", "off"):
        return judge == "on"
    from mcp_eval.llm import parse_model

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
        typer.Option("--store", help="Trace SQLite path (default ./.mcp-eval/traces.sqlite3)"),
    ] = None,
    jsonl: Annotated[
        str | None, typer.Option("--jsonl", help="Also write traces to this JSONL file")
    ] = None,
    http_mode: Annotated[HttpMode, typer.Option("--http-mode")] = HttpMode.STATEFUL,
) -> None:
    """Run eval tasks against an MCP server and print scored results."""
    from mcp_eval.datasets import load_tasks
    from mcp_eval.runner import RunConfig, run_task
    from mcp_eval.scoring import JudgeCache, JudgeConfig, MatchMode, score_runs
    from mcp_eval.tracing import TraceStore, default_store_path, write_jsonl

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


if __name__ == "__main__":
    app()
