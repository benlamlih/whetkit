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


def _load_tasks(path: str):
    """load_tasks with fat-finger errors (missing path, invalid YAML, schema
    violations) rendered as clean CLI errors instead of tracebacks."""
    import yaml

    from whetkit.datasets import load_tasks

    try:
        return load_tasks(path)
    except ValueError as exc:
        # load_tasks messages already name the offending file
        raise typer.BadParameter(str(exc)) from exc
    except (OSError, yaml.YAMLError) as exc:
        raise typer.BadParameter(f"cannot read tasks from {path}: {exc}") from exc


def _load_plan(path: str):
    """load_plan with YAML/schema/read errors rendered as clean CLI errors."""
    import yaml

    from whetkit.curation import load_plan

    try:
        return load_plan(path)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        raise typer.BadParameter(f"cannot load curation plan {path}: {exc}") from exc


_PROVIDER_KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def _provider_key_var(model: str) -> str | None:
    """The env var that holds this model's provider API key (None if unknown)."""
    from whetkit.llm import parse_model

    try:
        provider_name, _ = parse_model(model)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return _PROVIDER_KEY_VARS.get(provider_name)


def _require_provider_keys(models: dict[str, str]) -> None:
    """Fail fast — before any eval spend — when a provider API key is missing.

    ``models`` maps a role label ('agent model', 'judge model', ...) to its
    provider:model string. A missing key today surfaces only after paid/timed
    runs as 0% scores; refusing up front is the honest behavior.
    """
    for role, model in models.items():
        key_var = _provider_key_var(model)
        if key_var and not os.environ.get(key_var):
            raise typer.BadParameter(
                f"{key_var} is not set — required by the {role} {model!r}. "
                f"Export {key_var} (or pick a different provider)."
            )


def _judge_key_var(judge_model: str) -> str | None:
    """The env var that holds the judge provider's API key."""
    return _provider_key_var(judge_model)


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


def _out_path(value: str) -> Path:
    """User-supplied output path with ``~`` expanded — a literal '~' directory
    in the CWD is never what anyone meant."""
    return Path(value).expanduser()


def _write_report(report, out_dir: str) -> tuple[str, str]:
    from whetkit.report import render_html

    out = _out_path(out_dir)
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


def _exit_on_errored_runs(summaries: list) -> None:
    """Exit 3 when any task run ERRORed or TIMEOUTed: those scores measure the
    infrastructure, not tool selection, and CI must not stay green on them.
    Called after the full summary has been printed."""
    failed = sum(s.error_run_count + s.timeout_run_count for s in summaries)
    if failed:
        typer.echo(
            f"{failed} task run(s) errored — exit 3 "
            "(infrastructure failure, not a tool-selection result)",
            err=True,
        )
        raise typer.Exit(code=3)


def _echo_run_errors(task_runs: list) -> None:
    """Surface each failed run's real error (it otherwise hides in the trace DB)."""
    for task_run in task_runs:
        if task_run.error:
            typer.echo(f"  ⚠ {task_run.task_id}: {task_run.error}", err=True)


def _task_required_tools(task_list: list) -> dict[str, str]:
    """Every tool named in any task's expected_tools, mapped to the first
    task id that requires it."""
    required: dict[str, str] = {}
    for task in task_list:
        for slot in task.expected_tool_slots:
            for name in slot:
                required.setdefault(name, task.id)
    return required


def _drop_task_breaking_hides(plan, task_list: list) -> list[str]:
    """Remove hidden overrides that would break the task set.

    An optimizer that hides a tool some task's expected_tools require
    guarantees the regression it then measures — a deterministic cross-check
    catches it before the curated eval spends anything. Renames are fine
    (rename_map already handles scoring). Returns the warnings to print.
    """
    required = _task_required_tools(task_list)
    warnings: list[str] = []
    kept = []
    for override in plan.overrides:
        if override.hidden and override.original_name in required:
            warnings.append(
                f"kept {override.original_name}: required by task "
                f"{required[override.original_name]} (dropped the plan's hide)"
            )
            continue
        kept.append(override)
    plan.overrides = kept
    return warnings


def _warn_plan_hides_required_tools(plan, task_list: list) -> None:
    """run --plan keeps the user's plan untouched, but a plan that hides a
    tool the tasks require deserves a loud heads-up before the eval runs."""
    required = _task_required_tools(task_list)
    for override in plan.overrides:
        if override.hidden and override.original_name in required:
            typer.echo(
                f"warning: plan hides {override.original_name!r} but task "
                f"{required[override.original_name]!r} expects it — that task "
                "can only miss through this view",
                err=True,
            )


def _est_cost(model: str, task_runs: list) -> float | None:
    """Estimated $ of these runs (None when the model isn't in the table)."""
    from whetkit.llm import parse_model
    from whetkit.llm.pricing import estimate_cost_usd

    return estimate_cost_usd(
        parse_model(model)[1],
        sum(r.total_usage.input_tokens for r in task_runs),
        sum(r.total_usage.output_tokens for r in task_runs),
    )


def _usage_cost_line(model: str, task_runs: list) -> str:
    """Token totals, wall-clock latency, and the estimated $ cost of the runs."""
    from whetkit.llm import parse_model
    from whetkit.llm.pricing import estimate_cost_usd

    tokens_in = sum(r.total_usage.input_tokens for r in task_runs)
    tokens_out = sum(r.total_usage.output_tokens for r in task_runs)
    cost = estimate_cost_usd(parse_model(model)[1], tokens_in, tokens_out)
    cost_note = f"   ≈ ${cost:.4f} (est.)" if cost is not None else ""
    return (
        f"Tokens in/out: {tokens_in}/{tokens_out}   "
        f"Total latency: {sum(r.total_latency_ms for r in task_runs) / 1000:.1f}s"
        f"{cost_note}"
    )


_CLOUD_WAITLIST_LINE = (
    "☁ whetkit Cloud (hosted history, team dashboards, CI gating) — join the "
    "waitlist: https://github.com/benlamlih/whetkit/issues/new?template=cloud-waitlist.yml"
)


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"whetkit {version('whetkit')}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
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
    # Opt-in anonymous telemetry: one event naming the command (nothing else),
    # only when the user enabled it. See `whetkit telemetry status`.
    if ctx.invoked_subcommand and ctx.invoked_subcommand != "telemetry":
        from whetkit import telemetry as _telemetry

        _telemetry.record_event(ctx.invoked_subcommand)


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

    _require_provider_keys({"generator model": model})
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

    _out_path(out).parent.mkdir(parents=True, exist_ok=True)
    write_tasks_yaml(tasks, str(_out_path(out)))
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

    keep_set = {name.strip() for name in keep.split(",") if name.strip()}
    if from_tasks:
        for task in _load_tasks(from_tasks):
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
    _out_path(out).parent.mkdir(parents=True, exist_ok=True)
    save_plan(plan, str(_out_path(out)))
    typer.echo(f"kept {len(names & keep_set)}, hidden {len(hidden)} — wrote {out}")
    typer.echo(f"score it:  whetkit run --server {server} --tasks <tasks> --plan {out}")


@app.command()
def slim(
    config: Annotated[
        str,
        typer.Option(
            "--config",
            help=(
                "MCP client config file (Claude Code ~/.claude.json or .mcp.json, "
                "Cursor ~/.cursor/mcp.json, Claude Desktop claude_desktop_config.json)"
            ),
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="Reference model for the $/message estimate"),
    ] = "anthropic:claude-sonnet-5",
    dedupe: Annotated[
        bool,
        typer.Option(
            "--dedupe",
            help="Build hide plans for cross-server duplicate tools (loser side)",
        ),
    ] = False,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help=(
                "Write per-server plans plus a slimmed client config (the original "
                "config is never touched). Requires --dedupe and/or --hide."
            ),
        ),
    ] = False,
    out: Annotated[
        str, typer.Option("--out", "-o", help="Output directory for --apply")
    ] = "slim-out",
    keep: Annotated[
        str,
        typer.Option("--keep", help="Comma-separated server names slim must never touch"),
    ] = "",
    hide: Annotated[
        str,
        typer.Option("--hide", help="Comma-separated server names to hide entirely"),
    ] = "",
    share: Annotated[
        bool,
        typer.Option(
            "--share",
            help="Also print a copy-pasteable markdown snippet of the audit (with badge)",
        ),
    ] = False,
    recommend_hot: Annotated[
        bool,
        typer.Option(
            "--recommend-hot",
            help=(
                "Recommend which servers deserve alwaysLoad: true under Claude "
                "Code's tool search (uses --from-traces when given)"
            ),
        ),
    ] = False,
    from_traces: Annotated[
        str | None,
        typer.Option(
            "--from-traces",
            help="whetkit trace store: servers whose tools real runs called count as hot",
        ),
    ] = None,
    json_out: Annotated[
        bool, typer.Option("--json", help="Emit the audit as JSON instead of text")
    ] = False,
    plugins: Annotated[
        bool,
        typer.Option(
            "--plugins/--no-plugins",
            help=(
                "Also audit MCP servers shipped by installed Claude Code plugins "
                "(~/.claude/plugins) — measured but never modified"
            ),
        ),
    ] = True,
) -> None:
    """Audit — and optionally shrink — the union tool surface your MCP client
    sends with every message. The audit needs no tasks and no API key."""
    from whetkit.llm import parse_model
    from whetkit.llm.pricing import estimate_cost_usd
    from whetkit.slim import (
        build_dedupe_plans,
        cross_server_duplicates,
        discover_plugin_servers,
        parse_client_config,
        recommend_hot_servers,
        share_markdown,
        write_hot_config,
        write_slim_output,
    )

    try:
        # plugins can carry the whole surface: defer the nothing-to-audit
        # decision until after plugin discovery
        client_config = parse_client_config(config, allow_empty=True)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    for named in client_config.defer_loading_entries:
        typer.echo(
            f"warning: {named!r} sets defer_loading — Claude Code parses and "
            "silently ignores that field (anthropics/claude-code#26844); the "
            "real mechanism is alwaysLoad + tool search",
            err=True,
        )
    keep_servers = {s.strip() for s in keep.split(",") if s.strip()}
    hide_servers = {s.strip() for s in hide.split(",") if s.strip()}
    for named in (keep_servers | hide_servers) - set(client_config.servers):
        typer.echo(f"warning: --keep/--hide names unknown server {named!r}", err=True)
    if apply and not (dedupe or hide_servers or (recommend_hot and from_traces)):
        raise typer.BadParameter(
            "--apply needs --dedupe, --hide, and/or --recommend-hot --from-traces "
            "to have work to do"
        )

    plugin_names: set[str] = set()
    all_servers = dict(client_config.servers)
    if not all_servers and not plugins:
        raise typer.BadParameter(
            f"{client_config.path} has no mcpServers entries and --no-plugins "
            "was passed — nothing to audit."
        )
    if plugins:
        plugins_dir = os.environ.get("CLAUDE_PLUGINS_DIR") or (Path.home() / ".claude" / "plugins")
        plugin_servers, plugin_warnings = discover_plugin_servers(plugins_dir)
        for warning in plugin_warnings:
            typer.echo(f"warning: {warning}", err=True)
        plugin_names = set(plugin_servers)
        all_servers.update(plugin_servers)
        if not all_servers:
            raise typer.BadParameter(
                f"{client_config.path} has no mcpServers entries and no installed "
                "plugin ships MCP servers — nothing to audit."
            )
        if not client_config.servers and plugin_names:
            typer.echo(
                "note: this config declares no mcpServers — auditing the "
                "plugin-provided surface only",
                err=True,
            )

    async def _inventories() -> tuple[dict, dict[str, str]]:
        inventories: dict = {}
        failures: dict[str, str] = {}
        for name, spec in all_servers.items():
            typer.echo(f"inspecting {name} ...", err=True)
            try:
                inventories[name] = await inspect_server(spec)
            except Exception as exc:  # one dead server must not kill the audit
                failures[name] = f"{type(exc).__name__}: {exc}"
        return inventories, failures

    inventories, failures = asyncio.run(_inventories())
    if not inventories:
        typer.echo("error: no server in the config could be inspected", err=True)
        raise typer.Exit(code=1)

    for named in sorted((hide_servers | keep_servers) & set(failures)):
        typer.echo(
            f"warning: cannot act on {named!r} — it could not be inspected ({failures[named]})",
            err=True,
        )
    hide_servers -= set(failures)

    duplicates = cross_server_duplicates(inventories)
    model_id = parse_model(model)[1]
    total_tokens = sum(inv.total_definition_tokens for inv in inventories.values())
    per_message = estimate_cost_usd(model_id, total_tokens, 0)

    for named in sorted(hide_servers & plugin_names):
        typer.echo(
            f"warning: {named!r} is plugin-provided — whetkit measures it but "
            "cannot hide it; disable the plugin instead",
            err=True,
        )
    plans = (
        build_dedupe_plans(
            inventories,
            duplicates if dedupe else [],
            hide_servers=hide_servers - plugin_names,
            keep_servers=keep_servers | plugin_names,
        )
        if (dedupe or hide_servers)
        else {}
    )
    hidden_tokens = 0
    for server, plan in plans.items():
        hidden = {o.original_name for o in plan.overrides if o.hidden}
        hidden_tokens += sum(
            t.definition_tokens for t in inventories[server].tools if t.name in hidden
        )
    after_tokens = total_tokens - hidden_tokens
    after_cost = estimate_cost_usd(model_id, after_tokens, 0)

    def cost_of(server_names) -> tuple[int, float | None]:
        tokens = sum(
            inventories[n].total_definition_tokens for n in server_names if n in inventories
        )
        return tokens, estimate_cost_usd(model_id, tokens, 0)

    hot_recommendation: dict | None = None
    if recommend_hot:
        if from_traces:
            try:
                hot, cold, hot_warnings = recommend_hot_servers(inventories, from_traces)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            for warning in hot_warnings:
                typer.echo(f"warning: {warning}", err=True)
            hot_tokens, hot_cost = cost_of(hot)
            hot_recommendation = {
                "hot": sorted(hot),
                "cold": sorted(cold),
                "hot_definition_tokens": hot_tokens,
                "hot_cost_per_message_usd": hot_cost,
            }
        else:
            typer.echo(
                "note: --recommend-hot without --from-traces has no usage signal — "
                "the per-server table above IS the cost-if-always-loaded view. "
                "Run your usual tasks with 'whetkit run --store traces.sqlite3' "
                "and pass --from-traces for a real recommendation.",
                err=True,
            )

    snippet = share_markdown(
        client_config.path,
        [(name, inv.tool_count, inv.total_definition_tokens) for name, inv in inventories.items()],
        total_tokens,
        per_message,
        after_tokens if plans else None,
        after_cost if plans else None,
    )

    if json_out:
        import json as jsonlib

        document = {
            "config": client_config.path,
            "model": model,
            "servers": {
                name: {
                    "tools": inv.tool_count,
                    "definition_tokens": inv.total_definition_tokens,
                    "cost_per_message_usd": estimate_cost_usd(
                        model_id, inv.total_definition_tokens, 0
                    ),
                }
                for name, inv in inventories.items()
            },
            "failures": failures,
            "skipped": [s.model_dump() for s in client_config.skipped],
            "total_definition_tokens": total_tokens,
            "cost_per_message_usd": per_message,
            "cross_server_duplicates": [d.model_dump() for d in duplicates],
            "plugin_servers": sorted(plugin_names),
            "always_load": client_config.always_load,
            "defer_loading_ignored": client_config.defer_loading_entries,
            "hot_recommendation": hot_recommendation,
            "share_markdown": snippet,
            "plan_servers": sorted(plans),
            "after_definition_tokens": after_tokens if plans else None,
            "after_cost_per_message_usd": after_cost if plans else None,
        }
        typer.echo(jsonlib.dumps(document, indent=2))
    else:
        typer.echo(f"\nMCP client config: {client_config.path}")
        name_w = max(len(n) for n in inventories)
        typer.echo(f"\n{'SERVER':<{name_w}}  {'TOOLS':>5}  {'DEF TOKENS':>10}  {'$/MSG':>8}")
        for name, inv in inventories.items():
            cost = estimate_cost_usd(model_id, inv.total_definition_tokens, 0)
            cost_s = f"${cost:.4f}" if cost is not None else "?"
            typer.echo(
                f"{name:<{name_w}}  {inv.tool_count:>5}  "
                f"{inv.total_definition_tokens:>10}  {cost_s:>8}"
            )
        for name, reason in failures.items():
            typer.echo(f"{name:<{name_w}}  (could not inspect: {reason})")
        for skipped in client_config.skipped:
            typer.echo(f"{skipped.name:<{name_w}}  (skipped: {skipped.reason})")
        typer.echo(
            f"\nUnion: {sum(i.tool_count for i in inventories.values())} tools, "
            f"~{total_tokens} definition tokens riding along with EVERY message"
        )
        if per_message is not None:
            typer.echo(
                f"≈ ${per_message:.4f} of context per message on {model_id} "
                f"(≈ ${per_message * 1000:.2f} per 1,000 messages)"
            )
        if client_config.always_load:
            hot_tokens, hot_cost = cost_of(client_config.always_load)
            cost_s = f", ${hot_cost:.4f}/message" if hot_cost is not None else ""
            typer.echo(
                f"\nTool search hot set (alwaysLoad): "
                f"{', '.join(client_config.always_load)} — ~{hot_tokens} tokens{cost_s}; "
                "the rest loads on demand"
            )
        elif len(inventories) > 1:
            typer.echo(
                "\nTip: Claude Code's tool search (ENABLE_TOOL_SEARCH) defers tool "
                "loading; mark only your hot servers with alwaysLoad: true — "
                "try --recommend-hot --from-traces"
            )
        if hot_recommendation is not None:
            rec_cost = hot_recommendation["hot_cost_per_message_usd"]
            cost_s = f", ${rec_cost:.4f}/message" if rec_cost is not None else ""
            typer.echo(
                f"\nRecommended alwaysLoad set (from traces): "
                f"{', '.join(hot_recommendation['hot']) or '(none)'} — "
                f"~{hot_recommendation['hot_definition_tokens']} tokens{cost_s} "
                f"(full surface: ~{total_tokens})"
            )
            if hot_recommendation["cold"]:
                typer.echo(f"Defer (no observed usage): {', '.join(hot_recommendation['cold'])}")
        if duplicates:
            typer.echo(f"\nCross-server duplicates ({len(duplicates)}):")
            for duplicate in duplicates:
                typer.echo(f"  • {duplicate.describe()}")
        elif len(inventories) > 1:
            typer.echo("\nNo cross-server duplicate tools detected.")
        if plans:
            typer.echo(
                f"\nSlim plan: hide "
                f"{sum(len(p.overrides) for p in plans.values())} tool(s) across "
                f"{len(plans)} server(s) -> ~{after_tokens} tokens"
                + (f", ${after_cost:.4f}/message" if after_cost is not None else "")
            )

    if share and not json_out:
        typer.echo("\n--- copy below ---")
        typer.echo(snippet)
        typer.echo("--- copy above ---")

    if apply and hot_recommendation is not None:
        hot_path = write_hot_config(client_config, set(hot_recommendation["hot"]), out)
        typer.echo(f"\nwrote {hot_path} (alwaysLoad stamped on the recommended servers)")

    if apply and plans:
        slimmed, removed = write_slim_output(client_config, plans, out, inventories=inventories)
        typer.echo(f"\nwrote {slimmed}")
        for name in removed:
            typer.echo(
                f"removed {name} from the slimmed config (all tools hidden) — "
                "restore by re-adding its original entry"
            )
        if client_config.standalone:
            typer.echo(
                "Point your client at the slimmed config to use it; your original "
                f"config was not modified — reverting is switching back to {client_config.path}."
            )
        else:
            typer.echo(
                f"{client_config.path} holds more than mcpServers (settings, "
                "projects), so the slimmed file is a fragment, not a drop-in "
                "replacement: use it as a project .mcp.json, or merge its "
                "entries into your config. Your original was not modified."
            )
    elif apply:
        typer.echo(
            "\nnothing to apply — no duplicate losers or hideable servers survived the checks above"
        )


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
    from whetkit.curation.export import plan_to_json, plan_to_markdown

    if to not in ("markdown", "json"):
        raise typer.BadParameter("--to must be 'markdown' or 'json'")
    if not Path(plan).is_file():
        raise typer.BadParameter(
            f"no curation plan at {plan} — run 'whetkit curate' first, or pass --plan"
        )
    curation_plan = _load_plan(plan)
    if not curation_plan.overrides:
        typer.echo("plan has no overrides — nothing to export", err=True)
        raise typer.Exit(code=1)

    rendered = (plan_to_markdown if to == "markdown" else plan_to_json)(curation_plan)
    if out:
        _out_path(out).parent.mkdir(parents=True, exist_ok=True)
        _out_path(out).write_text(rendered)
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

    def _load_summary(path: str) -> dict:
        """Parse one --summary-json file, refusing anything else cleanly."""
        if not Path(path).is_file():
            raise typer.BadParameter(f"no summary file at {path}")
        not_a_summary = f"{path} is not a 'whetkit run --summary-json' output"
        try:
            doc = jsonlib.loads(Path(path).read_text())
        except jsonlib.JSONDecodeError as exc:
            raise typer.BadParameter(f"{not_a_summary}: invalid JSON ({exc})") from exc
        runs_docs = doc.get("runs") if isinstance(doc, dict) else None
        if not isinstance(runs_docs, list) or not all(
            isinstance(run_doc, dict)
            and isinstance(run_doc.get("tasks"), list)
            and all(isinstance(t, dict) and "id" in t and "hit" in t for t in run_doc["tasks"])
            for run_doc in runs_docs
        ):
            raise typer.BadParameter(
                f"{not_a_summary} (expected a top-level 'runs' list with per-task "
                "'id'/'hit' entries) — pass files written by 'whetkit run --summary-json'"
            )
        return doc

    docs = [_load_summary(path) for path in (before, after)]

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

    task_list = _load_tasks(tasks)
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

        from whetkit.curation import CuratedMCPClient

        if not Path(plan).is_file():
            raise typer.BadParameter(f"no curation plan at {plan}")
        curation_plan = _load_plan(plan)
        _warn_plan_hides_required_tools(curation_plan, task_list)
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
        _echo_run_errors(task_runs)
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
        typer.echo(_usage_cost_line(model, task_runs))
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
            _out_path(summary_json).parent.mkdir(parents=True, exist_ok=True)
            _out_path(summary_json).write_text(jsonlib.dumps(document, indent=2) + "\n")
            typer.echo(f"Summary JSON: {summary_json}")

        typer.echo(_CLOUD_WAITLIST_LINE, err=True)
        _exit_on_errored_runs(summaries)

    _require_provider_keys(
        {"agent model": model, **({"judge model": judge_model} if use_judge else {})}
    )
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
    from whetkit.mcp import MCPClient, inspect_server
    from whetkit.runner import RunConfig
    from whetkit.scoring import (
        JudgeCache,
        JudgeConfig,
        MultiRunSummary,
        hit_rate_noise_caveat,
        score_runs,
    )
    from whetkit.tracing import default_store_path

    if runs < 1:
        raise typer.BadParameter("--runs must be at least 1")
    if task_timeout <= 0:
        raise typer.BadParameter("--task-timeout must be positive")
    mode = _match_mode(match_mode)
    task_list = _load_tasks(tasks)
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
            for warning in _drop_task_breaking_hides(plan, task_list):
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
        all_task_runs = [r for rep in baseline_all + curated_all for r in rep]
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
            baseline_summaries=baseline_summaries,
            curated_summaries=curated_summaries,
            est_cost_usd=_est_cost(model, all_task_runs),
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
        if caveat := hit_rate_noise_caveat(baseline_multi, curated_multi):
            typer.echo(caveat)
        tok_before = (report.before.input_tokens + report.before.output_tokens) // len(task_list)
        tok_after = (report.after.input_tokens + report.after.output_tokens) // len(task_list)
        typer.echo(
            f"Tools: {report.tools_before} -> {report.tools_after}   "
            f"Tokens/task: {tok_before} -> {tok_after}"
        )
        typer.echo(_usage_cost_line(model, all_task_runs))
        typer.echo(
            f"Traces saved to {store_path} (groups {_group_family_note('baseline', runs)}, "
            f"{_group_family_note('curated', runs)})"
        )
        typer.echo(f"Report: {html_path} (machine-readable: {json_path})")

        typer.echo(_CLOUD_WAITLIST_LINE, err=True)
        _exit_on_errored_runs(baseline_summaries + curated_summaries)
        if curated_multi.mean_hit_rate < baseline_multi.mean_hit_rate:
            typer.echo(
                f"\n⚠ REGRESSION: the curated view scored LOWER than baseline "
                f"({baseline_multi.mean_hit_rate:.0%} -> {curated_multi.mean_hit_rate:.0%}). "
                "Do not adopt this plan. Try 'whetkit fix' (iterative self-correction) "
                "or edit the plan and re-score with 'whetkit run --plan'.",
                err=True,
            )
            raise typer.Exit(code=4)

    _require_provider_keys(
        {
            "agent model": model,
            "optimizer model": optimizer_model,
            **({"judge model": judge_model} if use_judge else {}),
        }
    )
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
    from whetkit.runner import RunConfig
    from whetkit.scoring import (
        JudgeCache,
        JudgeConfig,
        MultiRunSummary,
        hit_rate_noise_caveat,
        score_runs,
    )
    from whetkit.tracing import default_store_path

    if max_iterations < 1:
        raise typer.BadParameter("--max-iterations must be at least 1")
    if runs < 1:
        raise typer.BadParameter("--runs must be at least 1")
    if task_timeout <= 0:
        raise typer.BadParameter("--task-timeout must be positive")
    mode = _match_mode(match_mode)
    task_list = _load_tasks(tasks)
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
            all_summaries = list(baseline_summaries)
            spent_runs = [r for rep in baseline_all for r in rep]
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
            for w in warnings + _drop_task_breaking_hides(plan, task_list):
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
                all_summaries.extend(curated_summaries)
                spent_runs.extend(r for rep in curated_all for r in rep)
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
                for w in warnings + _drop_task_breaking_hides(plan, task_list):
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
        if caveat := hit_rate_noise_caveat(baseline_multi, best_multi):
            typer.echo(caveat)
        typer.echo(_usage_cost_line(model, spent_runs))
        typer.echo(f"serve it:  whetkit overlay --server <origin> --plan {plan_path}")

        typer.echo(_CLOUD_WAITLIST_LINE, err=True)
        _exit_on_errored_runs(all_summaries)

    _require_provider_keys(
        {
            "agent model": model,
            "optimizer model": optimizer_model,
            **({"judge model": judge_model} if use_judge else {}),
        }
    )
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
    from whetkit.report import build_report
    from whetkit.scoring import JudgeCache, JudgeConfig, score_runs
    from whetkit.tracing import TraceStore, default_store_path

    mode = _match_mode(match_mode)
    task_list = _load_tasks(tasks)
    if not Path(plan).is_file():
        raise typer.BadParameter(
            f"no curation plan at {plan} — run 'whetkit curate' first, or pass --plan"
        )
    curation_plan = _load_plan(plan)
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

        # Best-effort tool counts: re-inspect the tasks' server so the rebuilt
        # report keeps the TOOLS EXPOSED metric. An unreachable server just
        # leaves the counts off the report, exactly as before.
        tools_before = tools_after = None
        try:
            origin_spec = resolve_server_spec(task_list[0].server)
            inventory = await inspect_server(origin_spec)
            origin_names = {t.name for t in inventory.tools}
            tools_before = inventory.tool_count
            tools_after = len(curation_plan.presented_to_original(origin_names))
        except Exception:
            pass

        comparison = build_report(
            task_list,
            before_runs,
            before_summary,
            after_runs,
            after_summary,
            curation_plan,
            model=before_runs[0].model if before_runs else "",
            server=before_runs[0].server if before_runs else "",
            tools_before=tools_before,
            tools_after=tools_after,
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
    from whetkit.curation import serve_overlay
    from whetkit.curation.overlay import InvalidPlanError

    origin = _resolve_server(server, http_mode)
    curation_plan = _load_plan(plan)
    try:
        asyncio.run(serve_overlay(origin, curation_plan))
    except InvalidPlanError as exc:
        typer.echo(f"error: curation plan is not valid for this origin: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def telemetry(
    action: Annotated[
        str, typer.Argument(help="'on' or 'off' to opt in/out, 'status' to inspect")
    ] = "status",
) -> None:
    """Manage opt-in anonymous usage telemetry (off by default)."""
    import uuid

    from whetkit import telemetry as tel

    if action not in ("on", "off", "status"):
        raise typer.BadParameter("action must be 'on', 'off', or 'status'")

    if action in ("on", "off"):
        config = tel.load_config()
        config["enabled"] = action == "on"
        if action == "on" and not config.get("anonymous_id"):
            config["anonymous_id"] = str(uuid.uuid4())
        tel.save_config(config)
        state = "enabled" if action == "on" else "disabled"
        typer.echo(f"telemetry {state} — recorded in {tel.config_path()}")
    else:
        env = os.environ.get("WHETKIT_TELEMETRY")
        state = "enabled" if tel.is_enabled() else "disabled"
        source = (
            f"WHETKIT_TELEMETRY={env} (env var overrides the config file)"
            if env is not None
            else f"{tel.config_path()}"
        )
        typer.echo(f"telemetry is {state}  [{source}]")

    typer.echo(f"when enabled, each command sends exactly: {tel.COLLECTED}.")
    typer.echo(f"never collected: {tel.NEVER_COLLECTED}.")
    typer.echo("toggle with 'whetkit telemetry on|off' or WHETKIT_TELEMETRY=1|0.")


if __name__ == "__main__":
    app()
