import json
from pathlib import Path

from typer.testing import CliRunner

from whetkit.cli import _judge_enabled, app
from whetkit.llm import LLMTurn, ToolCall

from .fakes import FakeProvider, SleepyProvider

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def _mini_task_file(tmp_path: Path) -> Path:
    """One task against the mini fixture server (expects the 'add' tool)."""
    tasks = tmp_path / "task.yaml"
    tasks.write_text(
        f"id: add-two\n"
        f"prompt: add 2 and 3\n"
        f"server: {FIXTURES / 'mini_server.py'}\n"
        f"expected_tools: [add]\n"
        f"success_criteria: says 5\n"
    )
    return tasks


def _agent_turns_hit() -> list[LLMTurn]:
    """One agent run that calls the expected tool, then answers."""
    return [
        LLMTurn(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]),
        LLMTurn(text="2 + 3 = 5"),
    ]


def _agent_turns_miss() -> list[LLMTurn]:
    """One agent run that answers without any tool call."""
    return [LLMTurn(text="no idea")]


def _patch_agent_provider(monkeypatch, script: list[LLMTurn]) -> FakeProvider:
    provider = FakeProvider(script)
    monkeypatch.setattr("whetkit.runner.agent.get_provider", lambda name: provider)
    return provider


def _patch_optimizer_provider(monkeypatch, overrides: list | None = None) -> FakeProvider:
    proposal = {"notes": "no changes needed", "overrides": overrides or []}
    provider = FakeProvider([LLMTurn(text=json.dumps(proposal))])
    monkeypatch.setattr("whetkit.curation.optimizer.get_provider", lambda name: provider)
    return provider


def _agent_run_count(provider: FakeProvider) -> int:
    """How many distinct agent runs the provider served (each starts with
    exactly one message: the task prompt)."""
    return sum(1 for call in provider.calls if len(call["messages"]) == 1)


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("inspect", "run", "curate", "report", "overlay"):
        assert command in result.output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("whetkit ")


def test_run_missing_plan_is_friendly(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--tasks",
            str(Path(__file__).parent.parent / "examples" / "tasks"),
            "--plan",
            str(tmp_path / "nope.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "no curation plan" in result.output


def test_report_missing_plan_is_friendly(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "report",
            "--tasks",
            str(Path(__file__).parent.parent / "examples" / "tasks"),
            "--plan",
            str(tmp_path / "nope.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "no curation plan" in result.output
    assert "Traceback" not in result.output


def test_inspect_prints_inventory() -> None:
    result = runner.invoke(app, ["inspect", "--server", str(FIXTURES / "mini_server.py")])
    assert result.exit_code == 0
    assert "Tools: 2" in result.output
    assert "add" in result.output
    assert "Greet a person by name." in result.output


def test_inspect_rejects_bad_server() -> None:
    result = runner.invoke(app, ["inspect", "--server", "does-not-exist"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "not a URL, directory" in result.output


def test_doctor_rejects_bad_server_without_traceback() -> None:
    result = runner.invoke(app, ["doctor", "--server", "typo.json"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_report_requires_existing_traces(tmp_path: Path) -> None:
    from whetkit.curation import CurationPlan, save_plan
    from whetkit.tracing import TraceStore

    plan_path = tmp_path / "plan.yaml"
    save_plan(CurationPlan(), plan_path)
    store_path = tmp_path / "traces.sqlite3"
    TraceStore(store_path).close()  # empty store

    result = runner.invoke(
        app,
        [
            "report",
            "--tasks",
            str(Path(__file__).parent.parent / "examples" / "tasks"),
            "--plan",
            str(plan_path),
            "--store",
            str(store_path),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "whetkit curate" in result.output


async def test_summary_payload_shape() -> None:
    from whetkit.cli import _summary_payload
    from whetkit.datasets import TaskSpec
    from whetkit.scoring import score_runs
    from whetkit.tracing import TaskRun, ToolCallRecord, TurnRecord

    task = TaskSpec(id="t", prompt="p", server="s", expected_tools=["a"], success_criteria="c")
    run = TaskRun(
        task_id="t",
        server="s",
        model="m",
        turns=[
            TurnRecord(
                index=0,
                tool_calls=[
                    ToolCallRecord(call_id="c1", name="a", result_text="ok"),
                    ToolCallRecord(call_id="c2", name="x", result_text="ok"),
                ],
            )
        ],
    )
    summary = await score_runs([task], [run])
    payload = _summary_payload("baseline-1", summary, [run])
    assert payload["group"] == "baseline-1"
    assert payload["hit_rate"] == 1.0
    assert payload["avg_extra_calls"] == 1.0
    (task_entry,) = payload["tasks"]
    assert task_entry["id"] == "t" and task_entry["hit"] is True
    assert task_entry["called"] == ["a", "x"]
    assert task_entry["extra_calls"] == ["x"]
    import json

    json.dumps(payload)  # must be plain-data serializable


def test_judge_enabled_logic(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _judge_enabled("on", "anthropic:m") is True
    assert _judge_enabled("off", "anthropic:m") is False
    assert _judge_enabled("auto", "anthropic:m") is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert _judge_enabled("auto", "anthropic:m") is True
    assert _judge_enabled("auto", "openai:m") is False


def test_diff_compares_two_summaries(tmp_path: Path) -> None:
    import json

    def doc(hit: float, extras: float, task_hit: bool):
        return {
            "runs": [
                {
                    "hit_rate": hit,
                    "tool_hit_rate": hit,
                    "judge_pass_rate": hit,
                    "avg_precision": hit,
                    "avg_extra_calls": extras,
                    "tokens_in": 1000,
                    "tasks": [{"id": "trap-task", "hit": task_hit}],
                }
            ]
        }

    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(doc(0.8, 1.8, False)))
    after.write_text(json.dumps(doc(1.0, 0.2, True)))

    result = runner.invoke(app, ["diff", str(before), str(after)])
    assert result.exit_code == 0
    assert "80%" in result.output and "100%" in result.output
    assert "trap-task" in result.output
    assert "MISS" in result.output and "PASS" in result.output and "↑" in result.output


def test_diff_missing_file_is_friendly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["diff", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert result.exit_code != 0
    assert "no summary file" in result.output


def test_reset_cmd_failure_is_friendly(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--tasks",
            str(Path(__file__).parent.parent / "examples" / "tasks"),
            "--reset-cmd",
            "exit 3",
            "--judge",
            "off",
            "--store",
            str(tmp_path / "t.sqlite3"),
        ],
    )
    assert result.exit_code == 1
    assert "reset-cmd failed with exit code 3" in result.output
    assert "Traceback" not in result.output


def test_plan_init_scaffolds_view_plan(tmp_path: Path) -> None:
    from whetkit.curation import load_plan

    out = tmp_path / "plan.yaml"
    result = runner.invoke(
        app,
        [
            "plan-init",
            "--server",
            str(FIXTURES / "mini_server.py"),
            "--keep",
            "add,ghost_tool",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert "ignoring: ghost_tool" in result.output
    plan = load_plan(out)
    hidden = {o.original_name for o in plan.overrides if o.hidden}
    assert "add" not in hidden and hidden  # everything except 'add' hidden


def test_plan_init_requires_a_keep_set() -> None:
    result = runner.invoke(app, ["plan-init", "--server", str(FIXTURES / "mini_server.py")])
    assert result.exit_code != 0
    assert "nothing to keep" in result.output


def _multi_server_task_file(tmp_path: Path) -> Path:
    """Two tasks against two genuinely different servers."""
    mini = FIXTURES / "mini_server.py"
    sample = Path(__file__).parent.parent / "examples" / "sample-server"
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(
        f"- id: on-mini\n"
        f"  prompt: add 2 and 3\n"
        f"  server: {mini}\n"
        f"  expected_tools: [add]\n"
        f"  success_criteria: says 5\n"
        f"- id: on-sample\n"
        f"  prompt: find a mouse\n"
        f"  server: {sample}\n"
        f"  expected_tools: [data_query_1]\n"
        f"  success_criteria: names a mouse\n"
    )
    return tasks


def _forbid_agent_runs(monkeypatch) -> None:
    """Fail loudly if the command reaches the agent loop (and its provider)."""

    async def _boom(*args, **kwargs):
        raise AssertionError("run_task must not be reached")

    monkeypatch.setattr("whetkit.runner.run_task", _boom)


def test_curate_refuses_multi_server_task_sets(tmp_path: Path, monkeypatch) -> None:
    _forbid_agent_runs(monkeypatch)
    result = runner.invoke(
        app,
        [
            "curate",
            "--tasks",
            str(_multi_server_task_file(tmp_path)),
            "--judge",
            "off",
            "--store",
            str(tmp_path / "t.sqlite3"),
            "--plan",
            str(tmp_path / "plan.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "2 different servers" in result.output
    assert "unsupported" in result.output
    assert "once per server" in result.output


def test_fix_refuses_multi_server_task_sets(tmp_path: Path, monkeypatch) -> None:
    _forbid_agent_runs(monkeypatch)
    result = runner.invoke(
        app,
        [
            "fix",
            "--tasks",
            str(_multi_server_task_file(tmp_path)),
            "--judge",
            "off",
            "--store",
            str(tmp_path / "t.sqlite3"),
            "--plan",
            str(tmp_path / "plan.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "2 different servers" in result.output
    assert "whetkit fix" in result.output


def test_curate_accepts_single_server_after_override(tmp_path: Path) -> None:
    # --server collapses a multi-server task set onto one spec: the guard
    # must not fire then.
    from whetkit.cli import _resolve_task_servers, _single_server_spec
    from whetkit.datasets import load_tasks
    from whetkit.mcp import HttpMode

    tasks = load_tasks(_multi_server_task_file(tmp_path))
    servers = _resolve_task_servers(tasks, str(FIXTURES / "mini_server.py"), HttpMode.STATEFUL)
    spec = _single_server_spec(servers, "curate")
    assert "mini_server.py" in spec.label()


def test_curate_runs_repeats_evals_and_reports_mean_range(tmp_path: Path, monkeypatch) -> None:
    from whetkit.tracing import TraceStore

    monkeypatch.chdir(tmp_path)  # keep the judge cache out of the repo
    # baseline: rep 1 hits, rep 2 misses; curated: both reps hit
    agent = _patch_agent_provider(
        monkeypatch,
        _agent_turns_hit() + _agent_turns_miss() + _agent_turns_hit() + _agent_turns_hit(),
    )
    _patch_optimizer_provider(monkeypatch)

    store_path = tmp_path / "traces.sqlite3"
    result = runner.invoke(
        app,
        [
            "curate",
            "--tasks",
            str(_mini_task_file(tmp_path)),
            "--runs",
            "2",
            "--judge",
            "off",
            "--model",
            "fake:agent",
            "--optimizer-model",
            "fake:opt",
            "--store",
            str(store_path),
            "--plan",
            str(tmp_path / "plan.yaml"),
            "--report-dir",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert _agent_run_count(agent) == 4  # 2 baseline reps + 2 curated reps

    # mean with range on the noisy side, collapsed mean on the stable side
    assert "Hit-rate: 50% [0%–100%] -> 100%" in result.output
    assert "groups 'baseline-1'..'-2', 'curated-1'..'-2'" in result.output

    with TraceStore(store_path) as store:
        groups = {g["run_group"] for g in store.list_groups()}
    assert groups == {"baseline-1", "baseline-2", "curated-1", "curated-2"}

    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert report["before_spread"] == "50% [0%–100%]"
    assert report["after_spread"] == "100%"


def test_curate_single_run_keeps_plain_groups(tmp_path: Path, monkeypatch) -> None:
    from whetkit.tracing import TraceStore

    monkeypatch.chdir(tmp_path)
    agent = _patch_agent_provider(monkeypatch, _agent_turns_hit() + _agent_turns_hit())
    _patch_optimizer_provider(monkeypatch)

    store_path = tmp_path / "traces.sqlite3"
    # simulate leftovers from an earlier --runs 2 invocation: they must be replaced
    from whetkit.tracing import TaskRun

    with TraceStore(store_path) as store:
        store.save_runs([TaskRun(task_id="add-two", server="s", model="m")], run_group="baseline-2")

    result = runner.invoke(
        app,
        [
            "curate",
            "--tasks",
            str(_mini_task_file(tmp_path)),
            "--judge",
            "off",
            "--model",
            "fake:agent",
            "--optimizer-model",
            "fake:opt",
            "--store",
            str(store_path),
            "--plan",
            str(tmp_path / "plan.yaml"),
            "--report-dir",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert _agent_run_count(agent) == 2
    assert "Hit-rate: 100% -> 100%" in result.output
    with TraceStore(store_path) as store:
        groups = {g["run_group"] for g in store.list_groups()}
    assert groups == {"baseline", "curated"}  # stale baseline-2 cleared


def test_fix_runs_repeats_evals_and_reports_mean_range(tmp_path: Path, monkeypatch) -> None:
    from whetkit.tracing import TraceStore

    monkeypatch.chdir(tmp_path)
    agent = _patch_agent_provider(
        monkeypatch,
        _agent_turns_hit() + _agent_turns_miss() + _agent_turns_hit() + _agent_turns_hit(),
    )
    _patch_optimizer_provider(monkeypatch)

    store_path = tmp_path / "traces.sqlite3"
    result = runner.invoke(
        app,
        [
            "fix",
            "--tasks",
            str(_mini_task_file(tmp_path)),
            "--runs",
            "2",
            "--max-iterations",
            "1",
            "--judge",
            "off",
            "--model",
            "fake:agent",
            "--optimizer-model",
            "fake:opt",
            "--store",
            str(store_path),
            "--plan",
            str(tmp_path / "plan.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert _agent_run_count(agent) == 4  # 2 baseline reps + 2 curated reps
    assert "Hit-rate: 50% [0%–100%] -> 100%" in result.output
    with TraceStore(store_path) as store:
        groups = {g["run_group"] for g in store.list_groups()}
    assert groups == {"baseline-1", "baseline-2", "fix-1-1", "fix-1-2"}


def _colliding_plan(tmp_path: Path) -> Path:
    """A plan renaming two sample-server tools to the same presented name."""
    from whetkit.curation import CurationPlan, ToolOverride, save_plan

    plan_path = tmp_path / "bad-plan.yaml"
    save_plan(
        CurationPlan(
            overrides=[
                ToolOverride(original_name="data_query_1", new_name="search"),
                ToolOverride(original_name="legacy_search", new_name="search"),
            ]
        ),
        plan_path,
    )
    return plan_path


def test_run_plan_is_validated_against_origin_before_running(tmp_path: Path, monkeypatch) -> None:
    _forbid_agent_runs(monkeypatch)
    result = runner.invoke(
        app,
        [
            "run",
            "--tasks",
            str(Path(__file__).parent.parent / "examples" / "tasks"),
            "--plan",
            str(_colliding_plan(tmp_path)),
            "--judge",
            "off",
            "--store",
            str(tmp_path / "t.sqlite3"),
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "not valid for" in result.output
    assert "collision: 'search'" in result.output


def test_run_accepts_a_valid_plan(tmp_path: Path, monkeypatch) -> None:
    from whetkit.curation import CurationPlan, ToolOverride, save_plan

    monkeypatch.chdir(tmp_path)
    agent = _patch_agent_provider(monkeypatch, _agent_turns_miss())
    plan_path = tmp_path / "plan.yaml"
    save_plan(
        CurationPlan(overrides=[ToolOverride(original_name="add", new_name="sum_numbers")]),
        plan_path,
    )
    result = runner.invoke(
        app,
        [
            "run",
            "--tasks",
            str(_mini_task_file(tmp_path)),
            "--plan",
            str(plan_path),
            "--judge",
            "off",
            "--model",
            "fake:agent",
            "--store",
            str(tmp_path / "t.sqlite3"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert _agent_run_count(agent) == 1  # validation passed, the eval ran


def test_overlay_refuses_invalid_plan_with_clear_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "overlay",
            "--server",
            str(Path(__file__).parent.parent / "examples" / "sample-server"),
            "--plan",
            str(_colliding_plan(tmp_path)),
        ],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "not valid for this origin" in result.output
    assert "collision: 'search'" in result.output


def test_task_timeout_must_be_positive(tmp_path: Path) -> None:
    tasks = str(_mini_task_file(tmp_path))
    for command in ("run", "curate", "fix"):
        result = runner.invoke(app, [command, "--tasks", tasks, "--task-timeout", "0"])
        assert result.exit_code != 0, command
        assert "--task-timeout must be positive" in result.output, command


def test_run_flags_timed_out_tasks_in_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    provider = SleepyProvider(delay_s=30.0)
    monkeypatch.setattr("whetkit.runner.agent.get_provider", lambda name: provider)

    result = runner.invoke(
        app,
        [
            "run",
            "--tasks",
            str(_mini_task_file(tmp_path)),
            "--judge",
            "off",
            "--model",
            "fake:sleepy",
            "--task-timeout",
            "0.2",
            "--store",
            str(tmp_path / "t.sqlite3"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Timed-out runs: 1/1" in result.output
    assert "Raise --task-timeout" in result.output


def test_plan_init_from_traces_keeps_called_tools(tmp_path: Path) -> None:
    from whetkit.curation import load_plan
    from whetkit.tracing import TaskRun, ToolCallRecord, TraceStore, TurnRecord

    store_path = tmp_path / "traces.sqlite3"
    run = TaskRun(
        task_id="t",
        server="s",
        model="m",
        turns=[
            TurnRecord(
                index=0,
                tool_calls=[ToolCallRecord(call_id="c", name="add", result_text="ok")],
            )
        ],
    )
    with TraceStore(store_path) as store:
        store.save_runs([run], run_group="baseline")

    out = tmp_path / "plan.yaml"
    result = runner.invoke(
        app,
        [
            "plan-init",
            "--server",
            str(FIXTURES / "mini_server.py"),
            "--from-traces",
            str(store_path),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    hidden = {o.original_name for o in load_plan(out).overrides if o.hidden}
    assert "add" not in hidden and hidden  # called tool kept, others hidden
