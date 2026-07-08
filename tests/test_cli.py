from pathlib import Path

from typer.testing import CliRunner

from whetkit.cli import _judge_enabled, app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


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
