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
