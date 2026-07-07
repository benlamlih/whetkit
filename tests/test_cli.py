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


def test_judge_enabled_logic(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _judge_enabled("on", "anthropic:m") is True
    assert _judge_enabled("off", "anthropic:m") is False
    assert _judge_enabled("auto", "anthropic:m") is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert _judge_enabled("auto", "anthropic:m") is True
    assert _judge_enabled("auto", "openai:m") is False
