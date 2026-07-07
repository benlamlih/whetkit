import sqlite3
from pathlib import Path

import pytest

from mcp_eval.llm import Usage
from mcp_eval.tracing import TaskRun, ToolCallRecord, TraceStore, TurnRecord, read_jsonl
from mcp_eval.tracing import write_jsonl as write_jsonl_file
from mcp_eval.tracing.records import RunStatus, utc_now


def make_run(task_id: str = "t1", status: RunStatus = RunStatus.COMPLETED) -> TaskRun:
    return TaskRun(
        task_id=task_id,
        server="stdio: python server.py",
        model="anthropic:claude-sonnet-5",
        finished_at=utc_now(),
        turns=[
            TurnRecord(
                index=0,
                assistant_text="calling a tool",
                tool_calls=[
                    ToolCallRecord(
                        call_id="c1",
                        name="add",
                        arguments={"a": 2, "b": 3},
                        result_text="5",
                        latency_ms=12.5,
                    )
                ],
                usage=Usage(input_tokens=100, output_tokens=20),
                latency_ms=800.0,
                stop_reason="tool_use",
            ),
            TurnRecord(
                index=1,
                assistant_text="the answer is 5",
                usage=Usage(input_tokens=140, output_tokens=15),
                latency_ms=600.0,
                stop_reason="end_turn",
            ),
        ],
        final_text="the answer is 5",
        status=status,
    )


def test_sqlite_roundtrip(tmp_path: Path) -> None:
    with TraceStore(tmp_path / "traces.sqlite3") as store:
        store.save_runs([make_run("t1"), make_run("t2")], run_group="baseline")
        store.save_run(make_run("t1", status=RunStatus.MAX_TURNS), run_group="curated")

        baseline = store.load_runs("baseline")
        assert [r.task_id for r in baseline] == ["t1", "t2"]
        assert baseline[0] == make_run("t1").model_copy(
            update={"started_at": baseline[0].started_at, "finished_at": baseline[0].finished_at}
        )
        assert baseline[0].called_tool_names == ["add"]
        assert baseline[0].total_usage.input_tokens == 240

        everything = store.load_runs()
        assert len(everything) == 3

        groups = store.list_groups()
        assert [(g["run_group"], g["runs"]) for g in groups] == [("baseline", 2), ("curated", 1)]


def test_schema_version_guard(tmp_path: Path) -> None:
    db = tmp_path / "traces.sqlite3"
    TraceStore(db).close()

    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.close()

    with pytest.raises(RuntimeError, match="schema v999"):
        TraceStore(db)


def test_indexed_columns_match_record(tmp_path: Path) -> None:
    db = tmp_path / "traces.sqlite3"
    with TraceStore(db) as store:
        store.save_run(make_run(), run_group="g")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs").fetchone()
    conn.close()
    assert row["task_id"] == "t1"
    assert row["status"] == "completed"
    assert row["input_tokens"] == 240
    assert row["output_tokens"] == 35
    assert row["latency_ms"] == pytest.approx(1412.5)


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    runs = [make_run("t1"), make_run("t2", status=RunStatus.ERROR)]
    path = tmp_path / "out" / "traces.jsonl"
    write_jsonl_file(runs, path)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2

    loaded = read_jsonl(path)
    assert loaded == runs
