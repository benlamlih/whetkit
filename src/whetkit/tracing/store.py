"""Trace persistence: local SQLite (queryable) + JSONL (portable).

The full pydantic record is the source of truth and is stored verbatim in
``record_json``; the indexed columns exist for querying and reporting.
The schema is versioned via the ``meta`` table — opening a store written by
a newer schema fails loudly instead of corrupting it.

Stage 1 is local-first by design: SQLite ships with CPython, needs no
daemon, and one file per project is easy to inspect and delete.
"""

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from whetkit.tracing.records import TaskRun

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_group TEXT NOT NULL,
    task_id TEXT NOT NULL,
    server TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_group ON runs (run_group);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs (task_id);
"""


class TraceStore:
    """A local store of TaskRun traces, grouped by run label.

    ``run_group`` names one eval batch (e.g. ``baseline`` or ``curated``) so
    before/after comparisons can pull exactly the runs they need.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.path} uses trace schema v{row['value']}, but this "
                    f"whetkit only understands up to v{SCHEMA_VERSION} — upgrade whetkit"
                )

    def save_run(self, run: TaskRun, run_group: str) -> int:
        usage = run.total_usage
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO runs (
                    run_group, task_id, server, model, status, error,
                    started_at, finished_at, input_tokens, output_tokens,
                    latency_ms, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_group,
                    run.task_id,
                    run.server,
                    run.model,
                    str(run.status),
                    run.error,
                    run.started_at.isoformat(),
                    run.finished_at.isoformat() if run.finished_at else None,
                    usage.input_tokens,
                    usage.output_tokens,
                    run.total_latency_ms,
                    run.model_dump_json(),
                ),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def save_runs(
        self, runs: Iterable[TaskRun], run_group: str, replace: bool = False
    ) -> list[int]:
        """Save a batch under ``run_group``.

        ``replace=True`` clears the group first — commands that write to
        fixed group names (curate's ``baseline``/``curated``) must not
        append across invocations, or report totals double-count.
        """
        if replace:
            with self._conn:
                self._conn.execute("DELETE FROM runs WHERE run_group = ?", (run_group,))
        return [self.save_run(run, run_group) for run in runs]

    def delete_group_family(self, base_group: str) -> int:
        """Delete ``base_group`` and every ``base_group-*`` variant.

        Repeated evals (--runs N) write suffixed groups (``baseline-1`` ..
        ``baseline-N``); a re-run with a smaller N must not leave stale
        repetitions behind, so replacement clears the whole family. Returns
        how many runs were deleted.
        """
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM runs WHERE run_group = ? OR run_group LIKE ?",
                (base_group, f"{base_group}-%"),
            )
        return cursor.rowcount

    def latest_runs_per_task(self, run_group: str) -> tuple[list[TaskRun], int]:
        """Load a group keeping only the most recent run per task.

        Returns (runs, dropped_count). Groups written more than once (or with
        --runs N) contain several runs per task; summing usage over all of
        them silently inflates token/cost totals in reports.
        """
        runs = self.load_runs(run_group)
        latest: dict[str, TaskRun] = {}
        for run in runs:  # load order is insertion order — last one wins
            latest[run.task_id] = run
        return list(latest.values()), len(runs) - len(latest)

    def load_runs(self, run_group: str | None = None) -> list[TaskRun]:
        query = "SELECT record_json FROM runs"
        params: tuple = ()
        if run_group is not None:
            query += " WHERE run_group = ?"
            params = (run_group,)
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        return [TaskRun.model_validate_json(row["record_json"]) for row in rows]

    def list_groups(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT run_group, COUNT(*) AS runs,
                   MIN(started_at) AS first_started, MAX(started_at) AS last_started
            FROM runs GROUP BY run_group ORDER BY MIN(id)
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TraceStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def write_jsonl(runs: Iterable[TaskRun], path: str | Path) -> None:
    """Write one JSON record per line — greppable and diffable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for run in runs:
            f.write(run.model_dump_json() + "\n")


def read_jsonl(path: str | Path) -> list[TaskRun]:
    with Path(path).open() as f:
        return [TaskRun.model_validate_json(line) for line in f if line.strip()]


def default_store_path(base_dir: str | Path = ".") -> Path:
    """Project-local default: ./.whetkit/traces.sqlite3 (gitignored)."""
    return Path(base_dir) / ".whetkit" / "traces.sqlite3"


def _summary_row(run: TaskRun) -> dict:
    usage = run.total_usage
    return {
        "task_id": run.task_id,
        "status": str(run.status),
        "tools": run.called_tool_names,
        "turns": len(run.turns),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "latency_ms": round(run.total_latency_ms, 1),
    }


def summarize_runs(runs: list[TaskRun]) -> str:
    return json.dumps([_summary_row(run) for run in runs], indent=2)
