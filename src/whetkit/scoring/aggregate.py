"""Aggregation: per-task scores and eval-level hit-rate metrics."""

from pydantic import BaseModel

from whetkit.datasets import TaskSpec
from whetkit.llm import LLMProvider
from whetkit.scoring.deterministic import MatchMode, ToolMatchResult, score_tool_match
from whetkit.scoring.judge import JudgeCache, JudgeConfig, JudgeVerdict, judge_run
from whetkit.tracing import TaskRun
from whetkit.tracing.records import RunStatus


class TaskScore(BaseModel):
    task_id: str
    run_status: RunStatus
    tool_match: ToolMatchResult
    judge: JudgeVerdict | None = None
    tool_errors: int = 0

    @property
    def tool_hit(self) -> bool:
        return self.tool_match.matched

    @property
    def hit(self) -> bool:
        """The headline metric: right tools AND (when judged) task success."""
        if self.judge is not None and not self.judge.passed:
            return False
        return self.tool_hit

    @property
    def spec_gap(self) -> bool:
        """Judge passed but the tool match failed: the agent reached a
        correct outcome via tools the task never listed. Nine times out of
        ten that means expected_tools is incomplete, not that the agent
        failed."""
        return self.judge is not None and self.judge.passed and not self.tool_hit


class EvalSummary(BaseModel):
    scores: list[TaskScore]

    @property
    def task_count(self) -> int:
        return len(self.scores)

    @property
    def hit_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.hit for s in self.scores) / len(self.scores)

    @property
    def tool_hit_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.tool_hit for s in self.scores) / len(self.scores)

    @property
    def judge_pass_rate(self) -> float | None:
        judged = [s for s in self.scores if s.judge is not None]
        if not judged:
            return None
        return sum(s.judge.passed for s in judged) / len(judged)

    @property
    def avg_precision(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.tool_match.precision for s in self.scores) / len(self.scores)

    @property
    def avg_recall(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.tool_match.recall for s in self.scores) / len(self.scores)

    @property
    def avg_extra_calls(self) -> float:
        """Unnecessary tool calls per task — a passing task can still waste
        calls (and tokens) looping through the wrong tools first."""
        if not self.scores:
            return 0.0
        return sum(len(s.tool_match.extra_calls) for s in self.scores) / len(self.scores)

    @property
    def error_run_count(self) -> int:
        return sum(s.run_status == RunStatus.ERROR for s in self.scores)

    @property
    def total_tool_errors(self) -> int:
        return sum(s.tool_errors for s in self.scores)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Tasks: {self.task_count}",
            f"Hit-rate: {self.hit_rate:.0%}",
            f"Tool-selection hit-rate: {self.tool_hit_rate:.0%}",
            f"Tool precision (avg): {self.avg_precision:.0%}",
            f"Tool recall (avg): {self.avg_recall:.0%}",
            f"Unnecessary calls (avg): {self.avg_extra_calls:.1f}/task",
        ]
        if (rate := self.judge_pass_rate) is not None:
            lines.insert(2, f"Judge pass-rate: {rate:.0%}")
        if self.error_run_count:
            lines.append(
                f"⚠ Errored runs: {self.error_run_count}/{self.task_count} — the agent "
                "loop failed (connection/provider); scores below reflect failures, "
                "not tool selection"
            )
        if self.total_tool_errors:
            lines.append(
                f"⚠ Failed tool calls: {self.total_tool_errors} — agents received "
                "errors from the server; read the traces before trusting the scores"
            )
        return lines


class MultiRunSummary(BaseModel):
    """Aggregate over repeated executions of the same task set.

    Single runs are noise: the same server, tasks, and model can flip a
    task between runs. The honest headline is mean plus range."""

    summaries: list[EvalSummary]

    @property
    def n(self) -> int:
        return len(self.summaries)

    def _spread(self, values: list[float]) -> str:
        mean = sum(values) / len(values)
        low, high = min(values), max(values)
        if low == high:
            return f"{mean:.0%}"
        return f"{mean:.0%} [{low:.0%}–{high:.0%}]"

    def summary_lines(self) -> list[str]:
        lines = [
            f"Runs: {self.n} × {self.summaries[0].task_count} task(s)",
            f"Hit-rate: {self._spread([s.hit_rate for s in self.summaries])}",
            f"Tool-selection hit-rate: {self._spread([s.tool_hit_rate for s in self.summaries])}",
            f"Tool precision (avg): {self._spread([s.avg_precision for s in self.summaries])}",
        ]
        judged = [s.judge_pass_rate for s in self.summaries if s.judge_pass_rate is not None]
        if judged:
            lines.insert(2, f"Judge pass-rate: {self._spread(judged)}")
        extras = [s.avg_extra_calls for s in self.summaries]
        lines.append(f"Unnecessary calls (avg): {sum(extras) / len(extras):.1f}/task")
        flaky = self.flaky_tasks()
        if flaky:
            lines.append(f"Flaky tasks (hit in some runs, missed in others): {', '.join(flaky)}")
        return lines

    def flaky_tasks(self) -> list[str]:
        """Task ids whose hit outcome differs across runs."""
        outcomes: dict[str, set[bool]] = {}
        for summary in self.summaries:
            for score in summary.scores:
                outcomes.setdefault(score.task_id, set()).add(score.hit)
        return sorted(task_id for task_id, seen in outcomes.items() if len(seen) > 1)


async def score_runs(
    tasks: list[TaskSpec],
    runs: list[TaskRun],
    mode: MatchMode = MatchMode.ORDER_TOLERANT,
    judge_config: JudgeConfig | None = None,
    judge_provider: LLMProvider | None = None,
    judge_cache: JudgeCache | None = None,
    use_judge: bool = False,
    name_map: dict[str, str] | None = None,
) -> EvalSummary:
    """Score each run against its task. Runs and tasks are matched by id;
    a task with no run is scored as a total miss.

    ``name_map`` translates called tool names before matching (curated runs
    call overlay-presented names, while ``expected_tools`` use origin names)."""
    runs_by_task = {run.task_id: run for run in runs}
    scores: list[TaskScore] = []
    for task in tasks:
        run = runs_by_task.get(task.id)
        if run is None:
            scores.append(
                TaskScore(
                    task_id=task.id,
                    run_status=RunStatus.ERROR,
                    tool_match=score_tool_match(task, [], mode),
                )
            )
            continue
        verdict = None
        if use_judge:
            verdict = await judge_run(task, run, judge_config, judge_provider, judge_cache)
        called = run.called_tool_names
        if name_map:
            called = [name_map.get(name, name) for name in called]
        tool_errors = sum(call.is_error for turn in run.turns for call in turn.tool_calls)
        scores.append(
            TaskScore(
                task_id=task.id,
                run_status=run.status,
                tool_match=score_tool_match(task, called, mode),
                judge=verdict,
                tool_errors=tool_errors,
            )
        )
    return EvalSummary(scores=scores)
