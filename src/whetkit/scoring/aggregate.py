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

    @property
    def tool_hit(self) -> bool:
        return self.tool_match.matched

    @property
    def hit(self) -> bool:
        """The headline metric: right tools AND (when judged) task success."""
        if self.judge is not None and not self.judge.passed:
            return False
        return self.tool_hit


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

    def summary_lines(self) -> list[str]:
        lines = [
            f"Tasks: {self.task_count}",
            f"Hit-rate: {self.hit_rate:.0%}",
            f"Tool-selection hit-rate: {self.tool_hit_rate:.0%}",
            f"Tool precision (avg): {self.avg_precision:.0%}",
            f"Tool recall (avg): {self.avg_recall:.0%}",
        ]
        if (rate := self.judge_pass_rate) is not None:
            lines.insert(2, f"Judge pass-rate: {rate:.0%}")
        return lines


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
        scores.append(
            TaskScore(
                task_id=task.id,
                run_status=run.status,
                tool_match=score_tool_match(task, called, mode),
                judge=verdict,
            )
        )
    return EvalSummary(scores=scores)
