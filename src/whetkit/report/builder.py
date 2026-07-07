"""Assemble the before/after comparison from two scored eval batches."""

from pydantic import BaseModel

from whetkit.curation.plan import CurationPlan, ToolOverride
from whetkit.datasets import TaskSpec
from whetkit.scoring import EvalSummary, TaskScore
from whetkit.tracing import TaskRun


class SideView(BaseModel):
    """One side (before or after) of a task comparison."""

    hit: bool
    tool_hit: bool
    judge_passed: bool | None = None
    judge_rationale: str | None = None
    tools_called: list[str] = []
    missing_slots: list[list[str]] = []
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @classmethod
    def from_score(cls, score: TaskScore, run: TaskRun | None) -> "SideView":
        usage = run.total_usage if run else None
        return cls(
            hit=score.hit,
            tool_hit=score.tool_hit,
            judge_passed=score.judge.passed if score.judge else None,
            judge_rationale=score.judge.rationale if score.judge else None,
            tools_called=score.tool_match.called,
            missing_slots=score.tool_match.missing_slots,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_ms=run.total_latency_ms if run else 0.0,
        )


class TaskComparison(BaseModel):
    task_id: str
    prompt: str
    expected_slots: list[list[str]]
    before: SideView
    after: SideView

    @property
    def outcome(self) -> str:
        if not self.before.hit and self.after.hit:
            return "improved"
        if self.before.hit and not self.after.hit:
            return "regressed"
        return "unchanged"


class ActionImpact(BaseModel):
    """One curation action and the tasks it plausibly affected.

    Attribution is heuristic: an override is linked to a task when the tool
    it touches appears in the task's expected slots or in either run's calls.
    """

    override: ToolOverride
    action: str
    touched_tasks: list[str] = []
    improved_tasks: list[str] = []


class Totals(BaseModel):
    hit_rate: float
    tool_hit_rate: float
    judge_pass_rate: float | None
    avg_precision: float
    avg_recall: float
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @classmethod
    def from_summary(cls, summary: EvalSummary, runs: list[TaskRun]) -> "Totals":
        return cls(
            hit_rate=summary.hit_rate,
            tool_hit_rate=summary.tool_hit_rate,
            judge_pass_rate=summary.judge_pass_rate,
            avg_precision=summary.avg_precision,
            avg_recall=summary.avg_recall,
            input_tokens=sum(r.total_usage.input_tokens for r in runs),
            output_tokens=sum(r.total_usage.output_tokens for r in runs),
            latency_ms=sum(r.total_latency_ms for r in runs),
        )


class ComparisonReport(BaseModel):
    title: str = "whetkit before/after report"
    server: str = ""
    model: str = ""
    before: Totals
    after: Totals
    tasks: list[TaskComparison]
    plan: CurationPlan
    action_impacts: list[ActionImpact]

    @property
    def improved(self) -> list[TaskComparison]:
        return [t for t in self.tasks if t.outcome == "improved"]

    @property
    def regressed(self) -> list[TaskComparison]:
        return [t for t in self.tasks if t.outcome == "regressed"]


def _override_action(override: ToolOverride) -> str:
    if override.hidden:
        return "prune"
    if override.new_name and override.new_description:
        return "rename + rewrite"
    if override.new_name:
        return "rename"
    if override.new_description:
        return "rewrite"
    return "keep"


def _attribute_actions(
    plan: CurationPlan, comparisons: list[TaskComparison], tasks: list[TaskSpec]
) -> list[ActionImpact]:
    tasks_by_id = {t.id: t for t in tasks}
    impacts: list[ActionImpact] = []
    for override in plan.overrides:
        names = {override.original_name, override.presented_name}
        impact = ActionImpact(override=override, action=_override_action(override))
        for comparison in comparisons:
            task = tasks_by_id.get(comparison.task_id)
            slot_names = {n for slot in (task.expected_tool_slots if task else []) for n in slot}
            involved = bool(
                names & slot_names
                or names & set(comparison.before.tools_called)
                or names & set(comparison.after.tools_called)
            )
            if involved:
                impact.touched_tasks.append(comparison.task_id)
                if comparison.outcome == "improved":
                    impact.improved_tasks.append(comparison.task_id)
        impacts.append(impact)
    return impacts


def build_report(
    tasks: list[TaskSpec],
    baseline_runs: list[TaskRun],
    baseline_summary: EvalSummary,
    curated_runs: list[TaskRun],
    curated_summary: EvalSummary,
    plan: CurationPlan,
    model: str = "",
    server: str = "",
) -> ComparisonReport:
    baseline_runs_by_id = {r.task_id: r for r in baseline_runs}
    curated_runs_by_id = {r.task_id: r for r in curated_runs}
    baseline_scores = {s.task_id: s for s in baseline_summary.scores}
    curated_scores = {s.task_id: s for s in curated_summary.scores}

    comparisons: list[TaskComparison] = []
    for task in tasks:
        before_score = baseline_scores.get(task.id)
        after_score = curated_scores.get(task.id)
        if before_score is None or after_score is None:
            continue
        comparisons.append(
            TaskComparison(
                task_id=task.id,
                prompt=task.prompt.strip(),
                expected_slots=task.expected_tool_slots,
                before=SideView.from_score(before_score, baseline_runs_by_id.get(task.id)),
                after=SideView.from_score(after_score, curated_runs_by_id.get(task.id)),
            )
        )

    return ComparisonReport(
        server=server,
        model=model,
        before=Totals.from_summary(baseline_summary, baseline_runs),
        after=Totals.from_summary(curated_summary, curated_runs),
        tasks=comparisons,
        plan=plan,
        action_impacts=_attribute_actions(plan, comparisons, tasks),
    )
