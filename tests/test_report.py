import json
import re

from whetkit.curation import CurationPlan, ToolOverride
from whetkit.datasets import TaskSpec
from whetkit.report import build_report, render_html
from whetkit.scoring import EvalSummary, TaskScore, score_tool_match
from whetkit.tracing import TaskRun, ToolCallRecord, TurnRecord
from whetkit.tracing.records import RunStatus


def task(task_id: str, expected: list) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        prompt=f"do {task_id}",
        server="s",
        expected_tools=expected,
        success_criteria="did it",
    )


def run_for(task_id: str, tools: list[str], in_tokens: int = 100) -> TaskRun:
    from whetkit.llm import Usage

    return TaskRun(
        task_id=task_id,
        server="s",
        model="anthropic:claude-sonnet-5",
        turns=[
            TurnRecord(
                index=0,
                usage=Usage(input_tokens=in_tokens, output_tokens=20),
                latency_ms=500.0,
                tool_calls=[
                    ToolCallRecord(call_id=f"c{i}", name=name, result_text="ok", latency_ms=10.0)
                    for i, name in enumerate(tools)
                ],
            )
        ],
        final_text="done",
    )


def summary_for(tasks: list[TaskSpec], runs: list[TaskRun]) -> EvalSummary:
    runs_by_id = {r.task_id: r for r in runs}
    return EvalSummary(
        scores=[
            TaskScore(
                task_id=t.id,
                run_status=RunStatus.COMPLETED,
                tool_match=score_tool_match(t, runs_by_id[t.id].called_tool_names),
            )
            for t in tasks
        ]
    )


PLAN = CurationPlan(
    server="s",
    notes="renamed the cryptic search tool",
    overrides=[
        ToolOverride(
            original_name="data_query_1",
            new_name="search_products",
            new_description="Search products.",
            reason="cryptic",
        ),
        ToolOverride(original_name="sys_ping", hidden=True, reason="noise"),
    ],
)


def build_fixture_report():
    tasks = [task("flips", ["data_query_1"]), task("stays-bad", ["cust_upd"])]
    baseline_runs = [
        run_for("flips", ["sys_ping", "data_query_2"], in_tokens=900),
        run_for("stays-bad", ["get_rec"]),
    ]
    curated_runs = [
        run_for("flips", ["search_products"], in_tokens=300),
        run_for("stays-bad", ["fetch_record"]),
    ]

    # curated runs call the *presented* names; score them against curated names
    curated_tasks = [task("flips", ["search_products"]), task("stays-bad", ["cust_upd"])]
    return (
        build_report(
            tasks,
            baseline_runs,
            summary_for(tasks, baseline_runs),
            curated_runs,
            summary_for(curated_tasks, curated_runs),
            PLAN,
            model="anthropic:claude-sonnet-5",
            server="stdio: sample",
        ),
        tasks,
    )


class TestBuildReport:
    def test_outcomes_and_metrics(self) -> None:
        report, _ = build_fixture_report()

        assert report.before.hit_rate == 0.0
        assert report.after.hit_rate == 0.5
        by_id = {t.task_id: t for t in report.tasks}
        assert by_id["flips"].outcome == "improved"
        assert by_id["stays-bad"].outcome == "unchanged"
        assert [t.task_id for t in report.improved] == ["flips"]
        assert report.regressed == []

        # token/latency deltas come from the traces
        assert report.before.input_tokens == 1000
        assert report.after.input_tokens == 400
        assert report.before.latency_ms > 0

    def test_action_attribution(self) -> None:
        report, _ = build_fixture_report()
        impacts = {i.override.original_name: i for i in report.action_impacts}

        rename = impacts["data_query_1"]
        assert rename.action == "rename + rewrite"
        assert "flips" in rename.touched_tasks
        assert rename.improved_tasks == ["flips"]

        prune = impacts["sys_ping"]
        assert prune.action == "prune"
        assert prune.touched_tasks == ["flips"]  # it was called in the baseline run

    def test_json_roundtrip(self) -> None:
        report, _ = build_fixture_report()
        data = json.loads(report.model_dump_json())
        assert data["before"]["hit_rate"] == 0.0
        assert data["after"]["hit_rate"] == 0.5
        assert len(data["tasks"]) == 2
        assert data["plan"]["overrides"][0]["new_name"] == "search_products"

    def test_spread_strings_carried_through(self) -> None:
        tasks = [task("flips", ["data_query_1"])]
        runs = [run_for("flips", ["data_query_1"])]
        summary = summary_for(tasks, runs)
        report = build_report(
            tasks,
            runs,
            summary,
            runs,
            summary,
            PLAN,
            before_spread="50% [0%–100%]",
            after_spread="100%",
        )
        assert report.before_spread == "50% [0%–100%]"
        assert report.after_spread == "100%"
        data = json.loads(report.model_dump_json())
        assert data["before_spread"] == "50% [0%–100%]"


class TestRenderHtml:
    def test_self_contained_and_complete(self) -> None:
        report, _ = build_fixture_report()
        html_text = render_html(report)

        assert html_text.startswith("<!DOCTYPE html>")
        # self-contained: no scripts, no external stylesheets/fonts/images
        assert "<script" not in html_text
        assert "<link" not in html_text
        assert "@import" not in html_text
        assert not re.search(r'src\s*=\s*["\']https?://', html_text)

        # headline: 0% -> 50% hit-rate, +50 points delta
        assert "before curation" in html_text
        assert "after curation" in html_text
        assert "+50" in html_text
        assert "TOOL-SELECTION ACCURACY" in html_text

        # per-task table, curation cards, and traces are present
        assert "do stays-bad" in html_text
        assert "search_products" in html_text
        assert "PRUNED · 1" in html_text
        assert "RENAMED · 1" in html_text
        assert "improved on 1 task" in html_text
        assert "renamed the cryptic search tool" in html_text
        assert "BEFORE · raw tool set" in html_text
        assert '<details class="row" open' in html_text  # first improved task starts expanded

    def test_escapes_untrusted_text(self) -> None:
        report, _ = build_fixture_report()
        report.plan.notes = '<script>alert("x")</script>'
        html_text = render_html(report)
        assert "<script>alert" not in html_text
        assert "&lt;script&gt;" in html_text

    def test_multi_run_headline_ranges_and_caveat(self) -> None:
        tasks = [task("flips", ["data_query_1"])]
        hit_run = run_for("flips", ["data_query_1"])
        miss_run = run_for("flips", ["sys_ping"])
        hit = summary_for(tasks, [hit_run])
        miss = summary_for(tasks, [miss_run])

        report = build_report(
            tasks,
            [miss_run],
            miss,
            [hit_run],
            hit,
            PLAN,
            baseline_summaries=[miss, hit],  # 50% mean [0–100%]
            curated_summaries=[hit, hit],  # 100%, ranges overlap with before
            est_cost_usd=0.42,
        )
        html_text = render_html(report)
        assert "Runs: 2 × 1 tasks" in html_text
        assert "[0–100%]" in html_text
        assert "ranges overlap" in html_text  # the noise caveat
        assert "EST. COST" in html_text and "$0.42" in html_text
        # per-task repetition cells
        assert "⚡ 1/2" in html_text
        assert "✓ 2/2" in html_text

    def test_warnings_strip_only_when_nonzero(self) -> None:
        report, _ = build_fixture_report()
        assert "runs errored" not in render_html(report)
        report.warnings = ["⚠ 1/4 runs timed out — raise --task-timeout"]
        html_text = render_html(report)
        assert "runs timed out" in html_text

    def test_judge_and_spec_gap_render(self) -> None:
        report, _ = build_fixture_report()
        flips = next(t for t in report.tasks if t.task_id == "flips")
        flips.after.judge_passed = True
        flips.after.judge_rationale = "order cancelled correctly"
        flips.after.spec_gap = True
        html_text = render_html(report)
        assert "judge ›" in html_text
        assert "order cancelled correctly" in html_text
        assert "⚠ spec-gap" in html_text
        assert "expected_tools may be incomplete" in html_text
