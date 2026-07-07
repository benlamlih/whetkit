from pathlib import Path

import pytest

from whetkit.datasets import TaskSpec
from whetkit.llm import LLMTurn
from whetkit.scoring import (
    JudgeCache,
    JudgeConfig,
    MatchMode,
    judge_run,
    score_runs,
    score_tool_match,
)
from whetkit.tracing import TaskRun, ToolCallRecord, TurnRecord
from whetkit.tracing.records import RunStatus

from .fakes import FakeProvider


def task(expected: list, ordered: bool = False) -> TaskSpec:
    return TaskSpec(
        id="t",
        prompt="p",
        server="s",
        expected_tools=expected,
        ordered=ordered,
        success_criteria="c",
    )


class TestDeterministicScorer:
    def test_unordered_hit_with_extras(self) -> None:
        result = score_tool_match(task(["a", "b"]), ["b", "x", "a"])
        assert result.matched is True
        assert result.extra_calls == ["x"]
        assert result.precision == pytest.approx(2 / 3)
        assert result.recall == 1.0

    def test_unordered_miss(self) -> None:
        result = score_tool_match(task(["a", "b"]), ["a", "x"])
        assert result.matched is False
        assert result.missing_slots == [["b"]]
        assert result.recall == 0.5

    def test_alternatives_satisfy_slot(self) -> None:
        result = score_tool_match(task([["a1", "a2"]]), ["a2"])
        assert result.matched is True
        assert result.precision == 1.0

    def test_one_call_cannot_satisfy_two_slots(self) -> None:
        result = score_tool_match(task(["a", "a"]), ["a"])
        assert result.matched is False
        assert result.satisfied_slots == 1

    def test_ordered_requires_subsequence(self) -> None:
        ordered_task = task(["first", "second"], ordered=True)
        assert score_tool_match(ordered_task, ["first", "x", "second"]).matched is True
        assert score_tool_match(ordered_task, ["second", "first"]).matched is False

    def test_exact_mode(self) -> None:
        exact = MatchMode.EXACT
        assert score_tool_match(task(["a", "b"]), ["a", "b"], exact).matched is True
        assert score_tool_match(task(["a", "b"]), ["b", "a"], exact).matched is False
        assert score_tool_match(task(["a"]), ["a", "a"], exact).matched is False
        assert score_tool_match(task([["a1", "a2"]]), ["a2"], exact).matched is True

    def test_no_calls(self) -> None:
        result = score_tool_match(task(["a"]), [])
        assert result.matched is False
        assert result.precision == 0.0
        assert result.recall == 0.0


def make_run(tools: list[str], final_text: str | None = "done") -> TaskRun:
    return TaskRun(
        task_id="t",
        server="s",
        model="m",
        turns=[
            TurnRecord(
                index=0,
                tool_calls=[
                    ToolCallRecord(call_id=f"c{i}", name=name, result_text="ok")
                    for i, name in enumerate(tools)
                ],
            )
        ],
        final_text=final_text,
    )


JUDGE = JudgeConfig(model="fake:judge")


class TestJudge:
    async def test_verdict_parsed(self) -> None:
        provider = FakeProvider([LLMTurn(text='{"passed": true, "rationale": "answer states 5"}')])
        verdict = await judge_run(task(["a"]), make_run(["a"]), JUDGE, provider)
        assert verdict.passed is True
        assert verdict.valid is True
        assert verdict.rationale == "answer states 5"
        # the judge saw the rubric and the transcript
        prompt = provider.calls[0]["messages"][0].content
        assert "<success_criteria>" in prompt
        assert "a({})" in prompt

    async def test_unparseable_after_retry_fails_closed(self) -> None:
        provider = FakeProvider([LLMTurn(text="PASS!"), LLMTurn(text="definitely passed")])
        verdict = await judge_run(task(["a"]), make_run(["a"]), JUDGE, provider)
        assert verdict.passed is False
        assert verdict.valid is False
        assert len(provider.calls) == 2

    async def test_cache_hit_skips_provider(self, tmp_path: Path) -> None:
        cache = JudgeCache(tmp_path / "judge.sqlite3")
        provider = FakeProvider([LLMTurn(text='{"passed": false, "rationale": "wrong value"}')])
        run = make_run(["a"])

        first = await judge_run(task(["a"]), run, JUDGE, provider, cache)
        second = await judge_run(task(["a"]), run, JUDGE, provider, cache)
        assert first == second
        assert len(provider.calls) == 1  # second verdict came from the cache

        different_run = make_run(["a"], final_text="other answer")
        provider.script.append(LLMTurn(text='{"passed": true, "rationale": "ok"}'))
        third = await judge_run(task(["a"]), different_run, JUDGE, provider, cache)
        assert third.passed is True
        cache.close()


class TestAggregate:
    async def test_score_runs_hit_rate(self) -> None:
        tasks = [
            task(["a"]).model_copy(update={"id": "hit"}),
            task(["b"]).model_copy(update={"id": "miss"}),
            task(["c"]).model_copy(update={"id": "norun"}),
        ]
        runs = [
            make_run(["a"]).model_copy(update={"task_id": "hit"}),
            make_run(["x"]).model_copy(update={"task_id": "miss"}),
        ]
        summary = await score_runs(tasks, runs)
        assert summary.task_count == 3
        assert summary.hit_rate == pytest.approx(1 / 3)
        assert summary.tool_hit_rate == pytest.approx(1 / 3)
        assert summary.judge_pass_rate is None
        by_id = {s.task_id: s for s in summary.scores}
        assert by_id["norun"].run_status == RunStatus.ERROR
        assert any("Hit-rate: 33%" in line for line in summary.summary_lines())

    async def test_avg_extra_calls_counts_waste(self) -> None:
        tasks = [
            task(["a"]).model_copy(update={"id": "clean"}),
            task(["b"]).model_copy(update={"id": "wasteful"}),
        ]
        runs = [
            make_run(["a"]).model_copy(update={"task_id": "clean"}),
            make_run(["x", "y", "b"]).model_copy(update={"task_id": "wasteful"}),
        ]
        summary = await score_runs(tasks, runs)
        assert summary.avg_extra_calls == 1.0  # 0 + 2 extras over 2 tasks
        assert any("Unnecessary calls (avg): 1.0/task" in s for s in summary.summary_lines())

    async def test_name_map_translates_curated_calls(self) -> None:
        # A curated run calls overlay-presented names; expected_tools use
        # origin names. Without the map this scored 0% (the curate bug).
        tasks = [task(["proc_ord", "inv_check"], ordered=True)]
        runs = [make_run(["process_order", "check_inventory_quantity"])]

        unmapped = await score_runs(tasks, runs)
        assert unmapped.tool_hit_rate == 0.0

        summary = await score_runs(
            tasks,
            runs,
            name_map={"process_order": "proc_ord", "check_inventory_quantity": "inv_check"},
        )
        assert summary.tool_hit_rate == 1.0
        (score,) = summary.scores
        assert score.tool_match.called == ["proc_ord", "inv_check"]

    async def test_multi_run_summary_spread_and_flaky(self) -> None:
        from whetkit.scoring import MultiRunSummary

        tasks = [task(["a"]).model_copy(update={"id": "flaky"})]
        good = await score_runs(tasks, [make_run(["a"]).model_copy(update={"task_id": "flaky"})])
        bad = await score_runs(tasks, [make_run(["x"]).model_copy(update={"task_id": "flaky"})])

        multi = MultiRunSummary(summaries=[good, good, bad])
        assert multi.n == 3
        assert multi.flaky_tasks() == ["flaky"]
        lines = multi.summary_lines()
        assert any("Runs: 3" in line for line in lines)
        assert any("Hit-rate: 67% [0%–100%]" in line for line in lines)
        assert any("Flaky tasks" in line and "flaky" in line for line in lines)

        stable = MultiRunSummary(summaries=[good, good])
        assert stable.flaky_tasks() == []
        assert any("Hit-rate: 100%" in line for line in stable.summary_lines())

    async def test_spec_gap_flags_judge_pass_with_tool_miss(self) -> None:
        provider = FakeProvider(
            [
                LLMTurn(text='{"passed": true, "rationale": "answer is correct"}'),
                LLMTurn(text='{"passed": true, "rationale": "answer is correct"}'),
            ]
        )
        tasks = [task(["expected_tool"])]
        runs = [make_run(["unlisted_tool"])]
        summary = await score_runs(
            tasks, runs, judge_config=JUDGE, judge_provider=provider, use_judge=True
        )
        (score,) = summary.scores
        assert score.spec_gap is True
        assert score.hit is False  # still a miss until the spec is fixed

        # a plain miss (judge also failed) is NOT a spec gap
        provider2 = FakeProvider([LLMTurn(text='{"passed": false, "rationale": "wrong"}')])
        summary2 = await score_runs(
            tasks,
            [make_run(["unlisted_tool"], final_text="nope")],
            judge_config=JUDGE,
            judge_provider=provider2,
            use_judge=True,
        )
        assert summary2.scores[0].spec_gap is False

    async def test_judge_failure_blocks_hit(self) -> None:
        tasks = [task(["a"])]
        runs = [make_run(["a"])]
        provider = FakeProvider([LLMTurn(text='{"passed": false, "rationale": "bad answer"}')])
        summary = await score_runs(
            tasks, runs, judge_config=JUDGE, judge_provider=provider, use_judge=True
        )
        (score,) = summary.scores
        assert score.tool_hit is True
        assert score.hit is False
        assert summary.judge_pass_rate == 0.0
