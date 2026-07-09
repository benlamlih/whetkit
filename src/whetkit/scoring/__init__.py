"""Scoring: deterministic tool-selection matching + LLM-as-judge grading."""

from whetkit.scoring.aggregate import (
    EvalSummary,
    MultiRunSummary,
    TaskScore,
    hit_rate_noise_caveat,
    score_runs,
)
from whetkit.scoring.deterministic import MatchMode, ToolMatchResult, score_tool_match
from whetkit.scoring.judge import JudgeCache, JudgeConfig, JudgeVerdict, judge_run

__all__ = [
    "EvalSummary",
    "MultiRunSummary",
    "hit_rate_noise_caveat",
    "JudgeCache",
    "JudgeConfig",
    "JudgeVerdict",
    "MatchMode",
    "TaskScore",
    "ToolMatchResult",
    "judge_run",
    "score_runs",
    "score_tool_match",
]
