"""LLM-as-judge grading of task success against the task's rubric.

The judge sees the task, the rubric, a compact transcript of tool calls and
results, and the agent's final answer. It returns a structured verdict.
Judgments are cached in SQLite keyed by a content hash, so re-scoring the
same run with the same judge model costs nothing.
"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from pydantic import BaseModel

from mcp_eval.datasets import TaskSpec
from mcp_eval.llm import ChatMessage, LLMProvider, get_provider, parse_model
from mcp_eval.tracing import TaskRun

DEFAULT_JUDGE_MODEL = "anthropic:claude-sonnet-5"

JUDGE_SYSTEM_PROMPT = """\
You are a strict, impartial grader of AI agent runs. You are given a task a
user asked an agent to do, the success criteria a correct run must satisfy,
the tool calls the agent made (with results), and the agent's final answer.

Grade ONLY against the success criteria:
- If the criteria name specific facts or values, the final answer must state
  them (equivalent phrasing and formatting are fine; wrong or missing values
  are a fail).
- Actions count only if the transcript shows they actually happened; the
  agent claiming success without a supporting tool result is a fail.
- Do not reward effort, verbosity, or plausible-sounding but unverified
  claims. Do not penalize style.

Respond with ONLY a JSON object, no markdown fences:
{"passed": true or false, "rationale": "one or two sentences citing the evidence"}
"""


class JudgeVerdict(BaseModel):
    passed: bool
    rationale: str
    judge_model: str
    valid: bool = True  # False when the judge's output could not be parsed


class JudgeConfig(BaseModel):
    model: str = DEFAULT_JUDGE_MODEL
    max_tokens: int = 512


class JudgeCache:
    """Content-addressed cache of judge verdicts (SQLite key/value)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS judgments (key TEXT PRIMARY KEY, verdict TEXT NOT NULL)"
            )

    def get(self, key: str) -> JudgeVerdict | None:
        row = self._conn.execute("SELECT verdict FROM judgments WHERE key = ?", (key,)).fetchone()
        return JudgeVerdict.model_validate_json(row[0]) if row else None

    def put(self, key: str, verdict: JudgeVerdict) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO judgments (key, verdict) VALUES (?, ?)",
                (key, verdict.model_dump_json()),
            )

    def close(self) -> None:
        self._conn.close()


def _transcript(run: TaskRun, max_result_chars: int = 500) -> str:
    lines: list[str] = []
    for turn in run.turns:
        for call in turn.tool_calls:
            result = call.result_text
            if len(result) > max_result_chars:
                result = result[:max_result_chars] + "…(truncated)"
            status = "ERROR" if call.is_error else "ok"
            lines.append(f"- {call.name}({json.dumps(call.arguments)}) [{status}] -> {result}")
    return "\n".join(lines) or "(the agent made no tool calls)"


def build_judge_prompt(task: TaskSpec, run: TaskRun) -> str:
    return (
        f"<task>\n{task.prompt}\n</task>\n\n"
        f"<success_criteria>\n{task.success_criteria}\n</success_criteria>\n\n"
        f"<tool_calls>\n{_transcript(run)}\n</tool_calls>\n\n"
        f"<final_answer>\n{run.final_text or '(the agent never gave a final answer)'}\n"
        f"</final_answer>"
    )


def _cache_key(task: TaskSpec, run: TaskRun, judge_model: str) -> str:
    payload = json.dumps(
        {
            "judge_model": judge_model,
            "task_id": task.id,
            "criteria": task.success_criteria,
            "prompt": build_judge_prompt(task, run),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_verdict(raw: str, judge_model: str) -> JudgeVerdict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return JudgeVerdict(
            passed=bool(data["passed"]),
            rationale=str(data.get("rationale", "")),
            judge_model=judge_model,
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


async def judge_run(
    task: TaskSpec,
    run: TaskRun,
    config: JudgeConfig | None = None,
    provider: LLMProvider | None = None,
    cache: JudgeCache | None = None,
) -> JudgeVerdict:
    """Grade one run. Never raises on judge misbehavior: an unparseable
    verdict comes back as ``passed=False, valid=False``."""
    config = config or JudgeConfig()
    key = _cache_key(task, run, config.model)
    if cache is not None and (cached := cache.get(key)) is not None:
        return cached

    provider_name, model_id = parse_model(config.model)
    provider = provider or get_provider(provider_name)

    verdict: JudgeVerdict | None = None
    for _attempt in range(2):
        turn = await provider.complete(
            model=model_id,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=build_judge_prompt(task, run))],
            tools=[],
            max_tokens=config.max_tokens,
        )
        verdict = _parse_verdict(turn.text or "", config.model)
        if verdict is not None:
            break
    if verdict is None:
        verdict = JudgeVerdict(
            passed=False,
            rationale="Judge output was not valid JSON after 2 attempts.",
            judge_model=config.model,
            valid=False,
        )

    if cache is not None and verdict.valid:
        cache.put(key, verdict)
    return verdict
