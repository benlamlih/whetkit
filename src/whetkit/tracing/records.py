"""Structured records of an agent run: every turn, tool call, and outcome.

These are the atoms the scorer, curator, and report all consume.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from whetkit.llm.base import Usage


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    COMPLETED = "completed"  # model produced a final answer
    MAX_TURNS = "max_turns"  # loop hit the turn limit before finishing
    ERROR = "error"  # provider/transport failure aborted the run


class ToolCallRecord(BaseModel):
    """One tool invocation and its outcome."""

    call_id: str
    name: str
    arguments: dict[str, Any] = {}
    result_text: str = ""
    is_error: bool = False
    latency_ms: float = 0.0


class TurnRecord(BaseModel):
    """One model completion plus the tool calls it triggered."""

    index: int
    assistant_text: str | None = None
    tool_calls: list[ToolCallRecord] = []
    usage: Usage = Usage()
    latency_ms: float = 0.0
    stop_reason: str | None = None


class TaskRun(BaseModel):
    """A full agent run of one task against one server."""

    task_id: str
    server: str
    model: str
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    turns: list[TurnRecord] = []
    final_text: str | None = None
    status: RunStatus = RunStatus.COMPLETED
    error: str | None = None

    @property
    def called_tool_names(self) -> list[str]:
        """Every tool called, in call order across turns."""
        return [call.name for turn in self.turns for call in turn.tool_calls]

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for turn in self.turns:
            total = total + turn.usage
        return total

    @property
    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.turns) + sum(
            c.latency_ms for t in self.turns for c in t.tool_calls
        )
