"""Deterministic tool-selection scoring.

Compares the tools an agent actually called against a task's
``expected_tools`` slots (each slot lists acceptable alternatives).

Two modes:

- ``order_tolerant`` (default): every slot must be satisfied by a distinct
  call; extra calls are allowed (they cost precision, not the hit). When the
  task sets ``ordered: true``, the satisfying calls must appear as a
  subsequence of the call sequence.
- ``exact``: the call sequence must be exactly one call per slot, in order,
  with no extra calls.
"""

from enum import StrEnum

from pydantic import BaseModel

from whetkit.datasets import TaskSpec


class MatchMode(StrEnum):
    EXACT = "exact"
    ORDER_TOLERANT = "order_tolerant"


class ToolMatchResult(BaseModel):
    matched: bool
    mode: MatchMode
    expected_slots: list[list[str]]
    called: list[str]
    satisfied_slots: int
    missing_slots: list[list[str]]
    extra_calls: list[str]
    precision: float
    recall: float


def _assign_unordered(slots: list[list[str]], called: list[str]) -> list[int | None]:
    """Greedily assign calls to slots ignoring order. Returns, per call, the
    slot index it satisfied (or None if it satisfied nothing)."""
    remaining = set(range(len(slots)))
    assignment: list[int | None] = []
    for name in called:
        hit = next((i for i in sorted(remaining) if name in slots[i]), None)
        if hit is not None:
            remaining.discard(hit)
        assignment.append(hit)
    return assignment


def _assign_ordered(slots: list[list[str]], called: list[str]) -> list[int | None]:
    """Greedily match slots as a subsequence of the call sequence."""
    next_slot = 0
    assignment: list[int | None] = []
    for name in called:
        if next_slot < len(slots) and name in slots[next_slot]:
            assignment.append(next_slot)
            next_slot += 1
        else:
            assignment.append(None)
    return assignment


def score_tool_match(
    task: TaskSpec,
    called: list[str],
    mode: MatchMode = MatchMode.ORDER_TOLERANT,
) -> ToolMatchResult:
    slots = task.expected_tool_slots

    exact_ok = len(called) == len(slots) and all(
        name in slot for name, slot in zip(called, slots, strict=True)
    )
    if exact_ok:
        assignment: list[int | None] = list(range(len(called)))
    else:
        assignment = (_assign_ordered if task.ordered else _assign_unordered)(slots, called)

    satisfied = {i for i in assignment if i is not None}
    missing = [slot for i, slot in enumerate(slots) if i not in satisfied]
    extras = [name for name, hit in zip(called, assignment, strict=True) if hit is None]

    precision = (len(called) - len(extras)) / len(called) if called else 0.0
    recall = len(satisfied) / len(slots) if slots else 1.0
    matched = exact_ok if mode == MatchMode.EXACT else not missing

    return ToolMatchResult(
        matched=matched,
        mode=mode,
        expected_slots=slots,
        called=called,
        satisfied_slots=len(satisfied),
        missing_slots=missing,
        extra_calls=extras,
        precision=precision,
        recall=recall,
    )
