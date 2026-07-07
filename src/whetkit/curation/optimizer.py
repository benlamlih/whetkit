"""The curation optimizer: an LLM looks at the server's tool inventory and
the failing eval traces, and proposes a safe overlay plan.

The model may only propose metadata changes (prune / rename / rewrite
descriptions; merges are expressed as hide-duplicates-keep-canonical), so a
bad proposal can degrade the overlay but can never break or mutate the
origin server. Every proposal is validated against the live tool list and
invalid entries are dropped with a warning.
"""

import json
import re

from pydantic import BaseModel

from whetkit.curation.plan import CurationPlan, ToolOverride
from whetkit.datasets import TaskSpec
from whetkit.llm import ChatMessage, LLMProvider, get_provider, parse_model
from whetkit.mcp.introspect import ServerInventory
from whetkit.scoring import TaskScore
from whetkit.tracing import TaskRun

DEFAULT_OPTIMIZER_MODEL = "anthropic:claude-sonnet-5"

OPTIMIZER_SYSTEM_PROMPT = """\
You are an expert MCP (Model Context Protocol) tool-set curator. You are
given the tools an MCP server exposes and traces of an AI agent failing (and
succeeding) at eval tasks that use this server. Agents fail when tool names
are cryptic, descriptions are vague, duplicates split their attention, or
noise tools distract them.

Propose a curation overlay that maximizes the agent's tool-selection
hit-rate. You may, per tool:
- "rename": give it a clear, specific snake_case name
- "rewrite": replace the description with one precise sentence saying what it
  does, over what data, and when to use it (mention key argument semantics)
- "prune": hide tools irrelevant to the eval tasks (admin/debug/noise)
- "merge": hide redundant duplicates and keep one canonical tool (express
  this as prune on the duplicates, rename/rewrite on the canonical one)
- "keep": leave a tool untouched

Rules:
- Only metadata changes. Never invent tools, change schemas, or alter behavior.
- Renamed tools MUST keep their exact argument schema.
- New names must be unique, snake_case, and descriptive (verb_object style).
- Do not prune a tool that any task needs.
- Rewrite descriptions for every tool you keep whose description is vague.
- Cost matters, not just correctness: when a trace shows the agent taking an
  expensive path (repeated calls, dump-everything tools, high token counts)
  where a cheaper targeted tool exists, write descriptions that steer the
  agent to the cheap path (say when to prefer this tool over its siblings).

Respond with ONLY a JSON object, no markdown fences:
{
  "notes": "one short paragraph on your strategy",
  "overrides": [
    {"original_name": "...", "action": "rename|rewrite|prune|keep",
     "new_name": "... or null", "new_description": "... or null",
     "reason": "one sentence"}
  ]
}
"""


class OptimizerConfig(BaseModel):
    model: str = DEFAULT_OPTIMIZER_MODEL
    max_tokens: int = 4096


def _inventory_block(inventory: ServerInventory) -> str:
    lines = []
    for tool in inventory.tools:
        schema = json.dumps(tool.input_schema.get("properties", {}), sort_keys=True)
        lines.append(f"- {tool.name}: {tool.description!r} | args: {schema}")
    return "\n".join(lines)


def _trace_block(tasks: list[TaskSpec], runs: list[TaskRun], scores: list[TaskScore]) -> str:
    tasks_by_id = {t.id: t for t in tasks}
    runs_by_id = {r.task_id: r for r in runs}
    lines = []
    for score in scores:
        task = tasks_by_id.get(score.task_id)
        run = runs_by_id.get(score.task_id)
        if task is None:
            continue
        called = " -> ".join(run.called_tool_names) if run else "(no run)"
        outcome = "HIT" if score.hit else "MISS"
        lines.append(f"### task {task.id} [{outcome}]")
        lines.append(f"user prompt: {task.prompt.strip()}")
        lines.append(f"tools called: {called or '(none)'}")
        if run is not None:
            usage = run.total_usage
            lines.append(
                f"cost: {len(run.called_tool_names)} call(s), "
                f"{usage.input_tokens}/{usage.output_tokens} tokens in/out"
            )
        if score.tool_match.missing_slots:
            lines.append(f"expected but never called (any of): {score.tool_match.missing_slots}")
        if score.tool_match.extra_calls:
            lines.append(f"unnecessary calls: {score.tool_match.extra_calls}")
        if score.judge is not None and not score.judge.passed:
            lines.append(f"judge: FAIL — {score.judge.rationale}")
    return "\n".join(lines)


def build_optimizer_prompt(
    inventory: ServerInventory,
    tasks: list[TaskSpec],
    runs: list[TaskRun],
    scores: list[TaskScore],
) -> str:
    return (
        f"## Tools exposed by the server\n{_inventory_block(inventory)}\n\n"
        f"## Eval traces\n{_trace_block(tasks, runs, scores)}\n\n"
        "Propose the curation overlay now."
    )


def _parse_overrides(raw: str) -> tuple[str, list[ToolOverride]] | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        overrides = []
        for entry in data.get("overrides", []):
            action = entry.get("action", "keep")
            if action == "keep" and not entry.get("new_name") and not entry.get("new_description"):
                continue
            overrides.append(
                ToolOverride(
                    original_name=str(entry["original_name"]),
                    new_name=entry.get("new_name") or None,
                    new_description=entry.get("new_description") or None,
                    hidden=action in ("prune", "hide"),
                    reason=str(entry.get("reason", "")),
                )
            )
        return str(data.get("notes", "")), overrides
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


async def propose_plan(
    inventory: ServerInventory,
    tasks: list[TaskSpec],
    runs: list[TaskRun],
    scores: list[TaskScore],
    config: OptimizerConfig | None = None,
    provider: LLMProvider | None = None,
) -> tuple[CurationPlan, list[str]]:
    """Ask the optimizer model for a plan. Returns (plan, warnings): entries
    that fail validation against the live tool list are dropped, never applied.
    """
    config = config or OptimizerConfig()
    provider_name, model_id = parse_model(config.model)
    provider = provider or get_provider(provider_name)

    prompt = build_optimizer_prompt(inventory, tasks, runs, scores)
    parsed = None
    for _attempt in range(2):
        turn = await provider.complete(
            model=model_id,
            system=OPTIMIZER_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=prompt)],
            tools=[],
            max_tokens=config.max_tokens,
        )
        parsed = _parse_overrides(turn.text or "")
        if parsed is not None:
            break
    if parsed is None:
        return CurationPlan(server=inventory.server), [
            "optimizer output was not valid JSON after 2 attempts; keeping origin tool set"
        ]

    notes, overrides = parsed
    plan = CurationPlan(server=inventory.server, notes=notes, overrides=overrides)

    origin_names = {t.name for t in inventory.tools}
    warnings: list[str] = []
    while problems := plan.validate_against(origin_names):
        # Drop offending overrides one problem at a time until the plan is safe.
        problem = problems[0]
        name_match = re.search(r"'([^']+)'", problem)
        offender = name_match.group(1) if name_match else None
        before = len(plan.overrides)
        plan.overrides = [
            o
            for o in plan.overrides
            if offender not in (o.original_name, o.new_name, o.presented_name)
        ]
        warnings.append(f"dropped unsafe override(s): {problem}")
        if len(plan.overrides) == before:  # nothing removable matched — bail out safely
            warnings.append("could not repair plan; keeping origin tool set")
            plan.overrides = []
            break
    return plan, warnings
