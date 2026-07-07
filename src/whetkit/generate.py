"""Draft eval tasks from a server's tool inventory.

Writing task YAML by hand is the slowest step of adopting whetkit. Here an
LLM drafts candidate tasks from the tool names, descriptions, and schemas;
every draft is validated against :class:`TaskSpec` and against the live
tool list, and the output file says loudly that a human should review it.
"""

import json
import re

import yaml
from pydantic import BaseModel

from whetkit.datasets import TaskSpec
from whetkit.llm import ChatMessage, LLMProvider, get_provider, parse_model
from whetkit.mcp.introspect import ServerInventory

DEFAULT_GENERATOR_MODEL = "anthropic:claude-sonnet-5"

GENERATOR_SYSTEM_PROMPT = """\
You write eval tasks for an AI-agent benchmark. You are given the tools an
MCP server exposes. Draft tasks that measure whether an agent picks the
right tools for realistic user requests.

Rules for every task:
- The prompt reads like something a real user would type. It must NEVER
  mention tool names or hint at which tool to use — finding the tool is
  what the eval measures.
- expected_tools lists the calls a correct run makes, in order. Each entry
  is either one tool name, or a list of genuinely interchangeable
  alternatives for that step. Use ONLY tool names from the given list.
- success_criteria is one or two concrete, checkable sentences a grader can
  verify from the agent's final answer alone.
- Prefer read-only tasks. Include a write task only when the tool set is
  clearly built for writes, and never anything destructive or irreversible
  (no deletes, resets, or purges).
- Set "ordered": true only when the steps must happen in sequence.
- ids are short kebab-case slugs, unique across the set.

Respond with ONLY a JSON array, no markdown fences:
[
  {"id": "...", "prompt": "...", "expected_tools": ["tool" or ["a", "b"]],
   "ordered": false, "success_criteria": "..."}
]
"""


class GeneratorConfig(BaseModel):
    model: str = DEFAULT_GENERATOR_MODEL
    max_tokens: int = 4096


def _inventory_block(inventory: ServerInventory) -> str:
    lines = []
    for tool in inventory.tools:
        schema = json.dumps(tool.input_schema.get("properties", {}), sort_keys=True)
        lines.append(f"- {tool.name}: {tool.description!r} | args: {schema}")
    return "\n".join(lines)


def build_generator_prompt(inventory: ServerInventory, count: int) -> str:
    return (
        f"## Tools exposed by the server\n{_inventory_block(inventory)}\n\n"
        f"Draft exactly {count} eval tasks now."
    )


def _validate_draft(
    raw: dict, server: str, known_tools: set[str], seen_ids: set[str]
) -> tuple[TaskSpec | None, list[str]]:
    """Turn one raw draft into a TaskSpec, or explain why it was dropped."""
    warnings: list[str] = []
    draft_id = str(raw.get("id", "?"))

    slots = []
    for slot in raw.get("expected_tools", []):
        alternatives = [slot] if isinstance(slot, str) else list(slot)
        valid = [name for name in alternatives if name in known_tools]
        if unknown := [name for name in alternatives if name not in known_tools]:
            warnings.append(f"task {draft_id!r}: dropped unknown tool(s) {unknown}")
        if not valid:
            warnings.append(f"task {draft_id!r}: dropped — a step has no valid tool left")
            return None, warnings
        slots.append(valid[0] if len(valid) == 1 else valid)

    if draft_id in seen_ids:
        warnings.append(f"task {draft_id!r}: dropped — duplicate id")
        return None, warnings

    try:
        task = TaskSpec.model_validate({**raw, "server": server, "expected_tools": slots})
    except Exception as exc:
        warnings.append(f"task {draft_id!r}: dropped — invalid: {exc}")
        return None, warnings
    return task, warnings


async def generate_tasks(
    inventory: ServerInventory,
    server: str,
    count: int = 5,
    config: GeneratorConfig | None = None,
    provider: LLMProvider | None = None,
) -> tuple[list[TaskSpec], list[str]]:
    """Draft ``count`` tasks for ``server``. Returns (tasks, warnings);
    drafts that fail validation are dropped, never written."""
    config = config or GeneratorConfig()
    provider_name, model_id = parse_model(config.model)
    provider = provider or get_provider(provider_name)

    prompt = build_generator_prompt(inventory, count)
    drafts = None
    for _attempt in range(2):
        turn = await provider.complete(
            model=model_id,
            system=GENERATOR_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=prompt)],
            tools=[],
            max_tokens=config.max_tokens,
        )
        match = re.search(r"\[.*\]", turn.text or "", re.DOTALL)
        if match:
            try:
                drafts = json.loads(match.group(0))
                break
            except json.JSONDecodeError:
                drafts = None
    if not isinstance(drafts, list):
        return [], ["generator output was not a valid JSON array after 2 attempts"]

    known_tools = {t.name for t in inventory.tools}
    tasks: list[TaskSpec] = []
    warnings: list[str] = []
    for raw in drafts:
        if not isinstance(raw, dict):
            warnings.append("dropped a non-mapping draft entry")
            continue
        task, task_warnings = _validate_draft(raw, server, known_tools, {t.id for t in tasks})
        warnings.extend(task_warnings)
        if task is not None:
            tasks.append(task)
    return tasks, warnings


def write_tasks_yaml(tasks: list[TaskSpec], path: str) -> None:
    """Write tasks as one reviewable YAML list, loadable by ``load_tasks``."""
    header = (
        "# Drafted by 'whetkit generate' — review before trusting.\n"
        "# Check that expected_tools really are the right calls and that\n"
        "# success_criteria name facts your server actually returns.\n"
    )
    body = yaml.safe_dump(
        [task.model_dump(exclude_defaults=True) for task in tasks],
        sort_keys=False,
        allow_unicode=True,
    )
    with open(path, "w") as fh:
        fh.write(header + body)
