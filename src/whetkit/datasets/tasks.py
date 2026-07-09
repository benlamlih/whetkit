"""Task schema and YAML loader.

A task file is YAML holding either a single task mapping or a list of task
mappings. The format is documented in docs/task-format.md; the source of
truth for validation is :class:`TaskSpec`.
"""

import difflib
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class TaskSpec(BaseModel):
    """One eval task: a user request plus what a correct agent run looks like."""

    # A typo'd field (orderd: true) must fail loudly, not silently drop the
    # constraint it was meant to set.
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str = Field(min_length=1)
    server: str = Field(min_length=1, description="MCP server: URL, directory, or file path")
    expected_tools: list[str | list[str]] = Field(
        min_length=1,
        description=(
            "Tool calls a correct run makes. Each entry is one expected call; "
            "an entry may list acceptable alternatives for that call."
        ),
    )
    ordered: bool = Field(
        default=False,
        description="If true, expected calls must happen in the listed order.",
    )
    success_criteria: str = Field(
        min_length=1,
        description="Natural-language rubric the LLM judge grades the final answer against.",
    )
    tags: list[str] = []

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(
                f"task id {v!r} must be lowercase alphanumeric with '-' or '_' separators"
            )
        return v

    @field_validator("expected_tools")
    @classmethod
    def _non_empty_slots(cls, v: list[str | list[str]]) -> list[str | list[str]]:
        for slot in v:
            if isinstance(slot, list) and not slot:
                raise ValueError("an expected_tools alternatives list may not be empty")
        return v

    @property
    def expected_tool_slots(self) -> list[list[str]]:
        """expected_tools normalized so every slot is a list of alternatives."""
        return [[s] if isinstance(s, str) else list(s) for s in self.expected_tools]

    def resolve_server(self, base_dir: Path | None = None) -> str:
        """Resolve a relative ``server`` path against the task file's directory."""
        if self.server.startswith(("http://", "https://")):
            return self.server
        path = Path(self.server)
        if not path.is_absolute() and base_dir is not None:
            candidate = (base_dir / path).resolve()
            if candidate.exists():
                return str(candidate)
        return self.server


def _validation_message(exc: Exception) -> str:
    """Human-first message for a failed task validation. Unknown fields get
    named explicitly with a closest-match suggestion (a typo'd field used to
    be silently ignored — the error must say exactly what was wrong)."""
    if isinstance(exc, ValidationError):
        unknown = [str(err["loc"][-1]) for err in exc.errors() if err["type"] == "extra_forbidden"]
        if unknown:
            known = list(TaskSpec.model_fields)
            parts = []
            for field in unknown:
                matches = difflib.get_close_matches(field, known, n=3)
                hint = (
                    f" — did you mean one of: {', '.join(matches)}?"
                    if matches
                    else f" (valid fields: {', '.join(known)})"
                )
                parts.append(f"unknown field {field!r}{hint}")
            return "; ".join(parts)
    return str(exc)


def load_task_file(path: Path) -> list[TaskSpec]:
    """Load one YAML file containing a task mapping or a list of them."""
    data = yaml.safe_load(path.read_text())
    if data is None:
        raise ValueError(f"{path}: file is empty")
    raw_tasks = data if isinstance(data, list) else [data]
    tasks = []
    for i, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: entry {i} is not a mapping")
        try:
            task = TaskSpec.model_validate(raw)
        except Exception as exc:
            raise ValueError(f"{path}: entry {i} is invalid: {_validation_message(exc)}") from exc
        task = task.model_copy(update={"server": task.resolve_server(path.parent)})
        tasks.append(task)
    return tasks


def load_tasks(path: str | Path) -> list[TaskSpec]:
    """Load tasks from a YAML file or every ``*.yaml``/``*.yml`` in a directory.

    Raises ValueError on validation failures or duplicate task ids.
    """
    root = Path(path)
    if root.is_dir():
        files = sorted(p for p in root.iterdir() if p.suffix in (".yaml", ".yml"))
        if not files:
            raise ValueError(f"{root}: no .yaml/.yml task files found")
    elif root.is_file():
        files = [root]
    else:
        raise ValueError(f"{root}: no such file or directory")

    tasks: list[TaskSpec] = []
    seen: dict[str, Path] = {}
    for file in files:
        for task in load_task_file(file):
            if task.id in seen:
                raise ValueError(
                    f"duplicate task id {task.id!r} in {file} (first seen in {seen[task.id]})"
                )
            seen[task.id] = file
            tasks.append(task)
    return tasks
