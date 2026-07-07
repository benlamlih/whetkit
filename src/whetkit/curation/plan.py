"""Curation plans: a declarative, reversible description of tool-set changes.

A plan never touches the origin server. It only says how the overlay should
*present* the origin's tools: hide (prune), rename, or rewrite descriptions.
Merging duplicate tools is expressed as hiding the redundant copies and
renaming/redescribing the canonical one — call behavior is always delegated
1:1 to an existing origin tool, which is what keeps the overlay fully
reversible (delete the plan and nothing remains).

Plans serialize to YAML so they can be reviewed, versioned, and hand-edited.
"""

import re
from pathlib import Path

import mcp.types as types
import yaml
from pydantic import BaseModel, Field

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class ToolOverride(BaseModel):
    """How the overlay presents one origin tool."""

    original_name: str
    new_name: str | None = None
    new_description: str | None = None
    hidden: bool = False
    reason: str = ""

    @property
    def presented_name(self) -> str:
        return self.new_name or self.original_name


class CurationPlan(BaseModel):
    server: str = ""
    notes: str = ""
    overrides: list[ToolOverride] = Field(default_factory=list)

    def override_for(self, original_name: str) -> ToolOverride | None:
        return next((o for o in self.overrides if o.original_name == original_name), None)

    def validate_against(self, origin_tool_names: set[str]) -> list[str]:
        """Return every problem that makes the plan unsafe to apply."""
        problems: list[str] = []
        seen_originals: set[str] = set()
        for override in self.overrides:
            if override.original_name not in origin_tool_names:
                problems.append(f"override targets unknown tool {override.original_name!r}")
            if override.original_name in seen_originals:
                problems.append(f"duplicate override for {override.original_name!r}")
            seen_originals.add(override.original_name)
            if override.new_name is not None and not _NAME_RE.match(override.new_name):
                problems.append(f"invalid new name {override.new_name!r}")

        presented = [
            self.override_for(name).presented_name if self.override_for(name) else name
            for name in sorted(origin_tool_names)
            if not (self.override_for(name) and self.override_for(name).hidden)
        ]
        duplicates = {name for name in presented if presented.count(name) > 1}
        problems.extend(f"presented tool name collision: {name!r}" for name in sorted(duplicates))
        return problems

    def presented_to_original(self, origin_tool_names: set[str]) -> dict[str, str]:
        """Map every name the agent sees to the origin tool it delegates to."""
        mapping: dict[str, str] = {}
        for name in origin_tool_names:
            override = self.override_for(name)
            if override and override.hidden:
                continue
            mapping[override.presented_name if override else name] = name
        return mapping

    def rename_map(self) -> dict[str, str]:
        """Map renamed presented names back to their origin tools. Tasks
        declare ``expected_tools`` in origin names, so runs made through the
        overlay must be translated through this map before scoring."""
        return {
            override.new_name: override.original_name
            for override in self.overrides
            if override.new_name is not None and not override.hidden
        }

    def transform_tools(self, tools: list[types.Tool]) -> list[types.Tool]:
        """Present the origin's tool list through this plan."""
        presented: list[types.Tool] = []
        for tool in tools:
            override = self.override_for(tool.name)
            if override is None:
                presented.append(tool)
                continue
            if override.hidden:
                continue
            presented.append(
                tool.model_copy(
                    update={
                        "name": override.presented_name,
                        "description": (
                            override.new_description
                            if override.new_description is not None
                            else tool.description
                        ),
                    }
                )
            )
        return presented


def save_plan(plan: CurationPlan, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(plan.model_dump(exclude_defaults=True), sort_keys=False))


def load_plan(path: str | Path) -> CurationPlan:
    return CurationPlan.model_validate(yaml.safe_load(Path(path).read_text()))
