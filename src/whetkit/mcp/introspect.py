"""Tool introspection: inventory a server's tools and summarize their cost."""

import json
from typing import Any

from pydantic import BaseModel

from whetkit.mcp.client import MCPClient
from whetkit.mcp.transport import ServerSpec


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token for English prose). Good enough
    for comparing tool-set sizes; not a billing meter."""
    return max(1, round(len(text) / 4)) if text else 0


def schema_complexity(schema: dict[str, Any] | None) -> int:
    """Score a JSON schema: one point per property/branch, plus nesting depth.

    A flat 2-arg tool scores ~3; deeply nested or union-heavy schemas score
    much higher — which correlates with how hard the schema is for a model
    to fill correctly.
    """
    if not schema:
        return 0

    def walk(node: Any, depth: int) -> int:
        if not isinstance(node, dict):
            return 0
        score = 0
        props = node.get("properties")
        if isinstance(props, dict):
            for sub in props.values():
                score += 1 + walk(sub, depth + 1)
        for branch_key in ("anyOf", "oneOf", "allOf"):
            branches = node.get(branch_key)
            if isinstance(branches, list):
                for sub in branches:
                    score += 1 + walk(sub, depth + 1)
        items = node.get("items")
        if isinstance(items, dict):
            score += walk(items, depth + 1)
        return score + (1 if depth == 0 else 0)

    return walk(schema, 0)


class ToolInfo(BaseModel):
    name: str
    title: str | None = None
    description: str = ""
    input_schema: dict[str, Any] = {}

    @property
    def description_tokens(self) -> int:
        return estimate_tokens(self.description)

    @property
    def definition_tokens(self) -> int:
        """What this tool costs in context on EVERY request: the client sends
        name + description + full input schema with each message."""
        schema = json.dumps(self.input_schema, sort_keys=True) if self.input_schema else ""
        return estimate_tokens(f"{self.name} {self.description} {schema}")

    @property
    def complexity(self) -> int:
        return schema_complexity(self.input_schema)

    @property
    def param_count(self) -> int:
        props = self.input_schema.get("properties")
        return len(props) if isinstance(props, dict) else 0


class ServerInventory(BaseModel):
    server: str
    tools: list[ToolInfo]

    @property
    def tool_count(self) -> int:
        return len(self.tools)

    @property
    def total_description_tokens(self) -> int:
        return sum(t.description_tokens for t in self.tools)

    @property
    def total_definition_tokens(self) -> int:
        return sum(t.definition_tokens for t in self.tools)

    @property
    def total_complexity(self) -> int:
        return sum(t.complexity for t in self.tools)

    def summary_lines(self) -> list[str]:
        avg = self.total_complexity / self.tool_count if self.tools else 0.0
        return [
            f"Server: {self.server}",
            f"Tools: {self.tool_count}",
            f"Total description tokens (est.): {self.total_description_tokens}",
            f"Schema complexity: total {self.total_complexity}, avg {avg:.1f}",
        ]


async def inspect_server(spec: ServerSpec) -> ServerInventory:
    async with MCPClient(spec) as client:
        tools = await client.list_tools()
    infos = [
        ToolInfo(
            name=t.name,
            title=t.title,
            description=t.description or "",
            input_schema=t.inputSchema or {},
        )
        for t in tools
    ]
    return ServerInventory(server=spec.label(), tools=infos)
