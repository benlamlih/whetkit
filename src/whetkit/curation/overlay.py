"""The overlay: presents a curated tool set while delegating every call to
the untouched origin server.

Two ways to use it:

- :class:`CuratedMCPClient` — in-process overlay used by the eval runner for
  before/after comparisons.
- :func:`serve_overlay` — a real stdio MCP server (``whetkit overlay``) so
  any MCP client (Claude Code, an IDE, another agent) can talk to the
  curated view. The origin server is never modified; stop the proxy and
  nothing remains.
"""

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from whetkit.curation.plan import CurationPlan
from whetkit.mcp import MCPClient, ServerSpec


class UnknownCuratedTool(Exception):
    pass


class InvalidPlanError(Exception):
    """The plan fails validation against the origin's live tool list."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class CuratedMCPClient(MCPClient):
    """MCPClient that shows plan-transformed tools and un-maps names on call."""

    def __init__(self, spec: ServerSpec, plan: CurationPlan):
        super().__init__(spec)
        self.plan = plan
        self._name_map: dict[str, str] | None = None

    async def _mapping(self) -> dict[str, str]:
        if self._name_map is None:
            origin_tools = await super().list_tools()
            self._name_map = self.plan.presented_to_original({t.name for t in origin_tools})
        return self._name_map

    async def origin_tool_names(self) -> set[str]:
        """The origin server's tool names, untransformed by the plan."""
        return {t.name for t in await super().list_tools()}

    async def list_tools(self) -> list[types.Tool]:
        tools = await super().list_tools()
        self._name_map = self.plan.presented_to_original({t.name for t in tools})
        return self.plan.transform_tools(tools)

    async def call_tool(self, name: str, arguments: dict) -> types.CallToolResult:
        mapping = await self._mapping()
        if name not in mapping:
            raise UnknownCuratedTool(f"tool {name!r} is not part of the curated tool set")
        return await super().call_tool(mapping[name], arguments)


def build_overlay_server(client: CuratedMCPClient, name: str = "whetkit-overlay") -> Server:
    server = Server(name)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return await client.list_tools()

    @server.call_tool()
    async def _call_tool(tool_name: str, arguments: dict) -> types.CallToolResult:
        # Pass the origin's result through verbatim (content, structured
        # content, and error flag) — the overlay transforms metadata only.
        return await client.call_tool(tool_name, arguments)

    return server


async def serve_overlay(origin: ServerSpec, plan: CurationPlan) -> None:
    """Run the overlay as a stdio MCP server until the client disconnects.

    The plan is validated against the origin's live tool list first; serving
    a plan with unknown targets or name collisions would silently present a
    broken tool surface, so it raises :class:`InvalidPlanError` instead.
    """
    async with CuratedMCPClient(origin, plan) as client:
        if problems := plan.validate_against(await client.origin_tool_names()):
            raise InvalidPlanError(problems)
        server = build_overlay_server(client)
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
