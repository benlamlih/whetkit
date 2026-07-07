"""High-level MCP client used by the inspector, runner, and overlay proxy."""

from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Self

import mcp.types as types
from mcp import ClientSession

from whetkit.mcp.transport import HttpMode, HttpSpec, ServerSpec, open_session


class MCPClient:
    """Async client over any ServerSpec.

    For stdio and stateful HTTP the client holds one session for its whole
    lifetime. For stateless HTTP (2026-07-28 spec semantics) every operation
    runs on a fresh exchange, so no session state is assumed server-side.
    """

    def __init__(self, spec: ServerSpec):
        self.spec = spec
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def _stateless(self) -> bool:
        return isinstance(self.spec, HttpSpec) and self.spec.mode == HttpMode.STATELESS

    async def __aenter__(self) -> Self:
        if not self._stateless:
            self._stack = AsyncExitStack()
            self._session = await self._stack.enter_async_context(open_session(self.spec))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def list_tools(self) -> list[types.Tool]:
        if self._session is not None:
            result = await self._session.list_tools()
            return list(result.tools)
        async with open_session(self.spec) as session:
            result = await session.list_tools()
            return list(result.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        if self._session is not None:
            return await self._session.call_tool(name, arguments)
        async with open_session(self.spec) as session:
            return await session.call_tool(name, arguments)
