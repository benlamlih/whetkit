"""MCP connectivity: transports, client, and tool introspection."""

from whetkit.mcp.client import MCPClient
from whetkit.mcp.introspect import ServerInventory, ToolInfo, inspect_server
from whetkit.mcp.transport import HttpMode, HttpSpec, ServerSpec, StdioSpec, resolve_server_spec

__all__ = [
    "HttpMode",
    "HttpSpec",
    "MCPClient",
    "ServerInventory",
    "ServerSpec",
    "StdioSpec",
    "ToolInfo",
    "inspect_server",
    "resolve_server_spec",
]
