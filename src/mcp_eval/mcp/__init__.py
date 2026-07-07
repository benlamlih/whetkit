"""MCP connectivity: transports, client, and tool introspection."""

from mcp_eval.mcp.client import MCPClient
from mcp_eval.mcp.introspect import ServerInventory, ToolInfo, inspect_server
from mcp_eval.mcp.transport import HttpMode, HttpSpec, ServerSpec, StdioSpec, resolve_server_spec

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
