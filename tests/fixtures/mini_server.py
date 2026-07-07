"""Minimal MCP server used by the test suite.

Run modes:
    python mini_server.py                      # stdio
    python mini_server.py --http --port 8123   # streamable HTTP (stateful)
    python mini_server.py --http --port 8123 --stateless
"""

import sys

from mcp.server.fastmcp import FastMCP


def build_server(stateless: bool = False, port: int = 8000) -> FastMCP:
    server = FastMCP("mini", host="127.0.0.1", port=port, stateless_http=stateless)

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        return a + b

    @server.tool()
    def greet(name: str) -> str:
        """Greet a person by name."""
        return f"Hello, {name}!"

    return server


if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
        server = build_server(stateless="--stateless" in sys.argv, port=port)
        server.run(transport="streamable-http")
    else:
        build_server().run()
