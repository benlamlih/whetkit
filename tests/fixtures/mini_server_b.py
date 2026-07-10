"""Second minimal MCP server: overlaps mini_server on purpose.

``sum_two`` duplicates mini_server's ``add`` (near-identical description) so
the suite can exercise whetkit slim's cross-server duplicate detection.
``shout`` is unique to this server.
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("mini-b")


@server.tool()
def sum_two(x: int, y: int) -> int:
    """Add two integers and return their sum."""
    return x + y


@server.tool()
def shout(text: str) -> str:
    """Uppercase the given text and append an exclamation mark."""
    return text.upper() + "!"


if __name__ == "__main__":
    server.run()
