"""Command-line entry point for mcp-eval."""

import asyncio
from typing import Annotated

import typer

from mcp_eval.mcp import HttpMode, inspect_server, resolve_server_spec

app = typer.Typer(
    name="mcp-eval",
    help="Evaluate and improve LLM agent tool selection on MCP servers.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Evaluate and improve LLM agent tool selection on MCP servers."""


@app.command()
def inspect(
    server: Annotated[
        str,
        typer.Option("--server", help="MCP server: URL, directory, server.json, or server.py"),
    ],
    http_mode: Annotated[
        HttpMode,
        typer.Option(
            "--http-mode",
            help="HTTP session mode: 'stateful' (legacy 2025) or 'stateless' (2026-07-28 spec)",
        ),
    ] = HttpMode.STATEFUL,
) -> None:
    """Connect to an MCP server and print its tool inventory."""
    spec = resolve_server_spec(server, http_mode=http_mode)
    inventory = asyncio.run(inspect_server(spec))

    for line in inventory.summary_lines():
        typer.echo(line)
    typer.echo()

    name_w = max((len(t.name) for t in inventory.tools), default=4)
    header = f"{'NAME':<{name_w}}  {'PARAMS':>6}  {'CPLX':>4}  {'TOKENS':>6}  DESCRIPTION"
    typer.echo(header)
    typer.echo("-" * len(header))
    for tool in inventory.tools:
        desc = " ".join(tool.description.split())
        if len(desc) > 70:
            desc = desc[:67] + "..."
        typer.echo(
            f"{tool.name:<{name_w}}  {tool.param_count:>6}  {tool.complexity:>4}  "
            f"{tool.description_tokens:>6}  {desc}"
        )


if __name__ == "__main__":
    app()
