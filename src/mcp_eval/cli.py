"""Command-line entry point for mcp-eval."""

import typer

app = typer.Typer(
    name="mcp-eval",
    help="Evaluate and improve LLM agent tool selection on MCP servers.",
    no_args_is_help=True,
)


if __name__ == "__main__":
    app()
