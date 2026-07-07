# mcp-eval

**Measure — then improve — how well LLM agents pick and use the tools your MCP
server exposes.**

`mcp-eval` is a local-first CLI that runs an agent against your
[MCP](https://modelcontextprotocol.io) server on a set of eval tasks, scores
tool-selection hit-rate, and then *curates* the tool set (rename / prune /
merge / rewrite descriptions) via a reversible overlay proxy — showing a
measurable before/after improvement.

> Status: Stage 1, under active development. Quickstart lands with the CLI phase.

## Planned commands

```sh
uv run mcp-eval inspect --server examples/sample-server   # tool inventory
uv run mcp-eval run     --server examples/sample-server --tasks examples/tasks
uv run mcp-eval curate  --server examples/sample-server --tasks examples/tasks
uv run mcp-eval report  # before/after hit-rate, HTML + JSON
```

## Development

Requires [uv](https://docs.astral.sh/uv/). Python version is pinned in
`.python-version`; exact dependency versions are pinned in `pyproject.toml`
and documented in [`VERSIONS.md`](VERSIONS.md).

```sh
uv sync
uv run pytest
uv run ruff check .
```

API keys are read from environment variables (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`) — never hardcoded, never committed.

## License

Apache-2.0 — see [LICENSE](LICENSE).
