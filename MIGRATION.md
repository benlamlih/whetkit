# MIGRATION

## MCP Python SDK: v1 → v2

**Status (checked 2026-07-07):** PyPI's `mcp` package has no stable 2.x release —
only pre-releases (`2.0.0a1`, `2.0.0a2`, `2.0.0a3`, `2.0.0b1`). Per the project's
dependency policy, core functionality is **not** built on a beta that warns of
breaking changes between pre-releases, so we pin the latest stable v1 line:
`mcp==1.28.1`.

**Action item:** when `mcp` publishes a stable `2.x` (expected around the
2026-07-28 stateless spec release), migrate:

1. Bump the pin in `pyproject.toml` and re-verify on PyPI; update `VERSIONS.md`.
2. `FastMCP` is renamed `MCPServer` in v2 — update the sample server in
   `examples/` and the curation overlay proxy.
3. Re-check the streamable-HTTP client APIs against the 2026-07-28 stateless
   spec. All transport construction is isolated in
   `src/whetkit/mcp/transport.py` behind the `ServerSpec` → session-factory
   interface, so the swap should not leak outside that module.
4. Keep support for all three connection modes (stdio, legacy stateful
   streamable-HTTP, stateless streamable-HTTP) — real-world servers will stay
   mixed for months.
