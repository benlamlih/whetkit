# Examples

## `sample-server/`

A small, deterministic e-commerce MCP server (products, customers, orders)
that is **intentionally badly curated**: cryptic names (`proc_ord`,
`do_thing`, `inv_check`), vague descriptions ("Query data."), duplicate tools
(`get_rec` vs `fetch_record`, `data_query_1` vs `legacy_search`), one
absurdly verbose name, and noise tools (`sys_ping`, `util_helper`,
`admin_reset`). It exists so `mcp-eval curate` has something measurable to
fix. All state is in-process; nothing on your machine is touched.

Inspect it:

```sh
uv run mcp-eval inspect --server examples/sample-server
```

## `tasks/`

Five eval tasks against the sample server, covering search, lookups, ordered
write-then-verify flows, and a worst-case-named notification tool. The task
format is documented in [docs/task-format.md](../docs/task-format.md).
