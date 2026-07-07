# Curation and the overlay proxy

`whetkit curate` measures how well an agent picks your server's tools, asks
an LLM to fix the tool set's *metadata*, and measures again — all without
touching your server.

## How it works

1. **Baseline eval** — every task runs against the origin server and is
   scored (deterministic tool-match + optional LLM judge).
2. **Proposal** — the optimizer model sees the tool inventory and the eval
   traces (what was called, what was expected but never called, judge
   failures) and proposes a `CurationPlan`.
3. **Overlay eval** — the same tasks run again, but the agent sees the tools
   *through* the plan. Calls are un-mapped to original names and delegated
   to the untouched origin server.
4. **Comparison** — before/after hit-rate, per task.

## The plan is declarative and reversible

The plan is YAML (default `.whetkit/curation-plan.yaml`) — review it, edit
it, commit it. Each entry transforms how one origin tool is *presented*:

```yaml
server: 'stdio: python server.py'
notes: Renamed cryptic tools, hid duplicates and noise.
overrides:
  - original_name: data_query_1
    new_name: search_products
    new_description: Search the product catalog by name keywords.
    reason: Name and description said nothing about products.
  - original_name: legacy_search
    hidden: true
    reason: Duplicate of search_products.
```

Supported actions and their guarantees:

| Action | Plan form | Guarantee |
|---|---|---|
| Rename | `new_name` | Argument schema and behavior are untouched. |
| Rewrite description | `new_description` | Metadata only. |
| Prune | `hidden: true` | The tool still exists on the origin; it is just not shown. |
| Merge duplicates | `hidden` on the copies + rename/rewrite on the canonical one | Every presented tool delegates 1:1 to one origin tool. |

Because every presented tool maps 1:1 to an origin tool and only metadata
changes, deleting the plan (or stopping the overlay) restores the original
world exactly. The origin server is **never** modified. Higher-level
composite/workflow tools are out of Stage 1's mechanical scope — the
optimizer may suggest them in `notes`, but the overlay will not synthesize
new behavior.

Every proposal is validated against the live tool list (unknown tools, name
collisions, invalid names); unsafe entries are dropped with a warning, so a
bad LLM proposal can degrade the overlay but never break delegation.

## Using the overlay outside evals

Serve the curated view as a real stdio MCP server for any MCP client:

```sh
uv run whetkit overlay --server examples/sample-server --plan .whetkit/curation-plan.yaml
```

Point Claude Code (or any MCP-capable agent) at that command as a stdio
server and it sees the curated tool set.
