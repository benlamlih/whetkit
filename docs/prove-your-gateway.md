# Prove your gateway

Tool search, persona toolsets (hypertool), glob filters (mcp-filter),
`alwaysLoad` hot sets, hand-written gateways — every one of them reduces the
tool surface your agent sees. None of them tells you whether your workflows
still succeed afterwards. whetkit does; that's the whole point of it.

**Slim with anything — prove it with whetkit.**

## The recipe

1. **Write (or generate) the tasks that represent your real workflows** —
   see [task-format.md](task-format.md), or draft them:

   ```sh
   whetkit generate --server your-server.json --out tasks/workflows.yaml
   ```

2. **Baseline: run them against the raw server**, repeated so noise can't
   masquerade as signal:

   ```sh
   whetkit run --server your-server.json --tasks tasks/workflows.yaml \
     --runs 3 --summary-json before.json
   ```

3. **Point a `server.json` at your reduced view** — whatever produced it.
   The gateway is just another stdio command:

   ```json
   { "kind": "stdio", "command": "npx",
     "args": ["mcp-filter", "--include", "read_*", "--", "npx", "your-server"] }
   ```

   (For a whetkit plan, skip the wrapper entirely: `whetkit run --plan
   your-plan.yaml` scores the curated view directly.)

4. **Same tasks, through the reduced view:**

   ```sh
   whetkit run --server gateway.json --tasks tasks/workflows.yaml \
     --runs 3 --summary-json after.json
   ```

5. **The verdict, in one table:**

   ```sh
   whetkit diff before.json after.json
   ```

   Hit-rate, precision, unnecessary calls, tokens — with per-task
   PASS/MISS/FLAKY transitions. If the reduced surface broke a workflow,
   this is where it shows, before your users find it.

## Reading the result

- A hit-rate drop on specific tasks names exactly which workflow the
  reduced surface starved — check which tool those tasks needed.
- `FLAKY` transitions with overlapping ranges are noise, not signal; the
  summary's ⚠ caveats say so explicitly. Add `--runs` before believing
  either direction.
- Token deltas here measure the *conversation*, not the tool-definition
  surface — pair with `whetkit slim` for the per-message definition bill.
