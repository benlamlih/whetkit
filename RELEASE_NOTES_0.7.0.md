# whetkit 0.7.0

Everything on main since the 0.6.0 PyPI release, in one installable version.
0.6.0 worked; 0.7.0 is the release that makes its numbers **trustworthy** —
failures now refuse loudly instead of scoring quietly.

## Trustworthiness

- **Preflight provider checks.** `run` / `curate` / `fix` / `generate` verify
  every API key they will need (agent, judge when enabled, optimizer,
  generator) *before* any eval spend, and refuse with a clean error naming
  the exact env var (e.g. `ANTHROPIC_API_KEY`). No more "0% hit-rate, exit 0"
  from a missing key.
- **Honest exit codes.** Any task run that ERRORs or times out makes
  `run`/`curate`/`fix` exit **3** after printing the full summary, with an
  explicit "infrastructure failure, not a tool-selection result" line — and
  each failed run's real error (previously buried in `traces.sqlite3`) is
  printed with the results. CI can no longer stay green while measuring
  nothing.
- **Regression guardrail.** `curate` and `fix` drop any optimizer override
  that hides a tool the task set's `expected_tools` require ("kept `<tool>`:
  required by task `<id>`"); `run --plan` warns about the same without
  touching your hand-edited plan. And when the curated view still scores
  below baseline, `curate` prints a loud REGRESSION block with next steps
  and exits **4** instead of handing you a worse server without comment.
- **Strict task schema.** Unknown task fields are rejected with the file, the
  field, and a closest-match suggestion (`unknown field 'orderd' — did you
  mean one of: ordered?`). A typo can no longer silently drop a constraint.

## Hardening (landed on main after 0.6.0)

- `--runs N` on `run`/`curate`/`fix`: repeat the eval and report mean plus
  range and flaky tasks — single runs are noise.
- Per-task `--task-timeout`: a hung server or provider fails one task, not
  the batch, and is flagged as TIMEOUT.
- Prompt-injection delimiting of all server-controlled text in the judge,
  optimizer, and generator prompts; judge/optimizer fail closed on provider
  errors.
- Curation plans are validated against the origin's live tool list before
  `run --plan` spends anything and at `overlay` startup.
- `curate`/`fix` refuse multi-server task sets instead of silently curating
  one server against runs it never touched.
- Strict flag validation (`--judge`, `--match-mode`, `--fail-on`, …) with
  clean messages; `tools/list` `nextCursor` pagination; `max_tokens`
  truncation flagging.

## Bug fixes & UX

- `whetkit report` rebuilds no longer lose the TOOLS EXPOSED metric: the
  server is best-effort re-inspected and the plan mapping recomputes the
  after-count (unreachable servers just leave it blank, as before).
- Clean errors (no tracebacks) for the common fat-finger inputs: missing or
  invalid `--tasks` files, malformed task YAML, unreadable plans, and `diff`
  on files that aren't `--summary-json` outputs (the message names which
  file is wrong).
- `curate` and `fix` print the same tokens / latency / estimated-$ line as
  `run`.

## Telemetry (opt-in, off by default)

New `whetkit telemetry on|off|status`. Nothing is sent unless you opt in
(`whetkit telemetry on` or `WHETKIT_TELEMETRY=1`). When enabled, each command
sends exactly: command name, whetkit version, Python major.minor,
`sys.platform`, and a random anonymous id — never arguments, paths, server
names, prompts, or results. Fire-and-forget with a 2 s timeout; it can never
block or break a command. See the README "Telemetry" section.

whetkit Cloud (hosted history, team dashboards, CI gating) is being explored —
join the waitlist:
https://github.com/benlamlih/whetkit/issues/new?template=cloud-waitlist.yml

## Compatibility notes

- Exit codes are new: `3` = errored/timed-out runs, `4` = curated view
  regressed below baseline. Scripts that only checked `0`/non-zero are
  unaffected; CI that must tolerate infra flakes should branch on `3`.
- Task files with unknown/typo'd fields that previously loaded silently now
  fail validation — fix the field name (the error tells you the closest
  match).
- `httpx` is now a direct dependency (same version the lock already pinned).
