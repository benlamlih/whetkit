# VERSIONS

Every runtime, tool, and library used by this project, with the exact pinned
version, the date it was verified, and the authoritative source consulted.
Re-verify at the start of every working session and bump when a newer stable
release exists.

## Runtime & tooling

| Tool | Version | Checked | Source | Notes |
|---|---|---|---|---|
| Python | 3.13 (3.13.12 locally) | 2026-07-07 | `uv python list` / python.org | 3.14.x is current stable upstream, but the environment's uv (0.8.17) only offers 3.14.0rc2 (pre-release). 3.13 is the conservative, broadly supported choice; every pinned dependency supports it. Recorded in `.python-version`. |
| uv | 0.8.17 | 2026-07-07 | `uv --version` (preinstalled) | Package/env manager. Build backend pinned to matching `uv_build>=0.8.17,<0.9.0`. |

## Runtime dependencies (pinned in `pyproject.toml`)

| Package | Version | Checked | Source | Notes |
|---|---|---|---|---|
| mcp | 1.28.1 | 2026-07-07 | https://pypi.org/pypi/mcp/json | **Decision rule applied:** PyPI shows v2 only as pre-releases (`2.0.0a1`–`2.0.0b1`, no stable `2.x`), so we build on the stable v1 line. Transport lives behind a thin interface; see `MIGRATION.md`. |
| anthropic | 0.116.0 | 2026-07-07 | https://pypi.org/pypi/anthropic/json | Latest stable. Key from `ANTHROPIC_API_KEY`. |
| openai | 2.44.0 | 2026-07-07 | https://pypi.org/pypi/openai/json | Latest stable. Key from `OPENAI_API_KEY`. |
| pydantic | 2.13.4 | 2026-07-07 | https://pypi.org/pypi/pydantic/json | Task schema validation. |
| pyyaml | 6.0.3 | 2026-07-07 | https://pypi.org/pypi/pyyaml/json | Task file parsing. |
| typer | 0.26.8 | 2026-07-07 | https://pypi.org/pypi/typer/json | CLI framework. |

## Dev dependencies

| Package | Version | Checked | Source | Notes |
|---|---|---|---|---|
| pytest | 9.1.1 | 2026-07-07 | https://pypi.org/pypi/pytest/json | |
| pytest-asyncio | 1.4.0 | 2026-07-07 | https://pypi.org/pypi/pytest-asyncio/json | `asyncio_mode = "auto"`. |
| ruff | 0.15.20 | 2026-07-07 | https://pypi.org/pypi/ruff/json | Lint + format (also via pre-commit). |
| pre-commit | 4.6.0 | 2026-07-07 | https://pypi.org/pypi/pre-commit/json | |

## Storage

| Choice | Version | Checked | Source | Notes |
|---|---|---|---|---|
| SQLite (stdlib `sqlite3`) | ships with CPython 3.13 | 2026-07-07 | python.org stdlib docs | Local trace storage. No Postgres in Stage 1 by design. |

## CI

| Item | Version | Checked | Source | Notes |
|---|---|---|---|---|
| actions/checkout | v5 | 2026-07-07 | training-data knowledge (v5.0.0, Aug 2025) | ⚠️ Live verification of GitHub release tags is blocked by this environment's network proxy (403 on github.com/api.github.com outside repo scope). Re-verify from an unrestricted environment. |
| actions/upload-artifact / download-artifact | v4 | 2026-07-07 | training-data knowledge | Same proxy caveat as above. |
| pypa/gh-action-pypi-publish | release/v1 | 2026-07-07 | PyPA's documented pin (tracking branch) | Officially recommended way to consume this action; handles PyPI Trusted Publishing (OIDC). |
| gh CLI (release job) | preinstalled on ubuntu-latest | 2026-07-07 | GitHub-hosted runner docs | Used only to create the GitHub Release. |
| uv in CI | latest via https://astral.sh/uv/install.sh | 2026-07-07 | Astral install docs | Install script used instead of the `setup-uv` action to avoid pinning an unverifiable action tag. |
| ruff-pre-commit | v0.15.20 | 2026-07-07 | mirrors ruff releases on PyPI | Tag mirrors the pinned ruff version. |
