# Contributing to whetkit

Thanks for helping make agent↔MCP tool selection measurable. Issues and pull
requests are welcome — small, focused changes land fastest.

## Dev setup

whetkit uses [uv](https://docs.astral.sh/uv/) for everything. Python is
pinned via `.python-version` (3.13); uv fetches it automatically.

```sh
git clone https://github.com/benlamlih/whetkit && cd whetkit
uv sync
```

That's it — no API key is needed for development (the test suite runs on a
scripted fake provider).

## Tests and lint

```sh
uv run pytest        # full suite, offline, no key needed
uv run ruff check .  # lint  (add --fix to auto-fix)
uv run ruff format . # format
```

Optionally install the pre-commit hooks so lint/format run on every commit:

```sh
uv run pre-commit install
```

Every change should come with tests. CLI-output tests must assert against
rich-normalized text — see the `plain()` helper in `tests/test_cli.py` (CI
renders errors in ANSI panels that split words).

Dependency versions are pinned exactly and documented (with sources and check
dates) in [VERSIONS.md](VERSIONS.md); update it when you touch a pin.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scoring): add per-slot recall
fix(cli): clean error for missing --tasks path
docs: clarify task-format alternatives
chore(release): v0.7.0
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`. Scope is
the module or command when it helps (`cli`, `curate`, `datasets`, ...).

## Pull request flow

1. Fork (or branch) off `main` and make your change.
2. Run `uv run ruff check . && uv run ruff format . && uv run pytest` — all
   green before pushing.
3. Open a PR against `main` with a short description of what and why.
4. `main` is protected: CI must pass and the PR is merged by a maintainer —
   no direct pushes.

## Reporting bugs and requesting features

Please use the issue templates (bug report / feature request). For security
problems, do **not** open a public issue — see [SECURITY.md](SECURITY.md).
