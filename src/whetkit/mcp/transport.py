"""Transport layer: how to reach an MCP server.

All construction of SDK transports is isolated here so the planned move to
the MCP Python SDK v2 (see MIGRATION.md) stays contained in this module.

Three connection modes are supported, because real-world servers are mixed:

- ``stdio``: a local server subprocess speaking JSON-RPC over stdin/stdout.
- ``http`` with ``mode="stateful"``: legacy (2025 spec) streamable HTTP —
  one long-lived session per connection, identified by a server-issued
  session id.
- ``http`` with ``mode="stateless"``: the 2026-07-28 stateless spec —
  no session affinity; the client opens a fresh exchange per operation.
  On the v1 SDK this is modeled by re-connecting per operation (see
  ``MCPClient``), which matches the per-POST semantics from the client side.
"""

import json
import os
import re
import shutil
import sys
from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field


class HttpMode(StrEnum):
    STATEFUL = "stateful"
    STATELESS = "stateless"


class StdioSpec(BaseModel):
    kind: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = []
    env: dict[str, str] | None = None
    cwd: str | None = None

    def label(self) -> str:
        return f"stdio: {self.command} {' '.join(self.args)}".strip()


class HttpSpec(BaseModel):
    kind: Literal["http"] = "http"
    url: str
    headers: dict[str, str] | None = None
    mode: HttpMode = HttpMode.STATEFUL

    def label(self) -> str:
        return f"http ({self.mode}): {self.url}"


ServerSpec = Annotated[StdioSpec | HttpSpec, Field(discriminator="kind")]


class _SpecAdapter(BaseModel):
    spec: ServerSpec


def spec_from_dict(data: dict) -> ServerSpec:
    return _SpecAdapter(spec=data).spec


def _python_command(command: str) -> str:
    """Map a bare ``python`` to the running interpreter so stdio servers get
    the project virtualenv."""
    if command in ("python", "python3"):
        return sys.executable
    return shutil.which(command) or command


def resolve_server_spec(value: str, http_mode: HttpMode = HttpMode.STATEFUL) -> ServerSpec:
    """Turn a CLI-friendly string into a ServerSpec.

    Accepted forms:
    - ``http(s)://...``            -> streamable-HTTP server
    - directory                    -> ``server.json`` inside it, else ``server.py`` via stdio
    - ``*.json`` file              -> full spec document
    - ``*.py`` file                -> stdio python server
    """
    if value.startswith(("http://", "https://")):
        return HttpSpec(url=value, mode=http_mode)

    path = Path(value)
    if path.is_dir():
        config = path / "server.json"
        if config.is_file():
            return _spec_from_json(config)
        script = path / "server.py"
        if script.is_file():
            return StdioSpec(command=sys.executable, args=[str(script)])
        raise ValueError(f"{value}: directory has neither server.json nor server.py")
    if path.suffix == ".json" and path.is_file():
        return _spec_from_json(path)
    if path.suffix == ".py" and path.is_file():
        return StdioSpec(command=sys.executable, args=[str(path)])
    raise ValueError(f"{value}: not a URL, directory, .json spec, or .py server")


_ENV_REF_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value):
    """Substitute ``${VAR}`` references from the environment in every string
    of a spec document — so credentials (HTTP headers, child env) never have
    to be written into server.json itself."""
    if isinstance(value, str):

        def sub(match: re.Match) -> str:
            name = match.group(1)
            if (resolved := os.environ.get(name)) is None:
                raise ValueError(
                    f"server spec references ${{{name}}} but it is not set in the environment"
                )
            return resolved

        return _ENV_REF_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _spec_from_json(path: Path) -> ServerSpec:
    data = _expand_env(json.loads(path.read_text()))
    spec = spec_from_dict(data)
    if isinstance(spec, StdioSpec):
        spec.command = _python_command(spec.command)
        # Relative script paths are relative to the spec file, not the CWD.
        spec.args = [
            str((path.parent / a).resolve()) if (path.parent / a).is_file() else a
            for a in spec.args
        ]
        if spec.cwd is None:
            spec.cwd = str(path.parent)
    return spec


@asynccontextmanager
async def open_session(spec: ServerSpec) -> AsyncIterator[ClientSession]:
    """Open one initialized MCP session over the spec's transport.

    Stdio servers' stderr (SDK request logs, npm install chatter) is
    discarded so it can't garble whetkit's output; set ``WHETKIT_SERVER_LOGS=1``
    to pass it through when debugging a server that won't start."""
    if isinstance(spec, StdioSpec):
        params = StdioServerParameters(
            command=spec.command, args=spec.args, env=spec.env, cwd=spec.cwd
        )
        with ExitStack() as stack:
            errlog = sys.stderr
            if not os.environ.get("WHETKIT_SERVER_LOGS"):
                errlog = stack.enter_context(open(os.devnull, "w"))
            async with (
                stdio_client(params, errlog=errlog) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield session
    else:
        http_client = (
            httpx.AsyncClient(headers=spec.headers, follow_redirects=True, timeout=30)
            if spec.headers
            else None
        )
        try:
            async with (
                streamable_http_client(spec.url, http_client=http_client) as (read, write, _sid),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield session
        finally:
            if http_client is not None:
                await http_client.aclose()
