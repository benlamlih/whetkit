import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mcp_eval.mcp import (
    HttpMode,
    HttpSpec,
    MCPClient,
    StdioSpec,
    inspect_server,
    resolve_server_spec,
)
from mcp_eval.mcp.introspect import estimate_tokens, schema_complexity

FIXTURES = Path(__file__).parent / "fixtures"
MINI_SERVER = FIXTURES / "mini_server.py"


def stdio_spec() -> StdioSpec:
    return StdioSpec(command=sys.executable, args=[str(MINI_SERVER)])


async def test_stdio_list_and_call() -> None:
    async with MCPClient(stdio_spec()) as client:
        tools = await client.list_tools()
        assert sorted(t.name for t in tools) == ["add", "greet"]

        result = await client.call_tool("add", {"a": 2, "b": 3})
        assert not result.isError
        assert result.content[0].text == "5"


async def test_inspect_server_summary() -> None:
    inventory = await inspect_server(stdio_spec())
    assert inventory.tool_count == 2
    assert inventory.total_description_tokens > 0
    add = next(t for t in inventory.tools if t.name == "add")
    assert add.param_count == 2
    assert add.complexity >= 2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"server on port {port} never came up")


@pytest.fixture(params=["stateful", "stateless"])
def http_server(request: pytest.FixtureRequest):
    port = _free_port()
    args = [sys.executable, str(MINI_SERVER), "--http", "--port", str(port)]
    if request.param == "stateless":
        args.append("--stateless")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_port(port)
        yield HttpSpec(url=f"http://127.0.0.1:{port}/mcp", mode=HttpMode(request.param))
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def test_http_list_and_call(http_server: HttpSpec) -> None:
    async with MCPClient(http_server) as client:
        tools = await client.list_tools()
        assert sorted(t.name for t in tools) == ["add", "greet"]
        result = await client.call_tool("greet", {"name": "MCP"})
        assert not result.isError
        assert "Hello, MCP!" in result.content[0].text


def test_resolve_server_spec_forms(tmp_path: Path) -> None:
    url_spec = resolve_server_spec("http://localhost:1234/mcp", http_mode=HttpMode.STATELESS)
    assert isinstance(url_spec, HttpSpec)
    assert url_spec.mode == HttpMode.STATELESS

    py_spec = resolve_server_spec(str(MINI_SERVER))
    assert isinstance(py_spec, StdioSpec)
    assert py_spec.command == sys.executable

    server_dir = tmp_path / "srv"
    server_dir.mkdir()
    (server_dir / "server.py").write_text("# stub\n")
    dir_spec = resolve_server_spec(str(server_dir))
    assert isinstance(dir_spec, StdioSpec)

    (server_dir / "server.json").write_text(
        '{"kind": "stdio", "command": "python", "args": ["server.py"]}'
    )
    json_spec = resolve_server_spec(str(server_dir))
    assert isinstance(json_spec, StdioSpec)
    assert json_spec.command == sys.executable
    assert json_spec.args[0] == str((server_dir / "server.py").resolve())

    with pytest.raises(ValueError):
        resolve_server_spec(str(tmp_path / "missing"))


def test_estimate_tokens_and_complexity() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 10) == 10

    assert schema_complexity(None) == 0
    flat = {"type": "object", "properties": {"a": {"type": "int"}, "b": {"type": "int"}}}
    nested = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"field": {"type": "string"}, "op": {"type": "string"}},
            },
            "options": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }
    assert schema_complexity(nested) > schema_complexity(flat) > 0
