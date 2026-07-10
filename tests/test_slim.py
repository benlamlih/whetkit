import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_cli import plain
from whetkit.cli import app
from whetkit.mcp.introspect import ServerInventory, ToolInfo
from whetkit.mcp.transport import HttpSpec, StdioSpec
from whetkit.slim import (
    build_dedupe_plans,
    cross_server_duplicates,
    parse_client_config,
    write_slim_output,
)

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def write_config(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(document))
    return path


class TestParseClientConfig:
    def test_stdio_http_and_env_passthrough(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "mcpServers": {
                    "files": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                        "env": {"TOKEN": "${MY_TOKEN}"},
                    },
                    "remote": {
                        "type": "http",
                        "url": "https://api.example.com/mcp/",
                        "headers": {"Authorization": "Bearer ${PAT}"},
                    },
                }
            },
        )
        config = parse_client_config(path)
        files = config.servers["files"]
        assert isinstance(files, StdioSpec)
        assert files.env == {"TOKEN": "${MY_TOKEN}"}  # verbatim, expanded only at spawn
        remote = config.servers["remote"]
        assert isinstance(remote, HttpSpec)
        assert remote.headers == {"Authorization": "Bearer ${PAT}"}
        assert config.skipped == []

    def test_claude_code_project_blocks_merge(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "mcpServers": {"global-srv": {"command": "a"}},
                "projects": {
                    "/home/me/proj": {"mcpServers": {"proj-srv": {"command": "b"}}},
                    "/home/me/other": "not-a-dict-is-tolerated",
                },
            },
        )
        config = parse_client_config(path)
        assert set(config.servers) == {"global-srv", "proj-srv"}

    def test_sse_is_skipped_with_reason(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "mcpServers": {
                    "old": {"type": "sse", "url": "https://x/sse"},
                    "ok": {"command": "python"},
                }
            },
        )
        config = parse_client_config(path)
        assert set(config.servers) == {"ok"}
        (skipped,) = config.skipped
        assert skipped.name == "old" and "sse" in skipped.reason

    def test_missing_file_names_known_locations(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="claude_desktop_config.json"):
            parse_client_config(tmp_path / "nope.json")

    def test_no_mcp_servers_key(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"something": 1})
        with pytest.raises(ValueError, match="no mcpServers entries"):
            parse_client_config(path)


def tool(name: str, description: str) -> ToolInfo:
    return ToolInfo(name=name, description=description)


def inventory(*tools: ToolInfo) -> ServerInventory:
    return ServerInventory(server="s", tools=list(tools))


class TestCrossServerDuplicates:
    def test_flags_lookalikes_across_servers_only(self) -> None:
        inventories = {
            "files": inventory(
                tool("search_files", "Recursively search for files matching a pattern."),
                tool("read_file", "Read the complete contents of one file."),
            ),
            "repo": inventory(
                tool("search_files", "Recursively search for files matching a pattern."),
            ),
        }
        (duplicate,) = cross_server_duplicates(inventories)
        assert {duplicate.keep_server, duplicate.hide_server} == {"files", "repo"}
        assert duplicate.keep_tool == duplicate.hide_tool == "search_files"

    def test_crud_family_exempt_and_unrelated_ignored(self) -> None:
        inventories = {
            "a": inventory(tool("create_note", "Create a new note in the store.")),
            "b": inventory(
                tool("delete_note", "Delete a note from the store."),
                tool("weather", "Get the current weather for a city."),
            ),
        }
        assert cross_server_duplicates(inventories) == []

    def test_winner_has_more_informative_description(self) -> None:
        inventories = {
            "terse": inventory(tool("fetch_page", "Fetch page.")),
            "verbose": inventory(
                tool(
                    "fetch_page",
                    "Fetch a web page over HTTP and return its main text content.",
                )
            ),
        }
        (duplicate,) = cross_server_duplicates(inventories)
        assert duplicate.keep_server == "verbose"
        assert duplicate.hide_server == "terse"


class TestBuildDedupePlans:
    def test_plans_hide_losers_and_whole_hidden_servers(self) -> None:
        inventories = {
            "a": inventory(tool("x", "Do the x thing to the record.")),
            "b": inventory(
                tool("x", "Do the x thing to the record, with details."),
                tool("y", "Unrelated tool that stays."),
            ),
            "noisy": inventory(tool("z1", "Z one."), tool("z2", "Z two.")),
        }
        duplicates = cross_server_duplicates({"a": inventories["a"], "b": inventories["b"]})
        plans = build_dedupe_plans(inventories, duplicates, hide_servers={"noisy"})
        assert set(plans) == {"a", "noisy"}  # b won the duplicate, keeps everything
        assert [o.original_name for o in plans["a"].overrides] == ["x"]
        assert {o.original_name for o in plans["noisy"].overrides} == {"z1", "z2"}
        assert all(o.hidden for p in plans.values() for o in p.overrides)

    def test_keep_servers_are_never_touched(self) -> None:
        inventories = {
            "a": inventory(tool("x", "Do the x thing.")),
            "b": inventory(tool("x", "Do the x thing, with details.")),
        }
        duplicates = cross_server_duplicates(inventories)
        plans = build_dedupe_plans(inventories, duplicates, keep_servers={"a", "b"})
        assert plans == {}


class TestSlimCli:
    def _two_server_config(self, tmp_path: Path) -> Path:
        return write_config(
            tmp_path,
            {
                "mcpServers": {
                    "mini": {
                        "command": sys.executable,
                        "args": [str(FIXTURES / "mini_server.py")],
                    },
                    "mini-b": {
                        "command": sys.executable,
                        "args": [str(FIXTURES / "mini_server_b.py")],
                    },
                }
            },
        )

    def test_audit_needs_no_api_key(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = runner.invoke(app, ["slim", "--config", str(self._two_server_config(tmp_path))])
        assert result.exit_code == 0, result.output
        assert "EVERY message" in result.output
        assert "$" in result.output  # the screenshot number
        # add (mini) vs sum_two (mini-b) share a near-identical description
        assert "Cross-server duplicates" in result.output

    def test_apply_writes_round_trippable_output(self, tmp_path: Path) -> None:
        from whetkit.curation import load_plan

        out = tmp_path / "slim-out"
        result = runner.invoke(
            app,
            [
                "slim",
                "--config",
                str(self._two_server_config(tmp_path)),
                "--dedupe",
                "--apply",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        slimmed = out / "mcp.slimmed.json"
        assert slimmed.is_file()

        # the slimmed config parses with the same parser (round trip)
        reparsed = parse_client_config(slimmed)
        assert set(reparsed.servers) == {"mini", "mini-b"}
        rewritten = [
            name
            for name, spec in reparsed.servers.items()
            if isinstance(spec, StdioSpec) and spec.command == "whetkit"
        ]
        assert len(rewritten) == 1  # exactly the duplicate loser was rewritten
        loser = rewritten[0]
        plan = load_plan(out / loser / "plan.yaml")
        assert len(plan.overrides) == 1 and plan.overrides[0].hidden
        assert "not modified" in result.output

    def test_apply_without_work_is_refused(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["slim", "--config", str(self._two_server_config(tmp_path)), "--apply"],
        )
        assert result.exit_code != 0
        assert "--dedupe and/or --hide" in plain(result.output)

    def test_missing_config_is_friendly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["slim", "--config", str(tmp_path / "ghost.json")])
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "known client locations" in plain(result.output)


class TestWriteSlimOutput:
    def test_untouched_and_skipped_entries_copy_verbatim(self, tmp_path: Path) -> None:
        config_path = write_config(
            tmp_path,
            {
                "mcpServers": {
                    "planned": {"command": "python", "args": ["s.py"]},
                    "untouched": {"command": "node", "args": ["x.js"]},
                    "old-sse": {"type": "sse", "url": "https://x/sse"},
                }
            },
        )
        config = parse_client_config(config_path)
        from whetkit.curation import CurationPlan, ToolOverride

        plans = {
            "planned": CurationPlan(
                overrides=[ToolOverride(original_name="t", hidden=True, reason="r")]
            )
        }
        slimmed_path = write_slim_output(config, plans, tmp_path / "out")
        document = json.loads(slimmed_path.read_text())["mcpServers"]
        assert document["untouched"] == {"command": "node", "args": ["x.js"]}
        assert document["old-sse"] == {"type": "sse", "url": "https://x/sse"}
        assert document["planned"]["command"] == "whetkit"
        assert (tmp_path / "out" / "planned" / "server.json").is_file()
