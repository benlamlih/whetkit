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
            if isinstance(spec, StdioSpec) and Path(spec.command).name == "whetkit"
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
        assert "--apply needs" in plain(result.output)

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
        slimmed_path, _removed = write_slim_output(config, plans, tmp_path / "out")
        document = json.loads(slimmed_path.read_text())["mcpServers"]
        assert document["untouched"] == {"command": "node", "args": ["x.js"]}
        assert document["old-sse"] == {"type": "sse", "url": "https://x/sse"}
        assert Path(document["planned"]["command"]).name == "whetkit"
        assert (tmp_path / "out" / "planned" / "server.json").is_file()


class TestUltratestRegressions:
    def test_tie_chain_reasons_point_at_a_visible_survivor(self) -> None:
        # a==b==c descriptions: pairwise ties chain; every hide reason must
        # name a copy that actually stays visible.
        inventories = {
            name: inventory(tool("search", "Search the files.")) for name in ("a", "b", "c")
        }
        plans = build_dedupe_plans(inventories, cross_server_duplicates(inventories))
        hidden = {s for s, p in plans.items() if p.overrides}
        (visible,) = set(inventories) - hidden
        for plan in plans.values():
            for override in plan.overrides:
                assert f"Duplicate of {visible}.search" in override.reason

    def test_out_dir_expands_tilde(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        config = parse_client_config(
            write_config(tmp_path, {"mcpServers": {"s": {"command": "python"}}})
        )
        from whetkit.curation import CurationPlan, ToolOverride

        plans = {"s": CurationPlan(overrides=[ToolOverride(original_name="t", hidden=True)])}
        slimmed, _removed = write_slim_output(config, plans, "~/tilde-out")
        assert slimmed == tmp_path / "tilde-out" / "mcp.slimmed.json"

    def test_wrapper_command_is_absolute_when_resolvable(self, tmp_path: Path, monkeypatch) -> None:
        fake = tmp_path / "bin" / "whetkit"
        fake.parent.mkdir()
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake.parent))
        config = parse_client_config(
            write_config(tmp_path, {"mcpServers": {"s": {"command": "python"}}})
        )
        from whetkit.curation import CurationPlan, ToolOverride

        plans = {"s": CurationPlan(overrides=[ToolOverride(original_name="t", hidden=True)])}
        slimmed, _removed = write_slim_output(config, plans, tmp_path / "out")
        entry = json.loads(slimmed.read_text())["mcpServers"]["s"]
        assert entry["command"] == str(fake)

    def test_rich_config_flagged_not_standalone(self, tmp_path: Path) -> None:
        rich = write_config(
            tmp_path,
            {
                "mcpServers": {"s": {"command": "python"}},
                "projects": {},
                "someSetting": True,
            },
        )
        assert parse_client_config(rich).standalone is False
        bare = tmp_path / "bare.json"
        bare.write_text(json.dumps({"mcpServers": {"s": {"command": "python"}}}))
        assert parse_client_config(bare).standalone is True

    def test_hide_of_uninspectable_server_warns_and_is_dropped(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "mcpServers": {
                    "mini": {
                        "command": sys.executable,
                        "args": [str(FIXTURES / "mini_server.py")],
                    },
                    "dead": {"command": "/nonexistent/binary"},
                }
            },
        )
        result = runner.invoke(
            app,
            [
                "slim",
                "--config",
                str(path),
                "--hide",
                "dead",
                "--apply",
                "--out",
                str(tmp_path / "o"),
            ],
        )
        assert result.exit_code == 0, result.output
        norm = plain(result.output)
        assert "cannot act on 'dead'" in norm
        assert "nothing to apply" in norm
        assert "no --hide servers" not in norm  # the old lying message


class TestDedupeV2:
    def test_cluster_consolidates_on_single_winner(self) -> None:
        # a<b<c description informativeness: one cluster, c wins, a+b hidden
        inventories = {
            "a": inventory(tool("search", "Search the files.")),
            "b": inventory(tool("search", "Search the files quickly now.")),
            "c": inventory(tool("search", "Search the files quickly with ranked results.")),
        }
        plans = build_dedupe_plans(inventories, cross_server_duplicates(inventories))
        assert set(plans) == {"a", "b"}
        for plan in plans.values():
            (override,) = plan.overrides
            assert "Duplicate of c.search" in override.reason

    def test_tie_cluster_keeps_earliest_config_server(self) -> None:
        inventories = {
            name: inventory(tool("search", "Search the files."))
            for name in ("first", "second", "third")
        }
        plans = build_dedupe_plans(inventories, cross_server_duplicates(inventories))
        assert set(plans) == {"second", "third"}  # 'first' wins the tie
        for plan in plans.values():
            assert "Duplicate of first.search" in plan.overrides[0].reason

    def test_fully_hidden_server_dropped_from_config(self, tmp_path: Path) -> None:
        path = write_config(
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
        out = tmp_path / "o"
        result = runner.invoke(
            app,
            ["slim", "--config", str(path), "--hide", "mini-b", "--apply", "--out", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert "removed mini-b from the slimmed config" in plain(result.output)
        document = json.loads((out / "mcp.slimmed.json").read_text())["mcpServers"]
        assert "mini-b" not in document
        assert "mini" in document  # untouched server copied verbatim


class TestToolSearchAwareness:
    def test_defer_loading_lint_and_always_load_parsing(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "mcpServers": {
                    "hot-srv": {"command": "python", "alwaysLoad": True},
                    "trap-srv": {"command": "python", "defer_loading": True},
                    "plain": {"command": "python"},
                }
            },
        )
        config = parse_client_config(path)
        assert config.always_load == ["hot-srv"]
        assert config.defer_loading_entries == ["trap-srv"]

    def test_audit_reports_hot_set_and_lints_defer(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "mcpServers": {
                    "mini": {
                        "command": sys.executable,
                        "args": [str(FIXTURES / "mini_server.py")],
                        "alwaysLoad": True,
                        "defer_loading": True,
                    },
                    "mini-b": {
                        "command": sys.executable,
                        "args": [str(FIXTURES / "mini_server_b.py")],
                    },
                }
            },
        )
        result = runner.invoke(app, ["slim", "--config", str(path)])
        assert result.exit_code == 0, result.output
        norm = plain(result.output)
        assert "silently ignores" in norm and "#26844" in norm
        assert "Tool search hot set (alwaysLoad): mini" in norm

    def test_recommend_hot_from_traces(self, tmp_path: Path) -> None:
        from whetkit.tracing import TaskRun, ToolCallRecord, TraceStore, TurnRecord

        store_path = tmp_path / "traces.sqlite3"
        run = TaskRun(
            task_id="t",
            server="unmatched-label",
            model="m",
            turns=[
                TurnRecord(
                    index=0,
                    tool_calls=[
                        # 'add' exists only on mini -> tool-name attribution
                        ToolCallRecord(call_id="c", name="add", result_text="ok")
                    ],
                )
            ],
        )
        with TraceStore(store_path) as store:
            store.save_runs([run], run_group="baseline")

        path = write_config(
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
        out = tmp_path / "o"
        result = runner.invoke(
            app,
            [
                "slim",
                "--config",
                str(path),
                "--recommend-hot",
                "--from-traces",
                str(store_path),
                "--apply",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        norm = plain(result.output)
        assert "Recommended alwaysLoad set (from traces): mini" in norm
        assert "Defer (no observed usage): mini-b" in norm
        document = json.loads((out / "hot.mcp.json").read_text())["mcpServers"]
        assert document["mini"]["alwaysLoad"] is True
        assert "alwaysLoad" not in document["mini-b"]

    def test_recommend_hot_without_traces_points_at_them(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            {
                "mcpServers": {
                    "mini": {
                        "command": sys.executable,
                        "args": [str(FIXTURES / "mini_server.py")],
                    }
                }
            },
        )
        result = runner.invoke(app, ["slim", "--config", str(path), "--recommend-hot"])
        assert result.exit_code == 0, result.output
        assert "--from-traces" in plain(result.output)

    def test_hot_config_drops_ignored_defer_field(self, tmp_path: Path) -> None:
        from whetkit.slim import write_hot_config

        config = parse_client_config(
            write_config(
                tmp_path,
                {
                    "mcpServers": {
                        "a": {"command": "python", "defer_loading": True},
                        "b": {"command": "python", "alwaysLoad": True},
                    }
                },
            )
        )
        path = write_hot_config(config, {"a"}, tmp_path / "o")
        document = json.loads(path.read_text())["mcpServers"]
        assert document["a"] == {"command": "python", "alwaysLoad": True}
        assert document["b"] == {"command": "python"}  # demoted, defer field gone
