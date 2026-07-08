import sys
from pathlib import Path

import pytest

from whetkit.curation import (
    CuratedMCPClient,
    CurationPlan,
    ToolOverride,
    load_plan,
    propose_plan,
    save_plan,
)
from whetkit.curation.optimizer import OptimizerConfig
from whetkit.curation.overlay import UnknownCuratedTool
from whetkit.datasets import TaskSpec
from whetkit.llm import LLMTurn
from whetkit.mcp import StdioSpec, inspect_server, resolve_server_spec
from whetkit.scoring import score_tool_match
from whetkit.scoring.aggregate import TaskScore
from whetkit.tracing import TaskRun
from whetkit.tracing.records import RunStatus

from .fakes import FakeProvider

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_SERVER = REPO_ROOT / "examples" / "sample-server"


def sample_plan() -> CurationPlan:
    return CurationPlan(
        server="sample",
        notes="tidy the shop tools",
        overrides=[
            ToolOverride(
                original_name="data_query_1",
                new_name="search_products",
                new_description="Search the product catalog by name keywords.",
                reason="cryptic name",
            ),
            ToolOverride(original_name="legacy_search", hidden=True, reason="duplicate"),
            ToolOverride(original_name="sys_ping", hidden=True, reason="noise"),
            ToolOverride(
                original_name="do_thing",
                new_name="send_order_notification",
                new_description="Send an order confirmation notification to an email address.",
                reason="opaque name",
            ),
        ],
    )


class TestPlan:
    def test_transform_and_mapping(self) -> None:
        plan = sample_plan()
        names = {"data_query_1", "legacy_search", "sys_ping", "do_thing", "inv_check"}
        assert plan.validate_against(names) == []
        mapping = plan.presented_to_original(names)
        assert mapping == {
            "search_products": "data_query_1",
            "send_order_notification": "do_thing",
            "inv_check": "inv_check",
        }

    def test_validation_catches_unknown_and_collisions(self) -> None:
        plan = CurationPlan(
            overrides=[
                ToolOverride(original_name="ghost", new_name="x"),
                ToolOverride(original_name="a", new_name="b"),
                ToolOverride(original_name="a", new_name="bad name!"),
            ]
        )
        problems = plan.validate_against({"a", "b"})
        assert any("unknown tool 'ghost'" in p for p in problems)
        assert any("duplicate override" in p for p in problems)
        assert any("invalid new name" in p for p in problems)
        assert any("collision: 'b'" in p for p in problems)

    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        plan = sample_plan()
        save_plan(plan, tmp_path / "plan.yaml")
        assert load_plan(tmp_path / "plan.yaml") == plan

    def test_rename_map_covers_renames_only(self) -> None:
        plan = sample_plan()
        plan.overrides.append(
            ToolOverride(original_name="inv_check", new_description="Check stock.")
        )
        assert plan.rename_map() == {
            "search_products": "data_query_1",
            "send_order_notification": "do_thing",
        }


class TestCuratedClient:
    async def test_overlay_presents_and_delegates(self) -> None:
        spec = resolve_server_spec(str(SAMPLE_SERVER))
        async with CuratedMCPClient(spec, sample_plan()) as client:
            tools = {t.name: t for t in await client.list_tools()}

            assert "search_products" in tools
            assert "legacy_search" not in tools
            assert "sys_ping" not in tools
            assert "data_query_1" not in tools
            assert tools["search_products"].description.startswith("Search the product catalog")
            # schema is untouched by the overlay
            assert "q" in tools["search_products"].inputSchema["properties"]

            result = await client.call_tool("search_products", {"q": "wireless mouse"})
            assert "AeroGlide" in result.content[0].text

            with pytest.raises(UnknownCuratedTool):
                await client.call_tool("data_query_1", {"q": "x"})  # hidden original name

    async def test_overlay_stdio_proxy_end_to_end(self, tmp_path: Path) -> None:
        """The overlay must also work as a real MCP server over stdio."""
        from whetkit.mcp import MCPClient

        plan_file = tmp_path / "plan.yaml"
        save_plan(sample_plan(), plan_file)
        proxy_spec = StdioSpec(
            command=sys.executable,
            args=[
                "-m",
                "whetkit.cli",
                "overlay",
                "--server",
                str(SAMPLE_SERVER),
                "--plan",
                str(plan_file),
            ],
        )
        async with MCPClient(proxy_spec) as client:
            names = {t.name for t in await client.list_tools()}
            assert "search_products" in names
            assert "legacy_search" not in names

            result = await client.call_tool("search_products", {"q": "keyboard"})
            assert "KeyForge" in result.content[0].text
            assert not result.isError

            bad = await client.call_tool("send_order_notification", {})
            assert bad.isError


def _score(task: TaskSpec, run: TaskRun) -> TaskScore:
    return TaskScore(
        task_id=task.id,
        run_status=RunStatus.COMPLETED,
        tool_match=score_tool_match(task, run.called_tool_names),
    )


class TestOptimizer:
    async def test_propose_plan_parses_and_validates(self) -> None:
        inventory = await inspect_server(resolve_server_spec(str(SAMPLE_SERVER)))
        task = TaskSpec(
            id="find-product",
            prompt="Find wireless mice",
            server="s",
            expected_tools=["data_query_1"],
            success_criteria="names the mouse",
        )
        run = TaskRun(task_id="find-product", server="s", model="m")
        proposal = {
            "notes": "clean up",
            "overrides": [
                {
                    "original_name": "data_query_1",
                    "action": "rename",
                    "new_name": "search_products",
                    "new_description": "Search products by name.",
                    "reason": "cryptic",
                },
                {"original_name": "legacy_search", "action": "prune", "reason": "dup"},
                {"original_name": "not_a_tool", "action": "rename", "new_name": "oops"},
                {"original_name": "sys_ping", "action": "keep"},
            ],
        }
        import json

        provider = FakeProvider([LLMTurn(text=json.dumps(proposal))])
        plan, warnings = await propose_plan(
            inventory,
            [task],
            [run],
            [_score(task, run)],
            OptimizerConfig(model="fake:opt"),
            provider,
        )

        assert plan.notes == "clean up"
        by_original = {o.original_name: o for o in plan.overrides}
        assert by_original["data_query_1"].new_name == "search_products"
        assert by_original["legacy_search"].hidden is True
        assert "not_a_tool" not in by_original  # invalid entry dropped
        assert "sys_ping" not in by_original  # no-op keep dropped
        assert any("unknown tool" in w for w in warnings)

        # the optimizer saw the inventory, the failure evidence, and the cost
        prompt = provider.calls[0]["messages"][0].content
        assert "data_query_1" in prompt
        assert "Find wireless mice" in prompt
        assert "tokens in/out" in prompt

    async def test_unparseable_proposal_keeps_origin_tool_set(self) -> None:
        inventory = await inspect_server(resolve_server_spec(str(SAMPLE_SERVER)))
        provider = FakeProvider([LLMTurn(text="no json"), LLMTurn(text="still no json")])
        plan, warnings = await propose_plan(
            inventory, [], [], [], OptimizerConfig(model="fake:opt"), provider
        )
        assert plan.overrides == []
        assert any("keeping origin tool set" in w for w in warnings)


class TestPruneUnused:
    async def test_hides_only_untouched_tools(self) -> None:
        from whetkit.curation.optimizer import prune_unused
        from whetkit.mcp.introspect import ServerInventory, ToolInfo

        inventory = ServerInventory(
            server="s",
            tools=[ToolInfo(name=n) for n in ("used_called", "used_expected", "kept", "noise")],
        )
        task = TaskSpec(
            id="t",
            prompt="p",
            server="s",
            expected_tools=[["used_expected"]],
            success_criteria="c",
        )
        run = TaskRun(task_id="t", server="s", model="m")
        run.turns = []  # called_tool_names -> []
        plan = CurationPlan(
            overrides=[ToolOverride(original_name="kept", new_description="Clear.")]
        )
        # simulate a call to used_called via a run with one tool call
        from whetkit.tracing import ToolCallRecord, TurnRecord

        run.turns = [
            TurnRecord(
                index=0,
                tool_calls=[ToolCallRecord(call_id="c", name="used_called", result_text="ok")],
            )
        ]
        added = prune_unused(plan, inventory, [task], [run])
        assert added == 1
        hidden = {o.original_name for o in plan.overrides if o.hidden}
        assert hidden == {"noise"}


class TestRevision:
    async def test_revision_prompt_carries_plan_and_regressions(self) -> None:
        from whetkit.curation.optimizer import build_revision_prompt
        from whetkit.mcp.introspect import ServerInventory, ToolInfo
        from whetkit.tracing import ToolCallRecord, TurnRecord

        inventory = ServerInventory(server="s", tools=[ToolInfo(name="proc_ord")])
        task = TaskSpec(
            id="place-order",
            prompt="p",
            server="s",
            expected_tools=["proc_ord"],
            success_criteria="c",
        )
        good_run = TaskRun(
            task_id="place-order",
            server="s",
            model="m",
            turns=[
                TurnRecord(
                    index=0,
                    tool_calls=[ToolCallRecord(call_id="c", name="proc_ord", result_text="ok")],
                )
            ],
        )
        bad_run = TaskRun(task_id="place-order", server="s", model="m")
        baseline_score = _score(task, good_run)
        curated_score = _score(task, bad_run)
        plan = CurationPlan(
            overrides=[ToolOverride(original_name="proc_ord", new_name="process_order")]
        )
        prompt = build_revision_prompt(
            plan, inventory, [task], [good_run], [baseline_score], [bad_run], [curated_score]
        )
        assert "process_order" in prompt  # previous plan visible
        assert "REGRESSIONS your plan caused" in prompt
        assert "place-order" in prompt

    async def test_unparseable_revision_keeps_previous_plan(self) -> None:
        from whetkit.curation.optimizer import propose_revision
        from whetkit.mcp.introspect import ServerInventory, ToolInfo

        inventory = ServerInventory(server="s", tools=[ToolInfo(name="a")])
        previous = CurationPlan(overrides=[ToolOverride(original_name="a", hidden=True)])
        provider = FakeProvider([LLMTurn(text="nope"), LLMTurn(text="still nope")])
        revised, warnings = await propose_revision(
            previous,
            inventory,
            [],
            [],
            [],
            [],
            [],
            OptimizerConfig(model="fake:opt"),
            provider,
        )
        assert revised is previous
        assert any("keeping previous plan" in w for w in warnings)
