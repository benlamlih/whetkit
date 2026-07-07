import json
from pathlib import Path

from whetkit.datasets import load_tasks
from whetkit.generate import GeneratorConfig, generate_tasks, write_tasks_yaml
from whetkit.llm import LLMTurn
from whetkit.mcp.introspect import ServerInventory, ToolInfo

from .fakes import FakeProvider

CONFIG = GeneratorConfig(model="fake:gen")


def inventory() -> ServerInventory:
    return ServerInventory(
        server="test",
        tools=[
            ToolInfo(name="search_products", description="Search the catalog by name."),
            ToolInfo(name="check_stock", description="Return stock count for a product id."),
        ],
    )


def draft(**overrides) -> dict:
    base = {
        "id": "find-mouse",
        "prompt": "Find products matching 'mouse' and report the cheapest.",
        "expected_tools": ["search_products"],
        "ordered": False,
        "success_criteria": "The answer names the cheapest matching product.",
    }
    return {**base, **overrides}


def provider_returning(*drafts: dict) -> FakeProvider:
    return FakeProvider([LLMTurn(text=json.dumps(list(drafts)))])


class TestGenerateTasks:
    async def test_valid_drafts_become_tasks(self) -> None:
        provider = provider_returning(
            draft(),
            draft(id="stock-check", expected_tools=[["check_stock", "search_products"]]),
        )
        tasks, warnings = await generate_tasks(
            inventory(), "srv", count=2, config=CONFIG, provider=provider
        )
        assert [t.id for t in tasks] == ["find-mouse", "stock-check"]
        assert all(t.server == "srv" for t in tasks)
        assert warnings == []
        # the generator saw the tool inventory
        assert "search_products" in provider.calls[0]["messages"][0].content

    async def test_unknown_tools_filtered_and_task_dropped_when_empty(self) -> None:
        provider = provider_returning(
            draft(expected_tools=[["search_products", "ghost_tool"]]),
            draft(id="all-ghost", expected_tools=["nonexistent"]),
        )
        tasks, warnings = await generate_tasks(inventory(), "srv", config=CONFIG, provider=provider)
        assert [t.id for t in tasks] == ["find-mouse"]
        assert tasks[0].expected_tool_slots == [["search_products"]]
        assert any("ghost_tool" in w for w in warnings)
        assert any("all-ghost" in w and "no valid tool" in w for w in warnings)

    async def test_duplicate_ids_dropped(self) -> None:
        provider = provider_returning(draft(), draft())
        tasks, warnings = await generate_tasks(inventory(), "srv", config=CONFIG, provider=provider)
        assert len(tasks) == 1
        assert any("duplicate id" in w for w in warnings)

    async def test_invalid_draft_dropped_with_reason(self) -> None:
        provider = provider_returning(draft(id="BAD ID!"))
        tasks, warnings = await generate_tasks(inventory(), "srv", config=CONFIG, provider=provider)
        assert tasks == []
        assert any("invalid" in w for w in warnings)

    async def test_unparseable_output_fails_soft(self) -> None:
        provider = FakeProvider([LLMTurn(text="sure! here are tasks"), LLMTurn(text="[not json")])
        tasks, warnings = await generate_tasks(inventory(), "srv", config=CONFIG, provider=provider)
        assert tasks == []
        assert "not a valid JSON array" in warnings[0]


class TestWriteTasksYaml:
    async def test_written_file_round_trips_through_load_tasks(self, tmp_path: Path) -> None:
        provider = provider_returning(draft(), draft(id="second", expected_tools=["check_stock"]))
        tasks, _ = await generate_tasks(
            inventory(), str(tmp_path / "server"), config=CONFIG, provider=provider
        )
        out = tmp_path / "generated.yaml"
        write_tasks_yaml(tasks, str(out))

        content = out.read_text()
        assert content.startswith("# Drafted by 'whetkit generate'")
        loaded = load_tasks(out)
        assert [t.id for t in loaded] == ["find-mouse", "second"]


class TestPromptShaping:
    async def test_server_context_reaches_the_prompt(self) -> None:
        provider = provider_returning(draft())
        await generate_tasks(
            inventory(),
            "srv",
            config=CONFIG,
            provider=provider,
            server_context="stdio: uvx mcp-server-git --repository /real/checkout",
        )
        prompt = provider.calls[0]["messages"][0].content
        assert "/real/checkout" in prompt
        assert "Never invent placeholder" in prompt

    async def test_read_only_by_default_writes_on_opt_in(self) -> None:
        provider = provider_returning(draft())
        await generate_tasks(inventory(), "srv", config=CONFIG, provider=provider)
        assert "ONLY read-only" in provider.calls[0]["system"]

        provider2 = provider_returning(draft())
        await generate_tasks(
            inventory(), "srv", config=CONFIG, provider=provider2, allow_writes=True
        )
        assert "Write tasks are allowed" in provider2.calls[0]["system"]
