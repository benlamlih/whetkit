import sys
from pathlib import Path

import pytest

from whetkit.datasets import TaskSpec
from whetkit.llm import LLMTurn, ToolCall, Usage, get_provider, parse_model
from whetkit.llm.anthropic_provider import _to_anthropic_messages
from whetkit.llm.base import ChatMessage, ToolResult
from whetkit.llm.openai_provider import _to_openai_messages
from whetkit.mcp import StdioSpec
from whetkit.runner import RunConfig, run_task
from whetkit.tracing.records import RunStatus

from .fakes import FakeProvider

MINI_SERVER = Path(__file__).parent / "fixtures" / "mini_server.py"


def make_task() -> TaskSpec:
    return TaskSpec(
        id="add-numbers",
        prompt="What is 2 + 3? Use the add tool.",
        server="unused",
        expected_tools=["add"],
        success_criteria="The answer is 5.",
    )


def mini_spec() -> StdioSpec:
    return StdioSpec(command=sys.executable, args=[str(MINI_SERVER)])


CONFIG = RunConfig(model="fake:fake-model", max_turns=4)


async def test_loop_executes_tools_and_feeds_back_results() -> None:
    provider = FakeProvider(
        [
            LLMTurn(
                text="Let me compute that.",
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})],
                usage=Usage(input_tokens=10, output_tokens=5),
                stop_reason="tool_use",
            ),
            LLMTurn(text="2 + 3 = 5.", usage=Usage(input_tokens=20, output_tokens=7)),
        ]
    )
    run = await run_task(make_task(), mini_spec(), CONFIG, provider)

    assert run.status == RunStatus.COMPLETED
    assert run.final_text == "2 + 3 = 5."
    assert run.called_tool_names == ["add"]
    assert run.turns[0].tool_calls[0].result_text == "5"
    assert run.turns[0].tool_calls[0].is_error is False
    assert run.total_usage.input_tokens == 30
    assert run.total_usage.output_tokens == 12

    # the provider saw the server's tools and got the tool result back
    first_call = provider.calls[0]
    assert {t.name for t in first_call["tools"]} == {"add", "greet"}
    second_messages = provider.calls[1]["messages"]
    assert second_messages[-1].tool_results[0].content == "5"
    assert second_messages[-1].tool_results[0].call_id == "c1"


async def test_bad_tool_call_is_fed_back_as_error_not_raised() -> None:
    provider = FakeProvider(
        [
            LLMTurn(tool_calls=[ToolCall(id="c1", name="no_such_tool", arguments={})]),
            LLMTurn(text="That tool does not exist."),
        ]
    )
    run = await run_task(make_task(), mini_spec(), CONFIG, provider)

    assert run.status == RunStatus.COMPLETED
    call = run.turns[0].tool_calls[0]
    assert call.is_error is True
    assert call.result_text
    fed_back = provider.calls[1]["messages"][-1].tool_results[0]
    assert fed_back.is_error is True


async def test_max_turns_cap() -> None:
    looping = LLMTurn(tool_calls=[ToolCall(id="c", name="add", arguments={"a": 1, "b": 1})])
    provider = FakeProvider([looping.model_copy(deep=True) for _ in range(4)])
    run = await run_task(make_task(), mini_spec(), CONFIG, provider)

    assert run.status == RunStatus.MAX_TURNS
    assert len(run.turns) == 4
    assert run.final_text is None


async def test_provider_failure_yields_error_status() -> None:
    run = await run_task(make_task(), mini_spec(), CONFIG, FakeProvider([]))
    assert run.status == RunStatus.ERROR
    assert "script exhausted" in (run.error or "")


def test_parse_model() -> None:
    assert parse_model("anthropic:claude-sonnet-5") == ("anthropic", "claude-sonnet-5")
    assert parse_model("openai:gpt-5.2") == ("openai", "gpt-5.2")
    assert parse_model("claude-sonnet-5") == ("anthropic", "claude-sonnet-5")
    with pytest.raises(ValueError):
        parse_model("anthropic:")
    with pytest.raises(ValueError):
        get_provider("mystery")


def _sample_conversation() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="hi"),
        ChatMessage(
            role="assistant",
            content="calling",
            tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})],
        ),
        ChatMessage(
            role="user",
            tool_results=[ToolResult(call_id="c1", content="3", is_error=False)],
        ),
    ]


def test_anthropic_message_translation() -> None:
    wire = _to_anthropic_messages(_sample_conversation())
    assert wire[0] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assistant_blocks = wire[1]["content"]
    assert assistant_blocks[0] == {"type": "text", "text": "calling"}
    assert assistant_blocks[1]["type"] == "tool_use"
    assert assistant_blocks[1]["input"] == {"a": 1, "b": 2}
    result_block = wire[2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "c1"


def test_openai_message_translation() -> None:
    wire = _to_openai_messages("be helpful", _sample_conversation())
    assert wire[0] == {"role": "system", "content": "be helpful"}
    assert wire[1] == {"role": "user", "content": "hi"}
    assert wire[2]["tool_calls"][0]["function"]["name"] == "add"
    assert wire[3] == {"role": "tool", "tool_call_id": "c1", "content": "3"}
