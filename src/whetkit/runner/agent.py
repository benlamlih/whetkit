"""The core eval loop.

Give the model the task prompt and the server's tools; execute every tool
call it makes against the real MCP server; feed results back; repeat until
the model answers without tool calls or the turn limit is hit. Tool failures
are returned to the model as error results, not raised — an agent that
recovers from a bad call can still succeed.
"""

import asyncio
import json
import time
from collections.abc import Callable

import mcp.types as types
from pydantic import BaseModel

from whetkit.datasets import TaskSpec
from whetkit.llm import ChatMessage, LLMProvider, ToolDef, ToolResult, get_provider, parse_model
from whetkit.mcp import MCPClient, ServerSpec
from whetkit.tracing import TaskRun, ToolCallRecord, TurnRecord
from whetkit.tracing.records import RunStatus, utc_now

DEFAULT_SYSTEM_PROMPT = (
    "You are a capable assistant with access to tools from an MCP server. "
    "Use the tools to complete the user's request. When you are done, reply "
    "with a clear final answer and no further tool calls."
)

DEFAULT_MODEL = "anthropic:claude-sonnet-5"


class RunConfig(BaseModel):
    model: str = DEFAULT_MODEL
    max_turns: int = 10
    max_tokens: int = 1024
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Wall-clock budget for one task's whole agent loop (provider turns plus
    # tool calls). A hung server or provider must fail one task, not the batch.
    task_timeout_s: float = 120.0


def _render_tool_result(result: types.CallToolResult) -> str:
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
        else:
            parts.append(f"[{block.type} content]")
    return "\n".join(parts) if parts else "(empty result)"


async def _execute_call(client: MCPClient, name: str, arguments: dict) -> tuple[str, bool]:
    """Run one tool call; never raises. Returns (result_text, is_error)."""
    try:
        result = await client.call_tool(name, arguments)
        return _render_tool_result(result), bool(result.isError)
    except Exception as exc:
        return f"Tool call failed: {exc}", True


async def run_task(
    task: TaskSpec,
    server: ServerSpec,
    config: RunConfig | None = None,
    provider: LLMProvider | None = None,
    client_factory: Callable[[ServerSpec], MCPClient] = MCPClient,
) -> TaskRun:
    """Run one task's agent loop against a live MCP server.

    ``client_factory`` lets callers interpose a transforming client (e.g. the
    curation overlay) between the agent and the origin server.
    """
    config = config or RunConfig()
    provider_name, model_id = parse_model(config.model)
    provider = provider or get_provider(provider_name)

    run = TaskRun(task_id=task.id, server=server.label(), model=config.model)
    messages: list[ChatMessage] = [ChatMessage(role="user", content=task.prompt)]

    try:
        async with client_factory(server) as client:
            # The timeout is handled INSIDE the client context so the client's
            # __aexit__ runs on an un-cancelled task and the transport (stdio
            # subprocess / HTTP session) still shuts down cleanly.
            try:
                async with asyncio.timeout(config.task_timeout_s):
                    mcp_tools = await client.list_tools()
                    tool_defs = [
                        ToolDef(
                            name=t.name,
                            description=t.description or "",
                            input_schema=t.inputSchema or {"type": "object"},
                        )
                        for t in mcp_tools
                    ]

                    for turn_index in range(config.max_turns):
                        started = time.perf_counter()
                        turn = await provider.complete(
                            model=model_id,
                            system=config.system_prompt,
                            messages=messages,
                            tools=tool_defs,
                            max_tokens=config.max_tokens,
                        )
                        turn_latency = (time.perf_counter() - started) * 1000

                        record = TurnRecord(
                            index=turn_index,
                            assistant_text=turn.text,
                            usage=turn.usage,
                            latency_ms=turn_latency,
                            stop_reason=turn.stop_reason,
                        )
                        run.turns.append(record)

                        if not turn.tool_calls:
                            run.final_text = turn.text
                            run.status = RunStatus.COMPLETED
                            break

                        messages.append(
                            ChatMessage(
                                role="assistant", content=turn.text, tool_calls=turn.tool_calls
                            )
                        )
                        results: list[ToolResult] = []
                        for call in turn.tool_calls:
                            arguments = call.arguments if isinstance(call.arguments, dict) else {}
                            call_started = time.perf_counter()
                            result_text, is_error = await _execute_call(
                                client, call.name, arguments
                            )
                            call_latency = (time.perf_counter() - call_started) * 1000
                            record.tool_calls.append(
                                ToolCallRecord(
                                    call_id=call.id,
                                    name=call.name,
                                    arguments=arguments,
                                    result_text=result_text,
                                    is_error=is_error,
                                    latency_ms=call_latency,
                                )
                            )
                            results.append(
                                ToolResult(call_id=call.id, content=result_text, is_error=is_error)
                            )
                        messages.append(ChatMessage(role="user", tool_results=results))
                    else:
                        run.status = RunStatus.MAX_TURNS
            except TimeoutError:
                run.status = RunStatus.TIMEOUT
                run.error = f"task exceeded --task-timeout after {config.task_timeout_s:g}s"
    except Exception as exc:
        run.status = RunStatus.ERROR
        run.error = f"{type(exc).__name__}: {exc}"

    run.finished_at = utc_now()
    return run


async def run_tasks(
    tasks: list[TaskSpec],
    servers: dict[str, ServerSpec],
    config: RunConfig | None = None,
    provider: LLMProvider | None = None,
) -> list[TaskRun]:
    """Run several tasks sequentially. ``servers`` maps task.server strings
    (as stored on each task) to resolved specs."""
    runs = []
    for task in tasks:
        runs.append(await run_task(task, servers[task.server], config, provider))
    return runs


def summarize_run(run: TaskRun) -> str:
    """One-line human summary, used by the CLI."""
    tools = " -> ".join(run.called_tool_names) or "(no tool calls)"
    usage = run.total_usage
    return (
        f"{run.task_id}: {run.status} in {len(run.turns)} turn(s), "
        f"tools: {tools}, tokens in/out: {usage.input_tokens}/{usage.output_tokens}"
    )


def dump_run_json(run: TaskRun) -> str:
    return json.dumps(run.model_dump(mode="json"), indent=2)
