"""Anthropic provider (Messages API, anthropic SDK).

Reads the API key from ``ANTHROPIC_API_KEY``.
"""

import json
from typing import Any

from anthropic import AsyncAnthropic

from whetkit.llm.base import ChatMessage, LLMProvider, LLMTurn, ToolCall, ToolDef, Usage


def _to_anthropic_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        blocks: list[dict[str, Any]] = []
        if msg.role == "assistant":
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
        else:
            for result in msg.tool_results:
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                )
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
        out.append({"role": msg.role, "content": blocks})
    return out


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, client: AsyncAnthropic | None = None):
        self._client = client or AsyncAnthropic()

    async def complete(
        self,
        *,
        model: str,
        system: str | None,
        messages: list[ChatMessage],
        tools: list[ToolDef],
        max_tokens: int = 1024,
    ) -> LLMTurn:
        kwargs: dict[str, Any] = {}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=_to_anthropic_messages(messages),
            **kwargs,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                arguments = block.input
                if isinstance(arguments, str):  # defensive: some models emit JSON strings
                    arguments = json.loads(arguments or "{}")
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))

        return LLMTurn(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason,
        )
