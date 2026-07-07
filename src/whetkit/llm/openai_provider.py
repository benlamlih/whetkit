"""OpenAI provider (Chat Completions API, openai SDK).

Chat Completions (rather than the Responses API) keeps the tool-use loop
symmetric with other providers; both live behind :class:`LLMProvider`, so
swapping APIs is contained here. Reads the API key from ``OPENAI_API_KEY``.
"""

import json
from typing import Any

from openai import AsyncOpenAI

from whetkit.llm.base import ChatMessage, LLMProvider, LLMTurn, ToolCall, ToolDef, Usage


def _to_openai_messages(system: str | None, messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in msg.tool_calls
                ]
            out.append(entry)
        else:
            for result in msg.tool_results:
                out.append(
                    {"role": "tool", "tool_call_id": result.call_id, "content": result.content}
                )
            if msg.content:
                out.append({"role": "user", "content": msg.content})
    return out


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, client: AsyncOpenAI | None = None):
        self._client = client or AsyncOpenAI()

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
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        response = await self._client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=_to_openai_messages(system, messages),
            **kwargs,
        )

        choice = response.choices[0]
        tool_calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
            )
            for call in (choice.message.tool_calls or [])
            if call.type == "function"
        ]
        usage = response.usage
        return LLMTurn(
            text=choice.message.content or None,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            stop_reason=choice.finish_reason,
        )
