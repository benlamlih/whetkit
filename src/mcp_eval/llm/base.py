"""Provider-neutral chat/tool-use types and the provider interface.

The runner speaks only these types; each provider module translates to and
from its SDK's wire format. Adding a provider means implementing
:class:`LLMProvider` and registering it in ``registry.py``.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolDef(BaseModel):
    """A tool offered to the model."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = {}


class ToolResult(BaseModel):
    """The outcome of executing a ToolCall, fed back to the model."""

    call_id: str
    content: str
    is_error: bool = False


class ChatMessage(BaseModel):
    """One conversation message in provider-neutral form.

    - role="user": ``content`` holds user text and/or ``tool_results`` holds
      results for the assistant's previous tool calls.
    - role="assistant": ``content`` holds assistant text and ``tool_calls``
      holds any tool invocations it requested.
    """

    role: Literal["user", "assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class LLMTurn(BaseModel):
    """One assistant completion."""

    text: str | None = None
    tool_calls: list[ToolCall] = []
    usage: Usage = Usage()
    stop_reason: str | None = None


class LLMProvider(ABC):
    """One chat completion with tool use. Implementations must be stateless
    across calls: the full conversation is passed in every time."""

    name: str

    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        system: str | None,
        messages: list[ChatMessage],
        tools: list[ToolDef],
        max_tokens: int = 1024,
    ) -> LLMTurn: ...
