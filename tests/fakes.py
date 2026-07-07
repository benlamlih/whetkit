"""Test doubles shared across the suite."""

from whetkit.llm import ChatMessage, LLMProvider, LLMTurn, ToolDef


class FakeProvider(LLMProvider):
    """Replays a script of LLMTurns and records everything it was asked."""

    name = "fake"

    def __init__(self, script: list[LLMTurn]):
        self.script = list(script)
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        model: str,
        system: str | None,
        messages: list[ChatMessage],
        tools: list[ToolDef],
        max_tokens: int = 1024,
    ) -> LLMTurn:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": [m.model_copy(deep=True) for m in messages],
                "tools": [t.model_copy(deep=True) for t in tools],
            }
        )
        if not self.script:
            raise AssertionError("FakeProvider script exhausted")
        return self.script.pop(0)
