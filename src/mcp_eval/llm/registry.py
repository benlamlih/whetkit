"""Model-string parsing and provider lookup.

Models are addressed as ``provider:model_id`` (e.g. ``anthropic:claude-sonnet-5``,
``openai:gpt-5.2``). A bare model id defaults to the Anthropic provider.
"""

from functools import cache

from mcp_eval.llm.base import LLMProvider

DEFAULT_PROVIDER = "anthropic"


def parse_model(model: str) -> tuple[str, str]:
    """Split ``provider:model_id`` -> (provider_name, model_id)."""
    if ":" in model:
        provider_name, model_id = model.split(":", 1)
    else:
        provider_name, model_id = DEFAULT_PROVIDER, model
    if not model_id:
        raise ValueError(f"invalid model string {model!r}")
    return provider_name, model_id


@cache
def get_provider(provider_name: str) -> LLMProvider:
    """Instantiate (and cache) a provider by name. Imports lazily so one
    missing SDK/key never blocks the other provider."""
    if provider_name == "anthropic":
        from mcp_eval.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if provider_name == "openai":
        from mcp_eval.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"unknown provider {provider_name!r} (expected 'anthropic' or 'openai')")
