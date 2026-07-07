from whetkit.llm.pricing import estimate_cost_usd


def test_known_model_estimates() -> None:
    cost = estimate_cost_usd("gpt-4.1-mini", 1_000_000, 0)
    assert cost == 0.40


def test_dated_model_id_matches_base_entry() -> None:
    cost = estimate_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
    assert cost == 6.00


def test_unknown_model_returns_none() -> None:
    assert estimate_cost_usd("mystery-model-9000", 1000, 1000) is None
