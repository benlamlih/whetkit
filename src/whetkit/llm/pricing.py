"""Rough USD cost estimates for known models.

Prices are per million tokens (input, output), hand-checked 2026-07-08
against the providers' public pricing pages. They rot — treat every
number this module produces as an estimate, which is how the CLI labels
it. Unknown models simply get no estimate.
"""

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-mini": (0.25, 2.00),
}


def estimate_cost_usd(model_id: str, tokens_in: int, tokens_out: int) -> float | None:
    """Best-effort estimate; None when the model isn't in the table.
    Dated model ids (claude-haiku-4-5-20251001) match their base entry."""
    # longest key first, so gpt-4.1-mini never matches the gpt-4.1 entry
    for known in sorted(PRICES_PER_MTOK, key=len, reverse=True):
        price_in, price_out = PRICES_PER_MTOK[known]
        if model_id == known or model_id.startswith(f"{known}-"):
            return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
    return None
