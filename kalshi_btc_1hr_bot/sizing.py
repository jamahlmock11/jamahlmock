"""Fractional Kelly position sizing with hard dollar cap."""

from __future__ import annotations

from kalshi_btc_1hr_bot.config import SizingConfig


def kelly_contracts(
    *,
    win_prob: float,
    price: float,
    sizing: SizingConfig,
    confidence: float = 1.0,
    size_multiplier: float = 1.0,
) -> int:
    """Fractional Kelly sizing capped at max_trade_usd (0 = full bankroll pct)."""
    if price <= 0 or price >= 1 or win_prob <= price:
        return 0

    f_star = (win_prob - price) / (1.0 - price)
    mult = max(1.0, size_multiplier)
    f = max(0.0, min(1.0, f_star * sizing.kelly_fraction * confidence * mult))
    bankroll_cap = sizing.bankroll_usd * sizing.max_bankroll_pct
    if sizing.max_trade_usd > 0:
        max_dollars = min(bankroll_cap, sizing.max_trade_usd)
    else:
        max_dollars = bankroll_cap
    dollars = min(sizing.bankroll_usd * f, max_dollars)
    contracts = int(dollars / price)

    if contracts == 0 and max_dollars >= price and f_star > 0:
        contracts = 1

    while contracts > 0 and contracts * price > max_dollars + 1e-9:
        contracts -= 1

    return max(0, contracts)
