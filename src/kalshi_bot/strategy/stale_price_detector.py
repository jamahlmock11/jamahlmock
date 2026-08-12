"""Detect Kalshi prices that lag material BTC moves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StalePriceAssessment:
    btc_move_pct: float
    expected_prob_shift: float
    kalshi_prob_shift: float
    lag_pp: float
    is_stale: bool
    reason: str


def assess_stale_kalshi_price(
    *,
    prev_btc: float | None,
    curr_btc: float,
    prev_yes_mid: float | None,
    curr_yes_mid: float | None,
    model_prob_yes: float,
    strike: float,
    seconds_to_expiry: float,
    min_btc_move_pct: float = 0.0008,
    min_lag_pp: float = 0.04,
) -> StalePriceAssessment:
    """Flag when BTC moved but Kalshi mid did not adjust proportionally."""
    if prev_btc is None or prev_btc <= 0 or curr_yes_mid is None or prev_yes_mid is None:
        return StalePriceAssessment(0.0, 0.0, 0.0, 0.0, False, "insufficient_history")

    btc_move = (curr_btc - prev_btc) / prev_btc
    kalshi_shift = curr_yes_mid - prev_yes_mid

    # Rough sensitivity: closer to expiry and strike → larger prob response per $ move
    dist = abs(curr_btc - strike) / max(curr_btc, 1.0)
    time_factor = max(0.2, min(1.0, 900.0 / max(seconds_to_expiry, 60.0)))
    sensitivity = time_factor / max(dist, 0.0005)
    expected_shift = btc_move * sensitivity * 0.5
    expected_shift = max(-0.25, min(0.25, expected_shift))

    lag = abs(expected_shift - kalshi_shift)
    stale = abs(btc_move) >= min_btc_move_pct and lag >= min_lag_pp
    reason = "kalshi_lagging_btc" if stale else "aligned"
    return StalePriceAssessment(
        btc_move_pct=btc_move,
        expected_prob_shift=expected_shift,
        kalshi_prob_shift=kalshi_shift,
        lag_pp=lag,
        is_stale=stale,
        reason=reason,
    )
