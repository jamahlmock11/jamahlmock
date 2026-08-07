"""Map Kalshi binary contracts → options-implied probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from kalshi_bot.models.black_scholes import BSInputs, digital_call_probability
from kalshi_bot.models.smile import VolSmile


class MarketKind(str, Enum):
    UP_DOWN_15M = "up_down_15m"
    ABOVE_STRIKE = "above_strike"


@dataclass(frozen=True)
class ImpliedProbResult:
    probability: float
    vol_used: float
    spot: float
    strike: float
    time_years: float
    log_moneyness: float
    kind: MarketKind
    smile_expiry: str
    smile_age_seconds: float


def years_to_expiry(close_time: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)
    seconds = (close_time - now).total_seconds()
    return max(seconds, 1.0) / (365.25 * 24 * 3600)


def options_implied_prob_above(
    *,
    spot_btc: float,
    strike_btc: float,
    close_time: datetime,
    smile: VolSmile,
    rate: float = 0.0,
    dividend: float = 0.0,
    now: datetime | None = None,
    include_equal: bool = True,
) -> ImpliedProbResult:
    """Black-Scholes digital probability that BRTI settles above (or at) strike.

    Vol is read from the IBIT-derived smile at the strike's log-moneyness,
    then applied over the Kalshi contract's remaining life. This is the core
    mispricing signal: compare this probability to Kalshi's traded price.
    """
    t = years_to_expiry(close_time, now)
    # Re-anchor smile spot to live BRTI while keeping relative smile shape.
    lm = math.log(strike_btc / spot_btc)
    # Smile was built vs smile.spot_btc; interpolate in log-moneyness space
    # so shape transfers when spot moves between smile snapshot and now.
    vol = smile.iv_at_log_m(lm)
    # Floor extremely short-dated vols to avoid numerical blow-ups while
    # still reflecting the smile level.
    vol = max(vol, 0.05)
    inp = BSInputs(
        spot=spot_btc,
        strike=strike_btc,
        time_years=t,
        vol=vol,
        rate=rate,
        dividend=dividend,
    )
    # Continuous BS: P(S>K)=N(d2). For greater_or_equal markets the
    # continuous adjustment is negligible at BTC price precision.
    p = digital_call_probability(inp)
    if not include_equal:
        # same continuous limit
        pass
    return ImpliedProbResult(
        probability=p,
        vol_used=vol,
        spot=spot_btc,
        strike=strike_btc,
        time_years=t,
        log_moneyness=lm,
        kind=MarketKind.ABOVE_STRIKE,
        smile_expiry=smile.expiry,
        smile_age_seconds=smile.age_seconds,
    )


def options_implied_prob_up(
    *,
    spot_btc: float,
    open_level: float,
    close_time: datetime,
    smile: VolSmile,
    rate: float = 0.0,
    dividend: float = 0.0,
    now: datetime | None = None,
) -> ImpliedProbResult:
    """Probability that end BRTI >= open BRTI (KXBTC15M style)."""
    result = options_implied_prob_above(
        spot_btc=spot_btc,
        strike_btc=open_level,
        close_time=close_time,
        smile=smile,
        rate=rate,
        dividend=dividend,
        now=now,
        include_equal=True,
    )
    return ImpliedProbResult(
        probability=result.probability,
        vol_used=result.vol_used,
        spot=result.spot,
        strike=result.strike,
        time_years=result.time_years,
        log_moneyness=result.log_moneyness,
        kind=MarketKind.UP_DOWN_15M,
        smile_expiry=result.smile_expiry,
        smile_age_seconds=result.smile_age_seconds,
    )
