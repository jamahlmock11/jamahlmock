"""Mispricing detection: Kalshi price vs options-implied probability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from kalshi_bot.config import SeriesConfig, SmileConfig
from kalshi_bot.models.probability import (
    ImpliedProbResult,
    options_implied_prob_above,
    options_implied_prob_up,
)
from kalshi_bot.models.smile import VolSmile
from kalshi_bot.strategy.fees import quadratic_fee_per_contract


class Side(str, Enum):
    YES = "yes"
    NO = "no"


@dataclass(frozen=True)
class Mispricing:
    ticker: str
    series: str
    side: Side
    kalshi_price: float
    options_prob: float
    edge_pp: float
    edge_after_fees_pp: float
    strike: float
    spot: float
    vol: float
    seconds_to_expiry: float
    yes_bid: float | None
    yes_ask: float | None
    implied: ImpliedProbResult
    reason: str

    @property
    def edge_raw(self) -> float:
        return self.edge_pp / 100.0


def _seconds_to_expiry(close: datetime | None, now: datetime | None = None) -> float:
    if close is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    return (close - now).total_seconds()


def evaluate_market(
    market: dict,
    *,
    spot: float,
    smile: VolSmile,
    series_cfg: SeriesConfig,
    smile_cfg: SmileConfig,
    fee_rate: float = 0.07,
    fee_multiplier: float = 1.0,
    now: datetime | None = None,
) -> Mispricing | None:
    """Return a tradeable mispricing if edge clears the threshold after fees.

    Example: Kalshi YES ask = 0.22, options imply 0.378 → +15.8pp edge → BUY YES.
    """
    now = now or datetime.now(timezone.utc)
    strike = market.get("strike")
    close = market.get("close_time")
    if strike is None or close is None:
        return None
    if spot <= 0:
        return None

    series = market.get("series_ticker") or series_cfg.ticker
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")

    if series == "KXBTC15M":
        implied = options_implied_prob_up(
            spot_btc=spot,
            open_level=float(strike),
            close_time=close,
            smile=smile,
            rate=smile_cfg.risk_free_rate,
            dividend=smile_cfg.dividend_yield,
            now=now,
        )
    else:
        include_equal = (market.get("strike_type") or "").startswith("greater_or_equal")
        implied = options_implied_prob_above(
            spot_btc=spot,
            strike_btc=float(strike),
            close_time=close,
            smile=smile,
            rate=smile_cfg.risk_free_rate,
            dividend=smile_cfg.dividend_yield,
            now=now,
            include_equal=include_equal or True,
        )

    p = implied.probability
    secs = _seconds_to_expiry(close, now)

    # Effective min edge: widen when smile is stale or synthetic
    min_edge = series_cfg.min_edge_pp
    if smile.age_seconds > smile_cfg.max_smile_age_seconds:
        min_edge *= smile_cfg.stale_edge_multiplier
    if smile.is_synthetic:
        # Synthetic smiles are for plumbing demos only — do not take live risk
        # unless the edge is enormous.
        min_edge = max(min_edge * 3.0, 25.0)

    candidates: list[Mispricing] = []

    # Buy YES if options prob >> Kalshi ask
    if yes_ask is not None and 0 < yes_ask < 1:
        fee = quadratic_fee_per_contract(yes_ask, fee_rate=fee_rate, fee_multiplier=fee_multiplier)
        edge_pp = (p - yes_ask) * 100
        edge_after = (p - yes_ask - fee) * 100
        if edge_after >= min_edge:
            candidates.append(
                Mispricing(
                    ticker=market["ticker"],
                    series=series,
                    side=Side.YES,
                    kalshi_price=yes_ask,
                    options_prob=p,
                    edge_pp=edge_pp,
                    edge_after_fees_pp=edge_after,
                    strike=float(strike),
                    spot=spot,
                    vol=implied.vol_used,
                    seconds_to_expiry=secs,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    implied=implied,
                    reason=(
                        f"options imply {p*100:.1f}% vs Kalshi ask {yes_ask*100:.1f}% "
                        f"({edge_pp:.1f}pp raw, {edge_after:.1f}pp after fees)"
                    ),
                )
            )

    # Buy NO if (1-p) >> NO ask  (NO ask ≈ 1 - yes_bid)
    no_ask = market.get("no_ask")
    if no_ask is None and yes_bid is not None:
        no_ask = max(0.0, 1.0 - yes_bid)
    if no_ask is not None and 0 < no_ask < 1:
        q = 1.0 - p
        fee = quadratic_fee_per_contract(no_ask, fee_rate=fee_rate, fee_multiplier=fee_multiplier)
        edge_pp = (q - no_ask) * 100
        edge_after = (q - no_ask - fee) * 100
        if edge_after >= min_edge:
            candidates.append(
                Mispricing(
                    ticker=market["ticker"],
                    series=series,
                    side=Side.NO,
                    kalshi_price=no_ask,
                    options_prob=q,
                    edge_pp=edge_pp,
                    edge_after_fees_pp=edge_after,
                    strike=float(strike),
                    spot=spot,
                    vol=implied.vol_used,
                    seconds_to_expiry=secs,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    implied=implied,
                    reason=(
                        f"options imply NO {q*100:.1f}% vs Kalshi NO ask {no_ask*100:.1f}% "
                        f"({edge_pp:.1f}pp raw, {edge_after:.1f}pp after fees)"
                    ),
                )
            )

    if not candidates:
        return None
    return max(candidates, key=lambda m: m.edge_after_fees_pp)
