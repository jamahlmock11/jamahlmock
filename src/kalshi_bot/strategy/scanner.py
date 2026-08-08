"""Scan Kalshi BTC 1-hour markets with evidence-gated forecasts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from kalshi_bot.config import BotConfig, SeriesConfig
from kalshi_bot.data.brti import SpotSnapshot, resolve_spot
from kalshi_bot.data.kalshi_client import KalshiClient, normalize_market
from kalshi_bot.models.smile import VolSmile
from kalshi_bot.strategy.decision import DecisionVerdict, TradeDecision, evaluate_forecast_market
from kalshi_bot.strategy.mispricing import Mispricing, evaluate_market

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    spot: SpotSnapshot
    smile: VolSmile
    opportunities: list[Mispricing]
    markets_scanned: int
    asof: datetime


@dataclass
class ForecastScanResult:
    spot: SpotSnapshot
    smile: VolSmile | None
    decisions: list[TradeDecision]
    trades: list[TradeDecision]
    no_trades: list[TradeDecision]
    markets_scanned: int
    asof: datetime
    top_blockers: list[tuple[str, int]] = field(default_factory=list)


def _is_hourly_event(raw: dict) -> bool:
    """KXBTCD includes daily buckets; keep hourly cadence only when gated."""
    meta = (raw.get("product_metadata") or {}) if isinstance(raw, dict) else {}
    cadence = str(meta.get("cadence") or "").lower()
    if cadence == "hourly":
        return True
    if cadence == "daily":
        return False
    # Heuristic: hourly event tickers end with hour code like ...0804 (HH)
    # Daily often ends with ...0817 for 5pm. Prefer open→close span near 1h.
    open_t = raw.get("open_time")
    close_t = raw.get("close_time")
    if open_t and close_t:
        try:
            from kalshi_bot.data.kalshi_client import _parse_ts

            o = _parse_ts(open_t)
            c = _parse_ts(close_t)
            if o and c:
                span = (c - o).total_seconds()
                return 50 * 60 <= span <= 70 * 60
        except Exception:
            pass
    return True


class MispricingScanner:
    def __init__(self, client: KalshiClient, config: BotConfig) -> None:
        self.client = client
        self.config = config

    def _series_cfg(self, ticker: str) -> SeriesConfig | None:
        for s in self.config.series:
            if s.ticker == ticker and s.enabled:
                return s
        return None

    def _within_horizon(self, series: str, seconds: float) -> bool:
        risk = self.config.risk
        if seconds < risk.min_seconds_to_expiry:
            return False
        if series == "KXBTC15M":
            return seconds <= risk.max_seconds_to_expiry_15m
        if series == "KXBTCD":
            return seconds <= risk.max_seconds_to_expiry_1h
        return True

    def scan(self, smile: VolSmile, spot_override: float | None = None) -> ScanResult:
        spot = resolve_spot(self.client, fallback_btc=spot_override or smile.spot_btc)
        opportunities: list[Mispricing] = []
        scanned = 0
        now = datetime.now(timezone.utc)

        for series_cfg in self.config.series:
            if not series_cfg.enabled:
                continue
            for raw in self.client.iter_markets(series_cfg.ticker, status="open"):
                if series_cfg.ticker == "KXBTCD" and self.config.forecast_gates.hourly_only:
                    if not _is_hourly_event(raw):
                        continue
                market = normalize_market(raw)
                scanned += 1
                close = market.get("close_time")
                if close is None:
                    continue
                secs = (close - now).total_seconds()
                if not self._within_horizon(series_cfg.ticker, secs):
                    continue
                if not market.get("yes_ask") and not market.get("yes_bid"):
                    continue
                mis = evaluate_market(
                    market,
                    spot=spot.brti,
                    smile=smile,
                    series_cfg=series_cfg,
                    smile_cfg=self.config.smile,
                    fee_rate=self.config.fee_rate,
                    fee_multiplier=self.config.fee_multiplier,
                    now=now,
                )
                if mis is not None:
                    opportunities.append(mis)

        opportunities.sort(key=lambda m: m.edge_after_fees_pp, reverse=True)
        logger.info(
            "scan complete markets=%d opps=%d spot=%.2f (%s) smile_atm=%.1f%%",
            scanned,
            len(opportunities),
            spot.brti,
            spot.source,
            smile.atm_iv * 100,
        )
        return ScanResult(
            spot=spot,
            smile=smile,
            opportunities=opportunities,
            markets_scanned=scanned,
            asof=now,
        )


class ForecastScanner:
    """Institutional 1-hour forecasting scanner (accuracy-first)."""

    def __init__(self, client: KalshiClient, config: BotConfig) -> None:
        self.client = client
        self.config = config

    def scan(self, smile: VolSmile | None, spot_override: float | None = None) -> ForecastScanResult:
        fallback = spot_override or (smile.spot_btc if smile else None)
        spot = resolve_spot(self.client, fallback_btc=fallback)
        decisions: list[TradeDecision] = []
        scanned = 0
        now = datetime.now(timezone.utc)
        gates = self.config.forecast_gates

        for series_cfg in self.config.series:
            if not series_cfg.enabled or series_cfg.ticker != "KXBTCD":
                continue
            for raw in self.client.iter_markets(series_cfg.ticker, status="open"):
                if gates.hourly_only and not _is_hourly_event(raw):
                    continue
                market = normalize_market(raw)
                scanned += 1
                close = market.get("close_time")
                if close is None:
                    continue
                secs = (close - now).total_seconds()
                if secs < gates.min_seconds_to_expiry or secs > gates.max_seconds_to_expiry:
                    continue
                if not market.get("yes_ask") and not market.get("yes_bid"):
                    continue
                decision = evaluate_forecast_market(
                    market,
                    spot=spot.brti,
                    smile=smile,
                    series_cfg=series_cfg,
                    smile_cfg=self.config.smile,
                    gates=gates,
                    fee_rate=self.config.fee_rate,
                    fee_multiplier=self.config.fee_multiplier,
                    now=now,
                )
                decisions.append(decision)

        trades = [d for d in decisions if d.verdict == DecisionVerdict.TRADE]
        no_trades = [d for d in decisions if d.verdict == DecisionVerdict.NO_TRADE]
        trades.sort(key=lambda d: d.expected_value_per_contract, reverse=True)
        # Rank near-misses by conservative edge for diagnostics
        no_trades.sort(key=lambda d: d.edge_after_fees_pp, reverse=True)

        blocker_counts: dict[str, int] = {}
        for d in no_trades:
            for b in d.blockers:
                key = b.split(" (")[0]
                blocker_counts[key] = blocker_counts.get(key, 0) + 1
        top_blockers = sorted(blocker_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]

        logger.info(
            "forecast scan markets=%d trades=%d no_trade=%d spot=%.2f (%s) smile=%s",
            scanned,
            len(trades),
            len(no_trades),
            spot.brti,
            spot.source,
            "none" if smile is None else f"atm={smile.atm_iv*100:.1f}%",
        )
        return ForecastScanResult(
            spot=spot,
            smile=smile,
            decisions=decisions,
            trades=trades,
            no_trades=no_trades,
            markets_scanned=scanned,
            asof=now,
            top_blockers=top_blockers,
        )
