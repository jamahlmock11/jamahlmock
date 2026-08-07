"""Scan open Kalshi BTC markets for IBIT-smile mispricings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from kalshi_bot.config import BotConfig, SeriesConfig
from kalshi_bot.data.brti import SpotSnapshot, resolve_spot
from kalshi_bot.data.kalshi_client import KalshiClient, normalize_market
from kalshi_bot.models.smile import VolSmile
from kalshi_bot.strategy.mispricing import Mispricing, evaluate_market

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    spot: SpotSnapshot
    smile: VolSmile
    opportunities: list[Mispricing]
    markets_scanned: int
    asof: datetime


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
        # Prefer live spot for probability; keep smile shape in log-moneyness.
        opportunities: list[Mispricing] = []
        scanned = 0
        now = datetime.now(timezone.utc)

        for series_cfg in self.config.series:
            if not series_cfg.enabled:
                continue
            for raw in self.client.iter_markets(series_cfg.ticker, status="open"):
                market = normalize_market(raw)
                scanned += 1
                close = market.get("close_time")
                if close is None:
                    continue
                secs = (close - now).total_seconds()
                if not self._within_horizon(series_cfg.ticker, secs):
                    continue
                # Skip empty books
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
