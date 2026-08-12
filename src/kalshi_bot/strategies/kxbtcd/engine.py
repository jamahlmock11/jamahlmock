"""KXBTCD 1-hour forecast strategy — separate from 15m mispricing."""

from __future__ import annotations

import time

from kalshi_bot.config import BotConfig
from kalshi_bot.data.ibit_options import load_ibit_smile
from kalshi_bot.data.kalshi_client import KalshiClient
from kalshi_bot.models.volatility_regime import classify_vol_regime, neutral_regime
from kalshi_bot.platform.decision_states import TradeSignal
from kalshi_bot.platform.observation import DataQuality, ObservationBundle
from kalshi_bot.strategies.base import MarketDecision, StrategyEngine
from kalshi_bot.strategy.decision import DecisionVerdict
from kalshi_bot.strategy.scanner import ForecastScanner


class KxbtcdStrategy(StrategyEngine):
    name = "KXBTCD"
    series = "KXBTCD"

    def __init__(self, client: KalshiClient, config: BotConfig) -> None:
        self.client = client
        self.config = config
        self.scanner = ForecastScanner(client, config)
        self._smile = None

    def evaluate_all(self) -> list[MarketDecision]:
        obs = ObservationBundle()
        if self._smile is None:
            try:
                self._smile = load_ibit_smile(self.config.smile, allow_synthetic=False)
            except Exception:
                return []

        t0 = time.time()
        result = self.scanner.scan(self._smile)
        obs.add(
            "brti",
            result.spot.brti,
            source=result.spot.source,
            quality=DataQuality.FRESH if result.spot.is_official else DataQuality.DEGRADED,
            latency_ms=(time.time() - t0) * 1000,
        )
        if self.config.platform.enable_vol_regime:
            regime = classify_vol_regime(result.realized_vol_ann)
            regime_label = regime.regime.value
        else:
            regime = neutral_regime(result.realized_vol_ann)
            regime_label = "off"

        decisions: list[MarketDecision] = []
        for d in result.decisions:
            if d.verdict == DecisionVerdict.TRADE:
                signal = TradeSignal.BUY_YES if d.side and d.side.value == "yes" else TradeSignal.BUY_NO
            else:
                signal = TradeSignal.NO_TRADE
            why_trade = d.reason if signal in (TradeSignal.BUY_YES, TradeSignal.BUY_NO) else ""
            why_not = d.reason if signal == TradeSignal.NO_TRADE else ""

            decisions.append(
                MarketDecision(
                    ticker=d.ticker,
                    series=self.series,
                    signal=signal.value,
                    reason=d.reason or signal.value,
                    model_prob_yes=d.forecast_prob,
                    model_prob_no=1.0 - d.forecast_prob,
                    fair_yes=d.forecast_prob,
                    fair_no=1.0 - d.forecast_prob,
                    executable_yes=d.kalshi_price if d.side and d.side.value == "yes" else None,
                    executable_no=d.kalshi_price if d.side and d.side.value == "no" else None,
                    raw_edge=d.edge_pp / 100.0,
                    net_edge=d.edge_after_fees_pp / 100.0,
                    confidence=d.forecast.confidence,
                    regime=regime_label,
                    strike=d.strike,
                    seconds_to_expiry=d.seconds_to_expiry,
                    btc_spot=result.spot.brti,
                    yes_price=d.kalshi_price,
                    no_price=None,
                    why_trade=why_trade,
                    why_not_trade=why_not,
                    observations=obs,
                    features={"hourly_scan": True, "realized_vol": result.realized_vol_ann},
                    execute_verdict="TRADE_YES" if signal == TradeSignal.BUY_YES else "TRADE_NO" if signal == TradeSignal.BUY_NO else None,
                    contracts=0,
                )
            )
        decisions.sort(key=lambda x: x.net_edge, reverse=True)
        return decisions
