"""Monitor open Kalshi positions and exit before expiry on drawdown or edge flip."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kalshi_bot.config import BotConfig
from kalshi_bot.data.brti import resolve_series_spot, resolve_spot
from kalshi_bot.data.kalshi_client import KalshiClient, normalize_market
from kalshi_bot.data.markets_15m import get_series_spec, parse_series_ticker
from kalshi_bot.data.realized_vol import estimate_realized_vol
from kalshi_bot.execution.executor import Executor
from kalshi_bot.models.forecast import forecast_prob_above
from kalshi_bot.models.smile import VolSmile
from kalshi_bot.strategy.fees import quadratic_fee_per_contract
from kalshi_bot.strategy.mispricing import Side

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenPosition:
    ticker: str
    series: str
    side: str  # "yes" | "no"
    contracts: int
    cost_usd: float
    avg_entry_price: float


@dataclass(frozen=True)
class ExitSignal:
    ticker: str
    side: str
    contracts: int
    reason: str
    drawdown_pct: float
    net_edge_dollars: float
    model_prob_side: float
    exit_bid: float
    seconds_to_expiry: float


def _series_from_ticker(ticker: str) -> str:
    parsed = parse_series_ticker(ticker)
    if parsed:
        return parsed
    if ticker.startswith("KXBTCD"):
        return "KXBTCD"
    return ticker.split("-")[0]


def parse_open_positions(payload: dict[str, Any]) -> list[OpenPosition]:
    """Parse Kalshi /portfolio/positions market_positions into open legs."""
    positions: list[OpenPosition] = []
    for raw in payload.get("market_positions") or []:
        fp = float(raw.get("position_fp") or raw.get("position") or 0)
        if abs(fp) < 0.01:
            continue
        ticker = raw.get("ticker") or ""
        if not ticker:
            continue
        cost = float(raw.get("market_exposure_dollars") or 0.0)
        contracts = int(abs(fp))
        if contracts <= 0:
            continue
        side = Side.YES.value if fp > 0 else Side.NO.value
        avg_entry = cost / contracts if cost > 0 and contracts > 0 else 0.0
        positions.append(
            OpenPosition(
                ticker=ticker,
                series=_series_from_ticker(ticker),
                side=side,
                contracts=contracts,
                cost_usd=cost,
                avg_entry_price=avg_entry,
            )
        )
    return positions


def drawdown_pct(*, avg_entry_price: float, exit_bid: float) -> float:
    """Fractional loss from entry to executable exit bid (0..1)."""
    if avg_entry_price <= 0 or exit_bid <= 0:
        return 0.0
    loss = (avg_entry_price - exit_bid) / avg_entry_price
    return max(0.0, loss)


def net_exit_edge_dollars(*, model_prob_side: float, exit_bid: float, fee_rate: float = 0.07) -> float:
    """Net edge for holding the side at the executable exit bid."""
    fee = quadratic_fee_per_contract(exit_bid, fee_rate=fee_rate)
    return model_prob_side - exit_bid - fee


def should_exit_position(
    *,
    drawdown: float,
    net_edge: float,
    max_drawdown_pct: float,
    exit_on_edge_flip: bool,
) -> tuple[bool, str]:
    if drawdown >= max_drawdown_pct:
        return True, f"drawdown {drawdown*100:.1f}% >= {max_drawdown_pct*100:.0f}%"
    if exit_on_edge_flip and net_edge < 0:
        return True, f"edge flipped (net {net_edge*100:.1f}¢ vs exit bid)"
    return False, ""


class PositionMonitor:
    """Poll open positions and trigger exits before settlement."""

    def __init__(
        self,
        client: KalshiClient,
        config: BotConfig,
        executor: Executor,
    ) -> None:
        self.client = client
        self.config = config
        self.executor = executor

    def manage_open_positions(self, *, smile: VolSmile | None = None) -> list[ExitSignal]:
        exit_cfg = self.config.exit
        if not exit_cfg.enabled or not self.client.authenticated:
            return []

        try:
            payload = self.client.get_positions(countFilter="position")
        except Exception as exc:
            logger.warning("position poll failed: %s", exc)
            return []

        signals: list[ExitSignal] = []
        for pos in parse_open_positions(payload):
            if pos.series not in exit_cfg.series_tickers:
                continue
            signal = self._evaluate_position(pos, smile=smile)
            if signal is None:
                continue
            signals.append(signal)
            fill = self.executor.close_position(
                ticker=signal.ticker,
                side=signal.side,
                contracts=signal.contracts,
                price=signal.exit_bid,
                reason=signal.reason,
            )
            if fill is not None:
                self.executor.risk.release_position(pos.ticker, pos.contracts, pos.cost_usd)
        return signals

    def _evaluate_position(self, pos: OpenPosition, *, smile: VolSmile | None) -> ExitSignal | None:
        exit_cfg = self.config.exit
        try:
            raw = self.client.get(f"/markets/{pos.ticker}")
            market = normalize_market(raw.get("market", raw))
        except Exception as exc:
            logger.warning("market fetch failed for %s: %s", pos.ticker, exc)
            return None

        close = market.get("close_time")
        now = datetime.now(timezone.utc)
        secs = max((close - now).total_seconds(), 0) if close else 0.0
        if secs < exit_cfg.min_seconds_to_expiry:
            logger.info("skip exit %s: too close to expiry (%.0fs)", pos.ticker, secs)
            return None

        if pos.side == Side.YES.value:
            exit_bid = market.get("yes_bid")
        else:
            exit_bid = market.get("no_bid")
        if exit_bid is None or exit_bid <= 0:
            logger.info("skip exit %s: no exit bid", pos.ticker)
            return None

        strike = market.get("strike")
        if strike is None or strike <= 0:
            return None

        series_spec = get_series_spec(pos.series)
        if series_spec is not None:
            spot_snap = resolve_series_spot(self.client, series_spec, brti_cfg=self.config.brti)
            kraken_pair = series_spec.kraken_pair or "XBTUSD"
        else:
            spot_snap = resolve_spot(self.client, brti_cfg=self.config.brti, series_ticker=pos.series)
            kraken_pair = "XBTUSD"
        realized = estimate_realized_vol(horizon_seconds=max(secs, 60.0), kraken_pair=kraken_pair)
        forecast = forecast_prob_above(
            spot=spot_snap.brti,
            strike=float(strike),
            close_time=close,
            smile=smile,
            realized=realized,
            yes_bid=market.get("yes_bid"),
            yes_ask=market.get("yes_ask"),
        )
        model_prob_yes = forecast.probability_yes
        model_prob_side = model_prob_yes if pos.side == Side.YES.value else 1.0 - model_prob_yes

        dd = drawdown_pct(avg_entry_price=pos.avg_entry_price, exit_bid=exit_bid)
        net_edge = net_exit_edge_dollars(
            model_prob_side=model_prob_side,
            exit_bid=exit_bid,
            fee_rate=self.config.fee_rate * self.config.fee_multiplier,
        )
        should_exit, reason = should_exit_position(
            drawdown=dd,
            net_edge=net_edge,
            max_drawdown_pct=exit_cfg.max_drawdown_pct,
            exit_on_edge_flip=exit_cfg.exit_on_edge_flip,
        )
        if not should_exit:
            return None

        logger.info(
            "EXIT SIGNAL %s %s x%d | %s | dd=%.1f%% edge=%.1f¢ model=%.1f%% bid=%.0f¢ t=%.0fs",
            pos.side.upper(),
            pos.ticker,
            pos.contracts,
            reason,
            dd * 100,
            net_edge * 100,
            model_prob_side * 100,
            exit_bid * 100,
            secs,
        )
        return ExitSignal(
            ticker=pos.ticker,
            side=pos.side,
            contracts=pos.contracts,
            reason=reason,
            drawdown_pct=dd,
            net_edge_dollars=net_edge,
            model_prob_side=model_prob_side,
            exit_bid=exit_bid,
            seconds_to_expiry=secs,
        )
