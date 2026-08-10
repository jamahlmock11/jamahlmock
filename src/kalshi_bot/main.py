"""Main trading / forecasting loop."""

from __future__ import annotations

import argparse
import logging
import time

from kalshi_bot.config import Settings, kalshi_base_url, load_config
from kalshi_bot.data.ibit_options import load_ibit_smile
from kalshi_bot.data.kalshi_client import KalshiClient
from kalshi_bot.execution.executor import Executor
from kalshi_bot.execution.position_monitor import PositionMonitor
from kalshi_bot.execution.risk import RiskManager
from kalshi_bot.models.probability import ImpliedProbResult, MarketKind
from kalshi_bot.strategy.decision import DecisionVerdict, TradeDecision
from kalshi_bot.strategy.mispricing import Mispricing, Side
from kalshi_bot.strategy.scanner import ForecastScanner, MispricingScanner
from kalshi_bot.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_runtime(settings: Settings | None = None):
    settings = settings or Settings()
    config = load_config(settings.config_path)
    setup_logging(settings.log_level)
    client = KalshiClient(
        base_url=kalshi_base_url(settings, config),
        api_key_id=settings.kalshi_api_key_id,
        private_key_pem=settings.resolve_private_key_pem(),
    )
    risk = RiskManager(config)
    executor = Executor(client, config, risk)
    scanner = MispricingScanner(client, config)
    return settings, config, client, scanner, executor


def decision_to_mispricing(decision: TradeDecision) -> Mispricing | None:
    """Adapt a forecast TRADE decision into the executor's Mispricing shape."""
    if decision.verdict != DecisionVerdict.TRADE or decision.side is None or decision.kalshi_price is None:
        return None
    implied = decision.forecast.options or ImpliedProbResult(
        probability=decision.forecast.probability_yes,
        vol_used=decision.forecast.components[0].vol_used if decision.forecast.components else 0.0,
        spot=decision.spot,
        strike=decision.strike,
        time_years=max(decision.seconds_to_expiry, 1.0) / (365.25 * 24 * 3600),
        log_moneyness=0.0,
        kind=MarketKind.ABOVE_STRIKE,
        smile_expiry="ensemble",
        smile_age_seconds=0.0,
    )
    return Mispricing(
        ticker=decision.ticker,
        series="KXBTCD",
        side=decision.side,
        kalshi_price=decision.kalshi_price,
        options_prob=decision.forecast_prob,
        edge_pp=decision.edge_pp,
        edge_after_fees_pp=decision.edge_after_fees_pp,
        strike=decision.strike,
        spot=decision.spot,
        vol=implied.vol_used,
        seconds_to_expiry=decision.seconds_to_expiry,
        yes_bid=decision.forecast.probability_lo,  # unused for sizing
        yes_ask=decision.kalshi_price if decision.side == Side.YES else None,
        implied=implied,
        reason="; ".join(decision.reasons) or decision.action.value,
    )


def run_forecast_once(allow_synthetic_smile: bool = False) -> int:
    settings, config, client, _scanner, executor = build_runtime()
    forecast_scanner = ForecastScanner(client, config)
    try:
        try:
            smile = load_ibit_smile(config.smile, allow_synthetic=allow_synthetic_smile)
        except Exception as exc:
            logger.warning("smile load failed (%s); continuing with realized-vol ensemble", exc)
            smile = None
        result = forecast_scanner.scan(smile)
        logger.info(
            "spot=%.2f source=%s official=%s scanned=%d trades=%d no_trade=%d",
            result.spot.brti,
            result.spot.source,
            result.spot.is_official,
            result.markets_scanned,
            len(result.trades),
            len(result.no_trades),
        )
        if not result.trades:
            logger.info("ENGINE VERDICT: NO TRADE")
            return 0
        for decision in result.trades:
            logger.info(
                "TRADE %s %s | %s gap=%.1fpp kalshi=%.1f%% forecast=%.1f%% EV=%.3f conf=%.2f Δ=%.1fpp t=%.0fs",
                decision.action.value,
                decision.ticker,
                decision.bot_action.value,
                decision.gap_pp,
                (decision.kalshi_price or 0) * 100,
                decision.forecast_prob * 100,
                decision.expected_value_per_contract,
                decision.confidence,
                decision.disagreement_pp,
                decision.seconds_to_expiry,
            )
            mis = decision_to_mispricing(decision)
            if mis is None:
                continue
            size = executor.risk.size(mis)
            executor.execute(mis, size, ignore_cooldown=True)
        return len(result.trades)
    finally:
        client.close()


def run_once(allow_synthetic_smile: bool = True) -> int:
    """Legacy mispricing path (kept for kalshi-scan / demos)."""
    settings, config, client, scanner, executor = build_runtime()
    try:
        smile = load_ibit_smile(config.smile, allow_synthetic=allow_synthetic_smile)
        result = scanner.scan(smile)
        logger.info(
            "spot=%.2f source=%s official=%s scanned=%d opportunities=%d",
            result.spot.brti,
            result.spot.source,
            result.spot.is_official,
            result.markets_scanned,
            len(result.opportunities),
        )
        for mis in result.opportunities[:20]:
            logger.info(
                "EDGE %s %s | kalshi=%.1f%% opts=%.1f%% edge=%.1fpp | K=%.2f S=%.2f σ=%.1f%% t=%.0fs",
                mis.side.value.upper(),
                mis.ticker,
                mis.kalshi_price * 100,
                mis.options_prob * 100,
                mis.edge_after_fees_pp,
                mis.strike,
                mis.spot,
                mis.vol * 100,
                mis.seconds_to_expiry,
            )
            size = executor.risk.size(mis)
            executor.execute(mis, size, ignore_cooldown=True)
        return len(result.opportunities)
    finally:
        client.close()


def run_loop(max_iterations: int | None = None, *, forecast: bool = True) -> None:
    settings, config, client, scanner, executor = build_runtime()
    forecast_scanner = ForecastScanner(client, config)
    position_monitor = PositionMonitor(client, config, executor)
    iterations = 0
    try:
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            try:
                smile = None
                if forecast:
                    try:
                        smile = load_ibit_smile(config.smile, allow_synthetic=False)
                    except Exception:
                        smile = None
                    position_monitor.manage_open_positions(smile=smile)
                    result = forecast_scanner.scan(smile)
                    for decision in result.trades:
                        mis = decision_to_mispricing(decision)
                        if mis is None:
                            continue
                        size = executor.risk.size(mis)
                        executor.execute(mis, size, ignore_cooldown=True)
                    if not result.trades:
                        logger.info("NO TRADE this cycle")
                else:
                    smile = load_ibit_smile(config.smile, allow_synthetic=True)
                    position_monitor.manage_open_positions(smile=smile)
                    result = scanner.scan(smile)
                    for mis in result.opportunities:
                        size = executor.risk.size(mis)
                        executor.execute(mis, size, ignore_cooldown=True)
            except Exception:
                logger.exception("scan iteration failed")
            time.sleep(config.scan_interval_seconds)
    finally:
        client.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kalshi BTC 1-hour forecasting engine")
    parser.add_argument("--once", action="store_true", help="Run a single forecast cycle")
    parser.add_argument("--iterations", type=int, default=None, help="Loop N times then exit")
    parser.add_argument(
        "--legacy-mispricing",
        action="store_true",
        help="Use legacy options-only mispricing path instead of forecast gates",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic smile fallback (demo only; raises trade bar)",
    )
    args = parser.parse_args(argv)
    if args.once:
        if args.legacy_mispricing:
            run_once(allow_synthetic_smile=args.allow_synthetic)
        else:
            run_forecast_once(allow_synthetic_smile=args.allow_synthetic)
    else:
        run_loop(max_iterations=args.iterations, forecast=not args.legacy_mispricing)


if __name__ == "__main__":
    main()
