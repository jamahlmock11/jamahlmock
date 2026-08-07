"""Main trading loop."""

from __future__ import annotations

import argparse
import logging
import time

from kalshi_bot.config import Settings, kalshi_base_url, load_config
from kalshi_bot.data.ibit_options import load_ibit_smile
from kalshi_bot.data.kalshi_client import KalshiClient
from kalshi_bot.execution.executor import Executor
from kalshi_bot.execution.risk import RiskManager
from kalshi_bot.strategy.scanner import MispricingScanner
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


def run_once(allow_synthetic_smile: bool = True) -> int:
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


def run_loop(max_iterations: int | None = None) -> None:
    settings, config, client, scanner, executor = build_runtime()
    iterations = 0
    try:
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            try:
                smile = load_ibit_smile(config.smile, allow_synthetic=True)
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
    parser = argparse.ArgumentParser(description="Kalshi BTC IBIT-smile mispricing bot")
    parser.add_argument("--once", action="store_true", help="Run a single scan cycle")
    parser.add_argument("--iterations", type=int, default=None, help="Loop N times then exit")
    args = parser.parse_args(argv)
    if args.once:
        run_once()
    else:
        run_loop(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
