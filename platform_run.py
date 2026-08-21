#!/usr/bin/env python3
"""Production Kalshi dual-strategy platform runner.

KXBTC15M — settlement-aware mispricing (Strategy A)
KXBTCD   — 1-hour forecast (Strategy B)

Live market data and account balances are always real.
Order execution requires platform.trading_enabled=true AND live execution config.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from kalshi_bot.platform.runner import ProductionPlatform
from kalshi_bot.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kalshi production platform (KXBTC15M + KXBTCD)")
    parser.add_argument("--loop", action="store_true", help="Continuous scan loop")
    parser.add_argument("--interval", type=float, default=None, help="Scan interval seconds")
    parser.add_argument("--execute", action="store_true", help="Submit live orders when safety allows")
    parser.add_argument("--reset", action="store_true", help="Reset risk/kill-switch state before starting")
    parser.add_argument("--web", action="store_true", help="Start web dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-15m", action="store_true", help="Disable KXBTC15M strategy")
    parser.add_argument("--no-1h", action="store_true", help="Disable KXBTCD strategy")
    args = parser.parse_args(argv)

    setup_logging("INFO")

    if args.web:
        _run_with_web(args)
        return

    platform = ProductionPlatform(enable_15m=not args.no_15m, enable_1h=not args.no_1h)
    if args.reset:
        platform.reset_trading_state()
    interval = args.interval or platform.config.scan_interval_seconds
    try:
        if args.loop:
            while True:
                result = platform.run_cycle(execute=args.execute)
                logger.info(
                    "cycle: 15m=%d 1h=%d status=%s",
                    len(result.decisions_15m),
                    len(result.decisions_1h),
                    result.status.get("status_label"),
                )
                time.sleep(interval)
        else:
            platform.run_cycle(execute=args.execute)
    finally:
        platform.close()


def _run_with_web(args: argparse.Namespace) -> None:
    import threading

    from kalshi_bot.web.web_server import run_server

    platform = ProductionPlatform(enable_15m=not args.no_15m, enable_1h=not args.no_1h)
    if args.reset:
        platform.reset_trading_state()
    interval = args.interval or platform.config.scan_interval_seconds
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            try:
                platform.run_cycle(execute=args.execute)
            except Exception:
                logger.exception("platform cycle failed")
            stop.wait(interval)

    t = threading.Thread(target=loop, name="platform-scan", daemon=True)
    t.start()
    try:
        run_server(host=args.host, port=args.port, with_scan=False)
    finally:
        stop.set()
        platform.close()


if __name__ == "__main__":
    main()
