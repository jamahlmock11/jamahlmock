from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from kalshi_bot.config import BotConfig, V6Config
from kalshi_bot.data.markets_15m import CRYPTO_15M_SERIES
from kalshi_bot.strategy.v6_upgrades import V6IntelligenceEngine

from run import V6Scanner, movement_vs_open


def _market(ticker: str, series: str, strike: float) -> dict:
    close = datetime.now(timezone.utc) + timedelta(minutes=10)
    open_t = close - timedelta(minutes=14)
    return {
        "ticker": ticker,
        "event_ticker": f"{series}-TEST",
        "floor_strike": strike,
        "close_time": close.isoformat().replace("+00:00", "Z"),
        "open_time": open_t.isoformat().replace("+00:00", "Z"),
        "yes_bid_dollars": "0.4500",
        "yes_ask_dollars": "0.4700",
        "no_ask_dollars": "0.5500",
    }


def test_movement_vs_open():
    assert movement_vs_open(101.0, 100.0) == 1.0
    assert movement_vs_open(99.0, 100.0) == -1.0
    assert movement_vs_open(100.0, None) is None


def test_scanner_iterates_multiple_series():
    client = MagicMock()
    eth_market = _market("KXETH15M-TEST", "KXETH15M", 3000.0)
    sol_market = _market("KXSOL15M-TEST", "KXSOL15M", 150.0)

    def iter_side_effect(series_ticker, status="open"):
        if series_ticker == "KXETH15M":
            yield eth_market
        elif series_ticker == "KXSOL15M":
            yield sol_market

    client.iter_markets.side_effect = iter_side_effect

    config = BotConfig(v6=V6Config(strict_min_gap_dollars=0.25))
    engine = V6IntelligenceEngine(config.v6, client=client)
    engine.evaluate = MagicMock(
        side_effect=lambda market, **kwargs: MagicMock(
            verdict="NO_TRADE",
            strict_gap_dollars=0.0,
            audit_record=None,
            blockers=[],
        )
    )

    specs = [CRYPTO_15M_SERIES["KXETH15M"], CRYPTO_15M_SERIES["KXSOL15M"]]
    scanner = V6Scanner(client, config, engine, specs)

    import run as run_module

    original = run_module.resolve_series_spot
    try:
        run_module.resolve_series_spot = lambda client, spec, **kwargs: MagicMock(
            brti=3000.0 if spec.asset == "ETH" else 150.0,
            source="test",
            is_official=True,
            asset=spec.asset,
            series_ticker=spec.ticker,
        )
        run_module.estimate_realized_vol = lambda **kwargs: MagicMock(
            annualized_vol=0.5,
            source="test",
            spot=1.0,
        )
        result = scanner.scan(smile=None)
    finally:
        run_module.resolve_series_spot = original

    assert result.markets_scanned == 2
    assert client.iter_markets.call_count == 2
    assert engine.evaluate.call_count == 2
