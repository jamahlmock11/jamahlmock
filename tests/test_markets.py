from datetime import datetime, timedelta, timezone

from kalshi_bot.config import SeriesConfig, SmileConfig
from kalshi_bot.data.kalshi_client import normalize_market
from kalshi_bot.models.smile import synthetic_smile
from kalshi_bot.strategy.mispricing import evaluate_market


def test_normalize_kxbtc15m():
    raw = {
        "ticker": "KXBTC15M-26AUG070730-30",
        "event_ticker": "KXBTC15M-26AUG070730",
        "title": "BTC price up in next 15 mins?",
        "status": "active",
        "floor_strike": 64920.88,
        "strike_type": "greater_or_equal",
        "close_time": "2026-08-07T11:30:00Z",
        "open_time": "2026-08-07T11:15:00Z",
        "yes_bid_dollars": "0.9700",
        "yes_ask_dollars": "0.9710",
        "no_bid_dollars": "0.0290",
        "no_ask_dollars": "0.0300",
        "volume_fp": "100.00",
        "open_interest_fp": "50.00",
        "rules_primary": "test",
    }
    m = normalize_market(raw)
    assert m["series_ticker"] == "KXBTC15M"
    assert m["strike"] == 64920.88
    assert m["yes_ask"] == 0.971


def test_no_trade_when_fairly_priced():
    spot = 65000.0
    smile = synthetic_smile(spot, atm_iv=0.5)
    close = datetime.now(timezone.utc) + timedelta(minutes=30)
    # Deep ITM, Kalshi near 1.0 — no meaningful edge to buy
    market = {
        "ticker": "X",
        "series_ticker": "KXBTCD",
        "strike": 60000.0,
        "close_time": close,
        "yes_ask": 0.99,
        "yes_bid": 0.98,
        "no_ask": 0.02,
        "strike_type": "greater",
    }
    mis = evaluate_market(
        market,
        spot=spot,
        smile=smile,
        series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=8.0),
        smile_cfg=SmileConfig(),
    )
    assert mis is None
