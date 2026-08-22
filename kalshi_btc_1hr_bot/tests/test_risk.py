"""Risk manager tests."""

from __future__ import annotations

from kalshi_btc_1hr_bot.config import BotConfig
from kalshi_btc_1hr_bot.risk import RiskManager


def test_allow_trade_second_hour_attempt():
    cfg = BotConfig()
    cfg.risk.max_trades_per_hour = 2
    risk = RiskManager(cfg)
    ok, reason = risk.allow_trade(
        ticker="KXBTCD-A",
        seconds_to_expiry=600,
        hour_trade_count=1,
    )
    assert ok
    assert reason == "ok"


def test_allow_trade_blocks_after_hour_limit():
    cfg = BotConfig()
    cfg.risk.max_trades_per_hour = 2
    risk = RiskManager(cfg)
    ok, reason = risk.allow_trade(
        ticker="KXBTCD-B",
        seconds_to_expiry=600,
        hour_trade_count=2,
    )
    assert not ok
    assert reason == "max_trades_per_hour"
