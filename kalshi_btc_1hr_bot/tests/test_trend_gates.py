"""Tests for trend alignment and flow confirmation gates."""

from __future__ import annotations

from kalshi_btc_1hr_bot.config import BotConfig
from kalshi_btc_1hr_bot.data_feed import FundingRate, MarketData
from kalshi_btc_1hr_bot.edge import TradeSignal
from kalshi_btc_1hr_bot.ensemble import ModelVote
from kalshi_btc_1hr_bot.evidence import DirectionalEvidence
from kalshi_btc_1hr_bot.trend_gates import (
    FlowSnapshot,
    apply_confirmation_gates,
    check_flow_confirmation,
    check_trend_alignment,
)


def _data(*, mu_5m: float = 0.0001, spot: float = 65000.0) -> MarketData:
    import numpy as np

    return MarketData(
        spot=spot,
        vwap=spot,
        funding_rate=0.0,
        annualized_vol=0.5,
        mu_5m=mu_5m,
        mu_15m=mu_5m,
        mu_30m=mu_5m,
        closes_1m=np.array([spot]),
        funding=FundingRate(0.0),
    )


def test_below_allowed_when_spot_above_strike_by_default():
    """Mean-reversion: BELOW with spot above strike is OK when position check is off."""
    ok, msg = check_trend_alignment(
        side="no",
        spot=66000,
        strike=65000,
        data=_data(mu_5m=-0.0002),
        min_momentum=0.0,
        require_spot_vs_strike=False,
    )
    assert ok, msg


def test_spot_vs_strike_optional_strict_mode():
    ok, _ = check_trend_alignment(
        side="no",
        spot=66000,
        strike=65000,
        data=_data(mu_5m=-0.0002),
        min_momentum=0.0,
        require_spot_vs_strike=True,
    )
    assert not ok


def test_trend_alignment_no_requires_down_momentum():
    ok, _ = check_trend_alignment(
        side="no", spot=64000, strike=65000, data=_data(mu_5m=-0.0002), min_momentum=0.0
    )
    assert ok
    ok2, msg = check_trend_alignment(
        side="no", spot=64000, strike=65000, data=_data(mu_5m=0.0002), min_momentum=0.0
    )
    assert not ok2
    assert "momentum not down" in msg


def test_flow_confirmation():
    flow = FlowSnapshot(yes_volume=100, no_volume=40, trade_count=5, net_side="yes")
    assert check_flow_confirmation(side="yes", flow=flow)[0]
    assert not check_flow_confirmation(side="no", flow=flow)[0]


def test_apply_confirmation_allows_below_above_strike():
    cfg = BotConfig()
    cfg.gates.trend_gate_enabled = True
    cfg.gates.flow_confirm_enabled = False
    cfg.gates.trend_require_spot_vs_strike = False
    direction = DirectionalEvidence("no", 0.0, 0.1, 0.1, (ModelVote("m", 0.3, 1, 1),))
    edge = TradeSignal(True, "no", 0.4, 0.3, 5.0, 0.1, "ok")
    out, trend_ok, _, _, _ = apply_confirmation_gates(
        edge,
        direction,
        data=_data(spot=66000, mu_5m=-0.0002),
        strike=65000,
        flow=None,
        cfg=cfg,
    )
    assert trend_ok
    assert out.should_trade


def test_daily_pnl_today(tmp_path):
    from kalshi_btc_1hr_bot.trade_journal import TradeJournal, daily_pnl_today

    journal = TradeJournal(path=tmp_path / "t.db")
    journal.record_trade(
        ticker="T", side="yes", contracts=1, entry_price=0.4, mode="PAPER", passed=True
    )
    pnl = daily_pnl_today(journal)
    assert pnl == -0.4
