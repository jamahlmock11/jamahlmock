from datetime import datetime, timedelta, timezone

import pytest

from kalshi_bot.models.black_scholes import BSInputs, digital_call_probability
from kalshi_bot.models.probability import options_implied_prob_above
from kalshi_bot.models.smile import build_smile_from_ibit_chain, synthetic_smile
from kalshi_bot.strategy.fees import quadratic_fee_per_contract
from kalshi_bot.strategy.mispricing import Side, evaluate_market
from kalshi_bot.config import SeriesConfig, SmileConfig
from kalshi_bot.execution.risk import kelly_contracts
from kalshi_bot.backtest.engine import run_backtest


def test_digital_atm_near_half():
    inp = BSInputs(spot=100, strike=100, time_years=1 / 365, vol=0.5)
    p = digital_call_probability(inp)
    # With negative vol-drag, ATM digital call < 0.5
    assert 0.45 < p < 0.50


def test_ibit_to_btc_log_moneyness_preserved():
    spot_ibit = 36.5
    spot_btc = 65000.0
    # 10% OTM IBIT call
    k_ibit = spot_ibit * 1.10
    smile = build_smile_from_ibit_chain(
        strikes_ibit=[spot_ibit * 0.9, spot_ibit, k_ibit, spot_ibit * 1.2],
        ivs=[0.65, 0.55, 0.52, 0.58],
        weights=[1, 1, 1, 1],
        spot_ibit=spot_ibit,
        spot_btc=spot_btc,
        expiry="2026-08-14",
        t_years=7 / 365,
    )
    k_btc = k_ibit * spot_btc / spot_ibit
    assert abs(smile.iv_at_strike(k_btc) - 0.52) < 1e-9


def test_classic_edge_example_triggers_yes():
    """Kalshi 22% vs options ~37.8% → buy YES."""
    spot = 65000.0
    strike = 65200.0  # ~0.3% OTM — realistic for a 1h bucket
    close = datetime.now(timezone.utc) + timedelta(minutes=55)
    target = 0.378
    best_iv, best_diff = 1.0, 1.0
    for iv in [i / 100 for i in range(40, 201)]:
        s = synthetic_smile(spot, atm_iv=iv)
        p = options_implied_prob_above(
            spot_btc=spot, strike_btc=strike, close_time=close, smile=s
        ).probability
        if abs(p - target) < best_diff:
            best_diff, best_iv = abs(p - target), iv
    smile = synthetic_smile(spot, atm_iv=best_iv)
    smile.is_synthetic = False  # controlled fixture, not a live synthetic fallback
    market = {
        "ticker": "KXBTCD-TEST-T65200",
        "series_ticker": "KXBTCD",
        "strike": strike,
        "close_time": close,
        "yes_ask": 0.22,
        "yes_bid": 0.20,
        "no_ask": 0.80,
        "strike_type": "greater",
    }
    mis = evaluate_market(
        market,
        spot=spot,
        smile=smile,
        series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=8.0),
        smile_cfg=SmileConfig(),
    )
    assert mis is not None
    assert mis.side == Side.YES
    assert mis.edge_pp > 10
    assert abs(mis.options_prob - target) < 0.03


def test_fees_quadratic():
    fee = quadratic_fee_per_contract(0.50)
    # 0.07 * 0.5 * 0.5 = 0.0175 → ceil to 0.02
    assert fee == 0.02


def test_kelly_positive_edge():
    from kalshi_bot.strategy.mispricing import Mispricing
    from kalshi_bot.models.probability import ImpliedProbResult, MarketKind

    implied = ImpliedProbResult(
        probability=0.378,
        vol_used=0.55,
        spot=65000,
        strike=66000,
        time_years=1 / 24 / 365,
        log_moneyness=0.015,
        kind=MarketKind.ABOVE_STRIKE,
        smile_expiry="x",
        smile_age_seconds=0,
    )
    mis = Mispricing(
        ticker="T",
        series="KXBTCD",
        side=Side.YES,
        kalshi_price=0.22,
        options_prob=0.378,
        edge_pp=15.8,
        edge_after_fees_pp=14.0,
        strike=66000,
        spot=65000,
        vol=0.55,
        seconds_to_expiry=3000,
        yes_bid=0.20,
        yes_ask=0.22,
        implied=implied,
        reason="test",
    )
    n = kelly_contracts(
        mis,
        bankroll=1000,
        kelly_fraction=0.25,
        max_contracts=500,
        max_notional=250,
        max_loss=75,
    )
    assert n > 0


def test_backtest_counts_pnl():
    spot = 65000.0
    smile = synthetic_smile(spot, atm_iv=0.7)
    close = datetime.now(timezone.utc) + timedelta(minutes=40)
    market = {
        "ticker": "T1",
        "series_ticker": "KXBTCD",
        "strike": 65500.0,
        "close_time": close,
        "yes_ask": 0.15,
        "yes_bid": 0.12,
        "no_ask": 0.88,
        "strike_type": "greater",
    }
    report = run_backtest(
        [{"market": market, "spot": spot, "settled_yes": True}],
        smile=smile,
        series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=5.0),
        smile_cfg=SmileConfig(),
    )
    # May or may not trade depending on implied; just ensure API works
    assert report.total_pnl == sum(t.pnl for t in report.trades)
