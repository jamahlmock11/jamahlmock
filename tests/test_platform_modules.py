"""Tests for stale price detection and calibration metrics."""

from kalshi_bot.calibration.metrics import calibration_table
from kalshi_bot.strategy.stale_price_detector import assess_stale_kalshi_price


def test_stale_price_detects_lag():
    result = assess_stale_kalshi_price(
        prev_btc=65000.0,
        curr_btc=65300.0,
        prev_yes_mid=0.50,
        curr_yes_mid=0.51,
        model_prob_yes=0.55,
        strike=65000.0,
        seconds_to_expiry=300.0,
        min_btc_move_pct=0.001,
        min_lag_pp=0.02,
    )
    assert result.btc_move_pct > 0
    assert result.is_stale


def test_calibration_buckets():
    records = [(0.72, True), (0.74, False), (0.71, True), (0.88, True)]
    rows = calibration_table(records, min_trades=2)
    bucket_70 = next(r for r in rows if r["range"] == "70-75%")
    assert bucket_70["n_trades"] == 3
    assert bucket_70["empirical_win_rate"] is not None
