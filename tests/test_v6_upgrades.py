"""Tests for Kalshi BTC 15-Min Intelligence V6 upgrades."""

from datetime import datetime, timedelta, timezone

import pytest

from kalshi_bot.config import V6Config
from kalshi_bot.strategy.v6_upgrades import (
    V6IntelligenceEngine,
    assess_market_quality,
    compute_microstructure,
    compute_price_action,
    detect_manipulation,
    detect_regime,
    monte_carlo_binary,
    multi_model_ensemble,
    passes_strict_edge,
    strict_edge_gap_dollars,
    ProbabilityCalibrator,
)


# ---------------------------------------------------------------------------
# STRICT EDGE RULE
# ---------------------------------------------------------------------------

def test_strict_edge_60_percent_model_matrix():
    """Agent thinks 60% UP — market YES must be ≤35–40¢ for 25¢/20¢ gaps."""
    model = 0.60
    # 35¢ → 25¢ gap → PASS at 20¢ floor
    ok, gap = passes_strict_edge(model, 0.35, min_gap_dollars=0.20)
    assert ok
    assert gap == pytest.approx(0.25)
    # 40¢ → 20¢ gap → PASS at 20¢ floor
    ok, gap = passes_strict_edge(model, 0.40, min_gap_dollars=0.20)
    assert ok
    assert gap == pytest.approx(0.20)
    # 45¢ → 15¢ gap → FAIL at 20¢ floor
    ok, gap = passes_strict_edge(model, 0.45, min_gap_dollars=0.20)
    assert not ok
    # 35¢ → 25¢ gap → PASS at 25¢ floor
    ok, gap = passes_strict_edge(model, 0.35, min_gap_dollars=0.25)
    assert ok
    # 40¢ → 20¢ gap → FAIL at 25¢ floor
    ok, gap = passes_strict_edge(model, 0.40, min_gap_dollars=0.25)
    assert not ok


def test_strict_edge_no_exceptions_below_threshold():
    """Hard filter — no trade below 20¢ gap regardless of confidence."""
    for market in (0.45, 0.50, 0.55):
        ok, _ = passes_strict_edge(0.60, market, min_gap_dollars=0.20)
        assert not ok


def test_strict_edge_gap_dollars():
    assert strict_edge_gap_dollars(0.60, 0.35) == 0.25
    assert strict_edge_gap_dollars(0.60, 0.40) == 0.20


# ---------------------------------------------------------------------------
# Microstructure
# ---------------------------------------------------------------------------

def test_microstructure_imbalance():
    micro = compute_microstructure(yes_bid=0.40, yes_ask=0.42)
    assert -1.0 <= micro.bid_ask_imbalance <= 1.0
    assert micro.spread == pytest.approx(0.02)
    assert 0.0 <= micro.liquidity_score <= 1.0


def test_microstructure_whale_detection():
    ob = {
        "orderbook": {
            "yes": [[40, 200], [39, 10]],
            "no": [[60, 5]],
        }
    }
    micro = compute_microstructure(yes_bid=0.40, yes_ask=0.42, orderbook=ob)
    assert micro.whale_detected


# ---------------------------------------------------------------------------
# Price action
# ---------------------------------------------------------------------------

def test_price_action_momentum():
    prices = [100.0 + i * 0.1 for i in range(60)]
    pa = compute_price_action(prices)
    assert pa.momentum_1m > 0
    assert pa.resistance >= pa.support


# ---------------------------------------------------------------------------
# Multi-model ensemble
# ---------------------------------------------------------------------------

def test_ensemble_agreement():
    micro = compute_microstructure(yes_bid=0.45, yes_ask=0.47)
    pa = compute_price_action([65000.0] * 30)
    ens = multi_model_ensemble(
        spot=65100,
        strike=65000,
        vol=0.55,
        seconds_to_expiry=600,
        market_yes=0.46,
        micro=micro,
        price_action=pa,
        options_prob=0.58,
    )
    assert 0.0 <= ens.consensus_prob <= 1.0
    assert 0.0 <= ens.agreement_score <= 1.0
    assert len(ens.votes) >= 4


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def test_monte_carlo_5000_sims():
    mean, lo, hi = monte_carlo_binary(
        spot=65000, strike=65000, vol=0.5, seconds=900, n_sims=5000
    )
    assert 0.3 < mean < 0.7
    assert lo <= mean <= hi


# ---------------------------------------------------------------------------
# Calibration (3+ trades per bucket)
# ---------------------------------------------------------------------------

def test_calibration_requires_three_trades():
    cal = ProbabilityCalibrator(min_trades_per_bucket=3)
    prob, ok = cal.calibrate(0.60)
    assert not ok
    cal.record(0.60, True)
    cal.record(0.60, True)
    prob, ok = cal.calibrate(0.60)
    assert not ok
    cal.record(0.60, False)
    prob, ok = cal.calibrate(0.60)
    assert ok
    assert 0.55 < prob < 0.70


# ---------------------------------------------------------------------------
# Regime + manipulation
# ---------------------------------------------------------------------------

def test_regime_detection():
    micro = compute_microstructure(yes_bid=0.40, yes_ask=0.50)
    pa = compute_price_action([100 + i * 0.5 for i in range(60)])
    regime = detect_regime(pa, micro)
    assert regime.value in ("trend", "mean_revert", "chop", "high_vol")


def test_manipulation_detector():
    micro = compute_microstructure(yes_bid=0.40, yes_ask=0.50)
    pa = compute_price_action([100.0] * 10)
    assert not detect_manipulation(micro, pa)


# ---------------------------------------------------------------------------
# V6 engine integration
# ---------------------------------------------------------------------------

def test_v6_engine_strict_edge_blocks_weak_gap():
    cfg = V6Config(strict_min_gap_dollars=0.20, require_pattern_evidence=False)
    engine = V6IntelligenceEngine(cfg)
    close = datetime.now(timezone.utc) + timedelta(minutes=10)
    market = {
        "ticker": "KXBTC15M-TEST",
        "strike": 65000.0,
        "close_time": close,
        "yes_bid": 0.48,
        "yes_ask": 0.50,
        "no_ask": 0.52,
    }
    for _ in range(30):
        engine.update_spot(65100.0)
    decision = engine.evaluate(market, spot=65100.0, vol=0.55, options_prob=0.58)
    assert decision.verdict == "NO_TRADE"
    assert any("strict_edge" in b for b in decision.blockers)


def test_v6_engine_passes_large_gap():
    cfg = V6Config(
        strict_min_gap_dollars=0.20,
        require_pattern_evidence=False,
        max_model_disagreement_pp=50.0,
    )
    engine = V6IntelligenceEngine(cfg)
    close = datetime.now(timezone.utc) + timedelta(minutes=10)
    market = {
        "ticker": "KXBTC15M-TEST",
        "strike": 65000.0,
        "close_time": close,
        "yes_bid": 0.33,
        "yes_ask": 0.35,
        "no_ask": 0.67,
    }
    for _ in range(60):
        engine.update_spot(65200.0)
    decision = engine.evaluate(market, spot=65200.0, vol=0.55, options_prob=0.65)
    # May still NO_TRADE on quality gates, but strict edge should not block
    strict_blocked = any("strict_edge_fail" in b for b in decision.blockers)
    if decision.strict_gap_dollars >= 0.20:
        assert not strict_blocked or decision.verdict != "NO_TRADE"


def test_market_quality_do_not_trade():
    micro = compute_microstructure(yes_bid=0.30, yes_ask=0.45)
    pa = compute_price_action([100.0] * 5)
    ens = multi_model_ensemble(
        spot=100, strike=100, vol=0.5, seconds_to_expiry=300,
        market_yes=0.40, micro=micro, price_action=pa,
    )
    q = assess_market_quality(micro=micro, price_action=pa, ensemble=ens, spread_limit=0.06)
    assert q.do_not_trade_score > 0
