"""Tests for per-side trade gate evaluation."""

from __future__ import annotations

from kalshi_bot.config import TradeGatesConfig
from kalshi_bot.strategy.trade_gates import GateStatus, evaluate_trade_gates


def test_gates_match_screenshot_scenario():
    """Crowd UP, model 42%, NO has +EV but alignment blocks."""
    result = evaluate_trade_gates(
        model_prob_yes=0.42,
        yes_net_ev=-0.327,
        no_net_ev=0.277,
        yes_ask=0.73,
        yes_bid=0.71,
        no_ask=0.27,
        seconds_to_expiry=7.1 * 60,
        uncertainty_pct=12.2,
        contracts=0,
        gates_cfg=TradeGatesConfig(),
        bucket_overrides={"7_5": {"min_net_edge_dollars": 0.122}},
    )
    assert result.crowd_direction == "UP"
    assert result.crowd_yes_pct == 72.0
    assert result.yes_passes_all is False
    assert result.no_passes_all is False
    assert result.ready_side is None

    by_name = {g.name: g for g in result.gates}
    assert by_name["Time to expiry"].status == GateStatus.PASS
    assert by_name["Uncertainty"].status == GateStatus.FAIL
    assert by_name["BUY YES alignment"].status == GateStatus.FAIL
    assert by_name["BUY YES NET EV"].status == GateStatus.FAIL
    assert by_name["BUY NO alignment"].status == GateStatus.FAIL
    assert by_name["BUY NO NET EV"].status == GateStatus.PASS
    assert by_name["Position size"].status == GateStatus.WARN
    assert "Waiting for a side" in by_name["Position size"].detail


def test_yes_ready_when_all_gates_pass():
    result = evaluate_trade_gates(
        model_prob_yes=0.58,
        yes_net_ev=0.20,
        no_net_ev=-0.05,
        yes_ask=0.40,
        yes_bid=0.38,
        no_ask=0.62,
        seconds_to_expiry=8 * 60,
        uncertainty_pct=6.0,
        contracts=2,
        gates_cfg=TradeGatesConfig(),
    )
    assert result.ready_side == "YES"
    assert result.yes_passes_all is True
    assert result.no_passes_all is False
    position = [g for g in result.gates if g.name == "Position size"][0]
    assert position.status == GateStatus.PASS
    assert "YES x2" in position.detail


def test_to_dict_serializes_for_dashboard():
    result = evaluate_trade_gates(
        model_prob_yes=0.55,
        yes_net_ev=0.15,
        no_net_ev=0.05,
        yes_ask=0.45,
        yes_bid=0.43,
        no_ask=0.57,
        seconds_to_expiry=600,
        uncertainty_pct=5.0,
    )
    data = result.to_dict()
    assert "gates" in data
    assert len(data["gates"]) == 7
    assert data["gates"][0]["status"] in ("pass", "fail", "warn")
