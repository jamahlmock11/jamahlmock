"""Tests for gap-tiered bot action policy.

Canonical matrix at model probability = 60%:

| Market YES | Gap  | Bot action                    |
|------------|------|-------------------------------|
| 35¢        | 25pp | Strong BUY candidate          |
| 40¢        | 20pp | Strong BUY candidate          |
| 45¢        | 15pp | Only if other signals confirm |
| 50¢        | 10pp | No trade                      |
| 55¢        |  5pp | No trade                      |
"""

from kalshi_bot.config import BotActionConfig
from kalshi_bot.strategy.bot_action import (
    BotAction,
    assess_buy_gap,
    classify_buy_gap,
    other_signals_confirm,
    raw_gap_pp,
)


def test_raw_gap_pp_at_60_percent_matrix():
    model = 0.60
    assert raw_gap_pp(model, 0.35) == 25.0
    assert raw_gap_pp(model, 0.40) == 20.0
    assert raw_gap_pp(model, 0.45) == 15.0
    assert raw_gap_pp(model, 0.50) == 10.0
    assert raw_gap_pp(model, 0.55) == 5.0


def test_classify_matrix_strong_buy():
    assert classify_buy_gap(25.0) == BotAction.STRONG_BUY
    assert classify_buy_gap(20.0) == BotAction.STRONG_BUY
    assert classify_buy_gap(20.0).label == "Strong BUY candidate"


def test_classify_matrix_conditional():
    assert classify_buy_gap(15.0) == BotAction.CONDITIONAL
    assert classify_buy_gap(19.9) == BotAction.CONDITIONAL
    assert classify_buy_gap(15.0).label == "Only if other signals confirm"


def test_classify_matrix_no_trade():
    assert classify_buy_gap(10.0) == BotAction.NO_TRADE
    assert classify_buy_gap(5.0) == BotAction.NO_TRADE
    assert classify_buy_gap(14.9) == BotAction.NO_TRADE
    assert classify_buy_gap(10.0).label == "No trade"


def test_assess_buy_gap_full_matrix_at_60():
    model = 0.60
    expected = [
        (0.35, 25.0, BotAction.STRONG_BUY),
        (0.40, 20.0, BotAction.STRONG_BUY),
        (0.45, 15.0, BotAction.CONDITIONAL),
        (0.50, 10.0, BotAction.NO_TRADE),
        (0.55, 5.0, BotAction.NO_TRADE),
    ]
    for market_yes, gap, action in expected:
        result = assess_buy_gap(model, market_yes)
        assert result.gap_pp == gap
        assert result.action == action


def test_other_signals_confirm_gates_conditional():
    cfg = BotActionConfig(
        conditional_min_confidence=0.70,
        conditional_max_disagreement_pp=6.0,
    )
    ok, fails = other_signals_confirm(
        confidence=0.80,
        disagreement_pp=3.0,
        sufficient_evidence=True,
        config=cfg,
    )
    assert ok
    assert fails == ()

    ok, fails = other_signals_confirm(
        confidence=0.60,
        disagreement_pp=3.0,
        sufficient_evidence=True,
        config=cfg,
    )
    assert not ok
    assert any("confidence" in f for f in fails)

    ok, fails = other_signals_confirm(
        confidence=0.80,
        disagreement_pp=9.0,
        sufficient_evidence=True,
        config=cfg,
    )
    assert not ok
    assert any("disagreement" in f for f in fails)


def test_custom_thresholds():
    cfg = BotActionConfig(strong_buy_min_gap_pp=25.0, conditional_min_gap_pp=20.0)
    assert classify_buy_gap(22.0, config=cfg) == BotAction.CONDITIONAL
    assert classify_buy_gap(25.0, config=cfg) == BotAction.STRONG_BUY
    assert classify_buy_gap(19.0, config=cfg) == BotAction.NO_TRADE
