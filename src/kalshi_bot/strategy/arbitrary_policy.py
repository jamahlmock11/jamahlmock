"""Arbitrary decision policy — independent YES/NO judgment.

Principles:
- Do not blindly follow the market favorite.
- Fade overpriced favorites (NO TRADE on favorite, trade the other side if underpriced).
- Buy underpriced underdogs when calibrated edge clears the bar.
- Weight edge requirements by time remaining.
- Evaluate both YES and NO every cycle.
- Do not chase after edge decays or price runs away.
- Treat uncalibrated model output with skepticism.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from kalshi_bot.config import ArbitraryPolicyConfig, BotActionConfig
from kalshi_bot.strategy.bot_action import BotAction, GapAssessment, assess_buy_gap, other_signals_confirm
from kalshi_bot.strategy.fees import quadratic_fee_per_contract


class CalibratorLike(Protocol):
    def calibrate(self, prob: float) -> tuple[float, bool]: ...


@dataclass(frozen=True)
class SideArbitraryAssessment:
  side: str
  model_probability: float
  calibrated_probability: float
  market_ask: float | None
  is_favorite: bool
  is_underdog: bool
  raw_gap_pp: float
  net_edge_pp: float
  expected_value: float
  bot_action: BotAction
  gap_assessment: GapAssessment | None
  blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArbitraryVerdict:
  verdict: str  # NO_TRADE | TRADE_YES | TRADE_NO
  chosen_side: str | None
  yes: SideArbitraryAssessment
  no: SideArbitraryAssessment
  calibrated: bool
  time_edge_multiplier: float
  chase_blocked: bool
  reasons: tuple[str, ...] = ()
  blockers: tuple[str, ...] = ()


class EdgeChaseGuard:
  """Block entries when edge has decayed or price chased since last scan."""

  def __init__(self, *, ttl_seconds: float = 120.0) -> None:
    self.ttl_seconds = ttl_seconds
    self._history: dict[str, tuple[float, float, float]] = {}

  def check(
    self,
    ticker: str,
    side: str,
    *,
    gap_pp: float,
    ask: float,
    min_gap_decay_pp: float,
    max_ask_rise: float,
  ) -> tuple[bool, str]:
    key = f"{ticker}:{side}"
    now = time.time()
    prev = self._history.get(key)
    self._history[key] = (gap_pp, ask, now)
    if prev is None:
      return True, "ok"
    prev_gap, prev_ask, prev_ts = prev
    if now - prev_ts > self.ttl_seconds:
      return True, "ok"
    if gap_pp < prev_gap - min_gap_decay_pp:
      return False, f"edge_decayed ({prev_gap:.1f}pp → {gap_pp:.1f}pp)"
    if ask > prev_ask + max_ask_rise:
      return False, f"price_chased ({prev_ask*100:.0f}¢ → {ask*100:.0f}¢)"
    return True, "ok"


def market_yes_mid(yes_bid: float | None, yes_ask: float | None) -> float | None:
  if yes_bid is not None and yes_ask is not None:
    return (yes_bid + yes_ask) / 2.0
  if yes_ask is not None:
    return yes_ask
  if yes_bid is not None:
    return yes_bid
  return None


def detect_favorite(yes_mid: float | None, *, config: ArbitraryPolicyConfig) -> str | None:
  if yes_mid is None:
    return None
  if yes_mid >= config.favorite_threshold:
    return "YES"
  if yes_mid <= config.underdog_threshold:
    return "NO"
  return None


def time_edge_multiplier(
  seconds_to_expiry: float,
  *,
  min_seconds: float,
  max_seconds: float,
  config: ArbitraryPolicyConfig,
) -> float:
  """More time remaining → higher edge bar."""
  span = max_seconds - min_seconds
  if span <= 0:
    return 1.0
  remaining = max(0.0, min(seconds_to_expiry - min_seconds, span))
  return 1.0 + config.time_edge_bonus_max * (remaining / span)


def apply_model_skepticism(
  raw_prob: float,
  *,
  calibrated: bool,
  calibrated_prob: float,
  config: ArbitraryPolicyConfig,
) -> float:
  if calibrated:
    return calibrated_prob
  return (raw_prob - 0.5) * config.uncalibrated_shrink + 0.5


def _conservative_prob(side: str, model_yes: float, yes_lo: float, yes_hi: float) -> float:
  if side == "YES":
    return yes_lo
  return 1.0 - yes_hi


def _assess_side(
  *,
  side: str,
  model_prob: float,
  conservative_prob: float,
  ask: float | None,
  favorite: str | None,
  min_edge_pp: float,
  fee_rate: float,
  fee_multiplier: float,
  bot_action_cfg: BotActionConfig,
  calibrated: bool,
  require_calibration_for_conditional: bool,
  block_favorite_without_edge: bool,
) -> SideArbitraryAssessment:
  is_favorite = favorite == side
  is_underdog = favorite is not None and favorite != side
  blockers: list[str] = []

  if ask is None or not (0 < ask < 1):
    return SideArbitraryAssessment(
      side=side,
      model_probability=model_prob,
      calibrated_probability=model_prob,
      market_ask=ask,
      is_favorite=is_favorite,
      is_underdog=is_underdog,
      raw_gap_pp=0.0,
      net_edge_pp=0.0,
      expected_value=0.0,
      bot_action=BotAction.NO_TRADE,
      gap_assessment=None,
      blockers=("missing executable ask",),
    )

  gap = assess_buy_gap(model_prob, ask, config=bot_action_cfg)
  fee = quadratic_fee_per_contract(ask, fee_rate=fee_rate, fee_multiplier=fee_multiplier)
  net_edge_pp = (conservative_prob - ask - fee) * 100.0
  ev = conservative_prob - ask - fee

  if block_favorite_without_edge and is_favorite and gap.gap_pp < bot_action_cfg.conditional_min_gap_pp:
    blockers.append(
      f"overpriced_favorite: {side} is market favorite at {ask*100:.0f}¢ "
      f"with only {gap.gap_pp:.1f}pp edge"
    )
  if gap.action == BotAction.NO_TRADE:
    blockers.append(f"gap {gap.gap_pp:.1f}pp below {bot_action_cfg.conditional_min_gap_pp:.0f}pp floor")
  if gap.action == BotAction.CONDITIONAL and require_calibration_for_conditional and not calibrated:
    blockers.append("uncalibrated model — conditional entry blocked")
  if net_edge_pp < min_edge_pp:
    blockers.append(
      f"post-fee edge {net_edge_pp:.1f}pp < {min_edge_pp:.1f}pp "
      f"(time-weighted)"
    )

  return SideArbitraryAssessment(
    side=side,
    model_probability=model_prob,
    calibrated_probability=model_prob,
    market_ask=ask,
    is_favorite=is_favorite,
    is_underdog=is_underdog,
    raw_gap_pp=gap.gap_pp,
    net_edge_pp=net_edge_pp,
    expected_value=ev,
    bot_action=gap.action,
    gap_assessment=gap,
    blockers=tuple(blockers),
  )


def evaluate_arbitrary(
  *,
  ticker: str,
  model_prob_yes: float,
  model_prob_yes_lo: float,
  model_prob_yes_hi: float,
  yes_ask: float | None,
  yes_bid: float | None,
  no_ask: float | None,
  seconds_to_expiry: float,
  min_seconds_to_expiry: float,
  max_seconds_to_expiry: float,
  base_min_edge_pp: float,
  fee_rate: float = 0.07,
  fee_multiplier: float = 1.0,
  confidence: float = 0.0,
  disagreement_pp: float = 0.0,
  sufficient_evidence: bool = True,
  calibrator: CalibratorLike | None = None,
  chase_guard: EdgeChaseGuard | None = None,
  bot_action_cfg: BotActionConfig | None = None,
  policy_cfg: ArbitraryPolicyConfig | None = None,
) -> ArbitraryVerdict:
  """Evaluate both sides with Arbitrary policy and return the best independent verdict."""
  policy = policy_cfg or ArbitraryPolicyConfig()
  action_cfg = bot_action_cfg or BotActionConfig()
  calibrator = calibrator

  raw_yes = model_prob_yes
  calibrated_yes, is_calibrated = (
    calibrator.calibrate(raw_yes) if calibrator is not None else (raw_yes, False)
  )
  model_yes = apply_model_skepticism(
    raw_yes,
    calibrated=is_calibrated,
    calibrated_prob=calibrated_yes,
    config=policy,
  )
  model_no = 1.0 - model_yes

  yes_lo = max(0.01, min(0.99, model_prob_yes_lo if is_calibrated else model_yes - policy.uncalibrated_band_pp / 100.0))
  yes_hi = max(0.01, min(0.99, model_prob_yes_hi if is_calibrated else model_yes + policy.uncalibrated_band_pp / 100.0))

  if no_ask is None and yes_bid is not None:
    no_ask = max(0.0, 1.0 - yes_bid)

  favorite = detect_favorite(market_yes_mid(yes_bid, yes_ask), config=policy)
  time_mult = time_edge_multiplier(
    seconds_to_expiry,
    min_seconds=min_seconds_to_expiry,
    max_seconds=max_seconds_to_expiry,
    config=policy,
  )
  min_edge_pp = base_min_edge_pp * time_mult

  yes = _assess_side(
    side="YES",
    model_prob=model_yes,
    conservative_prob=_conservative_prob("YES", model_yes, yes_lo, yes_hi),
    ask=yes_ask,
    favorite=favorite,
    min_edge_pp=min_edge_pp,
    fee_rate=fee_rate,
    fee_multiplier=fee_multiplier,
    bot_action_cfg=action_cfg,
    calibrated=is_calibrated,
    require_calibration_for_conditional=policy.require_calibration_for_conditional,
    block_favorite_without_edge=policy.block_favorite_without_edge,
  )
  no = _assess_side(
    side="NO",
    model_prob=model_no,
    conservative_prob=_conservative_prob("NO", model_yes, yes_lo, yes_hi),
    ask=no_ask,
    favorite=favorite,
    min_edge_pp=min_edge_pp,
    fee_rate=fee_rate,
    fee_multiplier=fee_multiplier,
    bot_action_cfg=action_cfg,
    calibrated=is_calibrated,
    require_calibration_for_conditional=policy.require_calibration_for_conditional,
    block_favorite_without_edge=policy.block_favorite_without_edge,
  )

  candidates: list[tuple[str, SideArbitraryAssessment]] = [("YES", yes), ("NO", no)]
  viable: list[tuple[str, SideArbitraryAssessment]] = []
  blockers: list[str] = []
  reasons: list[str] = []
  chase_blocked = False

  for side_name, assessment in candidates:
    if assessment.market_ask is None:
      continue
    if assessment.blockers:
      continue
    if assessment.bot_action == BotAction.CONDITIONAL:
      ok, fails = other_signals_confirm(
        confidence=confidence,
        disagreement_pp=disagreement_pp,
        sufficient_evidence=sufficient_evidence,
        config=action_cfg,
      )
      if not ok:
        blockers.extend(fails)
        continue
    if chase_guard is not None:
      ok, chase_reason = chase_guard.check(
        ticker,
        side_name,
        gap_pp=assessment.raw_gap_pp,
        ask=assessment.market_ask,
        min_gap_decay_pp=policy.chase_min_gap_decay_pp,
        max_ask_rise=policy.chase_max_ask_rise,
      )
      if not ok:
        chase_blocked = True
        blockers.append(f"{side_name} {chase_reason}")
        continue
    viable.append((side_name, assessment))

  verdict = "NO_TRADE"
  chosen: str | None = None
  if viable:
    side_name, assessment = max(viable, key=lambda item: item[1].expected_value)
    verdict = f"TRADE_{side_name}"
    chosen = side_name
    role = "underdog" if assessment.is_underdog else ("favorite" if assessment.is_favorite else "neutral")
    reasons.append(
      f"Arbitrary {side_name} ({role}): gap={assessment.raw_gap_pp:.1f}pp "
      f"net={assessment.net_edge_pp:.1f}pp EV=${assessment.expected_value:.3f}; "
      f"time_mult={time_mult:.2f}; calibrated={is_calibrated}"
    )
    if assessment.is_underdog:
      reasons.append("underpriced underdog entry")
    if favorite and chosen != favorite:
      reasons.append(f"fading overpriced favorite ({favorite})")
  else:
    if not blockers:
      blockers.append("no side clears Arbitrary edge + calibration gates")
    if favorite:
      fav_assessment = yes if favorite == "YES" else no
      if fav_assessment.raw_gap_pp < action_cfg.conditional_min_gap_pp:
        blockers.append(
          f"market favorite {favorite} overpriced — NO TRADE on favorite"
        )

  return ArbitraryVerdict(
    verdict=verdict,
    chosen_side=chosen,
    yes=yes,
    no=no,
    calibrated=is_calibrated,
    time_edge_multiplier=time_mult,
    chase_blocked=chase_blocked,
    reasons=tuple(reasons),
    blockers=tuple(dict.fromkeys(blockers)),
  )
