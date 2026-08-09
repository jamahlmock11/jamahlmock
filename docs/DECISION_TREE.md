# Kalshi BTC 15-Min V6 — Complete Decision Tree

Every path that can produce `NO_TRADE`, `SKIP`, or abstain.

## Layer 1: Market Discovery (`run.py` → `V6Scanner.scan`)

```
iter_markets(KXBTC15M, status=open)
│
├─ close_time missing ──────────────────────────► SKIP (not counted)
├─ secs < v6.min_seconds_to_expiry (60s) ───────► SKIP → TIMING_RESTRICTION
├─ secs > v6.max_seconds_to_expiry (840s) ──────► SKIP → TIMING_RESTRICTION
├─ no yes_ask AND no yes_bid ───────────────────► SKIP → MISSING_DATA
└─ passes ──────────────────────────────────────► evaluate()
```

**Note:** Scanner filters happen BEFORE evaluation. `markets_scanned=0` means NO_MARKET or all filtered by timing.

## Layer 2: Data Acquisition (`v6_evaluator.evaluate_market_audited`)

```
resolve_spot()
├─ authenticated BRTI ──► cf_benchmark=FRESH
└─ Kraken proxy ────────► cf_benchmark=PROXY

get_orderbook(ticker)
├─ success ─────────────► order_book=FRESH
└─ API failure ─────────► order_book=MISSING (continues with top-of-book)
```

## Layer 3: Model Pipeline

```
multi_model_ensemble() ──► consensus_prob, models_agree, disagreement_pp
monte_carlo_binary() ────► mc_mean (5000 sims)
options_implied_prob_up() ► options_prob (if IBIT smile available)
calibrator.calibrate() ──► adjusted prob (needs ≥3 trades/bucket)
```

## Layer 4: Hard Filters (block ANY trade)

| Filter | Code | Location |
|--------|------|----------|
| Kill switch / daily loss | `KILL_SWITCH` | `RiskControllerV6.allow_trade` |
| Max exposure | `RISK_LIMIT` | `RiskControllerV6.allow_trade` |
| Post-loss cooldown | `COOLDOWN` | `RiskControllerV6.allow_trade` |
| Too close/far from expiry | `TIMING_RESTRICTION` | scanner + evaluator |
| Market too new (<30s) | `TIMING_RESTRICTION` | evaluator `min_open_seconds` |
| No bid/ask | `MISSING_DATA` | scanner + evaluator |
| Model disagreement >12pp | `MODEL_CONFLICT` | `multi_model_ensemble` + hard block |
| Manipulation detected | `MANIPULATION_SUSPECTED` | `detect_manipulation` |

## Layer 5: Soft Quality Filters (contribute to rejection)

| Filter | Code | Threshold |
|--------|------|-----------|
| Spread | `SPREAD_TOO_WIDE` | > 8¢ default |
| Liquidity score | `INSUFFICIENT_LIQUIDITY` | < 0.15 |
| Do-not-trade composite | `QUALITY_SCORE_TOO_HIGH` | score ≥ 0.55 |
| Fake breakout | `FAKE_BREAKOUT` | price action |
| Pattern evidence | `PATTERN_EVIDENCE_INSUFFICIENT` | if `require_pattern_evidence=true` |

## Layer 6: Side Evaluation (BOTH YES and NO)

For each side independently:

```
raw_edge = model_probability - executable_ASK
fee = quadratic_fee(ask)
slippage = estimate_slippage(spread, liquidity)
net_edge = raw_edge - fee - slippage

├─ raw_edge < strict_min_gap (20¢) ──► EDGE_TOO_SMALL
├─ net_edge ≤ 0 ───────────────────► EXPECTED_VALUE_NEGATIVE
└─ passes both ──────────────────────► candidate for trade
```

**Best side** = highest `net_edge` (not direction of BTC move).

## Layer 7: Final Verdict

```
IF hard_blockers: NO_TRADE
ELIF best_side passes edge + net_ev AND NOT quality_soft_block: TRADE_YES/NO
ELIF tiers.enabled_for_live AND tier qualifies: TRADE (paper experiment only)
ELSE: NO_TRADE with primary_rejection = highest-priority code
```

## Redundant / Overlapping Filters (audit finding)

1. **MODEL_CONFLICT** checked in:
   - `assess_market_quality` (adds to do_not_trade_score)
   - `evaluate_market_audited` hard block
   - Legacy `evaluate` blockers list
   → Same signal counted 2–3 times.

2. **EDGE_TOO_SMALL** (20¢) + **QUALITY_SCORE** often both fire when model≈market.

3. **Liquidity score** uses synthetic depth (1+1) when orderbook empty → borderline fail.

## Tier System (paper analysis only by default)

| Tier | Min net edge | Min confidence | Live enabled |
|------|-------------|----------------|--------------|
| A+ | 22¢ | 75% | `tiers.enabled_for_live=false` |
| A | 17¢ | 65% | paper only |
| B | 12¢ | 55% | paper only |

Current live gate: **20¢ raw edge** via `strict_min_gap_dollars`.
