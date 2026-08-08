# Kalshi BTC 15-Min Bot — Diagnostic Audit Report

**Date:** 2026-08-08  
**Evaluations collected:** 100 (live API, paper mode)  
**Trades emitted:** 0  

---

## 1. Why the bot currently says NO TRADE

The bot rejected **100%** of evaluations. Three filters account for all primary rejections:

| Rank | Code | Count | % | Meaning |
|------|------|-------|---|---------|
| 1 | `MODEL_CONFLICT` | 82 | 82% | Multi-model spread > 12pp — **hard block** |
| 2 | `FAKE_BREAKOUT` | 11 | 11% | Price action filter (partially false positive on flat feeds) |
| 3 | `EDGE_TOO_SMALL` | 7 | 7% | Net/raw edge below 20¢ floor |

**Root cause:** The bot is not failing because BTC is moving or because markets are missing. It fails because:

1. **The ensemble models disagree with each other** on almost every tick (logistic proxies vs GBM time-series vs options-implied).
2. **The 20¢ strict edge floor** is rarely the binding constraint — only 7% of evals hit it as primary rejection.
3. **Best net edge distribution:** 40 evals had <5¢ edge, 59 had 5–10¢, 1 had 10–15¢, **zero had ≥20¢**.

The model and market prices are **too aligned** for a 20¢ mispricing to exist most of the time, AND when any edge appears, model conflict blocks it first.

---

## 2. Top rejection reasons (100 evaluations)

```
MODEL_CONFLICT:     82 (82.0%)
FAKE_BREAKOUT:      11 (11.0%)
EDGE_TOO_SMALL:      7 ( 7.0%)
TRADES:              0 ( 0.0%)
```

---

## 3. Which filters are overly restrictive

### MODEL_CONFLICT (overly restrictive — architectural)

- Checked as **hard block** in `v6_evaluator.py`
- **Also** penalized in `assess_market_quality()` do-not-trade score
- The 4 "models" are not independent: 3 are logistic transforms of the same 5 features with different scalings. They will **systematically disagree** with the GBM time-series model.
- **Evidence:** 82% primary rejection rate
- **Recommendation:** Refactor ensemble to use genuinely independent signals OR downgrade disagreement from hard block to confidence penalty. **Do not simply raise the 12pp ceiling** without fixing model architecture.

### FAKE_BREAKOUT (partially buggy)

- Triggered when price history is flat (scanner feeds same BRTI spot each cycle)
- **Fixed:** breakout detection now requires ≥5 unique price levels
- **Expected impact:** ~11% fewer false rejections on next collection run

### EDGE_TOO_SMALL at 20¢ (appropriate but not the bottleneck)

- Only 7% primary rejections
- 99% of evals have best net edge < 10¢
- **Lowering to 15¢ would NOT materially increase trades** without fixing model conflict first
- Tier B at 12¢: **0 hypothetical qualifications** in 100 evals

### TIMING / LIQUIDITY / SPREAD

- Not appearing as primary rejections in this sample
- Spread (8¢ max) and liquidity (0.15 min) are reasonable — **do not loosen**

---

## 4. Which filters should NOT be loosened

| Filter | Reason |
|--------|--------|
| Kill switch / daily loss | Hard risk control |
| Max exposure | Account is ~$4 |
| 20¢ edge for **live** | Only 1 eval had >10¢ net edge; lowering live threshold without calibration data increases random trades |
| Spread limit (8¢) | Not binding in sample |
| Manipulation detector | Not binding in sample |
| `live_trading_enabled=false` | Keep until paper tier experiment shows positive EV |

---

## 5. Recommended thresholds (evidence-based)

| Parameter | Current | Recommendation | Rationale |
|-----------|---------|----------------|-----------|
| `strict_min_gap_dollars` (live) | 20¢ | **Keep 20¢** | Not the bottleneck; only 7% rejections |
| `max_model_disagreement_pp` | 12pp | **Fix ensemble first**, then re-test | 82% rejections; raising to 20pp is a band-aid |
| `min_liquidity_score` | 0.15 | Keep | Not binding |
| `max_spread` | 8¢ | Keep | Not binding |
| Tier B (paper only) | 12¢ | Enable paper tracking | 0 qualifications in 100 evals — tiers won't help until model conflict resolved |
| `min_open_seconds` | 30s | Keep | Prevents trading on empty book at market open |

### If model conflict is fixed (paper experiment):

Re-collect 100+ evals. If net edge 10–15¢ setups appear with calibrated win rate > break-even, **then** consider enabling Tier B in paper mode only.

---

## 6. Expected change in trade frequency

| Change | Expected trades/100 evals | Risk |
|--------|--------------------------|------|
| Current | 0 | None |
| Fix fake_breakout bug | +0–2 | Low |
| Fix model ensemble architecture | Unknown — could unlock 10–30% if edges exist | Medium — needs re-calibration |
| Lower edge to 15¢ (without other fixes) | +1–3 estimated | **High** — edges are 5–10¢, not statistically significant |
| Remove MODEL_CONFLICT hard block | +40–80 | **Very high** — trading on conflicting signals |

---

## 7. Expected change in risk

- **Lowering edge threshold alone:** Higher trade count but negative EV (market and model already agree within 10¢).
- **Removing model conflict block:** Trades on low-conviction, internally contradictory signals.
- **Safe path:** Fix ensemble → re-collect diagnostics → paper-test tiers → calibrate predicted vs actual win rates.

---

## 8. Backtest / paper results

```
100 opportunities evaluated
TRADES: 0
NO_TRADE: 100

Edge distribution (best net edge):
  <5¢:    40
  5-10¢:  59
  10-15¢:  1
  15-20¢:  0
  20¢+:    0

Hypothetical tier qualifications (edge-only, ignoring model block):
  A+: 0
  A:  0
  B:  0
```

**Conclusion:** There is no evidence that lowering thresholds would produce legitimate opportunities. The market is efficiently priced relative to the current model on these 100 snapshots.

---

## 9. Files changed

| File | Purpose |
|------|---------|
| `src/kalshi_bot/strategy/rejection_codes.py` | Standardized rejection enums |
| `src/kalshi_bot/strategy/decision_record.py` | Structured audit records |
| `src/kalshi_bot/strategy/v6_evaluator.py` | Full audited evaluation path |
| `src/kalshi_bot/strategy/tiered_edge.py` | A+/A/B tier classification (paper) |
| `src/kalshi_bot/strategy/opportunity_monitor.py` | SQLite diagnostics + breakdown |
| `src/kalshi_bot/strategy/v6_upgrades.py` | Wired audit path; fake_breakout fix |
| `src/kalshi_bot/config.py` | `live_trading_enabled`, tier config |
| `config/default.yaml` | Paper default, `live_trading_enabled: false` |
| `run.py` | `--collect`, `--report`, opportunity monitor UI |
| `docs/DECISION_TREE.md` | Complete decision tree |
| `docs/AUDIT_REPORT.md` | This report |
| `tests/test_diagnostics.py` | Diagnostic system tests |

---

## 10. Tests added

- `tests/test_diagnostics.py` — rejection priority, side evaluation, tiers, audit records
- Updated `tests/test_v6_upgrades.py` for new rejection code format

**43 tests passing.**

---

## 11. Bugs discovered

1. **Fake breakout on flat price feed** — scanner appends same BRTI each cycle; support/resistance logic fired on noise. **Fixed.**
2. **MODEL_CONFLICT double-counting** — hard block + quality score penalty for same signal.
3. **Ensemble not independent** — 3 logistic "models" are correlated; disagreement metric is inflated.
4. **`markets_scanned=0`** when outside timing window — can look like "no markets" but is timing filter (documented in decision tree).
5. **Live trading was enabled without `live_trading_enabled` gate** — **Fixed:** requires both flags now.

---

## Commands

```bash
# Single scan with full audit dashboard
python run.py

# Collect N diagnostic evaluations (paper)
python run.py --collect 100 --interval 2

# View rejection breakdown
python run.py --report
```

## Next steps (recommended order)

1. Refactor `multi_model_ensemble()` to use independent signals (options-implied, GBM, realized-vol digital, order-flow tilt)
2. Re-run `--collect 200` after ensemble fix
3. Compare rejection breakdown — target MODEL_CONFLICT < 30%
4. If edges 10–15¢ appear with agreement, paper-test Tier B for 48+ hours
5. Build calibration table (predicted vs actual) before any live tier changes
