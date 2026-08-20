# Strategy Guide — Kalshi KXBTCD 1-Hour BTC Forecasting Bot

## 1. The Market

Every hour, Kalshi lists a binary contract on `KXBTCD`: **"Will BTC's BRTI
settlement average be above the strike at the top of the hour?"**

- Strike K = BRTI at window open (e.g. 10:00:00 UTC)
- Settlement = 60-second average of BRTI prints from 10:59:00 → 11:00:00
- Same settlement mechanism as 15-min — the only difference is the window length
- YES pays $1.00 if settlement > strike; NO pays $1.00 otherwise

Sources: [Kalshi Help Center](https://help.kalshi.com/en/articles/13823838-crypto-markets),
[CF Benchmarks BRTI](https://www.cfbenchmarks.com/data/indices/BRTI)

---

## 2. Why 1-Hour Is Different From 15-Minute

| Dimension | 15-min (KXBTC15M) | 1-hour (KXBTCD) |
|-----------|-------------------|-----------------|
| Window | 900s | 3600s |
| Price moves | Smaller, noisier | Larger, more trend-driven |
| Mean reversion | Weak | Moderate (BTC shows some MR over 1hr) |
| Momentum signal | 90s lookback | 5m/15m/30m multi-timeframe |
| Funding rate | Less relevant | More predictive (hourly sentiment) |
| HFT competition | High | Lower (more retail-friendly) |
| Execution urgency | 2s cycle | 5s cycle (more relaxed) |
| Trades/day | 60-80 | 10-16 |

The longer window means:
- More time for directional drift → momentum matters more
- More uncertainty → higher vol → wider Kalshi spreads → bigger potential edge
- Less latency-critical → more realistic for a retail bot
- Fewer trades per day → each trade matters more → tighter risk management

---

## 3. The 5-Layer Ensemble Model

### Layer 1: GBM Core (same as 15m)
Lognormal probability with the 60-second averaging adjustment:

    P(S_settle > K) = N(d2),  d2 = [ln(S/K) + (mu - 0.5*sigma_eff^2)*T_mid] / (sigma_eff * sqrt(T_mid))

Where `sigma_eff` accounts for the 60-print averaging reducing single-tick noise.

### Layer 2: Multi-Timeframe Momentum
Blends drift estimates from 3 timeframes:

    mu_blend = 0.5 * mu_5min + 0.3 * mu_15min + 0.2 * mu_30min

Longer lookbacks capture trend; shorter ones capture recent acceleration.
Weights are tunable in backtest.

### Layer 3: Funding Rate Signal
BTC perpetual futures funding rate (from Binance) as a sentiment proxy:

- Funding > 0: longs pay shorts → bullish sentiment (but extreme = overcrowded)
- Funding < 0: shorts pay longs → bearish sentiment
- Extreme funding (> 0.05%) → contrarian signal (reversal likely)

The model maps funding to a probability adjustment with a sigmoid, weighted by
time remaining (funding matters more early in the window).

### Layer 4: Mean Reversion
BTC shows moderate mean reversion over 1 hour. If price is extended far from
VWAP, the model pulls probability toward 0.5 (expects a pullback):

    mr_pull = -0.15 * (S - VWAP) / VWAP * 100 * min(1, T)

### Layer 5: Volatility Regime Adjustment
Classifies current vol as low (<35%), medium (35-80%), or high (>80%):

- Low vol → high confidence in directional signal (weight 1.0)
- Medium vol → moderate confidence (weight 0.85)
- High vol → reduced confidence, blend toward 0.5 (weight 0.65)

### Final: Logistic Calibration
All features (18 total) are fed through a logistic regression calibrator trained
on backtest data to produce the final `p_fair`.

---

## 4. Edge Calculation

Same structure as 15m bot — trade only when:

    p_fair - market_price > min_edge (2.5 cents for hourly)

The slightly higher threshold (2.5c vs 2.0c for 15m) accounts for the wider
spreads typical in hourly markets.

---

## 5. Position Sizing

More conservative than 15m:
- 20% Kelly (vs 25% for 15m) — each trade has more at stake
- 4% max bankroll per trade (vs 5%)
- 6% daily loss stop (vs 8%)

---

## 6. Realistic Expectations

**Same warning as 15m: 75-80% win rate is not sustainable.** The longer window
adds signal richness but also adds uncertainty. Professional targets:

- 53-56% win rate with positive EV is excellent
- 10-16 trades/day means ~300-480/month
- Each trade has higher EV potential (wider spreads) but also higher risk

The hourly bot's advantage over 15m: the multi-signal ensemble (funding rate,
multi-TF momentum, mean reversion, vol regime) captures edges that a pure GBM
model misses. The disadvantage: fewer trades means slower feedback and
higher variance per-day.

---

## 7. Backtesting

The synthetic generator includes:
- **Regime-switching volatility** (vol randomly jumps between 30%, 50%, 80%, 120%)
- **Mean reversion** component in the price path
- **Funding rate** simulation with noise
- **VWAP** computation

For real data: KalshiBackTest.com currently serves 15-minute markets. For
hourly backtesting, either aggregate 4 consecutive 15m windows or use Kalshi's
public historical market API.
