# Changelog

## 0.1.28

- Learning keys policy v2: allowlisted ENTRY/EXIT buckets only (no session/symbol/chart/strategy effect keys)
- Confirmed ENTRY losses hard-reject; confirm threshold 10; sanitize wipe on deploy
- hist15 seed priority-file only (no pattern_evidence chart poison)

## 0.1.27

- Capital auto-refill triggers when flat and cash/equity ≤ `capital_policy.refill_below` (default **30**, was ~0)

## 0.1.26

- Cleanup: collapse redundant weekend branch in FX `get_ohlcv`; clarify OTC-off comments (crypto-only weekend)
- Keep `_synthetic` / `SimulatedCrypto` only as fallbacks and offline tools; no PO leftovers

## 0.1.25

- Rollback: removed Pocket Option / synthetic weekend FX chart feeds (`865c40c`, `e068238`)
- Weekend again **crypto only** (`weekend_forex_otc: false`); crypto still uses Pepperstone FX fee schedule
- Weekday FX unchanged (yfinance + Pepperstone); peak-lock exits kept

## 0.1.22

- Peak-profit lock: after `peak_pnl>0`, exit on ≥2 soft clues (MACD fade, RSI rollover/extreme, rejection, hard/chart reversal) even if current `net_est<=0`
- Overrides `continuation_hold` so trades like #65 lock instead of riding giveback to SL
- Never-profit path unchanged (limbo / SL)

## 0.1.21

- Weekend paper **forex OTC** (`weekend_forex_otc`): Sat/Sun entries on all configured FX pairs with Pepperstone fees; crypto weekend entries kept
- Expanded FX universe (NZD/USD, USD/CAD, USD/CHF, EUR/GBP, EUR/JPY, GBP/JPY); synthetic 1m marks on weekend so OTC keeps moving
- OTC weekend treated as open for the paper book (no `session_end` kills); real calendar still closed Sat/Sun

## 0.1.20

- Pepperstone Razor FX fee proxy (`forex_fee_bps` 0.35 / `forex_slippage_bps` 1.0); paper leverage **50x**
- Weekday entries: FX only; weekend entries: crypto only, charged with FX fee schedule (`weekend_use_forex_costs`)
- EXIT FIX: `trend_reversal` only when net>0; flat/loss waits SL or `limbo_timeout` (10m never-profit); classify forming pattern (`hard_reversal` lock / `continuation` hold / ambiguous)
- Engine context row passes `active_patterns` into exit classifier

## 0.1.19

- Paper live bridge: priority patterns (3 confirmations ∩ hist15) obligatory when indicators pass
- Soft discovery outside priority (less hard-edge / soft-reject); promote to priority at ≥90% net WR
- Session `weekend` (Sat–Sun UTC) for FX-closed / crypto book
- `seed-hist15-learning`: seed pattern_evidence from hist15 wins + loss context + priority keys

## 0.1.18

- `historical-pattern-validate`: OOS walk on hist15 successful `chart_pattern` whitelist (default 2026-07-20..22)
- Whitelist artifact `data/ml/hist15_clean/successful_patterns.json`; report under `data/ml/hist15_validate_3d/`
- Extra window report: `data/ml/hist15_validate_aug2_9/` (2026-08-02..09; overlaps hist15)
- ML confirmation run: `data/ml/hist15_validate_jul26_aug2/` — whitelist + hist15 LightGBM + live fee edge; win = net_pnl>0
- No live bridge yet — after OOS confirms, learning will be passed to the bot (pattern_evidence / optional model)
- Other PC: `git pull` + same CLI; needs whitelist JSON + local `models/hist15_clean/win_model.joblib`

## 0.1.17

- `historical-ml-run`: 15d causal walk with EXIT FIX labels, pre-move features, incremental ML
- Soft-gate by p_win (net-win label only); outputs `data/ml/hist15_clean` + `models/hist15_clean`

## 0.1.16

- Bulkowski Phase 1: pipes/horns/pennants detectors; HTF BB mean-rev vs continuation filter
- LTF 15s/30s patterns feed chart-reversal exits; momentum uses triangle/flag/pennant/rectangle with HTF

## 0.1.15

- EXIT ENGINE Gen-5: adaptive exits with MFE/giveback tracking, weakening vs reversal vs stale
- Remove "fade only if net>0" gate that ignored valid reversals after giveback
- New exit reasons: `trend_reversal`, `profit_protection`, `stale_position` (SL unchanged; no fixed early TP)
- Config block `exit:` (stale 60m safety, giveback thresholds) — does not touch entries or +50% objective

## 0.1.14

- Monitor charts: keep entry + stop-loss inside the visible price scale (FX SL was off-screen because candle zoom only covered BB width); draw SL as a clear red price line

## 0.1.13

- `bulkowski_pattern` strategy: enter on confirmed chart patterns (1m or 15m/30m/1h) after HTF MACD agrees and backing indicators line up — BB stays, but entries are no longer BB-only
- Extra detectors: triangles, rectangles, rising/falling wedges
- Pattern/continuation edge uses measure-rule target (not BB mid), so breakouts are not starved by the fee gate
- HTF charts are also scanned for patterns; opposing HTF patterns feed fade exits

## 0.1.12

- Margin stays at `base_trade_size` (10) for crypto and FX; size only scales up after +€50 equity (no more size 9)
- Monitor Trades = closed row count in DB, plus last sqlite id (ids are not reused after deletes)
- HTF confirm 15m/30m/1h (MACD 13,21,9): skip fading into the higher-timeframe trend; add `momentum_continuation`
- LTF 15s/30s (crypto 1s resample) + Bulkowski reversals feed trend-fade so TP can exit near the turn
- FX opens without SL get the same 4% margin stop as crypto; missing SL is patched live

## 0.1.11

- Forex (and any open without SL) gets the same 4% margin stop as crypto; missing SL is patched live

## 0.1.10

- Explicit objective: maximize net equity; +50%/UTC-day is the exploitation north star (discovery does not chase it)
- time_stop removed — exits are fade / SL / liquidation / FX session_end
- `regime=` excluded from hard soft-reject
- `reset-loss-learning` CLI/script: wipe loss evidence + delete time_stop trades (keep other wins); run on each machine after pull

## 0.1.9

- Monitor session banner (current UTC region + hours)
- Daily report: active adjustments context + per-session winrate/TP/SL/patterns/changes

## 0.1.8

- Stop-loss budget is fee-aware (max NET loss); take-profit math unchanged

## 0.1.7

- Session-aware pattern learning (UTC buckets); no cross-session win/loss bleed
- Monitor shows current session

## 0.1.6

- Add tight stop_loss (`sl_min_bps` 10 / `sl_band_fraction` 0.12) to cut large adverse losses
- Take-profit math unchanged (still used for fee-edge entry checks)

## 0.1.5

- Paper defaults: margin `base_trade_size=10`, leverage `20x` (notional €200)

## 0.1.4

- Adaptive time-stop: prefer early window (~3m), extend to 10m only if still progressing toward TP
- Monitor Recent Closed: entry price, exit price, learning label (ganancia / pérdida)

## 0.1.3

- Take-profit no longer uses BB mid; early rejection target (`tp_band_fraction` / `tp_min_bps`) for crypto and forex

## 0.1.2

- Rejection candle hard-required; keep 3/5 otherwise
- Skip entries that retraced too far from BB extreme toward mid (`max_extreme_retrace_pct`)
- `max_hold_minutes` default 10
- Take-profit only when estimated **net PnL > 0** (not ≥ 0)
- Paper leverage model (default 5x): margin=`base_trade_size`, fees/PnL on notional, soft liquidation

## 0.1.1

- Pattern evidence: confirm win/loss only at ≥20 occurrences
- Win confirmed → confidence boost only (strategy unchanged)
- Loss confirmed → confidence penalty / soft-reject
- Automatic UTC daily learning report (5 sections)
- Paper capital auto-refill to €100 when exhausted (no stop)

## 0.1.0

- Initial adaptive paper trading system
- Crypto (CCXT / simulated) + Forex session-gated adapters
- BB mean-reversion strategy, soft risk, learning engine, ML stub
- FastAPI monitor (5s), backtester, pytest suite
