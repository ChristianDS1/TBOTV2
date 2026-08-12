# Changelog

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
