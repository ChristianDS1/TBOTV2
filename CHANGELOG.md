# Changelog

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
