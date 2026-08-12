# Configuration

- YAML: `trading_system/config/default.yaml`
- Env: `.env` from `.env.example` (`TRADING_MODE`, `INITIAL_CAPITAL`, dashboard host/port, DB path, exchange, forex provider)

## Important keys

| Key | Meaning |
|---|---|
| `mode` | `paper` only for now |
| `symbols.crypto` / `symbols.forex` | Universe |
| `strategy.min_conditions` | Entry threshold (default 3) |
| `strategy.require_rejection_candle` | Rejection wick hard-required |
| `strategy.max_extreme_retrace_pct` | Max distance from extreme toward mid (fraction of band width) |
| `strategy.max_hold_minutes` | Time stop (default 10) |
| `learning.exploration_budget` | Explore fraction |
| `learning.pattern_min_occurrences` | Confirm win/loss patterns only at ≥N (default 20) |
| `learning.win_confidence_boost` | Confidence-only effect for confirmed wins |
| `capital_policy.auto_refill` | Top up paper capital when flat & broke |
| `risk.soft_mode` | No daily kill |
| `forex_session.*` | UTC session window |
| `execution.leverage` | Paper margin multiplier (default 5); fees/PnL on notional |
| `execution.tp_require_positive_net` | TP only if estimated net &gt; 0 |
| `execution.poll_interval_seconds` | Engine loop (5) |
| `dashboard.refresh_seconds` | UI poll (5) |
