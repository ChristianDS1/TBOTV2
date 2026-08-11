# Configuration

- YAML: `trading_system/config/default.yaml`
- Env: `.env` from `.env.example` (`TRADING_MODE`, `INITIAL_CAPITAL`, dashboard host/port, DB path, exchange, forex provider)

## Important keys

| Key | Meaning |
|---|---|
| `mode` | `paper` only for now |
| `symbols.crypto` / `symbols.forex` | Universe |
| `strategy.min_conditions` | Entry threshold (default 3) |
| `learning.exploration_budget` | Explore fraction |
| `learning.pattern_min_occurrences` | Confirm win/loss patterns only at ≥N (default 20) |
| `learning.win_confidence_boost` | Confidence-only effect for confirmed wins |
| `capital_policy.auto_refill` | Top up paper capital when flat & broke |
| `risk.soft_mode` | No daily kill |
| `forex_session.*` | UTC session window |
| `execution.poll_interval_seconds` | Engine loop (5) |
| `dashboard.refresh_seconds` | UI poll (5) |
