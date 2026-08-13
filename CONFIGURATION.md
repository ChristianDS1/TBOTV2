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
| `strategy.tp_mode` | `trend_fade` (default, no fixed TP) or legacy `band_fraction` / `fixed_bps` |
| `strategy.trend_fade_min_score` | Min fade signals to exit (default 2) |
| `strategy.tp_band_fraction` | Legacy early TP fraction of band width |
| `strategy.tp_min_bps` | Legacy min TP distance / still feeds mid proxy context |
| `strategy.sl_mode` | `margin_pct` (default), `rr_from_tp`, or `band` |
| `strategy.sl_margin_pct` | Max NET loss as % of margin (default 4) |
| `strategy.tp_rr_multiple` | Legacy net TP/SL ratio when `rr_from_tp` |
| `strategy.sl_band_fraction` | Legacy band SL fraction of band width |
| `strategy.sl_min_bps` | Legacy band SL min budget in bps |
| `strategy.sl_include_exit_fees` | Include exit fee in SL net sizing (default true) |
| `objective.daily_equity_gain_pct` | North-star daily equity gain in exploitation (default 50). Discovery does not chase it |
| `objective.chase_target_in_discovery` | If true, treat daily target as active now (default false) |
| `strategy.min_hold_minutes` | Minimum hold before trend-fade exit (default 1) |
| `strategy.preferred_hold_minutes` | Expected early-rejection window (metadata; not a closer) |
| `strategy.max_hold_minutes` | Expected hold metadata — **not** a time_stop (time_stop disabled) |
| `learning.exploration_budget` | Explore fraction |
| `learning.pattern_min_occurrences` | Confirm win/loss patterns only at ≥N (default 20) |
| `learning.win_confidence_boost` | Confidence-only effect for confirmed wins |
| `learning.session_aware` | Scope pattern evidence/effects by UTC session (default true) |
| `learning.session_buckets` | Named UTC hour ranges (asia/europe/us_open/us_afternoon/night) |
| `capital_policy.auto_refill` | Top up paper capital when flat & broke |
| `risk.soft_mode` | No daily kill |
| `forex_session.*` | UTC session window |
| `execution.leverage` | Paper margin multiplier (default 20); fees/PnL on notional |
| `execution.tp_require_positive_net` | `trend_exit` / price-TP only if estimated net &gt; 0 |
| `execution.poll_interval_seconds` | Engine loop (5) |
| `dashboard.refresh_seconds` | UI poll (5) |

## Maintenance (both machines)

`data/trading.db` is local (gitignored). After `git pull`, stop the bot then:

```bash
python -m trading_system reset-loss-learning --dry-run
python -m trading_system reset-loss-learning --yes
```

Wipes loss/cost_erosion patterns, deletes `time_stop` trades, keeps other win patterns. Does **not** rebuild losses from remaining history.
