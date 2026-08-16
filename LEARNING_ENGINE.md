# Learning Engine

## Objective

North star: **maximize net equity**. Discovery learns which **indicator bucket** setups win; exploitation applies those toward **+50% equity per UTC day**. Config: `objective.daily_equity_gain_pct` (default 50).

## Allowlisted keys (policy v2)

Learning **never** increments or applies confidence effects on:

- bare `session=*`, `symbol=*`, `chart=*` / Bulkowski names, bare `strategy=*`, `confidence_bucket=*`, bare `exit_reason=*`
- raw floats (rsi values, prices, trade_id, p_win, …)

### ENTRY track (win/loss → confidence boost / **hard-reject**)

Bucket dims only, e.g. `rsi_zone`, `rsi_slope`, `rsi_vs_side`, `bb_pos`, `bb_width_bin`, `macd_*`, `rejection`, `htf_ltf_combo`, `backing_quality`, `edge_ratio_bin`, plus priority compounds `rsi_zone|bb_pos`, `rsi_vs_side|macd_fast_vs_slow`.

Uneconomic edge (`edge_bps < hard_min_edge_multiple × round_trip_cost_bps`): **no ENTRY win/loss count** (cost track / EXIT diagnostics only).

### EXIT track (insight only — never bans entries)

`exit_class`, `mfe_bin`, `mae_bin`, `giveback_bin`, `hold_min_bin`, compound `exit_class|mfe_bin`.

### Cost erosion

Unchanged: insight-only when strategy win but net ≤ 0. Does **not** ban entries.

## Confirmation

- Threshold: `pattern_min_occurrences` (default **10**).
- Confirmed **ENTRY win**: confidence boost only.
- Confirmed **ENTRY loss**: **hard-reject** matching signals (including priority setups).
- Bot may still *enter* via H&S etc.; learning does **not** treat chart/symbol/session as winner/loser.

## Session display

UTC session buckets remain for **insights / reports / priority promote** display. They are **not** pattern_evidence effect keys.

## Deploy cleanup

On first bot start (or `python -m trading_system sanitize-pattern-keys --yes`), wipe non-allowlist `pattern_evidence` + matching `applied_changes`. Trades history kept. Flag: `system_state.pattern_keys_policy_v2`.

`seed-hist15-learning` only ensures the priority JSON — it does **not** seed chart/session evidence.

## Daily report (UTC rollover)

Unchanged sections into `reports/daily/daily_YYYY-MM-DD.md`. Manual: `python -m trading_system report`.

## Exploration / ranking / ML

Exploration budget, strategy ranking, `P(win|features)` retrain path unchanged.
