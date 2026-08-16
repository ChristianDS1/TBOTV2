# Learning Engine

## Objective (north-star — code law)

**Maximize net equity growth** (compounding positive PnL over time). Every UTC day should move toward higher net equity. Opportunity cost of not trading when the book is self-blocked by learning bans is a **failure mode**.

- Monitor tracks `objective.daily_equity_gain_pct`.
- Discovery does **not** chase +50%/day with runaway size/leverage (`chase_target_in_discovery: false`).
- Paper leverage remains **50x** (risk/size floors and size_anomaly guards still apply).
- Safety kill-switches (stale data, API failure, size sanity) stay.

## Allowlisted keys (policy v2+)

Learning **never** increments or applies confidence effects on:

- bare `session=*`, `symbol=*`, `chart=*` / Bulkowski names, bare `strategy=*`, `confidence_bucket=*`, bare `exit_reason=*`
- raw floats (rsi values, prices, trade_id, p_win, …)

### ENTRY track

Bucket dims: `rsi_zone`, `rsi_slope`, `rsi_vs_side`, `bb_pos`, `bb_width_bin`, `macd_*`, `rejection`, `htf_ltf_combo`, `backing_quality`, `edge_ratio_bin`, plus compounds `rsi_zone|bb_pos`, `rsi_vs_side|macd_fast_vs_slow`.

Uneconomic edge: **no ENTRY win/loss count** (cost / EXIT only).

### Loss effects (anti-lockup)

| Key type | Effect |
|---|---|
| Ultra-common / 1-dim (e.g. `rejection=none`, `macd_cross=none`, `rsi_vs_side=counter`, `bb_width_bin=squeeze`) | **confidence_penalty only** — never `hard_reject` |
| Compounds | `soft_reject` with **exploration bypass** (default); optional `hard_reject` only if `compound_loss_hard_reject: true` |
| Discovery phase | Learning bans do not freeze fill rate |

Confirmed **WIN compounds** get higher boost (`win_compound_boost`) and entry preference — path toward consistent gains.

### EXIT track / cost_erosion

Insight only — never bans entries. `pattern_min_occurrences` default **10**.

## Idle governor (mandatory)

If **fills in last ~45m == 0** and recent rejects are dominated by `confirmed_loss_pattern:*`:

→ auto-demote those hard_reject / blocking loss rows back to `observing` + `confidence_penalty`.

`trade_rate≈0` + mass hard-reject = policy failure; bans revert automatically.

## Deploy cleanup

- `pattern_keys_policy_v2`: wipe non-allowlist evidence.
- `pattern_keys_policy_v3_no_single_hard_reject`: neutralize blocking hard_rejects (esp. 1-dim defaults).
- Trades history kept.

CLI: `python -m trading_system sanitize-pattern-keys --yes`

## Session display

UTC session buckets for insights/reports only — not pattern_evidence effect keys.
