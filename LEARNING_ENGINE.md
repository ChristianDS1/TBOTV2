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

**Does NOT train ENTRY win/loss when:**

- `exit_reason` is limbo (`limbo_timeout` / flat timeout equivalents) — EXIT/cost insights still allowed
- `cost_erosion` (strategy OK / TP but net ≤ 0 by fees) — cost track only; never inflate ENTRY wins
- uneconomic edge at entry

### Exclusive WR labels (policy v4)

For each ENTRY key, `n = wins + losses`:

| Condition | Label | Effect |
|---|---|---|
| `n < pattern_min_occurrences` (10) | observing | none |
| `wr ≥ 0.55` and mean net > 0 (when known) | **winner** | `confidence_boost` only (compounds → `win_compound_boost`) |
| `wr ≤ 0.45` | **loser** | `confidence_penalty` (1-dim) or `soft_reject` (compounds); win side inactive |
| `0.45 < wr < 0.55` | **neutral** | neither boost nor penalty |

`apply_confidence_effects` is **XOR per key** — never boost + penalty on the same key. No Fase 2 equity-reward/bandit.

### Loss effects (anti-lockup)

| Key type | Effect |
|---|---|
| Ultra-common / 1-dim (e.g. `rejection=none`, `macd_cross=none`, `rsi_vs_side=counter`, `bb_width_bin=squeeze`) | **confidence_penalty only** — never `hard_reject` |
| Compounds | `soft_reject` with **exploration bypass** (default); optional `hard_reject` only if `compound_loss_hard_reject: true` |
| Discovery phase | Learning bans do not freeze fill rate |

### EXIT track / cost_erosion

Insight only — never bans entries. `pattern_min_occurrences` default **10**.

## Idle governor (mandatory)

If **fills in last ~45m == 0** and recent rejects are dominated by `confirmed_loss_pattern:*`:

→ auto-demote those hard_reject / blocking loss rows back to `observing` + `confidence_penalty`.

`trade_rate≈0` + mass hard-reject = policy failure; bans revert automatically.

## Deploy cleanup / historical reclassify

- `pattern_keys_policy_v2`: wipe non-allowlist evidence.
- `pattern_keys_policy_v3_no_single_hard_reject`: neutralize blocking hard_rejects (esp. 1-dim defaults).
- `pattern_keys_policy_v4_limbo_wr_exclusive`: one-shot rebuild of `pattern_evidence` + `applied_changes` from all closed trades (limbo/cost skip ENTRY; WR-exclusive labels). **Trades table kept.**
- On engine start: ensure v2 → v3 → v4 (idempotent flags).

CLI:

```text
python -m trading_system rebuild-patterns
python -m trading_system reclassify-pattern-effects   # alias; same rebuild + sets v4 flag
python -m trading_system sanitize-pattern-keys --yes
```

## Session display

UTC session buckets for insights/reports only — not pattern_evidence effect keys.
