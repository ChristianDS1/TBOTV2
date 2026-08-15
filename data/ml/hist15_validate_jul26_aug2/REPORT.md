# Pattern validation `2026-07-26` → `2026-08-02`

## Scope

- Window: `2026-07-26` → `2026-08-02` UTC
- Hist15 train window: `2026-07-27` → `2026-08-14` — **OVERLAPS**
- Whitelist (hist15 successful patterns): `data\ml\hist15_clean\successful_patterns.json`
- ML model: `models\hist15_clean` loaded=True
- ml_min_p_win soft-gate: `0.12`
- Fee edge gate: live `assess_entry_edge` (hard reject / soft penalty)
- Win label: **net_pnl > 0** (fees included; fee-eaten = loss)
- Causal: bars only up to decision time; outcome revealed after EXIT FIX
- n trades: **71**
- net wins / losses: **15** / **56**
- net WR: **21.13%**
- cost_erosion (gross>0 net<=0): **11**

## Process (live-equivalent)

1. Whitelist chart_pattern from hist15 winners (wins>=10 & WR>=12%)
2. Fee-aware edge gate (same as engine)
3. Hist15 LightGBM `p_win` + confidence blend; soft-skip if below ml_min_p_win
4. Paper fill + EXIT FIX; score only on **net** PnL

## Live bridge (deferred)

Still does **not** write `pattern_evidence`. ML used offline for this confirmation only.

## Other PC

```text
git pull origin main
# needs local models/hist15_clean/win_model.joblib (gitignored) or re-run historical-ml-run
python -m trading_system historical-pattern-validate \
  --start 2026-07-26 --end 2026-08-02 \
  --from-dataset data/ml/hist15_clean \
  --model-dir models/hist15_clean --ml-min-p-win 0.12
```

## By pattern vs hist15 baseline

| pattern | n | wins | WR | hist15 WR |
|---|---:|---:|---:|---:|
| double_top | 22 | 5 | 22.73% | 12.24% |
| triangle_desc_down | 18 | 3 | 16.67% | 12.59% |
| bb_mean_reversion | 10 | 3 | 30.00% | 14.56% |
| v_top | 10 | 3 | 30.00% | 16.00% |
| triangle_sym_down | 8 | 1 | 12.50% | 18.33% |
| v_bottom | 3 | 0 | 0.00% | 13.01% |

## Whitelist names

`triangle_sym_up`, `triangle_sym_down`, `v_top`, `bb_mean_reversion`, `v_bottom`, `triangle_desc_up`, `triangle_desc_down`, `double_top`

- Dataset: `data\ml\hist15_validate_jul26_aug2`
- Created: `2026-08-15T12:36:59.387414+00:00`
