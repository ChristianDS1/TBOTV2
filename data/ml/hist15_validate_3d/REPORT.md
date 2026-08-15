# Pattern validation OOS (3d)

## Scope

- Window: `2026-07-20` → `2026-07-22` UTC
- Hist15 train window (disjoint): `2026-07-27` → `2026-08-14`
- Whitelist: `data\ml\hist15_clean\successful_patterns.json`
- n trades: **289**
- net wins / losses: **51** / **238**
- net WR: **17.65%**

## Live bridge (deferred)

This run does **not** write `pattern_evidence` or wire `models/hist15_clean` into the live engine.
After OOS confirms identification + success, a later bridge will pass learning to the bot so it
knows which patterns to follow for entries / continuation preference / soft-reject — on top of
existing EXIT FIX exits.

## Other PC

```text
git pull origin main
python -m trading_system historical-pattern-validate \
  --start 2026-07-20 --end 2026-07-22 \
  --from-dataset data/ml/hist15_clean
```

Needs `data/ml/hist15_clean/successful_patterns.json` (in git).
`examples.csv` is gitignored — only required to rebuild the whitelist.

## By pattern vs hist15 baseline

| pattern | n | wins | OOS WR | hist15 WR |
|---|---:|---:|---:|---:|
| double_top | 81 | 15 | 18.52% | 12.24% |
| v_bottom | 51 | 17 | 33.33% | 13.01% |
| v_top | 48 | 6 | 12.50% | 16.00% |
| triangle_desc_down | 47 | 6 | 12.77% | 12.59% |
| triangle_desc_up | 20 | 1 | 5.00% | 12.60% |
| bb_mean_reversion | 18 | 3 | 16.67% | 14.56% |
| triangle_sym_up | 14 | 2 | 14.29% | 19.05% |
| triangle_sym_down | 10 | 1 | 10.00% | 18.33% |

## Whitelist names

`triangle_sym_up`, `triangle_sym_down`, `v_top`, `bb_mean_reversion`, `v_bottom`, `triangle_desc_up`, `triangle_desc_down`, `double_top`

- Dataset: `D:\T-BOT V2\data\ml\hist15_validate_3d`
- Created: `2026-08-15T11:04:58.066890+00:00`
