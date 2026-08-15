# Pattern validation (2–9 Aug 2026)

## Scope

- Window: `2026-08-02` → `2026-08-09` UTC
- Hist15 train window: `2026-07-27` → `2026-08-14` — **OVERLAPS** (in-sample relative to hist15; not pure OOS)
- Same whitelist / objective as the 20–22 Jul OOS run
- Whitelist: `data\ml\hist15_clean\successful_patterns.json`
- n trades: **584**
- net wins / losses: **50** / **534**
- net WR: **8.56%**

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
| double_top | 178 | 10 | 5.62% | 12.24% |
| bb_mean_reversion | 79 | 12 | 15.19% | 14.56% |
| triangle_desc_down | 79 | 3 | 3.80% | 12.59% |
| v_top | 75 | 6 | 8.00% | 16.00% |
| triangle_desc_up | 62 | 6 | 9.68% | 12.60% |
| v_bottom | 61 | 7 | 11.48% | 13.01% |
| triangle_sym_down | 26 | 2 | 7.69% | 18.33% |
| triangle_sym_up | 24 | 4 | 16.67% | 19.05% |

## Whitelist names

`triangle_sym_up`, `triangle_sym_down`, `v_top`, `bb_mean_reversion`, `v_bottom`, `triangle_desc_up`, `triangle_desc_down`, `double_top`

- Dataset: `data\ml\hist15_validate_aug2_9`
- Created: `2026-08-15T11:18:46.569814+00:00`
