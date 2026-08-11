# Rollback — soft-reject / edge experiment

Checkpoint created before relaxing soft-reject + edge filters (2026-08-11).

## What changed in this experiment

1. Soft-reject **ignores** keys starting with `strategy=` (was blocking the whole bot).
2. If a key is confirmed as **both** win and loss → no soft-reject (ambiguous).
3. `hard_min_edge_multiple`: `0.75` → `0.5`
4. `soft_min_edge_multiple`: `1.25` → `1.15`

Learned pattern counters / trade history are **not** wiped by the code change.

## Fast rollback (config only — preferred)

In `trading_system/config/default.yaml`:

```yaml
learning:
  soft_reject_exclude_key_prefixes: []   # empty = old behavior (strategy= can soft-reject)
execution:
  hard_min_edge_multiple: 0.75
  soft_min_edge_multiple: 1.25
```

Restart the bot.

## Full code rollback (git tag)

```bash
git checkout rollback-pre-softreject-fix
# or reset main to the tag if you want that commit exactly
```

Tag: `rollback-pre-softreject-fix` (= commit `29dabb3` era).

## Database rollback (learning state)

Copies taken before this experiment:

- `data/trading.db.bak-pre-softreject-fix`
- `data/trading.db.bak-pre-softreject-fix-YYYYMMDD-HHMMSS`

**Stop the bot first**, then:

```powershell
Copy-Item "data\trading.db.bak-pre-softreject-fix" "data\trading.db" -Force
```

Restart. This restores patterns/rejects/cash as of the backup moment (you lose trades after the backup).
