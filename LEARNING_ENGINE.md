# Learning Engine

## Objective

North star: **maximize net equity**. Discovery learns which indicator/context setups win; exploitation should apply those toward **+50% equity per UTC day**. Config: `objective.daily_equity_gain_pct` (default 50). Discovery does **not** size or leverage to chase the daily target (`chase_target_in_discovery: false`).

## Phases

discovery → pattern → optimization → exploitation (suggested by trade count; config sets starting phase).

## Session-aware patterns (UTC)

Pattern evidence is scoped by time-of-day bucket so night results do not contaminate morning (and vice versa).

Default buckets (UTC):

| Session | Hours |
|---|---|
| `asia` | 00–07 |
| `europe` | 07–12 |
| `us_open` | 12–16 |
| `us_afternoon` | 16–21 |
| `night` | 21–24 |

Keys look like `session=europe|regime=low_vol`. Confirmed effects (boost / soft-reject) apply **only** in the matching session. Changing session = re-observe until that session reaches ≥20 again; other sessions keep their own memory.

Disable with `learning.session_aware: false`.

## Pattern evidence (≥20)

- Every closed trade increments counters for keys: `regime`, `symbol`, `exit_reason`, `confidence_bucket`, `strategy`, and `regime|exit` (each prefixed with `session=…` when session-aware).
- **Win and loss** both require `pattern_min_occurrences` (default **20**) before confirmation.
- Below 20: observation only — no hypothesis action, no confidence effect.

### Confirmed win pattern

- Effect: **confidence boost only** (`win_confidence_boost`, default +8).
- Does **not** modify entry rules / strategy / the pattern itself.
- Scope: **only the matching contextual key** (e.g. `session=europe|regime=ranging`), not the full 5-indicator entry checklist.

### Confirmed loss pattern

- Effect: confidence penalty and optional **soft-reject** (`loss_soft_reject`).
- Base strategy unchanged.
- Scope: **only that key** — does **not** treat BB+RSI+MACD+rejection as all wrong.
- Bare `session=…`, `…|strategy=…`, and `…|regime=…` keys are excluded from soft-reject (too broad — would halt a whole session or book). Symbol-level keys can still soft-reject.

### Strategy vs costs

Pattern win/loss uses **strategy outcome**:

1. `exit_reason == take_profit` → strategy **win**
2. else `gross_pnl > 0` → strategy **win**
3. else strategy **loss**

If strategy win but **net PnL ≤ 0** (fees/slippage): counted as **cost_erosion**, not as a strategy loss. Confirmed cost patterns are insight-only (no confidence penalty on entry rules).


## Daily report (UTC rollover)

Auto-generated at day change into `reports/daily/daily_YYYY-MM-DD.md`:

1. Aprendizaje del día  
2. Errores identificados (loss patterns)  
3. Oportunidades (win patterns)  
4. Cambios implementados  
5. Progreso del aprendizaje  

Manual: `python -m trading_system report` or `POST /api/report`.

## Reset loss learning (both PCs)

Stop the bot, then:

```bash
python -m trading_system reset-loss-learning --dry-run
python -m trading_system reset-loss-learning --yes
```

Deletes `time_stop` trades and all loss/cost_erosion pattern rows. Keeps win patterns except keys that contain `time_stop`. Does **not** replay remaining trades into loss counters.

## Exploration / ranking / ML

Unchanged: exploration budget, strategy ranking with sample-size uncertainty, `P(win|features)` retrain path.
