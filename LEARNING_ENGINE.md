# Learning Engine

## Phases

discovery → pattern → optimization → exploitation (suggested by trade count; config sets starting phase).

## Pattern evidence (≥20)

- Every closed trade increments counters for keys: `regime`, `symbol`, `exit_reason`, `confidence_bucket`, `strategy`, and `regime|exit`.
- **Win and loss** both require `pattern_min_occurrences` (default **20**) before confirmation.
- Below 20: observation only — no hypothesis action, no confidence effect.

### Confirmed win pattern

- Effect: **confidence boost only** (`win_confidence_boost`, default +8).
- Does **not** modify entry rules / strategy / the pattern itself.
- Scope: **only the matching contextual key** (e.g. `regime=ranging`), not the full 5-indicator entry checklist.

### Confirmed loss pattern

- Effect: confidence penalty and optional **soft-reject** (`loss_soft_reject`).
- Base strategy unchanged.
- Scope: **only that key** — does **not** treat BB+RSI+MACD+rejection as all wrong.

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

## Exploration / ranking / ML

Unchanged: exploration budget, strategy ranking with sample-size uncertainty, `P(win|features)` retrain path.
