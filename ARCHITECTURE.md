# Architecture

## Pipeline

Market data (crypto CCXT / forex OHLCV) → features → strategies → ranker / exploration → soft risk → paper execution → SQLite → learning / ML → monitor (5s).

## Modules

| Package | Role |
|---|---|
| `data/` | `MarketAdapter`, `CryptoAdapter`, `ForexAdapter`, `SessionCalendar` |
| `features/` | BB, RSI, dual MACD, rejection, regime |
| `strategies/` | `BBMeanReversionStrategy` (+ registry for more) |
| `risk/` | Soft paper risk + technical kill switch |
| `execution/` | Paper fills with fees/slippage |
| `portfolio/` | Cash, equity, metrics |
| `learning/` | Exploration budget, ranking, insights |
| `models/` | `P(win\|features)` baseline → sklearn/LightGBM |
| `backtesting/` | Historical sim with costs |
| `database/` | SQLite trades, rejected, insights, rankings |
| `dashboard/` | FastAPI + static monitor |
| `engine.py` | Orchestrator loop |

## Venues

- **Crypto**: always tradable when API healthy.
- **Forex**: entries only when `SessionCalendar.is_open()` (Mon–Fri UTC window). Intraday FX can close at session end.

## Safety

- `mode=live` raises at engine init.
- Kill switch on stale data / API failure (configurable).
- No daily-loss halt in paper soft mode (exploration continues).
