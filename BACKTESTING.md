# Backtesting

```bash
python -m trading_system backtest --bars 500
```

`Backtester` walks OHLCV bars, evaluates `bb_mean_reversion`, applies fee+slippage bps, time stop and TP.

## Metrics

win rate, expectancy, profit factor, Sharpe/Sortino (approx), max drawdown, avg win/loss, longest losing streak, net PnL.

Use for strategy comparison before trusting paper live-data runs.
