"""Simple event-driven backtester with costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_system.config import AppConfig, StrategyConfig
from trading_system.features import build_features
from trading_system.strategies import BBMeanReversionStrategy
from trading_system.types import Side, Venue


@dataclass
class BacktestResult:
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def compute_metrics(pnls: list[float], initial: float = 100.0) -> dict[str, Any]:
    if not pnls:
        return {
            "win_rate": 0,
            "expectancy": 0,
            "profit_factor": 0,
            "sharpe": 0,
            "sortino": 0,
            "max_drawdown": 0,
            "total_trades": 0,
            "net_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "longest_losing_streak": 0,
        }
    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    wr = len(wins) / len(arr)
    exp = float(arr.mean())
    gw = float(wins.sum()) if len(wins) else 0
    gl = float(abs(losses.sum())) if len(losses) else 0
    pf = gw / gl if gl > 0 else (999 if gw > 0 else 0)
    # equity
    eq = initial + np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak == 0, 1, peak)
    max_dd = float(dd.max())
    ret = np.diff(eq, prepend=initial) / initial
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252 * 24 * 60)) if ret.std() > 0 else 0
    downside = ret[ret < 0]
    sortino = (
        float(ret.mean() / downside.std() * np.sqrt(252 * 24 * 60))
        if len(downside) and downside.std() > 0
        else 0
    )
    streak = longest = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return {
        "win_rate": wr,
        "expectancy": exp,
        "profit_factor": pf,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "total_trades": len(pnls),
        "net_pnl": float(arr.sum()),
        "avg_win": float(wins.mean()) if len(wins) else 0,
        "avg_loss": float(losses.mean()) if len(losses) else 0,
        "longest_losing_streak": longest,
    }


class Backtester:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.strategy = BBMeanReversionStrategy()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "BTC/USDT",
        venue: Venue = Venue.CRYPTO,
        trade_size: float = 10.0,
    ) -> BacktestResult:
        scfg = self.cfg.strategy
        fee_bps, slip_bps = self.cfg.execution.costs_for_venue(venue)
        fee_one_way = fee_bps + slip_bps
        leverage = max(1.0, float(self.cfg.execution.leverage or 1.0))
        feat = build_features(
            df,
            bb_period=scfg.bb_period,
            bb_std=scfg.bb_std,
            rsi_period=scfg.rsi_period,
            macd_fast=scfg.macd_fast,
            macd_slow=scfg.macd_slow,
        )

        cash = self.cfg.capital.initial
        equity_curve = [cash]
        trades: list[dict[str, Any]] = []
        open_trade: dict[str, Any] | None = None
        max_hold = scfg.max_hold_minutes

        for i in range(max(scfg.bb_period + 5, 40), len(feat)):
            window = feat.iloc[: i + 1].copy()
            row = feat.iloc[i]
            price = float(row["close"])
            ts = row["timestamp"]

            if open_trade:
                bars_held = i - open_trade["entry_i"]
                exit_reason = None
                sl = open_trade.get("sl")
                if open_trade["side"] == "call" and sl is not None and price <= sl:
                    exit_reason = "stop_loss"
                elif open_trade["side"] == "put" and sl is not None and price >= sl:
                    exit_reason = "stop_loss"
                elif bars_held >= max_hold:
                    exit_reason = "time_stop"
                elif open_trade["side"] == "call" and price >= open_trade["tp"]:
                    exit_reason = "take_profit"
                elif open_trade["side"] == "put" and price <= open_trade["tp"]:
                    exit_reason = "take_profit"

                if exit_reason:
                    direction = 1 if open_trade["side"] == "call" else -1
                    notional = open_trade["notional"]
                    margin = open_trade["margin"]
                    raw = direction * (price - open_trade["entry"]) / open_trade["entry"] * notional
                    exit_fee = notional * fee_one_way / 10_000
                    net = raw - exit_fee
                    cash += margin + net
                    open_trade.update(
                        {
                            "exit": price,
                            "pnl": net,
                            "exit_reason": exit_reason,
                            "exit_i": i,
                        }
                    )
                    trades.append(open_trade)
                    open_trade = None

            if open_trade is None:
                ohlcv = window[
                    [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in window.columns]
                ]
                sig = self.strategy.evaluate(symbol, venue, ohlcv, scfg)
                if sig is not None:
                    margin = trade_size
                    notional = margin * leverage
                    entry_fee = notional * fee_one_way / 10_000
                    if cash >= margin + entry_fee:
                        cash -= margin + entry_fee
                        open_trade = {
                            "side": sig.side.value,
                            "entry": price,
                            "tp": sig.take_profit or price,
                            "sl": sig.stop_loss,
                            "entry_i": i,
                            "confidence": sig.confidence,
                            "symbol": symbol,
                            "margin": margin,
                            "notional": notional,
                            "entry_fee": entry_fee,
                            "leverage": leverage,
                        }

            marked = cash
            if open_trade:
                direction = 1 if open_trade["side"] == "call" else -1
                marked += open_trade["margin"] + direction * (price - open_trade["entry"]) / open_trade[
                    "entry"
                ] * open_trade["notional"]
            equity_curve.append(marked)

        pnls = [t["pnl"] for t in trades]
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=compute_metrics(pnls, self.cfg.capital.initial),
        )
