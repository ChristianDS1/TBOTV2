"""Higher/lower timeframe bias (MACD 13/21/9 on HTF; short-bar turn on LTF)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.features import macd


def macd_htf_bias(
    df: pd.DataFrame,
    *,
    fast: int = 13,
    slow: int = 21,
    signal: int = 9,
) -> str:
    """bull / bear / mixed / unknown from slow MACD histogram."""
    if df is None or len(df) < slow + signal + 3:
        return "unknown"
    _, _, hist = macd(df["close"], fast, slow, signal)
    h = float(hist.iloc[-1])
    p = float(hist.iloc[-2])
    if pd.isna(h) or pd.isna(p):
        return "unknown"
    if h > 0 and h >= p:
        return "bull"
    if h < 0 and h <= p:
        return "bear"
    return "mixed"


def combine_htf_votes(votes: dict[str, str]) -> str:
    vals = [v for v in votes.values() if v in ("bull", "bear")]
    if not vals:
        return "unknown"
    if all(v == "bull" for v in vals):
        return "bull"
    if all(v == "bear" for v in vals):
        return "bear"
    bull_n = sum(1 for v in vals if v == "bull")
    bear_n = sum(1 for v in vals if v == "bear")
    if bull_n >= 2 and bull_n > bear_n:
        return "bull"
    if bear_n >= 2 and bear_n > bull_n:
        return "bear"
    return "mixed"


def ltf_turn(df: pd.DataFrame) -> str | None:
    """Last few sub-minute bars: momentum turning up/down."""
    if df is None or len(df) < 8:
        return None
    c = df["close"].astype(float)
    ema = c.ewm(span=8, adjust=False).mean()
    last, prev = float(c.iloc[-1]), float(c.iloc[-2])
    e = float(ema.iloc[-1])
    if last > prev and last >= e:
        return "turn_up"
    if last < prev and last <= e:
        return "turn_down"
    return None


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.set_index("timestamp").sort_index()
    agg = x.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["open", "close"]).reset_index()
    return agg
