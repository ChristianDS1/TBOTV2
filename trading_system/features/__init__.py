"""Feature engineering: BB, RSI, MACD, rejection, regime."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_system.types import MarketRegime


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 10) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # When avg_loss ~ 0 (pure uptrend), RSI → 100
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return out


def macd(
    close: pd.Series, fast: int = 5, slow: int = 8, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def bollinger(
    close: pd.Series, period: int = 20, std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return lower, mid, upper


def rejection_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Lower wick rejection (bullish) and upper wick rejection (bearish)."""
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    bullish = (lower_wick > body * 1.2) & (lower_wick / range_ > 0.4)
    bearish = (upper_wick > body * 1.2) & (upper_wick / range_ > 0.4)
    return bullish.fillna(False), bearish.fillna(False)


def detect_regime(df: pd.DataFrame) -> MarketRegime:
    if len(df) < 30:
        return MarketRegime.UNKNOWN
    close = df["close"]
    ret = close.pct_change().dropna()
    vol = ret.tail(20).std()
    sma_fast = close.tail(10).mean()
    sma_slow = close.tail(30).mean()
    _, mid, upper = bollinger(close)
    band_width = ((upper - mid) / mid).iloc[-1] if mid.iloc[-1] else 0

    if vol > 0.003:
        return MarketRegime.HIGH_VOL
    if vol < 0.0005:
        return MarketRegime.LOW_VOL
    if band_width and band_width < 0.005:
        return MarketRegime.COMPRESSION
    if sma_fast > sma_slow * 1.002:
        return MarketRegime.BULLISH
    if sma_fast < sma_slow * 0.998:
        return MarketRegime.BEARISH
    return MarketRegime.RANGING


def build_features(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 10,
    macd_fast: list[int] | None = None,
    macd_slow: list[int] | None = None,
) -> pd.DataFrame:
    macd_fast = macd_fast or [5, 8, 9]
    macd_slow = macd_slow or [13, 21, 9]
    out = df.copy()
    out["rsi"] = rsi(out["close"], rsi_period)
    lower, mid, upper = bollinger(out["close"], bb_period, bb_std)
    out["bb_lower"] = lower
    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_width"] = (upper - lower) / mid.replace(0, np.nan)

    mf, ms, mh = macd(out["close"], *macd_fast)
    out["macd_fast"], out["macd_fast_sig"], out["macd_fast_hist"] = mf, ms, mh
    sf, ss, sh = macd(out["close"], *macd_slow)
    out["macd_slow"], out["macd_slow_sig"], out["macd_slow_hist"] = sf, ss, sh

    bull_rej, bear_rej = rejection_flags(out)
    out["rejection_bull"] = bull_rej
    out["rejection_bear"] = bear_rej

    out["touch_lower"] = out["low"] <= out["bb_lower"]
    out["touch_upper"] = out["high"] >= out["bb_upper"]
    out["mid_bands"] = (out["close"] > out["bb_lower"]) & (out["close"] < out["bb_upper"])
    out["pct_from_mid"] = (out["close"] - out["bb_mid"]) / out["bb_mid"].replace(0, np.nan)

    # Histogram weakening / cross hints
    out["macd_fast_hist_prev"] = out["macd_fast_hist"].shift(1)
    out["macd_fast_hist_prev2"] = out["macd_fast_hist"].shift(2)
    out["macd_slow_hist_prev"] = out["macd_slow_hist"].shift(1)
    out["rsi_prev"] = out["rsi"].shift(1)
    out["macd_fast_bull_cross"] = (out["macd_fast_hist_prev"] < 0) & (out["macd_fast_hist"] >= 0)
    out["macd_fast_bear_cross"] = (out["macd_fast_hist_prev"] > 0) & (out["macd_fast_hist"] <= 0)
    out["macd_slow_weakening_bear"] = (
        (out["macd_slow_hist"] < 0)
        & (out["macd_slow_hist"] > out["macd_slow_hist_prev"])
    )
    out["macd_slow_weakening_bull"] = (
        (out["macd_slow_hist"] > 0)
        & (out["macd_slow_hist"] < out["macd_slow_hist_prev"])
    )
    return out


def latest_feature_dict(df: pd.DataFrame) -> dict[str, Any]:
    row = df.iloc[-1]
    keys = [
        "close",
        "rsi",
        "rsi_prev",
        "bb_lower",
        "bb_mid",
        "bb_upper",
        "bb_width",
        "macd_fast_hist",
        "macd_fast_hist_prev",
        "macd_fast_hist_prev2",
        "macd_slow_hist",
        "macd_slow_hist_prev",
        "macd_fast_bull_cross",
        "macd_fast_bear_cross",
        "pct_from_mid",
        "touch_lower",
        "touch_upper",
        "rejection_bull",
        "rejection_bear",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        if k not in row.index:
            continue
        v = row[k]
        if isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        elif pd.isna(v):
            out[k] = None
        else:
            out[k] = float(v) if np.issubdtype(type(v), np.number) else v
    out["regime"] = detect_regime(df).value
    return out
