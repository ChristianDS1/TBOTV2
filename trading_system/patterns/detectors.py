"""High-priority Bulkowski detectors for 1–5m (geometry + confirmation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_system.patterns.pivots import find_pivots, is_near


@dataclass
class DetectedPattern:
    name: str
    direction: str  # bullish | bearish
    confidence: float
    breakout_price: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _last_of(pivots: list[tuple[int, float, str]], kind: str, n: int) -> list[tuple[int, float, str]]:
    picked = [p for p in pivots if p[2] == kind]
    return picked[-n:]


def scan_patterns(
    df: pd.DataFrame,
    *,
    pivot_left: int = 3,
    pivot_right: int = 3,
    tol_pct: float = 0.25,
) -> list[DetectedPattern]:
    """Confirmed patterns only (close already beyond the trigger). Prefer precision."""
    if df is None or len(df) < 30:
        return []
    highs = df["high"]
    lows = df["low"]
    close = float(df["close"].iloc[-1])
    pivots = find_pivots(highs, lows, left=pivot_left, right=pivot_right)
    found: list[DetectedPattern] = []
    found += _double_tops_bottoms(pivots, close, tol_pct)
    found += _head_shoulders(pivots, close, tol_pct)
    found += _triple(pivots, close, tol_pct)
    found += _three_peaks_valleys(pivots, close)
    found += _v_reversals(pivots, close, df)
    found += _gaps(df)
    found += _flags_pennants(df, close)
    return found


def best_reversal(patterns: list[DetectedPattern], side: str) -> DetectedPattern | None:
    """Opposing reversal vs an open CALL (want bearish) or PUT (want bullish)."""
    want = "bearish" if side == "call" else "bullish"
    cands = [p for p in patterns if p.direction == want]
    if not cands:
        return None
    return max(cands, key=lambda p: p.confidence)


def _double_tops_bottoms(
    pivots: list[tuple[int, float, str]], close: float, tol_pct: float
) -> list[DetectedPattern]:
    out: list[DetectedPattern] = []
    hs = [p for p in pivots if p[2] == "H"]
    ls = [p for p in pivots if p[2] == "L"]
    if len(hs) >= 2:
        (i1, p1, _), (i2, p2, _) = hs[-2], hs[-1]
        if i2 - i1 >= 5 and is_near(p1, p2, tol_pct):
            valley = min((px for j, px, k in pivots if k == "L" and i1 < j < i2), default=None)
            if valley is not None and close < valley:
                out.append(
                    DetectedPattern(
                        "double_top",
                        "bearish",
                        70.0,
                        valley,
                        {"p1": p1, "p2": p2, "neck": valley},
                    )
                )
    if len(ls) >= 2:
        (i1, p1, _), (i2, p2, _) = ls[-2], ls[-1]
        if i2 - i1 >= 5 and is_near(p1, p2, tol_pct):
            peak = max((px for j, px, k in pivots if k == "H" and i1 < j < i2), default=None)
            if peak is not None and close > peak:
                out.append(
                    DetectedPattern(
                        "double_bottom",
                        "bullish",
                        70.0,
                        peak,
                        {"p1": p1, "p2": p2, "neck": peak},
                    )
                )
    return out


def _head_shoulders(
    pivots: list[tuple[int, float, str]], close: float, tol_pct: float
) -> list[DetectedPattern]:
    out: list[DetectedPattern] = []
    hs = [p for p in pivots if p[2] == "H"]
    ls = [p for p in pivots if p[2] == "L"]
    if len(hs) >= 3:
        lsh, head, rsh = hs[-3], hs[-2], hs[-1]
        if head[1] > lsh[1] and head[1] > rsh[1] and is_near(lsh[1], rsh[1], tol_pct * 1.5):
            necks = [px for j, px, k in pivots if k == "L" and lsh[0] < j < rsh[0]]
            if necks:
                neck = sum(necks) / len(necks)
                if close < neck:
                    out.append(DetectedPattern("hs_top", "bearish", 75.0, neck, {"neck": neck}))
    if len(ls) >= 3:
        lsh, head, rsh = ls[-3], ls[-2], ls[-1]
        if head[1] < lsh[1] and head[1] < rsh[1] and is_near(lsh[1], rsh[1], tol_pct * 1.5):
            necks = [px for j, px, k in pivots if k == "H" and lsh[0] < j < rsh[0]]
            if necks:
                neck = sum(necks) / len(necks)
                if close > neck:
                    out.append(DetectedPattern("hs_bottom", "bullish", 75.0, neck, {"neck": neck}))
    return out


def _triple(
    pivots: list[tuple[int, float, str]], close: float, tol_pct: float
) -> list[DetectedPattern]:
    out: list[DetectedPattern] = []
    hs = [p for p in pivots if p[2] == "H"]
    ls = [p for p in pivots if p[2] == "L"]
    if len(hs) >= 3:
        a, b, c = hs[-3], hs[-2], hs[-1]
        if is_near(a[1], b[1], tol_pct) and is_near(b[1], c[1], tol_pct):
            valley = min((px for j, px, k in pivots if k == "L" and a[0] < j < c[0]), default=None)
            if valley is not None and close < valley:
                out.append(DetectedPattern("triple_top", "bearish", 72.0, valley))
    if len(ls) >= 3:
        a, b, c = ls[-3], ls[-2], ls[-1]
        if is_near(a[1], b[1], tol_pct) and is_near(b[1], c[1], tol_pct):
            peak = max((px for j, px, k in pivots if k == "H" and a[0] < j < c[0]), default=None)
            if peak is not None and close > peak:
                out.append(DetectedPattern("triple_bottom", "bullish", 72.0, peak))
    return out


def _three_peaks_valleys(
    pivots: list[tuple[int, float, str]], close: float
) -> list[DetectedPattern]:
    out: list[DetectedPattern] = []
    hs = [p for p in pivots if p[2] == "H"]
    ls = [p for p in pivots if p[2] == "L"]
    if len(hs) >= 3:
        a, b, c = hs[-3], hs[-2], hs[-1]
        if a[1] > b[1] > c[1]:
            valley = min((px for j, px, k in pivots if k == "L" and a[0] < j < c[0]), default=None)
            if valley is not None and close < valley:
                out.append(DetectedPattern("three_falling_peaks", "bearish", 68.0, valley))
    if len(ls) >= 3:
        a, b, c = ls[-3], ls[-2], ls[-1]
        if a[1] < b[1] < c[1]:
            peak = max((px for j, px, k in pivots if k == "H" and a[0] < j < c[0]), default=None)
            if peak is not None and close > peak:
                out.append(DetectedPattern("three_rising_valleys", "bullish", 68.0, peak))
    return out


def _v_reversals(
    pivots: list[tuple[int, float, str]], close: float, df: pd.DataFrame
) -> list[DetectedPattern]:
    out: list[DetectedPattern] = []
    if len(pivots) < 1:
        return out
    idx, px, kind = pivots[-1]
    # Last pivot near the end and sharp
    if len(df) - idx > 8:
        return out
    if kind == "L":
        after_high = float(df["high"].iloc[idx:].max())
        if close > px and (after_high - px) / max(px, 1e-9) > 0.0015:
            out.append(DetectedPattern("v_bottom", "bullish", 60.0, after_high))
    else:
        after_low = float(df["low"].iloc[idx:].min())
        if close < px and (px - after_low) / max(px, 1e-9) > 0.0015:
            out.append(DetectedPattern("v_top", "bearish", 60.0, after_low))
    return out


def _gaps(df: pd.DataFrame) -> list[DetectedPattern]:
    if len(df) < 3:
        return []
    prev_c = float(df["close"].iloc[-2])
    o = float(df["open"].iloc[-1])
    h, l = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
    if o > prev_c * 1.0008 and l > prev_c:
        return [DetectedPattern("gap_up", "bullish", 55.0, o)]
    if o < prev_c * 0.9992 and h < prev_c:
        return [DetectedPattern("gap_down", "bearish", 55.0, o)]
    return []


def _flags_pennants(df: pd.DataFrame, close: float) -> list[DetectedPattern]:
    """Micro flag: sharp 8–20 bar pole then 5–18 bar counter-consolidation + break."""
    if len(df) < 30:
        return []
    pole = df.iloc[-30:-8]
    cons = df.iloc[-8:]
    pole_move = float(pole["close"].iloc[-1] / pole["close"].iloc[0] - 1)
    cons_range = float(cons["high"].max() - cons["low"].min()) / max(float(cons["close"].mean()), 1e-9)
    if abs(pole_move) < 0.004 or cons_range > 0.006:
        return []
    if pole_move > 0 and close > float(cons["high"].max()):
        return [DetectedPattern("flag_bull", "bullish", 65.0, close)]
    if pole_move < 0 and close < float(cons["low"].min()):
        return [DetectedPattern("flag_bear", "bearish", 65.0, close)]
    return []
