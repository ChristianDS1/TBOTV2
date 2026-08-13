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
    found += _triangles(pivots, close)
    found += _rectangles(pivots, close, tol_pct)
    found += _wedges(pivots, close)
    return found


WEAK_PATTERNS = {"gap_up", "gap_down", "v_top", "v_bottom"}


def measure_target(pat: DetectedPattern, price: float) -> float:
    """Measure-rule target from pattern height (fallback ~25 bps)."""
    height = None
    d = pat.details or {}
    if d.get("p1") is not None and d.get("neck") is not None:
        height = abs(float(d["p1"]) - float(d["neck"]))
    elif pat.breakout_price is not None:
        height = abs(price - float(pat.breakout_price))
    if height is None or height <= 0:
        height = abs(price) * 0.0025
    if pat.direction == "bullish":
        return price + height
    return price - height


def pattern_opposes_htf(pat: DetectedPattern, htf: str) -> bool:
    if htf == "bull" and pat.direction == "bearish":
        return True
    if htf == "bear" and pat.direction == "bullish":
        return True
    return False


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


def _triangles(
    pivots: list[tuple[int, float, str]], close: float
) -> list[DetectedPattern]:
    hs = [p for p in pivots if p[2] == "H"]
    ls = [p for p in pivots if p[2] == "L"]
    if len(hs) < 2 or len(ls) < 2:
        return []
    h1, h2 = hs[-2][1], hs[-1][1]
    l1, l2 = ls[-2][1], ls[-1][1]
    upper_falling = h2 < h1 * 0.9995
    upper_flat = is_near(h1, h2, 0.35)
    lower_rising = l2 > l1 * 1.0005
    lower_flat = is_near(l1, l2, 0.35)
    out: list[DetectedPattern] = []
    if upper_falling and lower_rising:
        if close > h2:
            out.append(DetectedPattern("triangle_sym_up", "bullish", 66.0, h2))
        elif close < l2:
            out.append(DetectedPattern("triangle_sym_down", "bearish", 66.0, l2))
    elif upper_flat and lower_rising:
        cap = max(h1, h2)
        if close > cap:
            out.append(DetectedPattern("triangle_asc_up", "bullish", 68.0, cap))
        elif close < l2:
            out.append(DetectedPattern("triangle_asc_down", "bearish", 62.0, l2))
    elif upper_falling and lower_flat:
        floor = min(l1, l2)
        if close < floor:
            out.append(DetectedPattern("triangle_desc_down", "bearish", 68.0, floor))
        elif close > h2:
            out.append(DetectedPattern("triangle_desc_up", "bullish", 62.0, h2))
    return out


def _rectangles(
    pivots: list[tuple[int, float, str]], close: float, tol_pct: float
) -> list[DetectedPattern]:
    hs = [p for p in pivots if p[2] == "H"][-3:]
    ls = [p for p in pivots if p[2] == "L"][-3:]
    if len(hs) < 2 or len(ls) < 2:
        return []
    if not all(is_near(hs[0][1], h[1], tol_pct * 1.2) for h in hs):
        return []
    if not all(is_near(ls[0][1], l[1], tol_pct * 1.2) for l in ls):
        return []
    cap = sum(h[1] for h in hs) / len(hs)
    floor = sum(l[1] for l in ls) / len(ls)
    if cap <= floor:
        return []
    if close > cap:
        return [DetectedPattern("rectangle_up", "bullish", 64.0, cap, {"neck": cap, "p1": floor})]
    if close < floor:
        return [DetectedPattern("rectangle_down", "bearish", 64.0, floor, {"neck": floor, "p1": cap})]
    return []


def _wedges(
    pivots: list[tuple[int, float, str]], close: float
) -> list[DetectedPattern]:
    hs = [p for p in pivots if p[2] == "H"][-3:]
    ls = [p for p in pivots if p[2] == "L"][-3:]
    if len(hs) < 3 or len(ls) < 3:
        return []
    out: list[DetectedPattern] = []
    rising_h = hs[0][1] < hs[1][1] < hs[2][1]
    rising_l = ls[0][1] < ls[1][1] < ls[2][1]
    falling_h = hs[0][1] > hs[1][1] > hs[2][1]
    falling_l = ls[0][1] > ls[1][1] > ls[2][1]
    if rising_h and rising_l and close < ls[-1][1]:
        out.append(DetectedPattern("rising_wedge", "bearish", 67.0, ls[-1][1]))
    if falling_h and falling_l and close > hs[-1][1]:
        out.append(DetectedPattern("falling_wedge", "bullish", 67.0, hs[-1][1]))
    return []
