"""Pivot helpers for chart-pattern geometry."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

PivotKind = Literal["H", "L"]


def is_near(a: float, b: float, tol_pct: float = 0.25) -> bool:
    if a == 0 and b == 0:
        return True
    ref = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / ref * 100.0 <= float(tol_pct)


def find_pivots(
    highs: pd.Series | np.ndarray,
    lows: pd.Series | np.ndarray,
    *,
    left: int = 3,
    right: int = 3,
) -> list[tuple[int, float, PivotKind]]:
    """Local highs/lows. Right window means the last `right` bars cannot confirm yet."""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    n = len(h)
    out: list[tuple[int, float, PivotKind]] = []
    if n < left + right + 1:
        return out
    last = n - right
    for i in range(left, last):
        w_h = h[i - left : i + right + 1]
        w_l = l[i - left : i + right + 1]
        if h[i] >= np.max(w_h) and h[i] > h[i - 1] and h[i] > h[i + 1]:
            out.append((i, float(h[i]), "H"))
        elif l[i] <= np.min(w_l) and l[i] < l[i - 1] and l[i] < l[i + 1]:
            out.append((i, float(l[i]), "L"))
    return out
