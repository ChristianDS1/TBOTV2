"""Bulkowski high-priority coverage vs short-TF detectors (Phase 1 audit)."""

from __future__ import annotations

# From Bulkowski_Chart_Patterns_Prompt high-priority list (1–5m).
HIGH_PRIORITY_NAMES = frozenset(
    {
        "flag_bull",
        "flag_bear",
        "pennant_bull",
        "pennant_bear",
        "rectangle_up",
        "rectangle_down",
        "triangle_sym_up",
        "triangle_sym_down",
        "triangle_asc_up",
        "triangle_asc_down",
        "triangle_desc_up",
        "triangle_desc_down",
        "double_bottom",
        "double_top",
        "triple_bottom",
        "triple_top",
        "hs_top",
        "hs_bottom",
        "gap_up",
        "gap_down",
        "v_bottom",
        "v_top",
        "pipe_bottom",
        "pipe_top",
        "horn_bottom",
        "horn_top",
        "three_rising_valleys",
        "three_falling_peaks",
        "falling_wedge",
        "rising_wedge",
    }
)

# Names that count as continuation with HTF (breakout in trend direction).
CONTINUATION_PATTERN_PREFIXES = (
    "flag_",
    "pennant_",
    "triangle_",
    "rectangle_",
)


def is_continuation_pattern(name: str, direction: str, htf: str) -> bool:
    if htf == "bull" and direction != "bullish":
        return False
    if htf == "bear" and direction != "bearish":
        return False
    return any(name.startswith(p) for p in CONTINUATION_PATTERN_PREFIXES)


def htf_bb_entry_mode(*, htf: str, touch_lower: bool, touch_upper: bool) -> str:
    """
    How to treat a BB extreme given HTF MACD bias.

    mean_reversion — fade the extreme (against local BB, with HTF)
    continuation — extreme is a pullback into HTF trend (don't fade)
    conflict — fade would fight HTF (block mean-reversion)
    neutral — no clear HTF
    """
    if htf not in ("bull", "bear"):
        return "neutral"
    if htf == "bull":
        if touch_lower:
            return "mean_reversion"  # buy dip with trend
        if touch_upper:
            return "continuation"  # upper touch in uptrend → don't short; momentum
        return "neutral"
    # bear
    if touch_upper:
        return "mean_reversion"
    if touch_lower:
        return "continuation"
    return "neutral"


def coverage_report(detected_names: set[str]) -> dict[str, object]:
    covered = HIGH_PRIORITY_NAMES & detected_names
    missing = HIGH_PRIORITY_NAMES - detected_names
    return {
        "high_priority_total": len(HIGH_PRIORITY_NAMES),
        "implementable_names": len(HIGH_PRIORITY_NAMES),
        "covered_in_scan_api": sorted(covered),
        "missing_from_scan_api": sorted(missing),
    }
