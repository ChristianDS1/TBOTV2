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

# Classic reversal pattern names (direction encoded in suffix / name).
HARD_REVERSAL_BEARISH = frozenset(
    {
        "double_top",
        "triple_top",
        "hs_top",
        "v_top",
        "pipe_top",
        "horn_top",
        "three_falling_peaks",
        "rising_wedge",
        "gap_down",
    }
)
HARD_REVERSAL_BULLISH = frozenset(
    {
        "double_bottom",
        "triple_bottom",
        "hs_bottom",
        "v_bottom",
        "pipe_bottom",
        "horn_bottom",
        "three_rising_valleys",
        "falling_wedge",
        "gap_up",
    }
)


def is_continuation_pattern(name: str, direction: str, htf: str) -> bool:
    if htf == "bull" and direction != "bullish":
        return False
    if htf == "bear" and direction != "bearish":
        return False
    return any(name.startswith(p) for p in CONTINUATION_PATTERN_PREFIXES)


def is_hard_reversal_against_side(name: str, direction: str, side: str) -> bool:
    """True if pattern is a classic reversal against an open CALL/PUT."""
    side_l = (side.value if hasattr(side, "value") else str(side)).lower()
    name_l = str(name or "").lower()
    dir_l = str(direction or "").lower()
    if side_l in ("call", "long", "buy"):
        if name_l in HARD_REVERSAL_BEARISH:
            return True
        return dir_l == "bearish" and any(
            name_l.startswith(p) for p in ("double_top", "triple_top", "hs_", "v_top", "pipe_top")
        )
    if side_l in ("put", "short", "sell"):
        if name_l in HARD_REVERSAL_BULLISH:
            return True
        return dir_l == "bullish" and any(
            name_l.startswith(p)
            for p in ("double_bottom", "triple_bottom", "hs_", "v_bottom", "pipe_bottom")
        )
    return False


def classify_exit_pattern_context(
    *,
    side: str,
    htf: str,
    row: dict,
) -> str:
    """
    Classify forming structure for exit decisions.

    Returns: hard_reversal | continuation | ambiguous
    """
    side_l = (side.value if hasattr(side, "value") else str(side)).lower()
    htf_l = str(htf or "unknown")

    # Indicator / LTF chart reversal flags against the open side
    if side_l in ("call", "long", "buy") and bool(row.get("chart_reversal_bear")):
        hard = True
    elif side_l in ("put", "short", "sell") and bool(row.get("chart_reversal_bull")):
        hard = True
    else:
        hard = False

    cont = False
    patterns = row.get("active_patterns") or []
    for p in patterns:
        if isinstance(p, dict):
            name = str(p.get("name") or "")
            direction = str(p.get("direction") or "")
        else:
            name = str(getattr(p, "name", "") or "")
            direction = str(getattr(p, "direction", "") or "")
        if not name:
            continue
        if is_hard_reversal_against_side(name, direction, side_l):
            hard = True
        # Continuation must align WITH the trade
        want = "bullish" if side_l in ("call", "long", "buy") else "bearish"
        if direction == want and is_continuation_pattern(name, direction, htf_l):
            cont = True
        elif direction == want and any(
            name.startswith(p) for p in CONTINUATION_PATTERN_PREFIXES
        ):
            # Allow continuation even if HTF unknown/mixed when direction matches trade
            if htf_l in ("unknown", "mixed", ""):
                cont = True

    if hard and not cont:
        return "hard_reversal"
    if cont and not hard:
        return "continuation"
    if hard and cont:
        # Conflict: prefer protecting profit
        return "hard_reversal"
    return "ambiguous"

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
