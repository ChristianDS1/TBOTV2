"""Chart-pattern scan + multi-timeframe bias."""

from trading_system.patterns.coverage import (
    CONTINUATION_PATTERN_PREFIXES,
    HIGH_PRIORITY_NAMES,
    classify_exit_pattern_context,
    coverage_report,
    htf_bb_entry_mode,
    is_continuation_pattern,
    is_hard_reversal_against_side,
)
from trading_system.patterns.detectors import (
    DetectedPattern,
    WEAK_PATTERNS,
    best_reversal,
    measure_target,
    pattern_opposes_htf,
    scan_patterns,
)
from trading_system.patterns.mtf import combine_htf_votes, ltf_turn, macd_htf_bias, resample_ohlcv
from trading_system.patterns.pivots import find_pivots, is_near

__all__ = [
    "CONTINUATION_PATTERN_PREFIXES",
    "DetectedPattern",
    "HIGH_PRIORITY_NAMES",
    "WEAK_PATTERNS",
    "best_reversal",
    "combine_htf_votes",
    "classify_exit_pattern_context",
    "coverage_report",
    "find_pivots",
    "htf_bb_entry_mode",
    "is_continuation_pattern",
    "is_hard_reversal_against_side",
    "is_near",
    "ltf_turn",
    "macd_htf_bias",
    "measure_target",
    "pattern_opposes_htf",
    "resample_ohlcv",
    "scan_patterns",
]
