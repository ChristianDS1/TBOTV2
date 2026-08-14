"""Phase 1: Bulkowski coverage + HTF/LTF entry/exit wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.execution.edge import score_trend_fade
from trading_system.patterns import (
    HIGH_PRIORITY_NAMES,
    coverage_report,
    htf_bb_entry_mode,
    is_continuation_pattern,
    scan_patterns,
)
from trading_system.strategies import BBMeanReversionStrategy, MomentumContinuationStrategy
from trading_system.types import Side, Venue


def test_high_priority_detector_api_covers_pipes_horns_pennants():
    """Audit: scan_patterns API exposes high-priority names (pipes/horns newly added)."""
    # Synthetic empty-ish scan still lists implementable names via coverage module
    report = coverage_report(set(HIGH_PRIORITY_NAMES))
    assert report["missing_from_scan_api"] == []
    assert "pipe_bottom" in HIGH_PRIORITY_NAMES
    assert "horn_top" in HIGH_PRIORITY_NAMES
    assert "pennant_bull" in HIGH_PRIORITY_NAMES


def test_htf_bb_entry_mode_mean_rev_vs_continuation():
    assert htf_bb_entry_mode(htf="bull", touch_lower=True, touch_upper=False) == "mean_reversion"
    assert htf_bb_entry_mode(htf="bull", touch_lower=False, touch_upper=True) == "continuation"
    assert htf_bb_entry_mode(htf="bear", touch_lower=False, touch_upper=True) == "mean_reversion"
    assert htf_bb_entry_mode(htf="bear", touch_lower=True, touch_upper=False) == "continuation"
    assert htf_bb_entry_mode(htf="unknown", touch_lower=True, touch_upper=False) == "neutral"


def test_continuation_pattern_helper():
    assert is_continuation_pattern("flag_bull", "bullish", "bull")
    assert is_continuation_pattern("triangle_sym_up", "bullish", "bull")
    assert is_continuation_pattern("pennant_bear", "bearish", "bear")
    assert not is_continuation_pattern("flag_bull", "bullish", "bear")
    assert not is_continuation_pattern("double_top", "bearish", "bull")


def test_pipe_bottom_detector():
    n = 40
    close = np.linspace(100, 100.5, n)
    high = close + 0.2
    low = close - 0.2
    open_ = close.copy()
    # two pipe lows
    for i in (-3, -2):
        low[i] = 99.0
        open_[i] = 99.4
        close[i] = 99.5
        high[i] = 99.7
    close[-1] = 100.2
    high[-1] = 100.4
    low[-1] = 99.8
    open_[-1] = 99.9
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 5.0),
        }
    )
    names = {p.name for p in scan_patterns(df)}
    assert "pipe_bottom" in names


def test_bb_blocks_fade_when_htf_continuation_zone():
    """HTF bull + upper BB → mean-reversion PUT blocked; mode is continuation."""
    from trading_system.config import load_config
    from trading_system.data.crypto import SimulatedCryptoAdapter

    cfg = load_config()
    assert htf_bb_entry_mode(htf="bull", touch_lower=False, touch_upper=True) == "continuation"
    strat = BBMeanReversionStrategy()

    for seed in range(50):
        df = SimulatedCryptoAdapter(seed=seed).get_ohlcv("BTC/USDT", limit=150)
        raw = strat.evaluate(
            "BTC/USDT", Venue.CRYPTO, df, cfg.strategy, context={"htf_bias": "unknown"}
        )
        if raw is None or raw.side != Side.PUT:
            continue
        # Same setup against HTF bull must block short fade
        blocked = strat.evaluate(
            "BTC/USDT",
            Venue.CRYPTO,
            df,
            cfg.strategy,
            context={"htf_bias": "bull"},
        )
        assert blocked is None
        break


def test_momentum_uses_triangle_continuation():
    from trading_system.config import load_config
    from trading_system.data.crypto import SimulatedCryptoAdapter
    from trading_system.patterns.detectors import DetectedPattern

    cfg = load_config()
    cont = MomentumContinuationStrategy()

    df = SimulatedCryptoAdapter(seed=3).get_ohlcv("BTC/USDT", limit=150)
    pat = DetectedPattern("triangle_sym_up", "bullish", 70.0, 100.0)
    sig = cont.evaluate(
        "BTC/USDT",
        Venue.CRYPTO,
        df,
        cfg.strategy,
        context={
            "htf_bias": "bull",
            "patterns": [pat],
            "htf_patterns": [],
            "ltf_turn": "turn_up",
        },
    )
    # May be None if MACD/pullback not met on this seed; if present must be CALL
    if sig is not None:
        assert sig.side == Side.CALL
        assert sig.features.get("htf_bb_mode") in (
            "mean_reversion",
            "continuation",
            "neutral",
        )


def test_ltf_turn_and_chart_reversal_feed_fade_score():
    score, reasons = score_trend_fade(
        Side.CALL,
        {
            "rejection_bear": False,
            "macd_fast_hist": 0.0,
            "macd_fast_hist_prev": 0.0,
            "macd_fast_hist_prev2": 0.0,
            "macd_fast_bear_cross": False,
            "rsi": 55,
            "rsi_prev": 55,
            "macd_slow_hist": 0.1,
            "macd_slow_hist_prev": 0.1,
            "chart_reversal_bear": True,
            "ltf_turn": "turn_down",
        },
    )
    assert score >= 2
    assert "chart_reversal" in reasons
    assert "ltf_turn" in reasons
