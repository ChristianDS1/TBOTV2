"""Tests for historical-ml-run (causal, net label, incremental)."""

from __future__ import annotations

from trading_system.ml import HIST15_GENERATION
from trading_system.ml.audits import audit_examples
from trading_system.ml.hist_model import HistoricalWinModel, vectorize_example
from trading_system.ml.hist_run import run_historical_ml


def test_vectorize_pre_move_only():
    ex = {
        "rsi": 30.0,
        "bb_width": 0.02,
        "macd_fast_hist": 0.1,
        "macd_slow_hist": -0.1,
        "pct_from_mid": -0.5,
        "confidence": 70.0,
        "rejection_bull": True,
        "touch_lower": True,
        "strategy_family": "bb_mean_reversion",
        "htf_bias": "bull",
        "ltf_turn": "turn_up",
        "side": "call",
        "chart_direction": "bullish",
        # leakage fields must not crash vectorizer if present
        "mfe_pct": 9.9,
        "exit_reason": "trend_reversal",
    }
    v = vectorize_example(ex)
    assert len(v) > 10
    assert v[0] == 30.0


def test_hist_model_temporal_fit(tmp_path):
    rows = []
    for i in range(40):
        rows.append(
            {
                "rsi": 40 + i % 20,
                "bb_width": 0.01 + 0.001 * i,
                "macd_fast_hist": (i % 5) * 0.01,
                "macd_slow_hist": (i % 3) * 0.01,
                "pct_from_mid": (i % 7) * 0.01,
                "confidence": 40 + i % 30,
                "rejection_bull": i % 2 == 0,
                "rejection_bear": i % 2 == 1,
                "touch_lower": i % 2 == 0,
                "touch_upper": i % 2 == 1,
                "strategy_family": [
                    "bb_mean_reversion",
                    "momentum_continuation",
                    "bulkowski_pattern",
                ][i % 3],
                "htf_bias": "bull" if i % 2 == 0 else "bear",
                "ltf_turn": "turn_up" if i % 2 == 0 else "turn_down",
                "side": "call" if i % 2 == 0 else "put",
                "chart_direction": "bullish" if i % 2 == 0 else "bearish",
                "label_win": 1 if i % 3 else 0,
            }
        )
    m = HistoricalWinModel(tmp_path / "m")
    out = m.fit_examples(rows)
    assert out["status"] == "ok"
    assert m.model is not None
    p = m.predict_proba(rows[0])
    assert 0.0 < p < 1.0
    assert (tmp_path / "m" / "meta.json").exists()


def test_audit_net_labels():
    rows = [
        {
            "strategy_family": "bb_mean_reversion",
            "symbol": "BTC/USDT",
            "side": "call",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "signal_bar": 10,
            "fill_bar": 10,
            "exit_bar": 20,
            "exit_reason": "trend_reversal",
            "generation": HIST15_GENERATION,
            "label_win": 1,
        }
    ]
    a = audit_examples(rows)
    assert a["ok"]
    assert a["no_lookahead_bar_order_ok"]


def test_historical_ml_run_simulate(tmp_path):
    ds = tmp_path / "ds"
    md = tmp_path / "model"
    manifest = run_historical_ml(
        days=2,
        out_dir=ds,
        model_dir=md,
        simulate=True,
        max_bars=400,
        step=10,
        retrain_every=15,
        ml_min_p_win=0.0,
    )
    assert (ds / "examples.csv").exists()
    assert (ds / "manifest.json").exists()
    assert (ds / "REPORT.md").exists()
    assert manifest["generation"].startswith("HIST15_CLEAN_")
    assert manifest["n_examples"] >= 0
    if manifest["n_examples"] > 0:
        assert manifest["audits"]["no_lookahead_bar_order_ok"]
