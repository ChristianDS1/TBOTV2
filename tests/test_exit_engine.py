"""EXIT ENGINE Gen-5 — adaptive MFE / reversal / stale tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from trading_system.config import load_config
from trading_system.database import Database
from trading_system.execution import PaperExecutor
from trading_system.execution.exit_engine import decide_exit, update_excursion_state
from trading_system.portfolio import Portfolio
from trading_system.types import MarketRegime, Position, Side, Signal, TradeStatus, Venue


@pytest.fixture
def cfg():
    c = load_config()
    c.execution.leverage = 20.0
    c.execution.fee_bps = 4.0
    c.execution.slippage_bps = 2.0
    c.strategy.tp_mode = "trend_fade"
    c.strategy.min_hold_minutes = 0
    return c


@pytest.fixture
def tmp_db(tmp_path) -> Database:
    return Database(tmp_path / "exit.db")


def _open_call(ex: PaperExecutor, tmp_db: Database, price: float = 100.0, minutes_ago: float = 5.0):
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=None,
        stop_loss=price * 0.5,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
        features={"sl_budget_cash": 99.0, "sl_mode": "margin_pct"},
    )
    pos = ex.open_trade(sig, 10.0, price)
    past = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    with tmp_db.connection() as conn:
        conn.execute(
            "UPDATE trades SET entry_time=? WHERE id=?",
            (past.isoformat(), pos.id),
        )
    return pos


def _open_put(ex: PaperExecutor, tmp_db: Database, price: float = 100.0, minutes_ago: float = 5.0):
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.PUT,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=None,
        stop_loss=price * 1.5,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
        features={"sl_budget_cash": 99.0, "sl_mode": "margin_pct"},
    )
    pos = ex.open_trade(sig, 10.0, price)
    past = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    with tmp_db.connection() as conn:
        conn.execute(
            "UPDATE trades SET entry_time=? WHERE id=?",
            (past.isoformat(), pos.id),
        )
    return pos


def _neutral_row():
    """No fade components for LONG (CALL)."""
    return {
        "rejection_bear": False,
        "rejection_bull": False,
        "macd_fast_bear_cross": False,
        "macd_fast_bull_cross": False,
        "macd_fast_hist": 0.2,
        "macd_fast_hist_prev": 0.1,
        "macd_fast_hist_prev2": 0.05,
        "rsi": 55,
        "rsi_prev": 52,
        "macd_slow_hist": 0.05,
        "macd_slow_hist_prev": 0.04,
        "htf_bias": "bull",
    }


def _neutral_row_short():
    """No fade components for SHORT (PUT) — hist falling / RSI soft."""
    return {
        "rejection_bear": False,
        "rejection_bull": False,
        "macd_fast_bear_cross": False,
        "macd_fast_bull_cross": False,
        "macd_fast_hist": -0.2,
        "macd_fast_hist_prev": -0.1,
        "macd_fast_hist_prev2": -0.05,
        "rsi": 40,
        "rsi_prev": 42,
        "macd_slow_hist": -0.05,
        "macd_slow_hist_prev": -0.04,
        "htf_bias": "bear",
    }


def _reversal_row_long():
    return {
        "rejection_bear": True,
        "macd_fast_bear_cross": True,
        "macd_fast_hist": -0.2,
        "macd_fast_hist_prev": 0.1,
        "macd_fast_hist_prev2": 0.2,
        "rsi": 48,
        "rsi_prev": 58,
        "macd_slow_hist": 0.0,
        "macd_slow_hist_prev": 0.1,
        "chart_reversal_bear": True,
        "htf_bias": "bull",
    }


def _weak_fade_long():
    """Zero fade components — temporary pullback without deterioration."""
    return {
        "rejection_bear": False,
        "macd_fast_bear_cross": False,
        "macd_fast_hist": 0.15,
        "macd_fast_hist_prev": 0.10,
        "macd_fast_hist_prev2": 0.05,
        "rsi": 56,
        "rsi_prev": 54,
        "macd_slow_hist": 0.06,
        "macd_slow_hist_prev": 0.05,
        "htf_bias": "bull",
    }


def _one_component_fade_long():
    """Exactly one fade component (macd_slow only)."""
    return {
        "rejection_bear": False,
        "macd_fast_bear_cross": False,
        "macd_fast_hist": 0.15,
        "macd_fast_hist_prev": 0.10,
        "macd_fast_hist_prev2": 0.05,
        "rsi": 56,
        "rsi_prev": 54,
        "macd_slow_hist": 0.04,
        "macd_slow_hist_prev": 0.08,
        "htf_bias": "bull",
    }


def test1_long_rising_holds(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, 5)
    closed = ex.manage_open(
        {"BTC/USDT": 100.8},
        True,
        False,
        feature_rows={"BTC/USDT": _neutral_row()},
    )
    assert closed == []
    closed2 = ex.manage_open(
        {"BTC/USDT": 101.2},
        True,
        False,
        feature_rows={"BTC/USDT": _neutral_row()},
    )
    assert closed2 == []


def test2_long_small_pullback_holds(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, 5)
    ex.manage_open({"BTC/USDT": 101.0}, True, False, {"BTC/USDT": _neutral_row()})
    closed = ex.manage_open(
        {"BTC/USDT": 100.85},
        True,
        False,
        feature_rows={"BTC/USDT": _weak_fade_long()},
    )
    assert closed == []


def test3_long_moderate_giveback_weakening_holds(cfg, tmp_db):
    """MFE up, ~20% giveback + 1 component — below protect threshold → HOLD."""
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, 5)
    ex.manage_open({"BTC/USDT": 101.0}, True, False, {"BTC/USDT": _neutral_row()})
    closed = ex.manage_open(
        {"BTC/USDT": 100.80},
        True,
        False,
        feature_rows={"BTC/USDT": _one_component_fade_long()},
    )
    assert closed == []


def test4_long_confirmed_reversal_exits(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, 5)
    ex.manage_open({"BTC/USDT": 101.0}, True, False, {"BTC/USDT": _neutral_row()})
    closed = ex.manage_open(
        {"BTC/USDT": 100.70},
        True,
        False,
        feature_rows={"BTC/USDT": _reversal_row_long()},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason in ("trend_reversal", "profit_protection")


def test5_short_symmetric_reversal_exits(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_put(ex, tmp_db, 100.0, 5)
    ex.manage_open({"BTC/USDT": 99.0}, True, False, {"BTC/USDT": _neutral_row_short()})
    rev = {
        "rejection_bull": True,
        "macd_fast_bull_cross": True,
        "macd_fast_hist": 0.2,
        "macd_fast_hist_prev": -0.1,
        "macd_fast_hist_prev2": -0.2,
        "rsi": 52,
        "rsi_prev": 42,
        "macd_slow_hist": 0.0,
        "macd_slow_hist_prev": -0.1,
        "chart_reversal_bull": True,
    }
    closed = ex.manage_open(
        {"BTC/USDT": 99.40},
        True,
        False,
        feature_rows={"BTC/USDT": rev},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason in ("trend_reversal", "profit_protection")


def test6_exit_after_peak_even_if_pnl_reduced(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, 5)
    ex.manage_open({"BTC/USDT": 101.5}, True, False, {"BTC/USDT": _neutral_row()})
    closed = ex.manage_open(
        {"BTC/USDT": 100.6},
        True,
        False,
        feature_rows={"BTC/USDT": _reversal_row_long()},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason in ("trend_reversal", "profit_protection")


def test7_slightly_positive_enough_fade_exits(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, 5)
    closed = ex.manage_open(
        {"BTC/USDT": 100.35},
        True,
        False,
        feature_rows={"BTC/USDT": _reversal_row_long()},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "trend_reversal"


def test8_negative_weak_fade_holds(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, 5)
    closed = ex.manage_open(
        {"BTC/USDT": 99.90},
        True,
        False,
        feature_rows={"BTC/USDT": _weak_fade_long()},
    )
    assert closed == []


def test9_stale_without_progress_exits(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, minutes_ago=65)
    opens = port.open_positions()
    assert opens
    pos = opens[0]
    feat = json.loads(pos.features_json or "{}")
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    feat["peak_price"] = 100.05
    feat["mfe_pct"] = 0.05
    feat["last_favorable_extreme_ts"] = old
    pos.features_json = json.dumps(feat)
    tmp_db.update_trade(pos)
    closed = ex.manage_open(
        {"BTC/USDT": 100.02},
        True,
        False,
        feature_rows={"BTC/USDT": _neutral_row()},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "stale_position"


def test10_new_extremes_block_stale(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, minutes_ago=70)
    closed = ex.manage_open(
        {"BTC/USDT": 101.2},
        True,
        False,
        feature_rows={"BTC/USDT": _neutral_row()},
    )
    assert closed == []


def test11_stop_loss_always_works(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=None,
        stop_loss=99.5,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
        features={},
    )
    pos = ex.open_trade(sig, 10.0, 100.0)
    past = datetime.now(timezone.utc) - timedelta(minutes=3)
    with tmp_db.connection() as conn:
        conn.execute(
            "UPDATE trades SET entry_time=?, stop_loss=? WHERE id=?",
            (past.isoformat(), 99.5, pos.id),
        )
    closed = ex.manage_open(
        {"BTC/USDT": 99.4},
        True,
        False,
        feature_rows={"BTC/USDT": _reversal_row_long()},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop_loss"


def test12_daily_equity_target_does_not_alter_exit(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    cfg.objective.daily_equity_gain_pct = 50.0
    ex = PaperExecutor(cfg, tmp_db, port)
    _open_call(ex, tmp_db, 100.0, 5)
    ex.manage_open({"BTC/USDT": 101.0}, True, False, {"BTC/USDT": _neutral_row()})
    closed = ex.manage_open(
        {"BTC/USDT": 100.7},
        True,
        False,
        feature_rows={"BTC/USDT": _reversal_row_long()},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason in ("trend_reversal", "profit_protection")


def test_decide_exit_unit_mfe_giveback(cfg):
    pos = Position(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb",
        qty=10,
        entry_price=100,
        entry_mark=100,
        entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
        status=TradeStatus.OPEN,
        leverage=20,
        notional=200,
        features_json="{}",
    )
    feat: dict = {}
    update_excursion_state(feat, pos, 101.0)
    assert feat["mfe_pct"] == pytest.approx(1.0, rel=1e-3)
    update_excursion_state(feat, pos, 100.60)
    assert feat["giveback_pct"] == pytest.approx(0.40, rel=1e-2)
    d = decide_exit(
        pos,
        100.60,
        _one_component_fade_long(),
        feat,
        cfg.exit,
        fee_bps=4,
        slip_bps=2,
        min_hold_minutes=0,
    )
    assert d.reason == "profit_protection"
