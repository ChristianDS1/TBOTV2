"""Session bucket + weekend / FX OTC tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_system.config import ForexSessionConfig, load_config
from trading_system.data import SessionCalendar
from trading_system.database import Database
from trading_system.execution import PaperExecutor
from trading_system.learning.priority import SEED_PRIORITY_NAMES, ensure_priority_file, is_priority_setup
from trading_system.learning.sessions import is_weekend_utc, session_bucket, session_info
from trading_system.portfolio import Portfolio
from trading_system.types import MarketRegime, Side, Signal, Venue


def test_weekend_saturday_sunday():
    sat = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)  # Saturday
    sun = datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)  # Sunday
    mon = datetime(2026, 8, 17, 10, 0, tzinfo=timezone.utc)  # Monday
    assert session_bucket(sat) == "weekend"
    assert session_bucket(sun) == "weekend"
    assert session_bucket(mon) == "europe"
    info = session_info(sat)
    assert info["name"] == "weekend"
    assert any(b["name"] == "weekend" for b in info["buckets"])


def test_priority_seed_names():
    ensure_priority_file()
    for n in SEED_PRIORITY_NAMES:
        assert is_priority_setup(n)
    assert not is_priority_setup("triangle_sym_up")


def test_weekend_crypto_only_pepperstone_fees():
    cfg = load_config()
    assert cfg.execution.weekend_forex_otc is False
    assert cfg.execution.weekend_use_forex_costs is True
    sat = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)
    assert is_weekend_utc(sat)
    assert SessionCalendar(ForexSessionConfig()).is_open(sat) is False
    # Weekend crypto uses Pepperstone FX fee schedule; weekday FX same bps
    assert cfg.execution.costs_for_venue("forex", as_of=sat) == (0.35, 1.0)
    assert cfg.execution.costs_for_venue("crypto", as_of=sat) == (0.35, 1.0)


def test_otc_flag_can_keep_fx_open_when_enabled(tmp_path):
    """If OTC were enabled, forex_session_open=True avoids session_end."""
    cfg = load_config()
    cfg.execution.weekend_forex_otc = True
    cfg.execution.leverage = 20.0
    db = Database(tmp_path / "otc.db")
    port = Portfolio(db, 100)
    ex = PaperExecutor(cfg, db, port)
    sig = Signal(
        symbol="EUR/USD",
        venue=Venue.FOREX,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="otc",
        take_profit=None,
        stop_loss=1.0,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
        features={"sl_budget_cash": 50.0, "sl_mode": "margin_pct"},
    )
    pos = ex.open_trade(sig, 10.0, 1.08)
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    with db.connection() as conn:
        conn.execute(
            "UPDATE trades SET entry_time=? WHERE id=?",
            (past.isoformat(), pos.id),
        )
    closed = ex.manage_open(
        {"EUR/USD": 1.081},
        forex_session_open=True,
        close_fx_at_session_end=True,
        feature_rows={"EUR/USD": {}},
    )
    assert closed == []
    assert len(port.open_positions()) == 1


def test_weekday_session_end_still_closes_fx(tmp_path):
    cfg = load_config()
    cfg.execution.leverage = 20.0
    db = Database(tmp_path / "sess.db")
    port = Portfolio(db, 100)
    ex = PaperExecutor(cfg, db, port)
    sig = Signal(
        symbol="EUR/USD",
        venue=Venue.FOREX,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="fri",
        take_profit=None,
        stop_loss=1.0,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
        features={"sl_budget_cash": 50.0, "sl_mode": "margin_pct"},
    )
    pos = ex.open_trade(sig, 10.0, 1.08)
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    with db.connection() as conn:
        conn.execute(
            "UPDATE trades SET entry_time=? WHERE id=?",
            (past.isoformat(), pos.id),
        )
    closed = ex.manage_open(
        {"EUR/USD": 1.081},
        forex_session_open=False,
        close_fx_at_session_end=True,
        feature_rows={"EUR/USD": {}},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "session_end"
