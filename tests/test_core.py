"""Unit and integration tests."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from trading_system.backtesting import Backtester, compute_metrics
from trading_system.config import AppConfig, ForexSessionConfig, load_config
from trading_system.data import SessionCalendar
from trading_system.data.crypto import SimulatedCryptoAdapter
from trading_system.data.forex import ForexAdapter
from trading_system.database import Database
from trading_system.engine import TradingEngine
from trading_system.features import build_features, rsi
from trading_system.learning import LearningEngine
from trading_system.models import WinProbabilityModel
from trading_system.portfolio import Portfolio
from trading_system.risk import RiskManager
from trading_system.strategies import BBMeanReversionStrategy
from trading_system.types import Side, Signal, Venue, MarketRegime, Position, TradeStatus


@pytest.fixture
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture
def tmp_db(tmp_path) -> Database:
    return Database(tmp_path / "test.db")


def test_session_calendar_weekend_closed():
    cal = SessionCalendar(ForexSessionConfig())
    saturday = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)  # Saturday
    monday = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
    friday_late = datetime(2026, 8, 7, 22, 0, tzinfo=timezone.utc)  # Fri 22 UTC after 21 close
    assert cal.is_open(saturday) is False
    assert cal.is_open(monday) is True
    assert cal.is_open(friday_late) is False


def test_rsi_bounds():
    s = pd.Series([float(i) for i in range(1, 50)])
    r = rsi(s, 10)
    assert r.iloc[-1] >= 99  # pure uptrend → RSI ~ 100


def test_features_build():
    adapter = SimulatedCryptoAdapter(seed=1)
    df = adapter.get_ohlcv("BTC/USDT", limit=100)
    feat = build_features(df)
    assert "rsi" in feat.columns
    assert "bb_mid" in feat.columns
    assert "macd_fast_hist" in feat.columns


def test_strategy_returns_signal_or_none(cfg):
    adapter = SimulatedCryptoAdapter(seed=2)
    df = adapter.get_ohlcv("BTC/USDT", limit=150)
    strat = BBMeanReversionStrategy()
    # May or may not signal on random path — just ensure no crash
    sig = strat.evaluate("BTC/USDT", Venue.CRYPTO, df, cfg.strategy)
    assert sig is None or isinstance(sig, Signal)


def test_risk_blocks_live(cfg, tmp_db):
    cfg.mode = "paper"
    risk = RiskManager(cfg)
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        timestamp=datetime.now(timezone.utc),
    )
    # Force live policy check via approve with mode live while cfg paper
    d = risk.approve(sig, 100, [], mode="live")
    assert d.allowed is False


def test_risk_max_positions(cfg):
    risk = RiskManager(cfg)
    sig = Signal(
        symbol="ETH/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        timestamp=datetime.now(timezone.utc),
    )
    opens = [
        Position(
            symbol=f"X{i}",
            venue=Venue.CRYPTO,
            side=Side.CALL,
            strategy="bb_mean_reversion",
            qty=2,
            entry_price=1,
            entry_time=datetime.now(timezone.utc),
        )
        for i in range(cfg.risk.max_simultaneous_positions)
    ]
    d = risk.approve(sig, 100, opens, mode="paper")
    assert d.allowed is False
    assert d.reason == "max_positions"


def test_paper_roundtrip(cfg, tmp_db):
    cfg.database.path = str(tmp_db.path)
    port = Portfolio(tmp_db, 100)
    from trading_system.execution import PaperExecutor

    ex = PaperExecutor(cfg, tmp_db, port)
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=80,
        reason="test",
        take_profit=66000,
        timestamp=datetime.now(timezone.utc),
    )
    pos = ex.open_trade(sig, 2.5, 65000)
    assert pos.id is not None
    assert len(port.open_positions()) == 1
    closed = ex.close_trade(pos, 65500, "take_profit")
    assert closed.status == TradeStatus.CLOSED
    assert closed.pnl is not None


def test_no_duplicate_symbol(cfg):
    risk = RiskManager(cfg)
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        timestamp=datetime.now(timezone.utc),
    )
    opens = [
        Position(
            symbol="BTC/USDT",
            venue=Venue.CRYPTO,
            side=Side.CALL,
            strategy="bb_mean_reversion",
            qty=2,
            entry_price=1,
            entry_time=datetime.now(timezone.utc),
        )
    ]
    d = risk.approve(sig, 100, opens, mode="paper")
    assert d.reason == "already_in_symbol"


def test_forex_not_tradable_weekend(cfg):
    fx = ForexAdapter(cfg.forex_session, provider="synthetic")
    saturday = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    assert fx.is_tradable_now("EUR/USD", saturday) is False


def test_engine_simulate_tick(cfg, tmp_path):
    cfg.database.path = str(tmp_path / "eng.db")
    engine = TradingEngine(cfg, simulate=True)
    snap = engine.tick()
    assert snap.equity > 0
    assert snap.kill_switch is False
    payload = engine.get_monitor_payload()
    assert "snapshot" in payload
    assert "open_trades" in payload


def test_kill_switch_blocks(cfg):
    risk = RiskManager(cfg)
    risk.trip("stale_data:crypto")
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=90,
        reason="test",
        timestamp=datetime.now(timezone.utc),
    )
    d = risk.approve(sig, 100, [], mode="paper")
    assert d.allowed is False
    assert "kill_switch" in d.reason


def test_backtest_runs(cfg):
    adapter = SimulatedCryptoAdapter(seed=99)
    df = adapter.get_ohlcv("BTC/USDT", limit=300)
    result = Backtester(cfg).run(df)
    assert "total_trades" in result.metrics
    assert len(result.equity_curve) > 0


def test_compute_metrics():
    m = compute_metrics([1, -0.5, 1, -0.5, 2])
    assert m["total_trades"] == 5
    assert 0 <= m["win_rate"] <= 1


def test_learning_ranker(cfg, tmp_db):
    le = LearningEngine(cfg.learning, tmp_db)
    trades = [
        Position(
            id=i,
            symbol="BTC/USDT",
            venue=Venue.CRYPTO,
            side=Side.CALL,
            strategy="bb_mean_reversion",
            qty=2,
            entry_price=100,
            entry_time=datetime.now(timezone.utc),
            status=TradeStatus.CLOSED,
            pnl=1.0 if i % 2 == 0 else -0.5,
            confidence=60,
            regime="ranging",
            features_json="{}",
        )
        for i in range(10)
    ]
    for t in trades:
        tmp_db.insert_trade(t)
    closed = tmp_db.get_all_closed()
    # insert_trade leaves status as provided
    stats = le.ranker.update_from_trades(closed if closed else trades)
    assert isinstance(stats, list)


def test_win_model_baseline(tmp_path):
    model = WinProbabilityModel(tmp_path)
    p = model.predict_proba({"rsi": 25}, confidence=70)
    assert 0 < p < 1


def test_live_engine_blocked(cfg, tmp_path):
    cfg.mode = "live"
    cfg.database.path = str(tmp_path / "live.db")
    with pytest.raises(RuntimeError):
        TradingEngine(cfg, simulate=True)


def _closed_pos(
    *,
    pnl: float,
    regime: str = "high_vol",
    exit_reason: str = "time_stop",
    symbol: str = "BTC/USDT",
    confidence: float = 55,
) -> Position:
    return Position(
        symbol=symbol,
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        qty=2.5,
        entry_price=100,
        entry_time=datetime.now(timezone.utc),
        status=TradeStatus.CLOSED,
        exit_time=datetime.now(timezone.utc),
        exit_price=100,
        pnl=pnl,
        confidence=confidence,
        regime=regime,
        exit_reason=exit_reason,
        features_json="{}",
    )


def test_pattern_not_confirmed_below_20(cfg, tmp_db):
    cfg.learning.pattern_min_occurrences = 20
    le = LearningEngine(cfg.learning, tmp_db)
    for _ in range(19):
        pos = _closed_pos(pnl=-0.5)
        pos.id = tmp_db.insert_trade(pos)
        newly = le.on_trade_closed(pos)
        assert newly == []
    patterns = tmp_db.get_patterns(direction="loss", status="confirmed")
    assert patterns == []
    observing = tmp_db.get_patterns(direction="loss", status="observing")
    assert any(p["pattern_key"] == "regime=high_vol" and p["count"] == 19 for p in observing)


def test_win_pattern_confirms_at_20_boost_only(cfg, tmp_db):
    cfg.learning.pattern_min_occurrences = 20
    cfg.learning.win_confidence_boost = 8.0
    le = LearningEngine(cfg.learning, tmp_db)
    for i in range(20):
        pos = _closed_pos(pnl=0.5, regime="ranging")
        pos.id = tmp_db.insert_trade(pos)
        newly = le.on_trade_closed(pos)
        if i < 19:
            assert newly == []
        else:
            assert any(n["direction"] == "win" for n in newly)
    confirmed = tmp_db.get_patterns(direction="win", status="confirmed")
    assert any(p["pattern_key"] == "regime=ranging" for p in confirmed)
    changes = tmp_db.applied_changes_on_day(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    assert any(c["action"] == "confidence_boost" for c in changes)
    assert all(c["action"] != "rewrite_strategy" for c in changes)

    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=50,
        reason="test",
        regime=MarketRegime.RANGING,
        timestamp=datetime.now(timezone.utc),
    )
    adjusted, reject = le.apply_confidence_effects(sig)
    assert reject is None
    assert adjusted.confidence == pytest.approx(58.0)


def test_loss_pattern_soft_reject_at_20(cfg, tmp_db):
    cfg.learning.pattern_min_occurrences = 20
    cfg.learning.loss_soft_reject = True
    le = LearningEngine(cfg.learning, tmp_db)
    for _ in range(20):
        pos = _closed_pos(pnl=-0.4, regime="breakout")
        pos.id = tmp_db.insert_trade(pos)
        le.on_trade_closed(pos)
    sig = Signal(
        symbol="ETH/USDT",
        venue=Venue.CRYPTO,
        side=Side.PUT,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        regime=MarketRegime.BREAKOUT,
        timestamp=datetime.now(timezone.utc),
    )
    adjusted, reject = le.apply_confidence_effects(sig)
    assert reject is not None
    assert "confirmed_loss_pattern" in reject
    assert adjusted.confidence < 70


def test_capital_auto_refill(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    port.cash = 0.5
    tmp_db.set_state("cash", "0.5")
    did = port.maybe_refill(
        auto_refill=True,
        refill_to=100.0,
        min_trade_size=2.5,
    )
    assert did is True
    assert port.cash == 100.0
    assert port.capital_resets() == 1


def test_capital_no_refill_with_open_position(cfg, tmp_db):
    from trading_system.execution import PaperExecutor

    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=80,
        reason="test",
        take_profit=66000,
        timestamp=datetime.now(timezone.utc),
    )
    ex.open_trade(sig, 2.5, 65000)
    port.cash = 0.1
    tmp_db.set_state("cash", "0.1")
    did = port.maybe_refill(auto_refill=True, refill_to=100.0, min_trade_size=2.5)
    assert did is False


def test_daily_report_sections(cfg, tmp_db, tmp_path):
    from trading_system.reports import write_daily_report

    le = LearningEngine(cfg.learning, tmp_db)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pos = _closed_pos(pnl=0.2)
    pos.id = tmp_db.insert_trade(pos)
    le.on_trade_closed(pos)
    path = write_daily_report(db=tmp_db, learning=le, out_dir=tmp_path, day=day)
    text = path.read_text(encoding="utf-8")
    assert "Aprendizaje del día" in text
    assert "Errores identificados" in text
    assert "Oportunidades" in text
    assert "Cambios implementados" in text
    assert "Progreso del aprendizaje" in text
    assert "Decisión:" in text
    assert "Repeticiones" in text
    assert tmp_db.latest_daily_report() is not None


def test_report_shows_accept_reject_reasons(cfg, tmp_db, tmp_path):
    from trading_system.reports import write_daily_report

    cfg.learning.pattern_min_occurrences = 20
    le = LearningEngine(cfg.learning, tmp_db)
    for _ in range(20):
        pos = _closed_pos(pnl=-0.3, regime="high_vol")
        pos.id = tmp_db.insert_trade(pos)
        le.on_trade_closed(pos)
    # one observing win candidate
    pos_w = _closed_pos(pnl=0.4, regime="ranging")
    pos_w.id = tmp_db.insert_trade(pos_w)
    le.on_trade_closed(pos_w)

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = write_daily_report(db=tmp_db, learning=le, out_dir=tmp_path, day=day)
    text = path.read_text(encoding="utf-8")
    assert "ACEPTADO" in text
    assert "NO ACEPTADO" in text
    assert "Repeticiones al confirmar: **20**" in text
    assert "regime=high_vol" in text
    assert "solo esta condición contextual" in text or "solo esa key" in text or "NO invalida los 5" in text


def test_take_profit_negative_net_is_strategy_win_not_loss(cfg, tmp_db):
    from trading_system.learning import classify_strategy_outcome

    pos = _closed_pos(pnl=-0.01, regime="ranging", exit_reason="take_profit")
    pos.gross_pnl = 0.02
    pos.cost_erosion = True
    direction, erosion = classify_strategy_outcome(pos)
    assert direction == "win"
    assert erosion is True

    le = LearningEngine(cfg.learning, tmp_db)
    pos.id = tmp_db.insert_trade(pos)
    le.on_trade_closed(pos)
    wins = tmp_db.get_patterns(direction="win")
    losses = tmp_db.get_patterns(direction="loss")
    costs = tmp_db.get_patterns(direction="cost_erosion")
    assert any(p["pattern_key"] == "exit_reason=take_profit" for p in wins)
    assert not any(p["pattern_key"] == "exit_reason=take_profit" for p in losses)
    assert any("cost_erosion" in p["pattern_key"] for p in costs)


def test_close_trade_sets_gross_and_cost_erosion(cfg, tmp_db):
    from trading_system.execution import PaperExecutor
    from trading_system.portfolio import Portfolio

    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=65010,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
    )
    pos = ex.open_trade(sig, 2.5, 65000)
    # Tiny favorable move that fees can wipe
    closed = ex.close_trade(pos, 65005, "take_profit")
    assert closed.gross_pnl is not None
    assert closed.exit_reason == "take_profit"
    # With default fee+slip bps, small move often nets <= 0
    direction, erosion = __import__(
        "trading_system.learning", fromlist=["classify_strategy_outcome"]
    ).classify_strategy_outcome(closed)
    assert direction == "win"
    if closed.pnl is not None and closed.pnl <= 0:
        assert erosion is True
        assert closed.cost_erosion is True


def test_rebuild_reclassifies_legacy_take_profit(cfg, tmp_db):
    from trading_system.learning.rebuild import rebuild_patterns

    # Legacy-style trade: TP exit, negative net, no gross fields
    pos = _closed_pos(pnl=-0.004, regime="ranging", exit_reason="take_profit")
    pos.gross_pnl = None
    pos.cost_erosion = False
    pos.entry_mark = None
    pos.exit_price = 100.1
    pos.fees = 0.01
    pos.id = tmp_db.insert_trade(pos)

    # Wrong historical classification as loss
    tmp_db.increment_pattern("exit_reason=take_profit", "loss", datetime.now(timezone.utc).isoformat())

    summary = rebuild_patterns(tmp_db, cfg.learning, quiet=True)
    assert summary["closed_trades"] == 1
    assert summary["tp_net_negative_reclassified"] == 1

    refreshed = tmp_db.get_all_closed()[0]
    assert refreshed.gross_pnl is not None
    assert refreshed.cost_erosion is True

    wins = tmp_db.get_patterns(direction="win")
    losses = tmp_db.get_patterns(direction="loss")
    costs = tmp_db.get_patterns(direction="cost_erosion")
    assert any(p["pattern_key"] == "exit_reason=take_profit" for p in wins)
    assert not any(p["pattern_key"] == "exit_reason=take_profit" for p in losses)
    assert len(costs) >= 1
