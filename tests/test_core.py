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
    assert any(
        p["pattern_key"].endswith("regime=high_vol") and p["count"] == 19 for p in observing
    )


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
    assert any(p["pattern_key"].endswith("regime=ranging") for p in confirmed)
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
        pos = _closed_pos(pnl=-0.4, regime="breakout", symbol="ETH/USDT")
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
    assert "Ajustes activos" in text
    assert "Resumen por horario" in text
    assert "take_profit=" in text
    assert "stop_loss=" in text
    assert "Decisión:" in text
    assert "Repeticiones" in text
    assert tmp_db.latest_daily_report() is not None


def test_daily_report_session_breakdown(cfg, tmp_db, tmp_path):
    from trading_system.reports import write_daily_report

    le = LearningEngine(cfg.learning, tmp_db)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # asia-hour trade
    pos = _closed_pos(pnl=-0.1, exit_reason="stop_loss")
    pos.entry_time = datetime(2026, 8, 12, 3, 0, tzinfo=timezone.utc)
    pos.exit_time = datetime(2026, 8, 12, 3, 5, tzinfo=timezone.utc)
    pos.id = tmp_db.insert_trade(pos)
    le.on_trade_closed(pos)
    # europe-hour TP
    pos2 = _closed_pos(pnl=0.05, exit_reason="take_profit", regime="ranging")
    pos2.entry_time = datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc)
    pos2.exit_time = datetime(2026, 8, 12, 9, 2, tzinfo=timezone.utc)
    pos2.id = tmp_db.insert_trade(pos2)
    le.on_trade_closed(pos2)

    path = write_daily_report(db=tmp_db, learning=le, out_dir=tmp_path, day=day)
    text = path.read_text(encoding="utf-8")
    assert "`asia`" in text
    assert "`europe`" in text
    assert "00:00–07:00 UTC" in text or "00:00" in text
    assert "Patrones confirmados de ganancia" in text
    assert "Cambios / efectos aplicados" in text
    summary = tmp_db.latest_daily_report()
    assert summary is not None


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
    assert any(p["pattern_key"].endswith("exit_reason=take_profit") for p in wins)
    assert not any(p["pattern_key"].endswith("exit_reason=take_profit") for p in losses)
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
    assert any(p["pattern_key"].endswith("exit_reason=take_profit") for p in wins)
    assert not any(p["pattern_key"].endswith("exit_reason=take_profit") for p in losses)
    assert len(costs) >= 1


def test_entry_edge_hard_and_soft():
    from trading_system.execution.edge import assess_entry_edge

    # 5 bps to TP, ~12 bps round trip → hard reject
    hard = assess_entry_edge(
        price=100.0,
        take_profit=100.05,
        fee_bps=4.0,
        slippage_bps=2.0,
        hard_multiple=0.75,
        soft_multiple=1.25,
    )
    assert hard.hard_reject is True

    # 12 bps to TP vs 12 cost → soft zone (~1.0x with soft=1.25)
    soft = assess_entry_edge(
        price=100.0,
        take_profit=100.12,
        fee_bps=4.0,
        slippage_bps=2.0,
        hard_multiple=0.75,
        soft_multiple=1.25,
    )
    assert soft.hard_reject is False
    assert soft.soft_penalty is True
    # 20 bps → ok
    ok = assess_entry_edge(
        price=100.0,
        take_profit=100.20,
        fee_bps=4.0,
        slippage_bps=2.0,
        hard_multiple=0.75,
        soft_multiple=1.25,
    )
    assert ok.hard_reject is False
    assert ok.soft_penalty is False


def test_forex_costs_cheaper_than_crypto(cfg):
    crypto = cfg.execution.costs_for_venue("crypto")
    forex = cfg.execution.costs_for_venue("forex")
    assert crypto == (4.0, 2.0)
    assert forex == (1.0, 1.0)
    # round-trip: crypto 12 bps vs forex 4 bps
    assert 2 * sum(forex) < 2 * sum(crypto)


def test_tp_deferred_when_net_negative(cfg, tmp_db):
    from trading_system.execution import PaperExecutor
    from trading_system.portfolio import Portfolio

    port = Portfolio(tmp_db, 100)
    ex = PaperExecutor(cfg, tmp_db, port)
    cfg.execution.tp_require_positive_net = True
    cfg.execution.leverage = 1.0  # isolate fee geometry
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=65001,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
    )
    pos = ex.open_trade(sig, 2.5, 65000)
    # Mark barely above TP — fees make net <= 0
    closed = ex.manage_open(
        {"BTC/USDT": 65001.0},
        forex_session_open=True,
        close_fx_at_session_end=False,
    )
    assert closed == []
    assert len(port.open_positions()) == 1


def test_tp_requires_strictly_positive_net(cfg, tmp_db):
    from trading_system.execution.edge import can_take_profit_net_positive, estimate_close_net
    from trading_system.execution import PaperExecutor
    from trading_system.portfolio import Portfolio

    port = Portfolio(tmp_db, 100)
    cfg.execution.leverage = 1.0
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
    # Craft a mark where estimate is ~0 or negative — must not pass > 0 gate
    assert can_take_profit_net_positive(pos, 65001.0, 4.0, 2.0) is False
    # Large enough move should pass
    assert can_take_profit_net_positive(pos, 65200.0, 4.0, 2.0) is True
    assert estimate_close_net(pos, 65200.0, 4.0, 2.0) > 0


def test_leverage_scales_notional_and_fees(cfg, tmp_db):
    from trading_system.execution import PaperExecutor
    from trading_system.portfolio import Portfolio

    port = Portfolio(tmp_db, 100)
    cfg.execution.leverage = 20.0
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
    cash_before = port.cash
    pos = ex.open_trade(sig, 10.0, 65000)
    assert pos.leverage == 20.0
    assert abs(pos.notional - 200.0) < 1e-9
    # Entry fee on notional 200 * 6bps
    expected_entry_fee = 200.0 * 6.0 / 10_000
    assert abs(pos.fees - expected_entry_fee) < 1e-9
    assert abs(cash_before - port.cash - (10.0 + expected_entry_fee)) < 1e-9

    closed = ex.close_trade(pos, 65500, "take_profit")
    # PnL on notional 200, not on margin 10
    assert closed.pnl is not None
    assert closed.gross_pnl is not None
    assert abs(closed.gross_pnl) > abs(
        (65500 - 65000) / 65000 * 10.0
    )  # larger than 1x margin PnL


def test_near_extreme_and_rejection_required(cfg):
    from trading_system.strategies import (
        _near_extreme,
        compute_early_rejection_tp,
        compute_sl_from_tp_rr,
        compute_tight_stop_loss,
    )
    from trading_system.types import Side

    row = pd.Series(
        {
            "bb_upper": 110.0,
            "bb_lower": 100.0,
            "bb_mid": 105.0,
            "close": 101.0,  # 10% from lower toward mid
        }
    )
    assert _near_extreme(row, Side.CALL, 0.35) is True
    row["close"] = 108.0  # far from lower
    assert _near_extreme(row, Side.CALL, 0.35) is False
    row["close"] = 109.0
    assert _near_extreme(row, Side.PUT, 0.35) is True
    row["close"] = 103.0
    assert _near_extreme(row, Side.PUT, 0.35) is False

    # TP is early rejection — short of BB mid; SL from TP at 1:1.5 net
    tp_call = compute_early_rejection_tp(
        side=Side.CALL,
        price=100.5,
        bb_lower=100.0,
        bb_mid=105.0,
        bb_upper=110.0,
        cfg=cfg.strategy,
    )
    sl_call, budget_c, trigger_c, tp_net_c = compute_sl_from_tp_rr(
        side=Side.CALL,
        price=100.5,
        take_profit=tp_call,
        exit_fee_bps=6.0,
        reward_multiple=1.5,
    )
    assert tp_call > 100.5
    assert tp_call < 105.0
    assert sl_call < 100.5
    assert abs(tp_net_c / budget_c - 1.5) < 1e-6
    assert abs(budget_c - (trigger_c + 6.0)) < 1e-6 or trigger_c == 1.0

    tp_put = compute_early_rejection_tp(
        side=Side.PUT,
        price=109.5,
        bb_lower=100.0,
        bb_mid=105.0,
        bb_upper=110.0,
        cfg=cfg.strategy,
    )
    sl_put, budget_p, trigger_p, tp_net_p = compute_sl_from_tp_rr(
        side=Side.PUT,
        price=109.5,
        take_profit=tp_put,
        exit_fee_bps=6.0,
        reward_multiple=1.5,
    )
    assert tp_put < 109.5
    assert tp_put > 105.0
    assert sl_put > 109.5
    assert abs(tp_net_p / budget_p - 1.5) < 1e-6

    # Legacy band mode still fee-reserves inside fixed budget
    cfg.strategy.sl_mode = "band"
    sl_band, budget_b, trigger_b = compute_tight_stop_loss(
        side=Side.CALL,
        price=100.5,
        bb_lower=100.0,
        bb_upper=110.0,
        cfg=cfg.strategy,
        exit_fee_bps=6.0,
    )
    assert trigger_b < budget_b
    assert abs(budget_b - (trigger_b + 6.0)) < 1e-6 or trigger_b == 1.0
    cfg.strategy.sl_mode = "margin_pct"


def test_sl_margin_pct_four_percent(cfg, tmp_db):
    from trading_system.execution import PaperExecutor
    from trading_system.portfolio import Portfolio
    from trading_system.strategies import compute_sl_from_margin_pct

    sl, budget_bps, trigger_bps, budget_cash = compute_sl_from_margin_pct(
        side=Side.CALL,
        price=65000.0,
        margin=10.0,
        leverage=20.0,
        sl_margin_pct=4.0,
        exit_fee_bps=6.0,
    )
    assert abs(budget_cash - 0.40) < 1e-9
    assert abs(budget_bps - 20.0) < 1e-6  # 0.40 / 200 * 10000
    assert abs(trigger_bps - 14.0) < 1e-6
    assert sl < 65000.0

    port = Portfolio(tmp_db, 100)
    cfg.execution.leverage = 20.0
    cfg.execution.fee_bps = 4.0
    cfg.execution.slippage_bps = 2.0
    cfg.strategy.tp_mode = "trend_fade"
    ex = PaperExecutor(cfg, tmp_db, port)
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=None,
        stop_loss=sl,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
        features={
            "sl_budget_cash": budget_cash,
            "sl_budget_bps": budget_bps,
            "sl_mode": "margin_pct",
        },
    )
    ex.open_trade(sig, 10.0, 65000)
    # Adverse enough that net (move + exit fee) hits ~0.40 budget
    closed = ex.manage_open(
        {"BTC/USDT": 64890.0},
        forex_session_open=True,
        close_fx_at_session_end=False,
        feature_rows={},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop_loss"
    assert closed[0].pnl is not None
    assert closed[0].pnl > -0.55  # around 4% margin, not runaway


def test_trend_fade_exit_requires_score_and_net(cfg, tmp_db):
    from datetime import timedelta

    from trading_system.execution import PaperExecutor
    from trading_system.execution.edge import detect_trend_fade
    from trading_system.portfolio import Portfolio

    faded, reasons = detect_trend_fade(
        Side.CALL,
        {
            "rejection_bear": True,
            "macd_fast_bear_cross": True,
            "macd_fast_hist": -0.1,
            "macd_fast_hist_prev": 0.1,
            "macd_fast_hist_prev2": 0.2,
            "rsi": 48,
            "rsi_prev": 55,
            "macd_slow_hist": 0.01,
            "macd_slow_hist_prev": 0.05,
        },
        min_score=2,
    )
    assert faded is True
    assert len(reasons) >= 2

    not_faded, _ = detect_trend_fade(
        Side.CALL,
        {
            "rejection_bear": False,
            "macd_fast_bear_cross": False,
            "macd_fast_hist": 0.2,
            "macd_fast_hist_prev": 0.1,
            "macd_fast_hist_prev2": 0.05,
            "rsi": 60,
            "rsi_prev": 55,
            "macd_slow_hist": 0.05,
            "macd_slow_hist_prev": 0.01,
        },
        min_score=2,
    )
    assert not_faded is False

    port = Portfolio(tmp_db, 100)
    cfg.execution.leverage = 20.0
    cfg.execution.fee_bps = 4.0
    cfg.execution.slippage_bps = 2.0
    cfg.strategy.tp_mode = "trend_fade"
    cfg.strategy.min_hold_minutes = 0  # allow immediate fade in test
    ex = PaperExecutor(cfg, tmp_db, port)
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=None,
        stop_loss=60000,  # far
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=2),
        regime=MarketRegime.RANGING,
        features={"sl_budget_cash": 0.40, "sl_mode": "margin_pct"},
    )
    pos = ex.open_trade(sig, 10.0, 65000)
    # Persist past entry_time — manage_open reloads from DB
    past = datetime.now(timezone.utc) - timedelta(minutes=2)
    with tmp_db.connection() as conn:
        conn.execute(
            "UPDATE trades SET entry_time=? WHERE id=?",
            (past.isoformat(), pos.id),
        )
    # Favorable mark so net > 0 after fees (~ +40 bps on notional 200)
    fade_row = {
        "rejection_bear": True,
        "macd_fast_bear_cross": True,
        "macd_fast_hist": -0.1,
        "macd_fast_hist_prev": 0.2,
        "macd_fast_hist_prev2": 0.3,
        "rsi": 49,
        "rsi_prev": 58,
        "macd_slow_hist": 0.0,
        "macd_slow_hist_prev": 0.1,
    }
    closed = ex.manage_open(
        {"BTC/USDT": 65260.0},
        forex_session_open=True,
        close_fx_at_session_end=False,
        feature_rows={"BTC/USDT": fade_row},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "trend_exit"
    assert closed[0].pnl is not None and closed[0].pnl > 0


def test_stop_loss_cuts_before_large_adverse(cfg, tmp_db):
    from trading_system.execution import PaperExecutor
    from trading_system.portfolio import Portfolio

    port = Portfolio(tmp_db, 100)
    cfg.execution.leverage = 1.0
    ex = PaperExecutor(cfg, tmp_db, port)
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=65100,
        stop_loss=64950,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
        features={"sl_budget_bps": 10.0},
    )
    ex.open_trade(sig, 10.0, 65000)
    closed = ex.manage_open(
        {"BTC/USDT": 64940.0},
        forex_session_open=True,
        close_fx_at_session_end=False,
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop_loss"


def test_stop_loss_net_budget_includes_exit_fee(cfg, tmp_db):
    """SL fires when estimated NET (with exit fee) reaches budget, before price SL."""
    from trading_system.execution import PaperExecutor
    from trading_system.portfolio import Portfolio
    import json

    port = Portfolio(tmp_db, 100)
    cfg.execution.leverage = 1.0
    cfg.execution.fee_bps = 4.0
    cfg.execution.slippage_bps = 2.0
    ex = PaperExecutor(cfg, tmp_db, port)
    # Budget 10 bps on notional 10 = 0.01 max net loss
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=66000,
        stop_loss=64000,  # far away — should not be the price trigger
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
        features={"sl_budget_bps": 10.0},
    )
    pos = ex.open_trade(sig, 10.0, 65000)
    # Adverse enough that net+exit fee hits -0.01 budget but still above far SL
    closed = ex.manage_open(
        {"BTC/USDT": 64920.0},
        forex_session_open=True,
        close_fx_at_session_end=False,
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop_loss"
    # Net loss should not massively exceed budget+small overshoot
    assert closed[0].pnl is not None
    assert closed[0].pnl > -0.05


def test_no_time_stop_even_past_max_hold(cfg, tmp_db):
    """time_stop is disabled — winners and losers stay open past max_hold."""
    from datetime import timedelta

    from trading_system.execution import PaperExecutor
    from trading_system.portfolio import Portfolio

    port = Portfolio(tmp_db, 100)
    cfg.execution.leverage = 20.0
    cfg.execution.fee_bps = 4.0
    cfg.execution.slippage_bps = 2.0
    cfg.strategy.tp_mode = "trend_fade"
    cfg.strategy.max_hold_minutes = 1
    cfg.strategy.preferred_hold_minutes = 1
    cfg.strategy.min_hold_minutes = 0
    ex = PaperExecutor(cfg, tmp_db, port)

    def _open(symbol: str, px: float) -> None:
        sig = Signal(
            symbol=symbol,
            venue=Venue.CRYPTO,
            side=Side.CALL,
            strategy="bb_mean_reversion",
            confidence=70,
            reason="test",
            take_profit=None,
            stop_loss=px * 0.5,
            timestamp=datetime.now(timezone.utc),
            regime=MarketRegime.RANGING,
            features={"sl_budget_cash": 50.0, "sl_mode": "margin_pct"},
        )
        pos = ex.open_trade(sig, 10.0, px)
        past = datetime.now(timezone.utc) - timedelta(minutes=15)
        with tmp_db.connection() as conn:
            conn.execute(
                "UPDATE trades SET entry_time=? WHERE id=?",
                (past.isoformat(), pos.id),
            )

    _open("BTC/USDT", 65000.0)
    _open("ETH/USDT", 2000.0)
    closed = ex.manage_open(
        {"BTC/USDT": 65260.0, "ETH/USDT": 1990.0},
        forex_session_open=True,
        close_fx_at_session_end=False,
        feature_rows={"BTC/USDT": {}, "ETH/USDT": {}},
    )
    assert closed == []
    assert len(port.open_positions()) == 2


def test_learning_display_labels():
    from trading_system.learning import learning_display

    win = _closed_pos(pnl=0.01, exit_reason="take_profit")
    win.gross_pnl = 0.02
    d = learning_display(win)
    assert d["learning_direction"] == "win"
    assert "ganancia" in d["learning_label"]

    loss = _closed_pos(pnl=-0.02, exit_reason="time_stop")
    loss.gross_pnl = -0.01
    d2 = learning_display(loss)
    assert d2["learning_direction"] == "loss"
    assert "perdida" in d2["learning_label"]


def test_strategy_loss_pattern_does_not_soft_reject(cfg, tmp_db):
    from trading_system.learning.sessions import session_bucket

    cfg.learning.loss_soft_reject = True
    cfg.learning.soft_reject_exclude_key_prefixes = ["strategy=", "regime="]
    cfg.learning.session_aware = True
    le = LearningEngine(cfg.learning, tmp_db)
    sess = session_bucket(datetime.now(timezone.utc), cfg.learning.session_buckets)
    now = datetime.now(timezone.utc).isoformat()
    strat_key = f"session={sess}|strategy=bb_mean_reversion"
    regime_key = f"session={sess}|regime=breakout"
    symbol_key = f"session={sess}|symbol=ETH/USDT"
    for _ in range(20):
        tmp_db.increment_pattern(strat_key, "loss", now)
    tmp_db.confirm_pattern(
        strat_key,
        "loss",
        confirmed_count=20,
        decision_reason="test",
        effect_action="soft_reject",
    )
    for _ in range(20):
        tmp_db.increment_pattern(regime_key, "loss", now)
    tmp_db.confirm_pattern(
        regime_key,
        "loss",
        confirmed_count=20,
        decision_reason="test",
        effect_action="soft_reject",
    )
    for _ in range(20):
        tmp_db.increment_pattern(symbol_key, "loss", now)
    tmp_db.confirm_pattern(
        symbol_key,
        "loss",
        confirmed_count=20,
        decision_reason="test",
        effect_action="soft_reject",
    )

    sig_ok = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
    )
    adjusted, reject = le.apply_confidence_effects(sig_ok)
    assert reject is None  # strategy= excluded even with session prefix
    assert adjusted.confidence < 70  # penalty may still apply

    sig_regime = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.PUT,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BREAKOUT,
    )
    _, reject_regime = le.apply_confidence_effects(sig_regime)
    assert reject_regime is None  # regime= too broad to hard-reject

    sig_bad = Signal(
        symbol="ETH/USDT",
        venue=Venue.CRYPTO,
        side=Side.PUT,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
    )
    _, reject2 = le.apply_confidence_effects(sig_bad)
    assert reject2 is not None
    assert "symbol=ETH/USDT" in reject2


def test_session_bucket_mapping():
    from trading_system.learning.sessions import session_bucket
    from trading_system.config import SessionBucketConfig

    buckets = [
        SessionBucketConfig(name="asia", start_hour_utc=0, end_hour_utc=7),
        SessionBucketConfig(name="europe", start_hour_utc=7, end_hour_utc=12),
        SessionBucketConfig(name="us_open", start_hour_utc=12, end_hour_utc=16),
        SessionBucketConfig(name="us_afternoon", start_hour_utc=16, end_hour_utc=21),
        SessionBucketConfig(name="night", start_hour_utc=21, end_hour_utc=24),
    ]
    assert session_bucket(datetime(2026, 8, 12, 3, 0, tzinfo=timezone.utc), buckets) == "asia"
    assert session_bucket(datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc), buckets) == "europe"
    assert session_bucket(datetime(2026, 8, 12, 14, 0, tzinfo=timezone.utc), buckets) == "us_open"
    assert session_bucket(datetime(2026, 8, 12, 18, 0, tzinfo=timezone.utc), buckets) == "us_afternoon"
    assert session_bucket(datetime(2026, 8, 12, 22, 0, tzinfo=timezone.utc), buckets) == "night"


def test_session_patterns_do_not_cross_bleed(cfg, tmp_db):
    """Night confirmed loss must not soft-reject a europe entry."""
    from trading_system.learning.sessions import session_bucket

    cfg.learning.session_aware = True
    cfg.learning.loss_soft_reject = True
    le = LearningEngine(cfg.learning, tmp_db)
    now = datetime.now(timezone.utc).isoformat()
    night_key = "session=night|regime=breakout"
    for _ in range(20):
        tmp_db.increment_pattern(night_key, "loss", now)
    tmp_db.confirm_pattern(
        night_key, "loss", confirmed_count=20, decision_reason="test", effect_action="soft_reject"
    )

    # Force europe timestamp
    europe_ts = datetime(2026, 8, 12, 9, 30, tzinfo=timezone.utc)
    assert session_bucket(europe_ts, cfg.learning.session_buckets) == "europe"
    sig = Signal(
        symbol="ETH/USDT",
        venue=Venue.CRYPTO,
        side=Side.PUT,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        timestamp=europe_ts,
        regime=MarketRegime.BREAKOUT,
    )
    _, reject = le.apply_confidence_effects(sig)
    assert reject is None


def test_objective_config_and_daily_progress(cfg):
    from trading_system.learning import daily_objective_progress

    assert cfg.objective.daily_equity_gain_pct == 50.0
    assert cfg.objective.chase_target_in_discovery is False
    assert "regime=" in cfg.learning.soft_reject_exclude_key_prefixes
    prog = daily_objective_progress(
        start_equity=100.0,
        current_equity=110.0,
        target_pct=50.0,
        phase="discovery",
        chase_in_discovery=False,
    )
    assert prog["day_gain_pct"] == pytest.approx(10.0)
    assert prog["progress_vs_target"] == pytest.approx(0.2)
    assert prog["chase_now"] is False
    assert "aprender" in prog["blurb"]


def test_reset_loss_learning_keeps_wins(tmp_path):
    import importlib.util

    from trading_system.config import ROOT

    db = Database(tmp_path / "wipe.db")
    now = datetime.now(timezone.utc)
    win = _closed_pos(pnl=0.4, regime="ranging", exit_reason="trend_exit")
    win.gross_pnl = 0.5
    ts_loss = _closed_pos(pnl=-0.2, regime="low_vol", exit_reason="time_stop")
    sl_loss = _closed_pos(pnl=-0.3, regime="low_vol", exit_reason="stop_loss")
    sl_loss.symbol = "ETH/USDT"
    for p in (win, ts_loss, sl_loss):
        p.exit_time = now
        db.insert_trade(p)
    db.increment_pattern("session=europe|regime=ranging", "win", now.isoformat())
    db.confirm_pattern(
        "session=europe|regime=ranging",
        "win",
        confirmed_count=20,
        decision_reason="keep",
        effect_action="confidence_boost",
    )
    db.increment_pattern("session=europe|exit_reason=time_stop", "win", now.isoformat())
    db.increment_pattern("session=europe|regime=low_vol", "loss", now.isoformat())
    db.insert_applied_change(
        "session=europe|regime=low_vol",
        "loss",
        "soft_reject",
        "test",
        occurrences=20,
        threshold=20,
    )

    spec = importlib.util.spec_from_file_location(
        "reset_loss_learning", ROOT / "scripts" / "reset_loss_learning.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    out = mod.reset_loss_learning(tmp_path / "wipe.db", dry_run=False)
    assert out["time_stop_trades"] == 1

    remaining = db.get_all_closed()
    reasons = {t.exit_reason for t in remaining}
    assert "time_stop" not in reasons
    assert "trend_exit" in reasons
    assert "stop_loss" in reasons
    wins = db.get_patterns(direction="win")
    assert any(p["pattern_key"] == "session=europe|regime=ranging" for p in wins)
    assert not any("time_stop" in p["pattern_key"] for p in wins)
    assert db.get_patterns(direction="loss") == []

