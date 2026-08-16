"""Unit and integration tests."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json

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


def _entry_feats(**extra) -> str:
    import json

    base = {
        "rsi": 28.0,
        "rsi_prev": 32.0,
        "close": 100.0,
        "bb_lower": 99.0,
        "bb_mid": 100.5,
        "bb_upper": 102.0,
        "bb_width": 0.03,
        "macd_fast_hist": 0.02,
        "macd_fast_hist_prev": -0.01,
        "macd_slow_hist": 0.01,
        "macd_fast_bull_cross": True,
        "macd_fast_bear_cross": False,
        "rejection_bull": True,
        "rejection_bear": False,
        "htf_bias": "bear",
        "ltf_turn": "turn_down",
        "backing": "rsi,macd,rejection",
        "edge_bps": 20.0,
        "round_trip_cost_bps": 4.0,
        "edge_ratio": 5.0,
        "chart_pattern": "hs_top",
    }
    base.update(extra)
    return json.dumps(base)


def _closed_pos(
    *,
    pnl: float,
    regime: str = "high_vol",
    exit_reason: str = "time_stop",
    symbol: str = "BTC/USDT",
    confidence: float = 55,
    features_json: str | None = None,
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
        features_json=features_json if features_json is not None else _entry_feats(),
    )


def test_pattern_not_confirmed_below_10(cfg, tmp_db):
    cfg.learning.pattern_min_occurrences = 10
    le = LearningEngine(cfg.learning, tmp_db)
    for _ in range(9):
        pos = _closed_pos(pnl=-0.5)
        pos.id = tmp_db.insert_trade(pos)
        newly = le.on_trade_closed(pos)
        assert newly == []
    patterns = tmp_db.get_patterns(direction="loss", status="confirmed")
    assert patterns == []
    observing = tmp_db.get_patterns(direction="loss", status="observing")
    assert any(p["pattern_key"].startswith("rsi_zone=") and p["count"] == 9 for p in observing)
    assert not any(
        p["pattern_key"].startswith("session=")
        or p["pattern_key"].startswith("symbol=")
        or p["pattern_key"].startswith("chart=")
        for p in observing
    )


def test_win_pattern_confirms_at_10_boost_only(cfg, tmp_db):
    cfg.learning.pattern_min_occurrences = 10
    cfg.learning.win_confidence_boost = 8.0
    le = LearningEngine(cfg.learning, tmp_db)
    for i in range(10):
        pos = _closed_pos(pnl=0.5, regime="ranging")
        pos.id = tmp_db.insert_trade(pos)
        newly = le.on_trade_closed(pos)
        if i < 9:
            assert newly == []
        else:
            assert any(n["direction"] == "win" for n in newly)
    confirmed = tmp_db.get_patterns(direction="win", status="confirmed")
    assert any(p["pattern_key"].startswith("rsi_zone=") for p in confirmed)
    assert not any(p["pattern_key"].startswith("session=") for p in confirmed)
    changes = tmp_db.applied_changes_on_day(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    assert any(c["action"] == "confidence_boost" for c in changes)

    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=50,
        reason="test",
        regime=MarketRegime.RANGING,
        features={
            "rsi": 28.0,
            "rsi_prev": 32.0,
            "close": 100.0,
            "bb_lower": 99.0,
            "bb_mid": 100.5,
            "bb_upper": 102.0,
            "bb_width": 0.03,
            "macd_fast_hist": 0.02,
            "macd_fast_hist_prev": -0.01,
            "macd_slow_hist": 0.01,
            "rejection_bull": True,
            "htf_bias": "bear",
            "ltf_turn": "turn_down",
            "backing": "rsi,macd,rejection",
            "edge_bps": 20.0,
            "round_trip_cost_bps": 4.0,
            "edge_ratio": 5.0,
        },
        timestamp=datetime.now(timezone.utc),
    )
    adjusted, reject = le.apply_confidence_effects(sig)
    assert reject is None
    assert adjusted.confidence == pytest.approx(58.0)


def test_loss_pattern_hard_reject_at_10(cfg, tmp_db):
    cfg.learning.pattern_min_occurrences = 10
    le = LearningEngine(cfg.learning, tmp_db)
    for _ in range(10):
        pos = _closed_pos(pnl=-0.4, regime="breakout", symbol="ETH/USDT")
        pos.id = tmp_db.insert_trade(pos)
        le.on_trade_closed(pos)
    changes = tmp_db.applied_changes_on_day(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    assert any(c["action"] == "hard_reject" for c in changes)
    sig = Signal(
        symbol="ETH/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        regime=MarketRegime.BREAKOUT,
        features={
            "rsi": 28.0,
            "rsi_prev": 32.0,
            "close": 100.0,
            "bb_lower": 99.0,
            "bb_mid": 100.5,
            "bb_upper": 102.0,
            "bb_width": 0.03,
            "macd_fast_hist": 0.02,
            "macd_fast_hist_prev": -0.01,
            "macd_slow_hist": 0.01,
            "rejection_bull": True,
            "htf_bias": "bear",
            "ltf_turn": "turn_down",
            "backing": "rsi,macd,rejection",
            "edge_bps": 20.0,
            "round_trip_cost_bps": 4.0,
            "edge_ratio": 5.0,
        },
        timestamp=datetime.now(timezone.utc),
    )
    adjusted, reject = le.apply_confidence_effects(sig)
    assert reject is not None
    assert "confirmed_loss_pattern" in reject
    assert adjusted.confidence == pytest.approx(70.0)  # no penalty path; reject only


def test_capital_auto_refill(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    port.cash = 0.5
    tmp_db.set_state("cash", "0.5")
    did = port.maybe_refill(
        auto_refill=True,
        refill_to=100.0,
        min_trade_size=2.5,
        refill_below=30.0,
    )
    assert did is True
    assert port.cash == 100.0
    assert port.capital_resets() == 1


def test_capital_refill_at_threshold_30(cfg, tmp_db):
    port = Portfolio(tmp_db, 100)
    port.cash = 30.0
    tmp_db.set_state("cash", "30.0")
    did = port.maybe_refill(
        auto_refill=True,
        refill_to=100.0,
        min_trade_size=10.0,
        refill_below=30.0,
    )
    assert did is True
    assert port.cash == 100.0

    port.cash = 30.01
    tmp_db.set_state("cash", "30.01")
    did = port.maybe_refill(
        auto_refill=True,
        refill_to=100.0,
        min_trade_size=10.0,
        refill_below=30.0,
    )
    assert did is False
    assert port.cash == 30.01


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

    cfg.learning.pattern_min_occurrences = 10
    le = LearningEngine(cfg.learning, tmp_db)
    for _ in range(10):
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
    assert "Repeticiones al confirmar: **10**" in text or "≥ umbral 10" in text or "umbral 10" in text
    assert "rsi_zone=" in text or "hard_reject" in text or "confidence_boost" in text


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
    assert any(p["pattern_key"].startswith("rsi_zone=") for p in wins)
    assert not any(p["pattern_key"].startswith("exit_reason=") for p in wins)
    assert not any(p["pattern_key"].startswith("exit_reason=") for p in losses)
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

    # Wrong historical classification as loss (forbidden key — wiped on rebuild replay)
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
    assert any(p["pattern_key"].startswith("rsi_zone=") for p in wins)
    assert not any("exit_reason=take_profit" in p["pattern_key"] for p in wins + losses)
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
    weekday = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)  # Friday
    weekend = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)  # Saturday
    crypto = cfg.execution.costs_for_venue("crypto", as_of=weekday)
    forex = cfg.execution.costs_for_venue("forex", as_of=weekday)
    assert crypto == (4.0, 2.0)
    assert forex == (0.35, 1.0)  # Pepperstone Razor proxy
    assert 2 * sum(forex) < 2 * sum(crypto)
    # Weekend crypto practices FX fee schedule
    assert cfg.execution.costs_for_venue("crypto", as_of=weekend) == forex


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
    cfg.execution.weekend_use_forex_costs = False  # pin crypto 4+2 schedule
    cfg.execution.fee_bps = 4.0
    cfg.execution.slippage_bps = 2.0
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


def test_forex_open_without_sl_gets_margin_pct_stop(cfg, tmp_db):
    """FX (and any venue) must get the same 4% margin SL even if the signal omitted it."""
    from trading_system.execution import PaperExecutor
    from trading_system.portfolio import Portfolio

    port = Portfolio(tmp_db, 100)
    cfg.execution.leverage = 20.0
    cfg.execution.forex_fee_bps = 1.0
    cfg.execution.forex_slippage_bps = 1.0
    cfg.strategy.sl_margin_pct = 4.0
    cfg.strategy.tp_mode = "trend_fade"
    ex = PaperExecutor(cfg, tmp_db, port)
    sig = Signal(
        symbol="EUR/USD",
        venue=Venue.FOREX,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        take_profit=None,
        stop_loss=None,
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.RANGING,
        features={},
    )
    pos = ex.open_trade(sig, 10.0, 1.1500)
    assert pos.stop_loss is not None
    assert pos.stop_loss < pos.entry_price
    feat = json.loads(pos.features_json)
    assert feat["sl_budget_cash"] == pytest.approx(0.40)
    # Adverse ~25 bps on 20x notional blows the 4% margin budget
    closed = ex.manage_open(
        {"EUR/USD": 1.1470},
        forex_session_open=True,
        close_fx_at_session_end=False,
        feature_rows={},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop_loss"


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

    chart_fade, chart_reasons = detect_trend_fade(
        Side.CALL,
        {
            "rejection_bear": True,
            "chart_reversal_bear": True,
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
    assert chart_fade is True
    assert "chart_reversal" in chart_reasons

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
    # Favorable mark so net > 0; hard chart reversal → lock
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
        "chart_reversal_bear": True,
        "htf_bias": "bull",
    }
    closed = ex.manage_open(
        {"BTC/USDT": 65260.0},
        forex_session_open=True,
        close_fx_at_session_end=False,
        feature_rows={"BTC/USDT": fade_row},
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "trend_reversal"
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
    """time_stop disabled — past max_hold alone does not exit (raise limbo for this case)."""
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
    cfg.exit.limbo_flat_max_minutes = 60.0  # isolate from limbo_timeout
    cfg.exit.stale_position_max_minutes = 120.0
    cfg.exit.stale_soft_minutes = 90.0
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


def test_legacy_forbidden_keys_do_not_affect_entry(cfg, tmp_db):
    """session/strategy/symbol confirmed losses must not hard-reject (not allowlisted)."""
    le = LearningEngine(cfg.learning, tmp_db)
    now = datetime.now(timezone.utc).isoformat()
    for key in (
        "session=weekend",
        "session=weekend|strategy=bb_mean_reversion",
        "session=weekend|symbol=ETH/USDT",
        "chart=hs_top",
    ):
        for _ in range(10):
            tmp_db.increment_pattern(key, "loss", now)
        tmp_db.confirm_pattern(
            key, "loss", confirmed_count=10, decision_reason="legacy", effect_action="hard_reject"
        )
    sig = Signal(
        symbol="ETH/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        features={"rsi": 28.0, "chart_pattern": "hs_top"},
        timestamp=datetime.now(timezone.utc),
    )
    _, reject = le.apply_confidence_effects(sig)
    assert reject is None


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


def test_exit_confirmed_key_does_not_reject_entry(cfg, tmp_db):
    le = LearningEngine(cfg.learning, tmp_db)
    now = datetime.now(timezone.utc).isoformat()
    key = "exit_class=hard_reversal"
    for _ in range(10):
        tmp_db.increment_pattern(key, "loss", now)
    tmp_db.confirm_pattern(
        key, "loss", confirmed_count=10, decision_reason="exit", effect_action="exit_insight_only"
    )
    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.CALL,
        strategy="bulkowski_pattern",
        confidence=70,
        reason="test",
        features={"rsi": 28.0, "exit_pattern_class": "hard_reversal"},
        timestamp=datetime.now(timezone.utc),
    )
    _, reject = le.apply_confidence_effects(sig)
    assert reject is None


def test_session_patterns_do_not_cross_bleed(cfg, tmp_db):
    """Legacy session= keys never affect entry under keys policy v2."""
    le = LearningEngine(cfg.learning, tmp_db)
    now = datetime.now(timezone.utc).isoformat()
    night_key = "session=night|regime=breakout"
    for _ in range(20):
        tmp_db.increment_pattern(night_key, "loss", now)
    tmp_db.confirm_pattern(
        night_key, "loss", confirmed_count=20, decision_reason="test", effect_action="soft_reject"
    )
    europe_ts = datetime(2026, 8, 12, 9, 30, tzinfo=timezone.utc)
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
    assert cfg.learning.pattern_min_occurrences == 10
    assert cfg.learning.soft_reject_exclude_key_prefixes == []
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


def test_trade_size_never_below_base(cfg):
    risk = RiskManager(cfg)
    assert risk.trade_size(100) == 10
    assert risk.trade_size(91) == 10
    assert risk.trade_size(50) == 10
    assert risk.trade_size(150) == 11
    assert risk.trade_size(200) == 12


def test_trade_inventory_closed_count_vs_last_id(tmp_db):
    a = _closed_pos(pnl=0.1, exit_reason="trend_exit")
    b = _closed_pos(pnl=-0.1, exit_reason="stop_loss")
    tmp_db.insert_trade(a)
    tmp_db.insert_trade(b)
    inv = tmp_db.trade_inventory()
    assert inv["closed"] == 2
    assert inv["last_id"] == 2
    with tmp_db.connection() as conn:
        conn.execute("DELETE FROM trades WHERE id = 1")
    inv = tmp_db.trade_inventory()
    assert inv["closed"] == 1
    assert inv["last_id"] == 2
    assert inv["rows"] == 1


def test_double_top_detector():
    import numpy as np

    from trading_system.patterns import scan_patterns

    n = 80
    close = np.full(n, 105.0)
    high = np.full(n, 106.0)
    low = np.full(n, 104.0)
    open_ = np.full(n, 105.0)
    for i, h in ((22, 108.0), (23, 109.0), (24, 110.0), (25, 109.2), (26, 108.0)):
        high[i] = h
        close[i] = h - 0.4
        open_[i] = h - 0.8
        low[i] = h - 1.2
    for i, l in ((32, 104.2), (33, 103.4), (34, 103.0), (35, 103.5), (36, 104.2)):
        low[i] = l
        close[i] = l + 0.3
        open_[i] = l + 0.6
        high[i] = l + 1.2
    for i, h in ((47, 108.0), (48, 109.2), (49, 110.1), (50, 109.0), (51, 108.0)):
        high[i] = h
        close[i] = h - 0.4
        open_[i] = h - 0.8
        low[i] = h - 1.2
    close[-1] = 102.0
    high[-1] = 103.2
    low[-1] = 101.4
    open_[-1] = 103.0
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 10.0),
        }
    )
    names = {p.name for p in scan_patterns(df)}
    assert "double_top" in names


def test_htf_votes_and_bb_blocks_fade_into_trend(cfg):
    from trading_system.patterns import combine_htf_votes
    from trading_system.strategies import MomentumContinuationStrategy

    assert combine_htf_votes({"15m": "bull", "30m": "bull", "1h": "bull"}) == "bull"
    assert combine_htf_votes({"15m": "bull", "30m": "bear", "1h": "bull"}) == "bull"
    assert combine_htf_votes({"15m": "bull", "30m": "bear", "1h": "mixed"}) == "mixed"

    adapter = SimulatedCryptoAdapter(seed=2)
    df = adapter.get_ohlcv("BTC/USDT", limit=150)
    cont = MomentumContinuationStrategy()
    assert (
        cont.evaluate(
            "BTC/USDT", Venue.CRYPTO, df, cfg.strategy, context={"htf_bias": "unknown"}
        )
        is None
    )

    strat = BBMeanReversionStrategy()
    for seed in range(40):
        df = SimulatedCryptoAdapter(seed=seed).get_ohlcv("BTC/USDT", limit=150)
        raw = strat.evaluate(
            "BTC/USDT", Venue.CRYPTO, df, cfg.strategy, context={"htf_bias": "unknown"}
        )
        if raw is None:
            continue
        against = "bull" if raw.side == Side.PUT else "bear"
        blocked = strat.evaluate(
            "BTC/USDT",
            Venue.CRYPTO,
            df,
            cfg.strategy,
            context={"htf_bias": against},
        )
        assert blocked is None
        return


def test_signal_keys_entry_buckets_not_chart(cfg):
    from trading_system.learning import signal_pattern_keys

    sig = Signal(
        symbol="BTC/USDT",
        venue=Venue.CRYPTO,
        side=Side.PUT,
        strategy="bb_mean_reversion",
        confidence=70,
        reason="test",
        features={
            "rsi": 72.0,
            "rsi_prev": 68.0,
            "htf_bias": "bear",
            "ltf_turn": "turn_down",
            "chart_pattern": "double_top",
            "close": 100.0,
            "bb_lower": 98.0,
            "bb_mid": 100.0,
            "bb_upper": 102.0,
            "bb_width": 0.04,
            "macd_fast_hist": -0.01,
            "macd_slow_hist": -0.02,
            "edge_bps": 10.0,
            "round_trip_cost_bps": 4.0,
            "edge_ratio": 2.5,
        },
        timestamp=datetime.now(timezone.utc),
    )
    keys = signal_pattern_keys(sig, cfg.learning)
    assert any(k.startswith("htf_ltf_combo=") for k in keys)
    assert any(k.startswith("rsi_zone=") for k in keys)
    assert not any("chart=" in k for k in keys)
    assert not any(k.startswith("session=") for k in keys)
    assert not any(k.startswith("symbol=") for k in keys)


def test_bulkowski_pattern_enters_when_htf_agrees(cfg):
    from trading_system.patterns import DetectedPattern
    from trading_system.strategies import ChartPatternStrategy

    df = SimulatedCryptoAdapter(seed=3).get_ohlcv("BTC/USDT", limit=150)
    pat = DetectedPattern(
        "double_bottom", "bullish", 72.0, 100.0, {"p1": 99.0, "neck": 101.0}
    )
    strat = ChartPatternStrategy()
    sig = strat.evaluate(
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
    assert sig is not None
    assert sig.strategy == "bulkowski_pattern"
    assert sig.side == Side.CALL
    assert sig.features.get("measure_target") is not None
    assert sig.take_profit is None

    blocked = strat.evaluate(
        "BTC/USDT",
        Venue.CRYPTO,
        df,
        cfg.strategy,
        context={
            "htf_bias": "bear",
            "patterns": [pat],
            "htf_patterns": [],
            "ltf_turn": "turn_up",
        },
    )
    assert blocked is None



def test_weekend_hs_close_increments_buckets_not_coarse(cfg, tmp_db):
    """Acceptance: weekend H&S close increments buckets, not session/symbol/chart."""
    from trading_system.learning import pattern_keys_from_trade

    le = LearningEngine(cfg.learning, tmp_db)
    pos = _closed_pos(
        pnl=-0.2,
        symbol="BTC/USDT",
        exit_reason="trend_reversal",
        features_json=_entry_feats(
            chart_pattern="hs_top",
            exit_pattern_class="hard_reversal",
            mfe_pct=0.2,
            mae_pct=-0.1,
            giveback_pct=0.4,
        ),
    )
    pos.entry_time = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)  # Sat
    pos.exit_time = datetime(2026, 8, 15, 14, 12, tzinfo=timezone.utc)
    pos.id = tmp_db.insert_trade(pos)
    keys = pattern_keys_from_trade(pos, cfg.learning)
    assert any(k.startswith("rsi_zone=") for k in keys)
    assert any(k.startswith("bb_pos=") for k in keys)
    assert any("rsi_zone=" in k and "bb_pos=" in k for k in keys)
    assert not any(k.startswith("session=") for k in keys)
    assert not any(k.startswith("symbol=") for k in keys)
    assert not any(k.startswith("chart=") for k in keys)
    le.on_trade_closed(pos)
    evidence = tmp_db.get_patterns()
    keys_db = {p["pattern_key"] for p in evidence}
    assert any(k.startswith("rsi_zone=") for k in keys_db)
    assert "session=weekend" not in keys_db
    assert "symbol=BTC/USDT" not in keys_db
    assert "chart=hs_top" not in keys_db
    assert not any(k.startswith("chart=") for k in keys_db)


def test_uneconomic_skips_entry_win_loss(cfg, tmp_db):
    le = LearningEngine(cfg.learning, tmp_db, edge_multiple=0.5)
    pos = _closed_pos(
        pnl=-0.2,
        features_json=_entry_feats(
            edge_bps=1.0,
            round_trip_cost_bps=10.0,
            edge_ratio=0.1,
            exit_pattern_class="ambiguous",
            mfe_pct=0.05,
        ),
    )
    pos.id = tmp_db.insert_trade(pos)
    le.on_trade_closed(pos)
    evidence = tmp_db.get_patterns()
    entry_like = [
        p for p in evidence if p["pattern_key"].startswith("rsi_zone=") and p["direction"] in ("win", "loss")
    ]
    assert entry_like == []
    # EXIT diagnostics may still increment
    assert any(p["pattern_key"].startswith("exit_class=") for p in evidence)


def test_sanitize_wipes_legacy_keeps_allowlist(cfg, tmp_db):
    from trading_system.learning.sanitize import sanitize_pattern_evidence

    now = datetime.now(timezone.utc).isoformat()
    tmp_db.increment_pattern("session=weekend", "loss", now)
    tmp_db.increment_pattern("chart=hs_top", "win", now)
    tmp_db.increment_pattern("rsi_zone=lt30", "loss", now)
    tmp_db.increment_pattern("cost_erosion|exit=x|symbol=BTC/USDT", "cost_erosion", now)
    summary = sanitize_pattern_evidence(tmp_db, dry_run=False)
    assert summary["dropped_evidence"] >= 2
    left = {p["pattern_key"] for p in tmp_db.get_patterns()}
    assert "rsi_zone=lt30" in left
    assert "session=weekend" not in left
    assert "chart=hs_top" not in left
    assert any(k.startswith("cost_erosion") for k in left)
