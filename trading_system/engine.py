"""Main trading engine loop."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from trading_system.config import AppConfig, ROOT, load_config
from trading_system.data.crypto import CryptoAdapter, SimulatedCryptoAdapter
from trading_system.data.forex import ForexAdapter
from trading_system.database import Database
from trading_system.execution import PaperExecutor
from trading_system.execution.edge import assess_entry_edge
from trading_system.features import build_features, latest_feature_dict
from trading_system.learning import LearningEngine
from trading_system.models import WinProbabilityModel
from trading_system.portfolio import Portfolio
from trading_system.reports import maybe_rollover_daily_report, write_daily_report
from trading_system.risk import RiskManager
from trading_system.strategies import StrategyRegistry, compute_sl_from_margin_pct, compute_tight_stop_loss
from trading_system.types import PortfolioSnapshot, RejectedSignal, Venue

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, cfg: AppConfig | None = None, simulate: bool = False) -> None:
        self.cfg = cfg or load_config()
        if self.cfg.is_live:
            raise RuntimeError(
                "LIVE mode blocked — set mode=paper. Live requires explicit future activation."
            )

        self.db = Database(self.cfg.db_path())
        self.portfolio = Portfolio(self.db, self.cfg.capital.initial)
        self.risk = RiskManager(self.cfg)
        self.executor = PaperExecutor(self.cfg, self.db, self.portfolio)
        self.learning = LearningEngine(self.cfg.learning, self.db)
        self.model = WinProbabilityModel(ROOT / "models" / "artifacts")
        self.strategies = StrategyRegistry()

        self.simulate = simulate
        if simulate:
            self.crypto = SimulatedCryptoAdapter()
        else:
            try:
                self.crypto = CryptoAdapter(
                    self.cfg.crypto.exchange, sandbox=self.cfg.crypto.sandbox
                )
            except Exception as e:
                logger.warning("CCXT init failed (%s); using simulated crypto", e)
                self.crypto = SimulatedCryptoAdapter()

        self.forex = ForexAdapter(self.cfg.forex_session, provider=self.cfg.forex.provider)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_snapshot: PortfolioSnapshot | None = None
        self._lock = threading.Lock()
        self._cycle = 0
        self.status_message = "initialized"
        self._last_daily_report_path: str | None = None

        # Initialize daily report day marker
        if self.db.get_state("last_daily_report_day") is None:
            self.db.set_state(
                "last_daily_report_day",
                datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            )

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_forever, name="trading-loop", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def run_forever(self) -> None:
        interval = self.cfg.execution.poll_interval_seconds
        logger.info("Engine started interval=%ss mode=%s", interval, self.cfg.mode)
        while not self._stop.is_set():
            try:
                self.tick()
                self.status_message = "running"
            except Exception as e:
                logger.exception("tick failed")
                self.status_message = f"error:{e}"
                if self.cfg.risk.kill_on_api_failure:
                    self.risk.trip(f"tick_error:{e}")
            self._stop.wait(interval)

    def tick(self) -> PortfolioSnapshot:
        with self._lock:
            return self._tick_unlocked()

    def _tick_unlocked(self) -> PortfolioSnapshot:
        self._cycle += 1
        self.risk.check_stale()

        # Daily report rollover (UTC)
        _, report_path = maybe_rollover_daily_report(
            self.db, self.learning, self.db.get_state("last_daily_report_day")
        )
        if report_path is not None:
            self._last_daily_report_path = str(report_path)
            logger.info("Daily report written: %s", report_path)

        mark_prices: dict[str, float] = {}
        feature_rows: dict[str, dict[str, Any]] = {}
        venue_health: dict[str, Any] = {}

        # Refill before trying new entries if flat & broke
        self.portfolio.maybe_refill(
            auto_refill=self.cfg.capital_policy.auto_refill,
            refill_to=self.cfg.capital_policy.refill_to or self.cfg.capital.initial,
            min_trade_size=self.cfg.capital.base_trade_size,
            mark_prices=mark_prices,
        )

        # --- Crypto ---
        try:
            for sym in self.cfg.symbols.crypto:
                df = self.crypto.get_ohlcv(
                    sym, self.cfg.timeframes.primary, self.cfg.timeframes.lookback_bars
                )
                mark_prices[sym] = float(df["close"].iloc[-1])
                feature_rows[sym] = self._feature_row(df)
                self._process_symbol(sym, Venue.CRYPTO, df)
            self.risk.mark_data("crypto", True)
            venue_health["crypto"] = self.crypto.health()
        except Exception as e:
            logger.exception("crypto cycle")
            self.risk.mark_data("crypto", False)
            venue_health["crypto"] = {"ok": False, "error": str(e)}

        # --- Forex (session gated) ---
        fx_open = self.forex.calendar.is_open()
        try:
            for sym in self.cfg.symbols.forex:
                df = self.forex.get_ohlcv(
                    sym, self.cfg.timeframes.primary, self.cfg.timeframes.lookback_bars
                )
                mark_prices[sym] = float(df["close"].iloc[-1])
                feature_rows[sym] = self._feature_row(df)
                if self.forex.is_tradable_now(sym):
                    self._process_symbol(sym, Venue.FOREX, df)
            self.risk.mark_data("forex", True)
            venue_health["forex"] = self.forex.health()
        except Exception as e:
            logger.exception("forex cycle")
            self.risk.mark_data("forex", False)
            venue_health["forex"] = {"ok": False, "error": str(e)}

        closed = self.executor.manage_open(
            mark_prices,
            forex_session_open=fx_open,
            close_fx_at_session_end=self.cfg.forex_session.close_intraday_at_session_end,
            feature_rows=feature_rows,
        )
        for pos in closed:
            self.learning.on_trade_closed(pos)

        # After closes, refill if needed so learning never stalls
        self.portfolio.maybe_refill(
            auto_refill=self.cfg.capital_policy.auto_refill,
            refill_to=self.cfg.capital_policy.refill_to or self.cfg.capital.initial,
            min_trade_size=self.cfg.capital.base_trade_size,
            mark_prices=mark_prices,
        )

        closed_all = self.db.get_all_closed()
        if (
            closed
            and len(closed_all) >= 20
            and len(closed_all) % self.cfg.learning.retrain_every_n_trades < len(closed)
        ):
            result = self.model.train(closed_all)
            self.db.insert_insight("ml_train", str(result), result)

        self.portfolio.snapshot_equity(mark_prices)
        snap = self._build_snapshot(mark_prices, venue_health)
        self._last_snapshot = snap
        return snap

    def _feature_row(self, df) -> dict[str, Any]:
        cfg = self.cfg.strategy
        feat = build_features(
            df,
            bb_period=cfg.bb_period,
            bb_std=cfg.bb_std,
            rsi_period=cfg.rsi_period,
            macd_fast=cfg.macd_fast,
            macd_slow=cfg.macd_slow,
        )
        return latest_feature_dict(feat)

    def _process_symbol(self, symbol: str, venue: Venue, df) -> None:
        strategy = self.strategies.get(self.cfg.strategy.name)
        signal = strategy.evaluate(symbol, venue, df, self.cfg.strategy)
        if signal is None:
            return

        price = float(df["close"].iloc[-1])
        margin = float(self.cfg.capital.base_trade_size)
        leverage = max(1.0, float(self.cfg.execution.leverage or 1.0))
        exit_fee_bps = (
            self.cfg.execution.fee_bps + self.cfg.execution.slippage_bps
            if getattr(self.cfg.strategy, "sl_include_exit_fees", True)
            else 0.0
        )
        sl_mode = (getattr(self.cfg.strategy, "sl_mode", "margin_pct") or "margin_pct").lower()

        # Recompute SL with live margin/leverage/fees
        if sl_mode == "margin_pct":
            sl, budget_bps, trigger_bps, budget_cash = compute_sl_from_margin_pct(
                side=signal.side,
                price=price,
                margin=margin,
                leverage=leverage,
                sl_margin_pct=float(getattr(self.cfg.strategy, "sl_margin_pct", 4.0)),
                exit_fee_bps=float(exit_fee_bps),
            )
            signal.stop_loss = sl
            signal.features["stop_loss"] = sl
            signal.features["sl_budget_bps"] = budget_bps
            signal.features["sl_trigger_bps"] = trigger_bps
            signal.features["sl_budget_cash"] = budget_cash
            signal.features["sl_exit_fee_bps"] = float(exit_fee_bps)
            signal.features["sl_include_exit_fees"] = bool(
                getattr(self.cfg.strategy, "sl_include_exit_fees", True)
            )
            signal.features["sl_mode"] = "margin_pct"
            signal.features["sl_margin_pct"] = float(
                getattr(self.cfg.strategy, "sl_margin_pct", 4.0)
            )
        elif signal.stop_loss is not None:
            bb_lo = signal.features.get("bb_lower")
            bb_hi = signal.features.get("bb_upper")
            if bb_lo is not None and bb_hi is not None:
                sl, budget_bps, trigger_bps = compute_tight_stop_loss(
                    side=signal.side,
                    price=price,
                    bb_lower=float(bb_lo),
                    bb_upper=float(bb_hi),
                    cfg=self.cfg.strategy,
                    exit_fee_bps=float(exit_fee_bps),
                    take_profit=signal.take_profit,
                    margin=margin,
                    leverage=leverage,
                )
                signal.stop_loss = sl
                signal.features["stop_loss"] = sl
                signal.features["sl_budget_bps"] = budget_bps
                signal.features["sl_trigger_bps"] = trigger_bps
                signal.features["sl_budget_cash"] = (
                    margin * leverage * float(budget_bps) / 10_000.0
                )
                signal.features["sl_exit_fee_bps"] = float(exit_fee_bps)
                signal.features["sl_include_exit_fees"] = bool(
                    getattr(self.cfg.strategy, "sl_include_exit_fees", True)
                )
                signal.features["sl_mode"] = sl_mode

        # Soft fee-aware edge: use bb_mid distance as expected-move proxy when no fixed TP
        edge_tp = signal.take_profit
        if edge_tp is None:
            mid = signal.features.get("bb_mid")
            if mid is not None:
                edge_tp = float(mid)
        edge = assess_entry_edge(
            price=price,
            take_profit=edge_tp,
            fee_bps=self.cfg.execution.fee_bps,
            slippage_bps=self.cfg.execution.slippage_bps,
            hard_multiple=self.cfg.execution.hard_min_edge_multiple,
            soft_multiple=self.cfg.execution.soft_min_edge_multiple,
        )
        signal.features["edge_bps"] = edge.edge_bps
        signal.features["round_trip_cost_bps"] = edge.round_trip_cost_bps
        signal.features["edge_ratio"] = edge.ratio
        signal.features["edge_proxy"] = "bb_mid" if signal.take_profit is None else "take_profit"
        if edge.hard_reject:
            self.db.insert_rejected(
                RejectedSignal(
                    symbol=symbol,
                    venue=venue,
                    side=signal.side,
                    strategy=signal.strategy,
                    confidence=signal.confidence,
                    reason=edge.reason,
                    features=signal.features,
                    regime=signal.regime,
                    timestamp=datetime.now(timezone.utc),
                )
            )
            return
        if edge.soft_penalty:
            signal.confidence = max(
                0.0,
                signal.confidence - self.cfg.execution.soft_edge_confidence_penalty,
            )
            signal.features["thin_edge"] = True

        p_win = self.model.predict_proba(signal.features, signal.confidence)
        signal.features["p_win"] = p_win
        signal.confidence = 0.7 * signal.confidence + 0.3 * (p_win * 100)

        # Confirmed patterns: win=boost only; loss=penalty/soft-reject
        signal, reject_reason = self.learning.apply_confidence_effects(signal)
        if reject_reason:
            self.db.insert_rejected(
                RejectedSignal(
                    symbol=symbol,
                    venue=venue,
                    side=signal.side,
                    strategy=signal.strategy,
                    confidence=signal.confidence,
                    reason=reject_reason,
                    features=signal.features,
                    regime=signal.regime,
                    timestamp=datetime.now(timezone.utc),
                )
            )
            return

        signal = self.learning.tag_signal(signal)

        decision = self.risk.approve(
            signal,
            equity=self.portfolio.equity({symbol: float(df["close"].iloc[-1])}),
            open_positions=self.portfolio.open_positions(),
            mode=self.cfg.mode,
        )
        if not decision.allowed:
            self.db.insert_rejected(
                RejectedSignal(
                    symbol=symbol,
                    venue=venue,
                    side=signal.side,
                    strategy=signal.strategy,
                    confidence=signal.confidence,
                    reason=decision.reason,
                    features=signal.features,
                    regime=signal.regime,
                    timestamp=datetime.now(timezone.utc),
                )
            )
            return

        # Ensure cash can cover size (refill may have just run)
        if self.portfolio.cash < decision.size:
            self.portfolio.maybe_refill(
                auto_refill=self.cfg.capital_policy.auto_refill,
                refill_to=self.cfg.capital_policy.refill_to or self.cfg.capital.initial,
                min_trade_size=self.cfg.capital.base_trade_size,
                mark_prices={symbol: float(df["close"].iloc[-1])},
            )
            if self.portfolio.cash < decision.size:
                self.db.insert_rejected(
                    RejectedSignal(
                        symbol=symbol,
                        venue=venue,
                        side=signal.side,
                        strategy=signal.strategy,
                        confidence=signal.confidence,
                        reason="insufficient_cash",
                        features=signal.features,
                        regime=signal.regime,
                        timestamp=datetime.now(timezone.utc),
                    )
                )
                return

        self.executor.open_trade(signal, decision.size, price)

    def _build_snapshot(
        self, mark_prices: dict[str, float], venue_health: dict[str, Any]
    ) -> PortfolioSnapshot:
        m = self.portfolio.metrics()
        progress = self.learning.phase_progress(m["total_trades"])
        return PortfolioSnapshot(
            equity=self.portfolio.equity(mark_prices),
            cash=self.portfolio.cash,
            open_positions=len(self.portfolio.open_positions()),
            realized_pnl=self.portfolio.realized_pnl(),
            unrealized_pnl=self.portfolio.unrealized_pnl(mark_prices),
            win_rate=m["win_rate"],
            expectancy=m["expectancy"],
            profit_factor=m["profit_factor"],
            drawdown=m["drawdown"],
            total_trades=m["total_trades"],
            exploration_ratio=self.learning.exploration_ratio,
            learning_phase=progress["suggested_phase"],
            kill_switch=self.risk.kill_switch,
            venues=venue_health,
            timestamp=datetime.now(timezone.utc),
        )

    def force_daily_report(self, day: str | None = None) -> str:
        path = write_daily_report(self.db, learning=self.learning, day=day)
        self._last_daily_report_path = str(path)
        return str(path)

    def get_monitor_payload(self) -> dict[str, Any]:
        snap = self._last_snapshot
        if snap is None:
            try:
                snap = self.tick()
            except Exception as e:
                return {"error": str(e), "status": self.status_message}

        open_pos = [p.model_dump(mode="json") for p in self.portfolio.open_positions()]
        from trading_system.learning import learning_display

        closed = []
        for p in self.db.get_closed_trades(30):
            row = p.model_dump(mode="json")
            row.update(learning_display(p))
            closed.append(row)
        rejected = self.db.recent_rejected(20)
        insights = self.db.recent_insights(15)
        rankings = self.db.get_strategy_stats()
        latest_report = self.db.latest_daily_report()
        patterns_win = self.db.get_patterns("win")
        patterns_loss = self.db.get_patterns("loss")
        from trading_system.learning.sessions import session_info

        sess = session_info(buckets=self.cfg.learning.session_buckets)
        learning_payload = self.learning.phase_progress(snap.total_trades if snap else 0)
        learning_payload["session"] = sess
        return {
            "snapshot": snap.model_dump(mode="json"),
            "open_trades": open_pos,
            "closed_trades": closed,
            "rejected_signals": rejected,
            "insights": insights,
            "strategy_rankings": rankings,
            "model": {
                "backend": self.model.backend,
                "brier": self.model.brier,
            },
            "status": self.status_message,
            "kill_reason": self.risk.kill_reason,
            "cycle": self._cycle,
            "mode": self.cfg.mode,
            "learning": learning_payload,
            "session": sess,
            "capital_resets": self.portfolio.capital_resets(),
            "daily_report": latest_report,
            "patterns": {
                "wins": patterns_win[:10],
                "losses": patterns_loss[:10],
                "threshold": self.cfg.learning.pattern_min_occurrences,
            },
        }


_ENGINE: TradingEngine | None = None


def get_engine(simulate: bool = False) -> TradingEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TradingEngine(simulate=simulate)
    return _ENGINE
