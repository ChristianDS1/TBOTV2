"""Main trading engine loop."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from trading_system.config import AppConfig, ROOT, load_config
from trading_system.data.crypto import CryptoAdapter, SimulatedCryptoAdapter
from trading_system.data.forex import ForexAdapter
from trading_system.database import Database
from trading_system.execution import PaperExecutor
from trading_system.execution.edge import assess_entry_edge
from trading_system.features import build_features, latest_feature_dict
from trading_system.learning import LearningEngine, daily_objective_progress
from trading_system.models import WinProbabilityModel
from trading_system.patterns import (
    combine_htf_votes,
    ltf_turn,
    macd_htf_bias,
    resample_ohlcv,
    scan_patterns,
)
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
        self._ohlcv_cache: dict[str, Any] = {}  # symbol -> DataFrame (1m)
        self._tf_cache: dict[tuple[str, str], tuple[float, Any]] = {}

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
                ctx = self._market_context(sym, Venue.CRYPTO, df)
                row = self._feature_row(df)
                row.update(self._context_row(ctx))
                feature_rows[sym] = row
                self._ohlcv_cache[sym] = df
                self._process_symbol(sym, Venue.CRYPTO, df, ctx)
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
                ctx = self._market_context(sym, Venue.FOREX, df)
                row = self._feature_row(df)
                row.update(self._context_row(ctx))
                feature_rows[sym] = row
                self._ohlcv_cache[sym] = df
                if self.forex.is_tradable_now(sym):
                    self._process_symbol(sym, Venue.FOREX, df, ctx)
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

    def _context_row(self, ctx: dict[str, Any]) -> dict[str, Any]:
        pats = ctx.get("patterns") or []
        htf_pats = ctx.get("htf_patterns") or []
        ltf_pats = ctx.get("ltf_patterns") or []
        top = max(pats, key=lambda p: p.confidence) if pats else None
        htf_top = max(htf_pats, key=lambda p: p.confidence) if htf_pats else None
        bear = (
            any(getattr(p, "direction", "") == "bearish" for p in pats)
            or any(getattr(p, "direction", "") == "bearish" for p in htf_pats)
            or any(getattr(p, "direction", "") == "bearish" for p in ltf_pats)
        )
        bull = (
            any(getattr(p, "direction", "") == "bullish" for p in pats)
            or any(getattr(p, "direction", "") == "bullish" for p in htf_pats)
            or any(getattr(p, "direction", "") == "bullish" for p in ltf_pats)
        )
        return {
            "htf_bias": ctx.get("htf_bias") or "unknown",
            "ltf_turn": ctx.get("ltf_turn"),
            "chart_reversal_bear": bear,
            "chart_reversal_bull": bull,
            "chart_pattern": getattr(top, "name", None) if top else (
                getattr(htf_top, "name", None) if htf_top else None
            ),
        }

    def _fetch_tf(self, symbol: str, venue: Venue, tf: str, limit: int):
        key = (symbol, tf)
        now = time.monotonic()
        ttl = 5.0 if tf in ("1m", "15s", "30s", "1s") else 45.0
        hit = self._tf_cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        df = None
        try:
            adapter = self.forex if venue == Venue.FOREX else self.crypto
            if tf in ("15s", "30s"):
                raw = adapter.get_ohlcv(symbol, "1s", max(int(limit) * 30, 180))
                df = resample_ohlcv(raw, tf)
            else:
                df = adapter.get_ohlcv(symbol, tf, limit)
            if df is not None and getattr(df, "empty", False):
                df = None
        except Exception as e:
            logger.debug("ohlcv %s %s failed: %s", symbol, tf, e)
            df = None
        self._tf_cache[key] = (now, df)
        return df

    def _market_context(self, symbol: str, venue: Venue, df_1m) -> dict[str, Any]:
        votes: dict[str, str] = {}
        htf_patterns: list = []
        for tf in self.cfg.timeframes.confirm or []:
            d = self._fetch_tf(symbol, venue, tf, 80)
            votes[tf] = macd_htf_bias(d)
            if d is not None and len(d) >= 30:
                for p in scan_patterns(d):
                    p.details["tf"] = tf
                    htf_patterns.append(p)
        bias = combine_htf_votes(votes)
        patterns = scan_patterns(df_1m)
        ltf = None
        ltf_patterns: list = []
        if venue != Venue.FOREX:
            for tf in self.cfg.timeframes.anticipate or []:
                d = self._fetch_tf(symbol, venue, tf, 60)
                if d is None or len(d) < 8:
                    continue
                turn = ltf_turn(d)
                if turn and ltf is None:
                    ltf = turn
                if len(d) >= 30:
                    for p in scan_patterns(d):
                        p.details["tf"] = tf
                        ltf_patterns.append(p)
        return {
            "htf_bias": bias,
            "htf_votes": votes,
            "patterns": patterns,
            "htf_patterns": htf_patterns,
            "ltf_patterns": ltf_patterns,
            "ltf_turn": ltf,
        }

    def _process_symbol(self, symbol: str, venue: Venue, df, context: dict[str, Any] | None = None) -> None:
        ctx = context or {}
        preferred = self.cfg.strategy.name
        order = []
        try:
            order.append(self.strategies.get(preferred))
        except KeyError:
            pass
        for strat in self.strategies.all():
            if strat.name != preferred:
                order.append(strat)
        for strat in order:
            signal = strat.evaluate(symbol, venue, df, self.cfg.strategy, context=ctx)
            if signal is None:
                continue
            if self._try_enter(signal, symbol, venue, df):
                return

    def _try_enter(self, signal, symbol: str, venue: Venue, df) -> bool:
        price = float(df["close"].iloc[-1])
        margin = float(self.cfg.capital.base_trade_size)
        leverage = max(1.0, float(self.cfg.execution.leverage or 1.0))
        fee_bps, slip_bps = self.cfg.execution.costs_for_venue(venue)
        exit_fee_bps = (
            fee_bps + slip_bps
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

        # Soft fee-aware edge: BB uses mid; pattern/continuation use measure-rule (or skip gate)
        edge_tp = signal.take_profit
        proxy = "take_profit"
        if edge_tp is None:
            mt = signal.features.get("measure_target") or signal.features.get("edge_target")
            if mt is not None:
                edge_tp = float(mt)
                proxy = "measure_target"
            elif signal.strategy == "bb_mean_reversion":
                mid = signal.features.get("bb_mid")
                if mid is not None:
                    edge_tp = float(mid)
                    proxy = "bb_mid"
            else:
                proxy = "no_tp"
        edge = assess_entry_edge(
            price=price,
            take_profit=edge_tp,
            fee_bps=fee_bps,
            slippage_bps=slip_bps,
            hard_multiple=self.cfg.execution.hard_min_edge_multiple,
            soft_multiple=self.cfg.execution.soft_min_edge_multiple,
        )
        signal.features["edge_bps"] = edge.edge_bps
        signal.features["round_trip_cost_bps"] = edge.round_trip_cost_bps
        signal.features["edge_ratio"] = edge.ratio
        signal.features["edge_proxy"] = proxy
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
            return False
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
            return False

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
            return False

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
                return False

        self.executor.open_trade(signal, decision.size, price)
        return True

    def _build_snapshot(
        self, mark_prices: dict[str, float], venue_health: dict[str, Any]
    ) -> PortfolioSnapshot:
        m = self.portfolio.metrics()
        inv = self.db.trade_inventory()
        progress = self.learning.phase_progress(inv["closed"])
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
            total_trades=inv["closed"],
            last_trade_id=inv["last_id"],
            db_trade_rows=inv["rows"],
            exploration_ratio=self.learning.exploration_ratio,
            learning_phase=progress["suggested_phase"],
            kill_switch=self.risk.kill_switch,
            venues=venue_health,
            timestamp=datetime.now(timezone.utc),
        )

    def _fetch_ohlcv_df(self, symbol: str, venue: str, limit: int | None = None):
        import pandas as pd

        lim = int(limit or self.cfg.timeframes.lookback_bars)
        cached = self._ohlcv_cache.get(symbol)
        if cached is not None and len(cached) >= min(lim, 30):
            return cached.tail(lim).reset_index(drop=True)

        tf = self.cfg.timeframes.primary
        if venue == "forex" or symbol in self.cfg.symbols.forex:
            df = self.forex.get_ohlcv(symbol, tf, lim)
        else:
            df = self.crypto.get_ohlcv(symbol, tf, lim)
        self._ohlcv_cache[symbol] = df
        return df.tail(lim).reset_index(drop=True) if hasattr(df, "tail") else df

    @staticmethod
    def _bar_time_unix(ts) -> int:
        import pandas as pd

        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return int(t.timestamp())

    def _chart_from_df(
        self,
        df,
        *,
        markers: list[dict[str, Any]] | None = None,
        symbol: str = "",
        venue: str = "",
    ) -> dict[str, Any]:
        cfg = self.cfg.strategy
        feat = build_features(
            df,
            bb_period=cfg.bb_period,
            bb_std=cfg.bb_std,
            rsi_period=cfg.rsi_period,
            macd_fast=cfg.macd_fast,
            macd_slow=cfg.macd_slow,
        )
        candles: list[dict[str, Any]] = []
        bb: list[dict[str, Any]] = []
        for _, row in feat.iterrows():
            t = self._bar_time_unix(row["timestamp"])
            candles.append(
                {
                    "time": t,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
            mid = row.get("bb_mid")
            up = row.get("bb_upper")
            lo = row.get("bb_lower")
            if mid is not None and up is not None and lo is not None:
                import pandas as pd

                if not (pd.isna(mid) or pd.isna(up) or pd.isna(lo)):
                    bb.append(
                        {
                            "time": t,
                            "mid": float(mid),
                            "upper": float(up),
                            "lower": float(lo),
                        }
                    )
        payload = {
            "symbol": symbol,
            "venue": venue,
            "candles": candles,
            "bb": bb,
            "markers": markers or [],
        }
        if candles:
            times = [c["time"] for c in candles]
            snapped = []
            for m in payload["markers"]:
                t = int(m["time"])
                # snap to nearest candle time (lightweight-charts requirement)
                nearest = min(times, key=lambda x: abs(x - t))
                mm = dict(m)
                mm["time"] = nearest
                snapped.append(mm)
            payload["markers"] = snapped
        return payload

    def get_chart(
        self,
        symbol: str,
        venue: str = "crypto",
        limit: int = 120,
        trade_id: int | None = None,
    ) -> dict[str, Any]:
        df = self._fetch_ohlcv_df(symbol, venue, limit=max(limit, 60))
        markers: list[dict[str, Any]] = []
        if trade_id is not None:
            pos = self.db.get_trade(int(trade_id))
            if pos is not None:
                markers.extend(self._trade_markers(pos))
        else:
            for pos in self.portfolio.open_positions():
                if pos.symbol == symbol:
                    markers.extend(self._trade_markers(pos, exit_included=False))
        return self._chart_from_df(
            df.tail(limit).reset_index(drop=True),
            markers=markers,
            symbol=symbol,
            venue=venue,
        )

    def get_trade_chart(self, trade_id: int, pad_bars: int = 30) -> dict[str, Any]:
        pos = self.db.get_trade(int(trade_id))
        if pos is None:
            return {"error": "trade_not_found", "candles": [], "bb": [], "markers": []}
        venue = pos.venue.value if hasattr(pos.venue, "value") else str(pos.venue)
        df = self._fetch_ohlcv_df(pos.symbol, venue, limit=self.cfg.timeframes.lookback_bars)
        import pandas as pd

        entry_ts = pd.Timestamp(pos.entry_time)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        exit_ts = pd.Timestamp(pos.exit_time) if pos.exit_time else pd.Timestamp.now(tz="UTC")
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize("UTC")

        ts = pd.to_datetime(df["timestamp"], utc=True)
        start = entry_ts - pd.Timedelta(minutes=pad_bars)
        end = exit_ts + pd.Timedelta(minutes=pad_bars)
        mask = (ts >= start) & (ts <= end)
        window = df.loc[mask].reset_index(drop=True)
        if window.empty:
            window = df.tail(max(pad_bars * 2, 60)).reset_index(drop=True)
        return self._chart_from_df(
            window,
            markers=self._trade_markers(pos, exit_included=True),
            symbol=pos.symbol,
            venue=venue,
        )

    def _trade_markers(
        self, pos, *, exit_included: bool = True
    ) -> list[dict[str, Any]]:
        is_call = (pos.side.value if hasattr(pos.side, "value") else str(pos.side)).lower() == "call"
        color = "#3ecf8e" if is_call else "#e85d5d"
        markers = [
            {
                "time": self._bar_time_unix(pos.entry_time),
                "position": "belowBar" if is_call else "aboveBar",
                "shape": "arrowUp" if is_call else "arrowDown",
                "color": color,
                "text": f"IN {pos.side.value if hasattr(pos.side, 'value') else pos.side}",
                "price": float(pos.entry_price),
            }
        ]
        if exit_included and pos.exit_time is not None:
            markers.append(
                {
                    "time": self._bar_time_unix(pos.exit_time),
                    "position": "aboveBar" if is_call else "belowBar",
                    "shape": "circle",
                    "color": "#e6b35a",
                    "text": f"OUT {pos.exit_reason or ''}",
                    "price": float(pos.exit_price) if pos.exit_price is not None else None,
                }
            )
        if pos.stop_loss is not None:
            markers.append(
                {
                    "time": self._bar_time_unix(pos.entry_time),
                    "position": "aboveBar",
                    "shape": "square",
                    "color": "#8fa3b8",
                    "text": "SL",
                    "price": float(pos.stop_loss),
                    "line": True,
                }
            )
        return markers

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
        obj_cfg = self.cfg.objective
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_eq = self.db.first_equity_on_day(day)
        if start_eq is None:
            start_eq = float(self.cfg.capital.initial)
        objective = daily_objective_progress(
            start_equity=start_eq,
            current_equity=float(snap.equity) if snap else start_eq,
            target_pct=obj_cfg.daily_equity_gain_pct,
            phase=learning_payload.get("suggested_phase") or obj_cfg.name,
            chase_in_discovery=obj_cfg.chase_target_in_discovery,
            name=obj_cfg.name,
        )
        learning_payload["objective"] = objective
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
            "objective": objective,
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
