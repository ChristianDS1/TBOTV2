"""Paper execution engine — margin + leverage model (fees on notional)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from trading_system.config import AppConfig
from trading_system.database import Database
from trading_system.execution.edge import (
    can_take_profit_net_positive,
    position_notional,
    should_adaptive_time_stop,
    unrealized_pnl_on_notional,
)
from trading_system.portfolio import Portfolio
from trading_system.types import Position, Signal, TradeStatus

logger = logging.getLogger(__name__)


class PaperExecutor:
    def __init__(self, cfg: AppConfig, db: Database, portfolio: Portfolio) -> None:
        self.cfg = cfg
        self.db = db
        self.portfolio = portfolio

    def _leverage(self) -> float:
        return max(1.0, float(self.cfg.execution.leverage or 1.0))

    def _cost_on_notional(self, notional: float) -> float:
        bps = self.cfg.execution.fee_bps + self.cfg.execution.slippage_bps
        return notional * bps / 10_000

    def open_trade(self, signal: Signal, size: float, price: float) -> Position:
        # size = margin; notional = margin * leverage (perp-style paper)
        leverage = self._leverage()
        margin = float(size)
        notional = margin * leverage
        slip = self.cfg.execution.slippage_bps / 10_000
        fill = price * (1 + slip) if signal.side.value == "call" else price * (1 - slip)
        fees = self._cost_on_notional(notional)
        # Lock margin + pay entry fee from cash
        self.portfolio.debit(margin + fees)

        pos = Position(
            symbol=signal.symbol,
            venue=signal.venue,
            side=signal.side,
            strategy=signal.strategy,
            qty=margin,
            entry_price=fill,
            entry_mark=price,
            entry_time=datetime.now(timezone.utc),
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            confidence=signal.confidence,
            regime=signal.regime.value,
            features_json=json.dumps(signal.features),
            exploration=signal.exploration,
            status=TradeStatus.OPEN,
            fees=fees,
            leverage=leverage,
            notional=notional,
        )
        pos.id = self.db.insert_trade(pos)
        logger.info(
            "OPEN %s %s %s margin=%.2f notional=%.2f lev=%.1fx @ %.6f conf=%.1f",
            signal.side.value,
            signal.symbol,
            signal.strategy,
            margin,
            notional,
            leverage,
            fill,
            signal.confidence,
        )
        return pos

    def close_trade(
        self, pos: Position, price: float, reason: str
    ) -> Position:
        slip = self.cfg.execution.slippage_bps / 10_000
        fill = price * (1 - slip) if pos.side.value == "call" else price * (1 + slip)
        direction = 1 if pos.side.value == "call" else -1
        notional = position_notional(pos)
        margin = float(pos.qty)

        # Strategy gross: mark-to-mark on notional (no fees, no slip)
        entry_ref = pos.entry_mark if pos.entry_mark and pos.entry_mark > 0 else pos.entry_price
        gross = direction * (price - entry_ref) / entry_ref * notional

        # Net cash: leveraged move on fill minus exit fee (entry fee already debited)
        raw_fill_pnl = direction * (fill - pos.entry_price) / pos.entry_price * notional
        exit_fees = self._cost_on_notional(notional)
        net = raw_fill_pnl - exit_fees

        # Cost erosion: strategy achieved target / gross positive but net red
        strategy_ok = reason == "take_profit" or gross > 0
        cost_erosion = bool(strategy_ok and net <= 0)

        pos.exit_price = fill
        pos.exit_time = datetime.now(timezone.utc)
        pos.gross_pnl = gross
        pos.pnl = net
        pos.fees = (pos.fees or 0) + exit_fees
        pos.exit_reason = reason
        pos.cost_erosion = cost_erosion
        pos.status = TradeStatus.CLOSED
        # Return margin + net PnL to cash
        self.portfolio.credit(margin + net)
        self.db.update_trade(pos)
        logger.info(
            "CLOSE %s %s gross=%.4f net=%.4f reason=%s cost_erosion=%s lev=%.1fx",
            pos.side.value,
            pos.symbol,
            gross,
            net,
            reason,
            cost_erosion,
            pos.leverage or 1.0,
        )
        return pos

    def manage_open(
        self,
        mark_prices: dict[str, float],
        forex_session_open: bool,
        close_fx_at_session_end: bool,
    ) -> list[Position]:
        closed: list[Position] = []
        now = datetime.now(timezone.utc)
        liq_frac = float(self.cfg.execution.liquidation_margin_fraction or 0.9)
        require_pos_net = bool(
            getattr(self.cfg.execution, "tp_require_positive_net", True)
            or getattr(self.cfg.execution, "tp_require_non_negative_net", True)
        )

        for pos in list(self.portfolio.open_positions()):
            px = mark_prices.get(pos.symbol)
            if px is None:
                continue

            # Soft liquidation: unrealized loss eats most of margin
            u_pnl = unrealized_pnl_on_notional(pos, px)
            if u_pnl <= -abs(pos.qty) * liq_frac:
                closed.append(self.close_trade(pos, px, "liquidation"))
                continue

            # Tight stop-loss (cut losers before time_stop balloons R:R)
            if pos.stop_loss is not None:
                hit_sl = (
                    (pos.side.value == "call" and px <= pos.stop_loss)
                    or (pos.side.value == "put" and px >= pos.stop_loss)
                )
                if hit_sl:
                    closed.append(self.close_trade(pos, px, "stop_loss"))
                    continue

            max_hold = float(self.cfg.strategy.max_hold_minutes)
            pref_hold = float(
                getattr(self.cfg.strategy, "preferred_hold_minutes", 3) or 3
            )
            min_hold = float(getattr(self.cfg.strategy, "min_hold_minutes", 1) or 1)
            held_min = (now - pos.entry_time).total_seconds() / 60.0
            if should_adaptive_time_stop(
                pos,
                px,
                held_min,
                min_hold_minutes=min_hold,
                preferred_hold_minutes=pref_hold,
                max_hold_minutes=max_hold,
            ):
                closed.append(self.close_trade(pos, px, "time_stop"))
                continue

            if pos.take_profit is not None:
                hit_tp = (
                    (pos.side.value == "call" and px >= pos.take_profit)
                    or (pos.side.value == "put" and px <= pos.take_profit)
                )
                if hit_tp:
                    ok_net = can_take_profit_net_positive(
                        pos,
                        px,
                        self.cfg.execution.fee_bps,
                        self.cfg.execution.slippage_bps,
                    )
                    if (not require_pos_net) or ok_net:
                        closed.append(self.close_trade(pos, px, "take_profit"))
                        continue
                    # Hit early-rejection TP but net would be <= 0 — hold for better or time_stop
                    logger.debug(
                        "TP deferred (net not > 0) %s %s mark=%.6f",
                        pos.side.value,
                        pos.symbol,
                        px,
                    )

            if (
                close_fx_at_session_end
                and pos.venue.value == "forex"
                and not forex_session_open
            ):
                closed.append(self.close_trade(pos, px, "session_end"))
                continue

        return closed
