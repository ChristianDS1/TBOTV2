"""Paper execution engine."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta

from trading_system.config import AppConfig
from trading_system.database import Database
from trading_system.portfolio import Portfolio
from trading_system.types import Position, Signal, TradeStatus

logger = logging.getLogger(__name__)


class PaperExecutor:
    def __init__(self, cfg: AppConfig, db: Database, portfolio: Portfolio) -> None:
        self.cfg = cfg
        self.db = db
        self.portfolio = portfolio

    def _cost_bps(self, notional: float) -> float:
        bps = self.cfg.execution.fee_bps + self.cfg.execution.slippage_bps
        return notional * bps / 10_000

    def open_trade(self, signal: Signal, size: float, price: float) -> Position:
        # Slippage adverse on fill; keep pre-slip mark for strategy gross PnL
        slip = self.cfg.execution.slippage_bps / 10_000
        fill = price * (1 + slip) if signal.side.value == "call" else price * (1 - slip)
        fees = self._cost_bps(size)
        self.portfolio.debit(size + fees)

        pos = Position(
            symbol=signal.symbol,
            venue=signal.venue,
            side=signal.side,
            strategy=signal.strategy,
            qty=size,
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
        )
        pos.id = self.db.insert_trade(pos)
        logger.info(
            "OPEN %s %s %s size=%.2f @ %.6f conf=%.1f",
            signal.side.value,
            signal.symbol,
            signal.strategy,
            size,
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

        # Strategy gross: mark-to-mark (no fees, no slip)
        entry_ref = pos.entry_mark if pos.entry_mark and pos.entry_mark > 0 else pos.entry_price
        gross = direction * (price - entry_ref) / entry_ref * pos.qty

        # Net cash impact on this close: slipped fill move minus exit fee
        # (entry fee already debited at open)
        raw_fill_pnl = direction * (fill - pos.entry_price) / pos.entry_price * pos.qty
        exit_fees = self._cost_bps(pos.qty)
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
        self.portfolio.credit(pos.qty + net)
        self.db.update_trade(pos)
        logger.info(
            "CLOSE %s %s gross=%.4f net=%.4f reason=%s cost_erosion=%s",
            pos.side.value,
            pos.symbol,
            gross,
            net,
            reason,
            cost_erosion,
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
        for pos in list(self.portfolio.open_positions()):
            px = mark_prices.get(pos.symbol)
            if px is None:
                continue

            max_hold = timedelta(minutes=self.cfg.strategy.max_hold_minutes)
            if now - pos.entry_time >= max_hold:
                closed.append(self.close_trade(pos, px, "time_stop"))
                continue

            if pos.take_profit is not None:
                if pos.side.value == "call" and px >= pos.take_profit:
                    closed.append(self.close_trade(pos, px, "take_profit"))
                    continue
                if pos.side.value == "put" and px <= pos.take_profit:
                    closed.append(self.close_trade(pos, px, "take_profit"))
                    continue

            if (
                close_fx_at_session_end
                and pos.venue.value == "forex"
                and not forex_session_open
            ):
                closed.append(self.close_trade(pos, px, "session_end"))
                continue

        return closed
