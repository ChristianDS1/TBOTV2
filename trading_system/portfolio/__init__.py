"""Paper portfolio and metrics + auto capital refill."""

from __future__ import annotations

from trading_system.database import Database
from trading_system.types import Position


class Portfolio:
    def __init__(self, db: Database, initial_capital: float) -> None:
        self.db = db
        self.initial_capital = initial_capital
        cash = db.get_state("cash")
        self.cash = float(cash) if cash is not None else initial_capital
        if cash is None:
            db.set_state("cash", str(self.cash))
            db.set_state("initial_capital", str(initial_capital))
        if db.get_state("capital_resets") is None:
            db.set_state("capital_resets", "0")

    def open_positions(self) -> list[Position]:
        return self.db.get_open_trades()

    def realized_pnl(self) -> float:
        closed = self.db.get_all_closed()
        return sum(p.pnl or 0.0 for p in closed)

    def equity(self, mark_prices: dict[str, float] | None = None) -> float:
        unrealized = self.unrealized_pnl(mark_prices or {})
        return self.cash + unrealized

    def unrealized_pnl(self, mark_prices: dict[str, float]) -> float:
        total = 0.0
        for p in self.open_positions():
            px = mark_prices.get(p.symbol)
            if px is None:
                continue
            notional = p.notional if p.notional and p.notional > 0 else p.qty * max(p.leverage or 1.0, 1.0)
            direction = 1 if p.side.value == "call" else -1
            total += direction * (px - p.entry_price) / p.entry_price * notional
        return total

    def metrics(self) -> dict:
        closed = self.db.get_all_closed()
        if not closed:
            return {
                "win_rate": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "drawdown": 0.0,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
            }
        pnls = [p.pnl or 0.0 for p in closed]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        expectancy = sum(pnls) / len(pnls)
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

        equity = self.initial_capital
        peak = equity
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak else 0
            max_dd = max(max_dd, dd)

        return {
            "win_rate": win_rate,
            "expectancy": expectancy,
            "profit_factor": pf,
            "drawdown": max_dd,
            "total_trades": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
        }

    def debit(self, amount: float) -> None:
        self.cash -= amount
        self.db.set_state("cash", str(self.cash))

    def credit(self, amount: float) -> None:
        self.cash += amount
        self.db.set_state("cash", str(self.cash))

    def capital_resets(self) -> int:
        return int(self.db.get_state("capital_resets") or "0")

    def maybe_refill(
        self,
        *,
        auto_refill: bool,
        refill_to: float,
        min_trade_size: float,
        mark_prices: dict[str, float] | None = None,
        refill_below: float = 30.0,
    ) -> bool:
        """
        If flat and cash/equity <= refill_below, top up to refill_to.
        Never stops trading. Returns True if a refill happened.
        """
        if not auto_refill:
            return False
        if self.open_positions():
            # Wait until flat before full reset
            return False

        eq = self.equity(mark_prices or {})
        floor = float(refill_below)
        # Also refill if cash cannot cover one trade (even above floor)
        needs = (
            self.cash <= floor
            or eq <= floor
            or self.cash < float(min_trade_size)
        )
        if not needs:
            return False

        self.cash = float(refill_to)
        self.db.set_state("cash", str(self.cash))
        resets = self.capital_resets() + 1
        self.db.set_state("capital_resets", str(resets))
        self.db.insert_insight(
            "capital_reset",
            (
                f"Paper capital ≤€{floor:.2f} — refilled to €{refill_to:.2f} "
                f"(reset #{resets}). Trading continues."
            ),
            {
                "refill_to": refill_to,
                "refill_below": floor,
                "resets": resets,
            },
        )
        return True

    def snapshot_equity(self, mark_prices: dict[str, float]) -> None:
        eq = self.equity(mark_prices)
        self.db.record_equity(eq, self.cash, len(self.open_positions()))
