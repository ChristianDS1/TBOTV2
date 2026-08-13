"""Risk manager — soft paper mode + technical kill switch."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from trading_system.config import AppConfig
from trading_system.types import Position, Signal, Venue


CORRELATED_GROUPS = {
    "crypto_majors": {"BTC/USDT", "ETH/USDT"},
    "fx_usd": {"EUR/USD", "GBP/USD", "AUD/USD"},
}


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    size: float = 0.0


@dataclass
class RiskManager:
    cfg: AppConfig
    kill_switch: bool = False
    kill_reason: str = ""
    last_data_ok: dict[str, datetime] = field(default_factory=dict)

    def trip(self, reason: str) -> None:
        self.kill_switch = True
        self.kill_reason = reason

    def reset_kill(self) -> None:
        self.kill_switch = False
        self.kill_reason = ""

    def mark_data(self, venue: str, ok: bool) -> None:
        if ok:
            self.last_data_ok[venue] = datetime.now(timezone.utc)
        elif self.cfg.risk.kill_on_api_failure:
            self.trip(f"api_failure:{venue}")

    def check_stale(self) -> None:
        max_age = self.cfg.risk.stale_data_seconds
        now = datetime.now(timezone.utc)
        for venue, ts in self.last_data_ok.items():
            if (now - ts).total_seconds() > max_age:
                self.trip(f"stale_data:{venue}")

    def trade_size(self, equity: float) -> float:
        base = self.cfg.capital.base_trade_size
        step = self.cfg.capital.size_step_per_50_equity
        initial = self.cfg.capital.initial
        # Scale UP with equity only — never drop below base (avoids margin 9 when paper < €100)
        bumps = int(max(0, equity - initial) // 50)
        size = base + bumps * step
        return max(float(base), round(size, 2))

    def correlated_count(self, open_positions: list[Position], symbol: str) -> int:
        group = None
        for g, members in CORRELATED_GROUPS.items():
            if symbol in members:
                group = members
                break
        if not group:
            return 0
        return sum(1 for p in open_positions if p.symbol in group)

    def approve(
        self,
        signal: Signal,
        equity: float,
        open_positions: list[Position],
        mode: str,
    ) -> RiskDecision:
        if mode.lower() == "live" and not self.cfg.is_live:
            return RiskDecision(False, "live_not_enabled")
        if self.cfg.is_live:
            # Extra safety — plan says live disabled initially
            return RiskDecision(False, "live_blocked_by_policy")

        if self.kill_switch:
            return RiskDecision(False, f"kill_switch:{self.kill_reason}")

        if len(open_positions) >= self.cfg.risk.max_simultaneous_positions:
            return RiskDecision(False, "max_positions")

        if any(p.symbol == signal.symbol for p in open_positions):
            return RiskDecision(False, "already_in_symbol")

        corr = self.correlated_count(open_positions, signal.symbol)
        if corr >= self.cfg.risk.max_correlated_exposure:
            return RiskDecision(False, "correlated_exposure")

        size = self.trade_size(equity)
        if equity > 0 and size > equity * 0.5:
            # After refill equity is healthy; only block absurd sizing
            return RiskDecision(False, "size_anomaly")
        if equity <= 0:
            return RiskDecision(False, "equity_depleted_awaiting_refill")

        # Soft mode: no daily loss kill — exploration continues
        return RiskDecision(True, "ok", size=size)
