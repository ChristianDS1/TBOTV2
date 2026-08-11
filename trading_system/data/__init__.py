"""Market data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from trading_system.config import AppConfig, ForexSessionConfig
    from trading_system.types import Venue


class MarketAdapter(ABC):
    venue: "Venue"

    @abstractmethod
    def get_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 200
    ) -> pd.DataFrame:
        """Return DataFrame with columns: timestamp, open, high, low, close, volume."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def is_tradable_now(self, symbol: str, ts: datetime | None = None) -> bool:
        ...

    @abstractmethod
    def health(self) -> dict:
        ...


class SessionCalendar:
    """Forex session gate: open Mon–Fri within configured UTC hours; closed weekends."""

    def __init__(self, cfg: "ForexSessionConfig") -> None:
        self.cfg = cfg

    def is_open(self, ts: datetime | None = None) -> bool:
        now = ts or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        wd = now.weekday()  # Mon=0 .. Sun=6
        hour = now.hour + now.minute / 60.0

        open_wd = self.cfg.open_weekday
        close_wd = self.cfg.close_weekday
        open_h = float(self.cfg.open_hour_utc)
        close_h = float(self.cfg.close_hour_utc)

        # Weekend closed
        if wd > close_wd or wd < open_wd:
            return False
        if wd == open_wd and hour < open_h:
            return False
        if wd == close_wd and hour >= close_h:
            return False
        return True

    def status(self, ts: datetime | None = None) -> dict:
        open_now = self.is_open(ts)
        return {"forex_session_open": open_now, "reason": "open" if open_now else "closed"}
