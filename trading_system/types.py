"""Shared domain types."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Venue(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"


class Side(str, Enum):
    CALL = "call"  # long / buy
    PUT = "put"  # short / sell


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class SignalAction(str, Enum):
    ENTER = "enter"
    REJECT = "reject"
    EXIT = "exit"


class MarketRegime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    BREAKOUT = "breakout"
    COMPRESSION = "compression"
    MEAN_REVERSION = "mean_reversion"
    UNKNOWN = "unknown"


class LearningPhase(str, Enum):
    DISCOVERY = "discovery"
    PATTERN = "pattern"
    OPTIMIZATION = "optimization"
    EXPLOITATION = "exploitation"


class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class Signal(BaseModel):
    symbol: str
    venue: Venue
    side: Side
    strategy: str
    confidence: float
    reason: str
    features: dict[str, Any] = Field(default_factory=dict)
    regime: MarketRegime = MarketRegime.UNKNOWN
    expected_holding_minutes: int = 5
    take_profit: float | None = None
    stop_loss: float | None = None
    timestamp: datetime
    exploration: bool = False
    conditions_met: int = 0


class Position(BaseModel):
    id: int | None = None
    symbol: str
    venue: Venue
    side: Side
    strategy: str
    qty: float
    entry_price: float
    entry_time: datetime
    take_profit: float | None = None
    stop_loss: float | None = None
    confidence: float = 0.0
    regime: str = "unknown"
    features_json: str = "{}"
    exploration: bool = False
    status: TradeStatus = TradeStatus.OPEN
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl: float | None = None
    fees: float = 0.0
    exit_reason: str | None = None


class RejectedSignal(BaseModel):
    symbol: str
    venue: Venue
    side: Side | None
    strategy: str
    confidence: float
    reason: str
    features: dict[str, Any] = Field(default_factory=dict)
    regime: MarketRegime = MarketRegime.UNKNOWN
    timestamp: datetime


class PortfolioSnapshot(BaseModel):
    equity: float
    cash: float
    open_positions: int
    realized_pnl: float
    unrealized_pnl: float
    win_rate: float
    expectancy: float
    profit_factor: float
    drawdown: float
    total_trades: int
    exploration_ratio: float
    learning_phase: str
    kill_switch: bool
    venues: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
