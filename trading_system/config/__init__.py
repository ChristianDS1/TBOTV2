"""Configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("default.yaml")


class CapitalConfig(BaseModel):
    initial: float = 100.0
    currency: str = "EUR"
    base_trade_size: float = 2.5
    size_step_per_50_equity: float = 1.0


class RiskConfig(BaseModel):
    max_simultaneous_positions: int = 5
    risk_per_trade_pct: float = 2.0
    soft_mode: bool = True
    max_correlated_exposure: int = 3
    stale_data_seconds: int = 120
    kill_on_api_failure: bool = True


class StrategyConfig(BaseModel):
    name: str = "bb_mean_reversion"
    min_conditions: int = 3
    require_rejection_candle: bool = True
    max_extreme_retrace_pct: float = 0.35
    tp_mode: str = "band_fraction"  # band_fraction | fixed_bps (never bb_mid)
    tp_band_fraction: float = 0.25
    tp_min_bps: float = 12.0
    tp_fixed_bps: float = 15.0
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 10
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    macd_fast: list[int] = Field(default_factory=lambda: [5, 8, 9])
    macd_slow: list[int] = Field(default_factory=lambda: [13, 21, 9])
    max_hold_minutes: int = 10
    discovery_phase: bool = True
    entry_confidence_floor: float = 0


class LearningConfig(BaseModel):
    phase: str = "discovery"
    exploration_budget: float = 0.25
    min_sample_size: int = 20
    pattern_min_occurrences: int = 20
    win_confidence_boost: float = 8.0
    loss_confidence_penalty: float = 15.0
    loss_soft_reject: bool = True
    soft_reject_exclude_key_prefixes: list[str] = Field(
        default_factory=lambda: ["strategy="]
    )
    retrain_every_n_trades: int = 50
    recency_half_life_trades: int = 100


class CapitalPolicyConfig(BaseModel):
    auto_refill: bool = True
    refill_to: float = 100.0


class ForexSessionConfig(BaseModel):
    open_weekday: int = 0
    open_hour_utc: int = 0
    close_weekday: int = 4
    close_hour_utc: int = 21
    close_intraday_at_session_end: bool = True


class ExecutionConfig(BaseModel):
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    poll_interval_seconds: int = 5
    leverage: float = 5.0
    liquidation_margin_fraction: float = 0.9
    hard_min_edge_multiple: float = 0.5
    soft_min_edge_multiple: float = 1.15
    soft_edge_confidence_penalty: float = 8.0
    tp_require_positive_net: bool = True
    # Legacy alias kept for older configs/tests
    tp_require_non_negative_net: bool = True


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    refresh_seconds: int = 5


class DatabaseConfig(BaseModel):
    path: str = "data/trading.db"


class SymbolsConfig(BaseModel):
    crypto: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    forex: list[str] = Field(default_factory=lambda: ["EUR/USD", "GBP/USD"])


class TimeframesConfig(BaseModel):
    primary: str = "1m"
    lookback_bars: int = 200


class CryptoExchangeConfig(BaseModel):
    exchange: str = "binance"
    sandbox: bool = False


class ForexProviderConfig(BaseModel):
    provider: str = "yfinance"


class AppConfig(BaseModel):
    mode: str = "paper"
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    capital_policy: CapitalPolicyConfig = Field(default_factory=CapitalPolicyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    symbols: SymbolsConfig = Field(default_factory=SymbolsConfig)
    timeframes: TimeframesConfig = Field(default_factory=TimeframesConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    forex_session: ForexSessionConfig = Field(default_factory=ForexSessionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    crypto: CryptoExchangeConfig = Field(default_factory=CryptoExchangeConfig)
    forex: ForexProviderConfig = Field(default_factory=ForexProviderConfig)

    @property
    def all_symbols(self) -> list[str]:
        return list(self.symbols.crypto) + list(self.symbols.forex)

    @property
    def is_live(self) -> bool:
        return self.mode.lower() == "live"

    def db_path(self) -> Path:
        p = Path(self.database.path)
        if not p.is_absolute():
            p = ROOT / p
        return p


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | str | None = None) -> AppConfig:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Env overrides
    if os.getenv("TRADING_MODE"):
        raw["mode"] = os.getenv("TRADING_MODE")
    if os.getenv("INITIAL_CAPITAL"):
        raw.setdefault("capital", {})["initial"] = float(os.getenv("INITIAL_CAPITAL"))
    if os.getenv("DATABASE_PATH"):
        raw.setdefault("database", {})["path"] = os.getenv("DATABASE_PATH")
    if os.getenv("CRYPTO_EXCHANGE"):
        raw.setdefault("crypto", {})["exchange"] = os.getenv("CRYPTO_EXCHANGE")
    if os.getenv("FOREX_DATA_PROVIDER"):
        raw.setdefault("forex", {})["provider"] = os.getenv("FOREX_DATA_PROVIDER")
    if os.getenv("DASHBOARD_HOST"):
        raw.setdefault("dashboard", {})["host"] = os.getenv("DASHBOARD_HOST")
    if os.getenv("DASHBOARD_PORT"):
        raw.setdefault("dashboard", {})["port"] = int(os.getenv("DASHBOARD_PORT"))

    return AppConfig.model_validate(raw)
