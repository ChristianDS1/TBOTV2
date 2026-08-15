"""Configuration loading."""

from __future__ import annotations

import os
from datetime import datetime
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
    base_trade_size: float = 10.0
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
    # Forex majors move less — slightly softer entry (rejection candle still required)
    forex_min_conditions: int = 2
    # Soft discovery: lower bars when phase=discovery (indicators still required)
    discovery_min_conditions: int = 2
    discovery_forex_min_conditions: int = 1
    require_rejection_candle: bool = True
    max_extreme_retrace_pct: float = 0.35
    forex_max_extreme_retrace_pct: float = 0.45
    # trend_fade = no fixed TP (exit on momentum/trend fade); legacy: band_fraction | fixed_bps
    tp_mode: str = "trend_fade"
    tp_band_fraction: float = 0.25
    tp_min_bps: float = 12.0
    tp_fixed_bps: float = 15.0
    # SL: margin_pct (default) | rr_from_tp | band
    sl_mode: str = "margin_pct"
    sl_margin_pct: float = 4.0  # max NET loss as % of margin
    tp_rr_multiple: float = 1.5  # legacy rr_from_tp
    sl_band_fraction: float = 0.12
    sl_min_bps: float = 10.0
    sl_include_exit_fees: bool = True
    trend_fade_min_score: int = 2
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 10
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    macd_fast: list[int] = Field(default_factory=lambda: [5, 8, 9])
    macd_slow: list[int] = Field(default_factory=lambda: [13, 21, 9])
    min_hold_minutes: int = 1
    preferred_hold_minutes: int = 3
    max_hold_minutes: int = 10
    discovery_phase: bool = True
    entry_confidence_floor: float = 0


class SessionBucketConfig(BaseModel):
    name: str
    start_hour_utc: float  # inclusive
    end_hour_utc: float  # exclusive (use 24 for end-of-day)


class LearningConfig(BaseModel):
    phase: str = "discovery"
    exploration_budget: float = 0.45  # soft discovery: try more
    min_sample_size: int = 20
    pattern_min_occurrences: int = 20
    win_confidence_boost: float = 8.0
    loss_confidence_penalty: float = 15.0
    loss_soft_reject: bool = False  # soft discovery: don't starve exploration
    soft_reject_exclude_key_prefixes: list[str] = Field(
        default_factory=lambda: ["strategy=", "regime="]
    )
    # Pattern evidence and confidence effects scoped by UTC session buckets
    session_aware: bool = True
    session_buckets: list[SessionBucketConfig] = Field(
        default_factory=lambda: [
            SessionBucketConfig(name="asia", start_hour_utc=0.0, end_hour_utc=7.0),
            SessionBucketConfig(name="europe", start_hour_utc=7.0, end_hour_utc=12.0),
            SessionBucketConfig(name="us_open", start_hour_utc=12.0, end_hour_utc=16.0),
            SessionBucketConfig(name="us_afternoon", start_hour_utc=16.0, end_hour_utc=21.0),
            SessionBucketConfig(name="night", start_hour_utc=21.0, end_hour_utc=24.0),
        ]
    )
    # Sat/Sun UTC → session=weekend (FX closed; crypto book)
    weekend_session_enabled: bool = True
    priority_boost: float = 25.0  # obligatory priority setups
    priority_min_net_wr: float = 0.90
    priority_min_n: int = 10
    discovery_skip_hard_edge: bool = True  # outside priority: don't hard-reject on thin edge
    retrain_every_n_trades: int = 50
    recency_half_life_trades: int = 100


class ObjectiveConfig(BaseModel):
    """North-star goal. Discovery learns; exploitation aims at daily equity target."""

    name: str = "maximize_net_equity"
    daily_equity_gain_pct: float = 50.0
    chase_target_in_discovery: bool = False


class CapitalPolicyConfig(BaseModel):
    auto_refill: bool = True
    refill_to: float = 100.0


class ForexSessionConfig(BaseModel):
    open_weekday: int = 0
    open_hour_utc: int = 0
    close_weekday: int = 4
    close_hour_utc: int = 21
    close_intraday_at_session_end: bool = True


class ExitConfig(BaseModel):
    """Adaptive exit engine (Gen-5). Does not change entry logic."""

    stale_position_max_minutes: float = 60.0
    stale_soft_minutes: float = 30.0
    stale_progress_minutes: float = 10.0
    min_mfe_pct_for_protection: float = 0.30  # price %
    giveback_protect_pct: float = 0.35
    giveback_reversal_pct: float = 0.45
    fade_score_in_profit: int = 2
    fade_score_flat_or_loss: int = 3
    fade_score_with_mfe_giveback: int = 1
    log_every_seconds: float = 60.0
    # Profit-lock: only trend_reversal when net>0; classify reversal vs continuation
    trend_reversal_require_net_profit: bool = True
    limbo_flat_max_minutes: float = 10.0
    continuation_hold: bool = True
    # Soft floor for aggressive lock when pattern ambiguous (fraction of margin)
    min_lock_net_margin_pct: float = 0.15
    # After peak_pnl>0: lock on soft momentum/reversal clues even if net_est<=0
    lock_after_peak: bool = True
    peak_lock_min_clues: int = 2
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


class ExecutionConfig(BaseModel):
    # Crypto (Binance paper) — unused on weekend when weekend_use_forex_costs
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    # Pepperstone Razor FX proxy: ~$3.50/side/lot ≈ 0.35 bps + raw spread
    forex_fee_bps: float = 0.35
    forex_slippage_bps: float = 1.0
    broker: str = "pepperstone"
    weekend_use_forex_costs: bool = True  # weekend crypto practices Pepperstone FX fees
    # Sat/Sun paper FX OTC off — weekend is crypto-only (weekday FX unchanged)
    weekend_forex_otc: bool = False
    poll_interval_seconds: int = 5
    leverage: float = 50.0
    liquidation_margin_fraction: float = 0.9
    hard_min_edge_multiple: float = 0.5
    soft_min_edge_multiple: float = 1.15
    soft_edge_confidence_penalty: float = 8.0
    tp_require_positive_net: bool = True
    # Legacy alias kept for older configs/tests
    tp_require_non_negative_net: bool = True

    def costs_for_venue(
        self,
        venue: str | Any,
        *,
        as_of: datetime | None = None,
    ) -> tuple[float, float]:
        """Return (fee_bps, slippage_bps). Weekend crypto uses Pepperstone FX schedule."""
        from trading_system.learning.sessions import is_weekend_utc

        v = venue.value if hasattr(venue, "value") else str(venue)
        use_fx = v.lower() == "forex"
        if bool(self.weekend_use_forex_costs) and is_weekend_utc(as_of):
            use_fx = True
        if use_fx:
            return float(self.forex_fee_bps), float(self.forex_slippage_bps)
        return float(self.fee_bps), float(self.slippage_bps)


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
    confirm: list[str] = Field(default_factory=lambda: ["15m", "30m", "1h"])
    anticipate: list[str] = Field(default_factory=lambda: ["30s", "15s"])


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
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    forex_session: ForexSessionConfig = Field(default_factory=ForexSessionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)
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
