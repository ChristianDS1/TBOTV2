"""Strategy engine — BB mean reversion + pluggable base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_system.config import StrategyConfig
from trading_system.features import build_features, detect_regime, latest_feature_dict
from trading_system.types import MarketRegime, Side, Signal, Venue


class Strategy(ABC):
    name: str
    expected_holding_minutes: int = 5

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        venue: Venue,
        df: pd.DataFrame,
        cfg: StrategyConfig,
    ) -> Signal | None:
        ...


def _near_extreme(row: pd.Series, side: Side, max_retrace: float) -> bool:
    """True if close is still in the early zone from the extreme toward mid."""
    upper = float(row["bb_upper"])
    lower = float(row["bb_lower"])
    close = float(row["close"])
    width = upper - lower
    if width <= 0 or pd.isna(width):
        return False
    if side == Side.CALL:
        # From lower band upward: early rejection = still near lower
        return (close - lower) / width <= max_retrace
    # PUT: from upper band downward
    return (upper - close) / width <= max_retrace


def compute_early_rejection_tp(
    *,
    side: Side,
    price: float,
    bb_lower: float,
    bb_mid: float,
    bb_upper: float,
    cfg: StrategyConfig,
) -> float:
    """
    Short TP for early rejection scalp — NEVER BB mid.
    Default: move = max(tp_band_fraction * band_width, tp_min_bps of price) toward mid,
    then clamp so TP stays short of the midline. Also used for fee-edge checks at entry.
    """
    width = max(0.0, bb_upper - bb_lower)
    min_move = price * float(cfg.tp_min_bps) / 10_000.0
    mode = (cfg.tp_mode or "band_fraction").lower()
    if mode == "fixed_bps":
        move = price * float(cfg.tp_fixed_bps) / 10_000.0
    else:
        move = max(width * float(cfg.tp_band_fraction), min_move)

    # Keep target in the early half toward mid (never at/beyond mid)
    if side == Side.CALL:
        room = max(0.0, bb_mid - price)
        if room > 0:
            move = min(move, room * 0.85)
        return price + max(move, min_move * 0.5)
    room = max(0.0, price - bb_mid)
    if room > 0:
        move = min(move, room * 0.85)
    return price - max(move, min_move * 0.5)


def compute_tight_stop_loss(
    *,
    side: Side,
    price: float,
    bb_lower: float,
    bb_upper: float,
    cfg: StrategyConfig,
) -> float:
    """Tight stop to cut losers before time_stop balloons the loss."""
    width = max(0.0, bb_upper - bb_lower)
    min_move = price * float(cfg.sl_min_bps) / 10_000.0
    move = max(width * float(cfg.sl_band_fraction), min_move)
    if side == Side.CALL:
        return price - move
    return price + move


class BBMeanReversionStrategy(Strategy):
    """Expansion → overextension → mean reversion (Estrategia.txt)."""

    name = "bb_mean_reversion"
    expected_holding_minutes = 5

    def evaluate(
        self,
        symbol: str,
        venue: Venue,
        df: pd.DataFrame,
        cfg: StrategyConfig,
    ) -> Signal | None:
        if len(df) < max(cfg.bb_period, 30) + 5:
            return None

        feat = build_features(
            df,
            bb_period=cfg.bb_period,
            bb_std=cfg.bb_std,
            rsi_period=cfg.rsi_period,
            macd_fast=cfg.macd_fast,
            macd_slow=cfg.macd_slow,
        )
        row = feat.iloc[-1]
        if pd.isna(row.get("rsi")) or pd.isna(row.get("bb_mid")):
            return None

        regime = detect_regime(feat)
        features = latest_feature_dict(feat)
        now = datetime.now(timezone.utc)
        if hasattr(row.get("timestamp"), "to_pydatetime"):
            now = row["timestamp"].to_pydatetime()

        call_conds = [
            bool(row["touch_lower"]),
            float(row["rsi"]) < cfg.rsi_oversold,
            bool(row["macd_fast_bull_cross"])
            or (
                float(row["macd_fast_hist"]) > float(row["macd_fast_hist_prev"])
                if not pd.isna(row["macd_fast_hist_prev"])
                else False
            ),
            bool(row["macd_slow_weakening_bear"]) or float(row.get("macd_slow_hist", 0) or 0) < 0,
            bool(row["rejection_bull"]),
        ]
        put_conds = [
            bool(row["touch_upper"]),
            float(row["rsi"]) > cfg.rsi_overbought,
            bool(row["macd_fast_bear_cross"])
            or (
                float(row["macd_fast_hist"]) < float(row["macd_fast_hist_prev"])
                if not pd.isna(row["macd_fast_hist_prev"])
                else False
            ),
            bool(row["macd_slow_weakening_bull"]) or float(row.get("macd_slow_hist", 0) or 0) > 0,
            bool(row["rejection_bear"]),
        ]

        call_n = sum(call_conds)
        put_n = sum(put_conds)
        min_n = cfg.min_conditions

        # Soft filters reduce confidence rather than hard-block in discovery
        soft_penalty = 0.0
        rsi_v = float(row["rsi"])
        if 40 <= rsi_v <= 60:
            soft_penalty += 15
        if bool(row.get("mid_bands")) and not (
            bool(row["touch_lower"]) or bool(row["touch_upper"])
        ):
            soft_penalty += 20
        bb_w = row.get("bb_width")
        if bb_w is not None and not pd.isna(bb_w) and float(bb_w) > 0.05:
            soft_penalty += 15

        side: Side | None = None
        met = 0
        reasons: list[str] = []

        if call_n >= min_n and call_n >= put_n:
            side = Side.CALL
            met = call_n
            reasons = [
                "touch_lower",
                "rsi_oversold",
                "macd_fast_turn_up",
                "macd_slow_weak_bear",
                "rejection_bull",
            ]
            reasons = [r for r, ok in zip(reasons, call_conds) if ok]
        elif put_n >= min_n:
            side = Side.PUT
            met = put_n
            reasons = [
                "touch_upper",
                "rsi_overbought",
                "macd_fast_turn_down",
                "macd_slow_weak_bull",
                "rejection_bear",
            ]
            reasons = [r for r, ok in zip(reasons, put_conds) if ok]
        else:
            return None

        # Hard gate: rejection candle required for extreme-entry style
        if cfg.require_rejection_candle:
            if side == Side.CALL and not bool(row["rejection_bull"]):
                return None
            if side == Side.PUT and not bool(row["rejection_bear"]):
                return None

        # Too far from extreme toward mid → late entry, skip
        max_retrace = float(cfg.max_extreme_retrace_pct)
        features["extreme_retrace_pct"] = None
        upper = float(row["bb_upper"])
        lower = float(row["bb_lower"])
        width = upper - lower
        if width > 0:
            if side == Side.CALL:
                features["extreme_retrace_pct"] = (float(row["close"]) - lower) / width
            else:
                features["extreme_retrace_pct"] = (upper - float(row["close"])) / width
        if not _near_extreme(row, side, max_retrace):
            return None

        confidence = max(0.0, min(100.0, (met / 5.0) * 100.0 - soft_penalty))
        if cfg.discovery_phase is False and confidence < cfg.entry_confidence_floor:
            return None

        price = float(row["close"])
        tp = compute_early_rejection_tp(
            side=side,
            price=price,
            bb_lower=lower,
            bb_mid=float(row["bb_mid"]),
            bb_upper=upper,
            cfg=cfg,
        )
        sl = compute_tight_stop_loss(
            side=side,
            price=price,
            bb_lower=lower,
            bb_upper=upper,
            cfg=cfg,
        )
        features["tp_mode"] = cfg.tp_mode
        features["tp_band_fraction"] = float(cfg.tp_band_fraction)
        features["sl_band_fraction"] = float(cfg.sl_band_fraction)
        features["bb_mid"] = float(row["bb_mid"])
        features["take_profit"] = tp
        features["stop_loss"] = sl
        return Signal(
            symbol=symbol,
            venue=venue,
            side=side,
            strategy=self.name,
            confidence=confidence,
            reason="; ".join(reasons),
            features=features,
            regime=regime if isinstance(regime, MarketRegime) else MarketRegime.UNKNOWN,
            expected_holding_minutes=cfg.max_hold_minutes,
            take_profit=tp,
            stop_loss=sl,
            timestamp=now if isinstance(now, datetime) else datetime.now(timezone.utc),
            conditions_met=met,
        )


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, Strategy] = {}
        self.register(BBMeanReversionStrategy())

    def register(self, strategy: Strategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Strategy:
        return self._strategies[name]

    def all(self) -> list[Strategy]:
        return list(self._strategies.values())
