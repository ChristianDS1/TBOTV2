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


def compute_sl_from_margin_pct(
    *,
    side: Side,
    price: float,
    margin: float,
    leverage: float,
    sl_margin_pct: float = 4.0,
    exit_fee_bps: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    SL sized so max NET loss ≈ sl_margin_pct of margin.

    Returns (stop_price, budget_bps_on_notional, trigger_bps, budget_cash).
    """
    margin = max(1e-9, float(margin))
    lev = max(1.0, float(leverage))
    notional = margin * lev
    budget_cash = margin * float(sl_margin_pct) / 100.0
    budget_bps = budget_cash / notional * 10_000.0 if notional > 0 else 0.0
    fee = max(0.0, float(exit_fee_bps))
    trigger_bps = max(1.0, budget_bps - fee) if fee > 0 else max(1.0, budget_bps)
    move = price * trigger_bps / 10_000.0
    if side == Side.CALL:
        return price - move, budget_bps, trigger_bps, budget_cash
    return price + move, budget_bps, trigger_bps, budget_cash


def compute_sl_from_tp_rr(
    *,
    side: Side,
    price: float,
    take_profit: float,
    exit_fee_bps: float = 0.0,
    reward_multiple: float = 1.5,
) -> tuple[float, float, float, float]:
    """
    SL sized so NET reward:risk = reward_multiple : 1 (default 1.5:1).

    Fees on both sides (exit fee, matching close PnL):
      TP_net = tp_move_bps - exit_fee_bps
      SL_net = TP_net / reward_multiple
      trigger_bps = max(1, SL_net - exit_fee_bps)

    Returns (stop_price, sl_budget_bps=SL_net, trigger_bps, tp_net_bps).
    """
    if price <= 0 or take_profit is None:
        raise ValueError("price and take_profit required for rr_from_tp SL")
    fee = max(0.0, float(exit_fee_bps))
    mult = max(0.1, float(reward_multiple))
    tp_move_bps = abs(float(take_profit) - price) / price * 10_000.0
    # Floor so a tiny TP still yields a usable stop
    tp_net_bps = max(0.5, tp_move_bps - fee)
    sl_net_bps = tp_net_bps / mult
    trigger_bps = max(1.0, sl_net_bps - fee) if fee > 0 else max(1.0, sl_net_bps)
    move = price * trigger_bps / 10_000.0
    if side == Side.CALL:
        return price - move, sl_net_bps, trigger_bps, tp_net_bps
    return price + move, sl_net_bps, trigger_bps, tp_net_bps


def compute_tight_stop_loss(
    *,
    side: Side,
    price: float,
    bb_lower: float,
    bb_upper: float,
    cfg: StrategyConfig,
    exit_fee_bps: float = 0.0,
    take_profit: float | None = None,
    margin: float | None = None,
    leverage: float | None = None,
) -> tuple[float, float, float]:
    """
    Stop loss. Returns (stop_price, budget_bps, trigger_bps).

    sl_mode=margin_pct: NET loss cap as % of margin.
    sl_mode=rr_from_tp: derive from TP.
    sl_mode=band: legacy band/min budget.
    """
    fee = (
        float(exit_fee_bps)
        if getattr(cfg, "sl_include_exit_fees", True)
        else 0.0
    )
    mode = (getattr(cfg, "sl_mode", "margin_pct") or "margin_pct").lower()
    if mode == "margin_pct":
        m = float(margin) if margin is not None else 10.0
        lev = float(leverage) if leverage is not None else 20.0
        sl, budget_bps, trigger_bps, _cash = compute_sl_from_margin_pct(
            side=side,
            price=price,
            margin=m,
            leverage=lev,
            sl_margin_pct=float(getattr(cfg, "sl_margin_pct", 4.0)),
            exit_fee_bps=fee,
        )
        return sl, budget_bps, trigger_bps

    if mode == "rr_from_tp" and take_profit is not None:
        sl, budget_bps, trigger_bps, _tp_net = compute_sl_from_tp_rr(
            side=side,
            price=price,
            take_profit=float(take_profit),
            exit_fee_bps=fee,
            reward_multiple=float(getattr(cfg, "tp_rr_multiple", 1.5)),
        )
        return sl, budget_bps, trigger_bps

    width = max(0.0, bb_upper - bb_lower)
    min_move = price * float(cfg.sl_min_bps) / 10_000.0
    budget_move = max(width * float(cfg.sl_band_fraction), min_move)
    budget_bps = budget_move / price * 10_000.0 if price > 0 else float(cfg.sl_min_bps)

    trigger_bps = budget_bps
    if fee > 0:
        trigger_bps = max(1.0, budget_bps - fee)
    move = price * trigger_bps / 10_000.0

    if side == Side.CALL:
        return price - move, budget_bps, trigger_bps
    return price + move, budget_bps, trigger_bps


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
        tp_mode = (cfg.tp_mode or "trend_fade").lower()
        tp: float | None
        if tp_mode == "trend_fade":
            tp = None
        else:
            tp = compute_early_rejection_tp(
                side=side,
                price=price,
                bb_lower=lower,
                bb_mid=float(row["bb_mid"]),
                bb_upper=upper,
                cfg=cfg,
            )
        # Placeholder SL — engine recomputes with margin/leverage + exit fees
        sl, sl_budget_bps, sl_trigger_bps = compute_tight_stop_loss(
            side=side,
            price=price,
            bb_lower=lower,
            bb_upper=upper,
            cfg=cfg,
            exit_fee_bps=0.0,
            take_profit=tp,
            margin=10.0,
            leverage=20.0,
        )
        features["tp_mode"] = tp_mode
        features["tp_band_fraction"] = float(cfg.tp_band_fraction)
        features["sl_mode"] = getattr(cfg, "sl_mode", "margin_pct")
        features["sl_margin_pct"] = float(getattr(cfg, "sl_margin_pct", 4.0))
        features["tp_rr_multiple"] = float(getattr(cfg, "tp_rr_multiple", 1.5))
        features["sl_band_fraction"] = float(cfg.sl_band_fraction)
        features["sl_budget_bps"] = sl_budget_bps
        features["sl_trigger_bps"] = sl_trigger_bps
        features["bb_mid"] = float(row["bb_mid"])
        features["take_profit"] = tp
        features["stop_loss"] = sl
        features["trend_fade_min_score"] = int(
            getattr(cfg, "trend_fade_min_score", 2)
        )
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
