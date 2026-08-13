"""Adaptive exit engine (Gen-5) — MFE/giveback, weakening vs reversal vs stale.

Does not change entry logic. Stop-loss remains emergency protection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from trading_system.config import ExitConfig
from trading_system.execution.edge import (
    estimate_close_net,
    score_trend_fade,
    unrealized_pnl_on_notional,
)
from trading_system.types import Position, Side

logger = logging.getLogger(__name__)


@dataclass
class ExitDecision:
    reason: str | None = None
    state: str = "hold"  # hold | weakening | reversal | stale
    score: int = 0
    components: list[str] = field(default_factory=list)
    thresholds_used: dict[str, Any] = field(default_factory=dict)
    snapshot: dict[str, Any] = field(default_factory=dict)


def _side_str(side: Side | str) -> str:
    return side.value if isinstance(side, Side) else str(side).lower()


def _is_call(side: Side | str) -> bool:
    return _side_str(side) == "call"


def _favorable_move_pct(side: Side | str, entry: float, mark: float) -> float:
    if entry <= 0:
        return 0.0
    if _is_call(side):
        return (mark - entry) / entry * 100.0
    return (entry - mark) / entry * 100.0


def update_excursion_state(
    feat: dict[str, Any],
    pos: Position,
    mark: float,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Mutate/return features with MFE/MAE/peak/giveback tracking."""
    now = now or datetime.now(timezone.utc)
    entry = float(pos.entry_mark or pos.entry_price)
    move_pct = _favorable_move_pct(pos.side, entry, mark)
    u_pnl = unrealized_pnl_on_notional(pos, mark)

    peak_price = feat.get("peak_price")
    if peak_price is None:
        peak_price = entry
    peak_price = float(peak_price)

    if _is_call(pos.side):
        if mark >= float(peak_price):
            peak_price = mark
            feat["last_favorable_extreme_ts"] = now.isoformat()
    else:
        if mark <= float(peak_price):
            peak_price = mark
            feat["last_favorable_extreme_ts"] = now.isoformat()

    if feat.get("last_favorable_extreme_ts") is None:
        feat["last_favorable_extreme_ts"] = (
            pos.entry_time.isoformat() if pos.entry_time else now.isoformat()
        )

    mfe_pct = float(feat.get("mfe_pct") or 0.0)
    mae_pct = float(feat.get("mae_pct") or 0.0)
    mfe_pct = max(mfe_pct, move_pct)
    mae_pct = min(mae_pct, move_pct)  # most adverse (negative) move

    peak_pnl = float(feat.get("peak_pnl") or 0.0)
    peak_pnl = max(peak_pnl, u_pnl)

    giveback = 0.0
    if mfe_pct > 1e-9:
        giveback = max(0.0, (mfe_pct - move_pct) / mfe_pct)

    last_ext = feat.get("last_favorable_extreme_ts")
    try:
        last_dt = datetime.fromisoformat(str(last_ext).replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        mins_since = (now - last_dt).total_seconds() / 60.0
    except Exception:
        mins_since = 0.0

    held_min = 0.0
    if pos.entry_time:
        et = pos.entry_time
        if et.tzinfo is None:
            et = et.replace(tzinfo=timezone.utc)
        held_min = (now - et).total_seconds() / 60.0

    feat["mfe_pct"] = mfe_pct
    feat["mae_pct"] = mae_pct
    feat["peak_price"] = peak_price
    feat["peak_pnl"] = peak_pnl
    feat["giveback_pct"] = giveback
    feat["current_pnl"] = u_pnl
    feat["current_move_pct"] = move_pct
    feat["minutes_since_last_favorable_extreme"] = mins_since
    feat["trade_duration_minutes"] = held_min
    return feat


def _required_fade_score(
    cfg: ExitConfig,
    *,
    net_est: float,
    mfe_pct: float,
    giveback: float,
) -> int:
    if (
        mfe_pct >= float(cfg.min_mfe_pct_for_protection)
        and giveback >= float(cfg.giveback_protect_pct)
    ):
        return int(cfg.fade_score_with_mfe_giveback)
    if net_est > 0:
        return int(cfg.fade_score_in_profit)
    return int(cfg.fade_score_flat_or_loss)


def decide_exit(
    pos: Position,
    mark: float,
    row: dict[str, Any],
    feat: dict[str, Any],
    cfg: ExitConfig,
    *,
    fee_bps: float,
    slip_bps: float,
    min_hold_minutes: float = 1.0,
    now: datetime | None = None,
) -> ExitDecision:
    """Return adaptive exit decision. Caller applies SL before this."""
    now = now or datetime.now(timezone.utc)
    feat = update_excursion_state(feat, pos, mark, now=now)
    score, components = score_trend_fade(pos.side, row)
    net_est = estimate_close_net(pos, mark, fee_bps, slip_bps)

    mfe_pct = float(feat.get("mfe_pct") or 0.0)
    giveback = float(feat.get("giveback_pct") or 0.0)
    held = float(feat.get("trade_duration_minutes") or 0.0)
    mins_since = float(feat.get("minutes_since_last_favorable_extreme") or 0.0)
    move_pct = float(feat.get("current_move_pct") or 0.0)

    weakening = score >= 1
    has_chart_rev = "chart_reversal" in components
    need = _required_fade_score(cfg, net_est=net_est, mfe_pct=mfe_pct, giveback=giveback)
    thresholds = {
        "required_score": need,
        "mfe_pct": mfe_pct,
        "giveback_pct": giveback,
        "net_est": net_est,
        "held_min": held,
        "mins_since_extreme": mins_since,
    }

    state = "hold"
    if weakening and score < need:
        state = "weakening"
    if score >= need or (has_chart_rev and giveback >= float(cfg.giveback_reversal_pct)):
        state = "reversal"

    feat["trend_fade_score"] = score
    feat["trend_fade_components"] = components
    feat["weakening_state"] = weakening
    feat["reversal_state"] = state == "reversal"
    feat["htf_bias"] = row.get("htf_bias")
    feat["ltf_turn"] = row.get("ltf_turn")

    snapshot = {
        "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
        "entry_price": pos.entry_price,
        "entry_mark": pos.entry_mark,
        "side": _side_str(pos.side),
        "current_price": mark,
        "current_pnl": feat.get("current_pnl"),
        "mfe_pct": mfe_pct,
        "mae_pct": feat.get("mae_pct"),
        "peak_price": feat.get("peak_price"),
        "peak_pnl": feat.get("peak_pnl"),
        "giveback_pct": giveback,
        "trade_duration_minutes": held,
        "minutes_since_last_favorable_extreme": mins_since,
        "trend_fade_score": score,
        "trend_fade_components": components,
        "weakening_state": weakening,
        "reversal_state": state == "reversal",
        "htf_bias": row.get("htf_bias"),
        "ltf_turn": row.get("ltf_turn"),
        "net_est": net_est,
        "thresholds": thresholds,
    }
    feat["exit_eval"] = snapshot

    # Progressing favorably → never force adaptive exit (except hard stale handled below)
    making_progress = mins_since < float(cfg.stale_progress_minutes) and move_pct >= mfe_pct - 1e-9

    if held < float(min_hold_minutes):
        return ExitDecision(
            reason=None,
            state=state if state != "hold" else "hold",
            score=score,
            components=components,
            thresholds_used=thresholds,
            snapshot=snapshot,
        )

    # Hard stale > max: no recent favorable extreme
    if held >= float(cfg.stale_position_max_minutes):
        if mins_since >= float(cfg.stale_progress_minutes):
            snapshot["exit_reason"] = "stale_position"
            return ExitDecision(
                reason="stale_position",
                state="stale",
                score=score,
                components=components,
                thresholds_used=thresholds,
                snapshot=snapshot,
            )

    # Soft stale 30–60: no progress + not strongly favorable
    if held >= float(cfg.stale_soft_minutes):
        strongly_favorable = move_pct > 0 and score == 0
        if (
            mins_since >= float(cfg.stale_progress_minutes)
            and not strongly_favorable
            and not making_progress
        ):
            snapshot["exit_reason"] = "stale_position"
            return ExitDecision(
                reason="stale_position",
                state="stale",
                score=score,
                components=components,
                thresholds_used=thresholds,
                snapshot=snapshot,
            )

    # Profit protection: significant MFE + giveback + weakening
    if (
        mfe_pct >= float(cfg.min_mfe_pct_for_protection)
        and giveback >= float(cfg.giveback_protect_pct)
        and weakening
        and score >= int(cfg.fade_score_with_mfe_giveback)
    ):
        # Strong evidence → reversal; mild evidence → protect peak profit
        if score >= int(cfg.fade_score_in_profit) or has_chart_rev:
            reason = "trend_reversal"
        else:
            reason = "profit_protection"
        snapshot["exit_reason"] = reason
        return ExitDecision(
            reason=reason,
            state="reversal" if reason == "trend_reversal" else "weakening",
            score=score,
            components=components,
            thresholds_used=thresholds,
            snapshot=snapshot,
        )

    # Pure reversal path (adaptive score) — may exit even if net slipped from peak
    if state == "reversal" and score >= need:
        # Flat/loss with weak MFE already baked into higher `need`
        snapshot["exit_reason"] = "trend_reversal"
        return ExitDecision(
            reason="trend_reversal",
            state="reversal",
            score=score,
            components=components,
            thresholds_used=thresholds,
            snapshot=snapshot,
        )

    # High giveback + chart reversal confirmation
    if (
        mfe_pct >= float(cfg.min_mfe_pct_for_protection)
        and giveback >= float(cfg.giveback_reversal_pct)
        and has_chart_rev
    ):
        snapshot["exit_reason"] = "trend_reversal"
        return ExitDecision(
            reason="trend_reversal",
            state="reversal",
            score=score,
            components=components,
            thresholds_used=thresholds,
            snapshot=snapshot,
        )

    return ExitDecision(
        reason=None,
        state=state,
        score=score,
        components=components,
        thresholds_used=thresholds,
        snapshot=snapshot,
    )


def maybe_log_exit_eval(
    pos: Position,
    decision: ExitDecision,
    *,
    last_log_ts: dict[int, float],
    log_every_seconds: float,
    monotonic_now: float,
) -> None:
    tid = int(pos.id or 0)
    prev = last_log_ts.get(tid, 0.0)
    if decision.reason is None and (monotonic_now - prev) < float(log_every_seconds):
        return
    last_log_ts[tid] = monotonic_now
    snap = decision.snapshot
    logger.info(
        "EXIT_EVAL id=%s %s %s state=%s score=%s comps=%s mfe=%.3f%% giveback=%.2f "
        "held=%.1fm since_ext=%.1fm net_est=%.4f reason=%s",
        pos.id,
        _side_str(pos.side),
        pos.symbol,
        decision.state,
        decision.score,
        decision.components,
        float(snap.get("mfe_pct") or 0),
        float(snap.get("giveback_pct") or 0),
        float(snap.get("trade_duration_minutes") or 0),
        float(snap.get("minutes_since_last_favorable_extreme") or 0),
        float(snap.get("net_est") or 0),
        decision.reason,
    )
