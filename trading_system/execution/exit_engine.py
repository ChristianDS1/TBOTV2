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
from trading_system.patterns.coverage import classify_exit_pattern_context
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


def _peak_lock_clues(
    side: Side | str,
    row: dict[str, Any],
    components: list[str],
    pattern_class: str,
    cfg: ExitConfig,
) -> tuple[int, list[str], bool]:
    """
    Soft momentum / reversal clues after a profitable peak.

    Returns (count, clue_names, hardish) where hardish means prefer trend_reversal
    (hard pattern / chart / rejection) over profit_protection.
    """
    clues: list[str] = []
    hardish = False
    is_call = _is_call(side)
    comps = set(components)

    if pattern_class == "hard_reversal" or "chart_reversal" in comps:
        clues.append("hard_or_chart_reversal")
        hardish = True
    if is_call and bool(row.get("rejection_bear")):
        clues.append("rejection_bear")
        hardish = True
    elif (not is_call) and bool(row.get("rejection_bull")):
        clues.append("rejection_bull")
        hardish = True
    if "macd_fast_fade" in comps:
        clues.append("macd_fast_fade")
    if "macd_slow_fade" in comps:
        clues.append("macd_slow_fade")
    if "rsi_rollover" in comps:
        clues.append("rsi_rollover")

    rsi_v = row.get("rsi")
    try:
        rsi_f = float(rsi_v) if rsi_v is not None else None
    except (TypeError, ValueError):
        rsi_f = None
    ob = float(getattr(cfg, "rsi_overbought", 70.0) or 70.0)
    os_ = float(getattr(cfg, "rsi_oversold", 30.0) or 30.0)
    if rsi_f is not None:
        if is_call and rsi_f >= ob:
            clues.append("rsi_overbought")
        elif (not is_call) and rsi_f <= os_:
            clues.append("rsi_oversold")

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for c in clues:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return len(uniq), uniq, hardish


def _try_peak_lock(
    *,
    peak_pnl: float,
    cfg: ExitConfig,
    side: Side | str,
    row: dict[str, Any],
    components: list[str],
    pattern_class: str,
    score: int,
    thresholds: dict[str, Any],
    snapshot: dict[str, Any],
) -> ExitDecision | None:
    if not bool(getattr(cfg, "lock_after_peak", True)):
        return None
    if peak_pnl <= 0:
        return None
    n_clues, clue_names, hardish = _peak_lock_clues(
        side, row, components, pattern_class, cfg
    )
    min_clues = int(getattr(cfg, "peak_lock_min_clues", 2) or 2)
    thresholds["peak_lock_clues"] = clue_names
    thresholds["peak_lock_count"] = n_clues
    thresholds["peak_pnl"] = peak_pnl
    snapshot["peak_lock_clues"] = clue_names
    snapshot["peak_lock_count"] = n_clues
    if n_clues < min_clues:
        return None
    reason = "trend_reversal" if hardish else "profit_protection"
    snapshot["exit_reason"] = reason
    return ExitDecision(
        reason=reason,
        state="reversal" if reason == "trend_reversal" else "weakening",
        score=score,
        components=components,
        thresholds_used=thresholds,
        snapshot=snapshot,
    )


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

    pattern_class = classify_exit_pattern_context(
        side=pos.side,
        htf=str(row.get("htf_bias") or "unknown"),
        row=row,
    )
    feat["exit_pattern_class"] = pattern_class
    thresholds["pattern_class"] = pattern_class
    thresholds["require_net_profit"] = bool(
        getattr(cfg, "trend_reversal_require_net_profit", True)
    )

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
        "exit_pattern_class": pattern_class,
        "htf_bias": row.get("htf_bias"),
        "ltf_turn": row.get("ltf_turn"),
        "net_est": net_est,
        "thresholds": thresholds,
    }
    feat["exit_eval"] = snapshot

    # Progressing favorably → never force adaptive exit (except hard stale handled below)
    making_progress = mins_since < float(cfg.stale_progress_minutes) and move_pct >= mfe_pct - 1e-9
    peak_pnl = float(feat.get("peak_pnl") or 0.0)
    ever_net_profit = peak_pnl > 0 or net_est > 0
    require_net = bool(getattr(cfg, "trend_reversal_require_net_profit", True))
    continuation_hold = bool(getattr(cfg, "continuation_hold", True))
    limbo_max = float(getattr(cfg, "limbo_flat_max_minutes", 10.0) or 10.0)
    min_lock = float(getattr(cfg, "min_lock_net_margin_pct", 0.15) or 0.0)
    margin = abs(float(pos.qty or 0.0))
    worth_lock = net_est > 0 and (
        pattern_class == "hard_reversal"
        or min_lock <= 0
        or net_est
        >= (margin * min_lock if min_lock < 1.0 else margin * min_lock / 100.0)
    )

    def _hold(state_out: str = "hold") -> ExitDecision:
        return ExitDecision(
            reason=None,
            state=state_out,
            score=score,
            components=components,
            thresholds_used=thresholds,
            snapshot=snapshot,
        )

    if held < float(min_hold_minutes):
        return _hold(state if state != "hold" else "hold")

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

    # Peak lock: ever green + soft clues → exit even if current net_est <= 0
    # (overrides continuation_hold and the net>0 gate below)
    peak_lock = _try_peak_lock(
        peak_pnl=peak_pnl,
        cfg=cfg,
        side=pos.side,
        row=row,
        components=components,
        pattern_class=pattern_class,
        score=score,
        thresholds=thresholds,
        snapshot=snapshot,
    )
    if peak_lock is not None:
        return peak_lock

    # Flat/loss never-profit: wait SL or limbo timeout
    if require_net and net_est <= 0:
        if held >= limbo_max and not ever_net_profit:
            snapshot["exit_reason"] = "limbo_timeout"
            return ExitDecision(
                reason="limbo_timeout",
                state="stale",
                score=score,
                components=components,
                thresholds_used=thresholds,
                snapshot=snapshot,
            )
        return _hold("weakening" if weakening else "hold")

    # --- In net profit ---
    # Continuation forming → let it run (peak-lock already handled above)
    if continuation_hold and pattern_class == "continuation":
        # Still protect if giveback is severe after real MFE
        if (
            mfe_pct >= float(cfg.min_mfe_pct_for_protection)
            and giveback >= float(cfg.giveback_reversal_pct)
            and weakening
            and score >= int(cfg.fade_score_in_profit)
        ):
            snapshot["exit_reason"] = "profit_protection"
            return ExitDecision(
                reason="profit_protection",
                state="weakening",
                score=score,
                components=components,
                thresholds_used=thresholds,
                snapshot=snapshot,
            )
        return _hold("hold")

    # Hard reversal against the side → lock profit now
    if pattern_class == "hard_reversal" and (
        state == "reversal" or score >= need or has_chart_rev or weakening
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

    # Profit protection: significant MFE + giveback + weakening (not continuation)
    if (
        mfe_pct >= float(cfg.min_mfe_pct_for_protection)
        and giveback >= float(cfg.giveback_protect_pct)
        and weakening
        and score >= int(cfg.fade_score_with_mfe_giveback)
    ):
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

    # Ambiguous: only exit if fade is strong AND profit is worth locking
    if state == "reversal" and score >= need and worth_lock:
        snapshot["exit_reason"] = "trend_reversal"
        return ExitDecision(
            reason="trend_reversal",
            state="reversal",
            score=score,
            components=components,
            thresholds_used=thresholds,
            snapshot=snapshot,
        )

    if (
        mfe_pct >= float(cfg.min_mfe_pct_for_protection)
        and giveback >= float(cfg.giveback_reversal_pct)
        and has_chart_rev
        and worth_lock
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

    return _hold(state)


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
        "EXIT_EVAL id=%s %s %s state=%s score=%s comps=%s class=%s peak_lock=%s "
        "mfe=%.3f%% giveback=%.2f held=%.1fm since_ext=%.1fm net_est=%.4f peak_pnl=%.4f reason=%s",
        pos.id,
        _side_str(pos.side),
        pos.symbol,
        decision.state,
        decision.score,
        decision.components,
        snap.get("exit_pattern_class"),
        snap.get("peak_lock_clues"),
        float(snap.get("mfe_pct") or 0),
        float(snap.get("giveback_pct") or 0),
        float(snap.get("trade_duration_minutes") or 0),
        float(snap.get("minutes_since_last_favorable_extreme") or 0),
        float(snap.get("net_est") or 0),
        float(snap.get("peak_pnl") or 0),
        decision.reason,
    )
