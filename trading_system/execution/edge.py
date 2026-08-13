"""Fee/edge helpers — soft filters that avoid fee-suicide without starving discovery."""

from __future__ import annotations

from dataclasses import dataclass

from trading_system.types import Position, Side


@dataclass
class EdgeAssessment:
    edge_bps: float
    round_trip_cost_bps: float
    ratio: float
    hard_reject: bool
    soft_penalty: bool
    reason: str


def round_trip_cost_bps(fee_bps: float, slippage_bps: float) -> float:
    """One side costs (fee+slip) in our paper model; round trip = 2 sides."""
    return 2.0 * (fee_bps + slippage_bps)


def distance_to_tp_bps(price: float, take_profit: float | None) -> float | None:
    if take_profit is None or price <= 0:
        return None
    return abs(take_profit - price) / price * 10_000.0


def assess_entry_edge(
    *,
    price: float,
    take_profit: float | None,
    fee_bps: float,
    slippage_bps: float,
    hard_multiple: float = 1.0,
    soft_multiple: float = 1.5,
) -> EdgeAssessment:
    rt = round_trip_cost_bps(fee_bps, slippage_bps)
    edge = distance_to_tp_bps(price, take_profit)
    if edge is None:
        return EdgeAssessment(
            edge_bps=0.0,
            round_trip_cost_bps=rt,
            ratio=999.0,
            hard_reject=False,
            soft_penalty=False,
            reason="no_tp",
        )
    ratio = edge / rt if rt > 0 else 999.0
    if ratio < hard_multiple:
        return EdgeAssessment(
            edge_bps=edge,
            round_trip_cost_bps=rt,
            ratio=ratio,
            hard_reject=True,
            soft_penalty=False,
            reason=(
                f"insufficient_edge_vs_fees:"
                f"edge={edge:.1f}bps < {hard_multiple}x cost={rt:.1f}bps"
            ),
        )
    if ratio < soft_multiple:
        return EdgeAssessment(
            edge_bps=edge,
            round_trip_cost_bps=rt,
            ratio=ratio,
            hard_reject=False,
            soft_penalty=True,
            reason=(
                f"thin_edge:edge={edge:.1f}bps "
                f"({ratio:.2f}x of {rt:.1f}bps cost)"
            ),
        )
    return EdgeAssessment(
        edge_bps=edge,
        round_trip_cost_bps=rt,
        ratio=ratio,
        hard_reject=False,
        soft_penalty=False,
        reason="edge_ok",
    )


def position_notional(pos: Position) -> float:
    if pos.notional is not None and pos.notional > 0:
        return float(pos.notional)
    lev = float(pos.leverage or 1.0)
    return float(pos.qty) * max(lev, 1.0)


def estimate_close_net(
    pos: Position,
    mark_price: float,
    fee_bps: float,
    slippage_bps: float,
) -> float:
    """Estimate net PnL if we close now at mark (exit fee + exit slip; entry fee sunk)."""
    slip = slippage_bps / 10_000.0
    notional = position_notional(pos)
    if pos.side == Side.CALL or pos.side.value == "call":
        fill = mark_price * (1 - slip)
        direction = 1
    else:
        fill = mark_price * (1 + slip)
        direction = -1
    raw = direction * (fill - pos.entry_price) / pos.entry_price * notional
    exit_fee = notional * (fee_bps + slippage_bps) / 10_000.0
    return raw - exit_fee


def can_take_profit_net_positive(
    pos: Position,
    mark_price: float,
    fee_bps: float,
    slippage_bps: float,
) -> bool:
    """TP only when estimated net is strictly > 0."""
    return estimate_close_net(pos, mark_price, fee_bps, slippage_bps) > 0.0


def _f(row: dict, key: str, default: float | None = None) -> float | None:
    v = row.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def score_trend_fade(side: Side | str, row: dict) -> tuple[int, list[str]]:
    """
    Score opposing rejection + momentum fade components (no threshold).

    CALL fade signals (PUT mirrored):
      1) rejection_bear
      2) macd_fast hist falling 2 bars OR macd_fast_bear_cross
      3) RSI rollover from >=50
      4) macd_slow_hist deteriorating vs trade direction
      5) chart_reversal / ltf_turn when present
    """
    side_v = side.value if isinstance(side, Side) else str(side).lower()
    is_call = side_v == "call"
    reasons: list[str] = []

    if is_call:
        if bool(row.get("rejection_bear")):
            reasons.append("rejection_bear")
        hist = _f(row, "macd_fast_hist")
        prev = _f(row, "macd_fast_hist_prev")
        prev2 = _f(row, "macd_fast_hist_prev2")
        bear_cross = bool(row.get("macd_fast_bear_cross"))
        falling2 = (
            hist is not None
            and prev is not None
            and prev2 is not None
            and hist < prev < prev2
        )
        if bear_cross or falling2:
            reasons.append("macd_fast_fade")
        rsi = _f(row, "rsi")
        rsi_prev = _f(row, "rsi_prev")
        if (
            rsi is not None
            and rsi_prev is not None
            and rsi < rsi_prev
            and (rsi_prev >= 50 or (rsi_prev >= 50 and rsi < 50))
        ):
            reasons.append("rsi_rollover")
        elif rsi is not None and rsi_prev is not None and rsi_prev >= 50 and rsi < 50:
            reasons.append("rsi_rollover")
        slow = _f(row, "macd_slow_hist")
        slow_prev = _f(row, "macd_slow_hist_prev")
        if slow is not None and slow_prev is not None and slow < slow_prev:
            reasons.append("macd_slow_fade")
    else:
        if bool(row.get("rejection_bull")):
            reasons.append("rejection_bull")
        hist = _f(row, "macd_fast_hist")
        prev = _f(row, "macd_fast_hist_prev")
        prev2 = _f(row, "macd_fast_hist_prev2")
        bull_cross = bool(row.get("macd_fast_bull_cross"))
        rising2 = (
            hist is not None
            and prev is not None
            and prev2 is not None
            and hist > prev > prev2
        )
        if bull_cross or rising2:
            reasons.append("macd_fast_fade")
        rsi = _f(row, "rsi")
        rsi_prev = _f(row, "rsi_prev")
        if (
            rsi is not None
            and rsi_prev is not None
            and rsi > rsi_prev
            and rsi_prev <= 50
        ):
            reasons.append("rsi_rollover")
        elif rsi is not None and rsi_prev is not None and rsi_prev <= 50 and rsi > 50:
            reasons.append("rsi_rollover")
        slow = _f(row, "macd_slow_hist")
        slow_prev = _f(row, "macd_slow_hist_prev")
        if slow is not None and slow_prev is not None and slow > slow_prev:
            reasons.append("macd_slow_fade")

    if is_call and bool(row.get("chart_reversal_bear")):
        reasons.append("chart_reversal")
    elif not is_call and bool(row.get("chart_reversal_bull")):
        reasons.append("chart_reversal")
    ltf = row.get("ltf_turn")
    if is_call and ltf == "turn_down":
        reasons.append("ltf_turn")
    elif not is_call and ltf == "turn_up":
        reasons.append("ltf_turn")

    return len(reasons), reasons


def detect_trend_fade(
    side: Side | str,
    row: dict,
    *,
    min_score: int = 2,
) -> tuple[bool, list[str]]:
    """True when fade component score >= min_score (back-compat wrapper)."""
    score, reasons = score_trend_fade(side, row)
    return score >= int(min_score), reasons


# Back-compat alias
def can_take_profit_net_non_negative(
    pos: Position,
    mark_price: float,
    fee_bps: float,
    slippage_bps: float,
) -> bool:
    return can_take_profit_net_positive(pos, mark_price, fee_bps, slippage_bps)


def unrealized_pnl_on_notional(pos: Position, mark_price: float) -> float:
    notional = position_notional(pos)
    direction = 1 if pos.side.value == "call" else -1
    return direction * (mark_price - pos.entry_price) / pos.entry_price * notional


def should_adaptive_time_stop(
    pos: Position,
    mark_price: float,
    held_minutes: float,
    *,
    min_hold_minutes: float = 1.0,
    preferred_hold_minutes: float = 3.0,
    max_hold_minutes: float = 10.0,
) -> bool:
    """
    Early-rejection bias: prefer closing after the primary window unless the
    trade is still progressing toward TP. Hard cap at max_hold_minutes.
    """
    max_m = max(float(max_hold_minutes), 1.0)
    pref_m = min(max(float(preferred_hold_minutes), float(min_hold_minutes)), max_m)
    min_m = min(float(min_hold_minutes), pref_m)

    if held_minutes >= max_m:
        return True
    if held_minutes < pref_m:
        return False
    # Between preferred and max: extend only if still working
    entry = pos.entry_mark if pos.entry_mark and pos.entry_mark > 0 else pos.entry_price
    if entry <= 0:
        return True
    is_call = pos.side.value == "call"
    direction = 1 if is_call else -1
    move = direction * (mark_price - entry) / entry

    if pos.take_profit is not None:
        tp = float(pos.take_profit)
        hit_tp = (is_call and mark_price >= tp) or ((not is_call) and mark_price <= tp)
        if hit_tp:
            # TP deferred for net>0 — keep extending until max
            return False
        dist_entry = abs(tp - entry)
        dist_now = abs(tp - mark_price)
        progressing = dist_now < dist_entry * 0.90
    else:
        progressing = move > 0

    in_favor = move > 0
    if in_favor or progressing:
        return False
    # Stalled / adverse after preferred early window
    return held_minutes >= max(pref_m, min_m)
