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


def estimate_close_net(
    pos: Position,
    mark_price: float,
    fee_bps: float,
    slippage_bps: float,
) -> float:
    """Estimate net PnL if we close now at mark (exit fee + exit slip only; entry fee sunk)."""
    slip = slippage_bps / 10_000.0
    if pos.side == Side.CALL or pos.side.value == "call":
        fill = mark_price * (1 - slip)
        direction = 1
    else:
        fill = mark_price * (1 + slip)
        direction = -1
    raw = direction * (fill - pos.entry_price) / pos.entry_price * pos.qty
    exit_fee = pos.qty * (fee_bps + slippage_bps) / 10_000.0
    return raw - exit_fee


def can_take_profit_net_non_negative(
    pos: Position,
    mark_price: float,
    fee_bps: float,
    slippage_bps: float,
) -> bool:
    return estimate_close_net(pos, mark_price, fee_bps, slippage_bps) >= 0.0
