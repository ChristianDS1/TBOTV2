"""Rebuild historical trade economics + pattern evidence."""

from __future__ import annotations

import logging
from typing import Any

from trading_system.config import LearningConfig
from trading_system.database import Database
from trading_system.learning import LearningEngine, classify_strategy_outcome
from trading_system.types import Position

logger = logging.getLogger(__name__)


def backfill_trade_economics(pos: Position) -> Position:
    """
    Estimate gross_pnl / cost_erosion for trades closed before those fields existed.

    Historical fills already include slippage; we approximate gross as the
    fill-to-fill move (before subtracting exit fees). Exit fee ≈ half of total
    fees when entry+exit used the same bps model.
    """
    net = pos.pnl if pos.pnl is not None else 0.0

    if pos.entry_mark is None:
        pos.entry_mark = pos.entry_price

    if pos.gross_pnl is None:
        if pos.exit_price is not None and pos.entry_price:
            exit_fee_approx = (pos.fees or 0.0) / 2.0
            # Old pnl ≈ fill_move - exit_fees → reconstruct gross ≈ net + exit_fee
            pos.gross_pnl = net + exit_fee_approx
            if pos.exit_reason == "take_profit" and pos.gross_pnl <= 0:
                # Target hit; net was only cost-drag — force small positive strategy gross
                pos.gross_pnl = abs(net) + exit_fee_approx + 1e-9
        elif pos.exit_reason == "take_profit":
            pos.gross_pnl = max(net, 0.0) + (pos.fees or 0.0) / 2.0 + 1e-9
        else:
            pos.gross_pnl = net

    _direction_s, erosion = classify_strategy_outcome(pos)
    if pos.exit_reason == "take_profit":
        erosion = net <= 0
    pos.cost_erosion = erosion
    return pos


def rebuild_patterns(
    db: Database,
    cfg: LearningConfig,
    *,
    quiet: bool = True,
) -> dict[str, Any]:
    """
    1) Backfill gross_pnl / cost_erosion on all closed trades
    2) Wipe pattern_evidence + applied_changes
    3) Re-run learning classification in trade order
    """
    closed = db.get_all_closed()
    # get_all_closed is ORDER BY id — good
    updated = 0
    tp_net_neg = 0
    for pos in closed:
        before = (pos.gross_pnl, pos.cost_erosion)
        pos = backfill_trade_economics(pos)
        db.update_trade(pos)
        updated += 1
        if pos.exit_reason == "take_profit" and (pos.pnl or 0) <= 0:
            tp_net_neg += 1
        logger.debug(
            "backfill trade=%s exit=%s net=%s gross=%s erosion=%s was=%s",
            pos.id,
            pos.exit_reason,
            pos.pnl,
            pos.gross_pnl,
            pos.cost_erosion,
            before,
        )

    db.clear_pattern_learning_state()

    engine = LearningEngine(cfg, db)
    # Avoid flooding insights during rebuild
    original_insert = db.insert_insight

    if quiet:

        def _quiet_insight(*_a, **_k) -> None:
            return None

        db.insert_insight = _quiet_insight  # type: ignore[method-assign]

    confirmed_events = 0
    for pos in db.get_all_closed():
        newly = engine.on_trade_closed(pos)
        confirmed_events += len(newly)

    if quiet:
        db.insert_insight = original_insert  # type: ignore[method-assign]

    db.insert_insight(
        "rebuild",
        (
            f"Rebuilt patterns from {updated} closed trades. "
            f"take_profit with net<=0 reclassified as strategy win + cost_erosion: {tp_net_neg}. "
            f"Confirmation events during rebuild: {confirmed_events}."
        ),
        {
            "closed_trades": updated,
            "tp_net_negative": tp_net_neg,
            "confirmation_events": confirmed_events,
        },
    )

    summary = {
        "closed_trades": updated,
        "tp_net_negative_reclassified": tp_net_neg,
        "confirmation_events": confirmed_events,
        "patterns_win": len(db.get_patterns(direction="win")),
        "patterns_loss": len(db.get_patterns(direction="loss")),
        "patterns_cost_erosion": len(db.get_patterns(direction="cost_erosion")),
    }
    logger.info("rebuild_patterns done: %s", summary)
    return summary
