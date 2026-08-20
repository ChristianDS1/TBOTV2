"""Rebuild historical trade economics + pattern evidence."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from trading_system.config import LearningConfig
from trading_system.database import Database
from trading_system.learning import LearningEngine, classify_strategy_outcome
from trading_system.learning.keys import (
    entry_keys_from_features,
    exit_keys_from_features,
    hold_minutes_from_position,
    is_limbo_exit,
    is_uneconomic,
)
from trading_system.learning.sessions import with_session
from trading_system.learning import trade_session
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


def _features_from_position(pos: Position) -> dict[str, Any]:
    import json

    raw = getattr(pos, "features_json", None) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def rebuild_patterns(
    db: Database,
    cfg: LearningConfig,
    *,
    quiet: bool = True,
) -> dict[str, Any]:
    """
    1) Backfill gross_pnl / cost_erosion on all closed trades
    2) Wipe pattern_evidence + applied_changes (NOT trades)
    3) Replay counts in-memory (fast), bulk write, then WR-exclusive reconcile once
    """
    closed = db.get_all_closed()
    updated = 0
    tp_net_neg = 0
    for pos in closed:
        pos = backfill_trade_economics(pos)
        db.update_trade(pos)
        updated += 1
        if pos.exit_reason == "take_profit" and (pos.pnl or 0) <= 0:
            tp_net_neg += 1

    db.clear_pattern_learning_state()

    edge_multiple = 0.5
    try:
        from trading_system.config import load_config

        edge_multiple = float(load_config().execution.hard_min_edge_multiple)
    except Exception:
        pass

    # (pattern_key, direction) -> {count, sum_pnl}
    acc: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "sum_pnl": 0.0}
    )
    entry_keys_seen: set[str] = set()
    exit_keys_seen: set[str] = set()
    cost_keys_seen: set[str] = set()

    for pos in closed:
        direction, cost_erosion = classify_strategy_outcome(pos)
        sess = trade_session(pos, cfg)
        feats = _features_from_position(pos)
        uneconomic = is_uneconomic(feats, edge_multiple=edge_multiple)
        net = float(pos.pnl) if pos.pnl is not None else 0.0

        if cost_erosion:
            cost_key = (
                f"cost_erosion|exit={pos.exit_reason or 'unknown'}|symbol={pos.symbol}"
            )
            if cfg.session_aware:
                cost_key = with_session(sess, cost_key)
            slot = acc[(cost_key, "cost_erosion")]
            slot["count"] += 1
            cost_keys_seen.add(cost_key)

        skip_entry = bool(uneconomic or cost_erosion or is_limbo_exit(pos.exit_reason))
        if not skip_entry:
            for key in entry_keys_from_features(
                feats, pos.side, edge_multiple=edge_multiple
            ):
                slot = acc[(key, direction)]
                slot["count"] += 1
                slot["sum_pnl"] += net
                entry_keys_seen.add(key)

        for key in exit_keys_from_features(feats, hold_minutes_from_position(pos)):
            slot = acc[(key, direction)]
            slot["count"] += 1
            exit_keys_seen.add(key)

    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (key, direction, int(data["count"]), float(data["sum_pnl"]), now)
        for (key, direction), data in acc.items()
        if int(data["count"]) > 0
    ]
    db.bulk_upsert_pattern_counts(rows)

    engine = LearningEngine(cfg, db, edge_multiple=edge_multiple)
    original_insert = db.insert_insight
    if quiet:

        def _quiet_insight(*_a, **_k) -> None:
            return None

        db.insert_insight = _quiet_insight  # type: ignore[method-assign]

    threshold = int(cfg.pattern_min_occurrences)
    confirmed_events = 0

    for key in sorted(entry_keys_seen):
        newly = engine._reconcile_entry_key_effects(
            key, now=now, threshold=threshold, sess="all"
        )
        if newly:
            confirmed_events += 1

    for key in sorted(exit_keys_seen):
        for direction in ("win", "loss"):
            row = db.get_pattern(key, direction)
            if not row:
                continue
            count = int(row["count"] or 0)
            if count < threshold or row.get("status") == "confirmed":
                continue
            action_info = engine._apply_confirmed_effect(
                key, direction, occurrences=count, threshold=threshold, track="exit"
            )
            db.confirm_pattern(
                key,
                direction,
                confirmed_count=count,
                decision_reason=(
                    f"ACEPTADO exit {direction}: '{key}' n={count}>={threshold}"
                ),
                effect_action=action_info["action"],
            )
            confirmed_events += 1

    for key in sorted(cost_keys_seen):
        row = db.get_pattern(key, "cost_erosion")
        if not row:
            continue
        count = int(row["count"] or 0)
        if count < threshold or row.get("status") == "confirmed":
            continue
        detail = (
            f"Cost erosion confirmed during rebuild. Occurrences={count} "
            f"(>={threshold}). Insight only — does not ban entries."
        )
        db.confirm_pattern(
            key,
            "cost_erosion",
            confirmed_count=count,
            decision_reason=f"ACEPTADO cost_erosion: {detail}",
            effect_action="cost_insight_only",
        )
        db.insert_applied_change(
            key,
            "cost_erosion",
            "cost_insight_only",
            detail,
            occurrences=count,
            threshold=threshold,
        )
        confirmed_events += 1

    engine.ranker.update_from_trades(db.get_all_closed())

    if quiet:
        db.insert_insight = original_insert  # type: ignore[method-assign]

    db.insert_insight(
        "rebuild",
        (
            f"Rebuilt patterns from {updated} closed trades "
            f"(limbo/cost skip ENTRY; WR-exclusive labels; bulk replay). "
            f"take_profit with net<=0 reclassified as strategy win + cost_erosion: {tp_net_neg}. "
            f"Confirmation events during rebuild: {confirmed_events}."
        ),
        {
            "closed_trades": updated,
            "tp_net_negative": tp_net_neg,
            "confirmation_events": confirmed_events,
            "policy": "v4_limbo_wr_exclusive",
            "bulk": True,
        },
    )

    summary = {
        "closed_trades": updated,
        "tp_net_negative_reclassified": tp_net_neg,
        "confirmation_events": confirmed_events,
        "patterns_win": len(db.get_patterns(direction="win")),
        "patterns_loss": len(db.get_patterns(direction="loss")),
        "patterns_cost_erosion": len(db.get_patterns(direction="cost_erosion")),
        "bulk": True,
    }
    logger.info("rebuild_patterns done: %s", summary)
    return summary
