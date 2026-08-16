"""Wipe non-allowlist pattern_evidence / applied_changes (learning keys policy v2)."""

from __future__ import annotations

import logging
from typing import Any

from trading_system.database import Database
from trading_system.learning.keys import is_allowlisted_key

logger = logging.getLogger(__name__)

POLICY_FLAG = "pattern_keys_policy_v2"


def sanitize_pattern_evidence(db: Database, *, dry_run: bool = False) -> dict[str, Any]:
    """
    DELETE pattern_evidence rows that are not on the ENTRY/EXIT allowlist.
    Also drop matching applied_changes. Keeps trades. cost_erosion rows kept
    (separate insight track; not win/loss entry effects).
    """
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT pattern_key, direction, count, status FROM pattern_evidence"
        ).fetchall()
        keep: list[str] = []
        drop: list[tuple[str, str]] = []
        for r in rows:
            key = r["pattern_key"]
            direction = r["direction"]
            # cost_erosion track is insight-only; keep as today
            if direction == "cost_erosion" or key.startswith("cost_erosion"):
                keep.append(key)
                continue
            if is_allowlisted_key(key):
                keep.append(key)
            else:
                drop.append((key, direction))

        dropped_changes = 0
        if not dry_run and drop:
            for key, direction in drop:
                conn.execute(
                    "DELETE FROM pattern_evidence WHERE pattern_key=? AND direction=?",
                    (key, direction),
                )
                cur = conn.execute(
                    "DELETE FROM applied_changes WHERE pattern_key=?",
                    (key,),
                )
                dropped_changes += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    summary = {
        "kept": len(set(keep)),
        "dropped_evidence": len(drop),
        "dropped_applied_changes": dropped_changes,
        "dry_run": dry_run,
        "sample_dropped": [f"{k}|{d}" for k, d in drop[:15]],
    }
    logger.info("sanitize_pattern_evidence: %s", summary)
    return summary


def ensure_pattern_keys_policy(db: Database) -> dict[str, Any] | None:
    """One-shot on deploy: wipe legacy keys if flag unset."""
    if db.get_state(POLICY_FLAG):
        return None
    summary = sanitize_pattern_evidence(db, dry_run=False)
    db.set_state(POLICY_FLAG, "1")
    db.insert_insight(
        "pattern_keys_policy_v2",
        (
            f"Sanitized legacy pattern keys: dropped={summary['dropped_evidence']} "
            f"kept={summary['kept']}"
        ),
        summary,
    )
    return summary
