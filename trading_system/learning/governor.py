"""North-star idle governor + hard-reject demotion helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from trading_system.database import Database
from trading_system.learning.keys import (
    HARD_REJECT_FORBIDDEN_KEYS,
    is_entry_compound,
    is_entry_key,
)

logger = logging.getLogger(__name__)

POLICY_V3_FLAG = "pattern_keys_policy_v3_no_single_hard_reject"


def is_hard_reject_forbidden_key(key: str) -> bool:
    """1-dim / ultra-common keys must never hard-reject the book."""
    if key in HARD_REJECT_FORBIDDEN_KEYS:
        return True
    if not is_entry_key(key):
        return True
    # All singles (non-compound) forbidden from hard_reject
    if not is_entry_compound(key):
        return True
    return False


def demote_hard_reject_keys(
    db: Database,
    keys: list[str] | None = None,
    *,
    reason: str = "demote",
) -> dict[str, Any]:
    """
    Clear hard_reject on confirmed ENTRY loss rows → observing + confidence_penalty.
    If keys is None, demote all hard_reject loss rows (and forbidden singles).
    """
    demoted: list[str] = []
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT pattern_key, direction, status, effect_action
            FROM pattern_evidence
            WHERE direction='loss'
            """
        ).fetchall()
        for r in rows:
            key = r["pattern_key"]
            action = r["effect_action"] or ""
            status = r["status"] or ""
            should = False
            if keys is not None:
                should = key in keys
            else:
                should = action == "hard_reject" or (
                    status == "confirmed" and is_hard_reject_forbidden_key(key)
                )
            if not should:
                continue
            if not is_entry_key(key) and not key.startswith("htf_ltf_combo="):
                continue
            conn.execute(
                """
                UPDATE pattern_evidence
                SET status='observing',
                    effect_action='confidence_penalty',
                    decision_reason=?,
                    confirmed_at=NULL
                WHERE pattern_key=? AND direction='loss'
                """,
                (
                    f"DEMOTADO ({reason}): hard_reject cleared — north-star idle/book unlock",
                    key,
                ),
            )
            conn.execute(
                "DELETE FROM applied_changes WHERE pattern_key=? AND action='hard_reject'",
                (key,),
            )
            demoted.append(key)

    summary = {"demoted": len(demoted), "keys": demoted[:30], "reason": reason}
    if demoted:
        db.insert_insight(
            "hard_reject_demote",
            f"Demoted {len(demoted)} hard_reject loss keys ({reason})",
            summary,
        )
        logger.info("demote_hard_reject_keys: %s", summary)
    return summary


def neutralize_blocking_hard_rejects(db: Database) -> dict[str, Any]:
    """Deploy: clear all hard_rejects especially default singles."""
    return demote_hard_reject_keys(db, keys=None, reason="deploy_v3")


def ensure_no_single_hard_reject_policy(db: Database) -> dict[str, Any] | None:
    """One-shot neutralize after v2 allowlist sanitize."""
    if db.get_state(POLICY_V3_FLAG):
        return None
    summary = neutralize_blocking_hard_rejects(db)
    db.set_state(POLICY_V3_FLAG, "1")
    db.insert_insight(
        "pattern_keys_policy_v3",
        f"Neutralized hard_reject bans (esp. 1-dim defaults): demoted={summary['demoted']}",
        summary,
    )
    return summary


def count_fills_since(db: Database, since: datetime) -> int:
    iso = since.astimezone(timezone.utc).isoformat()
    with db.connection() as conn:
        # Count opens or closes in window as "fills"
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM trades
            WHERE (entry_time >= ? OR exit_time >= ?)
            """,
            (iso, iso),
        ).fetchone()
    return int(row["n"] if row else 0)


def recent_reject_stats(db: Database, *, limit: int = 40) -> dict[str, Any]:
    rows = db.recent_rejected(limit)
    loss_ban = 0
    for r in rows:
        reason = str(r.get("reason") or "")
        if reason.startswith("confirmed_loss_pattern:"):
            loss_ban += 1
    return {
        "n": len(rows),
        "confirmed_loss_rejects": loss_ban,
        "fraction": (loss_ban / len(rows)) if rows else 0.0,
        "keys": [
            str(r.get("reason") or "").split(":", 1)[-1]
            for r in rows
            if str(r.get("reason") or "").startswith("confirmed_loss_pattern:")
        ],
    }


def maybe_idle_unban(
    db: Database,
    *,
    idle_minutes: float = 45.0,
    min_rejects: int = 8,
    loss_reject_fraction: float = 0.5,
) -> dict[str, Any] | None:
    """
    If no fills in idle window AND confirmed_loss rejects dominate → demote hard_rejects.
    North-star: trade_rate≈0 + mass hard-reject = policy failure.
    """
    since = datetime.now(timezone.utc) - timedelta(minutes=float(idle_minutes))
    fills = count_fills_since(db, since)
    if fills > 0:
        return None
    stats = recent_reject_stats(db, limit=40)
    if stats["n"] < int(min_rejects):
        return None
    if stats["fraction"] < float(loss_reject_fraction):
        return None
    # Demote keys seen in rejects + all remaining hard_rejects
    keys = list({k for k in stats["keys"] if k})
    summary = demote_hard_reject_keys(db, keys=keys or None, reason="idle_governor")
    # Also wipe any leftover hard_reject actions
    leftover = demote_hard_reject_keys(db, keys=None, reason="idle_governor_sweep")
    summary["fills"] = fills
    summary["reject_stats"] = stats
    summary["sweep_demoted"] = leftover["demoted"]
    db.insert_insight(
        "idle_governor",
        (
            f"Idle {idle_minutes:.0f}m with 0 fills; demoted hard_rejects "
            f"(n={summary['demoted']}+{leftover['demoted']})"
        ),
        summary,
    )
    logger.warning("idle_governor unban: %s", summary)
    return summary
