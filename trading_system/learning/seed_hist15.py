"""Seed live learning DB from hist15 wins/losses + priority patterns."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_system.config import ROOT, AppConfig, load_config
from trading_system.database import Database
from trading_system.learning.priority import (
    SEED_PRIORITY_NAMES,
    ensure_priority_file,
    load_priority,
)
from trading_system.learning.sessions import session_bucket

logger = logging.getLogger(__name__)


def _seed_key(chart: str, *, session: str | None) -> str:
    base = f"chart={chart}"
    if session:
        return f"session={session}|{base}"
    return base


def seed_from_hist15(
    cfg: AppConfig | None = None,
    *,
    examples_csv: Path | str | None = None,
    confirm_wins: bool = True,
    max_loss_keys: int = 40,
) -> dict[str, Any]:
    """Seed pattern_evidence from hist15 + priority obligatory keys."""
    cfg = cfg or load_config()
    db = Database(cfg.db_path())
    prio_path = ensure_priority_file()
    prio = load_priority(prio_path)

    csv_path = (
        Path(examples_csv)
        if examples_csv
        else ROOT / "data" / "ml" / "hist15_clean" / "examples.csv"
    )
    now = datetime.now(timezone.utc).isoformat()
    threshold = max(int(cfg.learning.pattern_min_occurrences), 20)
    summary: dict[str, Any] = {
        "csv": str(csv_path),
        "priority_path": str(prio_path),
        "priority_names": list(prio.get("names") or SEED_PRIORITY_NAMES),
        "win_keys": 0,
        "loss_keys": 0,
        "priority_keys": 0,
    }
    prio_set = set(summary["priority_names"])

    win_counts: Counter[str] = Counter()
    loss_counts: Counter[str] = Counter()

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "chart_pattern" in df.columns and "label_win" in df.columns:
            ts_col = df["timestamp"] if "timestamp" in df.columns else [None] * len(df)
            for chart, label, ts in zip(
                df["chart_pattern"].astype(str),
                df["label_win"].fillna(0).astype(int),
                ts_col,
            ):
                if not chart or chart == "nan":
                    continue
                sess = None
                if ts is not None and cfg.learning.session_aware:
                    try:
                        t = pd.Timestamp(ts)
                        if t.tzinfo is None:
                            t = t.tz_localize("UTC")
                        sess = session_bucket(
                            t.to_pydatetime(), cfg.learning.session_buckets
                        )
                    except Exception:
                        sess = None
                key = _seed_key(chart, session=sess)
                if label == 1:
                    win_counts[key] += 1
                elif chart not in prio_set:
                    loss_counts[key] += 1
        else:
            logger.warning("examples.csv missing chart_pattern/label_win")
    else:
        logger.warning("hist15 examples.csv missing — priority keys only")

    with db.connection() as conn:

        def upsert(key: str, direction: str, count: int) -> None:
            row = conn.execute(
                "SELECT pattern_key FROM pattern_evidence WHERE pattern_key=? AND direction=?",
                (key, direction),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO pattern_evidence (pattern_key, direction, count, last_seen, status)
                    VALUES (?, ?, ?, ?, 'observing')
                    """,
                    (key, direction, int(count), now),
                )
            else:
                conn.execute(
                    """
                    UPDATE pattern_evidence
                    SET count=?, last_seen=?
                    WHERE pattern_key=? AND direction=?
                    """,
                    (int(count), now, key, direction),
                )

        def confirm(key: str, direction: str, count: int, reason: str, action: str) -> None:
            conn.execute(
                """
                UPDATE pattern_evidence
                SET status='confirmed',
                    confirmed_at=?,
                    confirmed_count=?,
                    decision_reason=?,
                    effect_action=?
                WHERE pattern_key=? AND direction=?
                """,
                (now, int(count), reason, action, key, direction),
            )

        for name in summary["priority_names"]:
            for key in (f"chart={name}", f"session=weekend|chart={name}"):
                upsert(key, "win", threshold)
                confirm(
                    key,
                    "win",
                    threshold,
                    f"ACEPTADO priority seed from 3 confirmations ∩ hist15: {name}",
                    "priority_boost",
                )
                summary["priority_keys"] += 1

        for key, count in win_counts.items():
            upsert(key, "win", int(count))
            if confirm_wins and count >= 5:
                confirm(
                    key,
                    "win",
                    int(count),
                    f"ACEPTADO hist15 net-win aggregate n={count} key={key}",
                    "confidence_boost",
                )
            summary["win_keys"] += 1

        for key, count in loss_counts.most_common(int(max_loss_keys)):
            c = min(int(count), 50)
            upsert(key, "loss", c)
            if c >= threshold:
                confirm(
                    key,
                    "loss",
                    c,
                    f"ACEPTADO hist15 loss-context n={count} key={key}",
                    "confidence_penalty",
                )
            summary["loss_keys"] += 1

        conn.execute(
            "INSERT INTO system_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("hist15_seed_at", now),
        )
        conn.execute(
            "INSERT INTO system_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("hist15_seed_summary", json.dumps(summary)),
        )

    db.insert_insight(
        "hist15_seed",
        (
            f"Seeded priority={summary['priority_keys']} "
            f"win_keys={summary['win_keys']} loss_keys={summary['loss_keys']}"
        ),
        summary,
    )
    logger.info("hist15 seed done: %s", summary)
    return summary
