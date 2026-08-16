"""Ensure priority patterns file from hist15 — does NOT write pattern_evidence win/loss."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.config import ROOT, AppConfig, load_config
from trading_system.database import Database
from trading_system.learning.priority import (
    SEED_PRIORITY_NAMES,
    ensure_priority_file,
    load_priority,
)

logger = logging.getLogger(__name__)


def seed_from_hist15(
    cfg: AppConfig | None = None,
    *,
    examples_csv: Path | str | None = None,
    confirm_wins: bool = True,
    max_loss_keys: int = 40,
) -> dict[str, Any]:
    """
    Learning keys policy v2: do not seed chart/session/symbol pattern_evidence.
    Only ensures the priority patterns JSON used for obligatory entry boost.
    """
    del confirm_wins, max_loss_keys  # legacy kwargs kept for CLI compatibility
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
    summary: dict[str, Any] = {
        "csv": str(csv_path),
        "priority_path": str(prio_path),
        "priority_names": list(prio.get("names") or SEED_PRIORITY_NAMES),
        "win_keys": 0,
        "loss_keys": 0,
        "priority_keys": 0,
        "pattern_evidence_seeded": False,
        "note": (
            "pattern_evidence chart/session seeding disabled (keys policy v2); "
            "priority file only"
        ),
    }
    summary["priority_keys"] = len(summary["priority_names"])

    with db.connection() as conn:
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
            f"Priority-only seed priority={summary['priority_keys']} "
            f"(no pattern_evidence win/loss keys)"
        ),
        summary,
    )
    logger.info("hist15 seed done (priority only): %s", summary)
    return summary
