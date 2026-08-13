#!/usr/bin/env python3
"""
Reset loss-pattern learning and drop time_stop trades.

Keeps non-time_stop win patterns. Does NOT rebuild loss evidence from remaining
trades (that would immediately re-confirm contaminated SL/time_stop losses).

Bot MUST be stopped first (SQLite lock). Each machine has its own trading.db —
run after git pull on every PC.

Usage (from repo root, venv active):

  python -m trading_system reset-loss-learning --dry-run
  python -m trading_system reset-loss-learning --yes

  python scripts/reset_loss_learning.py --dry-run
  python scripts/reset_loss_learning.py --yes --db data/trading.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def preview(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "time_stop_trades": _count(
            conn, "SELECT COUNT(*) FROM trades WHERE exit_reason = 'time_stop'"
        ),
        "loss_patterns": _count(
            conn, "SELECT COUNT(*) FROM pattern_evidence WHERE direction = 'loss'"
        ),
        "cost_erosion_patterns": _count(
            conn,
            "SELECT COUNT(*) FROM pattern_evidence WHERE direction = 'cost_erosion'",
        ),
        "win_time_stop_patterns": _count(
            conn,
            """
            SELECT COUNT(*) FROM pattern_evidence
            WHERE direction = 'win' AND pattern_key LIKE '%time_stop%'
            """,
        ),
        "applied_loss_or_timestop": _count(
            conn,
            """
            SELECT COUNT(*) FROM applied_changes
            WHERE direction IN ('loss', 'cost_erosion')
               OR action IN ('soft_reject', 'confidence_penalty')
               OR pattern_key LIKE '%time_stop%'
            """,
        ),
    }


def reset_loss_learning(db_path: Path, *, dry_run: bool) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=60000")
        counts = preview(conn)
        summary: dict[str, Any] = {
            "db": str(db_path),
            "dry_run": dry_run,
            **counts,
            "strategy_stats_rebuilt": False,
        }
        if dry_run:
            return summary

        conn.execute("DELETE FROM trades WHERE exit_reason = 'time_stop'")
        conn.execute(
            "DELETE FROM pattern_evidence WHERE direction IN ('loss', 'cost_erosion')"
        )
        conn.execute(
            """
            DELETE FROM pattern_evidence
            WHERE direction = 'win' AND pattern_key LIKE '%time_stop%'
            """
        )
        conn.execute(
            """
            DELETE FROM applied_changes
            WHERE direction IN ('loss', 'cost_erosion')
               OR action IN ('soft_reject', 'confidence_penalty')
               OR pattern_key LIKE '%time_stop%'
            """
        )
        conn.execute("DELETE FROM strategy_stats")
        conn.commit()
    finally:
        conn.close()

    sys.path.insert(0, str(ROOT))
    from trading_system.config import load_config
    from trading_system.database import Database
    from trading_system.learning import StrategyRanker

    cfg = load_config()
    db = Database(db_path)
    ranker = StrategyRanker(cfg.learning, db)
    ranker.update_from_trades(db.get_all_closed())
    summary["strategy_stats_rebuilt"] = True
    conn2 = sqlite3.connect(str(db_path))
    try:
        summary["remaining_closed_trades"] = _count(
            conn2, "SELECT COUNT(*) FROM trades WHERE lower(coalesce(status,'')) = 'closed'"
        )
    finally:
        conn2.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Wipe loss learning and delete time_stop trades. Keep other wins."
    )
    p.add_argument("--db", default=None, help="Path to trading.db")
    p.add_argument("--dry-run", action="store_true", help="Count only; do not delete")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = p.parse_args(argv)

    if args.db:
        db_path = Path(args.db)
        if not db_path.is_absolute():
            db_path = ROOT / db_path
    else:
        sys.path.insert(0, str(ROOT))
        from trading_system.config import load_config

        db_path = load_config().db_path()

    pre = reset_loss_learning(db_path, dry_run=True)
    print(
        f"DB: {pre['db']}\n"
        f"Would delete: time_stop_trades={pre['time_stop_trades']} "
        f"loss_patterns={pre['loss_patterns']} "
        f"cost_erosion={pre['cost_erosion_patterns']} "
        f"win_time_stop_keys={pre['win_time_stop_patterns']} "
        f"applied_changes={pre['applied_loss_or_timestop']}\n"
        "Keeps non-time_stop win patterns. Does NOT rebuild losses from remaining trades."
    )

    if args.dry_run:
        print("Dry-run only — nothing deleted.")
        return 0

    if not args.yes:
        ans = input("Type YES to apply this wipe (stop the bot first): ").strip()
        if ans != "YES":
            print("Aborted.")
            return 1

    out = reset_loss_learning(db_path, dry_run=False)
    print(
        f"Wiped time_stop_trades={out['time_stop_trades']} "
        f"loss_patterns={out['loss_patterns']} "
        f"remaining_closed={out.get('remaining_closed_trades')} "
        f"strategy_stats_rebuilt={out['strategy_stats_rebuilt']}"
    )
    print("Done. Restart the bot so it reloads the DB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
