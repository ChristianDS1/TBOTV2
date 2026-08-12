#!/usr/bin/env python3
"""
Purge paper trades (and related rows) at/after a cutoff timestamp, then rebuild
pattern learning from whatever remains.

Default cutoff: 2026-08-12 16:15:00 UTC (bad SL/TP window).

Usage (from repo root, with venv active):

  python scripts/purge_trades_after.py --dry-run
  python scripts/purge_trades_after.py --yes
  python scripts/purge_trades_after.py --after 2026-08-12T16:15:00+00:00 --yes

  # custom DB path (other machine):
  python scripts/purge_trades_after.py --db data/trading.db --yes
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AFTER = "2026-08-12T16:15:00+00:00"


def _parse_after(raw: str) -> str:
    """Return ISO timestamp comparable with TEXT columns in SQLite."""
    s = raw.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _count(conn: sqlite3.Connection, sql: str, params: tuple) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def purge(
    db_path: Path,
    after_iso: str,
    *,
    dry_run: bool,
    rebuild: bool,
) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=60000")
        # Closed: by exit_time. Open: by entry_time. Also catch closed with null exit via entry.
        trades_n = _count(
            conn,
            """
            SELECT COUNT(*) FROM trades
            WHERE (
                (exit_time IS NOT NULL AND exit_time >= ?)
                OR (exit_time IS NULL AND entry_time >= ?)
            )
            """,
            (after_iso, after_iso),
        )
        rejected_n = _count(
            conn,
            "SELECT COUNT(*) FROM rejected_signals WHERE timestamp >= ?",
            (after_iso,),
        )
        # applied_changes / insights tied to the bad window
        try:
            changes_n = _count(
                conn,
                "SELECT COUNT(*) FROM applied_changes WHERE created_at >= ?",
                (after_iso,),
            )
        except sqlite3.OperationalError:
            changes_n = 0
        try:
            insights_n = _count(
                conn,
                "SELECT COUNT(*) FROM insights WHERE created_at >= ?",
                (after_iso,),
            )
        except sqlite3.OperationalError:
            insights_n = 0

        summary = {
            "db": str(db_path),
            "after": after_iso,
            "dry_run": dry_run,
            "trades": trades_n,
            "rejected_signals": rejected_n,
            "applied_changes": changes_n,
            "insights": insights_n,
            "rebuild": False,
            "rebuild_summary": None,
        }

        if dry_run:
            return summary

        conn.execute(
            """
            DELETE FROM trades
            WHERE (
                (exit_time IS NOT NULL AND exit_time >= ?)
                OR (exit_time IS NULL AND entry_time >= ?)
            )
            """,
            (after_iso, after_iso),
        )
        conn.execute(
            "DELETE FROM rejected_signals WHERE timestamp >= ?",
            (after_iso,),
        )
        try:
            conn.execute(
                "DELETE FROM applied_changes WHERE created_at >= ?",
                (after_iso,),
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                "DELETE FROM insights WHERE created_at >= ?",
                (after_iso,),
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()
    finally:
        conn.close()

    if rebuild:
        sys.path.insert(0, str(ROOT))
        from trading_system.config import load_config
        from trading_system.database import Database
        from trading_system.learning.rebuild import rebuild_patterns

        cfg = load_config()
        db = Database(db_path)
        summary["rebuild"] = True
        summary["rebuild_summary"] = rebuild_patterns(db, cfg.learning, quiet=True)

    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Delete trades at/after a cutoff and rebuild pattern learning."
    )
    p.add_argument(
        "--after",
        default=DEFAULT_AFTER,
        help=f"UTC cutoff ISO datetime (default {DEFAULT_AFTER})",
    )
    p.add_argument(
        "--db",
        default=None,
        help="Path to trading.db (default: config database.path)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count only; do not delete",
    )
    p.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip pattern rebuild after delete",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = p.parse_args(argv)

    after_iso = _parse_after(args.after)
    if args.db:
        db_path = Path(args.db)
        if not db_path.is_absolute():
            db_path = ROOT / db_path
    else:
        sys.path.insert(0, str(ROOT))
        from trading_system.config import load_config

        db_path = load_config().db_path()

    preview = purge(db_path, after_iso, dry_run=True, rebuild=False)
    print(
        f"DB: {preview['db']}\n"
        f"After (>=): {preview['after']}\n"
        f"Would delete: trades={preview['trades']} "
        f"rejected={preview['rejected_signals']} "
        f"applied_changes={preview['applied_changes']} "
        f"insights={preview['insights']}"
    )

    if args.dry_run:
        print("Dry-run only — nothing deleted.")
        return 0

    if preview["trades"] == 0 and preview["rejected_signals"] == 0:
        print("Nothing to delete.")
        if not args.no_rebuild:
            print("Still rebuilding patterns from remaining trades…")
            out = purge(db_path, after_iso, dry_run=False, rebuild=True)
            print("rebuild:", out.get("rebuild_summary"))
        return 0

    if not args.yes:
        ans = input("Type YES to permanently delete these rows: ").strip()
        if ans != "YES":
            print("Aborted.")
            return 1

    out = purge(
        db_path,
        after_iso,
        dry_run=False,
        rebuild=not args.no_rebuild,
    )
    print(
        f"Deleted trades={out['trades']} rejected={out['rejected_signals']} "
        f"applied_changes={out['applied_changes']} insights={out['insights']}"
    )
    if out.get("rebuild_summary") is not None:
        print("rebuild:", out["rebuild_summary"])
    print("Done. Restart the bot if it is running so it reloads the DB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
