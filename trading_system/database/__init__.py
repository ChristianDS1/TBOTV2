"""SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from trading_system.types import (
    Position,
    RejectedSignal,
    Side,
    TradeStatus,
    Venue,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    venue TEXT NOT NULL,
    side TEXT NOT NULL,
    strategy TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    take_profit REAL,
    stop_loss REAL,
    confidence REAL,
    regime TEXT,
    features_json TEXT,
    exploration INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    exit_price REAL,
    exit_time TEXT,
    pnl REAL,
    fees REAL DEFAULT 0,
    exit_reason TEXT
);

CREATE TABLE IF NOT EXISTS rejected_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    venue TEXT NOT NULL,
    side TEXT,
    strategy TEXT NOT NULL,
    confidence REAL,
    reason TEXT,
    features_json TEXT,
    regime TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    meta_json TEXT
);

CREATE TABLE IF NOT EXISTS strategy_stats (
    strategy TEXT PRIMARY KEY,
    trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    expectancy REAL DEFAULT 0,
    profit_factor REAL DEFAULT 0,
    rank_score REAL DEFAULT 0,
    status TEXT DEFAULT 'active',
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    open_positions INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pattern_evidence (
    pattern_key TEXT NOT NULL,
    direction TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    last_seen TEXT,
    status TEXT NOT NULL DEFAULT 'observing',
    confirmed_at TEXT,
    confirmed_count INTEGER,
    decision_reason TEXT,
    effect_action TEXT,
    PRIMARY KEY (pattern_key, direction)
);

CREATE TABLE IF NOT EXISTS applied_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    pattern_key TEXT NOT NULL,
    direction TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_reports (
    day TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    summary_json TEXT
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)
        self._migrate_pattern_columns()

    def _migrate_pattern_columns(self) -> None:
        needed = {
            "confirmed_at": "TEXT",
            "confirmed_count": "INTEGER",
            "decision_reason": "TEXT",
            "effect_action": "TEXT",
        }
        with self.connection() as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(pattern_evidence)").fetchall()
            }
            for name, typ in needed.items():
                if name not in cols:
                    conn.execute(
                        f"ALTER TABLE pattern_evidence ADD COLUMN {name} {typ}"
                    )

    def insert_trade(self, pos: Position) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO trades (
                    symbol, venue, side, strategy, qty, entry_price, entry_time,
                    take_profit, stop_loss, confidence, regime, features_json,
                    exploration, status, exit_price, exit_time, pnl, fees, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pos.symbol,
                    pos.venue.value,
                    pos.side.value,
                    pos.strategy,
                    pos.qty,
                    pos.entry_price,
                    pos.entry_time.isoformat(),
                    pos.take_profit,
                    pos.stop_loss,
                    pos.confidence,
                    pos.regime,
                    pos.features_json,
                    int(pos.exploration),
                    pos.status.value,
                    pos.exit_price,
                    pos.exit_time.isoformat() if pos.exit_time else None,
                    pos.pnl,
                    pos.fees,
                    pos.exit_reason,
                ),
            )
            return int(cur.lastrowid)

    def update_trade(self, pos: Position) -> None:
        assert pos.id is not None
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE trades SET
                    status=?, exit_price=?, exit_time=?, pnl=?, fees=?, exit_reason=?
                WHERE id=?
                """,
                (
                    pos.status.value,
                    pos.exit_price,
                    pos.exit_time.isoformat() if pos.exit_time else None,
                    pos.pnl,
                    pos.fees,
                    pos.exit_reason,
                    pos.id,
                ),
            )

    def get_open_trades(self) -> list[Position]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY id"
            ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def get_closed_trades(self, limit: int = 200) -> list[Position]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='closed' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def get_all_closed(self) -> list[Position]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='closed' ORDER BY id"
            ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def insert_rejected(self, sig: RejectedSignal) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO rejected_signals
                (symbol, venue, side, strategy, confidence, reason, features_json, regime, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sig.symbol,
                    sig.venue.value,
                    sig.side.value if sig.side else None,
                    sig.strategy,
                    sig.confidence,
                    sig.reason,
                    json.dumps(sig.features),
                    sig.regime.value,
                    sig.timestamp.isoformat(),
                ),
            )

    def recent_rejected(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM rejected_signals ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_insight(self, category: str, content: str, meta: dict | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO insights (created_at, category, content, meta_json) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), category, content, json.dumps(meta or {})),
            )

    def recent_insights(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM insights ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_strategy_stats(self, stats: dict[str, Any]) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO strategy_stats
                (strategy, trades, wins, losses, total_pnl, expectancy, profit_factor, rank_score, status, updated_at)
                VALUES (:strategy, :trades, :wins, :losses, :total_pnl, :expectancy, :profit_factor, :rank_score, :status, :updated_at)
                ON CONFLICT(strategy) DO UPDATE SET
                    trades=excluded.trades,
                    wins=excluded.wins,
                    losses=excluded.losses,
                    total_pnl=excluded.total_pnl,
                    expectancy=excluded.expectancy,
                    profit_factor=excluded.profit_factor,
                    rank_score=excluded.rank_score,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                stats,
            )

    def get_strategy_stats(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM strategy_stats ORDER BY rank_score DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def record_equity(self, equity: float, cash: float, open_positions: int) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO equity_curve (timestamp, equity, cash, open_positions) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), equity, cash, open_positions),
            )

    def set_state(self, key: str, value: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO system_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_state(self, key: str, default: str | None = None) -> str | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key=?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def increment_pattern(
        self, pattern_key: str, direction: str, now_iso: str
    ) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM pattern_evidence WHERE pattern_key=? AND direction=?",
                (pattern_key, direction),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO pattern_evidence (pattern_key, direction, count, last_seen, status)
                    VALUES (?, ?, 1, ?, 'observing')
                    """,
                    (pattern_key, direction, now_iso),
                )
                return {
                    "pattern_key": pattern_key,
                    "direction": direction,
                    "count": 1,
                    "status": "observing",
                    "just_confirmed": False,
                }
            new_count = int(row["count"]) + 1
            status = row["status"]
            just_confirmed = False
            conn.execute(
                """
                UPDATE pattern_evidence
                SET count=?, last_seen=?
                WHERE pattern_key=? AND direction=?
                """,
                (new_count, now_iso, pattern_key, direction),
            )
            return {
                "pattern_key": pattern_key,
                "direction": direction,
                "count": new_count,
                "status": status,
                "just_confirmed": just_confirmed,
            }

    def confirm_pattern(
        self,
        pattern_key: str,
        direction: str,
        *,
        confirmed_count: int,
        decision_reason: str,
        effect_action: str,
    ) -> None:
        with self.connection() as conn:
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
                (
                    datetime.now(timezone.utc).isoformat(),
                    confirmed_count,
                    decision_reason,
                    effect_action,
                    pattern_key,
                    direction,
                ),
            )

    def get_patterns(
        self, direction: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM pattern_evidence WHERE 1=1"
        params: list[Any] = []
        if direction:
            q += " AND direction=?"
            params.append(direction)
        if status:
            q += " AND status=?"
            params.append(status)
        q += " ORDER BY count DESC"
        with self.connection() as conn:
            rows = conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def insert_applied_change(
        self,
        pattern_key: str,
        direction: str,
        action: str,
        detail: str,
        *,
        occurrences: int | None = None,
        threshold: int | None = None,
    ) -> None:
        meta = detail
        if occurrences is not None and threshold is not None:
            meta = (
                f"occurrences={occurrences} (threshold≥{threshold}). {detail}"
            )
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO applied_changes (created_at, pattern_key, direction, action, detail)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    pattern_key,
                    direction,
                    action,
                    meta,
                ),
            )

    def applied_changes_since(self, day_prefix: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM applied_changes WHERE created_at >= ? ORDER BY id DESC",
                (day_prefix,),
            ).fetchall()
        return [dict(r) for r in rows]

    def applied_changes_on_day(self, day: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM applied_changes WHERE substr(created_at,1,10)=? ORDER BY id",
                (day,),
            ).fetchall()
        return [dict(r) for r in rows]

    def closed_trades_on_day(self, day: str) -> list[Position]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE status='closed' AND substr(COALESCE(exit_time, entry_time),1,10)=?
                ORDER BY id
                """,
                (day,),
            ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def insights_on_day(self, day: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM insights WHERE substr(created_at,1,10)=? ORDER BY id",
                (day,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_daily_report(self, day: str, path: str, summary: dict[str, Any]) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_reports (day, path, created_at, summary_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    path=excluded.path,
                    created_at=excluded.created_at,
                    summary_json=excluded.summary_json
                """,
                (
                    day,
                    path,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(summary),
                ),
            )

    def latest_daily_report(self) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_reports ORDER BY day DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _row_to_position(row: sqlite3.Row) -> Position:
        return Position(
            id=row["id"],
            symbol=row["symbol"],
            venue=Venue(row["venue"]),
            side=Side(row["side"]),
            strategy=row["strategy"],
            qty=row["qty"],
            entry_price=row["entry_price"],
            entry_time=datetime.fromisoformat(row["entry_time"]),
            take_profit=row["take_profit"],
            stop_loss=row["stop_loss"],
            confidence=row["confidence"] or 0.0,
            regime=row["regime"] or "unknown",
            features_json=row["features_json"] or "{}",
            exploration=bool(row["exploration"]),
            status=TradeStatus(row["status"]),
            exit_price=row["exit_price"],
            exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
            pnl=row["pnl"],
            fees=row["fees"] or 0.0,
            exit_reason=row["exit_reason"],
        )
