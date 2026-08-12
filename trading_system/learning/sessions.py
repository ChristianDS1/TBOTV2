"""UTC session buckets for session-aware pattern learning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from trading_system.config import SessionBucketConfig


DEFAULT_SESSION_BUCKETS: list[SessionBucketConfig] = [
    SessionBucketConfig(name="asia", start_hour_utc=0.0, end_hour_utc=7.0),
    SessionBucketConfig(name="europe", start_hour_utc=7.0, end_hour_utc=12.0),
    SessionBucketConfig(name="us_open", start_hour_utc=12.0, end_hour_utc=16.0),
    SessionBucketConfig(name="us_afternoon", start_hour_utc=16.0, end_hour_utc=21.0),
    SessionBucketConfig(name="night", start_hour_utc=21.0, end_hour_utc=24.0),
]


def _hour_utc(ts: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.hour + ts.minute / 60.0 + ts.second / 3600.0


def session_bucket(
    ts: datetime | None = None,
    buckets: Sequence[SessionBucketConfig] | None = None,
) -> str:
    """
    Map a timestamp to a named UTC session bucket.
    Buckets are [start, end); end=24 means until midnight.
    """
    now = ts or datetime.now(timezone.utc)
    hour = _hour_utc(now)
    specs = list(buckets) if buckets else DEFAULT_SESSION_BUCKETS
    for b in specs:
        start = float(b.start_hour_utc)
        end = float(b.end_hour_utc)
        if end > start:
            if start <= hour < end:
                return b.name
        else:
            if hour >= start or hour < end:
                return b.name
    return specs[-1].name if specs else "unknown"


def with_session(prefix_session: str, key: str) -> str:
    if key.startswith("session="):
        return key
    return f"session={prefix_session}|{key}"


def _fmt_hour(h: float) -> str:
    hh = int(h) % 24
    mm = int(round((h - int(h)) * 60)) % 60
    if mm:
        return f"{hh:02d}:{mm:02d}"
    return f"{hh:02d}:00"


def session_hours_label(bucket: SessionBucketConfig) -> str:
    end = float(bucket.end_hour_utc)
    end_s = "24:00" if end >= 24 else _fmt_hour(end)
    return f"{_fmt_hour(float(bucket.start_hour_utc))}–{end_s} UTC"


def session_info(
    ts: datetime | None = None,
    buckets: Sequence[SessionBucketConfig] | None = None,
) -> dict[str, Any]:
    """Current session name + hours window for monitor/report."""
    specs = list(buckets) if buckets else DEFAULT_SESSION_BUCKETS
    now = ts or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    name = session_bucket(now, specs)
    match = next((b for b in specs if b.name == name), None)
    hours = session_hours_label(match) if match else "—"
    return {
        "name": name,
        "hours": hours,
        "start_hour_utc": float(match.start_hour_utc) if match else None,
        "end_hour_utc": float(match.end_hour_utc) if match else None,
        "utc_now": now.strftime("%H:%M:%S"),
        "buckets": [{"name": b.name, "hours": session_hours_label(b)} for b in specs],
    }


def pattern_session_name(pattern_key: str) -> str | None:
    if not pattern_key.startswith("session="):
        return None
    rest = pattern_key[len("session=") :]
    return rest.split("|", 1)[0]
