"""UTC session buckets for session-aware pattern learning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

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
            # wrap past midnight (e.g. 22 → 6)
            if hour >= start or hour < end:
                return b.name
    return specs[-1].name if specs else "unknown"


def with_session(prefix_session: str, key: str) -> str:
    if key.startswith("session="):
        return key
    return f"session={prefix_session}|{key}"
