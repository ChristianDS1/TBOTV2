"""Session bucket + weekend tests."""

from __future__ import annotations

from datetime import datetime, timezone

from trading_system.learning.priority import SEED_PRIORITY_NAMES, ensure_priority_file, is_priority_setup
from trading_system.learning.sessions import session_bucket, session_info


def test_weekend_saturday_sunday():
    sat = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)  # Saturday
    sun = datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)  # Sunday
    mon = datetime(2026, 8, 17, 10, 0, tzinfo=timezone.utc)  # Monday
    assert session_bucket(sat) == "weekend"
    assert session_bucket(sun) == "weekend"
    assert session_bucket(mon) == "europe"
    info = session_info(sat)
    assert info["name"] == "weekend"
    assert any(b["name"] == "weekend" for b in info["buckets"])


def test_priority_seed_names():
    ensure_priority_file()
    for n in SEED_PRIORITY_NAMES:
        assert is_priority_setup(n)
    assert not is_priority_setup("triangle_sym_up")
