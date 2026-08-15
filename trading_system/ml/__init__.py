"""Offline historical ML: 15d learning run with EXIT FIX labels."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _short_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


HIST15_GENERATION = f"HIST15_CLEAN_{_short_sha()}"

__all__ = ["HIST15_GENERATION"]
