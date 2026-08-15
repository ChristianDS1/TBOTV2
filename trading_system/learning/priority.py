"""Priority pattern list — elite setups applied obligatorily when seen."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.config import ROOT

logger = logging.getLogger(__name__)

DEFAULT_PRIORITY_PATH = ROOT / "data" / "ml" / "hist15_clean" / "priority_patterns.json"

# Seed: wins in all 3 confirmation runs ∩ hist15 successful whitelist
SEED_PRIORITY_NAMES = [
    "bb_mean_reversion",
    "double_top",
    "triangle_desc_down",
    "triangle_sym_down",
    "v_top",
]


def default_priority_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "source": "3_confirmations_intersect_hist15_successful",
        "min_net_wr_promote": 0.90,
        "min_n_promote": 10,
        "names": list(SEED_PRIORITY_NAMES),
        "patterns": [
            {
                "chart_pattern": n,
                "seed": True,
                "note": "obligatory when indicators pass",
            }
            for n in SEED_PRIORITY_NAMES
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def ensure_priority_file(path: Path | None = None) -> Path:
    out = Path(path) if path else DEFAULT_PRIORITY_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text(json.dumps(default_priority_payload(), indent=2), encoding="utf-8")
    return out


def load_priority(path: Path | None = None) -> dict[str, Any]:
    p = ensure_priority_file(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if "names" not in data:
        data["names"] = [x["chart_pattern"] for x in data.get("patterns", [])]
    return data


def priority_names(path: Path | None = None) -> set[str]:
    return {str(n) for n in load_priority(path).get("names", [])}


def is_priority_setup(chart_pattern: str | None, path: Path | None = None) -> bool:
    if not chart_pattern:
        return False
    return str(chart_pattern) in priority_names(path)


def promote_pattern(
    chart_pattern: str,
    *,
    path: Path | None = None,
    net_wr: float,
    n: int,
    session: str | None = None,
) -> bool:
    """Add to priority list if meets elite bar. Returns True if newly added."""
    data = load_priority(path)
    names = list(data.get("names") or [])
    if chart_pattern in names:
        return False
    min_wr = float(data.get("min_net_wr_promote") or 0.90)
    min_n = int(data.get("min_n_promote") or 10)
    if n < min_n or net_wr < min_wr:
        return False
    names.append(chart_pattern)
    data["names"] = names
    pats = list(data.get("patterns") or [])
    pats.append(
        {
            "chart_pattern": chart_pattern,
            "seed": False,
            "net_wr": net_wr,
            "n": n,
            "session": session,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    data["patterns"] = pats
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    out = Path(path) if path else DEFAULT_PRIORITY_PATH
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(
        "promoted to priority: %s wr=%.2f n=%d session=%s",
        chart_pattern,
        net_wr,
        n,
        session,
    )
    return True
