"""Tests for OOS pattern validation whitelist + gated walk."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_system.config import load_config
from trading_system.ml.validate_patterns import (
    HIST15_WINDOW,
    build_successful_whitelist,
    load_whitelist,
    run_pattern_validation,
    save_whitelist,
    walk_whitelisted,
)


def test_build_successful_whitelist(tmp_path: Path):
    rows = []
    for i in range(30):
        rows.append({"chart_pattern": "double_top", "label_win": 1 if i < 12 else 0})
    for i in range(20):
        rows.append({"chart_pattern": "noise_pat", "label_win": 1 if i < 1 else 0})
    csv = tmp_path / "examples.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    wl = build_successful_whitelist(csv, min_wins=10, min_wr=0.12)
    assert "double_top" in wl["names"]
    assert "noise_pat" not in wl["names"]
    save_whitelist(wl, tmp_path / "successful_patterns.json")
    assert load_whitelist(tmp_path / "successful_patterns.json") == {"double_top"}


def test_walk_only_whitelisted(tmp_path: Path):
    cfg = load_config()
    # Tiny simulate window — only empty or whitelisted names may appear
    rows = walk_whitelisted(
        "BTC/USDT",
        cfg,
        start="2026-07-20",
        end="2026-07-22",
        whitelist={"__never_match__"},
        step=10,
        simulate=True,
        max_bars=250,
    )
    assert rows == []


def test_validation_window_disjoint_from_hist15():
    assert HIST15_WINDOW[0] == "2026-07-27"
    # OOS dates must be before hist15 start
    assert "2026-07-22" < HIST15_WINDOW[0]


def test_run_pattern_validation_simulate(tmp_path: Path):
    cfg = load_config()
    ds = tmp_path / "hist15"
    ds.mkdir()
    # Minimal whitelist JSON (no examples.csv needed)
    wl = {
        "names": ["bb_mean_reversion", "double_top", "v_top", "v_bottom"],
        "patterns": [
            {"chart_pattern": "bb_mean_reversion", "n": 100, "wins": 15, "win_rate": 0.15},
            {"chart_pattern": "double_top", "n": 100, "wins": 12, "win_rate": 0.12},
            {"chart_pattern": "v_top", "n": 100, "wins": 16, "win_rate": 0.16},
            {"chart_pattern": "v_bottom", "n": 100, "wins": 13, "win_rate": 0.13},
        ],
        "min_wins": 10,
        "min_wr": 0.12,
    }
    (ds / "successful_patterns.json").write_text(json.dumps(wl), encoding="utf-8")
    out = tmp_path / "val"
    manifest = run_pattern_validation(
        cfg,
        start="2026-07-20",
        end="2026-07-22",
        from_dataset=ds,
        out_dir=out,
        step=15,
        simulate=True,
        max_bars=300,
    )
    assert manifest["bridge_live"] is False
    assert manifest["start"] == "2026-07-20"
    assert (out / "REPORT.md").exists()
    assert (out / "manifest.json").exists()
    for row_pat in (manifest.get("by_pattern") or {}):
        assert row_pat in wl["names"]
