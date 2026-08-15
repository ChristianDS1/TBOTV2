"""Audits for Gen-5 clean dataset."""

from __future__ import annotations

from typing import Any


def audit_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    issues: list[str] = []
    if n == 0:
        return {"ok": False, "n": 0, "issues": ["empty_dataset"]}

    missing_family = sum(1 for r in rows if not r.get("strategy_family"))
    if missing_family:
        issues.append(f"missing_strategy_family={missing_family}")

    # fill_bar >= signal_bar; exit_bar >= fill_bar
    bad_order = sum(
        1
        for r in rows
        if int(r.get("exit_bar", 0)) < int(r.get("fill_bar", 0))
        or int(r.get("fill_bar", 0)) < int(r.get("signal_bar", 0))
    )
    if bad_order:
        issues.append(f"bar_order_violations={bad_order}")

    # duplicate key: family|symbol|side|timestamp
    keys = [
        f"{r.get('strategy_family')}|{r.get('symbol')}|{r.get('side')}|{r.get('timestamp')}"
        for r in rows
    ]
    dup = len(keys) - len(set(keys))
    if dup:
        issues.append(f"duplicate_keys={dup}")

    legacy_labels = sum(
        1 for r in rows if r.get("exit_reason") in ("trend_exit", "time_stop")
    )
    # trend_exit shouldn't appear from Gen-5 simulator
    if legacy_labels:
        issues.append(f"legacy_exit_reasons={legacy_labels}")

    gen_ok = all(r.get("generation") for r in rows)
    if not gen_ok:
        issues.append("missing_generation_tag")

    return {
        "ok": len(issues) == 0,
        "n": n,
        "issues": issues,
        "no_lookahead_bar_order_ok": bad_order == 0,
        "duplicates": dup,
        "families_present": sorted(
            {str(r.get("strategy_family")) for r in rows if r.get("strategy_family")}
        ),
    }
