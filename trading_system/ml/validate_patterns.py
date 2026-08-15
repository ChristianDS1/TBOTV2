"""OOS pattern validation — whitelist from hist15, causal walk, no retrain/bridge."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_system.config import ROOT, AppConfig, load_config
from trading_system.features import build_features
from trading_system.ml.reprocess import (
    FAMILIES,
    _htf_bias_from_1m,
    _parse_utc,
    _simulate_trade,
    _venue_for,
    fetch_ohlcv_range,
)
from trading_system.patterns import scan_patterns
from trading_system.strategies import (
    BBMeanReversionStrategy,
    ChartPatternStrategy,
    MomentumContinuationStrategy,
)
from trading_system.types import Side

logger = logging.getLogger(__name__)

HIST15_WINDOW = ("2026-07-27", "2026-08-14")
DEFAULT_MIN_WINS = 10
DEFAULT_MIN_WR = 0.12


def build_successful_whitelist(
    examples_csv: Path | str,
    *,
    min_wins: int = DEFAULT_MIN_WINS,
    min_wr: float = DEFAULT_MIN_WR,
) -> dict[str, Any]:
    """Freeze chart_pattern names that cleared hist15 win threshold."""
    path = Path(examples_csv)
    df = pd.read_csv(path)
    if "chart_pattern" not in df.columns or "label_win" not in df.columns:
        raise ValueError("examples.csv needs chart_pattern and label_win")
    g = (
        df.groupby("chart_pattern", dropna=False)
        .agg(n=("label_win", "size"), wins=("label_win", "sum"))
        .reset_index()
    )
    g["win_rate"] = g["wins"] / g["n"].clip(lower=1)
    ok = g[(g["wins"] >= int(min_wins)) & (g["win_rate"] >= float(min_wr))].copy()
    ok = ok.sort_values(["win_rate", "wins"], ascending=[False, False])
    patterns = [
        {
            "chart_pattern": str(r.chart_pattern),
            "n": int(r.n),
            "wins": int(r.wins),
            "win_rate": float(r.win_rate),
        }
        for r in ok.itertuples(index=False)
    ]
    return {
        "source": str(path),
        "min_wins": int(min_wins),
        "min_wr": float(min_wr),
        "n_patterns": len(patterns),
        "patterns": patterns,
        "names": [p["chart_pattern"] for p in patterns],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Offline hist15 whitelist only. Live bridge (pattern_evidence / models/artifacts) "
            "is deferred until OOS validation confirms these patterns."
        ),
    }


def save_whitelist(wl: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(wl, indent=2), encoding="utf-8")
    return out


def load_whitelist(path: Path | str) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = data.get("names") or [p["chart_pattern"] for p in data.get("patterns", [])]
    return {str(n) for n in names}


def _pattern_key(feats: dict[str, Any], strat_name: str) -> str:
    return str(
        feats.get("chart_pattern")
        or feats.get("setup")
        or feats.get("setup_type")
        or strat_name
    )


def _enrich(feats: dict[str, Any], *, htf: str, pats: list) -> dict[str, Any]:
    out = dict(feats)
    out["htf_bias"] = htf
    if pats and not out.get("chart_pattern"):
        top = max(pats, key=lambda p: p.confidence)
        out["chart_pattern"] = top.name
        out["chart_direction"] = top.direction
    return out


def walk_whitelisted(
    symbol: str,
    cfg: AppConfig,
    *,
    start: str,
    end: str,
    whitelist: set[str],
    step: int = 5,
    simulate: bool = False,
    max_bars: int | None = None,
) -> list[dict[str, Any]]:
    """Causal walk: only take strategy signals whose chart_pattern is whitelisted."""
    venue = _venue_for(symbol, cfg)
    start_dt = _parse_utc(start)
    end_dt = _parse_utc(end)
    if (
        end_dt.hour == 0
        and end_dt.minute == 0
        and end_dt.second == 0
        and end_dt.microsecond == 0
    ):
        from datetime import timedelta

        end_dt = end_dt + timedelta(days=1) - timedelta(milliseconds=1)

    df = fetch_ohlcv_range(
        symbol, venue, cfg, start=start_dt, end=end_dt, simulate=simulate, warmup_days=2.0
    )
    if df is None or len(df) < 80:
        logger.warning(
            "skip %s: insufficient bars (%s)",
            symbol,
            0 if df is None else len(df),
        )
        return []
    if max_bars:
        df = df.tail(max_bars).reset_index(drop=True)

    feat = build_features(
        df,
        bb_period=cfg.strategy.bb_period,
        bb_std=cfg.strategy.bb_std,
        rsi_period=cfg.strategy.rsi_period,
        macd_fast=cfg.strategy.macd_fast,
        macd_slow=cfg.strategy.macd_slow,
    )
    strategies = [
        BBMeanReversionStrategy(),
        MomentumContinuationStrategy(),
        ChartPatternStrategy(),
    ]
    examples: list[dict[str, Any]] = []
    cool_until = {s.name: 0 for s in strategies}
    warm = max(cfg.strategy.bb_period + 10, 50)
    skipped = 0

    for i in range(warm, len(feat), max(int(step), 1)):
        ts = pd.Timestamp(feat.iloc[i]["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if ts < pd.Timestamp(start_dt) or ts > pd.Timestamp(end_dt):
            continue

        window = df.iloc[: i + 1].reset_index(drop=True)
        htf = _htf_bias_from_1m(window.tail(min(len(window), 500)))
        pats = scan_patterns(window.tail(min(120, len(window))))
        ctx = {
            "htf_bias": htf,
            "htf_votes": {},
            "patterns": pats,
            "htf_patterns": [],
            "ltf_turn": None,
            "ltf_patterns": [],
        }
        for strat in strategies:
            if i < cool_until[strat.name]:
                continue
            sig = strat.evaluate(symbol, venue, window, cfg.strategy, context=ctx)
            if sig is None:
                continue
            feats = _enrich(dict(sig.features), htf=htf, pats=pats)
            feats["symbol"] = symbol
            feats["strategy_family"] = strat.name
            feats["side"] = sig.side.value
            key = _pattern_key(feats, strat.name)
            feats["chart_pattern"] = key
            if key not in whitelist:
                skipped += 1
                continue
            # Direction from pattern when available
            direction = str(feats.get("chart_direction") or "").lower()
            if direction == "bullish":
                side = Side.CALL
            elif direction == "bearish":
                side = Side.PUT
            else:
                side = sig.side

            setup = feats.get("setup") or key
            ex = _simulate_trade(
                side=side,
                strategy=strat.name,
                entry_i=i,
                feat=feat,
                df=df,
                cfg=cfg,
                venue=venue,
                confidence=float(sig.confidence),
                setup_type=str(setup),
                features=feats,
            )
            if not ex:
                continue
            ex["hist15_wr"] = None
            examples.append(ex)
            cool_until[strat.name] = i + 5

    logger.info("%s validate n=%d skipped_non_wl=%d", symbol, len(examples), skipped)
    return examples


def _hist15_wr_map(wl: dict[str, Any]) -> dict[str, float]:
    return {p["chart_pattern"]: float(p["win_rate"]) for p in wl.get("patterns", [])}


def write_validation_report(
    out_dir: Path,
    *,
    manifest: dict[str, Any],
    df: pd.DataFrame,
    wl: dict[str, Any],
) -> Path:
    wr_map = _hist15_wr_map(wl)
    lines = [
        "# Pattern validation OOS (3d)",
        "",
        "## Scope",
        "",
        f"- Window: `{manifest.get('start')}` → `{manifest.get('end')}` UTC",
        f"- Hist15 train window (disjoint): `{HIST15_WINDOW[0]}` → `{HIST15_WINDOW[1]}`",
        f"- Whitelist: `{manifest.get('whitelist_path')}`",
        f"- n trades: **{manifest.get('n_examples', 0)}**",
        f"- net wins / losses: **{manifest.get('wins', 0)}** / **{manifest.get('losses', 0)}**",
        f"- net WR: **{manifest.get('win_rate', 0):.2%}**",
        "",
        "## Live bridge (deferred)",
        "",
        "This run does **not** write `pattern_evidence` or wire `models/hist15_clean` into the live engine.",
        "After OOS confirms identification + success, a later bridge will pass learning to the bot so it",
        "knows which patterns to follow for entries / continuation preference / soft-reject — on top of",
        "existing EXIT FIX exits.",
        "",
        "## Other PC",
        "",
        "```text",
        "git pull origin main",
        "python -m trading_system historical-pattern-validate \\",
        "  --start 2026-07-20 --end 2026-07-22 \\",
        "  --from-dataset data/ml/hist15_clean",
        "```",
        "",
        "Needs `data/ml/hist15_clean/successful_patterns.json` (in git).",
        "`examples.csv` is gitignored — only required to rebuild the whitelist.",
        "",
        "## By pattern vs hist15 baseline",
        "",
        "| pattern | n | wins | OOS WR | hist15 WR |",
        "|---|---:|---:|---:|---:|",
    ]
    if len(df):
        g = (
            df.groupby("chart_pattern")
            .agg(n=("label_win", "size"), wins=("label_win", "sum"))
            .reset_index()
        )
        g["wr"] = g["wins"] / g["n"]
        for r in g.sort_values("n", ascending=False).itertuples(index=False):
            base = wr_map.get(str(r.chart_pattern))
            base_s = f"{base:.2%}" if base is not None else "—"
            lines.append(
                f"| {r.chart_pattern} | {int(r.n)} | {int(r.wins)} | {float(r.wr):.2%} | {base_s} |"
            )
    else:
        lines.append("| _(none)_ | 0 | 0 | — | — |")

    lines.extend(
        [
            "",
            "## Whitelist names",
            "",
            ", ".join(f"`{n}`" for n in wl.get("names", [])),
            "",
            f"- Dataset: `{out_dir}`",
            f"- Created: `{manifest.get('created_at')}`",
            "",
        ]
    )
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_pattern_validation(
    cfg: AppConfig | None = None,
    *,
    start: str = "2026-07-20",
    end: str = "2026-07-22",
    from_dataset: str | Path | None = None,
    out_dir: str | Path | None = None,
    whitelist_path: str | Path | None = None,
    min_wins: int = DEFAULT_MIN_WINS,
    min_wr: float = DEFAULT_MIN_WR,
    step: int = 5,
    simulate: bool = False,
    max_bars: int | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    dataset = Path(from_dataset) if from_dataset else ROOT / "data" / "ml" / "hist15_clean"
    out = Path(out_dir) if out_dir else ROOT / "data" / "ml" / "hist15_validate_3d"
    out.mkdir(parents=True, exist_ok=True)

    wl_path = Path(whitelist_path) if whitelist_path else dataset / "successful_patterns.json"
    examples_csv = dataset / "examples.csv"
    if wl_path.exists():
        wl = json.loads(wl_path.read_text(encoding="utf-8"))
        if "names" not in wl and "patterns" in wl:
            wl["names"] = [p["chart_pattern"] for p in wl["patterns"]]
    elif examples_csv.exists():
        wl = build_successful_whitelist(examples_csv, min_wins=min_wins, min_wr=min_wr)
        save_whitelist(wl, wl_path)
    else:
        raise FileNotFoundError(
            f"Need {wl_path} or {examples_csv} to build whitelist"
        )
    if not wl_path.exists():
        save_whitelist(wl, wl_path)

    names = set(wl.get("names") or [])
    if not names:
        raise ValueError("empty whitelist")

    # Guard: validation window must not overlap hist15
    vs = _parse_utc(start)
    ve = _parse_utc(end)
    hs = _parse_utc(HIST15_WINDOW[0])
    he = _parse_utc(HIST15_WINDOW[1])
    if vs <= he and ve >= hs:
        logger.warning(
            "validation window overlaps hist15 (%s–%s); continuing as requested",
            HIST15_WINDOW[0],
            HIST15_WINDOW[1],
        )

    symbols = list(cfg.symbols.crypto) + list(cfg.symbols.forex)
    all_ex: list[dict[str, Any]] = []
    periods: dict[str, Any] = {}
    wr_map = _hist15_wr_map(wl)

    for sym in symbols:
        logger.info("validate %s ...", sym)
        try:
            rows = walk_whitelisted(
                sym,
                cfg,
                start=start,
                end=end,
                whitelist=names,
                step=step,
                simulate=simulate,
                max_bars=max_bars,
            )
        except Exception as e:
            logger.exception("validate failed %s: %s", sym, e)
            periods[sym] = {"error": str(e), "n": 0}
            continue
        for r in rows:
            r["hist15_wr"] = wr_map.get(str(r.get("chart_pattern")))
        all_ex.extend(rows)
        periods[sym] = {"n": len(rows)}
        if rows:
            periods[sym]["from"] = rows[0]["timestamp"]
            periods[sym]["to"] = rows[-1]["timestamp"]

    df = pd.DataFrame(all_ex)
    csv_path = out / "examples.csv"
    if len(df):
        df.to_csv(csv_path, index=False)
    else:
        csv_path.write_text("", encoding="utf-8")

    wins = int(df["label_win"].sum()) if len(df) else 0
    n = len(df)
    losses = n - wins
    by_family = {
        f: int((df["strategy_family"] == f).sum()) if len(df) else 0 for f in FAMILIES
    }
    by_pattern = (
        df.groupby("chart_pattern")["label_win"]
        .agg(["count", "sum"])
        .rename(columns={"count": "n", "sum": "wins"})
        .assign(wr=lambda x: x["wins"] / x["n"])
        .to_dict(orient="index")
        if len(df)
        else {}
    )
    # JSON-serialize nested
    by_pattern_out = {
        k: {"n": int(v["n"]), "wins": int(v["wins"]), "wr": float(v["wr"])}
        for k, v in by_pattern.items()
    }

    manifest = {
        "kind": "pattern_validation_oos",
        "start": start,
        "end": end,
        "hist15_window": list(HIST15_WINDOW),
        "simulate": simulate,
        "step": step,
        "whitelist_path": str(wl_path),
        "whitelist_names": sorted(names),
        "symbols": symbols,
        "periods": periods,
        "n_examples": n,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / n) if n else 0.0,
        "by_family": by_family,
        "by_pattern": by_pattern_out,
        "bridge_live": False,
        "bridge_note": (
            "Deferred until OOS confirms; then pass learning to bot "
            "(pattern_evidence + optional hist15 model)."
        ),
        "csv": str(csv_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    write_validation_report(out, manifest=manifest, df=df, wl=wl)
    return manifest
