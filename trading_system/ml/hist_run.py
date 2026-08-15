"""15-day historical ML learning run — incremental model during causal walk."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_system.config import ROOT, AppConfig, load_config
from trading_system.features import build_features
from trading_system.ml import HIST15_GENERATION
from trading_system.ml.audits import audit_examples
from trading_system.ml.hist_model import HistoricalWinModel
from trading_system.ml.reprocess import (
    FAMILIES,
    ReprocessConfig,
    _htf_bias_from_1m,
    _simulate_trade,
    fetch_ohlcv_history,
    _venue_for,
)
from trading_system.patterns import scan_patterns
from trading_system.strategies import (
    BBMeanReversionStrategy,
    ChartPatternStrategy,
    MomentumContinuationStrategy,
)

logger = logging.getLogger(__name__)


def _enrich_pre_move(feats: dict[str, Any], *, htf: str, ltf: Any, pats: list) -> dict[str, Any]:
    out = dict(feats)
    out["htf_bias"] = htf
    out["ltf_turn"] = ltf
    if pats:
        top = max(pats, key=lambda p: p.confidence)
        out.setdefault("chart_pattern", top.name)
        out.setdefault("chart_direction", top.direction)
    for k in (
        "rejection_bull",
        "rejection_bear",
        "touch_lower",
        "touch_upper",
        "macd_fast_bull_cross",
        "macd_fast_bear_cross",
    ):
        if k in out:
            out[k] = bool(out[k])
    return out


def walk_symbol_with_ml(
    symbol: str,
    cfg: AppConfig,
    rcfg: ReprocessConfig,
    model: HistoricalWinModel,
    *,
    global_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    venue = _venue_for(symbol, cfg)
    df = fetch_ohlcv_history(
        symbol, venue, cfg, days=rcfg.days, simulate=rcfg.simulate
    )
    if df is None or len(df) < 80:
        logger.warning(
            "skip %s: insufficient bars (%s)",
            symbol,
            0 if df is None else len(df),
        )
        return []
    if rcfg.max_bars:
        df = df.tail(rcfg.max_bars).reset_index(drop=True)

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
    start = max(cfg.strategy.bb_period + 10, 50)
    skipped_ml = 0

    for i in range(start, len(feat), rcfg.step):
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
            feats = _enrich_pre_move(
                dict(sig.features), htf=htf, ltf=ctx.get("ltf_turn"), pats=pats
            )
            feats["symbol"] = symbol
            feats["strategy_family"] = strat.name
            feats["side"] = sig.side.value
            feats["confidence"] = float(sig.confidence)

            # ML active: score every signal; soft-gate only after warmup fit
            p_win = model.predict_proba(feats)
            feats["p_win"] = p_win
            warmup = max(50, int(rcfg.retrain_every) * 2)
            if (
                model.model is not None
                and model.n_trained >= warmup
                and float(rcfg.ml_min_p_win) > 0
                and p_win < float(rcfg.ml_min_p_win)
            ):
                skipped_ml += 1
                continue
            # Blend confidence like live engine when model has fitted
            if model.model is not None:
                sig.confidence = 0.7 * float(sig.confidence) + 0.3 * (p_win * 100.0)
                feats["confidence"] = float(sig.confidence)

            setup = feats.get("setup") or feats.get("chart_pattern") or strat.name
            ex = _simulate_trade(
                side=sig.side,
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
            examples.append(ex)
            global_examples.append(ex)
            cool_until[strat.name] = i + 5

            # Incremental retrain on temporal prefix only
            if (
                len(global_examples) >= 20
                and len(global_examples) % max(int(rcfg.retrain_every), 1) == 0
            ):
                model.fit_examples(list(global_examples))

    logger.info(
        "%s done n=%d skipped_ml=%d model_n=%d",
        symbol,
        len(examples),
        skipped_ml,
        model.n_trained,
    )
    return examples


def write_report(
    dataset_dir: Path,
    *,
    manifest: dict[str, Any],
    model: HistoricalWinModel,
) -> Path:
    csv = dataset_dir / "examples.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    lines = [
        f"# Historical ML Learning Run - `{HIST15_GENERATION}`",
        "",
        "## Scope",
        f"- Days requested: `{manifest.get('days_requested')}`",
        f"- Simulate: `{manifest.get('simulate')}`",
        f"- Step: `{manifest.get('step')}`",
        f"- Retrain every: `{manifest.get('retrain_every')}`",
        f"- ml_min_p_win: `{manifest.get('ml_min_p_win')}`",
        f"- Objective north star: +50% equity/day (net of fees) - documented, not guaranteed by this walk",
        "",
        "## Examples",
        f"- n_examples: **{manifest.get('n_examples', len(df))}**",
        f"- by_family: `{json.dumps(manifest.get('by_family', {}))}`",
        f"- by_symbol: `{json.dumps(manifest.get('by_symbol', {}), default=str)}`",
        "",
        "## Net economics",
    ]
    if len(df):
        lines += [
            f"- win_rate (net>0): `{float(df['label_win'].mean()):.4f}`",
            f"- net_pnl mean/sum: `{float(df['net_pnl'].mean()):.4f}` / `{float(df['net_pnl'].sum()):.4f}`",
            f"- cost_erosion rate: `{float(df['cost_erosion'].mean()) if 'cost_erosion' in df else 0:.4f}`",
            f"- exit_reason: `{df['exit_reason'].value_counts().to_dict()}`",
        ]
    lines += [
        "",
        "## ML",
        f"- generation: `{model.generation}`",
        f"- backend: `{model.backend}`",
        f"- n_trained: `{model.n_trained}`",
        f"- brier: `{model.brier}`",
        f"- auc: `{model.auc}`",
        f"- loaded_legacy_weights: `False`",
        "",
        "## Periods",
        f"```json\n{json.dumps(manifest.get('periods', {}), indent=2, default=str)}\n```",
        "",
        "## Audits",
        f"```json\n{json.dumps(manifest.get('audits', {}), indent=2, default=str)}\n```",
        "",
        "## Paths",
        f"- Dataset: `{dataset_dir}`",
        f"- Model: `{model.artifact_dir}`",
        "",
        "## Note",
        "Bot 24/7 was **not** started. Activate later with a dedicated model-dir wiring if desired.",
        "",
    ]
    path = dataset_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_historical_ml(
    cfg: AppConfig | None = None,
    *,
    days: int = 15,
    out_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    simulate: bool = False,
    max_bars: int | None = None,
    step: int = 5,
    retrain_every: int = 25,
    ml_min_p_win: float = 0.0,
) -> dict[str, Any]:
    """
    Causal 15d walk with EXIT FIX labels + incremental ML.

    Default step=5 for tractable runtime on real 15d×6 symbols; override with --step 1.
    """
    cfg = cfg or load_config()
    dataset = Path(out_dir) if out_dir else ROOT / "data" / "ml" / "hist15_clean"
    mdir = Path(model_dir) if model_dir else ROOT / "models" / "hist15_clean"
    dataset.mkdir(parents=True, exist_ok=True)
    rcfg = ReprocessConfig(
        days=days,
        out_dir=dataset,
        simulate=simulate,
        max_bars=max_bars,
        step=max(1, int(step)),
        ml_min_p_win=float(ml_min_p_win),
        retrain_every=max(1, int(retrain_every)),
        generation=HIST15_GENERATION,
    )
    model = HistoricalWinModel(mdir, generation=HIST15_GENERATION)
    symbols = list(cfg.symbols.crypto) + list(cfg.symbols.forex)
    all_ex: list[dict[str, Any]] = []
    periods: dict[str, Any] = {}

    for sym in symbols:
        logger.info("historical-ml-run %s ...", sym)
        try:
            rows = walk_symbol_with_ml(sym, cfg, rcfg, model, global_examples=all_ex)
        except Exception as e:
            logger.exception("walk failed %s: %s", sym, e)
            periods[sym] = {"error": str(e), "n": 0}
            continue
        periods[sym] = {"n": len(rows)}
        if rows:
            periods[sym]["from"] = rows[0]["timestamp"]
            periods[sym]["to"] = rows[-1]["timestamp"]

    # Final fit on full temporal prefix
    if len(all_ex) >= 20:
        model.fit_examples(all_ex)

    audits = audit_examples(all_ex)
    df = pd.DataFrame(all_ex)
    csv_path = dataset / "examples.csv"
    df.to_csv(csv_path, index=False)

    by_family = {
        f: int((df["strategy_family"] == f).sum()) if len(df) else 0 for f in FAMILIES
    }
    by_symbol = df["symbol"].value_counts().to_dict() if len(df) else {}
    manifest = {
        "generation": HIST15_GENERATION,
        "days_requested": days,
        "simulate": simulate,
        "step": rcfg.step,
        "retrain_every": rcfg.retrain_every,
        "ml_min_p_win": rcfg.ml_min_p_win,
        "symbols": symbols,
        "periods": periods,
        "n_examples": len(all_ex),
        "by_family": by_family,
        "by_symbol": by_symbol,
        "audits": audits,
        "csv": str(csv_path),
        "model_dir": str(mdir),
        "model_backend": model.backend,
        "model_brier": model.brier,
        "model_auc": model.auc,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (dataset / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    report = write_report(dataset, manifest=manifest, model=model)
    manifest["report"] = str(report)
    (dataset / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest
