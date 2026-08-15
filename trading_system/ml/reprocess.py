"""Causal historical reprocess with Gen-5 exit labeling."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_system.config import ROOT, AppConfig, ExitConfig, load_config
from trading_system.execution.edge import estimate_close_net, position_notional
from trading_system.execution.exit_engine import decide_exit
from trading_system.features import build_features, latest_feature_dict
from trading_system.ml import HIST15_GENERATION
from trading_system.ml.audits import audit_examples
from trading_system.patterns import combine_htf_votes, macd_htf_bias, scan_patterns
from trading_system.strategies import (
    BBMeanReversionStrategy,
    ChartPatternStrategy,
    MomentumContinuationStrategy,
    compute_sl_from_margin_pct,
)
from trading_system.types import Position, Side, TradeStatus, Venue

logger = logging.getLogger(__name__)

FAMILIES = ("bb_mean_reversion", "momentum_continuation", "bulkowski_pattern")


@dataclass
class ReprocessConfig:
    days: int = 15
    out_dir: Path = ROOT / "data" / "ml" / "hist15_clean"
    simulate: bool = False
    max_bars: int | None = None  # test hook
    step: int = 1  # evaluate every N bars (1=full)
    ml_min_p_win: float = 0.0  # soft gate when model present
    retrain_every: int = 25
    generation: str = HIST15_GENERATION


def _venue_for(symbol: str, cfg: AppConfig) -> Venue:
    return Venue.FOREX if symbol in cfg.symbols.forex else Venue.CRYPTO


def fetch_ohlcv_history(
    symbol: str,
    venue: Venue,
    cfg: AppConfig,
    *,
    days: int,
    simulate: bool,
) -> pd.DataFrame:
    """Fetch ~days of 1m OHLCV. Document shorter windows in caller."""
    if simulate or venue == Venue.CRYPTO and simulate:
        from trading_system.data.crypto import SimulatedCryptoAdapter

        bars = min(days * 24 * 60, 3000)
        return SimulatedCryptoAdapter(seed=abs(hash(symbol)) % 10_000).get_ohlcv(
            symbol, "1m", bars
        )

    if venue == Venue.CRYPTO:
        from trading_system.data.crypto import CryptoAdapter

        ex = CryptoAdapter(cfg.crypto.exchange, sandbox=cfg.crypto.sandbox)
        # paginate ~1000 bars/request
        need = days * 24 * 60
        timeframe = "1m"
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        frames: list[pd.DataFrame] = []
        got = 0
        cursor = since
        while got < need:
            batch = 1000
            try:
                raw = ex.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=cursor, limit=batch
                )
            except Exception as e:
                logger.warning("crypto fetch %s failed: %s", symbol, e)
                break
            if not raw:
                break
            df = pd.DataFrame(
                raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            frames.append(df)
            got += len(df)
            last_ms = int(raw[-1][0])
            nxt = last_ms + 60_000
            if nxt <= cursor:
                break
            cursor = nxt
            if len(raw) < batch:
                break
        if not frames:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        out = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
        return out.sort_values("timestamp").reset_index(drop=True)

    # Forex via yfinance ÃÃÃ¶ use 1m for ~5d or 5m for longer
    from trading_system.data.forex import ForexAdapter

    fx = ForexAdapter(cfg.forex_session, provider=cfg.forex.provider)
    if days <= 5:
        return fx.get_ohlcv(symbol, "1m", min(days * 24 * 60, 5000))
    # longer: 15m resample proxy labeled as 1m steps for offline (document)
    df = fx.get_ohlcv(symbol, "15m", min(days * 24 * 4, 2000))
    return df


def _htf_bias_from_1m(df_1m: pd.DataFrame) -> str:
    if df_1m is None or len(df_1m) < 50:
        return "unknown"
    x = df_1m.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.set_index("timestamp")
    votes = {}
    for tf, rule in (("15m", "15min"), ("30m", "30min"), ("1h", "1h")):
        agg = (
            x.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )
        votes[tf] = macd_htf_bias(agg)
    return combine_htf_votes(votes)


def _row_dict(feat_df: pd.DataFrame, i: int) -> dict[str, Any]:
    row = feat_df.iloc[i]
    d = latest_feature_dict(feat_df.iloc[: i + 1])
    for k in (
        "rejection_bear",
        "rejection_bull",
        "macd_fast_bear_cross",
        "macd_fast_bull_cross",
        "macd_fast_hist",
        "macd_fast_hist_prev",
        "macd_fast_hist_prev2",
        "rsi",
        "rsi_prev",
        "macd_slow_hist",
        "macd_slow_hist_prev",
    ):
        if k in row.index:
            v = row[k]
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            d[k] = bool(v) if isinstance(v, (bool, np.bool_)) else v
    return d


def _simulate_trade(
    *,
    side: Side,
    strategy: str,
    entry_i: int,
    feat: pd.DataFrame,
    df: pd.DataFrame,
    cfg: AppConfig,
    venue: Venue,
    confidence: float,
    setup_type: str | None,
    features: dict[str, Any],
) -> dict[str, Any] | None:
    entry_row = feat.iloc[entry_i]
    entry_mark = float(entry_row["close"])
    fee_bps, slip_bps = cfg.execution.costs_for_venue(venue)
    slip = slip_bps / 10_000.0
    if side == Side.CALL:
        fill = entry_mark * (1 + slip)
    else:
        fill = entry_mark * (1 - slip)
    margin = float(cfg.capital.base_trade_size)
    lev = max(1.0, float(cfg.execution.leverage or 1.0))
    notional = margin * lev
    entry_fee = notional * (fee_bps + slip_bps) / 10_000.0
    sl, _, _, budget_cash = compute_sl_from_margin_pct(
        side=side,
        price=entry_mark,
        margin=margin,
        leverage=lev,
        sl_margin_pct=float(cfg.strategy.sl_margin_pct),
        exit_fee_bps=fee_bps + slip_bps,
    )
    entry_ts = pd.Timestamp(entry_row["timestamp"])
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize("UTC")
    else:
        entry_ts = entry_ts.tz_convert("UTC")

    pos = Position(
        symbol=str(features.get("symbol") or "SYM"),
        venue=venue,
        side=side,
        strategy=strategy,
        qty=margin,
        entry_price=fill,
        entry_mark=entry_mark,
        entry_time=entry_ts.to_pydatetime(),
        stop_loss=sl,
        confidence=confidence,
        status=TradeStatus.OPEN,
        leverage=lev,
        notional=notional,
        fees=entry_fee,
        features_json=json.dumps(
            {
                **{k: features.get(k) for k in ("rsi", "bb_width", "macd_fast_hist", "macd_slow_hist", "pct_from_mid")},
                "sl_budget_cash": budget_cash,
                "sl_mode": "margin_pct",
                "strategy_family": strategy,
                "setup_type": setup_type,
            }
        ),
    )
    feat_state: dict[str, Any] = json.loads(pos.features_json)
    exit_cfg: ExitConfig = cfg.exit
    min_hold = float(cfg.strategy.min_hold_minutes or 1)

    # horizon windows for MFE/MAE labels (bars ~ minutes on 1m)
    horizons = {1: None, 2: None, 3: None, 5: None}
    mfe_h = {1: 0.0, 2: 0.0, 3: 0.0, 5: 0.0}
    mae_h = {1: 0.0, 2: 0.0, 3: 0.0, 5: 0.0}

    exit_reason = "end_of_data"
    exit_i = len(feat) - 1
    exit_px = float(feat.iloc[-1]["close"])

    for j in range(entry_i + 1, len(feat)):
        mark = float(feat.iloc[j]["close"])
        ts = pd.Timestamp(feat.iloc[j]["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        bars_held = j - entry_i
        for h in horizons:
            if bars_held <= h:
                move = (
                    (mark - entry_mark) / entry_mark * 100.0
                    if side == Side.CALL
                    else (entry_mark - mark) / entry_mark * 100.0
                )
                mfe_h[h] = max(mfe_h[h], move)
                mae_h[h] = min(mae_h[h], move)

        # SL
        if pos.stop_loss is not None:
            if side == Side.CALL and mark <= float(pos.stop_loss):
                exit_reason, exit_i, exit_px = "stop_loss", j, mark
                break
            if side == Side.PUT and mark >= float(pos.stop_loss):
                exit_reason, exit_i, exit_px = "stop_loss", j, mark
                break

        row = _row_dict(feat, j)
        row["htf_bias"] = features.get("htf_bias")
        # chart reversal from patterns on window
        win = df.iloc[max(0, j - 80) : j + 1]
        pats = scan_patterns(win)
        row["chart_reversal_bear"] = any(p.direction == "bearish" for p in pats)
        row["chart_reversal_bull"] = any(p.direction == "bullish" for p in pats)

        decision = decide_exit(
            pos,
            mark,
            row,
            feat_state,
            exit_cfg,
            fee_bps=fee_bps,
            slip_bps=slip_bps,
            min_hold_minutes=min_hold,
            now=ts.to_pydatetime(),
        )
        if decision.reason:
            exit_reason, exit_i, exit_px = decision.reason, j, mark
            break

    # PnL
    direction = 1 if side == Side.CALL else -1
    exit_slip = slip_bps / 10_000.0
    if side == Side.CALL:
        exit_fill = exit_px * (1 - exit_slip)
    else:
        exit_fill = exit_px * (1 + exit_slip)
    gross = direction * (exit_px - entry_mark) / entry_mark * notional
    exit_fee = notional * (fee_bps + slip_bps) / 10_000.0
    raw = direction * (exit_fill - fill) / fill * notional
    net = raw - exit_fee
    hold_min = float(exit_i - entry_i)

    final_move = (
        (exit_px - entry_mark) / entry_mark * 100.0
        if side == Side.CALL
        else (entry_mark - exit_px) / entry_mark * 100.0
    )
    mfe_pct = float(feat_state.get("mfe_pct") or max(0.0, final_move))
    mae_pct = float(feat_state.get("mae_pct") or min(0.0, final_move))
    giveback = float(feat_state.get("giveback_pct") or 0.0)

    return {
        "strategy_family": strategy,
        "setup_type": setup_type or strategy,
        "symbol": pos.symbol,
        "side": side.value,
        "venue": venue.value,
        "timestamp": entry_ts.isoformat(),
        "entry_mark": entry_mark,
        "entry_fill": fill,
        "exit_price": exit_px,
        "exit_reason": exit_reason,
        "hold_minutes": hold_min,
        "gross_pnl": gross,
        "fees": entry_fee + exit_fee,
        "net_pnl": net,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "giveback_pct": giveback,
        "peak_pnl": float(feat_state.get("peak_pnl") or 0.0),
        "mfe_1m": mfe_h[1],
        "mfe_2m": mfe_h[2],
        "mfe_3m": mfe_h[3],
        "mfe_5m": mfe_h[5],
        "mae_1m": mae_h[1],
        "mae_2m": mae_h[2],
        "mae_3m": mae_h[3],
        "mae_5m": mae_h[5],
        "future_return_pct": final_move,
        "future_net_return": net,
        "confidence": confidence,
        "rsi": features.get("rsi"),
        "bb_width": features.get("bb_width"),
        "macd_fast_hist": features.get("macd_fast_hist"),
        "macd_slow_hist": features.get("macd_slow_hist"),
        "pct_from_mid": features.get("pct_from_mid"),
        "htf_bias": features.get("htf_bias"),
        "ltf_turn": features.get("ltf_turn"),
        "rejection_bull": bool(features.get("rejection_bull")),
        "rejection_bear": bool(features.get("rejection_bear")),
        "touch_lower": bool(features.get("touch_lower")),
        "touch_upper": bool(features.get("touch_upper")),
        "macd_fast_bull_cross": bool(features.get("macd_fast_bull_cross")),
        "macd_fast_bear_cross": bool(features.get("macd_fast_bear_cross")),
        "chart_pattern": features.get("chart_pattern") or setup_type,
        "chart_direction": features.get("chart_direction"),
        "p_win": features.get("p_win"),
        "label_win": 1 if net > 0 else 0,
        "cost_erosion": 1 if (gross > 0 and net <= 0) else 0,
        "generation": HIST15_GENERATION,
        "signal_bar": int(entry_i),
        "fill_bar": int(entry_i),
        "exit_bar": int(exit_i),
    }


def reprocess_symbol(
    symbol: str,
    cfg: AppConfig,
    rcfg: ReprocessConfig,
) -> list[dict[str, Any]]:
    venue = _venue_for(symbol, cfg)
    df = fetch_ohlcv_history(
        symbol, venue, cfg, days=rcfg.days, simulate=rcfg.simulate
    )
    if df is None or len(df) < 80:
        logger.warning("skip %s: insufficient bars (%s)", symbol, 0 if df is None else len(df))
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
        }
        for strat in strategies:
            if i < cool_until[strat.name]:
                continue
            sig = strat.evaluate(symbol, venue, window, cfg.strategy, context=ctx)
            if sig is None:
                continue
            feats = dict(sig.features)
            feats["symbol"] = symbol
            feats["htf_bias"] = htf
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
            if ex:
                examples.append(ex)
                # avoid overlapping same-family spam
                cool_until[strat.name] = i + 5
    return examples


def run_reprocess(
    cfg: AppConfig | None = None,
    *,
    days: int = 15,
    out_dir: str | Path | None = None,
    simulate: bool = False,
    max_bars: int | None = None,
    step: int = 1,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    rcfg = ReprocessConfig(
        days=days,
        out_dir=Path(out_dir) if out_dir else ROOT / "data" / "ml" / "hist15_clean",
        simulate=simulate,
        max_bars=max_bars,
        step=step,
    )
    rcfg.out_dir.mkdir(parents=True, exist_ok=True)
    symbols = list(cfg.symbols.crypto) + list(cfg.symbols.forex)
    all_ex: list[dict[str, Any]] = []
    periods: dict[str, Any] = {}

    for sym in symbols:
        logger.info("reprocess %s ...", sym)
        try:
            rows = reprocess_symbol(sym, cfg, rcfg)
        except Exception as e:
            logger.exception("reprocess failed %s: %s", sym, e)
            periods[sym] = {"error": str(e), "n": 0}
            continue
        all_ex.extend(rows)
        periods[sym] = {"n": len(rows)}
        if rows:
            periods[sym]["from"] = rows[0]["timestamp"]
            periods[sym]["to"] = rows[-1]["timestamp"]

    audits = audit_examples(all_ex)
    df = pd.DataFrame(all_ex)
    csv_path = rcfg.out_dir / "examples.csv"
    df.to_csv(csv_path, index=False)
    try:
        pq = rcfg.out_dir / "examples.parquet"
        df.to_parquet(pq, index=False)
        parquet_path = str(pq)
    except Exception:
        parquet_path = None

    by_family = {f: int((df["strategy_family"] == f).sum()) if len(df) else 0 for f in FAMILIES}
    by_symbol = df["symbol"].value_counts().to_dict() if len(df) else {}

    manifest = {
        "generation": HIST15_GENERATION,
        "days_requested": days,
        "simulate": simulate,
        "symbols": symbols,
        "periods": periods,
        "n_examples": len(all_ex),
        "by_family": by_family,
        "by_symbol": by_symbol,
        "audits": audits,
        "csv": str(csv_path),
        "parquet": parquet_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (rcfg.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest
