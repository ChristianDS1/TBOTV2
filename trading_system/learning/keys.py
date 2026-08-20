"""Allowlisted ENTRY/EXIT learning keys — buckets only, no coarse session/symbol/chart."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from trading_system.types import Position, Side, Signal

ENTRY_DIMS = frozenset(
    {
        "rsi_zone",
        "rsi_slope",
        "rsi_vs_side",
        "bb_pos",
        "bb_width_bin",
        "macd_fast_sign",
        "macd_fast_slope",
        "macd_cross",
        "macd_fast_vs_slow",
        "rejection",
        "htf_ltf_combo",
        "backing_quality",
        "edge_ratio_bin",
    }
)

EXIT_DIMS = frozenset(
    {
        "exit_class",
        "mfe_bin",
        "mae_bin",
        "giveback_bin",
        "hold_min_bin",
    }
)

FORBIDDEN_DIMS = frozenset(
    {
        "session",
        "symbol",
        "chart",
        "strategy",
        "confidence_bucket",
        "exit_reason",
        "regime",
        "htf",  # bare htf= — use htf_ltf_combo
        "cost_erosion",  # handled separately
    }
)

# Ultra-common 1-dim values that ban almost the whole book if hard-rejected
HARD_REJECT_FORBIDDEN_KEYS = frozenset(
    {
        "rejection=none",
        "macd_cross=none",
        "rsi_vs_side=counter",
        "bb_width_bin=squeeze",
        "backing_quality=none",
        "macd_fast_slope=flat",
        "rsi_slope=flat",
        "edge_ratio_bin=lt1",
        "edge_ratio_bin=uneconomic",
    }
)

NEAR_ZERO_MACD = 1e-6
RSI_FLAT_EPS = 0.5

# Flat / "went nowhere" exits — do NOT train ENTRY win/loss evidence
LIMBO_EXIT_REASONS = frozenset(
    {
        "limbo_timeout",
        "limbo_flat",
        "flat_timeout",
    }
)

# Exclusive ENTRY label thresholds (policy v4)
ENTRY_WR_WIN = 0.55
ENTRY_WR_LOSS = 0.45


def is_limbo_exit(exit_reason: str | None) -> bool:
    if not exit_reason:
        return False
    return str(exit_reason).strip().lower() in LIMBO_EXIT_REASONS


def classify_entry_label(
    wins: int,
    losses: int,
    *,
    min_n: int = 10,
    wr_win: float = ENTRY_WR_WIN,
    wr_loss: float = ENTRY_WR_LOSS,
    mean_net: float | None = None,
) -> str:
    """
    Exclusive ENTRY label for one allowlisted key (never boost AND penalty).

    Returns: observing | winner | loser | neutral
    """
    w = max(0, int(wins))
    l = max(0, int(losses))
    n = w + l
    if n < int(min_n):
        return "observing"
    wr = w / n if n else 0.0
    if wr >= float(wr_win):
        if mean_net is not None and mean_net <= 0:
            return "neutral"
        return "winner"
    if wr <= float(wr_loss):
        return "loser"
    return "neutral"


def _f(features: dict[str, Any], key: str) -> float | None:
    v = features.get(key)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return x


def _kv(dim: str, value: str) -> str:
    return f"{dim}={value}"


def rsi_zone(rsi: float | None) -> str | None:
    if rsi is None:
        return None
    if rsi < 30:
        return "lt30"
    if rsi < 40:
        return "30_40"
    if rsi <= 60:
        return "40_60"
    if rsi <= 70:
        return "60_70"
    return "gt70"


def rsi_slope(rsi: float | None, rsi_prev: float | None) -> str | None:
    if rsi is None or rsi_prev is None:
        return None
    d = rsi - rsi_prev
    if abs(d) < RSI_FLAT_EPS:
        return "flat"
    return "up" if d > 0 else "down"


def rsi_vs_side(rsi: float | None, side: Side | str) -> str | None:
    if rsi is None:
        return None
    is_call = side == Side.CALL or str(side).lower() in ("call", "buy", "long")
    # Mean-reversion alignment: call prefers oversold, put prefers overbought
    if is_call:
        return "aligned" if rsi < 50 else "counter"
    return "aligned" if rsi > 50 else "counter"


def bb_pos(features: dict[str, Any]) -> str | None:
    close = _f(features, "close") or _f(features, "entry_mark") or _f(features, "price")
    lo = _f(features, "bb_lower")
    mid = _f(features, "bb_mid")
    hi = _f(features, "bb_upper")
    if close is None or lo is None or mid is None or hi is None:
        return None
    if hi <= lo:
        return None
    touch_lo = bool(features.get("touch_lower")) or close <= lo
    touch_hi = bool(features.get("touch_upper")) or close >= hi
    if close < lo:
        return "below_lower"
    if touch_lo and close <= lo * 1.0005:
        return "touch_lower"
    if close > hi:
        return "above_upper"
    if touch_hi and close >= hi * 0.9995:
        return "touch_upper"
    # halves relative to mid
    if close < mid:
        # lower half of [lo, mid]
        return "lower_half"
    if abs(close - mid) / max(abs(mid), 1e-12) < 0.001:
        return "mid"
    return "upper_half"


def bb_width_bin(width: float | None) -> str | None:
    if width is None:
        return None
    if width < 0.005:
        return "squeeze"
    if width < 0.02:
        return "normal"
    return "expansion"


def macd_fast_sign(hist: float | None) -> str | None:
    if hist is None:
        return None
    if abs(hist) < NEAR_ZERO_MACD:
        return "near_zero"
    return "pos" if hist > 0 else "neg"


def macd_fast_slope(hist: float | None, prev: float | None) -> str | None:
    if hist is None or prev is None:
        return None
    d = hist - prev
    if abs(d) < NEAR_ZERO_MACD:
        return "flat"
    return "rising" if d > 0 else "falling"


def macd_cross(features: dict[str, Any]) -> str | None:
    if bool(features.get("macd_fast_bull_cross")):
        return "bull_cross"
    if bool(features.get("macd_fast_bear_cross")):
        return "bear_cross"
    hist = _f(features, "macd_fast_hist")
    prev = _f(features, "macd_fast_hist_prev")
    if hist is None or prev is None:
        return "none"
    if prev < 0 <= hist:
        return "bull_cross"
    if prev > 0 >= hist:
        return "bear_cross"
    return "none"


def macd_fast_vs_slow(fast: float | None, slow: float | None) -> str | None:
    if fast is None or slow is None:
        return None
    if abs(fast) < NEAR_ZERO_MACD and abs(slow) < NEAR_ZERO_MACD:
        return "aligned"
    if fast == 0 or slow == 0:
        return "divergent" if fast * slow < 0 else "aligned"
    return "aligned" if (fast > 0) == (slow > 0) else "divergent"


def rejection_bin(features: dict[str, Any]) -> str | None:
    bull = bool(features.get("rejection_bull"))
    bear = bool(features.get("rejection_bear"))
    if bull and not bear:
        return "bull"
    if bear and not bull:
        return "bear"
    if bull and bear:
        return "none"  # conflicting → treat as none
    return "none"


def htf_ltf_combo(features: dict[str, Any]) -> str | None:
    htf = features.get("htf_bias")
    ltf = features.get("ltf_turn")
    if htf is None and ltf is None:
        return None
    htf_s = str(htf) if htf not in (None, "", "unknown", "None") else "na"
    ltf_s = str(ltf) if ltf not in (None, "", "None") else "na"
    if htf_s == "na" and ltf_s == "na":
        return None
    return f"htf={htf_s}|ltf={ltf_s}"


def backing_quality(features: dict[str, Any]) -> str | None:
    raw = features.get("backing")
    if raw is None:
        return "none"
    if isinstance(raw, (list, tuple)):
        n = len([x for x in raw if x])
    else:
        s = str(raw).strip()
        if not s or s.lower() in ("none", "nan"):
            return "none"
        n = len([p for p in s.split(",") if p.strip()])
    if n >= 3:
        return "strong"
    if n >= 1:
        return "weak"
    return "none"


def is_uneconomic(
    features: dict[str, Any],
    *,
    edge_multiple: float = 0.5,
) -> bool:
    edge_bps = _f(features, "edge_bps")
    cost = _f(features, "round_trip_cost_bps")
    if edge_bps is None or cost is None or cost <= 0:
        return False
    return edge_bps < float(edge_multiple) * cost


def edge_ratio_bin(
    features: dict[str, Any],
    *,
    edge_multiple: float = 0.5,
) -> str | None:
    if is_uneconomic(features, edge_multiple=edge_multiple):
        return "uneconomic"
    ratio = _f(features, "edge_ratio")
    if ratio is None:
        edge_bps = _f(features, "edge_bps")
        cost = _f(features, "round_trip_cost_bps")
        if edge_bps is not None and cost and cost > 0:
            ratio = edge_bps / cost
    if ratio is None:
        return None
    if ratio < 1:
        return "lt1"
    if ratio < 2:
        return "1_2"
    if ratio < 5:
        return "2_5"
    return "gt5"


def _pct_bin(pct: float | None, *, abs_scale: bool = False) -> str | None:
    if pct is None:
        return None
    x = abs(pct) if abs_scale else pct
    # bins in percent points (features store percent already, e.g. 0.30 = 0.30%)
    if x < 0.15:
        return "lt0.15"
    if x < 0.35:
        return "0.15_0.35"
    if x < 0.75:
        return "0.35_0.75"
    return "gt0.75"


def giveback_bin(gb: float | None) -> str | None:
    if gb is None:
        return None
    if gb < 0.25:
        return "lt0.25"
    if gb < 0.50:
        return "0.25_0.50"
    if gb < 0.75:
        return "0.50_0.75"
    return "gt0.75"


def hold_min_bin(minutes: float | None) -> str | None:
    if minutes is None:
        return None
    if minutes < 2:
        return "lt2"
    if minutes < 10:
        return "2_10"
    if minutes <= 30:
        return "10_30"
    return "gt30"


def hold_minutes_from_position(pos: Position) -> float | None:
    if not pos.entry_time or not pos.exit_time:
        return None
    t0 = pos.entry_time
    t1 = pos.exit_time
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=timezone.utc)
    return max(0.0, (t1 - t0).total_seconds() / 60.0)


def _dim_of_part(part: str) -> str | None:
    if "=" not in part:
        return None
    return part.split("=", 1)[0]


def is_exit_key(key: str) -> bool:
    """True if key is EXIT-track (singles or exit_class|mfe_bin compound)."""
    parts = key.split("|")
    dims = [_dim_of_part(p) for p in parts]
    if any(d is None for d in dims):
        return False
    if not all(d in EXIT_DIMS for d in dims):  # type: ignore[operator]
        return False
    if len(parts) == 1:
        return True
    # only allowed EXIT compound
    return set(dims) == {"exit_class", "mfe_bin"}  # type: ignore[arg-type]


def is_entry_compound(key: str) -> bool:
    """True for priority multi-dim ENTRY keys (not ultra-common singles)."""
    if key.startswith("htf_ltf_combo="):
        return True
    if "|" not in key:
        return False
    # htf_ltf_combo values embed | — already handled
    parts = key.split("|")
    dims = [_dim_of_part(p) for p in parts]
    if any(d is None for d in dims):
        return False
    allowed_pairs = {
        frozenset({"rsi_zone", "bb_pos"}),
        frozenset({"rsi_vs_side", "macd_fast_vs_slow"}),
    }
    return len(parts) == 2 and frozenset(dims) in allowed_pairs  # type: ignore[arg-type]


def is_entry_key(key: str) -> bool:
    # Combo stored as one dim whose value may contain '|'
    if key.startswith("htf_ltf_combo="):
        return True
    parts = key.split("|")
    dims = [_dim_of_part(p) for p in parts]
    if any(d is None for d in dims):
        return False
    if not all(d in ENTRY_DIMS for d in dims):  # type: ignore[operator]
        return False
    if len(parts) == 1:
        return True
    allowed_pairs = {
        frozenset({"rsi_zone", "bb_pos"}),
        frozenset({"rsi_vs_side", "macd_fast_vs_slow"}),
    }
    if len(parts) == 2 and frozenset(dims) in allowed_pairs:  # type: ignore[arg-type]
        return True
    return False


def is_allowlisted_key(key: str) -> bool:
    """Strict allowlist: ENTRY or EXIT track keys only (not cost_erosion)."""
    if not key:
        return False
    if key.startswith("session=") or key.startswith("symbol=") or key.startswith("chart="):
        return False
    return is_entry_key(key) or is_exit_key(key)


def _normalize_htf_ltf_key(raw: str) -> str:
    """Store combo as single dim: htf_ltf_combo=htf=bear|ltf=turn_down."""
    return _kv("htf_ltf_combo", raw)


def entry_keys_from_features(
    features: dict[str, Any] | None,
    side: Side | str,
    *,
    edge_multiple: float = 0.5,
) -> list[str]:
    feats = features or {}
    keys: list[str] = []
    singles: dict[str, str] = {}

    rz = rsi_zone(_f(feats, "rsi"))
    if rz:
        singles["rsi_zone"] = rz
    rs = rsi_slope(_f(feats, "rsi"), _f(feats, "rsi_prev"))
    if rs:
        singles["rsi_slope"] = rs
    rvs = rsi_vs_side(_f(feats, "rsi"), side)
    if rvs:
        singles["rsi_vs_side"] = rvs
    bp = bb_pos(feats)
    if bp:
        singles["bb_pos"] = bp
    bw = bb_width_bin(_f(feats, "bb_width"))
    if bw:
        singles["bb_width_bin"] = bw
    mfs = macd_fast_sign(_f(feats, "macd_fast_hist"))
    if mfs:
        singles["macd_fast_sign"] = mfs
    msl = macd_fast_slope(
        _f(feats, "macd_fast_hist"), _f(feats, "macd_fast_hist_prev")
    )
    if msl:
        singles["macd_fast_slope"] = msl
    mc = macd_cross(feats)
    if mc:
        singles["macd_cross"] = mc
    mvs = macd_fast_vs_slow(
        _f(feats, "macd_fast_hist"), _f(feats, "macd_slow_hist")
    )
    if mvs:
        singles["macd_fast_vs_slow"] = mvs
    rej = rejection_bin(feats)
    if rej:
        singles["rejection"] = rej
    combo = htf_ltf_combo(feats)
    if combo:
        # priority compound #1 — stored as one allowlisted dim
        keys.append(_normalize_htf_ltf_key(combo))
    bq = backing_quality(feats)
    if bq:
        singles["backing_quality"] = bq
    erb = edge_ratio_bin(feats, edge_multiple=edge_multiple)
    if erb:
        singles["edge_ratio_bin"] = erb

    for dim, val in singles.items():
        keys.append(_kv(dim, val))

    # Priority compounds 2–4
    if "rsi_zone" in singles and "bb_pos" in singles:
        keys.append(
            f"{_kv('rsi_zone', singles['rsi_zone'])}|{_kv('bb_pos', singles['bb_pos'])}"
        )
    if "rsi_vs_side" in singles and "macd_fast_vs_slow" in singles:
        keys.append(
            f"{_kv('rsi_vs_side', singles['rsi_vs_side'])}|"
            f"{_kv('macd_fast_vs_slow', singles['macd_fast_vs_slow'])}"
        )
    # edge_ratio_bin already emitted as single (#4)

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen and is_allowlisted_key(k):
            seen.add(k)
            out.append(k)
    return out


def exit_keys_from_features(
    features: dict[str, Any] | None,
    hold_minutes: float | None = None,
) -> list[str]:
    feats = features or {}
    keys: list[str] = []
    singles: dict[str, str] = {}

    ec = feats.get("exit_pattern_class") or feats.get("exit_class")
    if ec and str(ec) not in ("", "None", "none"):
        singles["exit_class"] = str(ec)
    mfe = _pct_bin(_f(feats, "mfe_pct"))
    if mfe:
        singles["mfe_bin"] = mfe
    mae = _pct_bin(_f(feats, "mae_pct"), abs_scale=True)
    if mae:
        singles["mae_bin"] = mae
    gb = giveback_bin(_f(feats, "giveback_pct"))
    if gb:
        singles["giveback_bin"] = gb
    hm = hold_min_bin(hold_minutes)
    if hm:
        singles["hold_min_bin"] = hm

    for dim, val in singles.items():
        keys.append(_kv(dim, val))

    # Priority compound #5 (EXIT track)
    if "exit_class" in singles and "mfe_bin" in singles:
        keys.append(
            f"{_kv('exit_class', singles['exit_class'])}|{_kv('mfe_bin', singles['mfe_bin'])}"
        )

    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen and is_allowlisted_key(k):
            seen.add(k)
            out.append(k)
    return out


def pattern_keys_from_trade(
    pos: Position,
    cfg: Any = None,
    *,
    edge_multiple: float = 0.5,
    include_entry: bool = True,
    include_exit: bool = True,
) -> list[str]:
    """Build allowlisted keys for a closed trade. No session/symbol/chart."""
    del cfg  # session scoping intentionally unused for pattern keys
    feats = _features_from_position(pos)
    keys: list[str] = []
    if include_entry:
        keys.extend(
            entry_keys_from_features(feats, pos.side, edge_multiple=edge_multiple)
        )
    if include_exit:
        keys.extend(
            exit_keys_from_features(feats, hold_minutes_from_position(pos))
        )
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def signal_pattern_keys(
    signal: Signal,
    cfg: Any = None,
    *,
    edge_multiple: float = 0.5,
) -> list[str]:
    """ENTRY keys only for confidence effects / hard-reject matching."""
    del cfg
    feats = getattr(signal, "features", None) or {}
    # Keep session on features for display only — never as a pattern key
    return entry_keys_from_features(feats, signal.side, edge_multiple=edge_multiple)


def _features_from_position(pos: Position) -> dict[str, Any]:
    import json

    raw = getattr(pos, "features_json", None) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
