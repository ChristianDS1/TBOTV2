"""Learning engine: ranking, exploration, allowlisted pattern evidence, confidence effects."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from trading_system.config import LearningConfig
from trading_system.database import Database
from trading_system.learning.keys import (
    entry_keys_from_features,
    exit_keys_from_features,
    hold_minutes_from_position,
    is_entry_key,
    is_exit_key,
    is_uneconomic,
    pattern_keys_from_trade,
    signal_pattern_keys,
)
from trading_system.learning.sessions import session_bucket, with_session
from trading_system.types import Position, Signal

# Re-export for callers / tests
__all__ = [
    "LearningEngine",
    "StrategyRanker",
    "classify_strategy_outcome",
    "confidence_bucket",
    "daily_objective_progress",
    "learning_display",
    "pattern_keys_from_trade",
    "signal_pattern_keys",
    "trade_session",
]

class StrategyRanker:
    def __init__(self, cfg: LearningConfig, db: Database) -> None:
        self.cfg = cfg
        self.db = db

    def update_from_trades(self, trades: list[Position]) -> list[dict[str, Any]]:
        by_strat: dict[str, list[Position]] = {}
        for t in trades:
            by_strat.setdefault(t.strategy, []).append(t)

        results = []
        for name, items in by_strat.items():
            pnls = [t.pnl or 0.0 for t in items]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            n = len(pnls)
            wr = len(wins) / n if n else 0
            exp = sum(pnls) / n if n else 0
            gw = sum(wins) if wins else 0
            gl = abs(sum(losses)) if losses else 0
            pf = gw / gl if gl > 0 else (999 if gw > 0 else 0)

            weights = self._recency_weights(n)
            w_exp = float(np.average(pnls, weights=weights)) if n else 0

            conf_penalty = 1 / math.sqrt(max(n, 1))
            rank = w_exp * (1 - 0.5 * conf_penalty) + 0.1 * (wr - 0.5)

            status = "active"
            if n >= self.cfg.min_sample_size and exp < 0 and pf < 1:
                status = "exploration-only"
            if n >= self.cfg.min_sample_size * 2 and exp < -0.05:
                status = "reduced"

            stats = {
                "strategy": name,
                "trades": n,
                "wins": len(wins),
                "losses": len(losses),
                "total_pnl": sum(pnls),
                "expectancy": exp,
                "profit_factor": pf,
                "rank_score": rank,
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.db.upsert_strategy_stats(stats)
            results.append(stats)
        return results

    def _recency_weights(self, n: int) -> np.ndarray:
        half = max(self.cfg.recency_half_life_trades, 1)
        idx = np.arange(n)
        age = (n - 1 - idx).astype(float)
        return np.power(0.5, age / half)


def daily_objective_progress(
    *,
    start_equity: float | None,
    current_equity: float,
    target_pct: float,
    phase: str,
    chase_in_discovery: bool = False,
    name: str = "maximize_net_equity",
) -> dict[str, Any]:
    """UTC-day progress vs north-star daily equity gain. Does not change sizing."""
    start = float(start_equity) if start_equity is not None else float(current_equity)
    cur = float(current_equity)
    tgt = float(target_pct)
    gain_pct = ((cur - start) / start * 100.0) if start > 0 else 0.0
    vs_target = (gain_pct / tgt) if tgt else 0.0
    exploiting = (phase or "").lower() == "exploitation"
    chase_now = bool(chase_in_discovery) or exploiting
    blurb = (
        f"explotar ganadoras hacia +{tgt:.0f}% equity/día"
        if chase_now
        else "ahora: aprender señales; después: aplicar ganadoras hacia +50%/día"
    )
    return {
        "name": name,
        "daily_target_pct": tgt,
        "day_start_equity": start,
        "current_equity": cur,
        "day_gain_pct": gain_pct,
        "progress_vs_target": vs_target,
        "chase_now": chase_now,
        "blurb": blurb,
    }


def confidence_bucket(confidence: float) -> str:
    c = max(0, min(100, confidence))
    lo = int(c // 5) * 5
    return f"{lo}-{lo + 5}"


def trade_session(pos: Position, cfg: LearningConfig) -> str:
    ts = pos.entry_time or pos.exit_time or datetime.now(timezone.utc)
    if not cfg.session_aware:
        return "all"
    return session_bucket(ts, cfg.session_buckets)


def _context_pattern_keys(features: dict[str, Any] | None) -> list[str]:
    """Deprecated — kept empty; allowlisted buckets live in learning.keys."""
    del features
    return []


def _features_from_position(pos: Position) -> dict[str, Any]:
    raw = getattr(pos, "features_json", None) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# pattern_keys_from_trade / signal_pattern_keys imported from learning.keys


def classify_strategy_outcome(pos: Position) -> tuple[str, bool]:
    """
    Strategy win/loss for pattern learning — not net cash.

    - take_profit => strategy win (signal hit its target)
    - trend_exit => strategy win only if net pnl > 0
    - else use gross_pnl if available
    - else fall back to net pnl

    cost_erosion: strategy win but net pnl <= 0 (fees/slippage), tracked separately.
    """
    net = pos.pnl if pos.pnl is not None else 0.0
    gross = pos.gross_pnl

    if pos.exit_reason == "take_profit":
        strategy_win = True
    elif pos.exit_reason in (
        "trend_exit",
        "trend_reversal",
        "profit_protection",
    ):
        strategy_win = net > 0
    elif pos.exit_reason == "stale_position":
        strategy_win = net > 0
    elif gross is not None:
        strategy_win = gross > 0
    else:
        strategy_win = net > 0

    cost_erosion = bool(strategy_win and net <= 0) or bool(pos.cost_erosion)
    direction = "win" if strategy_win else "loss"
    return direction, cost_erosion


def learning_display(pos: Position) -> dict[str, Any]:
    """Human-readable learning classification for monitor UI."""
    direction, cost_erosion = classify_strategy_outcome(pos)
    if direction == "win":
        label = "aprendizaje ganancia"
        if cost_erosion:
            label = "aprendizaje ganancia (fees)"
    else:
        label = "aprendizaje perdida"
    return {
        "learning_direction": direction,
        "learning_label": label,
        "cost_erosion": cost_erosion,
    }


class LearningEngine:
    def __init__(
        self,
        cfg: LearningConfig,
        db: Database,
        *,
        edge_multiple: float = 0.5,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.edge_multiple = float(edge_multiple)
        self.ranker = StrategyRanker(cfg, db)
        self._trades_since_retrain = 0
        self.exploration_count = 0
        self.exploitation_count = 0

    @property
    def exploration_ratio(self) -> float:
        total = self.exploration_count + self.exploitation_count
        if total == 0:
            return self.cfg.exploration_budget
        return self.exploration_count / total

    def current_session(self, ts: datetime | None = None) -> str:
        if not self.cfg.session_aware:
            return "all"
        return session_bucket(ts, self.cfg.session_buckets)

    def on_trade_closed(self, pos: Position) -> list[dict[str, Any]]:
        """Record allowlisted evidence; confirm at >= pattern_min_occurrences."""
        self._trades_since_retrain += 1
        closed = self.db.get_all_closed()
        self.ranker.update_from_trades(closed)

        direction, cost_erosion = classify_strategy_outcome(pos)
        now = datetime.now(timezone.utc).isoformat()
        threshold = self.cfg.pattern_min_occurrences
        newly_confirmed: list[dict[str, Any]] = []
        sess = trade_session(pos, self.cfg)
        feats = _features_from_position(pos)
        uneconomic = is_uneconomic(feats, edge_multiple=self.edge_multiple)

        label = "WIN" if direction == "win" else "LOSS"
        erosion_note = (
            " [COST_EROSION: strategy OK, net<=0 by fees/slip]" if cost_erosion else ""
        )
        unecon_note = " [UNECONOMIC_EDGE: ENTRY keys skipped]" if uneconomic else ""
        self.db.insert_insight(
            "trade_observation",
            (
                f"{label}(strategy) {pos.strategy} {pos.symbol} "
                f"session={sess} regime={pos.regime} conf={pos.confidence:.1f} "
                f"gross={pos.gross_pnl} net={pos.pnl} exit={pos.exit_reason}"
                f"{erosion_note}{unecon_note}"
            ),
            {
                "trade_id": pos.id,
                "direction": direction,
                "cost_erosion": cost_erosion,
                "uneconomic": uneconomic,
                "gross_pnl": pos.gross_pnl,
                "net_pnl": pos.pnl,
                "session": sess,
            },
        )

        if cost_erosion:
            cost_key = (
                f"cost_erosion|exit={pos.exit_reason or 'unknown'}|symbol={pos.symbol}"
            )
            if self.cfg.session_aware:
                cost_key = with_session(sess, cost_key)
            ev = self.db.increment_pattern(cost_key, "cost_erosion", now)
            count = int(ev["count"])
            if count >= threshold and ev["status"] != "confirmed":
                detail = (
                    f"Cost erosion confirmed: strategy outcome was win/TP but net PnL <= 0 "
                    f"due to fees/slippage. Occurrences={count} (≥{threshold}). "
                    f"Does NOT penalize strategy entry rules — review fee/size/hold instead."
                )
                self.db.confirm_pattern(
                    cost_key,
                    "cost_erosion",
                    confirmed_count=count,
                    decision_reason=f"ACEPTADO cost_erosion: {detail}",
                    effect_action="cost_insight_only",
                )
                self.db.insert_applied_change(
                    cost_key,
                    "cost_erosion",
                    "cost_insight_only",
                    detail,
                    occurrences=count,
                    threshold=threshold,
                )
                self.db.insert_insight(
                    "cost_erosion", detail, {"count": count, "key": cost_key}
                )
                newly_confirmed.append(
                    {
                        "pattern_key": cost_key,
                        "direction": "cost_erosion",
                        "count": count,
                        "action": "cost_insight_only",
                    }
                )

        # ENTRY track — skip win/loss when uneconomic (cost track only)
        if not uneconomic:
            for key in entry_keys_from_features(
                feats, pos.side, edge_multiple=self.edge_multiple
            ):
                newly = self._increment_and_maybe_confirm(
                    key,
                    direction,
                    now=now,
                    threshold=threshold,
                    sess=sess,
                    track="entry",
                )
                if newly:
                    newly_confirmed.append(newly)

        # EXIT track — diagnostics only (never bans entries)
        for key in exit_keys_from_features(feats, hold_minutes_from_position(pos)):
            newly = self._increment_and_maybe_confirm(
                key,
                direction,
                now=now,
                threshold=threshold,
                sess=sess,
                track="exit",
            )
            if newly:
                newly_confirmed.append(newly)

        if self._trades_since_retrain >= self.cfg.retrain_every_n_trades:
            self._trades_since_retrain = 0
            self.db.insert_insight(
                "retrain",
                f"Retrain trigger after {self.cfg.retrain_every_n_trades} closed trades",
                {"phase": self.cfg.phase},
            )

        # Promote high-gain chart patterns to obligatory priority list (>=90% net WR)
        try:
            from trading_system.learning.priority import promote_pattern

            chart = feats.get("chart_pattern") or feats.get("setup") or pos.strategy
            if chart:
                chart_s = str(chart)
                same = []
                for t in closed:
                    tf = _features_from_position(t)
                    c = tf.get("chart_pattern") or tf.get("setup") or t.strategy
                    if str(c) != chart_s:
                        continue
                    if self.cfg.session_aware and trade_session(t, self.cfg) != sess:
                        continue
                    same.append(t)
                n = len(same)
                wins = sum(1 for t in same if (t.pnl or 0) > 0)
                wr = wins / n if n else 0.0
                min_wr = float(getattr(self.cfg, "priority_min_net_wr", 0.90) or 0.90)
                min_n = int(getattr(self.cfg, "priority_min_n", 10) or 10)
                if n >= min_n and wr >= min_wr:
                    if promote_pattern(
                        chart_s, net_wr=wr, n=n, session=sess
                    ):
                        self.db.insert_insight(
                            "priority_promote",
                            f"Promoted {chart_s} to priority (net WR={wr:.0%} n={n} session={sess})",
                            {"chart_pattern": chart_s, "net_wr": wr, "n": n, "session": sess},
                        )
                        newly_confirmed.append(
                            {
                                "pattern_key": f"priority:{chart_s}",
                                "direction": "win",
                                "count": n,
                                "action": "priority_promote",
                            }
                        )
        except Exception:
            pass

        return newly_confirmed

    def _increment_and_maybe_confirm(
        self,
        key: str,
        direction: str,
        *,
        now: str,
        threshold: int,
        sess: str,
        track: str,
    ) -> dict[str, Any] | None:
        ev = self.db.increment_pattern(key, direction, now)
        count = int(ev["count"])
        status = ev["status"]
        if count < threshold or status == "confirmed":
            return None

        action_info = self._apply_confirmed_effect(
            key, direction, occurrences=count, threshold=threshold, track=track
        )
        decision_reason = (
            f"ACEPTADO como patrón {direction} ({track}): la key '{key}' se repitió "
            f"{count} veces (≥ umbral {threshold}). "
            f"Efecto aplicado: {action_info['action']}."
        )
        self.db.confirm_pattern(
            key,
            direction,
            confirmed_count=count,
            decision_reason=decision_reason,
            effect_action=action_info["action"],
        )
        self.db.insert_insight(
            "confirmed_pattern",
            decision_reason,
            {
                "pattern_key": key,
                "direction": direction,
                "count": count,
                "threshold": threshold,
                "action": action_info["action"],
                "track": track,
                "session": sess,
            },
        )
        return {
            **ev,
            "status": "confirmed",
            "confirmed_count": count,
            "decision_reason": decision_reason,
            **action_info,
        }

    def _apply_confirmed_effect(
        self,
        pattern_key: str,
        direction: str,
        *,
        occurrences: int,
        threshold: int,
        track: str = "entry",
    ) -> dict[str, str]:
        """
        ENTRY win: confidence_boost.
        ENTRY loss: hard_reject (banned from re-entry).
        EXIT: exit_insight_only (never bans / never moves entry confidence).
        """
        if track == "exit" or is_exit_key(pattern_key):
            action = "exit_insight_only"
            detail = (
                f"EXIT diagnostic on key '{pattern_key}' only. "
                f"Insight/observability — does NOT ban entries or change confidence."
            )
        elif direction == "win":
            action = "confidence_boost"
            detail = (
                f"Win pattern on ENTRY key '{pattern_key}' only. "
                f"Boost confidence by +{self.cfg.win_confidence_boost} when a new "
                f"signal matches this key."
            )
        else:
            action = "hard_reject"
            detail = (
                f"Loss pattern on ENTRY key '{pattern_key}' confirmed. "
                f"Hard-reject matching entries going forward (not confidence-only)."
            )
        self.db.insert_applied_change(
            pattern_key,
            direction,
            action,
            detail,
            occurrences=occurrences,
            threshold=threshold,
        )
        return {"action": action, "detail": detail}

    def apply_confidence_effects(self, signal: Signal) -> tuple[Signal, str | None]:
        """
        ENTRY keys only: confirmed win → boost; confirmed loss → hard-reject.
        EXIT keys never affect entry.
        """
        keys = set(
            signal_pattern_keys(
                signal, self.cfg, edge_multiple=self.edge_multiple
            )
        )
        # Display-only session tag
        if self.cfg.session_aware:
            sess = self.current_session(signal.timestamp)
            signal.features["session"] = sess

        confirmed_wins = {
            p["pattern_key"]
            for p in self.db.get_patterns(direction="win", status="confirmed")
            if is_entry_key(p["pattern_key"])
        }
        confirmed_losses = {
            p["pattern_key"]
            for p in self.db.get_patterns(direction="loss", status="confirmed")
            if is_entry_key(p["pattern_key"])
        }

        boost = 0.0
        reject_reason: str | None = None

        for k in keys:
            if k in confirmed_wins:
                boost = max(boost, self.cfg.win_confidence_boost)
            if k in confirmed_losses:
                reject_reason = f"confirmed_loss_pattern:{k}"
                break

        signal.confidence = max(0.0, min(100.0, signal.confidence + boost))
        signal.features["pattern_boost"] = boost
        signal.features["pattern_penalty"] = 0.0
        return signal, reject_reason

    def should_explore(self) -> bool:
        import random

        return random.random() < self.cfg.exploration_budget

    def tag_signal(self, signal: Signal) -> Signal:
        explore = self.should_explore() or self.cfg.phase == "discovery"
        signal.exploration = explore
        if explore:
            self.exploration_count += 1
        else:
            self.exploitation_count += 1
        return signal

    def phase_progress(self, total_trades: int) -> dict[str, Any]:
        phase = self.cfg.phase
        if total_trades >= 200 and phase == "discovery":
            phase = "pattern"
        elif total_trades >= 500 and phase == "pattern":
            phase = "optimization"
        elif total_trades >= 1000 and phase == "optimization":
            phase = "exploitation"
        return {
            "configured_phase": self.cfg.phase,
            "suggested_phase": phase,
            "total_trades": total_trades,
            "exploration_budget": self.cfg.exploration_budget,
            "exploration_ratio": self.exploration_ratio,
            "pattern_min_occurrences": self.cfg.pattern_min_occurrences,
            "session_aware": self.cfg.session_aware,
            "goal": "maximize_net_equity",
            "confirmed_wins": len(
                [
                    p
                    for p in self.db.get_patterns(direction="win", status="confirmed")
                    if is_entry_key(p["pattern_key"])
                ]
            ),
            "confirmed_losses": len(
                [
                    p
                    for p in self.db.get_patterns(direction="loss", status="confirmed")
                    if is_entry_key(p["pattern_key"])
                ]
            ),
            "observing_patterns": len(
                self.db.get_patterns(status="observing")
            ),
            "current_session": self.current_session(),
        }
