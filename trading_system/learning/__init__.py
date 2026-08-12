"""Learning engine: ranking, exploration, pattern evidence (>=20), confidence effects."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from trading_system.config import LearningConfig
from trading_system.database import Database
from trading_system.learning.sessions import session_bucket, with_session
from trading_system.types import Position, Signal


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


def confidence_bucket(confidence: float) -> str:
    c = max(0, min(100, confidence))
    lo = int(c // 5) * 5
    return f"{lo}-{lo + 5}"


def trade_session(pos: Position, cfg: LearningConfig) -> str:
    ts = pos.entry_time or pos.exit_time or datetime.now(timezone.utc)
    if not cfg.session_aware:
        return "all"
    return session_bucket(ts, cfg.session_buckets)


def pattern_keys_from_trade(pos: Position, cfg: LearningConfig | None = None) -> list[str]:
    base = [
        f"regime={pos.regime}",
        f"symbol={pos.symbol}",
        f"exit_reason={pos.exit_reason or 'unknown'}",
        f"confidence_bucket={confidence_bucket(pos.confidence)}",
        f"strategy={pos.strategy}",
        f"regime={pos.regime}|exit={pos.exit_reason or 'unknown'}",
    ]
    if cfg is None or not cfg.session_aware:
        return base
    sess = trade_session(pos, cfg)
    scoped = [with_session(sess, k) for k in base]
    scoped.append(f"session={sess}")
    return scoped


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
    elif pos.exit_reason == "trend_exit":
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


def signal_pattern_keys(signal: Signal, cfg: LearningConfig | None = None) -> list[str]:
    base = [
        f"regime={signal.regime.value}",
        f"symbol={signal.symbol}",
        f"confidence_bucket={confidence_bucket(signal.confidence)}",
        f"strategy={signal.strategy}",
    ]
    if cfg is None or not cfg.session_aware:
        return base
    ts = signal.timestamp or datetime.now(timezone.utc)
    sess = session_bucket(ts, cfg.session_buckets)
    signal.features["session"] = sess
    scoped = [with_session(sess, k) for k in base]
    scoped.append(f"session={sess}")
    return scoped


def _soft_reject_excluded(key: str, exclude_prefixes: list[str]) -> bool:
    # Bare session key is too broad (would halt whole session)
    if key.startswith("session=") and "|" not in key:
        return True
    for p in exclude_prefixes:
        if key.startswith(p) or f"|{p}" in key:
            return True
    return False


class LearningEngine:
    def __init__(self, cfg: LearningConfig, db: Database) -> None:
        self.cfg = cfg
        self.db = db
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
        """Record evidence; confirm patterns only at >= pattern_min_occurrences."""
        self._trades_since_retrain += 1
        closed = self.db.get_all_closed()
        self.ranker.update_from_trades(closed)

        direction, cost_erosion = classify_strategy_outcome(pos)
        now = datetime.now(timezone.utc).isoformat()
        threshold = self.cfg.pattern_min_occurrences
        newly_confirmed: list[dict[str, Any]] = []
        sess = trade_session(pos, self.cfg)

        label = "WIN" if direction == "win" else "LOSS"
        erosion_note = (
            " [COST_EROSION: strategy OK, net<=0 by fees/slip]" if cost_erosion else ""
        )
        self.db.insert_insight(
            "trade_observation",
            (
                f"{label}(strategy) {pos.strategy} {pos.symbol} "
                f"session={sess} regime={pos.regime} conf={pos.confidence:.1f} "
                f"gross={pos.gross_pnl} net={pos.pnl} exit={pos.exit_reason}"
                f"{erosion_note}"
            ),
            {
                "trade_id": pos.id,
                "direction": direction,
                "cost_erosion": cost_erosion,
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

        for key in pattern_keys_from_trade(pos, self.cfg):
            ev = self.db.increment_pattern(key, direction, now)
            count = int(ev["count"])
            status = ev["status"]

            if count < threshold:
                continue

            if status != "confirmed":
                action_info = self._apply_confirmed_effect(
                    key, direction, occurrences=count, threshold=threshold
                )
                decision_reason = (
                    f"ACEPTADO como patrón {direction}: la key '{key}' se repitió "
                    f"{count} veces (≥ umbral {threshold}). "
                    f"Clasificación por señal/gross (no por net cash). "
                    f"Scope: sesión/contexto de esta key — NO invalida los 5 "
                    f"indicadores de entrada (BB/RSI/MACD/rejection) en bloque. "
                    f"Efecto aplicado: {action_info['action']}."
                )
                self.db.confirm_pattern(
                    key,
                    direction,
                    confirmed_count=count,
                    decision_reason=decision_reason,
                    effect_action=action_info["action"],
                )
                newly_confirmed.append(
                    {
                        **ev,
                        "status": "confirmed",
                        "confirmed_count": count,
                        "decision_reason": decision_reason,
                        **action_info,
                    }
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
                        "session": sess,
                    },
                )

        if self._trades_since_retrain >= self.cfg.retrain_every_n_trades:
            self._trades_since_retrain = 0
            self.db.insert_insight(
                "retrain",
                f"Retrain trigger after {self.cfg.retrain_every_n_trades} closed trades",
                {"phase": self.cfg.phase},
            )
        return newly_confirmed

    def _apply_confirmed_effect(
        self,
        pattern_key: str,
        direction: str,
        *,
        occurrences: int,
        threshold: int,
    ) -> dict[str, str]:
        """
        Win: confidence boost ONLY — do not alter strategy/pattern rules.
        Loss: confidence penalty and optional soft-reject flag.
        Affects only the matching contextual key, not all 5 entry indicators.
        """
        if direction == "win":
            action = "confidence_boost"
            detail = (
                f"Win pattern on key '{pattern_key}' only. "
                f"Boost confidence by +{self.cfg.win_confidence_boost} when a new "
                f"signal matches this key. Entry checklist BB/RSI/MACD/rejection unchanged."
            )
        else:
            if self.cfg.loss_soft_reject:
                action = "soft_reject"
                detail = (
                    f"Loss pattern on key '{pattern_key}' only. "
                    f"Soft-reject (and -{self.cfg.loss_confidence_penalty} conf) when a "
                    f"new signal matches this key. Does NOT mark all 5 indicators as bad."
                )
            else:
                action = "confidence_penalty"
                detail = (
                    f"Loss pattern on key '{pattern_key}' only. "
                    f"Penalize confidence by -{self.cfg.loss_confidence_penalty} when a "
                    f"new signal matches this key. Entry checklist unchanged."
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
        Adjust signal confidence from confirmed patterns.
        Returns (signal, soft_reject_reason|None).
        Win patterns: boost only. Loss patterns: penalty and optional soft-reject.
        Session-aware: only keys for the current UTC session apply.
        """
        keys = set(signal_pattern_keys(signal, self.cfg))
        if not self.cfg.session_aware:
            keys.add(f"regime={signal.regime.value}")

        sess = signal.features.get("session") or self.current_session(signal.timestamp)
        signal.features["session"] = sess

        confirmed_wins = {
            p["pattern_key"]
            for p in self.db.get_patterns(direction="win", status="confirmed")
        }
        confirmed_losses = {
            p["pattern_key"]
            for p in self.db.get_patterns(direction="loss", status="confirmed")
        }
        # Never apply another session's confirmed keys
        if self.cfg.session_aware:
            prefix = f"session={sess}|"
            bare = f"session={sess}"
            confirmed_wins = {
                k for k in confirmed_wins if k.startswith(prefix) or k == bare
            }
            confirmed_losses = {
                k for k in confirmed_losses if k.startswith(prefix) or k == bare
            }

        exclude_prefixes = list(self.cfg.soft_reject_exclude_key_prefixes or [])

        boost = 0.0
        penalty = 0.0
        reject_reason: str | None = None

        for k in keys:
            if k in confirmed_wins:
                boost = max(boost, self.cfg.win_confidence_boost)
            if k in confirmed_losses:
                penalty = max(penalty, self.cfg.loss_confidence_penalty)
                if not self.cfg.loss_soft_reject:
                    continue
                if _soft_reject_excluded(k, exclude_prefixes):
                    continue
                if k in confirmed_wins:
                    continue
                reject_reason = f"confirmed_loss_pattern:{k}"

        signal.confidence = max(0.0, min(100.0, signal.confidence + boost - penalty))
        signal.features["pattern_boost"] = boost
        signal.features["pattern_penalty"] = penalty
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
            "exploration_ratio": self.exploration_ratio,
            "exploration_budget": self.cfg.exploration_budget,
            "pattern_min_occurrences": self.cfg.pattern_min_occurrences,
            "confirmed_wins": len(self.db.get_patterns("win", "confirmed")),
            "confirmed_losses": len(self.db.get_patterns("loss", "confirmed")),
            "observing_patterns": len(self.db.get_patterns(status="observing")),
            "session_aware": self.cfg.session_aware,
            "current_session": self.current_session(),
        }
