"""Historical win model — pre-move features only, temporal fit."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from trading_system.ml import HIST15_GENERATION

logger = logging.getLogger(__name__)

NUMERIC_KEYS = [
    "rsi",
    "bb_width",
    "macd_fast_hist",
    "macd_slow_hist",
    "pct_from_mid",
    "confidence",
]
BOOL_KEYS = [
    "rejection_bull",
    "rejection_bear",
    "touch_lower",
    "touch_upper",
    "macd_fast_bull_cross",
    "macd_fast_bear_cross",
]
FAMILIES = ["bb_mean_reversion", "momentum_continuation", "bulkowski_pattern"]
HTF_MAP = {"bull": 1.0, "bear": -1.0, "mixed": 0.0, "unknown": 0.0}
LTF_MAP = {"turn_up": 1.0, "turn_down": -1.0}


def vectorize_example(ex: dict[str, Any]) -> list[float]:
    row: list[float] = []
    for k in NUMERIC_KEYS:
        v = ex.get(k)
        try:
            row.append(float(v) if v is not None and v == v else 0.0)
        except (TypeError, ValueError):
            row.append(0.0)
    for k in BOOL_KEYS:
        row.append(1.0 if ex.get(k) else 0.0)
    fam = str(ex.get("strategy_family") or "")
    row.extend(1.0 if fam == f else 0.0 for f in FAMILIES)
    row.append(float(HTF_MAP.get(str(ex.get("htf_bias") or "unknown"), 0.0)))
    row.append(float(LTF_MAP.get(str(ex.get("ltf_turn") or ""), 0.0)))
    direction = str(ex.get("chart_direction") or ex.get("side") or "").lower()
    row.append(1.0 if direction in ("bullish", "call") else (-1.0 if direction in ("bearish", "put") else 0.0))
    # side
    side = str(ex.get("side") or "").lower()
    row.append(1.0 if side == "call" else (-1.0 if side == "put" else 0.0))
    return row


FEATURE_DIM = len(vectorize_example({}))


class HistoricalWinModel:
    """Incremental P(net_win | pre-entry features)."""

    def __init__(self, artifact_dir: Path, *, generation: str | None = None) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.generation = generation or HIST15_GENERATION
        self.model: Any | None = None
        self.backend = "none"
        self.brier: float | None = None
        self.auc: float | None = None
        self.n_trained = 0

    def _path(self) -> Path:
        return self.artifact_dir / "win_model.joblib"

    def predict_proba(self, ex: dict[str, Any]) -> float:
        if self.model is None:
            conf = float(ex.get("confidence") or 50.0)
            return max(0.05, min(0.95, conf / 100.0))
        x = np.asarray([vectorize_example(ex)], dtype=float)
        try:
            return float(self.model.predict_proba(x)[0][1])
        except Exception:
            conf = float(ex.get("confidence") or 50.0)
            return max(0.05, min(0.95, conf / 100.0))

    def fit_examples(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if len(examples) < 20:
            return {"status": "skipped", "reason": "insufficient_samples", "n": len(examples)}
        X = np.asarray([vectorize_example(e) for e in examples], dtype=float)
        y = np.asarray([int(e.get("label_win") or 0) for e in examples], dtype=int)
        if len(set(y.tolist())) < 2:
            return {"status": "skipped", "reason": "single_class", "n": len(examples)}

        # Temporal holdout: last 20% for metrics only; fit on first 80%
        cut = max(int(len(X) * 0.8), 15)
        Xtr, ytr = X[:cut], y[:cut]
        Xte, yte = X[cut:], y[cut:]
        if len(set(ytr.tolist())) < 2:
            Xtr, ytr = X, y
            Xte, yte = X[-min(10, len(X)) :], y[-min(10, len(y)) :]

        backend = "random_forest"
        try:
            import lightgbm as lgb

            base = lgb.LGBMClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.05, verbose=-1
            )
            backend = "lightgbm"
        except Exception:
            base = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        base.fit(Xtr, ytr)
        try:
            clf = CalibratedClassifierCV(base, cv=3)
            clf.fit(Xtr, ytr)
        except Exception:
            clf = base

        self.model = clf
        self.backend = backend
        self.n_trained = len(Xtr)
        metrics: dict[str, Any] = {"n_train": float(len(Xtr)), "n_test": float(len(Xte))}
        if len(Xte) >= 5 and len(set(yte.tolist())) >= 2:
            proba = clf.predict_proba(Xte)[:, 1]
            self.brier = float(brier_score_loss(yte, proba))
            self.auc = float(roc_auc_score(yte, proba))
            metrics["brier"] = self.brier
            metrics["auc"] = self.auc
            metrics["test_win_rate"] = float(yte.mean())
        self._save()
        logger.info(
            "hist model fit backend=%s n=%d brier=%s auc=%s",
            backend,
            len(Xtr),
            self.brier,
            self.auc,
        )
        return {"status": "ok", "backend": backend, **metrics}

    def _save(self) -> None:
        joblib.dump(
            {
                "model": self.model,
                "backend": self.backend,
                "brier": self.brier,
                "auc": self.auc,
                "generation": self.generation,
                "feature_dim": FEATURE_DIM,
            },
            self._path(),
        )
        meta = {
            "generation": self.generation,
            "backend": self.backend,
            "brier": self.brier,
            "auc": self.auc,
            "n_trained": self.n_trained,
            "feature_dim": FEATURE_DIM,
            "numeric_keys": NUMERIC_KEYS,
            "bool_keys": BOOL_KEYS,
            "families": FAMILIES,
            "loaded_legacy_weights": False,
            "label": "label_win (net_pnl > 0)",
        }
        (self.artifact_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        (self.artifact_dir / "metrics.json").write_text(
            json.dumps(
                {"brier": self.brier, "auc": self.auc, "n_trained": self.n_trained},
                indent=2,
            ),
            encoding="utf-8",
        )
