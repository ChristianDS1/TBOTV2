"""Probability model P(win | features) — baseline → sklearn → LightGBM."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_system.types import Position

logger = logging.getLogger(__name__)

FEATURE_KEYS = [
    "rsi",
    "bb_width",
    "macd_fast_hist",
    "macd_slow_hist",
    "pct_from_mid",
    "confidence",
]


class WinProbabilityModel:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.model: Any | None = None
        self.backend = "none"
        self.brier: float | None = None
        self._load()

    def _path(self) -> Path:
        return self.artifact_dir / "win_model.joblib"

    def _load(self) -> None:
        p = self._path()
        if p.exists():
            data = joblib.load(p)
            self.model = data["model"]
            self.backend = data.get("backend", "unknown")
            self.brier = data.get("brier")

    def _save(self) -> None:
        joblib.dump(
            {"model": self.model, "backend": self.backend, "brier": self.brier},
            self._path(),
        )

    def _row(self, features: dict, confidence: float) -> list[float]:
        row = []
        for k in FEATURE_KEYS:
            if k == "confidence":
                row.append(float(confidence))
            else:
                v = features.get(k)
                row.append(float(v) if v is not None else 0.0)
        return row

    def predict_proba(self, features: dict, confidence: float) -> float:
        if self.model is None:
            # Baseline: map confidence to probability
            return max(0.05, min(0.95, confidence / 100.0))
        x = np.array([self._row(features, confidence)])
        try:
            proba = self.model.predict_proba(x)[0][1]
            return float(proba)
        except Exception:
            return max(0.05, min(0.95, confidence / 100.0))

    def train(self, trades: list[Position], prefer_lightgbm: bool = True) -> dict[str, Any]:
        rows = []
        labels = []
        for t in trades:
            try:
                feats = json.loads(t.features_json or "{}")
            except json.JSONDecodeError:
                feats = {}
            rows.append(self._row(feats, t.confidence))
            labels.append(1 if (t.pnl or 0) > 0 else 0)

        if len(rows) < 20:
            return {"status": "skipped", "reason": "insufficient_samples", "n": len(rows)}

        X = np.array(rows)
        y = np.array(labels)
        if len(set(y)) < 2:
            return {"status": "skipped", "reason": "single_class", "n": len(rows)}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        backend = "logistic"
        clf: Any
        if prefer_lightgbm:
            try:
                import lightgbm as lgb

                clf = lgb.LGBMClassifier(
                    n_estimators=80,
                    max_depth=4,
                    learning_rate=0.05,
                    verbose=-1,
                )
                backend = "lightgbm"
            except Exception:
                clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                backend = "random_forest"
        else:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500)),
                ]
            )
            clf = CalibratedClassifierCV(pipe, cv=3)
            backend = "logistic"

        if backend == "logistic":
            clf.fit(X_train, y_train)
            self.model = clf
        else:
            base = clf
            base.fit(X_train, y_train)
            try:
                self.model = CalibratedClassifierCV(base, cv=3)
                self.model.fit(X_train, y_train)
            except Exception:
                self.model = base

        proba = self.model.predict_proba(X_test)[:, 1]
        self.brier = float(brier_score_loss(y_test, proba))
        self.backend = backend
        self._save()
        logger.info("Trained %s model brier=%.4f n=%d", backend, self.brier, len(rows))
        return {
            "status": "ok",
            "backend": backend,
            "brier": self.brier,
            "n": len(rows),
            "test_size": len(y_test),
        }
