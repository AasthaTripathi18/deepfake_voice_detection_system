"""
Multi-Layer Perceptron (MLP) baseline model for spoof detection on pooled acoustic features.

Uses a deep feed-forward neural architecture with Standard Scaling, non-linear ReLU
activations, L2 regularization, and early stopping.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseSpoofDetectionModel

MODEL_FILENAME = "model.joblib"
META_FILENAME = "mlp_meta.json"


class MLPBaseline(BaseSpoofDetectionModel):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
        alpha: float = 0.01,
        learning_rate_init: float = 0.001,
        max_iter: int = 100,
        seed: int = 42,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.seed = seed
        self._pipeline: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPBaseline":
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.seed,
                early_stopping=True,
                n_iter_no_change=10,
            )),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Model not fitted/loaded yet.")
        proba = self._pipeline.predict_proba(X)
        spoof_col = list(self._pipeline.classes_).index(1)
        return proba[:, spoof_col]

    def save(self, dir_path: str | Path) -> None:
        if self._pipeline is None:
            raise RuntimeError("Cannot save an unfitted model.")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, dir_path / MODEL_FILENAME)
        meta = {
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "seed": self.seed,
        }
        with open(dir_path / META_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, dir_path: str | Path) -> "MLPBaseline":
        dir_path = Path(dir_path)
        with open(dir_path / META_FILENAME, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["hidden_layer_sizes"] = tuple(meta["hidden_layer_sizes"])
        instance = cls(**meta)
        instance._pipeline = joblib.load(dir_path / MODEL_FILENAME)
        return instance
