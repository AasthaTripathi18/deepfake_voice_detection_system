"""
Ensemble model combining Support Vector Machine (SVM) and Multi-Layer Perceptron (MLP)
via soft probability voting on 240-dimensional dynamic acoustic features.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .base import BaseSpoofDetectionModel
from .svm_baseline import SVMBaseline
from .mlp_baseline import MLPBaseline

META_FILENAME = "ensemble_meta.json"


class EnsembleBaseline(BaseSpoofDetectionModel):
    def __init__(self, svm_weight: float = 0.35, svm_model=None, mlp_model=None):
        self.svm_weight = svm_weight
        self.mlp_weight = 1.0 - svm_weight
        self.svm = svm_model
        self.mlp = mlp_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleBaseline":
        if self.svm is None:
            self.svm = SVMBaseline(C=0.5, seed=42)
            self.svm.fit(X, y)
        if self.mlp is None:
            self.mlp = MLPBaseline(hidden_layer_sizes=(256, 128, 64), alpha=0.01, seed=42)
            self.mlp.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.svm is None or self.mlp is None:
            raise RuntimeError("Ensemble models not fitted or loaded.")
        p_svm = self.svm.predict_proba(X)
        p_mlp = self.mlp.predict_proba(X)
        return self.svm_weight * p_svm + self.mlp_weight * p_mlp

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.svm._pipeline, dir_path / "svm_pipeline.joblib")
        joblib.dump(self.mlp._pipeline, dir_path / "mlp_pipeline.joblib")
        meta = {
            "svm_weight": self.svm_weight,
            "mlp_weight": self.mlp_weight,
        }
        with open(dir_path / META_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, dir_path: str | Path) -> "EnsembleBaseline":
        dir_path = Path(dir_path)
        with open(dir_path / META_FILENAME, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        instance = cls(svm_weight=meta["svm_weight"])
        svm = SVMBaseline(C=0.5)
        svm._pipeline = joblib.load(dir_path / "svm_pipeline.joblib")
        mlp = MLPBaseline()
        mlp._pipeline = joblib.load(dir_path / "mlp_pipeline.joblib")
        instance.svm = svm
        instance.mlp = mlp
        return instance
