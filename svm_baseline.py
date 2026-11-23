"""
Primary baseline model: Support Vector Machine (RBF kernel) on the
mean/std-pooled MFCC feature vector (src/preprocessing/features.py's
extract_mfcc_stats, 80-dimensional).

Why this is the model to START with (the "fits available hardware" model):
  - Input is an 80-dimensional vector per utterance, not a spectrogram —
    training a kernel SVM on a few thousand such vectors is a CPU-only,
    sub-minute-to-a-few-minutes job on an ordinary laptop, with no GPU,
    no large memory footprint, and no framework beyond scikit-learn.
  - It directly mirrors the base paper's own SVM baseline configuration
    (RBF kernel), which keeps this project's matched-condition SVM result
    methodologically comparable to the base paper's reported SVM row,
    without assuming the two numbers will match.
  - Its simplicity makes debugging the REST of the pipeline (manifest ->
    features -> training -> thresholding -> evaluation -> latency
    measurement) fast and cheap before committing time to a heavier model.

Tradeoff being accepted: an SVM on pooled statistics discards the
frame-by-frame temporal structure a CNN/LCNN over a spectrogram could use,
so it is expected to underperform a well-trained deep model — exactly as
the base paper's own results show (SVM lowest, LCNN highest in their
Table 1). That expected gap is precisely why src/modeling/lcnn.py exists
as an optional next step, not a replacement.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import BaseSpoofDetectionModel

MODEL_FILENAME = "model.joblib"
META_FILENAME = "svm_meta.json"


class SVMBaseline(BaseSpoofDetectionModel):
    def __init__(self, C: float = 1.0, gamma: str | float = "scale", class_weight: str | None = "balanced",
                 seed: int = 42):
        """
        class_weight="balanced" (default): ASVspoof2019 LA is heavily
        imbalanced (far more spoof than bonafide utterances). Balancing
        class weights is a simple, standard mitigation so the SVM isn't
        just rewarded for always predicting "spoof" — an explicit choice
        worth stating rather than leaving implicit, since it changes what
        the raw predict_proba scores mean relative to an unweighted fit.
        """
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.seed = seed
        self._pipeline: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMBaseline":
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                C=self.C, gamma=self.gamma, kernel="rbf",
                class_weight=self.class_weight, probability=True,
                random_state=self.seed,
            )),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Model not fitted/loaded yet.")
        # predict_proba's column order follows self._pipeline.classes_;
        # we look up the index of class 1 (spoof) explicitly rather than
        # assuming column 1, so this stays correct even in the edge case
        # of a training set missing a class.
        proba = self._pipeline.predict_proba(X)
        spoof_col = list(self._pipeline.classes_).index(1)
        return proba[:, spoof_col]

    def save(self, dir_path: str | Path) -> None:
        if self._pipeline is None:
            raise RuntimeError("Cannot save an unfitted model.")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, dir_path / MODEL_FILENAME)
        with open(dir_path / META_FILENAME, "w", encoding="utf-8") as fh:
            json.dump({"C": self.C, "gamma": self.gamma, "class_weight": self.class_weight, "seed": self.seed}, fh)

    @classmethod
    def load(cls, dir_path: str | Path) -> "SVMBaseline":
        dir_path = Path(dir_path)
        with open(dir_path / META_FILENAME, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        instance = cls(**meta)
        instance._pipeline = joblib.load(dir_path / MODEL_FILENAME)
        return instance
