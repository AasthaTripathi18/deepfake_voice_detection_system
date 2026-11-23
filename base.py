"""
Common model interface. A model is anything that can:
  - fit(X, y)                    train on a feature matrix + integer labels
  - predict_proba(X) -> ndarray  return P(spoof) in [0, 1] for each row
  - save(dir_path)                write everything needed to reload it
  - load(dir_path)  [classmethod]  reconstruct a fitted model from disk

Keeping this interface identical across models is what lets
scripts/train_model.py, src/inference/scorer.py, and src/evaluation/*
stay model-agnostic — adding LCNN later (see lcnn.py) does not require
touching any of those.

Label convention (fixed project-wide, see config.py LABEL_TO_INT):
    bonafide -> 0
    spoof    -> 1
predict_proba returns P(class == 1) == P(spoof), i.e. higher score means
"more likely synthetic." This is the "spoof-risk score" referenced in the
project plan and architecture documents.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np


class BaseSpoofDetectionModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSpoofDetectionModel":
        ...

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D array of P(spoof) in [0, 1], same length as X."""
        ...

    @abc.abstractmethod
    def save(self, dir_path: str | Path) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, dir_path: str | Path) -> "BaseSpoofDetectionModel":
        ...
