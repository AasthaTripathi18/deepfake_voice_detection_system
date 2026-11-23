"""
OPTIONAL comparison model: a Light CNN (LCNN) with Max-Feature-Map (MFM)
activations, consuming the log-mel spectrogram feature
(src/preprocessing/features.py's extract_logmel_spectrogram, fixed shape
128 x 200), matching the base paper's description (Section VI-D: six
convolutional layers with MFM activations, two fully-connected layers,
sigmoid output).

WHY THIS IS OPTIONAL, NOT THE DEFAULT:
  - It requires PyTorch and meaningfully more compute than the SVM
    baseline — training even a small LCNN over a full ASVspoof2019 LA
    training split (25k+ utterances) for the base paper's stated 50
    epochs is a GPU-hours-scale job, not a laptop-CPU-minutes job. If
    your hardware doesn't have a usable GPU, start with the SVM baseline
    (svm_baseline.py) and only add this if time/compute allow, potentially
    with fewer epochs and a smaller training subset than the base paper
    used — which would need to be reported as a methodological difference,
    not hidden.
  - This project's environment for developing/testing this code did not
    have PyTorch installed and had no network access to install it (see
    the chat response for exactly what was and wasn't verified). This
    module has been checked for syntax correctness only
    (`python -m py_compile`), not run end-to-end — treat it as a
    starting point to adapt and test in your own environment, not as a
    verified-working training pipeline the way svm_baseline.py is.

HOW TO ADD IT AS A COMPARISON (once you have PyTorch + adequate compute):
  1. `pip install torch` (choose the CPU or CUDA build matching your machine).
  2. Build log-mel features if you haven't already:
         python scripts/build_features.py ... --feature-set logmel ...
     (or --feature-set both, which also keeps mfcc_stats for the SVM).
  3. Run scripts/train_model.py with --model lcnn --feature-key logmel
     instead of --model svm_baseline --feature-key mfcc_stats — the rest
     of the training/evaluation/latency-measurement scripts are model-
     agnostic and need no changes, because this class implements the same
     BaseSpoofDetectionModel interface as SVMBaseline.
  4. Expect a much longer training time and consider starting with fewer
     epochs than the base paper's 50 to get a first result quickly, then
     increasing if time allows — record whatever epoch count you actually
     used in the saved TrainConfig (this happens automatically).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import BaseSpoofDetectionModel

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    TORCH_AVAILABLE = False


MODEL_FILENAME = "model.pt"
META_FILENAME = "lcnn_meta.json"


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. LCNN is an optional comparison model — "
            "install it with `pip install torch` (see this module's docstring) "
            "before using --model lcnn. The SVM baseline (svm_baseline.py) does "
            "not require PyTorch and is the recommended starting model."
        )


if TORCH_AVAILABLE:

    class MaxFeatureMap2d(nn.Module):
        """Splits channels in half and takes the elementwise max — the MFM
        activation the base paper credits with LCNN's selective feature
        learning (Section VII.D)."""

        def forward(self, x):
            a, b = torch.chunk(x, 2, dim=1)
            return torch.max(a, b)

    class LightCNN(nn.Module):
        """
        Compact LCNN: matches the base paper's stated shape (six conv
        layers with MFM, two FC layers, sigmoid output) at a small enough
        channel width to be trainable without a large GPU. Channel widths
        are a reasonable, documented choice for this project's scale, not
        a claim of matching the base paper's exact (unpublished) layer
        sizes — the base paper does not specify per-layer channel counts.
        """

        def __init__(self, n_mels: int = 128, n_frames: int = 200, dropout: float = 0.3):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.mfm1 = MaxFeatureMap2d()
            self.pool1 = nn.MaxPool2d(2)

            self.conv2 = nn.Conv2d(16, 48, kernel_size=3, padding=1)
            self.mfm2 = MaxFeatureMap2d()
            self.pool2 = nn.MaxPool2d(2)

            self.conv3 = nn.Conv2d(24, 64, kernel_size=3, padding=1)
            self.mfm3 = MaxFeatureMap2d()

            self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.mfm4 = MaxFeatureMap2d()
            self.pool3 = nn.MaxPool2d(2)

            self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.mfm5 = MaxFeatureMap2d()

            self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
            self.mfm6 = MaxFeatureMap2d()
            self.pool4 = nn.MaxPool2d(2)

            self.dropout = nn.Dropout(dropout)

            # Compute flattened size dynamically instead of hardcoding it,
            # since it depends on n_mels/n_frames — hardcoding here is a
            # common, easy-to-silently-break-later source of shape errors.
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_mels, n_frames)
                flat_size = self._conv_forward(dummy).numel()

            self.fc1 = nn.Linear(flat_size, 160)
            self.fc_mfm = MaxFeatureMap1d()
            self.fc2 = nn.Linear(80, 1)

        def _conv_forward(self, x):
            x = self.pool1(self.mfm1(self.conv1(x)))
            x = self.pool2(self.mfm2(self.conv2(x)))
            x = self.mfm3(self.conv3(x))
            x = self.pool3(self.mfm4(self.conv4(x)))
            x = self.mfm5(self.conv5(x))
            x = self.pool4(self.mfm6(self.conv6(x)))
            return x

        def forward(self, x):
            x = self._conv_forward(x)
            x = torch.flatten(x, start_dim=1)
            x = self.dropout(x)
            x = self.fc1(x)
            x = self.fc_mfm(x)
            logit = self.fc2(x).squeeze(-1)
            return logit  # raw logit; sigmoid applied in predict_proba / loss

    class MaxFeatureMap1d(nn.Module):
        def forward(self, x):
            a, b = torch.chunk(x, 2, dim=1)
            return torch.max(a, b)


class LCNNBaseline(BaseSpoofDetectionModel):
    def __init__(self, n_mels: int = 128, n_frames: int = 200, lr: float = 1e-4,
                 batch_size: int = 64, epochs: int = 50, dropout: float = 0.3,
                 weight_decay: float = 1e-5, patience: int = 10, seed: int = 42,
                 device: str | None = None):
        _require_torch()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: "LightCNN | None" = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "LCNNBaseline":
        """
        X: shape (n, n_mels, n_frames). X_val/y_val are optional; if given,
        used ONLY for early stopping (monitoring validation loss), never
        for gradient updates — this keeps the same train/validation
        separation principle as the SVM baseline's threshold selection,
        just applied to early-stopping instead.
        """
        torch.manual_seed(self.seed)
        self._model = LightCNN(self.n_mels, self.n_frames, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_t = torch.from_numpy(X).float().unsqueeze(1).to(self.device)
        y_t = torch.from_numpy(y).float().to(self.device)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        n = X_t.shape[0]
        rng = np.random.default_rng(self.seed)

        for epoch in range(self.epochs):
            self._model.train()
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_x, batch_y = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= n

            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val)
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        return self

    def _eval_loss(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        self._model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_val).float().unsqueeze(1).to(self.device)
            y_t = torch.from_numpy(y_val).float().to(self.device)
            logits = self._model(X_t)
            return F.binary_cross_entropy_with_logits(logits, y_t).item()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted/loaded yet.")
        self._model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().unsqueeze(1).to(self.device)
            logits = self._model(X_t)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def save(self, dir_path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), dir_path / MODEL_FILENAME)
        meta = {
            "n_mels": self.n_mels, "n_frames": self.n_frames, "lr": self.lr,
            "batch_size": self.batch_size, "epochs": self.epochs, "dropout": self.dropout,
            "weight_decay": self.weight_decay, "patience": self.patience, "seed": self.seed,
        }
        with open(dir_path / META_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)

    @classmethod
    def load(cls, dir_path: str | Path) -> "LCNNBaseline":
        _require_torch()
        dir_path = Path(dir_path)
        with open(dir_path / META_FILENAME, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        instance = cls(**meta)
        instance._model = LightCNN(instance.n_mels, instance.n_frames, instance.dropout).to(instance.device)
        instance._model.load_state_dict(torch.load(dir_path / MODEL_FILENAME, map_location=instance.device))
        return instance
