"""
Configuration, label mapping, and experiment-metadata capture.

Every trained model artifact is saved with:
  model.joblib / model.pt   the weights
  config.json                the TrainConfig used to produce it (below)
  label_map.json              LABEL_TO_INT, fixed project-wide
  threshold.json              the validation-selected operating threshold
                               (see src/evaluation/metrics.py select_threshold)
  training_metadata.json       library versions, timestamp, git commit if
                               available, train/val sample counts

so that any saved model is fully traceable to exactly the data, code
version, and settings that produced it — this is what "reproducible"
means operationally for this project, not just a word in the report.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

LABEL_TO_INT = {"bonafide": 0, "spoof": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass
class TrainConfig:
    model_type: str                 # "svm_baseline" | "lcnn"
    feature_key: str                 # "mfcc_stats" | "logmel"
    train_condition: str              # e.g. "clean" — which degradation condition was trained on
    train_splits: list                # e.g. ["train"]
    val_splits: list                  # e.g. ["dev"] — used ONLY for threshold selection, never for fitting
    seed: int = 42
    hyperparameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 — git metadata is best-effort, never fatal
        pass
    return "unknown (not a git repo, or git unavailable)"


def _library_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for module_name in ("numpy", "scipy", "pandas", "sklearn", "joblib", "librosa", "torch"):
        try:
            mod = __import__(module_name)
            versions[module_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "not installed"
    return versions


def build_training_metadata(n_train: int, n_val: int, class_balance_train: dict, class_balance_val: dict) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "git_commit": _git_commit(),
        "library_versions": _library_versions(),
        "n_train": n_train,
        "n_val": n_val,
        "class_balance_train": class_balance_train,
        "class_balance_val": class_balance_val,
    }


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
