#!/usr/bin/env python
"""
Train and tune classical models on the 240-dimensional dynamic MFCC features.
Selects the operating threshold on the DEV split ONLY.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger("train_mfcc240")


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute EER via linear interpolation."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    frr = fpr
    far = 1.0 - tpr
    diffs = far - frr
    idx = np.argmin(np.abs(diffs))

    if idx > 0 and np.sign(diffs[idx]) != np.sign(diffs[idx - 1]):
        neighbor = idx - 1
    elif idx < len(diffs) - 1 and np.sign(diffs[idx]) != np.sign(diffs[idx + 1]):
        neighbor = idx + 1
    else:
        neighbor = idx

    if neighbor == idx:
        eer = (far[idx] + frr[idx]) / 2.0
        eer_thresh = float(thresholds[idx])
    else:
        lo, hi = sorted([idx, neighbor])
        x0, x1 = diffs[lo], diffs[hi]
        t = 0.0 if (x1 - x0) == 0 else -x0 / (x1 - x0)
        eer = far[lo] + t * (far[hi] - far[lo])
        eer_thresh = float(thresholds[lo] + t * (thresholds[hi] - thresholds[lo]))

    return float(eer) * 100.0, float(eer_thresh)


def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray, name: str) -> dict:
    t0 = time.time()
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_val)[:, 1]
    else:
        scores = model.decision_function(X_val)

    eer, eer_th = compute_eer(scores, y_val)
    preds = (scores >= eer_th).astype(int)

    acc = np.mean(preds == y_val) * 100.0
    prec = precision_score(y_val, preds, pos_label=1, zero_division=0) * 100.0
    rec = recall_score(y_val, preds, pos_label=1, zero_division=0) * 100.0
    f1 = f1_score(y_val, preds, pos_label=1, zero_division=0) * 100.0
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

    far = (fn / (fn + tp)) * 100.0
    frr = (fp / (fp + tn)) * 100.0

    elapsed = time.time() - t0
    logger.info(
        "%-25s | EER: %6.3f%% (th=%.6f) | Acc: %6.2f%% | Prec: %6.2f%% | Rec: %6.2f%% | F1: %6.2f%% | FRR: %5.2f%% | FAR: %5.2f%% | Time: %.1fs",
        name, eer, eer_th, acc, prec, rec, f1, frr, far, elapsed
    )
    return {
        "name": name,
        "eer": eer,
        "eer_th": eer_th,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "frr": frr,
        "far": far,
        "model": model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="data/features/mfcc240")
    parser.add_argument("--out-dir", default="models/svm_mfcc240_clean")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    feat_dir = ROOT_DIR / args.features_dir
    train_path = feat_dir / "mfcc240_train.npz"
    dev_path = feat_dir / "mfcc240_dev.npz"

    if not train_path.exists() or not dev_path.exists():
        logger.error("Missing feature files in %s. Please run build_dynamic_features.py first.", feat_dir)
        return 1

    logger.info("Loading training data from %s ...", train_path)
    train_data = np.load(train_path)
    X_train, y_train = train_data["X"], train_data["y"]

    logger.info("Loading dev data from %s ...", dev_path)
    dev_data = np.load(dev_path)
    X_dev, y_dev = dev_data["X"], dev_data["y"]

    logger.info("Train: X=%s, y=%s (Bona fide=%d, Spoof=%d)", X_train.shape, y_train.shape, sum(y_train == 0), sum(y_train == 1))
    logger.info("Dev:   X=%s, y=%s (Bona fide=%d, Spoof=%d)", X_dev.shape, y_dev.shape, sum(y_dev == 0), sum(y_dev == 1))

    logger.info("\n" + "=" * 110)
    logger.info("MODEL COMPARISON & HYPERPARAMETER TUNING ON DEV SPLIT ONLY")
    logger.info("=" * 110)

    candidates = []

    # 1. Gradient Boosted Trees (Fast, non-linear baseline)
    logger.info("Training HistGradientBoostingClassifier ...")
    hgb = HistGradientBoostingClassifier(
        max_iter=150, class_weight="balanced", random_state=args.seed, min_samples_leaf=30
    )
    t0 = time.time()
    hgb.fit(X_train, y_train)
    logger.info("Fitted HGB in %.1fs", time.time() - t0)
    candidates.append(evaluate_model(hgb, X_dev, y_dev, "HistGradientBoosting"))

    # 2. SVM with RBF kernel and C tuning
    for c_val in [0.5, 1.0, 2.0, 5.0]:
        name = f"SVM (C={c_val}, RBF)"
        logger.info("Fitting %s ...", name)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=c_val, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=args.seed)),
        ])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        logger.info("Fitted in %.1fs", time.time() - t0)
        candidates.append(evaluate_model(pipe, X_dev, y_dev, name))

    # Pick the model with the lowest EER on Dev
    best = min(candidates, key=lambda c: c["eer"])
    logger.info("\n" + "=" * 110)
    logger.info("BEST MODEL SELECTED ON DEV: %s with EER=%.3f%% (Threshold=%.6f)", best["name"], best["eer"], best["eer_th"])
    logger.info("=" * 110 + "\n")

    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best["model"], out_dir / "model.joblib")
    threshold_meta = {
        "method": "eer",
        "threshold": float(best["eer_th"]),
        "validation_eer": float(best["eer"] / 100.0),
        "validation_accuracy": float(best["acc"] / 100.0),
        "validation_f1": float(best["f1"] / 100.0),
        "model_name": best["name"],
    }
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as fh:
        json.dump(threshold_meta, fh, indent=2)

    config_meta = {
        "feature_key": "mfcc_deltas_stats",
        "n_features": int(X_train.shape[1]),
        "model_type": "svm_baseline" if "SVM" in best["name"] else "hist_gb",
        "train_splits": ["train"],
        "val_splits": ["dev"],
        "seed": args.seed,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config_meta, fh, indent=2)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as fh:
        json.dump({"bonafide": 0, "spoof": 1}, fh, indent=2)

    logger.info("Saved best model to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
