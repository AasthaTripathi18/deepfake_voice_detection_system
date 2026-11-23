#!/usr/bin/env python
"""
Train the Deep Multi-Layer Perceptron (MLP) model on the 240-dim dynamic MFCC features.
Tunes hyperparameters and chooses the operating threshold on the DEV split only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.modeling.mlp_baseline import MLPBaseline

logger = logging.getLogger("train_mlp")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="data/features/mfcc240")
    parser.add_argument("--feature-key", default=None,
                        help="Feature key: 'mfcc_deltas_stats', 'lfcc_deltas_stats', or 'lfcc_mfcc_fusion'")
    parser.add_argument("--out-dir", default="models/mlp_mfcc240_clean")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Calibrated operating threshold (default: auto-calibrated from Dev EER)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    feat_dir = ROOT_DIR / args.features_dir

    # Auto-detect feature key and prefix
    feat_name = feat_dir.name.lower()
    if "fused" in feat_name or "480" in feat_name:
        prefix = "fused480"
        feature_key = args.feature_key or "lfcc_mfcc_fusion"
        model_name = "Deep MLP (256-128-64) with 480-dim LFCC+MFCC Fusion"
    elif "lfcc" in feat_name:
        prefix = "lfcc240"
        feature_key = args.feature_key or "lfcc_deltas_stats"
        model_name = "Deep MLP (256-128-64) with 240-dim Dynamic LFCCs"
    else:
        prefix = "mfcc240"
        feature_key = args.feature_key or "mfcc_deltas_stats"
        model_name = "Deep MLP (256-128-64) with 240-dim Dynamic MFCCs"

    train_path = feat_dir / f"{prefix}_train.npz"
    dev_path = feat_dir / f"{prefix}_dev.npz"

    if not train_path.exists() or not dev_path.exists():
        logger.error("Missing feature files in %s: %s or %s", feat_dir, train_path, dev_path)
        return 1

    logger.info("Loading training data from %s ...", train_path)
    train_data = np.load(train_path)
    X_train, y_train = train_data["X"], train_data["y"]

    logger.info("Loading dev data from %s ...", dev_path)
    dev_data = np.load(dev_path)
    X_dev, y_dev = dev_data["X"], dev_data["y"]

    logger.info("Train: X=%s, y=%s (Bona fide=%d, Spoof=%d)", X_train.shape, y_train.shape, sum(y_train == 0), sum(y_train == 1))
    logger.info("Dev:   X=%s, y=%s (Bona fide=%d, Spoof=%d)", X_dev.shape, y_dev.shape, sum(y_dev == 0), sum(y_dev == 1))

    logger.info("Fitting Deep MLP (256, 128, 64) with L2 regularization alpha=0.01 ...")
    model = MLPBaseline(hidden_layer_sizes=(256, 128, 64), alpha=0.01, seed=args.seed)
    t0 = time.time()
    model.fit(X_train, y_train)
    logger.info("Fitted MLP in %.1fs", time.time() - t0)

    # Evaluate on Dev split
    dev_scores = model.predict_proba(X_dev)
    dev_eer, eer_th = compute_eer(dev_scores, y_dev)
    logger.info("Development Split EER: %.3f%% (EER threshold: %.6f)", dev_eer, eer_th)

    operating_thresh = args.threshold if args.threshold is not None else float(eer_th)
    preds = (dev_scores >= operating_thresh).astype(int)
    acc = np.mean(preds == y_dev) * 100.0
    prec = precision_score(y_dev, preds, pos_label=1, zero_division=0) * 100.0
    rec = recall_score(y_dev, preds, pos_label=1, zero_division=0) * 100.0
    f1 = f1_score(y_dev, preds, pos_label=1, zero_division=0) * 100.0
    tn, fp, fn, tp = confusion_matrix(y_dev, preds).ravel()
    frr = (fp / (fp + tn)) * 100.0
    far = (fn / (fn + tp)) * 100.0

    logger.info(
        "Dev Metrics at Operating Threshold %.4f: Acc=%.2f%% | Prec=%.2f%% | Rec=%.2f%% | F1=%.2f%% | FRR=%.2f%% | FAR=%.2f%%",
        operating_thresh, acc, prec, rec, f1, frr, far,
    )

    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir)

    threshold_meta = {
        "method": "calibrated_eer_or_custom",
        "threshold": float(operating_thresh),
        "validation_eer": float(dev_eer / 100.0),
        "eer_threshold": float(eer_th),
        "validation_accuracy": float(acc / 100.0),
        "validation_f1": float(f1 / 100.0),
        "model_name": model_name,
        "note": "Dev-calibrated operating threshold to balance detection on unseen evaluation attacks while keeping false alarms low",
    }
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as fh:
        json.dump(threshold_meta, fh, indent=2)

    config_meta = {
        "feature_key": feature_key,
        "n_features": int(X_train.shape[1]),
        "model_type": "mlp_baseline",
        "train_splits": ["train"],
        "val_splits": ["dev"],
        "seed": args.seed,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config_meta, fh, indent=2)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as fh:
        json.dump({"bonafide": 0, "spoof": 1}, fh, indent=2)

    logger.info("Saved trained MLP model to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
