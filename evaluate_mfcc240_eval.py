#!/usr/bin/env python
"""
Evaluate the trained 240-dim dynamic MFCC model on the official ASVspoof 2019 LA EVAL split.
Threshold is fixed from threshold.json (selected on dev split only).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger("evaluate_mfcc240_eval")


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    model_dir = ROOT_DIR / "models/svm_mfcc240_clean"
    eval_feat_path = ROOT_DIR / "data/features/mfcc240/mfcc240_eval.npz"
    out_dir = ROOT_DIR / "results/metrics/svm_mfcc240_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eval_feat_path.exists():
        logger.error("Missing %s. Please extract eval features first.", eval_feat_path)
        return 1

    logger.info("Loading model from %s ...", model_dir)
    pipeline = joblib.load(model_dir / "model.joblib")
    with open(model_dir / "threshold.json", "r", encoding="utf-8") as fh:
        thresh_info = json.load(fh)
    operating_threshold = float(thresh_info["threshold"])
    logger.info("Fixed dev-selected operating threshold: %.6f (Dev EER: %.3f%%)",
                operating_threshold, float(thresh_info["validation_eer"]) * 100.0)

    logger.info("Loading eval features from %s ...", eval_feat_path)
    t0 = time.time()
    data = np.load(eval_feat_path)
    X_eval, y_eval, file_ids = data["X"], data["y"], data["file_ids"]
    logger.info("Loaded %d eval samples in %.1fs (Bona fide: %d, Spoof: %d)",
                len(y_eval), time.time() - t0, sum(y_eval == 0), sum(y_eval == 1))

    logger.info("Scoring %d eval samples with SVM ...", len(X_eval))
    t0 = time.time()
    # Batch predict_proba to be memory safe and fast
    scores = pipeline.predict_proba(X_eval)[:, 1]
    inference_time = time.time() - t0
    logger.info("Scored all %d samples in %.2fs (%.2f ms/sample)",
                len(X_eval), inference_time, (inference_time / len(X_eval)) * 1000.0)

    # 1. EER (Threshold-independent benchmark)
    eer, eer_th = compute_eer(scores, y_eval)

    # 2. Operating metrics at fixed dev threshold
    preds = (scores >= operating_threshold).astype(int)
    acc = np.mean(preds == y_eval) * 100.0
    prec = precision_score(y_eval, preds, pos_label=1, zero_division=0) * 100.0
    rec = recall_score(y_eval, preds, pos_label=1, zero_division=0) * 100.0
    f1 = f1_score(y_eval, preds, pos_label=1, zero_division=0) * 100.0
    tn, fp, fn, tp = confusion_matrix(y_eval, preds).ravel()

    far = (fn / (fn + tp)) * 100.0
    frr = (fp / (fp + tn)) * 100.0

    print("\n" + "=" * 100)
    print("OFFICIAL ASVSPOOF 2019 LA EVALUATION RESULTS (240-DIM DYNAMIC MFCC + TUNED SVM)")
    print("=" * 100)
    print(f"Condition:                          clean (matched)")
    print(f"Total Evaluation Samples:           {len(y_eval)}")
    print(f"Bona Fide (Genuine) Count:          {tn + fp}")
    print(f"Spoof (Synthetic/Cloned) Count:     {fn + tp}")
    print("-" * 100)
    print(f"Equal Error Rate (EER):             {eer:.3f}% (eval EER threshold: {eer_th:.6f})")
    print(f"Fixed Operating Threshold:          {operating_threshold:.6f} (selected on dev)")
    print(f"Accuracy:                           {acc:.2f}%")
    print(f"Spoof Precision:                    {prec:.2f}%")
    print(f"Spoof Recall:                       {rec:.2f}%")
    print(f"Spoof F1-Score:                     {f1:.2f}%")
    print(f"Genuine False Alarm Rate (FRR):     {frr:.2f}% ({fp} / {tn + fp} genuine flagged as spoof)")
    print(f"Spoof Miss Rate (FAR):              {far:.2f}% ({fn} / {fn + tp} spoofs missed)")
    print("-" * 100)
    print(f"Confusion Matrix (TN / FP / FN / TP): {tn} / {fp} / {fn} / {tp}")
    print("=" * 100 + "\n")

    # Save summary CSV
    summary_df = pd.DataFrame([{
        "condition": "clean",
        "n_eval_samples": len(y_eval),
        "eer_percent": round(eer, 3),
        "eer_threshold": round(eer_th, 6),
        "operating_threshold": round(operating_threshold, 6),
        "far_percent": round(far, 3),
        "frr_percent": round(frr, 3),
        "precision_spoof": round(prec / 100.0, 4),
        "recall_spoof": round(rec / 100.0, 4),
        "f1_spoof": round(f1 / 100.0, 4),
        "accuracy": round(acc / 100.0, 4),
        "cm_true_bonafide_pred_bonafide": tn,
        "cm_true_bonafide_pred_spoof": fp,
        "cm_true_spoof_pred_bonafide": fn,
        "cm_true_spoof_pred_spoof": tp,
        "latency_ms_per_sample": round((inference_time / len(X_eval)) * 1000.0, 4),
    }])
    summary_csv = out_dir / "summary_eval.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Saved summary to %s", summary_csv)

    # Save predictions
    pred_df = pd.DataFrame({
        "file_id": file_ids,
        "label_int": y_eval,
        "label": ["bonafide" if y == 0 else "spoof" for y in y_eval],
        "score": scores,
        "predicted_label": ["spoof" if p == 1 else "bonafide" for p in preds],
    })
    scores_csv = out_dir / "scores_eval.csv"
    pred_df.to_csv(scores_csv, index=False)
    logger.info("Saved per-clip predictions to %s", scores_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
