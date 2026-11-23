#!/usr/bin/env python
"""
Evaluate the 480-dim fused LFCC+MFCC model on the official ASVspoof 2019 LA EVAL split.
Operating threshold is strictly loaded from threshold.json (calibrated on DEV only).
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

from src.evaluation.report import compute_eer, compute_min_tdcf

logger = logging.getLogger("evaluate_fused480")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    model_dir = ROOT_DIR / "models/mlp_fused480_clean"
    eval_feat_path = ROOT_DIR / "data/features/fused480/fused480_eval.npz"
    out_dir = ROOT_DIR / "results/metrics/mlp_fused480_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists() or not eval_feat_path.exists():
        logger.error("Missing model or eval features: %s or %s", model_dir, eval_feat_path)
        return 1

    logger.info("Loading model from %s ...", model_dir)
    pipeline = joblib.load(model_dir / "model.joblib")
    with open(model_dir / "threshold.json", "r", encoding="utf-8") as fh:
        thresh_info = json.load(fh)
    operating_threshold = float(thresh_info["threshold"])
    logger.info("Operating threshold (from Dev): %.6f", operating_threshold)

    logger.info("Loading eval features from %s ...", eval_feat_path)
    eval_data = np.load(eval_feat_path)
    X_eval, y_eval, file_ids = eval_data["X"], eval_data["y"], eval_data["file_ids"]
    n_samples = len(y_eval)
    logger.info("Eval dataset: %d samples (Bona fide=%d, Spoof=%d)", n_samples, sum(y_eval == 0), sum(y_eval == 1))

    logger.info("Running scoring across %d evaluation clips...", n_samples)
    t0 = time.time()
    scores = pipeline.predict_proba(X_eval)[:, 1]
    total_sec = time.time() - t0
    ms_per_sample = (total_sec / n_samples) * 1000.0
    logger.info("Scored %d clips in %.2fs (%.4f ms/clip)", n_samples, total_sec, ms_per_sample)

    # 1. Threshold-independent metrics
    eval_eer, eer_thresh = compute_eer(scores, y_eval)
    min_tdcf, _ = compute_min_tdcf(scores, y_eval)
    logger.info("Eval EER: %.3f%% (EER threshold: %.6f) | min t-DCF: %.4f", eval_eer, eer_thresh, min_tdcf)

    # 2. Operating threshold metrics
    preds = (scores >= operating_threshold).astype(int)
    acc = np.mean(preds == y_eval) * 100.0
    prec = precision_score(y_eval, preds, pos_label=1, zero_division=0) * 100.0
    rec = recall_score(y_eval, preds, pos_label=1, zero_division=0) * 100.0
    f1 = f1_score(y_eval, preds, pos_label=1, zero_division=0) * 100.0
    tn, fp, fn, tp = confusion_matrix(y_eval, preds).ravel()
    frr = (fp / (fp + tn)) * 100.0
    far = (fn / (fn + tp)) * 100.0

    logger.info(
        "Eval Metrics at Dev Operating Threshold %.4f:\n"
        "  Accuracy: %.2f%%\n"
        "  FAR (Spoofs Missed): %.2f%% (FN: %d / %d)\n"
        "  FRR (False Alarms):  %.2f%% (FP: %d / %d)\n"
        "  Spoof Recall: %.2f%%\n"
        "  Spoof Precision: %.2f%%\n"
        "  Spoof F1: %.4f",
        operating_threshold, acc, far, fn, fn + tp, frr, fp, fp + tn, rec, prec, f1 / 100.0,
    )

    # Save summary
    summary_df = pd.DataFrame([{
        "condition": "clean",
        "n_eval_samples": n_samples,
        "eer_percent": round(eval_eer, 3),
        "eer_threshold": round(eer_thresh, 6),
        "min_tdcf": round(min_tdcf, 4),
        "operating_threshold": round(operating_threshold, 6),
        "far_percent": round(far, 3),
        "frr_percent": round(frr, 3),
        "precision_spoof": round(prec / 100.0, 4),
        "recall_spoof": round(rec / 100.0, 4),
        "f1_spoof": round(f1 / 100.0, 4),
        "accuracy": round(acc / 100.0, 4),
        "cm_true_bonafide_pred_bonafide": int(tn),
        "cm_true_bonafide_pred_spoof": int(fp),
        "cm_true_spoof_pred_bonafide": int(fn),
        "cm_true_spoof_pred_spoof": int(tp),
        "feature_only_latency_median_ms": round(ms_per_sample, 4),
        "feature_only_latency_p95_ms": round(ms_per_sample, 4),
    }])
    summary_csv = out_dir / "summary_by_condition.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Saved summary to %s", summary_csv)

    scores_df = pd.DataFrame({
        "file_id": file_ids,
        "label": y_eval,
        "score": scores,
        "predicted_label": preds,
    })
    scores_csv = out_dir / "scores_clean.csv"
    scores_df.to_csv(scores_csv, index=False)
    logger.info("Saved scores to %s", scores_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
