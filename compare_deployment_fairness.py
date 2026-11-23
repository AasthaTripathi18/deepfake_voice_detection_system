#!/usr/bin/env python
"""
Deployment Decision Analysis:
Compares the Deep MLP and the Soft-Voting Ensemble at fixed target False Acceptance Rates (FAR)
and False Rejection Rates (FRR) for fair, operationally aligned benchmarking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.inference.scorer import Scorer
from src.evaluation.report import compute_eer, compute_min_tdcf


def evaluate_at_target_far(scores: np.ndarray, labels: np.ndarray, target_far_pct: float) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    far_curve = (1.0 - tpr) * 100.0
    idx = int(np.argmin(np.abs(far_curve - target_far_pct)))
    th = float(thresholds[idx])

    preds = (scores >= th).astype(int)
    acc = float(accuracy_score(labels, preds)) * 100.0
    rec = float(recall_score(labels, preds, pos_label=1, zero_division=0)) * 100.0
    prec = float(precision_score(labels, preds, pos_label=1, zero_division=0)) * 100.0
    f1 = float(f1_score(labels, preds, pos_label=1, zero_division=0)) * 100.0
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    act_far = float((fn / (fn + tp)) * 100.0)
    act_frr = float((fp / (fp + tn)) * 100.0)

    return {
        "target_far_pct": target_far_pct,
        "operating_threshold": round(th, 6),
        "actual_far_pct": round(act_far, 2),
        "actual_frr_pct": round(act_frr, 2),
        "accuracy_pct": round(acc, 2),
        "spoof_recall_pct": round(rec, 2),
        "spoof_precision_pct": round(prec, 2),
        "spoof_f1": round(f1, 4),
        "tn_bonafide_correct": int(tn),
        "fp_false_alarm": int(fp),
        "fn_spoof_missed": int(fn),
        "tp_spoof_caught": int(tp),
    }


def evaluate_at_target_frr(scores: np.ndarray, labels: np.ndarray, target_frr_pct: float) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    frr_curve = fpr * 100.0
    idx = int(np.argmin(np.abs(frr_curve - target_frr_pct)))
    th = float(thresholds[idx])

    preds = (scores >= th).astype(int)
    acc = float(accuracy_score(labels, preds)) * 100.0
    rec = float(recall_score(labels, preds, pos_label=1, zero_division=0)) * 100.0
    prec = float(precision_score(labels, preds, pos_label=1, zero_division=0)) * 100.0
    f1 = float(f1_score(labels, preds, pos_label=1, zero_division=0)) * 100.0
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    act_far = float((fn / (fn + tp)) * 100.0)
    act_frr = float((fp / (fp + tn)) * 100.0)

    return {
        "target_frr_pct": target_frr_pct,
        "operating_threshold": round(th, 6),
        "actual_far_pct": round(act_far, 2),
        "actual_frr_pct": round(act_frr, 2),
        "accuracy_pct": round(acc, 2),
        "spoof_recall_pct": round(rec, 2),
        "spoof_precision_pct": round(prec, 2),
        "spoof_f1": round(f1, 4),
        "tn_bonafide_correct": int(tn),
        "fp_false_alarm": int(fp),
        "fn_spoof_missed": int(fn),
        "tp_spoof_caught": int(tp),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-1", default="models/mlp_mfcc240_clean",
                        help="Path to Model 1 directory (default: models/mlp_mfcc240_clean)")
    parser.add_argument("--model-2", default="models/mlp_lfcc240_clean",
                        help="Path to Model 2 directory (default: models/mlp_lfcc240_clean)")
    parser.add_argument("--features-1", default=None,
                        help="Path to eval features for Model 1 (auto-detected if None)")
    parser.add_argument("--features-2", default=None,
                        help="Path to eval features for Model 2 (auto-detected if None)")
    args = parser.parse_args()

    scorer_1 = Scorer(ROOT_DIR / args.model_1)
    scorer_2 = Scorer(ROOT_DIR / args.model_2)

    # Auto-resolve feature paths
    def resolve_features(scorer, explicit_path):
        if explicit_path:
            return ROOT_DIR / explicit_path
        if "lfcc" in scorer.feature_key.lower():
            return ROOT_DIR / "data/features/lfcc240/lfcc240_eval.npz"
        return ROOT_DIR / "data/features/mfcc240/mfcc240_eval.npz"

    feat_path_1 = resolve_features(scorer_1, args.features_1)
    feat_path_2 = resolve_features(scorer_2, args.features_2)

    if not feat_path_1.exists() or not feat_path_2.exists():
        print(f"Error: Missing feature files: {feat_path_1} or {feat_path_2}", file=sys.stderr)
        return 1

    print("Loading evaluation split features and models...")
    data_1 = np.load(feat_path_1)
    X_ev_1, y_ev = data_1["X"], data_1["y"]

    data_2 = np.load(feat_path_2)
    X_ev_2 = data_2["X"]

    name_1 = scorer_1.threshold_info.get("model_name", Path(args.model_1).name)
    name_2 = scorer_2.threshold_info.get("model_name", Path(args.model_2).name)

    print(f"Scoring {len(y_ev)} evaluation clips with Model 1 ({name_1})...")
    scores_1 = scorer_1.model.predict_proba(X_ev_1)

    print(f"Scoring {len(y_ev)} evaluation clips with Model 2 ({name_2})...")
    scores_2 = scorer_2.model.predict_proba(X_ev_2)

    # 1. Threshold-Independent Metrics
    eer_1, eer_th_1 = compute_eer(scores_1, y_ev)
    eer_2, eer_th_2 = compute_eer(scores_2, y_ev)
    auc_1 = roc_auc_score(y_ev, scores_1)
    auc_2 = roc_auc_score(y_ev, scores_2)
    min_tdcf_1, _ = compute_min_tdcf(scores_1, y_ev)
    min_tdcf_2, _ = compute_min_tdcf(scores_2, y_ev)

    print("\n" + "=" * 115)
    print("SECTION 1: THRESHOLD-INDEPENDENT BENCHMARKS (GLOBAL DISCRIMINATION ABILITY)")
    print("=" * 115)
    bench_df = pd.DataFrame([
        {
            "Model Architecture": name_1,
            "Eval EER (%)": round(eer_1, 3),
            "EER Threshold": round(eer_th_1, 6),
            "ROC-AUC": round(auc_1, 4),
            "min t-DCF": round(min_tdcf_1, 4),
        },
        {
            "Model Architecture": name_2,
            "Eval EER (%)": round(eer_2, 3),
            "EER Threshold": round(eer_th_2, 6),
            "ROC-AUC": round(auc_2, 4),
            "min t-DCF": round(min_tdcf_2, 4),
        },
    ])
    print(bench_df.to_string(index=False))

    # 2. Fair Comparison at Same Target FAR (Fixed Security Risk)
    print("\n" + "=" * 115)
    print("SECTION 2: FAIR OPERATIONAL COMPARISON AT FIXED TARGET FAR (SPOOF MISS RATE)")
    print("In high-security deployment, FAR is fixed to guarantee that no more than a target % of attacks slip past.")
    print("The superior model is the one that achieves LOWER Genuine False Alarms (FRR) and HIGHER Accuracy at that FAR.")
    print("=" * 115)

    far_targets = [15.0, 18.0, 20.0, 25.0, 30.0]
    comp_rows = []
    for tgt in far_targets:
        res_1 = evaluate_at_target_far(scores_1, y_ev, tgt)
        res_2 = evaluate_at_target_far(scores_2, y_ev, tgt)

        comp_rows.append({
            "Target FAR": f"{tgt:.1f}%",
            "Model": name_1,
            "Threshold": res_1["operating_threshold"],
            "Actual FAR": f"{res_1['actual_far_pct']}%",
            "False Alarm FRR": f"{res_1['actual_frr_pct']}%",
            "Accuracy": f"{res_1['accuracy_pct']}%",
            "Spoof Caught (TP)": f"{res_1['tp_spoof_caught']:,}",
            "False Alarms (FP)": f"{res_1['fp_false_alarm']:,}",
            "Precision": f"{res_1['spoof_precision_pct']}%",
            "F1-Score": res_1["spoof_f1"],
        })
        comp_rows.append({
            "Target FAR": f"{tgt:.1f}%",
            "Model": name_2,
            "Threshold": res_2["operating_threshold"],
            "Actual FAR": f"{res_2['actual_far_pct']}%",
            "False Alarm FRR": f"{res_2['actual_frr_pct']}%",
            "Accuracy": f"{res_2['accuracy_pct']}%",
            "Spoof Caught (TP)": f"{res_2['tp_spoof_caught']:,}",
            "False Alarms (FP)": f"{res_2['fp_false_alarm']:,}",
            "Precision": f"{res_2['spoof_precision_pct']}%",
            "F1-Score": res_2["spoof_f1"],
        })

    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))

    # 3. Fair Comparison at Same Target FRR (Fixed Usability / Genuine Acceptance)
    print("\n" + "=" * 115)
    print("SECTION 3: FAIR OPERATIONAL COMPARISON AT FIXED TARGET FRR (GENUINE FALSE ALARMS)")
    print("In user-facing deployment, FRR is fixed to guarantee that no more than a target % of real users are annoyed.")
    print("The superior model is the one that catches MORE spoofs (LOWER FAR) at that fixed false-alarm budget.")
    print("=" * 115)

    frr_targets = [10.0, 15.0, 20.0, 25.0, 30.0]
    frr_rows = []
    for tgt in frr_targets:
        res_1 = evaluate_at_target_frr(scores_1, y_ev, tgt)
        res_2 = evaluate_at_target_frr(scores_2, y_ev, tgt)

        frr_rows.append({
            "Target FRR": f"{tgt:.1f}%",
            "Model": name_1,
            "Threshold": res_1["operating_threshold"],
            "Actual FRR": f"{res_1['actual_frr_pct']}%",
            "Spoof Miss FAR": f"{res_1['actual_far_pct']}%",
            "Spoof Caught (TP)": f"{res_1['tp_spoof_caught']:,}",
            "Accuracy": f"{res_1['accuracy_pct']}%",
            "Real Voices Correct (TN)": f"{res_1['tn_bonafide_correct']:,}",
            "Precision": f"{res_1['spoof_precision_pct']}%",
            "F1-Score": res_1["spoof_f1"],
        })
        frr_rows.append({
            "Target FRR": f"{tgt:.1f}%",
            "Model": name_2,
            "Threshold": res_2["operating_threshold"],
            "Actual FRR": f"{res_2['actual_frr_pct']}%",
            "Spoof Miss FAR": f"{res_2['actual_far_pct']}%",
            "Spoof Caught (TP)": f"{res_2['tp_spoof_caught']:,}",
            "Accuracy": f"{res_2['accuracy_pct']}%",
            "Real Voices Correct (TN)": f"{res_2['tn_bonafide_correct']:,}",
            "Precision": f"{res_2['spoof_precision_pct']}%",
            "F1-Score": res_2["spoof_f1"],
        })

    frr_df = pd.DataFrame(frr_rows)
    print(frr_df.to_string(index=False))

    out_csv = ROOT_DIR / "results/deployment_fair_comparison.csv"
    comp_df.to_csv(out_csv, index=False)
    print(f"\nSaved fair deployment comparison to: {out_csv}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
