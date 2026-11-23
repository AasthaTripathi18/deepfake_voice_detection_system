#!/usr/bin/env python
"""
Robustness Evaluation Runner:
Tests model degradation and resilience across controlled noise and channel conditions:
- Clean baseline
- Additive Gaussian Noise at SNR = 20 dB, 10 dB, 5 dB
- Telephony Bandpass Filtering (300 Hz - 3400 Hz)
"""

from __future__ import annotations

try:
    # Pre-load onnxruntime before sklearn/scipy to prevent Windows C++ runtime DLL crash
    import onnxruntime  # noqa: F401
except (ImportError, OSError):
    pass

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset.asvspoof2019 import resolve_asvspoof_audio_path
from src.inference.scorer import Scorer
from src.preprocessing import cleaning, degradations, features as feat_mod
from src.evaluation.report import compute_eer, far_frr_at_threshold, compute_min_tdcf

logger = logging.getLogger("evaluate_robustness")


def _extract_feature_for_scorer(waveform: np.ndarray, sr: int, feature_key: str, apply_cmvn: bool) -> np.ndarray:
    """Extract the exact feature representation required by `scorer.feature_key`."""
    if feature_key == "mfcc_deltas_stats":
        return feat_mod.extract_mfcc_deltas_stats(waveform, sr=sr, apply_cmvn=apply_cmvn)
    if feature_key == "lfcc_deltas_stats":
        return feat_mod.extract_lfcc_deltas_stats(waveform, sr=sr, apply_cmvn=apply_cmvn)
    if feature_key == "lfcc_mfcc_fusion":
        return feat_mod.extract_lfcc_mfcc_fusion(waveform, sr=sr, apply_cmvn=apply_cmvn)
    return feat_mod.build_feature_record(waveform, sr, feature_set=feature_key)[feature_key]


def evaluate_condition_on_audio(
    scorer: Scorer,
    audio_paths: list[Path],
    labels: list[int],
    condition_name: str,
    snr_db: float | None = None,
    apply_bandpass: bool = False,
    apply_cmvn: bool = False,
) -> dict:
    t0 = time.time()
    scores = []
    y_true = []

    for path, label in zip(audio_paths, labels):
        try:
            waveform, sr = cleaning.load_audio(path, target_sr=16000)
            waveform = cleaning.clean(waveform, sr, trim=True, normalize=True)

            if snr_db is not None:
                waveform = degradations.add_noise_at_snr(waveform, snr_db=snr_db)

            if apply_bandpass:
                waveform = degradations.bandpass_telephony(waveform, sr=sr)

            feat = _extract_feature_for_scorer(waveform, sr, scorer.feature_key, apply_cmvn)
            res = scorer.score_features(feat)
            scores.append(res["score"])
            y_true.append(label)
        except Exception as exc:
            logger.debug("Failed processing %s: %s", path, exc)
            continue

    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    elapsed = time.time() - t0

    if len(scores) == 0:
        return {"condition": condition_name, "error": "No audio processed"}

    eer, eer_th = compute_eer(scores, y_true)
    min_tdcf, _ = compute_min_tdcf(scores, y_true)
    far_frr = far_frr_at_threshold(scores, y_true, scorer.threshold)
    preds = (scores >= scorer.threshold).astype(int)

    acc = float(accuracy_score(y_true, preds)) * 100.0
    rec = float(recall_score(y_true, preds, pos_label=1, zero_division=0)) * 100.0
    prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0)) * 100.0
    f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0)) * 100.0

    return {
        "condition": condition_name,
        "n_samples": len(scores),
        "eer_percent": round(eer * 100.0, 3),
        "min_tdcf": round(min_tdcf, 4),
        "operating_threshold": round(scorer.threshold, 4),
        "accuracy": round(acc, 2),
        "spoof_recall": round(rec, 2),
        "spoof_precision": round(prec, 2),
        "spoof_f1": round(f1, 2),
        "far_percent": round(far_frr["far"] * 100.0, 2),
        "frr_percent": round(far_frr["frr"] * 100.0, 2),
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/mlp_mfcc240_clean")
    parser.add_argument("--manifest", default="data/manifests/asvspoof2019_la.csv")
    parser.add_argument("--split", default="dev", choices=["dev", "eval"])
    parser.add_argument("--audio-root", default=None, help="Optional override root directory containing ASVspoof2019 LA folders")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of evaluation samples to test per condition (default: 200 for rapid audit)")
    parser.add_argument("--out-dir", default="results/robustness")
    parser.add_argument("--cmvn", action="store_true", help="Apply Cepstral Mean and Variance Normalization")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    scorer = Scorer(args.model_dir)
    logger.info("Loaded model from %s (feature_key=%s) with threshold %.4f",
                args.model_dir, scorer.feature_key, scorer.threshold)

    # Load audio paths from manifest
    manifest_csv = Path(args.manifest)
    if not manifest_csv.is_absolute():
        manifest_csv = ROOT_DIR / manifest_csv
    if not manifest_csv.exists():
        logger.error("Missing manifest: %s", manifest_csv)
        return 1

    df = pd.read_csv(manifest_csv)
    df = df[(df["split"] == args.split) & (df["status"] == "valid")]
    bf_df = df[df["label"] == "bonafide"].head(args.n_samples // 2)
    sp_df = df[df["label"] == "spoof"].head(args.n_samples // 2)
    sample_df = pd.concat([bf_df, sp_df]).sample(frac=1.0, random_state=42)

    audio_paths = [
        resolve_asvspoof_audio_path(row.file_path, row.file_id, row.split, args.audio_root)
        for row in sample_df.itertuples(index=False)
    ]
    labels = [0 if l == "bonafide" else 1 for l in sample_df["label"]]

    conditions = [
        {"name": "clean", "snr_db": None, "apply_bandpass": False},
        {"name": "noise_snr_20db", "snr_db": 20.0, "apply_bandpass": False},
        {"name": "noise_snr_10db", "snr_db": 10.0, "apply_bandpass": False},
        {"name": "noise_snr_5db", "snr_db": 5.0, "apply_bandpass": False},
        {"name": "telephony_bandpass", "snr_db": None, "apply_bandpass": True},
    ]

    results = []
    print("\n" + "=" * 90)
    print(f"ROBUSTNESS BENCHMARK ACROSS CHANNEL & NOISE CONDITIONS (CMVN={args.cmvn})")
    print("=" * 90)

    for cond in conditions:
        logger.info("Evaluating condition: %s ...", cond["name"])
        res = evaluate_condition_on_audio(
            scorer,
            audio_paths,
            labels,
            condition_name=cond["name"],
            snr_db=cond["snr_db"],
            apply_bandpass=cond["apply_bandpass"],
            apply_cmvn=args.cmvn,
        )
        results.append(res)
        logger.info(
            "%-20s | EER: %5.2f%% | Acc: %5.2f%% | Rec: %5.2f%% | FRR: %5.2f%%",
            res["condition"], res["eer_percent"], res["accuracy"], res["spoof_recall"], res["frr_percent"],
        )

    out_path = ROOT_DIR / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    csv_file = out_path / f"robustness_{Path(args.model_dir).name}.csv"
    res_df.to_csv(csv_file, index=False)
    print("\n" + res_df[["condition", "n_samples", "eer_percent", "accuracy", "spoof_recall", "frr_percent", "min_tdcf"]].to_string(index=False))
    print(f"\nSaved robustness report to: {csv_file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
