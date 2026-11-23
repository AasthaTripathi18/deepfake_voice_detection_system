#!/usr/bin/env python
"""
Diagnostic evaluation on the ASVspoof 2019 LA DEVELOPMENT split ONLY.

Compares:
  Preprocessing:
    P1: Current live-microphone streaming behavior (raw 16kHz audio, unnormalized, sliced into 2s windows)
    P2: Whole-recording training-matched preprocessing (leading/trailing silence trimmed,
        whole recording peak-normalized to -3 dBFS, then sliced into 2s windows without per-window re-normalization)
  Aggregation Rules:
    A1a: Actual App Severity Hierarchy Rule (reproducing assess_clip discrete logic)
    A1b: Continuous Worst-Window Max Score (max_i s_i)
    A2:  Mean Score (1/M sum s_i)
    A3:  Median Score (median(s_i))
    A4:  75th-Percentile Score (P75(s_i))
    A5:  Time-Voting Ratio (fraction of unique active speech time with s_k >= tau_win, tau_win=0.937403)

STRICT GUARDRAILS:
  - Reads and evaluates ONLY split == 'dev' from data/manifests/asvspoof2019_la.csv.
  - Zero access to evaluation split ('eval').
  - Model weights and thresholds are strictly fixed; no retraining.
  - No app or core library files are modified.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_curve, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
import joblib
from src.preprocessing.cleaning import trim_silence, normalize_amplitude  # noqa: E402
from src.preprocessing.features import extract_mfcc_stats  # noqa: E402
from src.streaming.quality_gate import (  # noqa: E402
    assess_window_quality, estimate_voiced_fraction, Window,
)
from src.decision_logic.risk_assessment import (  # noqa: E402
    LOW_RISK, NEEDS_VERIFICATION, HIGH_RISK, INSUFFICIENT_EVIDENCE,
)

_SEVERITY = {INSUFFICIENT_EVIDENCE: 0, LOW_RISK: 1, NEEDS_VERIFICATION: 2, HIGH_RISK: 3}

logger = logging.getLogger("diagnose_dev_aggregation")


def compute_conditional_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute EER on scored clips only via linear interpolation."""
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

    return float(eer), float(eer_thresh)


def process_audio_file(
    file_path: str,
    sr: int = 16000,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    min_window_sec: float = 1.0,
    min_speech_total_sec: float = 0.5,
) -> dict:
    """
    Extract sliding window features and non-overlapping segment speech durations
    for both P1 (raw unnormalized) and P2 (whole-clip trimmed and peak-normalized).
    """
    # Load audio cleanly via soundfile (exact 16kHz float32)
    raw_wav, native_sr = sf.read(file_path, dtype="float32")
    if raw_wav.ndim > 1:
        raw_wav = np.mean(raw_wav, axis=1)

    win_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    min_samples = int(min_window_sec * sr)

    out = {}

    # ------------------ Preprocessing P1: Raw Unnormalized ------------------
    # Sliced immediately into sliding windows
    p1_windows = []
    p1_segments = []
    p1_len = len(raw_wav)
    if p1_len >= min_samples:
        pos = 0
        seg_covered = 0
        while pos < p1_len:
            end = min(pos + win_samples, p1_len)
            w_arr = raw_wav[pos:end]
            if len(w_arr) < min_samples:
                break

            # Quality assessment
            w_obj = Window(waveform=w_arr, start_sample=pos, end_sample=end, sample_rate=sr,
                           is_full=(len(w_arr) == win_samples))
            qa = assess_window_quality(w_obj, min_duration_sec=min_window_sec, min_voiced_fraction=0.15, min_snr_db=6.0)

            # Unique non-overlapping segment
            seg_start = seg_covered
            seg_end = end
            if seg_end > seg_start:
                seg_slice = raw_wav[seg_start:seg_end]
                vf_seg = estimate_voiced_fraction(seg_slice, sr, silence_dbfs_threshold=-40.0)
                speech_dur_seg = len(seg_slice) / sr * vf_seg
                seg_covered = seg_end
            else:
                speech_dur_seg = 0.0

            p1_windows.append({
                "waveform": w_arr,
                "status": qa.status,
                "proceed": qa.proceed_to_inference,
                "start_sec": pos / sr,
                "end_sec": end / sr,
                "duration_sec": len(w_arr) / sr,
                "voiced_fraction": qa.voiced_fraction,
                "speech_dur_seg": speech_dur_seg,
            })
            if end >= p1_len:
                break
            pos += hop_samples

    out["P1"] = p1_windows

    # ---------------- Preprocessing P2: Whole-Clip Clean -------------------
    # Resample (already 16kHz) -> Trim silence -> Peak normalize to -3 dBFS
    trimmed_wav = trim_silence(raw_wav, top_db=30.0)
    norm_wav = normalize_amplitude(trimmed_wav, method="peak", target_dbfs=-3.0)

    p2_windows = []
    p2_len = len(norm_wav)
    if p2_len >= min_samples:
        pos = 0
        seg_covered = 0
        while pos < p2_len:
            end = min(pos + win_samples, p2_len)
            w_arr = norm_wav[pos:end]
            if len(w_arr) < min_samples:
                break

            w_obj = Window(waveform=w_arr, start_sample=pos, end_sample=end, sample_rate=sr,
                           is_full=(len(w_arr) == win_samples))
            qa = assess_window_quality(w_obj, min_duration_sec=min_window_sec, min_voiced_fraction=0.15, min_snr_db=6.0)

            seg_start = seg_covered
            seg_end = end
            if seg_end > seg_start:
                seg_slice = norm_wav[seg_start:seg_end]
                vf_seg = estimate_voiced_fraction(seg_slice, sr, silence_dbfs_threshold=-40.0)
                speech_dur_seg = len(seg_slice) / sr * vf_seg
                seg_covered = seg_end
            else:
                speech_dur_seg = 0.0

            p2_windows.append({
                "waveform": w_arr,
                "status": qa.status,
                "proceed": qa.proceed_to_inference,
                "start_sec": pos / sr,
                "end_sec": end / sr,
                "duration_sec": len(w_arr) / sr,
                "voiced_fraction": qa.voiced_fraction,
                "speech_dur_seg": speech_dur_seg,
            })
            if end >= p2_len:
                break
            pos += hop_samples

    out["P2"] = p2_windows
    return out


def _score_windows_for_clip(
    file_id: str,
    label: str,
    file_path: str,
    pipeline,
    operating_threshold: float,
) -> dict:
    """Worker function to process one audio file under P1 and P2 and score windows."""
    audio_data = process_audio_file(file_path)
    res = {"file_id": file_id, "label": label, "label_int": 1 if label == "spoof" else 0}

    for prep in ("P1", "P2"):
        windows = audio_data[prep]
        scored_scores = []
        seg_speech_durs = []
        statuses = []
        cum_speech = 0.0

        for w in windows:
            speech_dur_seg = w["speech_dur_seg"]
            seg_speech_durs.append(speech_dur_seg)
            cum_speech += speech_dur_seg

            if w["proceed"]:
                feats = extract_mfcc_stats(w["waveform"], sr=16000)
                prob_spoof = float(pipeline.predict_proba(feats[np.newaxis, :])[0, 1])
                scored_scores.append(prob_spoof)

                # Window status in app
                is_high = prob_spoof >= operating_threshold
                if cum_speech < 2.0 and not is_high:
                    st = INSUFFICIENT_EVIDENCE
                elif is_high:
                    st = HIGH_RISK
                elif prob_spoof < (operating_threshold * 0.5):
                    st = LOW_RISK
                else:
                    st = NEEDS_VERIFICATION

                if w["status"] == "high_noise_low_confidence" and st == LOW_RISK:
                    st = NEEDS_VERIFICATION
                statuses.append(st)
            else:
                statuses.append(INSUFFICIENT_EVIDENCE)

        total_speech = sum(seg_speech_durs)
        n_scored = len(scored_scores)

        # Clip aggregation values
        if n_scored == 0 or total_speech < 0.5:
            has_score = False
            a1b_max = np.nan
            a2_mean = np.nan
            a3_med = np.nan
            a4_p75 = np.nan
            a5_vote = np.nan
            a1a_status = INSUFFICIENT_EVIDENCE
        else:
            has_score = True
            arr = np.array(scored_scores, dtype=np.float32)
            a1b_max = float(np.max(arr))
            a2_mean = float(np.mean(arr))
            a3_med = float(np.median(arr))
            a4_p75 = float(np.percentile(arr, 75))

            # A5: Time voting (weight each window by its unique segment active speech duration)
            # Match each scored window to its corresponding segment duration
            scored_idx = 0
            risky_speech = 0.0
            for idx_w, w in enumerate(windows):
                if w["proceed"]:
                    score = scored_scores[scored_idx]
                    if score >= operating_threshold:
                        risky_speech += seg_speech_durs[idx_w]
                    scored_idx += 1
            a5_vote = float(risky_speech / total_speech) if total_speech > 0 else np.nan

            # A1a: Actual app rule = highest severity status among all windows
            max_sev = max(_SEVERITY[s] for s in statuses) if statuses else 0
            sev_to_status = {0: INSUFFICIENT_EVIDENCE, 1: LOW_RISK, 2: NEEDS_VERIFICATION, 3: HIGH_RISK}
            a1a_status = sev_to_status[max_sev]

        res[f"{prep}_has_score"] = has_score
        res[f"{prep}_total_speech_sec"] = total_speech
        res[f"{prep}_n_scored_windows"] = n_scored
        res[f"{prep}_a1a_status"] = a1a_status
        res[f"{prep}_a1b_max"] = a1b_max
        res[f"{prep}_a2_mean"] = a2_mean
        res[f"{prep}_a3_med"] = a3_med
        res[f"{prep}_a4_p75"] = a4_p75
        res[f"{prep}_a5_vote"] = a5_vote

    return res


def evaluate_dev_rules(
    df_cache: pd.DataFrame,
    operating_threshold: float,
    out_csv: Path,
) -> pd.DataFrame:
    """Compute and format the full comparison table across all 12 configurations."""
    total_bonafide = int((df_cache["label"] == "bonafide").sum())
    total_spoof = int((df_cache["label"] == "spoof").sum())
    total_clips = len(df_cache)

    records = []

    for prep in ("P1", "P2"):
        prep_label = "P1_live_raw" if prep == "P1" else "P2_whole_clip_clean"

        # ------------------- Discrete Rule A1a: Actual App Severity Logic -------------------
        for stance in ("alarm_centric", "strict_auth"):
            rule_name = f"A1a_app_rule_{stance}"

            if stance == "alarm_centric":
                # Fail-open / Alarm-centric: Positive = High risk only. Insufficient evidence = safe/unflagged.
                preds = (df_cache[f"{prep}_a1a_status"] == HIGH_RISK).astype(int)
            else:
                # Fail-closed / Strict-auth: Positive = Not Low risk. Insufficient evidence = flagged/not bona fide.
                preds = (df_cache[f"{prep}_a1a_status"] != LOW_RISK).astype(int)

            y_true = df_cache["label_int"].values
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            op_frr = (fp / total_bonafide * 100.0) if total_bonafide > 0 else 0.0
            op_far = (fn / total_spoof * 100.0) if total_spoof > 0 else 0.0

            n_insuf_bona = int(((df_cache["label"] == "bonafide") & (df_cache[f"{prep}_a1a_status"] == INSUFFICIENT_EVIDENCE)).sum())
            n_insuf_spoof = int(((df_cache["label"] == "spoof") & (df_cache[f"{prep}_a1a_status"] == INSUFFICIENT_EVIDENCE)).sum())

            records.append({
                "preprocessing": prep_label,
                "rule": rule_name,
                "n_total_clips": total_clips,
                "n_scored_bonafide": total_bonafide - n_insuf_bona,
                "n_insufficient_evidence_bonafide": n_insuf_bona,
                "n_scored_spoof": total_spoof - n_insuf_spoof,
                "n_insufficient_evidence_spoof": n_insuf_spoof,
                "conditional_eer_dev_percent": "N/A (discrete rule)",
                "conditional_eer_threshold": "N/A (discrete rule)",
                "default_threshold_value": f"App Policy ({operating_threshold:.4f})",
                "default_operational_frr_percent": round(op_frr, 3),
                "default_operational_far_percent": round(op_far, 3),
                "cm_default_tn_fp_fn_tp": f"{tn}/{fp}/{fn}/{tp}",
                "cond_far_le_5pct_threshold": "N/A",
                "cond_far_le_5pct_frr_percent": "N/A",
                "cond_far_le_10pct_threshold": "N/A",
                "cond_far_le_10pct_frr_percent": "N/A",
                "fail_closed_far_le_10pct_threshold": "N/A",
                "fail_closed_far_le_10pct_frr_percent": "N/A",
            })

        # ------------------- Continuous Rules: A1b, A2, A3, A4, A5 -------------------
        rule_map = {
            "A1b_worst_window_max": f"{prep}_a1b_max",
            "A2_mean": f"{prep}_a2_mean",
            "A3_median": f"{prep}_a3_med",
            "A4_p75": f"{prep}_a4_p75",
            "A5_time_voting": f"{prep}_a5_vote",
        }

        for rule_name, col in rule_map.items():
            valid_mask = df_cache[f"{prep}_has_score"] & df_cache[col].notna()
            scored_df = df_cache[valid_mask]

            n_scored_bona = int((scored_df["label"] == "bonafide").sum())
            n_scored_spoof = int((scored_df["label"] == "spoof").sum())
            n_insuf_bona = total_bonafide - n_scored_bona
            n_insuf_spoof = total_spoof - n_scored_spoof

            scores_scored = scored_df[col].values
            y_scored = scored_df["label_int"].values

            # Conditional EER on scored clips only
            if len(scores_scored) > 0 and len(np.unique(y_scored)) > 1:
                cond_eer, cond_eer_thresh = compute_conditional_eer(scores_scored, y_scored)
                cond_eer_pct = round(cond_eer * 100.0, 3)
                cond_eer_th = round(cond_eer_thresh, 6)
            else:
                cond_eer_pct = "N/A"
                cond_eer_th = "N/A"

            default_thresh = 0.50 if rule_name == "A5_time_voting" else operating_threshold

            # Operational results under Fail-Open:
            preds_fo = np.zeros(total_clips, dtype=int)
            preds_fo[scored_df.index] = (scores_scored >= default_thresh).astype(int)
            cm_fo = confusion_matrix(df_cache["label_int"].values, preds_fo, labels=[0, 1])
            tn_fo, fp_fo, fn_fo, tp_fo = cm_fo.ravel()
            op_frr_fo = (fp_fo / total_bonafide * 100.0) if total_bonafide > 0 else 0.0
            op_far_fo = (fn_fo / total_spoof * 100.0) if total_spoof > 0 else 0.0

            # Target security points: Sweep unique scores to find MAX threshold where FAR <= target
            ths = np.sort(np.unique(scores_scored))
            s_sp = scores_scored[y_scored == 1]
            s_bo = scores_scored[y_scored == 0]

            def find_max_th_for_far(target_far, s_spoof, n_spoof, s_bona, n_bona, extra_sp_miss=0, extra_bo_alarm=0):
                best_th, best_frr = None, None
                for th in ths:
                    far = ((np.sum(s_spoof < th) + extra_sp_miss) / n_spoof) * 100.0
                    if far <= target_far:
                        best_th = th
                        best_frr = ((np.sum(s_bona >= th) + extra_bo_alarm) / n_bona) * 100.0
                    else:
                        break
                return best_th, best_frr

            th_c5, frr_c5 = find_max_th_for_far(5.0, s_sp, n_scored_spoof, s_bo, n_scored_bona)
            th_c10, frr_c10 = find_max_th_for_far(10.0, s_sp, n_scored_spoof, s_bo, n_scored_bona)
            th_fc10, frr_fc10 = find_max_th_for_far(10.0, s_sp, total_spoof, s_bo, total_bonafide, extra_sp_miss=0, extra_bo_alarm=n_insuf_bona)

            records.append({
                "preprocessing": prep_label,
                "rule": rule_name,
                "n_total_clips": total_clips,
                "n_scored_bonafide": n_scored_bona,
                "n_insufficient_evidence_bonafide": n_insuf_bona,
                "n_scored_spoof": n_scored_spoof,
                "n_insufficient_evidence_spoof": n_insuf_spoof,
                "conditional_eer_dev_percent": cond_eer_pct,
                "conditional_eer_threshold": cond_eer_th,
                "default_threshold_value": f"{default_thresh:.4f}",
                "default_operational_frr_percent": round(op_frr_fo, 3),
                "default_operational_far_percent": round(op_far_fo, 3),
                "cm_default_tn_fp_fn_tp": f"{tn_fo}/{fp_fo}/{fn_fo}/{tp_fo}",
                "cond_far_le_5pct_threshold": round(float(th_c5), 6) if th_c5 is not None else "N/A",
                "cond_far_le_5pct_frr_percent": round(float(frr_c5), 3) if th_c5 is not None else "N/A",
                "cond_far_le_10pct_threshold": round(float(th_c10), 6) if th_c10 is not None else "N/A",
                "cond_far_le_10pct_frr_percent": round(float(frr_c10), 3) if th_c10 is not None else "N/A",
                "fail_closed_far_le_10pct_threshold": round(float(th_fc10), 6) if th_fc10 is not None else "N/A",
                "fail_closed_far_le_10pct_frr_percent": round(float(frr_fc10), 3) if th_fc10 is not None else "N/A",
            })


    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False)
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/asvspoof2019_la.csv",
                        help="Path to manifest CSV")
    parser.add_argument("--model-dir", default="models/svm_baseline_clean",
                        help="Path to trained SVM baseline")
    parser.add_argument("--out-dir", default="results/diagnostics",
                        help="Output directory for diagnostics")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit for rapid sanity check before full run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    manifest_path = ROOT_DIR / args.manifest
    model_dir = ROOT_DIR / args.model_dir
    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "dev_clip_scores_cache.csv"
    summary_path = out_dir / "dev_preprocessing_and_aggregation_comparison.csv"

    # STRICT GUARDRAIL: Verify and enforce DEVELOPMENT split only
    logger.info("Reading manifest from %s ...", manifest_path)
    df_manifest = pd.read_csv(manifest_path)
    df_dev = df_manifest[df_manifest["split"] == "dev"].copy()

    # Hard assert: zero eval rows
    assert (df_dev["split"] == "dev").all(), "Fatal error: Non-dev split rows detected!"
    assert not (df_dev["split"] == "eval").any(), "Fatal error: Eval split detected in dataset!"
    logger.info("Strict dev-split filter verified: %d dev rows (Bona fide: %d, Spoof: %d). ZERO eval rows.",
                len(df_dev), (df_dev["label"] == "bonafide").sum(), (df_dev["label"] == "spoof").sum())

    if args.limit:
        logger.warning("Running with --limit=%d for rapid test", args.limit)
        df_dev = df_dev.head(args.limit)

    # Load SVM model and threshold directly (no torch import)
    logger.info("Loading fixed SVM baseline from %s ...", model_dir)
    svm_pipeline = joblib.load(model_dir / "model.joblib")
    with open(model_dir / "threshold.json", "r", encoding="utf-8") as fh:
        threshold_meta = json.load(fh)
    operating_threshold = float(threshold_meta["threshold"])
    logger.info("Loaded SVM operating threshold: %.6f (Validation EER: %.4f)",
                operating_threshold, threshold_meta.get("validation_eer", float("nan")))

    # Process all dev audio clips
    logger.info("Processing %d dev clips using %d workers...", len(df_dev), args.workers)
    t0 = time.perf_counter()

    results = []
    # Use ProcessPoolExecutor for parallel processing
    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in df_dev.itertuples(index=False):
            f_path = str(row.file_path)
            # Extra guardrail against accidental eval path
            assert "eval" not in f_path.lower() or "dev" in f_path.lower(), f"Unexpected path: {f_path}"
            tasks.append(
                executor.submit(
                    _score_windows_for_clip,
                    row.file_id,
                    row.label,
                    f_path,
                    svm_pipeline,
                    operating_threshold,
                )
            )

        completed = 0
        total = len(tasks)
        for fut in as_completed(tasks):
            res = fut.result()
            results.append(res)
            completed += 1
            if completed % 1000 == 0 or completed == total:
                elapsed = time.perf_counter() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info("Progress: %d / %d clips (%.1f%%) [Elapsed: %.1fs, Rate: %.1f clips/s]",
                            completed, total, (completed / total) * 100.0, elapsed, rate)

    df_cache = pd.DataFrame(results)
    df_cache.to_csv(cache_path, index=False)
    logger.info("Saved per-clip window diagnostics cache to %s (%d rows)", cache_path, len(df_cache))

    # Evaluate rules and output comparison table
    logger.info("Evaluating all 12 combinations and computing conditional EER + operational error rates...")
    out_df = evaluate_dev_rules(df_cache, operating_threshold, summary_path)
    logger.info("Comparison summary table written to %s", summary_path)

    # Print summary table to console
    print("\n" + "=" * 120)
    print("DEVELOPMENT SET DIAGNOSTIC COMPARISON (P1 vs P2, A1a to A5)")
    print("=" * 120)
    print(out_df.to_string(index=False))
    print("=" * 120 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
