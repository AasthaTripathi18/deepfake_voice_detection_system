#!/usr/bin/env python
"""
Audit LCNN ONNX model directly against the ASVspoof 2019 LA zip archive without
requiring 25 GB extraction to disk.

Supports:
  1. --probe-preprocessing: quickly compares raw vs clean(trim+normalize) on a dev subset
     to verify which preprocessing matches the saved model's threshold.json.
  2. --full-audit: scores full dev and eval splits, checks raw EER/min-tDCF/CM/per-attack
     breakdown, fits monotonic temperature/bias calibration strictly on dev, and verifies
     zero metric degradation on eval.
  3. --robustness: evaluates clean, additive noise (20/10/5 dB), and telephony bandpass
     (300-3400 Hz) directly on audio streamed from the zip archive.
"""

from __future__ import annotations

try:
    import onnxruntime  # noqa: F401
except (ImportError, OSError):
    pass

import argparse
import io
import json
import logging
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.report import (  # noqa: E402
    classification_metrics_at_threshold,
    compute_eer,
    compute_min_tdcf,
    far_frr_at_threshold,
    latency_summary,
)
from src.modeling.config import INT_TO_LABEL, LABEL_TO_INT  # noqa: E402
from src.modeling.lcnn_onnx import LCNNOnnx  # noqa: E402
from src.preprocessing import cleaning, degradations, features as feat_mod  # noqa: E402

logger = logging.getLogger("audit_lcnn_archive")

_WORKER_ZIP: zipfile.ZipFile | None = None
_WORKER_ZIP_PATH: str = ""
_WORKER_PREPROC: str = "clean"
_WORKER_SNR_DB: float | None = None
_WORKER_BANDPASS: bool = False


def _init_worker(zip_path: str, preproc: str, snr_db: float | None, apply_bandpass: bool):
    global _WORKER_ZIP, _WORKER_ZIP_PATH, _WORKER_PREPROC, _WORKER_SNR_DB, _WORKER_BANDPASS
    _WORKER_ZIP_PATH = zip_path
    _WORKER_ZIP = zipfile.ZipFile(zip_path, "r")
    _WORKER_PREPROC = preproc
    _WORKER_SNR_DB = snr_db
    _WORKER_BANDPASS = apply_bandpass


def _extract_worker(item: tuple[str, str]) -> np.ndarray:
    zip_member, file_id = item
    assert _WORKER_ZIP is not None
    raw_bytes = _WORKER_ZIP.read(zip_member)
    wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000

    if _WORKER_PREPROC == "clean":
        wav = cleaning.clean(wav, sr, trim=True, normalize=True, denoise=False)

    if _WORKER_SNR_DB is not None:
        import zlib
        wav = degradations.add_noise_at_snr(
            wav, snr_db=_WORKER_SNR_DB, rng=np.random.default_rng(zlib.crc32(file_id.encode("utf-8"))),
        )
    if _WORKER_BANDPASS:
        wav = degradations.bandpass_telephony(wav, sr=sr)

    return feat_mod.extract_lfcc_spectrogram_2d(wav, sr=sr)


def _zip_member_map(zip_path: str) -> dict[str, str]:
    """Map file_id -> zip member path for LA flac files."""
    mapping: dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".flac") and "ASVspoof2019_LA_" in name:
                fid = Path(name).stem
                mapping[fid] = name
    return mapping


def _score_split(
    model: LCNNOnnx,
    df_split: pd.DataFrame,
    zip_path: str,
    member_map: dict[str, str],
    preproc: str,
    workers: int = 6,
    batch_size: int = 128,
    snr_db: float | None = None,
    apply_bandpass: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features in worker processes and batch-score raw logits via ONNX."""
    items = [(member_map[fid], fid) for fid in df_split["file_id"]]
    n_total = len(items)
    logits_all = np.empty((n_total, 2), dtype=np.float32)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(zip_path, preproc, snr_db, apply_bandpass),
    ) as pool:
        batch_feats: list[np.ndarray] = []
        write_idx = 0
        for i, feat in enumerate(pool.map(_extract_worker, items, chunksize=64)):
            batch_feats.append(feat)
            if len(batch_feats) == batch_size or i == n_total - 1:
                batch_arr = np.stack(batch_feats, axis=0)
                out = model.predict_logits(batch_arr)
                logits_all[write_idx:write_idx + len(batch_feats)] = out
                write_idx += len(batch_feats)
                batch_feats.clear()
                if write_idx % 5000 < batch_size or write_idx == n_total:
                    elapsed = time.perf_counter() - t0
                    rate = write_idx / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "  Scored %d / %d (%.1f%%) [%.1f clips/s]",
                        write_idx, n_total, 100.0 * write_idx / n_total, rate,
                    )

    labels = np.array([LABEL_TO_INT[l] for l in df_split["label"]], dtype=np.int64)
    return logits_all, labels


def _softmax_spoof(logits: np.ndarray, temperature: float = 1.0, logit_bias: float = 0.0) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    if logit_bias != 0.0:
        scaled = scaled.copy()
        scaled[:, 1] += logit_bias
    exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    return probs[:, 1]


def _fit_dev_calibration(dev_logits: np.ndarray, dev_labels: np.ndarray) -> tuple[float, float]:
    """
    Fit strictly monotonic affine calibration (temperature T > 0, bias b) on DEV logits:
      p = sigmoid((logit_1 - logit_0 + b_raw) / T)
    where b_raw centers the dev EER operating point at 0.5, and T minimizes
    binary cross-entropy (NLL) on balanced dev logits so probabilities are smooth
    and well-spread without ever changing ROC/EER/min-tDCF.
    """
    llr = (dev_logits[:, 1] - dev_logits[:, 0]).astype(np.float64)
    _, llr_eer_th = compute_eer(llr, dev_labels)
    b_raw = -float(llr_eer_th)
    centered = llr + b_raw

    # Equal class weights during NLL fit so 1:9 imbalance doesn't pull scores back to edges
    pos_w = float(np.sum(dev_labels == 0)) / max(1.0, float(np.sum(dev_labels == 1)))
    sample_weights = np.where(dev_labels == 1, pos_w, 1.0)

    def _nll(log_t: np.ndarray) -> float:
        t = float(np.exp(log_t[0]))
        z = np.clip(centered / t, -60.0, 60.0)
        loss = np.logaddexp(0.0, z) - dev_labels * z
        return float(np.average(loss, weights=sample_weights))

    res = minimize(_nll, x0=np.array([np.log(5.0)]), bounds=[(-2.0, 5.0)], method="L-BFGS-B")
    temp = float(np.exp(res.x[0]))
    logit_bias = b_raw / temp
    return temp, logit_bias


def _summarize_metrics(scores: np.ndarray, labels: np.ndarray, op_threshold: float, condition: str) -> dict:
    eer, eer_th = compute_eer(scores, labels)
    min_tdcf, tdcf_th = compute_min_tdcf(scores, labels)
    far_frr = far_frr_at_threshold(scores, labels, op_threshold)
    cls = classification_metrics_at_threshold(scores, labels, op_threshold)
    return {
        "condition": condition,
        "n_eval_samples": int(len(labels)),
        "eer_percent": round(eer * 100.0, 3),
        "eer_threshold": round(eer_th, 6),
        "min_tdcf": round(min_tdcf, 4),
        "operating_threshold": round(op_threshold, 6),
        "far_percent": round(far_frr["far"] * 100.0, 3),
        "frr_percent": round(far_frr["frr"] * 100.0, 3),
        "precision_spoof": round(cls["precision_spoof"], 4),
        "recall_spoof": round(cls["recall_spoof"], 4),
        "f1_spoof": round(cls["f1_spoof"], 4),
        "accuracy": round(cls["accuracy"], 4),
        **{f"cm_{k}": v for k, v in cls["confusion_matrix"].items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--model-dir", default="models/lcnn_clean")
    parser.add_argument("--manifest", default="data/manifests/asvspoof2019_la.csv")
    parser.add_argument("--out-dir", default="results/metrics/lcnn_clean_audit")
    parser.add_argument("--preproc", default="auto", choices=["auto", "raw", "clean"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--probe-preprocessing", action="store_true")
    parser.add_argument("--probe-samples", type=int, default=1000)
    parser.add_argument("--full-audit", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--robustness-samples", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    model_dir = ROOT_DIR / args.model_dir
    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "threshold.json", "r", encoding="utf-8") as fh:
        thresh_meta = json.load(fh)
    op_thresh = float(thresh_meta["threshold"])

    model = LCNNOnnx.load(model_dir)
    manifest_df = pd.read_csv(ROOT_DIR / args.manifest)
    manifest_df = manifest_df[manifest_df["status"] == "valid"]
    member_map = _zip_member_map(args.zip_path)
    logger.info("Indexed %d LA .flac entries inside zip archive.", len(member_map))

    dev_df = manifest_df[manifest_df["split"] == "dev"].copy()
    eval_df = manifest_df[manifest_df["split"] == "eval"].copy()

    preproc = args.preproc
    if args.probe_preprocessing or preproc == "auto":
        bf_sub = dev_df[dev_df["label"] == "bonafide"].head(args.probe_samples // 2)
        sp_sub = dev_df[dev_df["label"] == "spoof"].head(args.probe_samples // 2)
        probe_df = pd.concat([bf_sub, sp_sub])
        probe_res = {}
        for p_mode in ("raw", "clean"):
            logger.info("Probing preprocessing=%s on %d dev samples...", p_mode, len(probe_df))
            logits, y = _score_split(model, probe_df, args.zip_path, member_map, p_mode, workers=args.workers)
            probs = _softmax_spoof(logits)
            llrs = (logits[:, 1] - logits[:, 0]).astype(np.float64)
            eer_prob, th_prob = compute_eer(probs, y)
            eer_llr, th_llr = compute_eer(llrs, y)
            far_frr = far_frr_at_threshold(probs, y, op_thresh)
            probe_res[p_mode] = eer_llr
            print(
                f"PROBE {p_mode:5s} | EER(prob)={eer_prob*100:.3f}% (th={th_prob:.6f}) | "
                f"EER(LLR)={eer_llr*100:.3f}% (th={th_llr:.4f}) | "
                f"at {op_thresh:.3f}: FAR={far_frr['far']*100:.2f}% FRR={far_frr['frr']*100:.2f}%"
            )
        if preproc == "auto":
            preproc = min(probe_res, key=probe_res.get)
            logger.info("Auto-selected preprocessing=%s (lowest probe dev EER)", preproc)

    if args.full_audit:
        logger.info("Running FULL DEV audit (%d clips, preproc=%s)...", len(dev_df), preproc)
        dev_logits, dev_y = _score_split(model, dev_df, args.zip_path, member_map, preproc, workers=args.workers)
        dev_probs_raw = _softmax_spoof(dev_logits)
        dev_llrs = (dev_logits[:, 1] - dev_logits[:, 0]).astype(np.float64)

        dev_raw_summary = _summarize_metrics(dev_probs_raw, dev_y, op_thresh, f"dev_raw_prob_{preproc}")
        dev_llr_eer, dev_llr_th = compute_eer(dev_llrs, dev_y)
        dev_llr_tdcf, _ = compute_min_tdcf(dev_llrs, dev_y)
        print("\n--- DEV AUDIT (RAW) ---")
        print(dev_raw_summary)
        print(f"DEV LLR EER: {dev_llr_eer*100:.3f}% (th={dev_llr_th:.4f}) | min_tdcf={dev_llr_tdcf:.4f}")

        # Fit strictly monotonic dev calibration
        temp, logit_bias = _fit_dev_calibration(dev_logits, dev_y)
        dev_probs_cal = _softmax_spoof(dev_logits, temperature=temp, logit_bias=logit_bias)
        raw_op_llr = float(np.log(op_thresh / (1.0 - op_thresh)))
        cal_op_thresh = float(1.0 / (1.0 + np.exp(-((raw_op_llr / temp) + logit_bias))))
        cal_eer_thresh = 0.5
        print(
            f"Fitted DEV calibration: temperature={temp:.6f}, logit_bias={logit_bias:.6f}, "
            f"mapped_op_thresh={cal_op_thresh:.6f} (preserves exact decisions of {op_thresh})"
        )

        logger.info("Running FULL EVAL audit (%d clips, preproc=%s)...", len(eval_df), preproc)
        ev_logits, ev_y = _score_split(model, eval_df, args.zip_path, member_map, preproc, workers=args.workers)
        ev_probs_raw = _softmax_spoof(ev_logits).astype(np.float32)
        ev_llrs = (ev_logits[:, 1] - ev_logits[:, 0]).astype(np.float64)
        ev_probs_cal = _softmax_spoof(ev_logits, temperature=temp, logit_bias=logit_bias).astype(np.float32)

        ev_raw_summary = _summarize_metrics(ev_probs_raw, ev_y, op_thresh, f"eval_raw_prob_f32_{preproc}")
        ev_llr_summary = _summarize_metrics(ev_llrs, ev_y, raw_op_llr, f"eval_llr_{preproc}")
        ev_cal_op_summary = _summarize_metrics(ev_probs_cal, ev_y, cal_op_thresh, f"eval_cal_mapped_op_f32_{preproc}")
        ev_cal_eer_summary = _summarize_metrics(ev_probs_cal, ev_y, cal_eer_thresh, f"eval_cal_dev_eer0.5_f32_{preproc}")

        # Measure real single-sample latency on 200 eval clips
        sample_files = [(member_map[fid], fid) for fid in eval_df["file_id"].head(200)]
        with zipfile.ZipFile(args.zip_path, "r") as zf:
            feat_latencies = []
            # Warmup
            for mem, _ in sample_files[:20]:
                wav, sr = sf.read(io.BytesIO(zf.read(mem)), dtype="float32")
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                if preproc == "clean":
                    wav = cleaning.clean(wav, sr, trim=True, normalize=True, denoise=False)
                spec = feat_mod.extract_lfcc_spectrogram_2d(wav, sr=sr)
                _ = model.predict_proba(spec)
            for mem, _ in sample_files:
                wav, sr = sf.read(io.BytesIO(zf.read(mem)), dtype="float32")
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                if preproc == "clean":
                    wav = cleaning.clean(wav, sr, trim=True, normalize=True, denoise=False)
                spec = feat_mod.extract_lfcc_spectrogram_2d(wav, sr=sr)
                t0 = time.perf_counter()
                _ = model.predict_proba(spec)
                feat_latencies.append(time.perf_counter() - t0)
        lat_stats = latency_summary(np.array(feat_latencies))

        for row in (ev_raw_summary, ev_llr_summary, ev_cal_op_summary, ev_cal_eer_summary):
            row["feature_only_latency_median_ms"] = round(lat_stats["median_ms"], 4)
            row["feature_only_latency_p95_ms"] = round(lat_stats["p95_ms"], 4)

        audit_df = pd.DataFrame([dev_raw_summary, ev_raw_summary, ev_llr_summary, ev_cal_op_summary, ev_cal_eer_summary])
        audit_df.to_csv(out_dir / "audit_summary.csv", index=False)
        print("\n--- FULL AUDIT SUMMARY ---")
        print(audit_df.to_string(index=False))

        # Per-attack breakdown on eval (A07-A19 vs bonafide)
        eval_out = eval_df[["file_id", "split", "label", "speaker_id", "system_id"]].copy()
        eval_out["label_int"] = ev_y
        eval_out["raw_logit_0"] = ev_logits[:, 0]
        eval_out["raw_logit_1"] = ev_logits[:, 1]
        eval_out["llr"] = ev_llrs
        eval_out["raw_prob"] = ev_probs_raw
        eval_out["cal_prob"] = ev_probs_cal
        eval_out["pred_raw_op"] = [INT_TO_LABEL[int(p)] for p in (ev_probs_raw >= op_thresh)]
        eval_out["pred_cal_op"] = [INT_TO_LABEL[int(p)] for p in (ev_probs_cal >= cal_op_thresh)]
        eval_out.to_csv(out_dir / "eval_scores_detailed.csv", index=False)

        bona_scores = ev_llrs[ev_y == 0]
        attack_rows = []
        for sys_id, grp in eval_out[eval_out["label"] == "spoof"].groupby("system_id"):
            s_arr = np.concatenate([bona_scores, grp["llr"].to_numpy()])
            l_arr = np.concatenate([np.zeros(len(bona_scores), dtype=int), np.ones(len(grp), dtype=int)])
            a_eer, _ = compute_eer(s_arr, l_arr)
            a_tdcf, _ = compute_min_tdcf(s_arr, l_arr)
            a_miss_op = float(np.mean(grp["raw_prob"] < op_thresh)) * 100.0
            attack_rows.append({
                "system_id": sys_id,
                "n_spoof": len(grp),
                "eer_percent": round(a_eer * 100.0, 3),
                "min_tdcf": round(a_tdcf, 4),
                "miss_rate_at_op_percent": round(a_miss_op, 2),
            })
        atk_df = pd.DataFrame(attack_rows).sort_values("eer_percent", ascending=False)
        atk_df.to_csv(out_dir / "eval_per_attack.csv", index=False)
        print("\n--- PER-ATTACK EVAL BREAKDOWN (worst EER first) ---")
        print(atk_df.to_string(index=False))

        cal_info = {
            "preproc": preproc,
            "temperature": temp,
            "logit_bias": logit_bias,
            "raw_operating_threshold": op_thresh,
            "mapped_operating_threshold": cal_op_thresh,
            "calibrated_dev_eer_threshold": 0.5,
            "dev_eer_percent": round(dev_llr_eer * 100.0, 3),
            "eval_eer_percent": ev_llr_summary["eer_percent"],
            "latency_median_ms": round(lat_stats["median_ms"], 4),
            "latency_p95_ms": round(lat_stats["p95_ms"], 4),
        }
        with open(out_dir / "calibration_audit.json", "w", encoding="utf-8") as fh:
            json.dump(cal_info, fh, indent=2)

    if args.robustness:
        bf_sub = eval_df[eval_df["label"] == "bonafide"].head(args.robustness_samples // 2)
        sp_sub = eval_df[eval_df["label"] == "spoof"].head(args.robustness_samples // 2)
        rob_df = pd.concat([bf_sub, sp_sub]).sample(frac=1.0, random_state=42)
        conditions = [
            {"name": "clean", "snr_db": None, "apply_bandpass": False},
            {"name": "noise_snr_20db", "snr_db": 20.0, "apply_bandpass": False},
            {"name": "noise_snr_10db", "snr_db": 10.0, "apply_bandpass": False},
            {"name": "noise_snr_5db", "snr_db": 5.0, "apply_bandpass": False},
            {"name": "telephony_bandpass", "snr_db": None, "apply_bandpass": True},
        ]
        rob_rows = []
        for cond in conditions:
            logger.info("Robustness condition=%s (n=%d, preproc=%s)...", cond["name"], len(rob_df), preproc)
            logits, y = _score_split(
                model, rob_df, args.zip_path, member_map, preproc, workers=args.workers,
                snr_db=cond["snr_db"], apply_bandpass=cond["apply_bandpass"],
            )
            llrs = (logits[:, 1] - logits[:, 0]).astype(np.float64)
            raw_op_llr = float(np.log(op_thresh / (1.0 - op_thresh)))
            row = _summarize_metrics(llrs, y, raw_op_llr, cond["name"])
            rob_rows.append(row)
        rob_out = pd.DataFrame(rob_rows)
        rob_out.to_csv(out_dir / "robustness_eval.csv", index=False)
        print("\n--- ROBUSTNESS ON EVAL SUBSET ---")
        print(rob_out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
