#!/usr/bin/env python
"""
Edge Readiness and Computational Efficiency Benchmark.
Profiles model parameter count, memory footprint, inference latency, RTF, and quantization.
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
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.inference.scorer import Scorer
from src.inference.edge_optimizer import profile_edge_footprint, quantize_weights_fp16

logger = logging.getLogger("benchmark_edge")


def _resolve_model_file(model_dir: Path) -> Path:
    for name in ("model.onnx", "model.joblib", "svm_pipeline.joblib"):
        cand = model_dir / name
        if cand.exists():
            return cand
    onnx_candidates = list(model_dir.glob("*.onnx"))
    return onnx_candidates[0] if onnx_candidates else (model_dir / "model.joblib")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dirs", nargs="+", default=[
        "models/lcnn_clean",
        "models/mlp_mfcc240_clean",
        "models/svm_mfcc240_clean",
        "models/ensemble_mfcc240_clean",
    ])
    parser.add_argument("--quantize", action="store_true", help="Perform FP16 quantization on compatible models")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    reports = []
    print("\n" + "=" * 105)
    print("COMPUTATIONAL EFFICIENCY & EDGE READINESS BENCHMARK")
    print("=" * 105)

    for m_dir in args.model_dirs:
        m_path = ROOT_DIR / m_dir
        if not m_path.exists():
            logger.warning("Skipping missing model dir: %s", m_path)
            continue

        scorer = Scorer(m_path)
        model_file = _resolve_model_file(m_path)
        feature_shape = None
        sim_dur = 2.0
        if scorer.feature_key == "lfcc_spectrogram_2d":
            n_filters = int(scorer.train_config.get("n_filters", 60))
            target_frames = int(scorer.train_config.get("target_frames", 400))
            feature_shape = (1, 1, n_filters, target_frames)
            sim_dur = target_frames * 0.01  # 10 ms hop

        rep = profile_edge_footprint(
            scorer.model,
            model_file,
            feature_shape=feature_shape,
            simulated_audio_duration_sec=sim_dur,
        )
        reports.append({
            "model": m_path.name,
            "params": rep.total_parameter_count,
            "size_kb": rep.model_file_size_kb,
            "quantized_kb": rep.quantized_size_kb,
            "latency_p50_ms": rep.median_inference_latency_ms,
            "latency_p95_ms": rep.p95_inference_latency_ms,
            "throughput_fps": rep.throughput_samples_per_sec,
            "rtf": rep.real_time_factor,
            "edge_status": rep.edge_ready_status,
        })

    rep_df = pd.DataFrame(reports)
    print("\n" + rep_df.to_string(index=False))
    print("=" * 105 + "\n")

    out_csv = ROOT_DIR / "results/edge_benchmark_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(out_csv, index=False)
    logger.info("Saved edge benchmark report to %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
