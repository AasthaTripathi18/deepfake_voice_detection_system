#!/usr/bin/env python
"""
Measure REAL, end-to-end, single-sample latency (audio file in -> risk
score out) on THIS machine, using Scorer.score_audio_file so the timing
includes audio loading, cleaning, feature extraction, AND model inference
— not just the model forward pass.

Methodology (documented here so the numbers this produces are
interpretable, not just a bare figure):
  - Single-sample calls, not batched — this matches how the eventual
    live/challenge-response demo will actually call the scorer (one clip
    at a time), which is a different (usually slower per-sample) regime
    than batch throughput.
  - A configurable number of WARM-UP calls are run first and discarded,
    to avoid counting one-time costs (e.g. first-call JIT/cache effects,
    lazy imports inside librosa) as if they were steady-state latency.
  - After warm-up, N repeated calls on the same (or a rotating small set
    of) audio file(s) are timed; mean, median, p95, p99, and max are
    reported per pipeline stage, not just an overall mean — a single mean
    (as in the base paper's unqualified "~85 ms") hides worst-case
    behavior that matters for an interactive flow.
  - Hardware/software context (CPU model where obtainable, core count,
    whether a GPU/CUDA device was used, key library versions) is recorded
    alongside the numbers, because latency is meaningless without knowing
    what it was measured on.

This script measures whatever machine it is run on — it does NOT reuse or
assume the base paper's ~85 ms figure, and its output should be labeled
with the actual hardware it ran on when reported.

Example:
    python scripts/measure_latency.py \
        --model-dir models/svm_baseline_clean \
        --audio-files data/manifests/sample1.flac data/manifests/sample2.flac \
        --n-runs 200 --n-warmup 20 \
        --out results/metrics/svm_baseline_clean/latency.json
"""

from __future__ import annotations

try:
    # Pre-load onnxruntime before sklearn/scipy to prevent Windows C++ runtime DLL crash
    import onnxruntime  # noqa: F401
except (ImportError, OSError):
    pass

import argparse
import json
import logging
import multiprocessing
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.inference.scorer import Scorer  # noqa: E402
from src.evaluation.report import latency_summary  # noqa: E402


def hardware_context() -> dict:
    ctx = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown (see 'platform' field)",
        "cpu_count_logical": multiprocessing.cpu_count(),
        "python_version": platform.python_version(),
        "cuda_used": False,
    }
    try:
        import torch
        ctx["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            ctx["cuda_used"] = True
            ctx["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        ctx["cuda_available"] = False
    return ctx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio-files", nargs="+", required=True,
                         help="One or more audio files to rotate through for timing (real files, "
                              "not synthetic — timing should reflect actual audio I/O cost).")
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--degradation-condition", default=None,
                         help="Optionally measure latency under a specific degradation "
                              "condition (e.g. codec_gsm) instead of clean audio.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("measure_latency")

    scorer = Scorer(args.model_dir)
    files = args.audio_files

    log.info("Warm-up: %d call(s) (discarded)...", args.n_warmup)
    for i in range(args.n_warmup):
        scorer.score_audio_file(files[i % len(files)], degradation_condition=args.degradation_condition)

    log.info("Timed run: %d call(s)...", args.n_runs)
    stage_timings: dict[str, list] = {"load": [], "preprocessing": [], "feature_extraction": [], "model": [], "total": []}
    for i in range(args.n_runs):
        result = scorer.score_audio_file(files[i % len(files)], degradation_condition=args.degradation_condition)
        for stage, value in result["timing_sec"].items():
            stage_timings[stage].append(value)

    report = {
        "hardware_context": hardware_context(),
        "methodology": {
            "n_runs": args.n_runs,
            "n_warmup_discarded": args.n_warmup,
            "n_audio_files_rotated": len(files),
            "degradation_condition": args.degradation_condition or "clean",
            "call_pattern": "single-sample sequential calls via Scorer.score_audio_file "
                             "(not batched; matches the intended live-demo call pattern)",
        },
        "per_stage_ms": {
            stage: latency_summary(np.array(values)) for stage, values in stage_timings.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    total = report["per_stage_ms"]["total"]
    log.info(
        "Total latency (this machine, n=%d, %s): median=%.1fms p95=%.1fms p99=%.1fms max=%.1fms",
        args.n_runs, report["methodology"]["degradation_condition"],
        total["median_ms"], total["p95_ms"], total["p99_ms"], total["max_ms"],
    )
    log.info("Full report (including per-stage breakdown and hardware context) written to %s", out_path)
    log.info(
        "This number was measured just now on THIS machine (see hardware_context in the report) — "
        "it is NOT the base paper's ~85ms figure and should not be presented as such."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
