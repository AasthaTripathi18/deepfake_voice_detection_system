#!/usr/bin/env python
"""
Evaluate a trained model on the EVAL (test) split, for one or more
conditions (clean and/or any degraded conditions you've already built
with scripts/build_features.py). Every condition is scored with the SAME
model and the SAME threshold (loaded from the model directory's
threshold.json, which was selected on the DEV split by train_model.py) —
this script never re-selects a threshold from eval data.

For each condition, reports:
  - EER, computed directly on that condition's eval scores/labels (EER is
    threshold-independent by construction, so this is a valid summary even
    though the fixed operating threshold below was chosen on a different
    (dev) split under different — for degraded conditions, DELIBERATELY
    mismatched — conditions).
  - FAR / FRR / precision / recall / F1 / accuracy / confusion matrix AT
    the fixed dev-selected threshold — this is what actually happens if
    you deploy the model as trained, at the threshold you already
    committed to before seeing this data.
  - Latency of the fast (features-already-extracted) scoring path, for
    reference; scripts/measure_latency.py separately measures full
    audio-in-to-score-out latency, which is the more meaningful number
    for anything demo/deployment-facing.

Conditions are reported SEPARATELY, never averaged together, so a reader
always sees matched vs. each mismatched condition side by side.

Example (clean + two degraded conditions already built):
    python scripts/evaluate_model.py \
        --model-dir models/svm_baseline_clean \
        --condition clean=data/features/clean/features_index.csv \
        --condition telephony_bandpass=data/features/telephony_bandpass/features_index.csv \
        --condition codec_gsm=data/features/codec_gsm/features_index.csv \
        --eval-split eval \
        --out-dir results/metrics/svm_baseline_clean
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from src.inference.scorer import Scorer  # noqa: E402
from src.inference.batch_runner import run_batch, save_batch_results  # noqa: E402
from src.evaluation.report import (  # noqa: E402
    compute_eer, far_frr_at_threshold, classification_metrics_at_threshold, latency_summary, compute_min_tdcf,
)


def parse_condition_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--condition must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--condition", action="append", required=True, type=parse_condition_arg,
                         metavar="NAME=PATH", help="Repeatable: a condition name and its features_index.csv path")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional override for the operating threshold")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("evaluate_model")

    scorer = Scorer(args.model_dir)
    operating_threshold = args.threshold if args.threshold is not None else scorer.threshold
    log.info("Loaded model_type=%s feature_key=%s operating_threshold=%.6f",
              scorer.train_config["model_type"], scorer.feature_key, operating_threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for condition_name, index_path in args.condition:
        log.info("Scoring condition=%s from %s ...", condition_name, index_path)
        results_df = run_batch(scorer, index_path, splits=[args.eval_split])
        save_batch_results(results_df, out_dir / f"scores_{condition_name}.csv")

        scores = results_df["score"].to_numpy()
        labels = results_df["label_int"].to_numpy()

        eer, eer_threshold = compute_eer(scores, labels)
        min_tdcf, tdcf_threshold = compute_min_tdcf(scores, labels)
        far_frr = far_frr_at_threshold(scores, labels, operating_threshold)
        cls_metrics = classification_metrics_at_threshold(scores, labels, operating_threshold)
        lat = latency_summary(results_df["model_sec"].to_numpy())

        row = {
            "condition": condition_name,
            "n_eval_samples": len(results_df),
            "eer_percent": round(eer * 100, 3),
            "eer_threshold": round(eer_threshold, 6),
            "min_tdcf": round(min_tdcf, 4),
            "operating_threshold": round(operating_threshold, 6),
            "far_percent": round(far_frr["far"] * 100, 3),
            "frr_percent": round(far_frr["frr"] * 100, 3),
            "precision_spoof": round(cls_metrics["precision_spoof"], 4),
            "recall_spoof": round(cls_metrics["recall_spoof"], 4),
            "f1_spoof": round(cls_metrics["f1_spoof"], 4),
            "accuracy": round(cls_metrics["accuracy"], 4),
            **{f"cm_{k}": v for k, v in cls_metrics["confusion_matrix"].items()},
            "feature_only_latency_median_ms": round(lat["median_ms"], 4),
            "feature_only_latency_p95_ms": round(lat["p95_ms"], 4),
        }
        summary_rows.append(row)

        log.info(
            "condition=%s: EER=%.2f%% FAR=%.2f%% FRR=%.2f%% F1=%.4f (n=%d) "
            "-- MEASURED JUST NOW, not assumed from any prior source.",
            condition_name, row["eer_percent"], row["far_percent"], row["frr_percent"],
            row["f1_spoof"], row["n_eval_samples"],
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_by_condition.csv"
    summary_df.to_csv(summary_path, index=False)

    log.info("Per-condition summary written to %s", summary_path)
    log.info(
        "This table intentionally keeps every condition on its own row. "
        "Compare it against results/base_paper_reference.yaml manually or via "
        "src/evaluation/report.py — do not blend rows into one average, and do "
        "not treat any base-paper number as this project's own result."
    )
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
