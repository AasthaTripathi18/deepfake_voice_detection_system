#!/usr/bin/env python
"""
Train a spoof-detection model and select its operating threshold — using
ONLY the official train split to fit, and ONLY the official dev split to
select the threshold (and, for a model that supports it, for early
stopping). The eval/test split is never touched by this script.

Examples:
    # Primary baseline (CPU-only, minutes not hours):
    python scripts/train_model.py \
        --features-index data/features/clean/features_index.csv \
        --model-type svm_baseline \
        --out-dir models/svm_baseline_clean

    # Comparison model (requires `pip install torch`; see src/modeling/lcnn.py):
    python scripts/train_model.py \
        --features-index data/features/clean/features_index.csv \
        --model-type lcnn \
        --hyperparams-json '{"epochs": 15, "batch_size": 32}' \
        --out-dir models/lcnn_clean
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modeling import get_model_class, default_feature_key  # noqa: E402
from src.modeling.config import TrainConfig, build_training_metadata, save_json  # noqa: E402
from src.modeling.data_io import load_feature_matrix  # noqa: E402
from src.evaluation.report import select_threshold_from_validation  # noqa: E402


def class_balance(y) -> dict:
    import numpy as np
    values, counts = np.unique(y, return_counts=True)
    names = {0: "bonafide", 1: "spoof"}
    return {names[int(v)]: int(c) for v, c in zip(values, counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-index", required=True,
                         help="features_index.csv for the CLEAN (matched) condition")
    parser.add_argument("--model-type", default="svm_baseline", choices=["svm_baseline", "lcnn"])
    parser.add_argument("--feature-key", default=None,
                         help="Override the model's default feature key (mfcc_stats for svm_baseline, "
                              "logmel for lcnn). Usually leave unset.")
    parser.add_argument("--train-split", default="train", help="Official split used to FIT the model")
    parser.add_argument("--val-split", default="dev",
                         help="Official split used ONLY for threshold selection / early stopping")
    parser.add_argument("--hyperparams-json", default="{}", help="JSON dict of extra model constructor kwargs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("train_model")

    feature_key = args.feature_key or default_feature_key(args.model_type)
    hyperparams = json.loads(args.hyperparams_json)
    hyperparams.setdefault("seed", args.seed)

    log.info("Loading train split '%s' (feature_key=%s)...", args.train_split, feature_key)
    X_train, y_train, _, _ = load_feature_matrix(args.features_index, feature_key, splits=[args.train_split])
    log.info("Loading val split '%s' (feature_key=%s)...", args.val_split, feature_key)
    X_val, y_val, _, _ = load_feature_matrix(args.features_index, feature_key, splits=[args.val_split])

    log.info("train: n=%d %s", len(y_train), class_balance(y_train))
    log.info("val:   n=%d %s", len(y_val), class_balance(y_val))

    model_cls = get_model_class(args.model_type)
    model = model_cls(**hyperparams)

    log.info("Fitting %s ...", args.model_type)
    try:
        # LCNNBaseline.fit accepts optional X_val/y_val for early stopping;
        # SVMBaseline.fit does not accept them at all. Try the richer
        # signature first, fall back to the plain one.
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    except TypeError:
        model.fit(X_train, y_train)
    log.info("Fitting complete.")

    log.info("Scoring validation split to select threshold (validation data only, never eval/test)...")
    val_scores = model.predict_proba(X_val)
    threshold_info = select_threshold_from_validation(val_scores, y_val, method="eer")
    log.info("Selected threshold=%.6f (validation EER=%.4f)",
              threshold_info["threshold"], threshold_info["validation_eer"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir)

    config = TrainConfig(
        model_type=args.model_type,
        feature_key=feature_key,
        train_condition="clean",
        train_splits=[args.train_split],
        val_splits=[args.val_split],
        seed=args.seed,
        hyperparameters=hyperparams,
    )
    save_json(config.to_dict(), out_dir / "config.json")
    save_json({"bonafide": 0, "spoof": 1}, out_dir / "label_map.json")
    save_json(threshold_info, out_dir / "threshold.json")
    save_json(
        build_training_metadata(len(y_train), len(y_val), class_balance(y_train), class_balance(y_val)),
        out_dir / "training_metadata.json",
    )

    log.info("Saved model + config + label_map + threshold + training_metadata to %s", out_dir)
    log.info(
        "NOTE: no test/eval-set number has been computed by this script. "
        "Run scripts/evaluate_model.py separately to get reportable metrics — "
        "nothing here should be quoted as a result yet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
