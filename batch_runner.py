"""
Batch runner: scores every "ok" row of a features_index.csv (as produced
by scripts/build_features.py for one condition) using Scorer's fast
feature-only path, and writes a results table.

This is the SOLE data source scripts/evaluate_model.py's metric
computation reads from — it never re-derives labels from anywhere else,
so a bug here would show up consistently across every reported metric
rather than silently in just one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.config import LABEL_TO_INT
from .scorer import Scorer


def run_batch(scorer: Scorer, features_index_csv: str | Path, splits: list[str] | None = None, max_workers: int = 16) -> pd.DataFrame:
    features_index_csv = Path(features_index_csv)
    index_df = pd.read_csv(features_index_csv)
    index_df = index_df[index_df["status"] == "ok"]
    if splits:
        index_df = index_df[index_df["split"].isin(splits)]

    if index_df.empty:
        raise ValueError(
            f"No usable rows in {features_index_csv} for splits={splits}. "
            f"Check that build_features.py succeeded for this condition/split."
        )

    base_dir = features_index_csv.resolve().parent.parent.parent.parent

    # Fast-path for consolidated 240-dim dynamic MFCC features if available
    consolidated_data = None
    if scorer.feature_key == "mfcc_deltas_stats":
        for s in (splits or ["eval", "dev", "train"]):
            cand = base_dir / "data" / "features" / "mfcc240" / f"mfcc240_{s}.npz"
            if cand.exists():
                z = np.load(cand)
                consolidated_data = dict(zip(z["file_ids"], z["X"]))
                break

    def _process_one(row):
        row_dict = row._asdict()
        fid = row_dict["file_id"]
        if consolidated_data is not None and fid in consolidated_data:
            feature_vector = consolidated_data[fid]
        else:
            fpath = Path(row_dict["feature_path"])
            if not fpath.exists():
                alt = base_dir / row_dict["feature_path"]
                if alt.exists():
                    fpath = alt
            with np.load(fpath) as npz:
                if scorer.feature_key not in npz:
                    raise KeyError(
                        f"Model expects feature_key={scorer.feature_key!r} but "
                        f"{row_dict['feature_path']} only has {list(npz.keys())}. "
                        f"Rebuild features for this condition with a matching --feature-set."
                    )
                feature_vector = npz[scorer.feature_key]

        result = scorer.score_features(feature_vector)
        return {
            "file_id": row_dict["file_id"],
            "split": row_dict["split"],
            "label": row_dict["label"],
            "label_int": LABEL_TO_INT[row_dict["label"]],
            "speaker_id": row_dict.get("speaker_id", ""),
            "condition": row_dict["condition"],
            "score": result["score"],
            "predicted_label": result["predicted_label"],
            "model_sec": result["timing_sec"]["model"],
        }

    from concurrent.futures import ThreadPoolExecutor
    rows_in = list(index_df.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(_process_one, rows_in))

    return pd.DataFrame(rows)


def save_batch_results(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
