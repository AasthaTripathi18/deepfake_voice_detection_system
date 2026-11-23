"""
Bridges src/preprocessing's saved .npz feature files (built by
scripts/build_features.py) into the (X, y, file_ids) matrices the modeling
and evaluation code needs. This is the ONLY place that reads a
features_index.csv, so training and evaluation always load data the same
way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABEL_TO_INT


def load_feature_matrix(
    features_index_csv: str | Path,
    feature_key: str = "mfcc_stats",
    splits: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Returns (X, y, file_ids, speaker_ids).

    X : float32 array, shape (n_samples, feature_dim) for "mfcc_stats",
        or (n_samples, n_mels, n_frames) for "logmel" (kept un-flattened
        so a CNN-style model can consume it directly).
    y : int array of 0 (bonafide) / 1 (spoof), per LABEL_TO_INT.

    Only rows with status == "ok" in the index are loaded; rows the
    feature-building step already marked "failed" are skipped here too
    (their absence should already have been logged when features were
    built — this function does not re-log them, to avoid duplicate noise,
    but does report a final count so a silent all-failed index is caught).
    """
    features_index_csv = Path(features_index_csv)
    index_df = pd.read_csv(features_index_csv)
    index_df = index_df[index_df["status"] == "ok"]
    if splits:
        index_df = index_df[index_df["split"].isin(splits)]

    if index_df.empty:
        raise ValueError(
            f"No usable (status=ok, splits={splits}) rows found in {features_index_csv}. "
            f"Check that scripts/build_features.py actually succeeded for this split/condition."
        )

    unknown_labels = set(index_df["label"].unique()) - set(LABEL_TO_INT)
    if unknown_labels:
        raise ValueError(f"Unexpected label(s) in index: {unknown_labels}; expected {list(LABEL_TO_INT)}")

    feats, labels, file_ids, speaker_ids = [], [], [], []
    base_dir = features_index_csv.resolve().parent.parent.parent.parent
    for row in index_df.itertuples(index=False):
        row = row._asdict()
        fpath = Path(row["feature_path"])
        if not fpath.exists():
            alt = base_dir / row["feature_path"]
            if alt.exists():
                fpath = alt
        with np.load(fpath) as npz:
            if feature_key not in npz:
                raise KeyError(
                    f"feature_key={feature_key!r} not found in {row['feature_path']} "
                    f"(available keys: {list(npz.keys())}). Was --feature-set at build time "
                    f"set to include this key?"
                )
            feats.append(npz[feature_key])
        labels.append(LABEL_TO_INT[row["label"]])
        file_ids.append(row["file_id"])
        speaker_ids.append(row.get("speaker_id", ""))

    X = np.stack(feats).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, file_ids, speaker_ids
