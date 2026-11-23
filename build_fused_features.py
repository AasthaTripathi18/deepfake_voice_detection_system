#!/usr/bin/env python
"""
Build 480-dimensional fused acoustic feature matrices (LFCC 240 + MFCC 240)
for train, dev, and eval splits by combining the aligned precomputed caches.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger("build_fused_features")


def build_fused_cache():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    mfcc_dir = ROOT_DIR / "data/features/mfcc240"
    lfcc_dir = ROOT_DIR / "data/features/lfcc240"
    out_dir = ROOT_DIR / "data/features/fused480"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "eval"]:
        mfcc_path = mfcc_dir / f"mfcc240_{split}.npz"
        lfcc_path = lfcc_dir / f"lfcc240_{split}.npz"
        out_path = out_dir / f"fused480_{split}.npz"

        logger.info("Processing %s split...", split)
        d_mfcc = np.load(mfcc_path)
        d_lfcc = np.load(lfcc_path)

        assert np.array_equal(d_mfcc["file_ids"], d_lfcc["file_ids"]), f"File IDs mismatch in {split}"
        assert np.array_equal(d_mfcc["y"], d_lfcc["y"]), f"Labels mismatch in {split}"

        # Concatenate LFCC (0:240) and MFCC (240:480) horizontally
        X_fused = np.hstack([d_lfcc["X"], d_mfcc["X"]]).astype(np.float32)
        y = d_lfcc["y"]
        file_ids = d_lfcc["file_ids"]
        success_mask = d_lfcc["success_mask"] & d_mfcc["success_mask"]

        logger.info("Saving %s: X shape %s ...", out_path.name, X_fused.shape)
        np.savez_compressed(
            out_path,
            X=X_fused,
            y=y,
            file_ids=file_ids,
            success_mask=success_mask,
        )
        logger.info("Successfully generated %s (%d samples, %d dims)", out_path.name, len(y), X_fused.shape[1])


if __name__ == "__main__":
    build_fused_cache()
