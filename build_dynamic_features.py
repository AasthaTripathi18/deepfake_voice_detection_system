#!/usr/bin/env python
"""
Extract 240-dimensional dynamic MFCC features (static + delta + delta-delta with mean and std pooling)
for the ASVspoof 2019 LA dataset, saving directly into consolidated matrix files.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.cleaning import trim_silence, normalize_amplitude
from src.preprocessing.features import extract_mfcc_deltas_stats

logger = logging.getLogger("build_dynamic_features")
LABEL_TO_INT = {"bonafide": 0, "spoof": 1}


def process_audio(file_path: str, sr_target: int = 16000) -> np.ndarray:
    """Load, clean (-3 dBFS peak norm + trim silence), and extract 240-dim MFCC features."""
    wav, sr = sf.read(file_path, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != sr_target:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_target)
        sr = sr_target

    # Clean (matched training preprocessing)
    trimmed = trim_silence(wav, top_db=30.0)
    if len(trimmed) > 0 and np.max(np.abs(trimmed)) > 0:
        waveform = normalize_amplitude(trimmed, method="peak", target_dbfs=-3.0)
    else:
        waveform = trimmed if len(trimmed) > 0 else wav

    return extract_mfcc_deltas_stats(waveform, sr)


def extract_split(df_split: pd.DataFrame, out_path: Path, max_workers: int = 16):
    """Extract features for a manifest split and save to a single consolidated .npz."""
    n_samples = len(df_split)
    logger.info("Extracting %d samples using %d threads...", n_samples, max_workers)
    t0 = time.time()

    def _worker(row):
        try:
            feat = process_audio(row.file_path)
            return feat, LABEL_TO_INT[row.label], row.file_id, True
        except Exception as e:
            logger.error("Failed on %s: %s", row.file_id, e)
            return np.zeros(240, dtype=np.float32), LABEL_TO_INT[row.label], row.file_id, False

    rows = list(df_split.itertuples(index=False))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, res in enumerate(executor.map(_worker, rows)):
            results.append(res)
            if (i + 1) % 2500 == 0 or (i + 1) == n_samples:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info("Progress: %d / %d (%.1f%%) [%.1f clips/sec]", i + 1, n_samples, (i + 1) / n_samples * 100, rate)

    feats = np.stack([r[0] for r in results]).astype(np.float32)
    labels = np.array([r[1] for r in results], dtype=np.int64)
    file_ids = np.array([r[2] for r in results])
    success_mask = np.array([r[3] for r in results], dtype=bool)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=feats,
        y=labels,
        file_ids=file_ids,
        success_mask=success_mask,
    )
    logger.info("Saved %s: X shape %s in %.1fs", out_path, feats.shape, time.time() - t0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/asvspoof2019_la.csv")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"], help="Splits to extract")
    parser.add_argument("--out-dir", default="data/features/mfcc240")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    manifest_path = ROOT_DIR / args.manifest
    df = pd.read_csv(manifest_path)
    out_dir = ROOT_DIR / args.out_dir

    for split in args.splits:
        df_split = df[df["split"] == split].copy()
        if args.limit:
            df_split = df_split.head(args.limit)
        out_file = out_dir / f"mfcc240_{split}.npz"
        logger.info("=== Processing split: %s (%d clips) -> %s ===", split, len(df_split), out_file)
        extract_split(df_split, out_file, max_workers=args.workers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
