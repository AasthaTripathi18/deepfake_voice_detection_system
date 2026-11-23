#!/usr/bin/env python
"""
Apply the cleaning pipeline, optionally a named degradation condition, and
extract features for every valid row of a manifest, writing one .npz per
utterance plus a features index CSV.

This script deliberately processes ONE condition per invocation (default:
"clean"). Run it multiple times with different --condition values to build
the matched-condition and each mismatched-condition feature set separately
— this keeps condition tagging explicit at the file-system level (separate
output directories) rather than relying on a column alone, so it's obvious
from the directory layout which files came from which condition.

Examples:
    # Matched-condition (clean) features, train+dev only, for training:
    python scripts/build_features.py \
        --manifest data/manifests/asvspoof2019_la.csv \
        --splits train dev \
        --condition clean \
        --out-dir data/features/clean

    # Mismatched-condition features, eval split only, for RQ1's degradation
    # evaluation (run the SAME already-trained model on these later):
    python scripts/build_features.py \
        --manifest data/manifests/asvspoof2019_la.csv \
        --splits eval \
        --condition telephony_bandpass \
        --out-dir data/features/telephony_bandpass

    python scripts/build_features.py \
        --manifest data/manifests/asvspoof2019_la.csv \
        --splits eval \
        --condition codec_gsm \
        --out-dir data/features/codec_gsm

    python scripts/build_features.py \
        --manifest data/manifests/asvspoof2019_la.csv \
        --splits eval \
        --condition noise_snr --snr-db 10 \
        --out-dir data/features/noise_snr10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import cleaning, degradations, features as feat_mod  # noqa: E402
from src.dataset.asvspoof2019 import resolve_asvspoof_audio_path  # noqa: E402
from src.dataset.schema import STATUS_VALID  # noqa: E402


def process_one_file(
    file_path: str,
    condition: str,
    feature_set: str,
    clean_denoise: bool,
    condition_params: dict,
) -> dict:
    """Load, clean, degrade (if requested), and extract features for one file.

    Returns the feature record dict on success. Raises on failure — the
    caller is responsible for catching and logging per-file failures so one
    bad file cannot silently take down a whole batch run.
    """
    waveform, sr = cleaning.load_audio(file_path, target_sr=None)
    waveform = cleaning.clean(waveform, sr, trim=True, normalize=True, denoise=clean_denoise)

    if condition != "clean":
        waveform = degradations.apply_condition(waveform, sr, condition, **condition_params)

    return feat_mod.build_feature_record(waveform, sr, feature_set=feature_set)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Path to a manifest CSV from build_manifest.py")
    parser.add_argument("--out-dir", required=True, help="Output directory for this condition's features")
    parser.add_argument("--splits", nargs="+", default=None, help="Restrict to these splits (default: all present)")
    parser.add_argument(
        "--condition", default="clean",
        choices=["clean"] + degradations.list_conditions(),
        help="Degradation condition to apply after cleaning (default: clean, i.e. none)",
    )
    parser.add_argument("--snr-db", type=float, default=10.0, help="Used only when --condition noise_snr")
    parser.add_argument(
        "--feature-set", default="both",
        choices=[
            "mfcc_stats", "mfcc_deltas_stats", "lfcc_stats", "lfcc_deltas_stats",
            "lfcc_mfcc_fusion", "lfcc_spectrogram_2d", "logmel", "both", "all",
        ],
        help="Which feature representation(s) to compute (default: both)",
    )
    parser.add_argument(
        "--clean-denoise", action="store_true",
        help="Enable spectral-subtraction denoising in the cleaning step "
             "(OFF by default — see src/preprocessing/cleaning.py docstring "
             "for why this is an explicit opt-in, not a default).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Thread worker count (default: 1)")
    parser.add_argument("--audio-root", default=None, help="Optional override root directory containing ASVspoof2019 LA folders")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N valid rows (debug)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("build_features")

    manifest_df = pd.read_csv(args.manifest)
    manifest_df = manifest_df[manifest_df["status"] == STATUS_VALID]
    if args.splits:
        manifest_df = manifest_df[manifest_df["split"].isin(args.splits)]
    if args.limit:
        manifest_df = manifest_df.head(args.limit)

    if manifest_df.empty:
        log.error("No valid rows selected from manifest (check --splits and manifest status column).")
        return 1

    condition_params = {"snr_db": args.snr_db} if args.condition == "noise_snr" else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    n_ok, n_failed = 0, 0

    log.info(
        "Processing %d valid manifest rows -> condition=%s feature_set=%s out_dir=%s (workers=%d)",
        len(manifest_df), args.condition, args.feature_set, out_dir, args.workers,
    )

    audio_root = Path(args.audio_root) if args.audio_root else None

    def _run_row(row_dict: dict) -> tuple[dict, bool]:
        try:
            audio_path = str(resolve_asvspoof_audio_path(
                row_dict["file_path"], row_dict["file_id"], row_dict["split"], audio_root
            ))
            record = process_one_file(
                audio_path, args.condition, args.feature_set, args.clean_denoise, condition_params,
            )
            feature_path = out_dir / f"{row_dict['file_id']}.npz"
            np.savez_compressed(feature_path, **record)
            return ({
                "file_id": row_dict["file_id"], "split": row_dict["split"], "label": row_dict["label"],
                "speaker_id": row_dict["speaker_id"], "system_id": row_dict["system_id"],
                "condition": args.condition, "feature_path": str(feature_path),
                "status": "ok", "error": "",
            }, True)
        except Exception as exc:  # noqa: BLE001 — any failure here is logged and skipped, not fatal
            log.warning("Skipping file_id=%s (%s): %s", row_dict["file_id"], type(exc).__name__, exc)
            return ({
                "file_id": row_dict["file_id"], "split": row_dict["split"], "label": row_dict["label"],
                "speaker_id": row_dict["speaker_id"], "system_id": row_dict["system_id"],
                "condition": args.condition, "feature_path": "",
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            }, False)

    rows_in = [r._asdict() for r in manifest_df.itertuples(index=False)]
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_run_row, rows_in))
    else:
        results = [_run_row(r) for r in rows_in]

    for row_out, ok in results:
        index_rows.append(row_out)
        if ok:
            n_ok += 1
        else:
            n_failed += 1

    index_df = pd.DataFrame(index_rows)
    index_path = out_dir / "features_index.csv"
    index_df.to_csv(index_path, index=False)

    log.info("Done: %d succeeded, %d failed (see %s for per-file detail).", n_ok, n_failed, index_path)
    if n_failed:
        log.info(
            "Failures were logged, not silently dropped — inspect the 'error' column "
            "in %s before assuming this condition's feature set is complete.",
            index_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
