#!/usr/bin/env python
"""
Build and validate the ASVspoof2019 LA manifest.

Example:
    python scripts/build_manifest.py \
        --la-root /path/to/LA \
        --out data/manifests/asvspoof2019_la.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/build_manifest.py` from the repo root
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import asvspoof2019, manifest as manifest_mod  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--la-root", required=True, help="Path to the ASVspoof2019 'LA' directory")
    parser.add_argument("--out", required=True, help="Output manifest CSV path")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "dev", "eval"],
        help="Official splits to include (default: all three)",
    )
    parser.add_argument(
        "--fail-on-leakage", action="store_true",
        help="Exit with an error if any speaker_id appears in more than one split "
             "(default: warn and continue, since this should never happen with the "
             "official ASVspoof2019 LA distribution and likely indicates a corrupted "
             "or mismatched protocol file if it does).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("build_manifest")

    log.info("Parsing protocol files under %s (splits=%s)", args.la_root, args.splits)
    rows = list(asvspoof2019.iter_rows(args.la_root, splits=tuple(args.splits)))
    log.info("Adapter produced %d raw manifest rows (before integrity check)", len(rows))

    raw_df = manifest_mod.build_raw_manifest(rows)

    log.info("Running integrity checks (this opens every audio file once)...")
    checked_df = manifest_mod.check_integrity(raw_df)

    leakage_df = manifest_mod.check_speaker_split_leakage(checked_df)
    if not leakage_df.empty:
        msg = (
            f"Speaker/split leakage detected for {leakage_df['speaker_id'].nunique()} speaker(s): "
            f"{sorted(leakage_df['speaker_id'].unique())[:10]}... "
            "This should not happen with the official ASVspoof2019 LA distribution — "
            "check that --la-root points at an unmodified copy."
        )
        if args.fail_on_leakage:
            log.error(msg)
            return 1
        log.warning(msg)
    else:
        log.info("Speaker/split leakage check passed: no speaker appears in more than one split.")

    print(manifest_mod.summarize(checked_df))

    manifest_mod.save_manifest(checked_df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
