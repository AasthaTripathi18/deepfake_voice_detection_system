"""
Manifest builder + integrity checker.

Turns an iterable of ManifestRow objects (from a corpus adapter) into a
single pandas DataFrame, verifies every file is actually readable and
matches expected audio properties, and reports (rather than silently
drops) anything invalid.

This module also runs a leakage check: it confirms that no speaker_id
appears in more than one of the official splits. For ASVspoof2019 LA this
should always pass, since the corpus is distributed pre-split and
speaker-disjoint — the check exists to catch adapter bugs or a corrupted/
mismatched protocol file, not to fix leakage after the fact. If it fails,
that is treated as a hard error requiring investigation, not something
this code attempts to silently repair by re-splitting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import soundfile as sf

from .schema import (
    ManifestRow,
    STATUS_VALID,
    STATUS_SKIPPED,
    DEFAULT_EXPECTED_SAMPLE_RATE,
    DEFAULT_EXPECTED_CHANNELS,
    DEFAULT_MIN_DURATION_SEC,
    DEFAULT_MAX_DURATION_SEC,
)

logger = logging.getLogger(__name__)


def build_raw_manifest(rows: Iterable[ManifestRow]) -> pd.DataFrame:
    """Materialize adapter output into a DataFrame, before integrity checking."""
    df = pd.DataFrame([r.__dict__ for r in rows], columns=ManifestRow.columns())
    if df.empty:
        raise ValueError("No manifest rows produced by the adapter — check corpus path/splits.")
    return df


def check_integrity(
    df: pd.DataFrame,
    expected_sample_rate: int = DEFAULT_EXPECTED_SAMPLE_RATE,
    expected_channels: int = DEFAULT_EXPECTED_CHANNELS,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
) -> pd.DataFrame:
    """
    Open every file referenced in df, record its actual sample rate,
    duration, and channel count, and mark status=valid/skipped with a
    human-readable skip_reason. Returns a NEW DataFrame (does not mutate
    the input) with these columns filled in for every row — nothing is
    dropped from the table, so the full accounting of what was skipped
    and why stays visible and auditable.

    Rows are skipped (status=STATUS_SKIPPED), not silently excluded, when:
      - the file does not exist or cannot be opened as audio
      - the sample rate differs from expected_sample_rate
      - the channel count differs from expected_channels
      - the duration falls outside [min_duration_sec, max_duration_sec]

    A mismatched sample rate/channel count is treated as a SKIP rather
    than something to silently fix here (e.g. auto-resampling at this
    stage), because doing so during manifest-building would hide corpus
    data-quality issues. Resampling as a deliberate, logged preprocessing
    step belongs in src/preprocessing, not here.
    """
    records = []
    n_valid = 0
    n_skipped = 0
    skip_reason_counts: dict[str, int] = {}

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        path = Path(row_dict["file_path"])

        if not path.exists():
            row_dict.update(status=STATUS_SKIPPED, skip_reason="file_not_found")
        else:
            try:
                info = sf.info(str(path))
                duration_sec = info.frames / float(info.samplerate) if info.samplerate else 0.0
                row_dict["sample_rate"] = info.samplerate
                row_dict["channels"] = info.channels
                row_dict["duration_sec"] = round(duration_sec, 4)

                if info.samplerate != expected_sample_rate:
                    row_dict.update(
                        status=STATUS_SKIPPED,
                        skip_reason=f"unexpected_sample_rate:{info.samplerate}",
                    )
                elif info.channels != expected_channels:
                    row_dict.update(
                        status=STATUS_SKIPPED,
                        skip_reason=f"unexpected_channels:{info.channels}",
                    )
                elif duration_sec < min_duration_sec:
                    row_dict.update(
                        status=STATUS_SKIPPED,
                        skip_reason=f"too_short:{duration_sec:.3f}s",
                    )
                elif duration_sec > max_duration_sec:
                    row_dict.update(
                        status=STATUS_SKIPPED,
                        skip_reason=f"too_long:{duration_sec:.3f}s",
                    )
                else:
                    row_dict.update(status=STATUS_VALID, skip_reason="")
            except Exception as exc:  # noqa: BLE001 — deliberately broad: any unreadable file is a skip
                row_dict.update(status=STATUS_SKIPPED, skip_reason=f"unreadable:{type(exc).__name__}")

        if row_dict["status"] == STATUS_VALID:
            n_valid += 1
        else:
            n_skipped += 1
            skip_reason_counts[row_dict["skip_reason"]] = skip_reason_counts.get(row_dict["skip_reason"], 0) + 1

        records.append(row_dict)

    checked = pd.DataFrame.from_records(records, columns=ManifestRow.columns())

    logger.info("Integrity check complete: %d valid, %d skipped (of %d total).", n_valid, n_skipped, len(checked))
    if skip_reason_counts:
        for reason, count in sorted(skip_reason_counts.items(), key=lambda kv: -kv[1]):
            logger.info("  skipped[%s] = %d", reason, count)

    return checked


def check_speaker_split_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of any speaker_id found in more than one split.
    Empty result == no leakage detected. Only considers rows with a
    non-empty speaker_id (some corpora may not provide one).
    """
    has_speaker = df[df["speaker_id"].astype(str).str.len() > 0]
    grouped = has_speaker.groupby("speaker_id")["split"].nunique()
    leaking_speakers = grouped[grouped > 1]
    if leaking_speakers.empty:
        return has_speaker.iloc[0:0]  # empty frame, correct columns
    leaked = has_speaker[has_speaker["speaker_id"].isin(leaking_speakers.index)]
    return leaked.sort_values(["speaker_id", "split"])


def summarize(df: pd.DataFrame) -> str:
    """Human-readable summary for logs / console output."""
    lines = ["Manifest summary:"]
    for split in sorted(df["split"].unique()):
        split_df = df[df["split"] == split]
        valid_df = split_df[split_df["status"] == STATUS_VALID]
        counts = valid_df["label"].value_counts().to_dict()
        lines.append(
            f"  split={split:5s}  valid={len(valid_df):6d}  skipped={len(split_df) - len(valid_df):5d}  "
            f"label_counts={counts}"
        )
    return "\n".join(lines)


def save_manifest(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Manifest written to %s (%d rows)", out_path, len(df))
