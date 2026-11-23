"""
Corpus adapter for ASVspoof 2019 (Logical Access partition).

Responsibility: translate ASVspoof2019 LA's native protocol files and audio
layout into ManifestRow records. This module does NOT open/validate audio —
see manifest.py's integrity checker for that — it only parses labels/splits
and locates the expected file path for each entry.

Expected on-disk layout (the standard ASVspoof2019 LA distribution layout):

    LA/
      ASVspoof2019_LA_train/flac/*.flac
      ASVspoof2019_LA_dev/flac/*.flac
      ASVspoof2019_LA_eval/flac/*.flac
      ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt

Protocol line format (whitespace-separated), e.g.:
    LA_0079 LA_T_1138215 - A01 spoof
    LA_0079 LA_T_1271820 - -   bonafide

We parse defensively by taking the FIRST token as speaker_id, the SECOND
token as the file_id, the LAST token as the label, and the token immediately
before the label as the system_id. This tolerates the placeholder "-" columns
without hardcoding an exact column count, since minor protocol-file format
differences exist across ASVspoof releases/mirrors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from .schema import ManifestRow, LABELS

logger = logging.getLogger(__name__)

CORPUS_NAME = "asvspoof2019_la"

_PROTOCOL_FILENAMES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

_AUDIO_SUBDIRS = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev": "ASVspoof2019_LA_dev/flac",
    "eval": "ASVspoof2019_LA_eval/flac",
}


def _parse_protocol_line(line: str) -> tuple[str, str, str, str] | None:
    """Return (speaker_id, file_id, system_id, label) or None if unparseable."""
    tokens = line.strip().split()
    if len(tokens) < 3:
        return None
    label = tokens[-1].lower()
    if label not in LABELS:
        return None
    system_id = tokens[-2]
    speaker_id = tokens[0]
    file_id = tokens[1]
    return speaker_id, file_id, system_id, label


def iter_rows(la_root: str | Path, splits: tuple[str, ...] = ("train", "dev", "eval")) -> Iterator[ManifestRow]:
    """
    Yield ManifestRow objects (with audio-file-existence NOT yet checked)
    for the requested official splits of ASVspoof2019 LA.

    Parameters
    ----------
    la_root : path to the top-level "LA" directory of the ASVspoof2019
        distribution (the directory that directly contains
        ASVspoof2019_LA_train/, ASVspoof2019_LA_cm_protocols/, etc.)
    splits : which official partitions to include. Defaults to all three.
        Callers building a training manifest should typically pass
        ("train", "dev") and build the eval manifest separately, to make
        the eval-only usage explicit at call sites rather than implicit.
    """
    la_root = Path(la_root)
    protocol_dir = la_root / "ASVspoof2019_LA_cm_protocols"

    for split in splits:
        if split not in _PROTOCOL_FILENAMES:
            raise ValueError(f"Unknown split '{split}'; expected one of {tuple(_PROTOCOL_FILENAMES)}")

        protocol_path = protocol_dir / _PROTOCOL_FILENAMES[split]
        audio_dir = la_root / _AUDIO_SUBDIRS[split]

        if not protocol_path.exists():
            raise FileNotFoundError(
                f"Protocol file not found for split='{split}': {protocol_path}. "
                f"Check that --la-root points at the LA/ directory of the official distribution."
            )

        n_parsed, n_unparsed = 0, 0
        with open(protocol_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                if not raw_line.strip():
                    continue
                parsed = _parse_protocol_line(raw_line)
                if parsed is None:
                    n_unparsed += 1
                    continue
                speaker_id, file_id, system_id, label = parsed
                n_parsed += 1
                yield ManifestRow(
                    file_id=file_id,
                    file_path=str(audio_dir / f"{file_id}.flac"),
                    corpus=CORPUS_NAME,
                    split=split,
                    label=label,
                    speaker_id=speaker_id,
                    system_id=system_id,
                )

        if n_unparsed:
            logger.warning(
                "asvspoof2019 adapter: split=%s had %d unparseable protocol line(s) "
                "(skipped, not included in manifest)", split, n_unparsed
            )
        logger.info("asvspoof2019 adapter: split=%s parsed %d protocol entries", split, n_parsed)


def resolve_asvspoof_audio_path(
    raw_path: str | Path,
    file_id: str,
    split: str,
    audio_root: str | Path | None = None,
) -> Path:
    """Resolve an audio path, falling back to `audio_root` if `raw_path` was moved."""
    p = Path(raw_path)
    if p.exists() or audio_root is None:
        return p
    root = Path(audio_root)
    split_dir = f"ASVspoof2019_LA_{split}"
    for candidate in (
        root / split_dir / "flac" / f"{file_id}.flac",
        root / "LA" / split_dir / "flac" / f"{file_id}.flac",
        root / "LA" / "LA" / split_dir / "flac" / f"{file_id}.flac",
        root / split / "flac" / f"{file_id}.flac",
        root / f"{file_id}.flac",
    ):
        if candidate.exists():
            return candidate
    return p
