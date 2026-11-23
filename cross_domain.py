"""
Cross-Domain Dataset Loader and Domain-Shift Analyzer.

Enables ingesting and evaluating external out-of-domain voice datasets
(e.g., In-the-Wild, WaveFake, FakeAVCeleb, or custom user folders)
with audio integrity validation and domain discrepancy diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

logger = logging.getLogger("cross_domain")


@dataclass
class DomainStatistics:
    dataset_name: str
    total_samples: int
    mean_duration_sec: float
    mean_rms_energy: float
    mean_zero_crossing_rate: float
    sample_rates: list[int]
    formats: list[str]


class CrossDomainDataset:
    """
    Ingests and standardizes external audio datasets with varying folder structures.
    """

    def __init__(self, root_dir: str | Path, dataset_name: str = "external"):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.samples: list[dict] = []

    def scan_directory(
        self,
        bonafide_dir: str = "bonafide",
        spoof_dir: str = "spoof",
        supported_exts: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg"),
    ) -> pd.DataFrame:
        """
        Scan directory containing bonafide/ and spoof/ subdirectories.
        """
        records = []
        for label, sub in [("bonafide", bonafide_dir), ("spoof", spoof_dir)]:
            target_path = self.root_dir / sub
            if not target_path.exists():
                logger.warning("Subdirectory %s does not exist in %s", sub, self.root_dir)
                continue

            for ext in supported_exts:
                for file_path in target_path.glob(f"*{ext}"):
                    try:
                        info = sf.info(file_path)
                        records.append({
                            "file_id": file_path.stem,
                            "file_path": str(file_path.resolve()),
                            "corpus": self.dataset_name,
                            "split": "eval",
                            "label": label,
                            "speaker_id": "unknown",
                            "system_id": "-",
                            "condition": "unconstrained",
                            "sample_rate": info.samplerate,
                            "duration_sec": info.duration,
                            "channels": info.channels,
                            "status": "valid",
                        })
                    except Exception as exc:
                        logger.debug("Skipping unreadable file %s: %s", file_path, exc)

        self.samples = records
        df = pd.DataFrame(records)
        logger.info("Scanned %s: %d total valid audio files", self.dataset_name, len(df))
        return df

    def scan_from_manifest(self, manifest_csv: str | Path) -> pd.DataFrame:
        """
        Load an external manifest CSV containing at least 'file_path' and 'label'.
        """
        df = pd.read_csv(manifest_csv)
        if "file_path" not in df.columns or "label" not in df.columns:
            raise ValueError("Manifest must contain at least 'file_path' and 'label' columns.")

        records = []
        for _, row in df.iterrows():
            fp = Path(row["file_path"])
            if not fp.is_absolute():
                fp = self.root_dir / fp

            if fp.exists():
                records.append({
                    "file_id": row.get("file_id", fp.stem),
                    "file_path": str(fp.resolve()),
                    "corpus": self.dataset_name,
                    "split": row.get("split", "eval"),
                    "label": row["label"],
                    "speaker_id": row.get("speaker_id", "unknown"),
                    "system_id": row.get("system_id", "-"),
                    "condition": row.get("condition", "unconstrained"),
                    "status": "valid",
                })
        self.samples = records
        return pd.DataFrame(records)

    def compute_domain_statistics(self, max_samples: int = 50) -> DomainStatistics:
        """
        Compute acoustic distribution properties to quantify cross-domain shift.
        """
        durations = []
        energies = []
        zcrs = []
        srs = set()
        formats = set()

        subset = self.samples[:max_samples]
        for s in subset:
            p = Path(s["file_path"])
            formats.add(p.suffix.lower())
            try:
                y, sr = librosa.load(p, sr=None)
                durations.append(len(y) / sr)
                energies.append(float(np.sqrt(np.mean(y**2))))
                zcrs.append(float(np.mean(librosa.feature.zero_crossing_rate(y))))
                srs.add(sr)
            except Exception:
                continue

        return DomainStatistics(
            dataset_name=self.dataset_name,
            total_samples=len(self.samples),
            mean_duration_sec=float(np.mean(durations)) if durations else 0.0,
            mean_rms_energy=float(np.mean(energies)) if energies else 0.0,
            mean_zero_crossing_rate=float(np.mean(zcrs)) if zcrs else 0.0,
            sample_rates=list(srs),
            formats=list(formats),
        )
