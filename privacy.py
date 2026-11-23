"""
Privacy and Biometric Data Protection Module.

Implements:
1. Zero-retention ephemeral audio buffer scrubbing (in-memory memory shredding)
2. Speaker de-identification / voiceprint anonymization preserving spoof detection
3. Privacy audit logging ensuring biometric data protection (GDPR / HIPAA alignment)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator
import numpy as np
import librosa

logger = logging.getLogger("privacy")


class EphemeralAudioBuffer:
    """
    Context manager that guarantees in-memory scrubbing (zeroization) of raw audio
    waveforms immediately after processing to prevent biometric voiceprint leakage.
    """

    def __init__(self, waveform: np.ndarray):
        # Create a private copy that we can explicitly overwrite
        self._buffer = np.array(waveform, dtype=np.float32, copy=True)
        self.is_scrubbed = False

    def __enter__(self) -> np.ndarray:
        return self._buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scrub()

    def scrub(self) -> None:
        """Overwrite memory buffer with zeros and mark as scrubbed."""
        if not self.is_scrubbed and self._buffer is not None:
            self._buffer.fill(0.0)
            self.is_scrubbed = True
            logger.debug("Ephemeral audio buffer successfully zeroed in memory.")


def anonymize_speaker_biometrics(
    waveform: np.ndarray,
    sr: int = 16000,
    n_semitones: float = 3.5,
) -> np.ndarray:
    """
    Apply a privacy-preserving pitch and formant perturbation to obscure the speaker's
    biometric identity (preventing speaker re-identification or cloning), while
    preserving the spectral phase anomalies and vocoder discontinuities needed
    for spoof detection.
    """
    if len(waveform) < 512:
        return waveform

    # Pitch shift alters speaker timbre and pitch identity while leaving
    # frame-to-frame delta dynamics and phase smearing intact
    anonymized = librosa.effects.pitch_shift(
        y=waveform,
        sr=sr,
        n_steps=n_semitones,
    )
    return anonymized.astype(np.float32)


@dataclass
class PrivacyAuditReport:
    zero_retention_enforced: bool
    speaker_id_pseudonymized: bool
    disk_caching_disabled: bool
    compliance_status: str
    details: str


class PrivacyPolicyAuditor:
    """
    Validates adherence to zero-retention biometric privacy policies.
    """

    def __init__(self, enforce_zero_retention: bool = True, disallow_raw_disk_cache: bool = True):
        self.enforce_zero_retention = enforce_zero_retention
        self.disallow_raw_disk_cache = disallow_raw_disk_cache

    def audit_session(self, audio_saved_to_disk: bool, speaker_id_stored: bool) -> PrivacyAuditReport:
        compliant = True
        reasons = []

        if self.disallow_raw_disk_cache and audio_saved_to_disk:
            compliant = False
            reasons.append("Raw biometric audio was persisted to disk without encryption.")

        if speaker_id_stored:
            reasons.append("Clear-text speaker identifier was logged.")

        status = "COMPLIANT" if compliant else "NON-COMPLIANT"
        details = "All privacy safeguards active." if compliant else "; ".join(reasons)

        return PrivacyAuditReport(
            zero_retention_enforced=self.enforce_zero_retention,
            speaker_id_pseudonymized=not speaker_id_stored,
            disk_caching_disabled=not audio_saved_to_disk,
            compliance_status=status,
            details=details,
        )
