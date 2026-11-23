"""
Quality gating for one analysis window, BEFORE it's passed to feature
extraction / the model. Pure numpy — deliberately kept independent of
librosa/soundfile so these checks run even in an environment that only
has the modeling stack installed (the same reasoning as
src/inference/scorer.py's lazy preprocessing import).

Statuses this module can assign (see QualityAssessment.status):
  "ok"                    — proceed to inference normally.
  "insufficient_audio"    — window is shorter than the configured minimum
                              (the sliding-window "short sample" case, or
                              the leftover partial window at stream end).
                              No score is produced.
  "silence"                — essentially no voiced content at all
                              (voiced_fraction ~ 0). No score is produced.
  "insufficient_speech"    — some voiced content, but below the configured
                              minimum fraction of the window. No score is
                              produced, since a score computed on mostly-
                              silence would not meaningfully reflect
                              spoof-likelihood.
  "high_noise_low_confidence" — a score IS still produced (noisy audio is
                              an expected, not a fatal, condition for a
                              live demo), but flagged so the uncertainty
                              estimate downstream is pushed higher rather
                              than silently reported as ordinary.

None of these heuristics are claimed to be a calibrated, validated VAD or
SNR estimator — they are simple, fast, explainable checks appropriate for
a near-real-time gate, not a replacement for the more careful offline
noise-condition evaluation this project runs separately (RQ1 experiment).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .windowing import Window


@dataclass
class QualityAssessment:
    status: str
    voiced_fraction: float
    estimated_snr_db: float | None
    proceed_to_inference: bool
    reasons: list[str]
    clipping_fraction: float = 0.0


def _frame_rms_db(waveform: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))

    if waveform.size == 0:
        return np.array([])
    if waveform.size < frame_len:
        frame_len = waveform.size
        hop_len = frame_len

    n_frames = max(1, 1 + (waveform.size - frame_len) // hop_len)
    rms_values = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        seg = waveform[start:start + frame_len]
        rms_values[i] = np.sqrt(np.mean(np.square(seg)) + 1e-12)

    return 20.0 * np.log10(np.maximum(rms_values, 1e-9))


def estimate_voiced_fraction(waveform: np.ndarray, sr: int, silence_dbfs_threshold: float = -40.0) -> float:
    """
    Fraction of ~25ms frames whose RMS level exceeds an ABSOLUTE dBFS
    floor. This is a simple energy-based proxy for "how much of this
    window looks like it contains speech at all" — not a phonetic or
    learned VAD. A fixed absolute threshold is used (rather than
    relative-to-peak) so a window that is uniformly quiet throughout
    (e.g. captured with the mic muted) is correctly scored as ~0 voiced,
    which a peak-relative measure could miss.
    """
    rms_db = _frame_rms_db(waveform, sr)
    if rms_db.size == 0:
        return 0.0
    return float(np.mean(rms_db > silence_dbfs_threshold))


def estimate_snr_db(waveform: np.ndarray, sr: int) -> float | None:
    """
    Rough single-channel SNR PROXY: the gap between the 90th and 10th
    percentile of frame RMS (dB), i.e. "how much louder are the loud
    (likely speech) parts than the quiet (likely background) parts."

    This is NOT a true SNR measurement — that requires a separate noise-
    only reference this single window doesn't have — but it is a fast,
    explainable, monotonic-enough proxy for "is there a clear speech-vs-
    background contrast in this window," which is the property that
    actually matters for deciding whether to trust a score more or less.
    Returns None if there isn't enough audio to compute frames at all.

    KNOWN LIMITATION: a signal with almost no frame-to-frame loudness
    variation at all — a sustained, unmodulated tone, or audio whose
    dynamic range has been heavily compressed/normalized — will score a
    near-zero gap here and get flagged "high_noise_low_confidence" even
    if it is actually clean, simply because this proxy has no other
    signal to go on. Real speech is expected to have natural envelope
    variation (syllables, pauses) that avoids this case in practice, but
    it is a real edge case worth knowing about, not a hidden one.
    """
    rms_db = _frame_rms_db(waveform, sr)
    if rms_db.size < 2:
        return None
    return float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10))


def estimate_clipping_fraction(waveform: np.ndarray, clip_threshold: float = 0.99) -> float:
    """
    Fraction of audio samples at or near digital ceiling (>= clip_threshold).
    Microphone saturation creates clipped harmonics that degrade cepstral features.
    """
    if waveform.size == 0:
        return 0.0
    return float(np.mean(np.abs(waveform) >= clip_threshold))


def assess_window_quality(
    window: Window,
    min_duration_sec: float = 1.0,
    min_voiced_fraction: float = 0.15,
    silence_dbfs_threshold: float = -40.0,
    min_snr_db: float = 6.0,
    max_clipping_fraction: float = 0.01,
) -> QualityAssessment:
    reasons = []

    if window.duration_sec < min_duration_sec:
        reasons.append(
            f"window duration {window.duration_sec:.2f}s < minimum {min_duration_sec:.2f}s "
            f"(is_full={window.is_full}) — too short to extract a reliable feature vector."
        )
        return QualityAssessment(
            status="insufficient_audio", voiced_fraction=0.0, estimated_snr_db=None,
            proceed_to_inference=False, reasons=reasons, clipping_fraction=0.0,
        )

    voiced_fraction = estimate_voiced_fraction(window.waveform, window.sample_rate, silence_dbfs_threshold)

    if voiced_fraction <= 0.01:
        reasons.append(f"voiced_fraction={voiced_fraction:.3f} — essentially no energy above the silence floor.")
        return QualityAssessment(
            status="silence", voiced_fraction=voiced_fraction, estimated_snr_db=None,
            proceed_to_inference=False, reasons=reasons, clipping_fraction=0.0,
        )

    if voiced_fraction < min_voiced_fraction:
        reasons.append(
            f"voiced_fraction={voiced_fraction:.3f} < minimum {min_voiced_fraction:.3f} "
            f"— too little apparent speech content in this window to score meaningfully."
        )
        return QualityAssessment(
            status="insufficient_speech", voiced_fraction=voiced_fraction, estimated_snr_db=None,
            proceed_to_inference=False, reasons=reasons, clipping_fraction=0.0,
        )

    clipping_fraction = estimate_clipping_fraction(window.waveform)
    if clipping_fraction > max_clipping_fraction:
        reasons.append(
            f"clipping_fraction={clipping_fraction:.3f} > maximum {max_clipping_fraction:.3f} "
            f"— microphone saturation/clipping detected; spectral distortion may reduce spoof detection reliability."
        )
        snr_db = estimate_snr_db(window.waveform, window.sample_rate)
        return QualityAssessment(
            status="high_noise_low_confidence", voiced_fraction=voiced_fraction, estimated_snr_db=snr_db,
            proceed_to_inference=True, reasons=reasons, clipping_fraction=clipping_fraction,
        )

    snr_db = estimate_snr_db(window.waveform, window.sample_rate)
    if snr_db is not None and snr_db < min_snr_db:
        reasons.append(
            f"estimated_snr_db={snr_db:.1f} < minimum {min_snr_db:.1f} "
            f"— proceeding to inference, but flagging elevated uncertainty."
        )
        return QualityAssessment(
            status="high_noise_low_confidence", voiced_fraction=voiced_fraction, estimated_snr_db=snr_db,
            proceed_to_inference=True, reasons=reasons, clipping_fraction=clipping_fraction,
        )

    if snr_db is not None and min_snr_db <= snr_db < 10.0:
        reasons.append(
            f"Note: estimated_snr_db={snr_db:.1f} is below optimal clean speech (10.0 dB); background ambience is present."
        )

    return QualityAssessment(
        status="ok", voiced_fraction=voiced_fraction, estimated_snr_db=snr_db,
        proceed_to_inference=True, reasons=reasons, clipping_fraction=clipping_fraction,
    )
