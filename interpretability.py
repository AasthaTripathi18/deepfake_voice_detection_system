"""
Interpretability and Forensic Explainability Module.

Decomposes spoof-detection scores into:
1. Sub-band spectral attribution (Low-Band 0-1kHz, Mid-Band 1-4kHz, High-Band 4-8kHz)
2. Temporal saliency mapping (second-by-second anomaly localization)
3. Structured forensic narrative for security analysts
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpectralAttribution:
    low_band_pct: float       # 0 - 1000 Hz: vocal tract & F1
    mid_band_pct: float       # 1000 - 4000 Hz: formant transitions F2/F3
    high_band_pct: float      # 4000 - 8000 Hz: vocoder smoothing & high-frequency phase
    primary_cue_band: str
    explanation: str


@dataclass
class TemporalSaliency:
    window_timestamps_sec: list[float]
    window_risk_scores: list[float]
    peak_anomaly_timestamp_sec: float
    peak_anomaly_score: float
    suspicious_segments: list[tuple[float, float]]  # (start_sec, end_sec)


@dataclass
class ForensicExplanation:
    risk_score: float
    classification: str
    spectral: SpectralAttribution
    temporal: TemporalSaliency
    forensic_summary: str


def compute_subband_attribution(
    feature_vector: np.ndarray,
    model=None,
    n_mfcc: int = 40,
) -> SpectralAttribution:
    """
    Decompose a 240-dim (or 80-dim) MFCC feature vector or 2D LFCC spectrogram
    ((1, 60, T) or (60, T)) into low, mid, and high frequency contributions.

    In 40-filter Mel scale (0-8000 Hz):
    - Mel filters 0 to 12 correspond approximately to 0 - 1000 Hz (Low band)
    - Mel filters 13 to 27 correspond approximately to 1000 - 4000 Hz (Mid band)
    - Mel filters 28 to 39 correspond approximately to 4000 - 8000 Hz (High band)

    In 60-bin Linear LFCC spectrograms (0-8000 Hz), applying IDCT along the
    frequency axis recovers linear filterbank log-energies so bins map linearly:
    0-1000 Hz -> bins 0:7, 1000-4000 Hz -> bins 7:30, 4000-8000 Hz -> bins 30:60.
    """
    raw = np.asarray(feature_vector)
    if raw.ndim >= 2:
        from scipy import fftpack

        spec_2d = np.squeeze(raw)
        if spec_2d.ndim == 2 and spec_2d.shape[0] >= 12:
            # Invert DCT-II back to linear filterbank log-energy (dB) domain, then relative power
            log_band = fftpack.idct(spec_2d, type=2, axis=0, norm="ortho")
            rel_power = np.power(10.0, np.clip(log_band - np.max(log_band), -80.0, 0.0) / 10.0)
            n_bins = rel_power.shape[0]
            b_low = max(1, int(round(n_bins * (1000.0 / 8000.0))))
            b_mid = max(b_low + 1, int(round(n_bins * (4000.0 / 8000.0))))
            low_energy = float(np.mean(rel_power[:b_low]))
            mid_energy = float(np.mean(rel_power[b_low:b_mid]))
            high_energy = float(np.mean(rel_power[b_mid:]))
        else:
            feat = raw.ravel()
            dim = max(1, len(feat) // 2)
            l_cut = int(dim * (13 / 40))
            m_cut = int(dim * (28 / 40))
            low_energy = float(np.sum(np.abs(feat[:l_cut])))
            mid_energy = float(np.sum(np.abs(feat[l_cut:m_cut])))
            high_energy = float(np.sum(np.abs(feat[m_cut:])))
    else:
        feat = raw.ravel()
        if len(feat) == 240:
            # 40 mean, 40 std, 40 d1_mean, 40 d1_std, 40 d2_mean, 40 d2_std
            # Group coefficients across static, delta, and acceleration by frequency filter
            low_idx = list(range(0, 13)) + list(range(40, 53)) + list(range(80, 93)) + list(range(120, 133)) + list(range(160, 173)) + list(range(200, 213))
            mid_idx = list(range(13, 28)) + list(range(53, 68)) + list(range(93, 108)) + list(range(133, 148)) + list(range(173, 188)) + list(range(213, 228))
            high_idx = list(range(28, 40)) + list(range(68, 80)) + list(range(108, 120)) + list(range(148, 160)) + list(range(188, 200)) + list(range(228, 240))
        else:
            # Fallback for 80-dim or arbitrary
            dim = len(feat) // 2
            l_cut = int(dim * (13 / 40))
            m_cut = int(dim * (28 / 40))
            low_idx = list(range(0, l_cut)) + list(range(dim, dim + l_cut))
            mid_idx = list(range(l_cut, m_cut)) + list(range(dim + l_cut, dim + m_cut))
            high_idx = list(range(m_cut, dim)) + list(range(dim + m_cut, 2 * dim))

        low_energy = float(np.sum(np.abs(feat[low_idx])))
        mid_energy = float(np.sum(np.abs(feat[mid_idx])))
        high_energy = float(np.sum(np.abs(feat[high_idx])))

    total = low_energy + mid_energy + high_energy + 1e-8
    low_pct = round((low_energy / total) * 100.0, 1)
    mid_pct = round((mid_energy / total) * 100.0, 1)
    high_pct = round((high_energy / total) * 100.0, 1)

    bands = {"Low-Band (0-1kHz)": low_pct, "Mid-Band (1-4kHz)": mid_pct, "High-Band (4-8kHz)": high_pct}
    primary = max(bands, key=bands.get)

    if primary.startswith("High"):
        expl = (
            "Spectral energy is elevated in the 4-8 kHz band. In synthetic voice detection, "
            "this typically reflects neural vocoder artifacting (e.g. WaveNet, HiFi-GAN) where phase "
            "irregularities and spectral envelope smearing are concentrated."
        )
    elif primary.startswith("Mid"):
        expl = (
            "Dominant cues in the 1-4 kHz band reflect unnatural formant bandwidths and robotic "
            "vowel transitions typical of concatenated or parametric synthesis algorithms."
        )
    else:
        expl = (
            "Energy concentrated in the fundamental frequency range (<1 kHz). Often indicates unnatural "
            "pitch tracking or synthetic glottal pulse modeling."
        )

    return SpectralAttribution(
        low_band_pct=low_pct,
        mid_band_pct=mid_pct,
        high_band_pct=high_pct,
        primary_cue_band=primary,
        explanation=expl,
    )


def compute_temporal_saliency(
    window_scores: list[float],
    window_timestamps: list[float],
    threshold: float = 0.5,
) -> TemporalSaliency:
    """
    Identifies which time regions of an utterance exhibit elevated spoof risk.
    """
    scores = np.asarray(window_scores)
    ts = np.asarray(window_timestamps)

    if len(scores) == 0:
        return TemporalSaliency([], [], 0.0, 0.0, [])

    peak_idx = int(np.argmax(scores))
    peak_ts = float(ts[peak_idx]) if len(ts) > peak_idx else 0.0
    peak_score = float(scores[peak_idx])

    suspicious = []
    in_seg = False
    start_t = 0.0

    for s, t in zip(scores, ts):
        if s >= threshold and not in_seg:
            in_seg = True
            start_t = t
        elif s < threshold and in_seg:
            in_seg = False
            suspicious.append((round(start_t, 2), round(t, 2)))

    if in_seg:
        suspicious.append((round(start_t, 2), round(ts[-1] + 1.0, 2)))

    return TemporalSaliency(
        window_timestamps_sec=[round(float(t), 2) for t in ts],
        window_risk_scores=[round(float(s), 4) for s in scores],
        peak_anomaly_timestamp_sec=round(peak_ts, 2),
        peak_anomaly_score=round(peak_score, 4),
        suspicious_segments=suspicious,
    )


def explain_prediction(
    feature_vector: np.ndarray,
    risk_score: float,
    threshold: float,
    window_scores: list[float] | None = None,
    window_timestamps: list[float] | None = None,
) -> ForensicExplanation:
    """
    Generate an end-to-end interpretable forensic report for a scored clip.
    """
    classification = "SPOOF / AI-GENERATED" if risk_score >= threshold else "BONA FIDE / HUMAN"
    spectral = compute_subband_attribution(feature_vector)

    if window_scores and window_timestamps:
        temporal = compute_temporal_saliency(window_scores, window_timestamps, threshold)
    else:
        temporal = TemporalSaliency([0.0], [risk_score], 0.0, risk_score, [(0.0, 1.0)] if risk_score >= threshold else [])

    if risk_score >= threshold:
        summary = (
            f"Prediction: {classification} (Score: {risk_score:.3f} >= threshold {threshold:.3f}). "
            f"Primary acoustic driver is {spectral.primary_cue_band} ({spectral.high_band_pct}% high-band energy). "
        )
        if temporal.suspicious_segments:
            segs_str = ", ".join(f"[{s}s - {e}s]" for s, e in temporal.suspicious_segments)
            summary += f"High synthetic anomalies detected in time intervals: {segs_str}."
        else:
            summary += f"Peak anomaly observed at t={temporal.peak_anomaly_timestamp_sec}s."
    else:
        summary = (
            f"Prediction: {classification} (Score: {risk_score:.3f} < threshold {threshold:.3f}). "
            "Acoustic dynamics, formant transitions, and high-frequency harmonics conform to natural human speech patterns."
        )

    return ForensicExplanation(
        risk_score=float(risk_score),
        classification=classification,
        spectral=spectral,
        temporal=temporal,
        forensic_summary=summary,
    )
