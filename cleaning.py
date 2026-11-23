"""
Cleaning pipeline.

Design decision (documented, not just implemented):

The base paper's preprocessing stage is described as: noise reduction via
spectral subtraction, silence removal via energy-based VAD, and amplitude
normalization. We implement all three functions here, but we do NOT enable
spectral-subtraction denoising by default in the main pipeline.

Why: spectral subtraction suppresses low-energy, high-frequency spectral
detail. That is exactly the kind of subtle spectral artifact that
distinguishes synthetic from genuine speech (this is the premise of using
MFCC/spectrogram features for detection in the first place). Aggressively
denoising before feature extraction risks removing the very signal the
classifier depends on, which is a plausible reason a detector could look
good on clean benchmark audio yet be fragile in general — a risk the base
paper's design does not discuss.

So: silence trimming + amplitude normalization are ON by default (they
affect segmentation/level, not fine spectral structure). Spectral-subtraction
denoising is implemented and available, but OFF by default, and is only
meant to be switched on deliberately for a specific, labeled ablation
(e.g. "does denoising help or hurt under noisy-condition evaluation?") —
never silently applied to the main matched/mismatched evaluation.
"""

from __future__ import annotations

import numpy as np
import librosa

DEFAULT_TARGET_SR = 16000


def load_audio(path: str, target_sr: int | None = DEFAULT_TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Load audio as a mono float32 waveform.

    If target_sr is given and differs from the file's native rate, resample.
    Resampling here is a data-representation step (making shapes consistent
    across a manifest that might mix sources), NOT a substitute for
    channel-degradation simulation — e.g. upsampling audio that has already
    been through a telephony bandpass filter does not restore the lost
    high-frequency content, it only changes the sample count. That
    distinction is made explicit here so it isn't confused with the
    degradation transforms in degradations.py.
    """
    waveform, sr = librosa.load(path, sr=None, mono=True)  # sr=None: keep native rate first
    if target_sr is not None and sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return waveform.astype(np.float32), sr


def trim_silence(
    waveform: np.ndarray,
    top_db: float = 30.0,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """
    Energy-based silence trimming.

    This is librosa's energy-threshold trim (`librosa.effects.trim`), which
    is an approximation of the base paper's "energy-based voice activity
    detection" — it trims leading/trailing low-energy regions using a dB
    threshold relative to the clip's peak, rather than a full frame-by-frame
    VAD that also removes *internal* silence gaps. This is a deliberate
    simplification: full internal-gap removal can distort word-level timing
    in a way that matters for the challenge-response content-matching step
    (built in a later module), so we only trim the edges here.
    """
    if waveform.size == 0:
        return waveform
    trimmed, _ = librosa.effects.trim(
        waveform, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return trimmed


def normalize_amplitude(waveform: np.ndarray, method: str = "peak", target_dbfs: float = -3.0) -> np.ndarray:
    """
    Normalize signal level.

    method="peak": scale so the maximum absolute sample reaches target_dbfs
        (converted from dBFS to linear). Simple, deterministic, and matches
        the base paper's stated "amplitude normalization" step most directly.
    method="rms": scale so the RMS level reaches target_dbfs. More
        perceptually consistent across clips of different dynamic range,
        at the cost of potential clipping on peaky signals — included as an
        option but not the default, to keep default behavior simple and
        predictable.
    """
    if waveform.size == 0:
        return waveform

    target_linear = 10.0 ** (target_dbfs / 20.0)

    if method == "peak":
        peak = np.max(np.abs(waveform))
        if peak < 1e-9:
            return waveform  # silence in, silence out — nothing to scale
        gain = target_linear / peak
    elif method == "rms":
        rms = np.sqrt(np.mean(np.square(waveform)))
        if rms < 1e-9:
            return waveform
        gain = target_linear / rms
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    return np.clip(waveform * gain, -1.0, 1.0).astype(np.float32)


def spectral_subtract_denoise(
    waveform: np.ndarray,
    sr: int,
    noise_estimate_frames: int = 6,
    n_fft: int = 512,
    hop_length: int = 160,
    oversubtraction: float = 1.0,
    spectral_floor: float = 0.02,
) -> np.ndarray:
    """
    Simple spectral-subtraction denoiser. OFF by default in the main
    pipeline (see module docstring) — provided for explicit, labeled
    ablation experiments only.

    Method: estimate a noise magnitude spectrum from the first
    `noise_estimate_frames` STFT frames (assumed non-speech / lead-in),
    subtract it (scaled by `oversubtraction`) from every frame's magnitude,
    floor at `spectral_floor` times the original magnitude to avoid musical-
    noise artifacts, and reconstruct with the original phase.

    This is a textbook-simple version of spectral subtraction, not a
    production-grade denoiser (e.g. no voice-activity-aware noise tracking,
    no smoothing across frames) — adequate for an ablation comparison, not
    claimed to be state-of-the-art.
    """
    if waveform.size <= n_fft:
        return waveform

    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)

    n_noise_frames = min(noise_estimate_frames, magnitude.shape[1])
    noise_mag = np.mean(magnitude[:, :n_noise_frames], axis=1, keepdims=True)

    subtracted = magnitude - oversubtraction * noise_mag
    floored = np.maximum(subtracted, spectral_floor * magnitude)

    denoised_stft = floored * np.exp(1j * phase)
    denoised = librosa.istft(denoised_stft, hop_length=hop_length, length=waveform.shape[0])
    return denoised.astype(np.float32)


def clean(
    waveform: np.ndarray,
    sr: int,
    trim: bool = True,
    normalize: bool = True,
    denoise: bool = False,
) -> np.ndarray:
    """
    Convenience wrapper applying the default cleaning pipeline in a fixed
    order: (optional denoise) -> trim -> normalize. Every step is optional
    and independently callable above; this function exists so the default
    pipeline used for the matched-condition experiment is defined in exactly
    one place.
    """
    out = waveform
    if denoise:
        out = spectral_subtract_denoise(out, sr)
    if trim:
        out = trim_silence(out)
    if normalize:
        out = normalize_amplitude(out)
    return out
