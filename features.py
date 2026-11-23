"""
Feature extraction.

Which of the base paper's three acoustic feature ideas (MFCC, pitch, and
spectrogram) we actually use, and how, is a deliberate choice made against
this project's specific research question (RQ1: how much does phone-channel
degradation hurt a passively-trained detector, and does augmentation help),
not a blanket reuse of the base paper's feature-fusion approach. The
tradeoffs below are the reasoning for that choice.

  MFCC (40 coefficients, 512-pt FFT, 25 ms / 10 ms framing) — KEPT, used as
  the primary feature for the classical baseline model (mirrors the base
  paper's stated configuration exactly, which keeps the matched-condition
  comparison to the base paper's numbers methodologically aligned).

  Log-mel spectrogram (128 mel bins, 0-8000 Hz) — KEPT, used as the input
  representation for the CNN-style model, since a 2D time-frequency map
  lets a convolutional model learn its own discriminative filters rather
  than relying on hand-pooled statistics; this is the representation most
  directly affected by bandpass/codec degradation, which is exactly what
  RQ1 needs to be able to see.

  Pitch / F0 (YIN) — IMPLEMENTED BUT NOT part of the default feature
  vector for either model, by design:
    1. F0 estimation is itself degraded by exactly the same channel
       conditions this project varies (telephony bandpass and codec
       transcoding distort low-frequency harmonic structure and introduce
       estimation artifacts), so under the mismatched-condition arm of
       RQ1, a pitch feature would partly measure "how noisy is our F0
       estimator under this condition," conflating a measurement artifact
       with the phenomenon under study.
    2. The base paper does not ablate pitch's individual contribution
       (Section VII.D flags this as unexamined even in the base paper's
       own results), so there is no established basis here for assuming
       it helps enough to justify the added fragility above.
  Pitch extraction is kept available as a function for optional exploratory
  analysis or a clearly-labeled future ablation, but is not silently baked
  into the main experiment's feature vectors. This keeps the default
  pipeline simpler and keeps its failure modes attributable to the
  representations actually being evaluated.
"""

from __future__ import annotations

import functools
import numpy as np
import librosa
from scipy import signal, fftpack

# Matches the base paper's stated MFCC configuration.
MFCC_N_MFCC = 40
MFCC_N_FFT = 512
MFCC_HOP_LENGTH = 160   # 10 ms at 16 kHz
MFCC_WIN_LENGTH = 400   # 25 ms at 16 kHz

# LFCC configuration (canonical ASVspoof 2019 baseline)
LFCC_N_LFCC = 40
LFCC_N_FILTERS = 70

MEL_N_MELS = 128
MEL_FMAX = 8000

# Fixed spectrogram time-length (frames) so batches have a consistent shape
# for the CNN. ~2.0 s of audio at the hop length above.
SPECTROGRAM_FIXED_FRAMES = 200


def extract_mfcc_stats(
    waveform: np.ndarray,
    sr: int,
    n_mfcc: int = MFCC_N_MFCC,
    n_fft: int = MFCC_N_FFT,
    hop_length: int = MFCC_HOP_LENGTH,
    win_length: int = MFCC_WIN_LENGTH,
) -> np.ndarray:
    """
    Extract frame-level MFCCs, then pool over time with mean and std,
    returning a fixed-length vector of size 2 * n_mfcc.

    Mean/std pooling is a deliberate simplicity choice for the classical
    (SVM) baseline: it collapses variable-length audio into a fixed-size
    vector without needing a sequence model, at the cost of discarding
    temporal ordering within the clip. That cost is acceptable for a
    baseline meant for direct comparability with the base paper's SVM
    result, not for capturing every nuance MFCCs could offer.
    """
    if waveform.size < n_fft:
        # Pad very short clips so librosa doesn't error; extremely short
        # clips should mostly have been filtered out by the integrity
        # checker's min-duration threshold already.
        waveform = np.pad(waveform, (0, n_fft - waveform.size))

    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length,
    )  # shape: (n_mfcc, n_frames)

    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


def extract_mfcc_deltas_stats(
    waveform: np.ndarray,
    sr: int,
    n_mfcc: int = MFCC_N_MFCC,
    n_fft: int = MFCC_N_FFT,
    hop_length: int = MFCC_HOP_LENGTH,
    win_length: int = MFCC_WIN_LENGTH,
    apply_cmvn: bool = False,
) -> np.ndarray:
    """
    Extract 40 static MFCCs plus first-order delta (velocity) and second-order
    delta-delta (acceleration) coefficients, pooled over time with mean and std.
    Returns a fixed-length vector of size 6 * n_mfcc = 240 dimensions.

    When apply_cmvn=True, applies frame-wise Cepstral Mean and Variance Normalization
    to reduce convolutional channel distortion and microphone frequency response mismatch.
    """
    if waveform.size < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.size))

    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length,
    )  # shape: (n_mfcc, n_frames)

    if apply_cmvn and mfcc.shape[1] > 1:
        mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
        mfcc_std = np.std(mfcc, axis=1, keepdims=True)
        mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)

    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    feats = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(d1, axis=1),
        np.std(d1, axis=1),
        np.mean(d2, axis=1),
        np.std(d2, axis=1),
    ])
    return feats.astype(np.float32)


@functools.lru_cache(maxsize=8)
def linear_filter_bank(
    n_filters: int = LFCC_N_FILTERS,
    n_fft: int = MFCC_N_FFT,
    sr: int = 16000,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Construct a linear-frequency triangular filterbank spanning [fmin, fmax] Hz.
    In contrast to mel-filterbanks that compress high frequencies logarithmically,
    LFCC maintains uniform linear resolution up to Nyquist, capturing spectral slope
    and high-frequency vocoder artifacts critical for spoof detection.
    """
    if fmax is None or fmax > sr / 2.0:
        fmax = sr / 2.0
    freq_bins = np.linspace(fmin, fmax, n_filters + 2)
    fft_bins = np.floor((n_fft + 1) * freq_bins / sr).astype(int)
    n_bins = n_fft // 2 + 1
    bank = np.zeros((n_filters, n_bins), dtype=np.float32)
    for m in range(1, n_filters + 1):
        f_left = fft_bins[m - 1]
        f_center = fft_bins[m]
        f_right = fft_bins[m + 1]
        for k in range(f_left, f_center):
            if f_center > f_left:
                bank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center and k < n_bins:
                bank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return bank


def _compute_deltas(feat: np.ndarray, width: int = 9) -> np.ndarray:
    """Fast vectorized 2D regression delta matching librosa.feature.delta."""
    if feat.shape[1] < 2:
        return np.zeros_like(feat)
    half = width // 2
    padded = np.pad(feat, ((0, 0), (half, half)), mode="edge")
    kernel = np.arange(-half, half + 1)
    denom = 2 * sum(i**2 for i in range(1, half + 1))
    delta = signal.convolve2d(padded, kernel[np.newaxis, ::-1], mode="valid") / denom
    return delta.astype(np.float32)


def extract_lfcc_stats(
    waveform: np.ndarray,
    sr: int,
    n_lfcc: int = LFCC_N_LFCC,
    n_filters: int = LFCC_N_FILTERS,
    n_fft: int = MFCC_N_FFT,
    hop_length: int = MFCC_HOP_LENGTH,
    win_length: int = MFCC_WIN_LENGTH,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Extract frame-level LFCCs, then pool over time with mean and std,
    returning a fixed-length vector of size 2 * n_lfcc = 80 dimensions.
    """
    if waveform.size < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.size))

    f, t, Zxx = signal.stft(
        waveform,
        fs=sr,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=True,
    )
    power_spec = (np.abs(Zxx) ** 2).astype(np.float32)
    bank = linear_filter_bank(n_filters=n_filters, n_fft=n_fft, sr=sr, fmin=fmin, fmax=fmax)
    filt_energies = np.dot(bank, power_spec)
    log_energies = 10.0 * np.log10(filt_energies + 1e-10)

    lfcc = fftpack.dct(log_energies, type=2, axis=0, norm="ortho")[:n_lfcc, :]
    mean = np.mean(lfcc, axis=1)
    std = np.std(lfcc, axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


def extract_lfcc_deltas_stats(
    waveform: np.ndarray,
    sr: int,
    n_lfcc: int = LFCC_N_LFCC,
    n_filters: int = LFCC_N_FILTERS,
    n_fft: int = MFCC_N_FFT,
    hop_length: int = MFCC_HOP_LENGTH,
    win_length: int = MFCC_WIN_LENGTH,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    apply_cmvn: bool = False,
) -> np.ndarray:
    """
    Extract Linear Frequency Cepstral Coefficients (LFCC) plus first-order
    delta (velocity) and second-order delta-delta (acceleration) coefficients,
    pooled over time with mean and standard deviation.
    Returns a fixed-length vector of size 6 * n_lfcc = 240 dimensions.
    """
    if waveform.size < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.size))

    f, t, Zxx = signal.stft(
        waveform,
        fs=sr,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=True,
    )
    power_spec = (np.abs(Zxx) ** 2).astype(np.float32)
    bank = linear_filter_bank(n_filters=n_filters, n_fft=n_fft, sr=sr, fmin=fmin, fmax=fmax)
    filt_energies = np.dot(bank, power_spec)
    log_energies = 10.0 * np.log10(filt_energies + 1e-10)

    lfcc = fftpack.dct(log_energies, type=2, axis=0, norm="ortho")[:n_lfcc, :]

    if apply_cmvn and lfcc.shape[1] > 1:
        lfcc_mean = np.mean(lfcc, axis=1, keepdims=True)
        lfcc_std = np.std(lfcc, axis=1, keepdims=True)
        lfcc = (lfcc - lfcc_mean) / (lfcc_std + 1e-8)

    d1 = _compute_deltas(lfcc)
    d2 = _compute_deltas(d1)

    feats = np.concatenate([
        np.mean(lfcc, axis=1),
        np.std(lfcc, axis=1),
        np.mean(d1, axis=1),
        np.std(d1, axis=1),
        np.mean(d2, axis=1),
        np.std(d2, axis=1),
    ])
    return feats.astype(np.float32)


def extract_lfcc_mfcc_fusion(
    waveform: np.ndarray,
    sr: int,
    apply_cmvn: bool = False,
) -> np.ndarray:
    """
    Feature-level acoustic fusion: Concatenates 240-dim LFCC deltas+stats and
    240-dim MFCC deltas+stats into a 480-dim joint acoustic representation.
    Provides complementary sensitivity to both high-frequency vocoder phase/smear
    cues (LFCC) and lower-frequency vocal-tract spectral shape (MFCC).
    """
    lfcc = extract_lfcc_deltas_stats(waveform, sr, apply_cmvn=apply_cmvn)
    mfcc = extract_mfcc_deltas_stats(waveform, sr, apply_cmvn=apply_cmvn)
    return np.concatenate([lfcc, mfcc]).astype(np.float32)


def extract_lfcc_spectrogram_2d(
    waveform: np.ndarray,
    sr: int = 16000,
    target_frames: int = 400,
    n_filters: int = 60,
    n_fft: int = MFCC_N_FFT,
    hop_length: int = MFCC_HOP_LENGTH,
    win_length: int = MFCC_WIN_LENGTH,
) -> np.ndarray:
    """
    Extract (1, n_filters, target_frames) LFCC 2D time-frequency spectrogram for LCNN.
    Wraps/pads or truncates time frames so every clip has exactly target_frames (~4.0 sec).
    """
    if waveform.size < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.size))

    f, t, Zxx = signal.stft(
        waveform,
        fs=sr,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=True,
    )
    power_spec = (np.abs(Zxx) ** 2).astype(np.float32)
    bank = linear_filter_bank(n_filters=n_filters, n_fft=n_fft, sr=sr, fmin=0.0, fmax=8000.0)
    filt_energies = np.dot(bank, power_spec)
    log_energies = 10.0 * np.log10(filt_energies + 1e-10)
    lfcc_2d = fftpack.dct(log_energies, type=2, axis=0, norm="ortho")

    n_frames = lfcc_2d.shape[1]
    if n_frames < target_frames:
        pad_width = target_frames - n_frames
        lfcc_2d = np.pad(lfcc_2d, ((0, 0), (0, pad_width)), mode="wrap")
    elif n_frames > target_frames:
        lfcc_2d = lfcc_2d[:, :target_frames]

    return lfcc_2d[np.newaxis, :, :].astype(np.float32)


def extract_logmel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_mels: int = MEL_N_MELS,
    fmax: int = MEL_FMAX,
    n_fft: int = MFCC_N_FFT,
    hop_length: int = MFCC_HOP_LENGTH,
    fixed_frames: int = SPECTROGRAM_FIXED_FRAMES,
) -> np.ndarray:
    """
    Extract a log-mel spectrogram and pad/truncate to a fixed number of
    time frames, returning shape (n_mels, fixed_frames).

    fmax=8000 matches the base paper's stated mel-filter range. Note that
    under telephony bandpass degradation (cutoff ~3400 Hz), the upper
    portion of this range will contain little or no energy by
    construction — that is an expected and informative effect of the
    degradation, not a bug in the extractor, and is worth checking
    visually (e.g. plotting a clean vs. degraded spectrogram side by side)
    before drawing conclusions from downstream model behavior.
    """
    if waveform.size < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.size))

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, fmax=fmax,
        n_fft=n_fft, hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    n_frames = log_mel.shape[1]
    if n_frames < fixed_frames:
        pad_width = fixed_frames - n_frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=log_mel.min())
    elif n_frames > fixed_frames:
        log_mel = log_mel[:, :fixed_frames]

    return log_mel


def extract_pitch_stats(
    waveform: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 500.0,
    frame_length: int = 1024,
) -> dict:
    """
    OPTIONAL / diagnostic-only pitch summary (see module docstring for why
    this is not in the default feature vector). Uses librosa's YIN
    implementation. Returns a small dict rather than a bare array so
    exploratory notebooks can inspect it without guessing field order:

        {"f0_mean_hz": ..., "f0_std_hz": ..., "voiced_fraction": ...}

    voiced_fraction is the proportion of frames where YIN's estimate falls
    strictly inside (fmin, fmax) rather than pinned at a boundary — a rough
    proxy for how much of the clip had a confidently estimated pitch at all,
    which is itself a useful diagnostic for how much a given degradation
    condition has damaged pitch-trackability.
    """
    if waveform.size < frame_length:
        return {"f0_mean_hz": float("nan"), "f0_std_hz": float("nan"), "voiced_fraction": 0.0}

    f0 = librosa.yin(waveform, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length)
    voiced_mask = (f0 > fmin + 1e-6) & (f0 < fmax - 1e-6)
    voiced_fraction = float(np.mean(voiced_mask)) if f0.size else 0.0

    if not np.any(voiced_mask):
        return {"f0_mean_hz": float("nan"), "f0_std_hz": float("nan"), "voiced_fraction": voiced_fraction}

    voiced_f0 = f0[voiced_mask]
    return {
        "f0_mean_hz": float(np.mean(voiced_f0)),
        "f0_std_hz": float(np.std(voiced_f0)),
        "voiced_fraction": voiced_fraction,
    }


def build_feature_record(waveform: np.ndarray, sr: int, feature_set: str = "both") -> dict:
    """
    Convenience entry point used by the batch feature-building script and inference pipelines.

    feature_set:
      "mfcc_stats"        -> classical-baseline feature (80 dims)
      "mfcc_deltas_stats" -> 240-dim dynamic MFCC features
      "lfcc_stats"        -> 80-dim static LFCC features
      "lfcc_deltas_stats" -> 240-dim dynamic LFCC features
      "lfcc_mfcc_fusion"  -> 480-dim joint LFCC + MFCC features
      "logmel"            -> 2D time-frequency log-mel spectrogram for CNN
      "both"              -> mfcc_stats + logmel
      "all"               -> compute all acoustic representations
    """
    record: dict = {}
    if feature_set in ("mfcc_stats", "both", "all"):
        record["mfcc_stats"] = extract_mfcc_stats(waveform, sr)
    if feature_set in ("mfcc_deltas_stats", "all"):
        record["mfcc_deltas_stats"] = extract_mfcc_deltas_stats(waveform, sr)
    if feature_set in ("lfcc_stats", "all"):
        record["lfcc_stats"] = extract_lfcc_stats(waveform, sr)
    if feature_set in ("lfcc_deltas_stats", "all"):
        record["lfcc_deltas_stats"] = extract_lfcc_deltas_stats(waveform, sr)
    if feature_set in ("lfcc_mfcc_fusion", "all"):
        record["lfcc_mfcc_fusion"] = extract_lfcc_mfcc_fusion(waveform, sr)
    if feature_set in ("lfcc_spectrogram_2d", "all"):
        record["lfcc_spectrogram_2d"] = extract_lfcc_spectrogram_2d(waveform, sr)
    if feature_set in ("logmel", "both", "all"):
        record["logmel"] = extract_logmel_spectrogram(waveform, sr)
    if not record:
        raise ValueError(
            f"Unknown feature_set {feature_set!r}; expected mfcc_stats, mfcc_deltas_stats, "
            f"lfcc_stats, lfcc_deltas_stats, lfcc_mfcc_fusion, lfcc_spectrogram_2d, logmel, both, or all"
        )
    return record
