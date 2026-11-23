"""
Degradation registry — named, parameterized, pure transforms simulating
phone-channel conditions. Each transform takes (waveform, sr, **params) and
returns a NEW waveform; none mutate their input, and none touch the
manifest or filesystem except codec_transcode, which uses a private
temp-file round trip through `ffmpeg` (unavoidable, since Python has no
reliable pure-code AMR/GSM codec).

These are approximations of real telephony conditions, not captures of
real calls (see project plan Section 6 for why that distinction matters
and what it limits about the conclusions this project can draw).

Every transform is independently unit-testable in isolation
(tests/test_degradation_transforms.py) precisely because each is a pure
function with no hidden state.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


class DegradationError(RuntimeError):
    """Raised when a degradation cannot be applied (e.g. ffmpeg/codec missing).

    Callers (the batch feature-building script) are expected to catch this,
    log it as a skipped condition for that file, and continue — never to
    let a missing codec silently fall back to "clean" and mislabel results.
    """


# --- 1. Telephony bandpass ---------------------------------------------------

def bandpass_telephony(
    waveform: np.ndarray,
    sr: int,
    low_hz: float = 300.0,
    high_hz: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """
    Approximate standard analog telephony bandwidth (300-3400 Hz) with a
    zero-phase Butterworth bandpass filter.

    Tradeoff: this captures the bandwidth restriction of a phone channel
    but NOT its compression artifacts (quantization, codec-specific
    distortion) — codec_transcode below covers that separately. Applying
    both in sequence is a reasonable "worst case" phone-channel simulation;
    applying bandpass alone isolates the bandwidth-restriction effect on its
    own, which is useful for understanding which part of channel
    degradation actually drives any accuracy drop.
    """
    nyquist = sr / 2.0
    if high_hz >= nyquist:
        raise ValueError(
            f"high_hz={high_hz} must be below Nyquist ({nyquist} Hz for sr={sr}); "
            f"check that sr matches the expected rate before filtering."
        )
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, waveform).astype(np.float32)


# --- 2. Codec transcode (round-trip through ffmpeg) -------------------------

# Codec choice note: AMR-NB is the most "authentically cellular" codec but
# is not bundled in every ffmpeg build due to licensing (requires
# --enable-libvo-amrwbenc or similar at compile time). GSM 06.10 ("gsm") is
# widely available in standard ffmpeg builds and is also a real, historically
# deployed cellular codec, so it is used as the DEFAULT here; AMR-NB is
# offered as an option that will raise a clear DegradationError (not a
# silent fallback) if the local ffmpeg build lacks it.
_CODEC_FFMPEG_ARGS = {
    "gsm": ["-ar", "8000", "-ac", "1", "-c:a", "gsm"],
    "amr_nb": ["-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"],
}
_CODEC_CONTAINER_EXT = {
    "gsm": ".wav",   # ffmpeg can mux gsm inside a wav container
    "amr_nb": ".amr",
}


def codec_transcode(waveform: np.ndarray, sr: int, codec: str = "gsm") -> np.ndarray:
    """
    Encode waveform through a lossy telephony codec and decode it back,
    simulating the compression artifacts of a real phone/VoIP channel
    (as opposed to bandpass_telephony's bandwidth-only restriction).

    Requires the `ffmpeg` binary on PATH. Raises DegradationError (not a
    silent skip) if ffmpeg is missing or the requested codec isn't
    supported by the local ffmpeg build — the caller must handle this
    explicitly so a missing codec never gets recorded as a successful
    "codec_gsm" condition when it was actually never applied.
    """
    if codec not in _CODEC_FFMPEG_ARGS:
        raise ValueError(f"Unknown codec {codec!r}; expected one of {list(_CODEC_FFMPEG_ARGS)}")

    if shutil.which("ffmpeg") is None:
        raise DegradationError(
            "ffmpeg not found on PATH. Install it (see setup commands) before "
            "running codec-based degradations; this file is being skipped, "
            "not silently treated as clean."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "in.wav"
        tmp_encoded = Path(tmpdir) / f"encoded{_CODEC_CONTAINER_EXT[codec]}"
        tmp_out = Path(tmpdir) / "out.wav"

        sf.write(tmp_in, waveform, sr)

        encode_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_in),
                      *_CODEC_FFMPEG_ARGS[codec], str(tmp_encoded)]
        decode_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_encoded),
                      "-ar", str(sr), "-ac", "1", str(tmp_out)]

        for cmd in (encode_cmd, decode_cmd):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise DegradationError(
                    f"ffmpeg command failed for codec={codec!r}: {' '.join(cmd)}\n"
                    f"stderr: {result.stderr.strip()}"
                )

        out_waveform, _ = sf.read(str(tmp_out), dtype="float32", always_2d=False)
        return out_waveform


# --- 3. Additive noise at target SNR ----------------------------------------

def add_noise_at_snr(
    waveform: np.ndarray,
    snr_db: float,
    noise: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add noise at a specified signal-to-noise ratio.

    If `noise` is not provided, synthetic white Gaussian noise is generated.
    Tradeoff: white noise is a simplification — real-world background noise
    (traffic, crowd, HVAC) has structured spectral content the base paper's
    MUSAN reference (cited in Related Work) is meant to provide. We do not
    bundle MUSAN here (it is a separate, large external download this
    project does not assume network access to fetch automatically); if a
    MUSAN-derived noise clip is available locally, pass it as `noise` and
    this function will use it instead of synthesizing white noise. This is
    flagged explicitly so a reader knows which noise condition a given
    result actually used.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.square(waveform))
    if signal_power < 1e-12:
        return waveform  # near-silent input; adding noise at a target SNR is undefined

    if noise is None:
        noise = rng.normal(0.0, 1.0, size=waveform.shape[0]).astype(np.float32)
    else:
        noise = _match_length(noise, waveform.shape[0], rng)

    noise_power = np.mean(np.square(noise))
    if noise_power < 1e-12:
        raise ValueError("Provided noise clip is (near) silent; cannot scale to a target SNR.")

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = waveform + scale * noise
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def _match_length(noise: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    if noise.shape[0] >= target_len:
        start = int(rng.integers(0, noise.shape[0] - target_len + 1))
        return noise[start:start + target_len]
    reps = int(np.ceil(target_len / noise.shape[0]))
    return np.tile(noise, reps)[:target_len]


# --- Registry ----------------------------------------------------------------

REGISTRY = {
    "telephony_bandpass": bandpass_telephony,
    "codec_gsm": lambda waveform, sr, **kw: codec_transcode(waveform, sr, codec="gsm", **kw),
    "codec_amr_nb": lambda waveform, sr, **kw: codec_transcode(waveform, sr, codec="amr_nb", **kw),
    "noise_snr": add_noise_at_snr,
}


def list_conditions() -> list[str]:
    return sorted(REGISTRY.keys())


def apply_condition(waveform: np.ndarray, sr: int, condition_name: str, **params) -> np.ndarray:
    """
    Dispatch to a registered degradation by name. Raises KeyError for an
    unregistered name (fail loudly rather than silently no-op) and lets
    DegradationError/ValueError from the underlying transform propagate,
    since the batch-building script is responsible for catching and
    logging those per-file, not this dispatcher.
    """
    if condition_name not in REGISTRY:
        raise KeyError(f"Unregistered condition {condition_name!r}; known: {list_conditions()}")
    fn = REGISTRY[condition_name]
    if condition_name == "noise_snr":
        return fn(waveform, **{"snr_db": params.pop("snr_db", 10.0), **params})
    return fn(waveform, sr, **params)
