"""
Synthesizes a short alert beep as WAV bytes, using only the Python
standard library's `wave` module plus numpy (both already available) —
no extra audio-file dependency for this small cosmetic feature. This is
the DEFAULT alert sound rendered in the Streamlit app for High-risk
results; a real deployment is free to substitute its own sound asset.

Pure interface-layer utility: no decision logic, no model access.
"""

from __future__ import annotations

import io
import wave

import numpy as np


def generate_beep_wav_bytes(
    freq_hz: float = 880.0,
    duration_sec: float = 0.3,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> bytes:
    if not (0.0 < amplitude <= 1.0):
        raise ValueError("amplitude must be in (0, 1].")

    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * freq_hz * t)

    # Short fade-out to avoid an audible click at the end of the clip.
    fade_len = min(int(0.05 * sample_rate), n_samples)
    if fade_len > 0:
        tone[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

    pcm16 = (tone * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buffer.getvalue()
