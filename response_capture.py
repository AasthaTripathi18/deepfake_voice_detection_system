"""
Captures ONE bounded-duration response clip from an
src.streaming.audio_source.AudioSource — distinct from the continuous
SlidingWindowBuffer used for passive streaming analysis, since a challenge
response is a single, time-boxed "listen for the answer" event, not an
ongoing stream of overlapping windows.

Reuses AudioSource as-is (no new source types): whatever is passed in —
microphone, file replay, or an app-controlled stream — is used exactly as
already defined and labeled in src/streaming/audio_source.py, so a
challenge response's audio source is visible to the user through the same
`.label` mechanism as the passive streaming mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.streaming.audio_source import AudioSource


@dataclass
class CapturedResponse:
    waveform: np.ndarray
    sample_rate: int
    duration_sec: float
    source_label: str
    ended_early: bool   # True if the source ran out of audio before max_duration_sec was reached


def capture_response(source: AudioSource, max_duration_sec: float = 6.0) -> CapturedResponse:
    """
    Reads chunks from `source` until either max_duration_sec of audio has
    been collected or the source signals it has no more audio
    (read_chunk() returns None — e.g. a replayed file ending, or a caller-
    supplied stream closing). Does NOT loop forever waiting for silence to
    end on its own; the caller decides how long to listen by setting
    max_duration_sec, which keeps this function's behavior identical
    regardless of source type.
    """
    if max_duration_sec <= 0:
        raise ValueError("max_duration_sec must be positive.")

    sample_rate = source.sample_rate
    max_samples = int(round(max_duration_sec * sample_rate))

    collected = []
    total_samples = 0
    ended_early = False

    while total_samples < max_samples:
        chunk = source.read_chunk()
        if chunk is None:
            ended_early = True
            break
        collected.append(chunk)
        total_samples += len(chunk)

    if collected:
        waveform = np.concatenate(collected)[:max_samples]
    else:
        waveform = np.zeros(0, dtype=np.float32)

    return CapturedResponse(
        waveform=waveform.astype(np.float32),
        sample_rate=sample_rate,
        duration_sec=len(waveform) / sample_rate,
        source_label=source.label,
        ended_early=ended_early,
    )
