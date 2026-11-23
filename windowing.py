"""
Sliding-window buffering over a stream of audio chunks. Pure numpy
arithmetic — no audio I/O, no model access — so it's fully unit-testable
without any audio library installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    waveform: np.ndarray
    start_sample: int
    end_sample: int
    sample_rate: int
    is_full: bool  # False for a short trailing window emitted at stream end

    @property
    def start_sec(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_sec(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def duration_sec(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


class SlidingWindowBuffer:
    """
    push(chunk) accepts newly-arrived audio and returns zero or more fully
    formed Windows (each exactly `window_sec` long) as soon as enough
    samples have accumulated to complete them.

    flush() is called once the source ends, and returns a final SHORT
    window (is_full=False) covering whatever audio is left over that
    never reached a full window's length — this is the "short sample"
    case: it is emitted, flagged as partial, and left for
    quality_gate.py to decide whether it's long enough to analyze, rather
    than being silently discarded or silently padded and presented as if
    it were a normal full-length window.
    """

    def __init__(self, sample_rate: int, window_sec: float, hop_sec: float):
        if window_sec <= 0 or hop_sec <= 0:
            raise ValueError("window_sec and hop_sec must both be positive.")
        if hop_sec > window_sec:
            raise ValueError(
                f"hop_sec ({hop_sec}) > window_sec ({window_sec}) would skip audio "
                f"entirely between windows — not supported, since every sample should "
                f"fall inside at least one analysis window."
            )

        self.sample_rate = sample_rate
        self.window_samples = int(round(window_sec * sample_rate))
        self.hop_samples = int(round(hop_sec * sample_rate))
        if self.window_samples <= 0 or self.hop_samples <= 0:
            raise ValueError("window_sec/hop_sec too small for the given sample_rate.")

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_index = 0   # sample index that self._buffer[0] corresponds to
        self._total_pushed = 0
        self._next_window_start = 0
        self._flushed = False

    def push(self, chunk: np.ndarray) -> list[Window]:
        if self._flushed:
            raise RuntimeError("push() called after flush(); this buffer is finished.")
        chunk = np.asarray(chunk, dtype=np.float32)
        self._buffer = np.concatenate([self._buffer, chunk])
        self._total_pushed += len(chunk)

        windows = []
        while self._total_pushed - self._next_window_start >= self.window_samples:
            start = self._next_window_start
            end = start + self.window_samples
            rel_start = start - self._buffer_start_index
            rel_end = end - self._buffer_start_index
            waveform = self._buffer[rel_start:rel_end].copy()
            windows.append(Window(
                waveform=waveform, start_sample=start, end_sample=end,
                sample_rate=self.sample_rate, is_full=True,
            ))
            self._next_window_start += self.hop_samples

        self._trim()
        return windows

    def flush(self) -> Window | None:
        """Call exactly once, after the source has ended, to retrieve any
        leftover audio shorter than a full window. Returns None if there's
        nothing left (e.g. the last full window ended exactly at the
        stream's end)."""
        self._flushed = True
        start = self._next_window_start
        remaining = self._total_pushed - start
        if remaining <= 0:
            return None
        rel_start = start - self._buffer_start_index
        waveform = self._buffer[rel_start:rel_start + remaining].copy()
        window = Window(
            waveform=waveform, start_sample=start, end_sample=start + remaining,
            sample_rate=self.sample_rate, is_full=False,
        )
        self._next_window_start = self._total_pushed
        return window

    def _trim(self) -> None:
        """Discard buffered samples that no future window will ever need
        (everything strictly before the next window's start), so memory
        use stays bounded for an indefinitely long stream."""
        discard_upto = self._next_window_start
        rel_discard = discard_upto - self._buffer_start_index
        if rel_discard > 0:
            self._buffer = self._buffer[rel_discard:]
            self._buffer_start_index = discard_upto
