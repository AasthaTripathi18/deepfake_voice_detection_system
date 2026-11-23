"""
Audio capture. See src/streaming/__init__.py for the hard scope limit this
module enforces: microphone, file-replay, or an app-controlled stream —
nothing else, ever.

Every source exposes:
  .label          human-readable string identifying exactly what this
                   source is and, where relevant, which device/file/stream
                   name — meant to be displayed to the user verbatim,
                   never summarized away as just "audio input".
  .sample_rate    native sample rate this source delivers audio at.
  .read_chunk()   returns the next chunk as a 1D float32 numpy array, or
                   None when the source has ended (file exhausted, stream
                   closed, capture stopped).
"""

from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class AudioSource(abc.ABC):
    label: str
    sample_rate: int

    @abc.abstractmethod
    def read_chunk(self) -> Optional[np.ndarray]:
        """Return the next float32 mono chunk, or None if the source ended."""

    def close(self) -> None:
        """Optional cleanup hook; default no-op."""


class MicrophoneSource(AudioSource):
    """
    Local microphone capture, for a controlled demo — the person runs the
    app and speaks into their own device's mic. Requires the optional
    `sounddevice` package (not part of this project's core requirements.txt
    since only the live-demo path needs it).
    """

    def __init__(self, chunk_sec: float = 0.25, sample_rate: int = 16000, device: str | int | None = None):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "MicrophoneSource requires the 'sounddevice' package "
                "(`pip install sounddevice`), which is optional and not in "
                "requirements.txt since it's only needed for live mic capture."
            ) from exc
        except OSError as exc:
            # sounddevice can be installed but still fail here if its native
            # PortAudio backend isn't available (common on headless/minimal
            # Linux setups, including this project's own sandboxed dev
            # environment) — surface a clear, actionable message instead of
            # letting a raw OSError propagate.
            raise RuntimeError(
                "The 'sounddevice' package is installed, but its native PortAudio "
                "backend could not be loaded (this is common on headless machines "
                "or minimal containers). Install PortAudio for your OS, e.g. "
                "`sudo apt-get install libportaudio2` on Debian/Ubuntu, "
                "`brew install portaudio` on macOS, then retry. "
                f"Original error: {exc}"
            ) from exc

        self._sd = sd
        self.sample_rate = sample_rate
        self._chunk_samples = int(round(chunk_sec * sample_rate))
        self._device = device

        device_info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
        device_name = device_info.get("name", "default") if isinstance(device_info, dict) else "default"
        self.label = f"microphone (device: {device_name})"

        self._stream = sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="float32",
            blocksize=self._chunk_samples, device=device,
        )
        self._stream.start()

    def read_chunk(self) -> Optional[np.ndarray]:
        data, _overflowed = self._stream.read(self._chunk_samples)
        return data[:, 0].astype(np.float32)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class FileReplaySource(AudioSource):
    """
    Replays an audio file in fixed-size chunks as if it were a live stream
    — for testing and demoing the streaming pipeline without a microphone.
    This is still "audio supplied to the app" (a file), just fed through
    the streaming path instead of scored as one whole clip.

    Two ways to construct it:
      FileReplaySource(path="some.wav", ...)   — loads via
          src/preprocessing/cleaning.py (requires librosa/soundfile).
      FileReplaySource(waveform=array, sample_rate=16000, ...) — takes an
          already-loaded waveform directly, so unit tests (and callers who
          already have audio in memory) don't need the audio-I/O stack.

    real_time=True sleeps between chunks to approximate real capture
    pacing (useful for an interactive demo); real_time=False (default)
    replays as fast as possible, which is what a test/batch run wants.
    """

    def __init__(
        self,
        path: str | None = None,
        waveform: np.ndarray | None = None,
        sample_rate: int | None = None,
        chunk_sec: float = 0.25,
        real_time: bool = False,
    ):
        if (path is None) == (waveform is None):
            raise ValueError("Provide exactly one of `path` or `waveform`.")

        if path is not None:
            from src.preprocessing import cleaning  # lazy: needs librosa/soundfile
            waveform, sample_rate = cleaning.load_audio(path, target_sr=sample_rate)
            self.label = f"file replay ({Path(path).name})"
        else:
            if sample_rate is None:
                raise ValueError("sample_rate is required when passing `waveform` directly.")
            self.label = "file replay (in-memory waveform)"

        self._waveform = np.asarray(waveform, dtype=np.float32)
        self.sample_rate = sample_rate
        self._chunk_samples = max(1, int(round(chunk_sec * sample_rate)))
        self._chunk_sec = chunk_sec
        self._real_time = real_time
        self._position = 0

    def read_chunk(self) -> Optional[np.ndarray]:
        if self._position >= len(self._waveform):
            return None
        end = min(self._position + self._chunk_samples, len(self._waveform))
        chunk = self._waveform[self._position:end]
        self._position = end
        if self._real_time:
            time.sleep(self._chunk_sec)
        return chunk.astype(np.float32)


class AppControlledStreamSource(AudioSource):
    """
    Wraps audio the APPLICATION'S OWN CODE already receives from a stream
    it controls — e.g. frames handed to the app by a WebRTC or SIP client
    library integrated into the app itself (the "self-controlled VoIP
    leg" named as a possible future extension in the project architecture).

    This class does NOT open any connection, does NOT know how to reach a
    telephony network, and does NOT and cannot access cellular-call audio.
    It only wraps a callable the caller supplies — `chunk_provider` — which
    the caller's own integration code must implement to pull the next
    chunk from whatever stream it is already receiving. If a caller passes
    a chunk_provider that reaches into a cellular call, that is entirely
    outside this class and outside what this project supports or condones.
    """

    def __init__(
        self,
        chunk_provider: Callable[[], Optional[np.ndarray]],
        sample_rate: int,
        source_name: str,
    ):
        self._chunk_provider = chunk_provider
        self.sample_rate = sample_rate
        self.label = f"app-controlled stream (source: {source_name})"

    def read_chunk(self) -> Optional[np.ndarray]:
        chunk = self._chunk_provider()
        if chunk is None:
            return None
        return np.asarray(chunk, dtype=np.float32)
