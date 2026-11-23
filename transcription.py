"""
Speech-to-text abstraction for verifying WHAT was said in a challenge
response. This project does not bundle a real ASR engine (no dependency
in requirements.txt) — plugging one in is a deliberate, explicit step,
same pattern as the optional LCNN (needs torch) and MicrophoneSource
(needs sounddevice) integrations elsewhere in this project.

Every transcriber returns a TranscriptionResult with a `status` field, so
"the ASR engine failed" or "no engine is configured" is a distinguishable,
handleable outcome for session.py — never a silent empty string that gets
treated the same as "the caller genuinely said nothing recognizable."
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptionResult:
    text: str            # best-effort transcript, "" if none available
    status: str           # "ok" | "backend_unavailable" | "transcription_failed"
    raw_backend_info: dict = None  # optional backend-specific detail (e.g. confidence), for logging only


class Transcriber(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> TranscriptionResult:
        ...


class NullTranscriber(Transcriber):
    """
    Default when no real ASR backend is configured. Does NOT silently
    return an empty transcript that would look identical to "caller said
    nothing" — it returns status="backend_unavailable" so session.py can
    report the content-match step as inconclusive-because-unconfigured,
    a materially different situation from inconclusive-because-inaudible.
    """

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> TranscriptionResult:
        return TranscriptionResult(
            text="", status="backend_unavailable",
            raw_backend_info={
                "note": "No speech-to-text backend is configured. Pass a real Transcriber "
                         "(e.g. SpeechRecognitionTranscriber below, or your own) to "
                         "ChallengeResponseSession — this project does not bundle an ASR engine."
            },
        )


class SpeechRecognitionTranscriber(Transcriber):
    """
    OPTIONAL backend using the third-party `speech_recognition` package
    (`pip install SpeechRecognition`; not in this project's requirements.txt
    since only this optional path needs it), which itself wraps an
    underlying engine — by default, a cloud API requiring network access
    at call time.

    NOT VERIFIED IN THIS PROJECT'S DEVELOPMENT ENVIRONMENT (no network
    access was available there) — this class is implemented and syntax-
    checked only. Treat it as a real starting point to test in your own
    environment, not as a confirmed-working integration, the same caveat
    given for src/modeling/lcnn.py.
    """

    def __init__(self, language: str = "en-US"):
        try:
            import speech_recognition as sr_lib
        except ImportError as exc:
            raise ImportError(
                "SpeechRecognitionTranscriber requires `pip install SpeechRecognition` "
                "(and, depending on the engine you call, network access or a local model). "
                "Use NullTranscriber or your own Transcriber implementation otherwise."
            ) from exc
        self._sr_lib = sr_lib
        self._recognizer = sr_lib.Recognizer()
        self.language = language

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> TranscriptionResult:
        import io
        import soundfile as sf  # already a core dependency of this project

        try:
            buffer = io.BytesIO()
            sf.write(buffer, waveform, sample_rate, format="WAV")
            buffer.seek(0)
            with self._sr_lib.AudioFile(buffer) as source:
                audio_data = self._recognizer.record(source)
            text = self._recognizer.recognize_google(audio_data, language=self.language)
            return TranscriptionResult(text=text, status="ok")
        except Exception as exc:  # noqa: BLE001 — any backend failure is a handled, reported outcome
            return TranscriptionResult(
                text="", status="transcription_failed",
                raw_backend_info={"error": f"{type(exc).__name__}: {exc}"},
            )


class FixedResponseTranscriber(Transcriber):
    """
    Test double: always returns a preset transcript, regardless of input
    audio. Used by this project's own unit tests (and usable for a scripted
    demo) to exercise matching/session/decision logic without any real ASR
    engine installed.
    """

    def __init__(self, fixed_text: str, status: str = "ok"):
        self.fixed_text = fixed_text
        self.status = status

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> TranscriptionResult:
        return TranscriptionResult(text=self.fixed_text, status=self.status)
