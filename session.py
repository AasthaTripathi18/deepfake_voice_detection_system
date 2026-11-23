"""
Orchestrates one challenge-response round: issue -> capture -> quality
check -> transcribe -> match content -> spoof-score the response audio.
Reuses src.streaming.quality_gate (the SAME gate used for passive
streaming windows, applied here to the whole captured response clip as
one "window") and src.inference.scorer.Scorer (the SAME trained model and
operating threshold used everywhere else in this project) rather than
re-implementing either.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.streaming.audio_source import AudioSource
from src.streaming.windowing import Window
from src.streaming.quality_gate import assess_window_quality
from .challenge_generator import Challenge, ChallengeGenerator, render_for_display, speak_challenge
from .response_capture import capture_response, CapturedResponse
from .transcription import Transcriber, NullTranscriber, TranscriptionResult
from .matching import match_challenge, MatchResult, NO_MATCH

LIVE_CLONING_CAVEAT = (
    "Passing this challenge shows the response contained the right words and did not "
    "trip the spoof detector on this clip. It does NOT rule out a live voice-cloning or "
    "real-time voice-conversion system that hears the challenge as it is issued and "
    "immediately re-synthesizes a spoken reply in the target's voice. This challenge "
    "defends mainly against naive replay of pre-recorded audio, not against that more "
    "sophisticated scenario."
)

# Audio-quality outcomes (from quality_gate) that mean "there wasn't
# enough usable audio to say anything meaningful about content or voice."
_NO_USABLE_AUDIO_STATUSES = {"insufficient_audio", "silence", "insufficient_speech"}


@dataclass
class ChallengeResult:
    challenge_id: str
    challenge_text: str
    source_label: str
    captured_duration_sec: float
    audio_quality_status: str          # from quality_gate: ok | high_noise_low_confidence | insufficient_audio | silence | insufficient_speech
    transcription_status: str           # ok | backend_unavailable | transcription_failed | skipped_no_usable_audio
    transcribed_text: str
    match_status: str                    # matched | partial_match | no_match | inconclusive
    match_ratio: Optional[float]
    spoof_risk_score: Optional[float]     # P(spoof) on the response clip, or None if not computed
    spoof_operating_threshold: Optional[float]
    processing_delay_sec: float
    warnings: list = field(default_factory=list)


class ChallengeResponseSession:
    def __init__(
        self,
        scorer,                        # src.inference.scorer.Scorer, or a compatible test double
        transcriber: Optional[Transcriber] = None,
        num_words: int = 2,
        digit_count: int = 3,
        min_duration_sec: float = 1.0,
        min_voiced_fraction: float = 0.15,
        min_snr_db: float = 6.0,
        match_thresholds: tuple = (0.75, 0.4),
    ):
        self.scorer = scorer
        self.transcriber = transcriber or NullTranscriber()
        self.generator = ChallengeGenerator(num_words=num_words, digit_count=digit_count)
        self.min_duration_sec = min_duration_sec
        self.min_voiced_fraction = min_voiced_fraction
        self.min_snr_db = min_snr_db
        self.match_thresholds = match_thresholds
        self._current_challenge: Optional[Challenge] = None

    def issue_challenge(self) -> Challenge:
        """Generates and remembers a fresh challenge. Any previous
        challenge from this session, consumed or not, is discarded — only
        the most recently issued challenge can be answered, preventing an
        old challenge from being checked against a later response."""
        self._current_challenge = self.generator.generate()
        return self._current_challenge

    def present_challenge(self, tts_backend=None) -> str:
        if self._current_challenge is None:
            raise RuntimeError("issue_challenge() must be called before present_challenge().")
        return speak_challenge(self._current_challenge, tts_backend=tts_backend)

    def run_challenge(self, source: AudioSource, max_duration_sec: float = 6.0) -> ChallengeResult:
        if self._current_challenge is None:
            raise RuntimeError("No challenge has been issued — call issue_challenge() first.")
        if self._current_challenge.consumed:
            raise RuntimeError(
                "This challenge has already been checked against a response and cannot be "
                "reused — call issue_challenge() to get a fresh one."
            )

        t0 = time.perf_counter()
        challenge = self._current_challenge

        captured = capture_response(source, max_duration_sec=max_duration_sec)

        window = Window(
            waveform=captured.waveform, start_sample=0, end_sample=len(captured.waveform),
            sample_rate=captured.sample_rate, is_full=(not captured.ended_early),
        )
        quality = assess_window_quality(
            window,
            min_duration_sec=self.min_duration_sec,
            min_voiced_fraction=self.min_voiced_fraction,
            min_snr_db=self.min_snr_db,
        )

        warnings = [LIVE_CLONING_CAVEAT]
        if captured.ended_early and captured.duration_sec < max_duration_sec:
            warnings.append(
                f"Audio source ended after {captured.duration_sec:.2f}s, before the "
                f"requested {max_duration_sec:.2f}s listening window completed."
            )

        transcription_status = "skipped_no_usable_audio"
        transcribed_text = ""
        match_status = "inconclusive"
        match_ratio = None
        spoof_score = None

        if quality.status not in _NO_USABLE_AUDIO_STATUSES:
            if quality.status == "high_noise_low_confidence":
                warnings.append(
                    f"Estimated audio quality is low (SNR proxy={quality.estimated_snr_db:.1f}dB); "
                    f"content and spoof-risk results below are less reliable than usual."
                )

            transcription: TranscriptionResult = self.transcriber.transcribe(
                captured.waveform, captured.sample_rate
            )
            transcription_status = transcription.status
            transcribed_text = transcription.text

            if transcription.status == "ok":
                match_result: MatchResult = match_challenge(
                    challenge.expected_tokens, transcribed_text,
                    matched_threshold=self.match_thresholds[0],
                    partial_threshold=self.match_thresholds[1],
                )
                match_status = match_result.status
                match_ratio = match_result.ratio
            else:
                match_status = "inconclusive"
                warnings.append(
                    f"Speech-to-text did not produce a usable transcript (status={transcription.status}); "
                    f"content of the response could not be checked."
                )

            spoof_score = self._score_spoof_risk(captured.waveform, captured.sample_rate)
        else:
            warnings.append(
                f"Captured response audio was not usable for analysis (status={quality.status}); "
                f"neither content nor spoof-risk could be checked."
            )

        challenge.consumed = True
        self._current_challenge = None

        return ChallengeResult(
            challenge_id=challenge.challenge_id,
            challenge_text=challenge.text,
            source_label=captured.source_label,
            captured_duration_sec=captured.duration_sec,
            audio_quality_status=quality.status,
            transcription_status=transcription_status,
            transcribed_text=transcribed_text,
            match_status=match_status,
            match_ratio=match_ratio,
            spoof_risk_score=spoof_score,
            spoof_operating_threshold=getattr(self.scorer, "threshold", None),
            processing_delay_sec=time.perf_counter() - t0,
            warnings=warnings,
        )

    def _score_spoof_risk(self, waveform, sample_rate) -> Optional[float]:
        try:
            from src.preprocessing import features as feat_mod  # lazy, needs librosa
            record = feat_mod.build_feature_record(waveform, sample_rate, feature_set=self.scorer.feature_key)
            feature_vector = record[self.scorer.feature_key]
            result = self.scorer.score_features(feature_vector)
            return result["score"]
        except Exception:  # noqa: BLE001 — a scoring failure should not crash the whole session
            return None
