"""
Service layer for the Streamlit app: EVERYTHING that touches the trained
model, the streaming detector, the challenge-response session, or the
decision logic lives here. This module imports nothing from Streamlit, so
it can be unit-tested (and reused by another front end) without a UI, and
the UI modules (ui_components.py, app.py) import nothing from src.inference,
src.streaming, or src.preprocessing directly — they only call the
functions and dataclasses below.

Audio sources this app actually supports (and no others):
  - an uploaded audio file, and
  - a clip recorded in the browser with the microphone (st.audio_input).
Both are handed to the existing streaming pipeline as an in-memory
waveform replayed through the sliding window. "Live" in this prototype
therefore means "record a short clip, then analyze it" — not continuous
capture, and not a phone-call tap.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

try:
    # Pre-load onnxruntime before sklearn/scipy to prevent Windows C++ runtime DLL crash
    import onnxruntime  # noqa: F401
except ImportError:
    pass

import numpy as np

from src.inference.scorer import Scorer
from src.streaming.audio_source import FileReplaySource
from src.streaming.stream_inference import StreamingSpoofDetector, WindowResult
from src.challenge_response.session import (
    ChallengeResponseSession, ChallengeResult, LIVE_CLONING_CAVEAT,
)
from src.challenge_response.decision import combine_assessment, CombinedAssessment
from src.challenge_response.transcription import NullTranscriber, FixedResponseTranscriber
from src.decision_logic.risk_assessment import (
    assess, RiskAssessment, DetectorReading, AudioQualityReading, ChallengeReading,
    ThresholdConfig, SpeechAccumulator,
    LOW_RISK, NEEDS_VERIFICATION, HIGH_RISK, INSUFFICIENT_EVIDENCE,
    BAND_VERY_LOW, BAND_LOW, BAND_NEEDS_VERIFICATION, BAND_CRITICAL, BAND_INSUFFICIENT,
    ACTION_FAST_PASS, ACTION_STANDARD_PASS, ACTION_STEP_UP, ACTION_HARD_BLOCK, ACTION_RETRY,
)
from src.decision_logic.presentation import build_display, DisplayPayload

__all__ = [
    "LIVE_CLONING_CAVEAT", "LOW_RISK", "NEEDS_VERIFICATION", "HIGH_RISK", "INSUFFICIENT_EVIDENCE",
    "ModelLoadError", "ModelHandle", "load_model", "AnalysisSettings", "ClipAnalysis", "ClipReport",
    "prepare_waveform", "estimate_window_count", "iter_analysis", "assess_clip",
    "ChallengeService", "ChallengeOutcome", "speech_recognition_available",
    "TRANSCRIBER_NONE", "TRANSCRIBER_MANUAL", "TRANSCRIBER_SPEECH_RECOGNITION",
]

MAX_ANALYSIS_SECONDS = 60.0
_SCORED_STATUSES = ("ok", "high_noise_low_confidence")
_SEVERITY = {INSUFFICIENT_EVIDENCE: 0, LOW_RISK: 1, NEEDS_VERIFICATION: 2, HIGH_RISK: 3}


# --------------------------------------------------------------------- model

class ModelLoadError(RuntimeError):
    """Raised with a user-readable message when a model directory can't be loaded."""


@dataclass
class ModelHandle:
    scorer: object                 # src.inference.scorer.Scorer (or a compatible test double)
    model_dir: str
    model_type: str
    feature_key: str
    operating_threshold: float
    threshold_info: dict = field(default_factory=dict)


def load_model(model_dir: str) -> ModelHandle:
    path = Path(model_dir)
    if not path.is_dir():
        raise ModelLoadError(
            f"Model directory not found: {model_dir!r}. Train one first with "
            f"scripts/train_model.py (see README), then enter its path in the sidebar."
        )
    missing = [n for n in ("config.json", "threshold.json") if not (path / n).exists()]
    if missing:
        raise ModelLoadError(
            f"{model_dir!r} is missing {', '.join(missing)}. It should be the output "
            f"directory of scripts/train_model.py."
        )
    try:
        scorer = Scorer(path)
    except Exception as exc:  # noqa: BLE001 - surface any load failure readably
        raise ModelLoadError(f"Could not load model from {model_dir!r}: {type(exc).__name__}: {exc}") from exc
    return ModelHandle(
        scorer=scorer, model_dir=str(path),
        model_type=scorer.train_config["model_type"], feature_key=scorer.feature_key,
        operating_threshold=float(scorer.threshold), threshold_info=dict(scorer.threshold_info),
    )


# ------------------------------------------------------------------ analysis

@dataclass
class AnalysisSettings:
    window_sec: float = 2.0
    hop_sec: float = 1.0
    low_risk_score_ratio: float = 0.5
    min_reliable_speech_duration_sec: float = 2.0
    aggregation_rule: str = "mean"

    def thresholds(self, handle: ModelHandle, operating_threshold: Optional[float] = None) -> ThresholdConfig:
        op_thresh = operating_threshold if operating_threshold is not None else handle.operating_threshold
        return ThresholdConfig(
            operating_threshold=op_thresh,
            low_risk_score_ratio=self.low_risk_score_ratio,
            min_reliable_speech_duration_sec=self.min_reliable_speech_duration_sec,
        )


@dataclass
class ClipAnalysis:
    """Raw per-window model output for one analyzed clip. No decision logic
    here — assess_clip() turns this into statuses, and can be re-run cheaply
    (no model calls) whenever thresholds or the challenge result change."""
    source_label: str
    duration_sec: float
    window_sec: float
    hop_sec: float
    windows: list = field(default_factory=list)      # list[WindowResult]
    notes: list = field(default_factory=list)          # decoding / truncation notes shown to the user


def prepare_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    max_seconds: float = MAX_ANALYSIS_SECONDS,
    match_training: bool = False,
):
    """
    Cap analysis length so the UI stays responsive; returns (waveform, notes).
    When match_training=True:
      - Trims leading/trailing silence using energy-based trim
      - Normalizes peak amplitude to -3 dBFS to match training distribution
    """
    notes = []
    waveform = np.asarray(waveform, dtype=np.float32)
    max_samples = int(max_seconds * sample_rate)
    if len(waveform) > max_samples:
        notes.append(f"Audio was longer than {max_seconds:.0f}s; only the first {max_seconds:.0f}s were analyzed.")
        waveform = waveform[:max_samples]

    if match_training and len(waveform) > 0:
        from src.preprocessing.cleaning import trim_silence, normalize_amplitude
        orig_dur = len(waveform) / sample_rate
        trimmed = trim_silence(waveform, top_db=30.0)
        trimmed_dur = len(trimmed) / sample_rate
        if orig_dur - trimmed_dur > 0.15:
            notes.append(f"Trimmed {orig_dur - trimmed_dur:.1f}s of silence to match training condition.")
        if len(trimmed) > 0 and np.max(np.abs(trimmed)) > 0:
            waveform = normalize_amplitude(trimmed, method="peak", target_dbfs=-3.0)
            notes.append("Peak-normalized amplitude to -3 dBFS (training-matched level).")
        else:
            waveform = trimmed

    return np.asarray(waveform, dtype=np.float32), notes


def estimate_window_count(duration_sec: float, window_sec: float, hop_sec: float) -> int:
    """Approximate number of windows the sliding window will emit (for a progress bar)."""
    if duration_sec < window_sec:
        return 1
    return int((duration_sec - window_sec) // hop_sec) + 1 + 1  # +1 for the trailing flushed window


def iter_analysis(
    handle: ModelHandle,
    waveform: np.ndarray,
    sample_rate: int,
    source_label: str,
    window_sec: float,
    hop_sec: float,
    feature_extractor: Optional[Callable] = None,
) -> Iterator[WindowResult]:
    """
    Replays `waveform` through the sliding window and yields one WindowResult
    per window as it is produced (so the UI can show live progress).
    feature_extractor is a test seam only: when given, it replaces the real
    MFCC/log-mel extraction (which needs librosa) so tests can run without it.
    """
    source = FileReplaySource(waveform=waveform, sample_rate=sample_rate, chunk_sec=0.25, real_time=False)
    source.label = source_label     # make the true origin visible instead of the generic replay label
    detector = StreamingSpoofDetector(handle.scorer, source, window_sec=window_sec, hop_sec=hop_sec)
    if feature_extractor is not None:
        detector._extract_feature_vector = feature_extractor
    try:
        yield from detector.run()
    finally:
        source.close()


# ------------------------------------------------------------------ report

@dataclass
class ClipReport:
    overall: RiskAssessment
    display: DisplayPayload
    per_window: list = field(default_factory=list)      # list[(WindowResult, RiskAssessment)]
    n_windows: int = 0
    n_scored: int = 0
    status_counts: dict = field(default_factory=dict)
    blocked_counts: dict = field(default_factory=dict)  # quality-gate statuses that withheld scoring
    unique_speech_sec: float = 0.0
    mean_uncertainty: Optional[float] = None
    max_uncertainty: Optional[float] = None
    uncertainty_label: str = "unknown"
    median_delay_ms: Optional[float] = None
    max_delay_ms: Optional[float] = None
    policy_note: str = (
        "Clip-level status = the most cautious status any window reached (High risk > Needs "
        "verification > Low risk). Windows that were themselves 'Insufficient evidence' only decide "
        "the outcome if no window produced anything else. This aggregation is a design choice, not "
        "a validated procedure."
    )


def _uncertainty_label(mean_u: Optional[float]) -> str:
    # Display-only buckets for a heuristic (uncalibrated) quantity.
    if mean_u is None:
        return "unknown"
    if mean_u < 0.33:
        return "lower"
    if mean_u < 0.66:
        return "moderate"
    return "high"


def assess_clip(
    clip: ClipAnalysis,
    thresholds: ThresholdConfig,
    challenge_tier: Optional[str] = None,
    aggregation_rule: str = "mean",
) -> ClipReport:
    challenge = ChallengeReading(overall_tier=challenge_tier) if challenge_tier else None
    accumulator = SpeechAccumulator()
    per_window = []
    status_counts: dict = {}
    blocked_counts: dict = {}

    for w in clip.windows:
        scored = w.status in _SCORED_STATUSES
        cumulative = accumulator.add(w.window_start_sec, w.window_end_sec, w.voiced_fraction, scored)
        a = assess(
            DetectorReading(spoof_score=w.risk_score),
            AudioQualityReading(status=w.status, analyzed_speech_duration_sec=cumulative),
            thresholds, challenge=challenge,
        )
        per_window.append((w, a))
        status_counts[a.status] = status_counts.get(a.status, 0) + 1
        if not scored:
            blocked_counts[w.status] = blocked_counts.get(w.status, 0) + 1

    n_windows = len(per_window)
    scored_windows = [w for w, _ in per_window if w.status in _SCORED_STATUSES]
    uncertainties = [w.uncertainty for w in scored_windows if w.uncertainty is not None]
    delays_ms = [w.processing_delay_sec * 1000.0 for w in clip.windows]

    if not per_window:
        overall = RiskAssessment(
            status=INSUFFICIENT_EVIDENCE,
            reasons=["No analysis windows were produced (the audio may be empty)."],
            raw_score=None, operating_threshold=thresholds.operating_threshold,
        )
        if challenge_tier == "elevated_concern":
            overall = RiskAssessment(
                status=HIGH_RISK,
                reasons=overall.reasons + ["Challenge-response check itself indicated elevated concern."],
                raw_score=None, operating_threshold=thresholds.operating_threshold,
            )
    elif aggregation_rule == "worst_window":
        best_idx = max(range(n_windows), key=lambda i: (_SEVERITY[per_window[i][1].status], i))
        chosen = per_window[best_idx][1]
        reasons = list(chosen.reasons)
        n_high = status_counts.get(HIGH_RISK, 0)
        n_needs = status_counts.get(NEEDS_VERIFICATION, 0)
        reasons.append(
            f"Across {n_windows} window(s): {n_high} High risk, {n_needs} Needs verification, "
            f"{status_counts.get(LOW_RISK, 0)} Low risk, {status_counts.get(INSUFFICIENT_EVIDENCE, 0)} "
            f"Insufficient evidence."
        )
        overall = RiskAssessment(
            status=chosen.status, reasons=reasons,
            raw_score=chosen.raw_score, operating_threshold=thresholds.operating_threshold,
        )
    else:
        valid_scores = [w.risk_score for w in scored_windows if w.risk_score is not None]
        if not valid_scores:
            overall = RiskAssessment(
                status=INSUFFICIENT_EVIDENCE,
                reasons=["No analysis windows produced usable scores (audio may lack sufficient voiced speech)."],
                raw_score=None, operating_threshold=thresholds.operating_threshold,
            )
            if challenge_tier == "elevated_concern":
                overall = RiskAssessment(
                    status=HIGH_RISK,
                    reasons=overall.reasons + ["Challenge-response check itself indicated elevated concern."],
                    raw_score=None, operating_threshold=thresholds.operating_threshold,
                )
        else:
            if aggregation_rule == "median":
                agg_score = float(np.median(valid_scores))
                agg_name = "Median"
            elif aggregation_rule == "p75":
                agg_score = float(np.percentile(valid_scores, 75))
                agg_name = "75th-percentile"
            else:  # "mean"
                agg_score = float(np.mean(valid_scores))
                agg_name = "Mean"

            total_speech = accumulator.speech_sec
            reasons = []
            n_high = status_counts.get(HIGH_RISK, 0)
            n_needs = status_counts.get(NEEDS_VERIFICATION, 0)
            n_low = status_counts.get(LOW_RISK, 0)
            n_insuf = status_counts.get(INSUFFICIENT_EVIDENCE, 0)

            if agg_score >= thresholds.operating_threshold:
                status = HIGH_RISK
                reasons.append(
                    f"Utterance {agg_name.lower()} spoof score ({agg_score:.4f}) meets or exceeds operating "
                    f"threshold ({thresholds.operating_threshold:.4f}), indicating high likelihood of synthetic audio."
                )
            elif total_speech < thresholds.min_reliable_speech_duration_sec:
                status = INSUFFICIENT_EVIDENCE
                reasons.append(
                    f"Only {total_speech:.1f}s of voiced speech was analyzed; at least "
                    f"{thresholds.min_reliable_speech_duration_sec:.1f}s is needed for a reliable non-high reading."
                )
            elif agg_score >= thresholds.low_risk_ceiling:
                status = NEEDS_VERIFICATION
                reasons.append(
                    f"Utterance {agg_name.lower()} spoof score ({agg_score:.4f}) falls in the indeterminate region "
                    f"between {thresholds.low_risk_ceiling:.4f} and {thresholds.operating_threshold:.4f}."
                )
            else:
                status = LOW_RISK
                reasons.append(
                    f"Utterance {agg_name.lower()} spoof score ({agg_score:.4f}) is below the low-risk ceiling "
                    f"({thresholds.low_risk_ceiling:.4f}), consistent with genuine human speech."
                )

            if challenge_tier == "elevated_concern":
                status = HIGH_RISK
                reasons.append("Challenge-response check itself indicated elevated concern.")

            reasons.append(
                f"Across {n_windows} window(s): {n_high} High risk, {n_needs} Needs verification, "
                f"{n_low} Low risk, {n_insuf} Insufficient evidence."
            )
            if status != HIGH_RISK and n_high > 0:
                reasons.append(
                    f"Note: {n_high} of {n_windows} individual window(s) crossed high-risk threshold individually, "
                    f"but overall utterance {agg_name.lower()} was {agg_score:.4f} ({status.replace('_', ' ')})."
                )

            if status == HIGH_RISK:
                risk_band = BAND_CRITICAL
                recommended_action = ACTION_HARD_BLOCK
            elif status == NEEDS_VERIFICATION:
                risk_band = BAND_NEEDS_VERIFICATION
                recommended_action = ACTION_STEP_UP
            elif status == LOW_RISK:
                if agg_score < thresholds.operating_threshold * 0.20:
                    risk_band = BAND_VERY_LOW
                    recommended_action = ACTION_FAST_PASS
                else:
                    risk_band = BAND_LOW
                    recommended_action = ACTION_STANDARD_PASS
            else:
                risk_band = BAND_INSUFFICIENT
                recommended_action = ACTION_RETRY

            overall = RiskAssessment(
                status=status,
                reasons=reasons,
                raw_score=agg_score,
                operating_threshold=thresholds.operating_threshold,
                risk_band=risk_band,
                recommended_action=recommended_action,
            )

    return ClipReport(
        overall=overall, display=build_display(overall), per_window=per_window,
        n_windows=n_windows, n_scored=len(scored_windows), status_counts=status_counts,
        blocked_counts=blocked_counts, unique_speech_sec=accumulator.speech_sec,
        mean_uncertainty=(float(np.mean(uncertainties)) if uncertainties else None),
        max_uncertainty=(float(np.max(uncertainties)) if uncertainties else None),
        uncertainty_label=_uncertainty_label(float(np.mean(uncertainties)) if uncertainties else None),
        median_delay_ms=(float(np.median(delays_ms)) if delays_ms else None),
        max_delay_ms=(float(np.max(delays_ms)) if delays_ms else None),
    )


# ------------------------------------------------------- challenge-response

TRANSCRIBER_NONE = "none"
TRANSCRIBER_MANUAL = "manual"
TRANSCRIBER_SPEECH_RECOGNITION = "speech_recognition"


def speech_recognition_available() -> bool:
    return importlib.util.find_spec("speech_recognition") is not None


@dataclass
class ChallengeOutcome:
    result: ChallengeResult
    assessment: CombinedAssessment
    transcriber_mode: str


class ChallengeService:
    """Holds one ChallengeResponseSession and hides its orchestration from the UI."""

    def __init__(self, handle: ModelHandle, num_words: int = 2, digit_count: int = 3):
        self.session = ChallengeResponseSession(
            scorer=handle.scorer, transcriber=NullTranscriber(),
            num_words=num_words, digit_count=digit_count,
        )

    def issue(self) -> str:
        """Generate a fresh challenge and return its display text."""
        challenge = self.session.issue_challenge()
        return challenge.text

    @property
    def has_pending_challenge(self) -> bool:
        current = self.session._current_challenge
        return current is not None and not current.consumed

    def run(
        self, waveform: np.ndarray, sample_rate: int, source_label: str,
        mode: str = TRANSCRIBER_NONE, manual_text: str = "", max_duration_sec: float = 8.0,
    ) -> ChallengeOutcome:
        extra_warnings = []

        if mode == TRANSCRIBER_MANUAL:
            if manual_text.strip():
                transcriber = FixedResponseTranscriber(fixed_text=manual_text)
                extra_warnings.append(
                    "The transcript was typed in by hand for this demonstration. It is NOT speech "
                    "recognition, so the content check only tests the demo's own plumbing."
                )
            else:
                transcriber = NullTranscriber()   # empty typed text must not read as "caller said nothing"
                extra_warnings.append("No demo transcript was entered, so the content check was skipped.")
        elif mode == TRANSCRIBER_SPEECH_RECOGNITION:
            try:
                from src.challenge_response.transcription import SpeechRecognitionTranscriber
                transcriber = SpeechRecognitionTranscriber()
                extra_warnings.append(
                    "Speech recognition used an online service, so the response audio left this computer."
                )
            except ImportError:
                transcriber = NullTranscriber()
                extra_warnings.append("SpeechRecognition is not installed, so the content check was skipped.")
        else:
            transcriber = NullTranscriber()

        self.session.transcriber = transcriber
        source = FileReplaySource(waveform=waveform, sample_rate=sample_rate, chunk_sec=0.25, real_time=False)
        source.label = source_label
        try:
            result = self.session.run_challenge(source, max_duration_sec=max_duration_sec)
        finally:
            source.close()

        result.warnings.extend(w for w in extra_warnings if w not in result.warnings)
        return ChallengeOutcome(result=result, assessment=combine_assessment(result), transcriber_mode=mode)
