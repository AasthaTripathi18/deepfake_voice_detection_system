"""
StreamingSpoofDetector: the only module in src/streaming that touches
feature extraction or the trained model. Everything else (capture,
windowing, quality gate) is deliberately model-agnostic and audio-source-
agnostic, per src/streaming/__init__.py's module-separation note.

Output is EXPLICITLY an experimental risk indicator per window, never an
identity verdict:
  - `risk_score`: the model's raw P(spoof) in [0, 1], or None if the
    quality gate withheld inference for this window.
  - `risk_tier`: an optional coarse, hedged label ("low"/"elevated"/
    "high"), only computed if the caller supplies tier boundaries — never
    "genuine"/"fake" or any identity-implying wording.
  - `uncertainty`: a heuristic in [0, 1] combining score-margin, voice
    activity, and the noise proxy — see _estimate_uncertainty's docstring
    for exactly what this is and, importantly, what it is NOT (a
    calibrated confidence).
  - `status`: the quality gate's status for this window (see
    quality_gate.py), always present even when a score is withheld.
  - `source_label`: the exact audio source string (see audio_source.py),
    always present, meant to be shown to the user verbatim.
  - `processing_delay_sec`: measured wall-clock time to process this
    window (quality check + any feature extraction + model call) — this
    is the "measured delay" this mode is asked to show, computed fresh
    for every window on whatever machine this actually runs on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Iterator, Optional

import numpy as np

from .audio_source import AudioSource
from .windowing import SlidingWindowBuffer, Window
from .quality_gate import assess_window_quality, QualityAssessment


@dataclass
class WindowResult:
    source_label: str
    window_start_sec: float
    window_end_sec: float
    window_duration_sec: float
    is_full_window: bool
    status: str
    voiced_fraction: float
    estimated_snr_db: Optional[float]
    risk_score: Optional[float]
    risk_tier: Optional[str]
    uncertainty: Optional[float]
    operating_threshold: Optional[float]
    processing_delay_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


class StreamingSpoofDetector:
    def __init__(
        self,
        scorer,                      # src.inference.scorer.Scorer, or a compatible test double
        source: AudioSource,
        window_sec: float = 2.0,
        hop_sec: float = 1.0,
        min_duration_sec: float = 1.0,
        min_voiced_fraction: float = 0.15,
        min_snr_db: float = 6.0,
        risk_tier_bounds: Optional[tuple[float, float]] = None,
    ):
        """
        risk_tier_bounds: optional (low_bound, high_bound). If given,
        risk_tier="low" when score < low_bound, "high" when
        score >= high_bound, else "elevated". These bounds are NOT derived
        from any validated threshold-selection procedure (unlike the
        model's own `scorer.threshold`, which IS selected on held-out
        validation data) — they are a display convenience the caller must
        choose deliberately, and default to None (no tiering, raw score
        only) so a tier label is never shown unless someone explicitly
        asked for one.
        """
        self.scorer = scorer
        self.source = source
        self.buffer = SlidingWindowBuffer(source.sample_rate, window_sec, hop_sec)
        self.min_duration_sec = min_duration_sec
        self.min_voiced_fraction = min_voiced_fraction
        self.min_snr_db = min_snr_db
        self.risk_tier_bounds = risk_tier_bounds

    def run(self, max_windows: Optional[int] = None) -> Iterator[WindowResult]:
        n_emitted = 0
        while True:
            chunk = self.source.read_chunk()
            if chunk is None:
                final_window = self.buffer.flush()
                if final_window is not None:
                    yield self._process_window(final_window)
                return

            for window in self.buffer.push(chunk):
                yield self._process_window(window)
                n_emitted += 1
                if max_windows is not None and n_emitted >= max_windows:
                    return

    def _process_window(self, window: Window) -> WindowResult:
        t0 = time.perf_counter()

        quality = assess_window_quality(
            window,
            min_duration_sec=self.min_duration_sec,
            min_voiced_fraction=self.min_voiced_fraction,
            min_snr_db=self.min_snr_db,
        )

        risk_score, uncertainty, risk_tier = None, None, None

        if quality.proceed_to_inference:
            feature_vector = self._extract_feature_vector(window.waveform, window.sample_rate)
            score_result = self.scorer.score_features(feature_vector)
            risk_score = score_result["score"]
            uncertainty = self._estimate_uncertainty(risk_score, quality)
            risk_tier = self._tier_for_score(risk_score)

        processing_delay_sec = time.perf_counter() - t0

        return WindowResult(
            source_label=self.source.label,
            window_start_sec=window.start_sec,
            window_end_sec=window.end_sec,
            window_duration_sec=window.duration_sec,
            is_full_window=window.is_full,
            status=quality.status,
            voiced_fraction=quality.voiced_fraction,
            estimated_snr_db=quality.estimated_snr_db,
            risk_score=risk_score,
            risk_tier=risk_tier,
            uncertainty=uncertainty,
            operating_threshold=getattr(self.scorer, "threshold", None),
            processing_delay_sec=processing_delay_sec,
        )

    def _extract_feature_vector(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Isolated as its own method (rather than inlined) so tests can
        monkeypatch this one call and exercise the rest of the streaming
        pipeline without needing librosa/soundfile installed."""
        from src.preprocessing import features as feat_mod  # lazy, see module note
        record = feat_mod.build_feature_record(waveform, sample_rate, feature_set=self.scorer.feature_key)
        return record[self.scorer.feature_key]

    def _estimate_uncertainty(self, score: float, quality: QualityAssessment) -> float:
        """
        HEURISTIC, NOT a calibrated confidence. Combines three easily
        computed signals, each mapped to [0, 1] where higher = less
        trustworthy, then averaged:

          margin_component:  how close `score` is to the model's
              validation-selected operating threshold. A score sitting
              right at the threshold is the case the model itself is
              least decisive about, regardless of anything else.
          voiced_component:   1 - voiced_fraction. Less speech content in
              the window (even if enough to pass the quality gate) means
              less signal the score could have been based on.
          noise_component:    inverse of the SNR proxy, saturating at a
              fixed reference range. No estimate available -> treated as
              neutral (0.5), not as reassuring (0.0).

        This has NOT been validated against labeled data under matching
        conditions — that would require running this project's own
        RQ1-style evaluation (see project plan) specifically on streaming
        windows, which has not been done. Treat this purely as a display
        aid for flagging "trust this one less," not as a statistically
        meaningful uncertainty quantity.
        """
        threshold = getattr(self.scorer, "threshold", 0.5)
        normalizer = max(threshold, 1.0 - threshold, 1e-6)
        margin_component = max(0.0, 1.0 - abs(score - threshold) / normalizer)

        voiced_component = max(0.0, min(1.0, 1.0 - quality.voiced_fraction))

        if quality.estimated_snr_db is None:
            noise_component = 0.5
        else:
            reference_range_db = 15.0
            noise_component = max(0.0, min(1.0, (reference_range_db - quality.estimated_snr_db) / reference_range_db))

        return round(float((margin_component + voiced_component + noise_component) / 3.0), 4)

    def _tier_for_score(self, score: float) -> Optional[str]:
        if self.risk_tier_bounds is None:
            return None
        low_bound, high_bound = self.risk_tier_bounds
        if score < low_bound:
            return "low"
        if score >= high_bound:
            return "high"
        return "elevated"
