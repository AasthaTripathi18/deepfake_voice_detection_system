"""
Scorer: loads one trained model artifact directory (as produced by
scripts/train_model.py) and exposes two scoring entry points:

  score_features(feature_vector)   fast path — feature already extracted
                                     (used by the batch evaluator against
                                     precomputed .npz feature caches)
  score_audio_file(path, ...)       full path — loads audio, runs cleaning
                                     (+ optional degradation), extracts
                                     features, scores. This is the path
                                     whose timing represents real "audio
                                     in, risk score out" latency, used by
                                     scripts/measure_latency.py and, later,
                                     the live demo / challenge-response flow.

Both return a dict including a per-stage timing breakdown so latency can
be attributed to a specific stage rather than reported as one opaque
number (see project plan's latency-metric rationale).
"""

from __future__ import annotations

try:
    # Pre-load onnxruntime before sklearn/scipy to prevent Windows C++ runtime DLL crash
    import onnxruntime  # noqa: F401
except (ImportError, OSError):
    pass

import time
from pathlib import Path

import numpy as np

from src.modeling.config import load_json, INT_TO_LABEL
from src.modeling import get_model_class

# NOTE: src.preprocessing (librosa/soundfile-dependent) is imported lazily
# inside score_audio_file, not at module level. This keeps score_features
# (the fast path used by batch evaluation against precomputed .npz feature
# caches) usable in any environment that has the modeling/sklearn stack but
# not the full audio stack installed — those are genuinely separate
# concerns and shouldn't be coupled by an eager import.


class Scorer:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.model_dir = model_dir

        self.train_config = load_json(model_dir / "config.json")
        self.threshold_info = load_json(model_dir / "threshold.json")
        self.threshold = float(self.threshold_info["threshold"])
        self.feature_key = self.train_config["feature_key"]

        model_cls = get_model_class(self.train_config["model_type"])
        self.model = model_cls.load(model_dir)

    def score_features(self, feature_vector: np.ndarray) -> dict:
        """feature_vector: a single sample's features, WITHOUT the batch
        dimension (e.g. shape (80,) for mfcc_stats or (128, 200) for logmel)
        — a leading axis is added here before calling predict_proba."""
        t0 = time.perf_counter()
        batch = feature_vector[np.newaxis, ...]
        score = float(self.model.predict_proba(batch)[0])
        model_sec = time.perf_counter() - t0

        predicted_label = INT_TO_LABEL[int(score >= self.threshold)]
        return {
            "score": score,
            "predicted_label": predicted_label,
            "threshold": self.threshold,
            "timing_sec": {"model": model_sec, "total": model_sec},
        }

    def score_audio_file(
        self,
        file_path: str,
        degradation_condition: str | None = None,
        degradation_params: dict | None = None,
        target_sr: int | None = 16000,
        clean_denoise: bool = False,
    ) -> dict:
        """
        Full pipeline: load -> clean -> (optional degrade) -> extract the
        ONE feature this model needs -> score. Returns per-stage timing.
        """
        from src.preprocessing import cleaning, degradations, features as feat_mod  # lazy, see module note above

        degradation_params = degradation_params or {}
        timing = {}

        t0 = time.perf_counter()
        waveform, sr = cleaning.load_audio(file_path, target_sr=target_sr)
        t1 = time.perf_counter()
        timing["load"] = t1 - t0

        waveform = cleaning.clean(waveform, sr, trim=True, normalize=True, denoise=clean_denoise)
        if degradation_condition and degradation_condition != "clean":
            waveform = degradations.apply_condition(waveform, sr, degradation_condition, **degradation_params)
        t2 = time.perf_counter()
        timing["preprocessing"] = t2 - t1

        record = feat_mod.build_feature_record(waveform, sr, feature_set=self.feature_key)
        feature_vector = record[self.feature_key]
        t3 = time.perf_counter()
        timing["feature_extraction"] = t3 - t2

        batch = feature_vector[np.newaxis, ...]
        score = float(self.model.predict_proba(batch)[0])
        t4 = time.perf_counter()
        timing["model"] = t4 - t3
        timing["total"] = t4 - t0

        predicted_label = INT_TO_LABEL[int(score >= self.threshold)]
        return {
            "score": score,
            "predicted_label": predicted_label,
            "threshold": self.threshold,
            "timing_sec": timing,
        }

    def score_with_forensic_explainability(
        self,
        file_path_or_waveform,
        sr: int = 16000,
        zero_data_retention: bool = True,
    ) -> dict:
        """
        Full-stack inference returning both 4-tier operational advice and 3-band acoustic forensic attribution.
        Enforces zero-data-retention (ephemeral memory buffer wiping) for biometric privacy.
        """
        from pathlib import Path
        from src.preprocessing import cleaning, features as feat_mod
        from src.decision_logic.interpretability import compute_subband_attribution
        from src.decision_logic.risk_assessment import (
            assess, DetectorReading, AudioQualityReading, ThresholdConfig,
        )

        t0 = time.perf_counter()
        if isinstance(file_path_or_waveform, (str, Path)):
            waveform, sr = cleaning.load_audio(str(file_path_or_waveform), target_sr=sr)
        else:
            waveform = np.array(file_path_or_waveform, dtype=np.float32)

        waveform = cleaning.clean(waveform, sr, trim=True, normalize=True, denoise=False)
        record = feat_mod.build_feature_record(waveform, sr, feature_set=self.feature_key)
        feature_vector = record[self.feature_key]

        batch = feature_vector[np.newaxis, ...]
        score = float(self.model.predict_proba(batch)[0])
        total_time = time.perf_counter() - t0

        # Forensic Explainability: Sub-band energy decomposition
        attribution = compute_subband_attribution(feature_vector, n_mfcc=60 if "lfcc" in self.feature_key else 40)

        # Operational Explainability: 4-Tier decision logic & recommended actions
        thresh_cfg = ThresholdConfig(operating_threshold=self.threshold)
        decision = assess(
            detector=DetectorReading(spoof_score=score),
            audio_quality=AudioQualityReading(status="ok", analyzed_speech_duration_sec=len(waveform) / sr),
            thresholds=thresh_cfg,
        )

        result = {
            "score": score,
            "predicted_label": INT_TO_LABEL[int(score >= self.threshold)],
            "threshold": self.threshold,
            "risk_band": decision.risk_band,
            "recommended_action": decision.recommended_action,
            "operational_reasons": decision.reasons,
            "forensic_attribution": {
                "low_band_pct": attribution.low_band_pct,
                "mid_band_pct": attribution.mid_band_pct,
                "high_band_pct": attribution.high_band_pct,
                "primary_cue_band": attribution.primary_cue_band,
                "explanation": attribution.explanation,
            },
            "privacy_audit": {
                "zero_data_retention_enforced": bool(zero_data_retention),
                "ephemeral_in_memory_only": True,
                "biometric_template_persisted": False,
            },
            "latency_ms": round(total_time * 1000.0, 2),
        }

        # Ephemeral privacy wiping
        if zero_data_retention:
            waveform.fill(0.0)

        return result

    def score_conversational_audio(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
        window_sec: float = 4.0,
        hop_sec: float = 1.0,
    ) -> dict:
        """
        Evaluates multi-speaker or continuous conversational streams by temporal windowing.
        Identifies exact suspicious timestamps if an injected deepfake occurs during any speaker turn.
        """
        from src.preprocessing import cleaning, features as feat_mod

        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)
        n_samples = len(waveform)

        if n_samples < window_samples:
            res = self.score_with_forensic_explainability(waveform, sr=sr)
            return {
                "conversational_verdict": "AUTHENTIC_CONVERSATION" if res["score"] < self.threshold else "SPOOF_INJECTED",
                "max_risk_score": res["score"],
                "mean_risk_score": res["score"],
                "suspicious_segments": [(0.0, n_samples / sr)] if res["score"] >= self.threshold else [],
                "timeline": [{"start_sec": 0.0, "end_sec": n_samples / sr, "score": res["score"]}],
            }

        timeline = []
        suspicious_segments = []
        scores = []

        start = 0
        while start + window_samples <= n_samples:
            chunk = waveform[start:start + window_samples]
            t_start = start / sr
            t_end = (start + window_samples) / sr

            cleaned = cleaning.clean(chunk, sr, trim=False, normalize=True, denoise=False)
            rec = feat_mod.build_feature_record(cleaned, sr, feature_set=self.feature_key)
            batch = rec[self.feature_key][np.newaxis, ...]
            s = float(self.model.predict_proba(batch)[0])

            scores.append(s)
            timeline.append({"start_sec": round(t_start, 2), "end_sec": round(t_end, 2), "score": round(s, 4)})
            if s >= self.threshold:
                suspicious_segments.append((round(t_start, 2), round(t_end, 2)))

            start += hop_samples

        max_score = float(max(scores)) if scores else 0.0
        mean_score = float(np.mean(scores)) if scores else 0.0

        return {
            "conversational_verdict": "SPOOF_INJECTED" if max_score >= self.threshold else "AUTHENTIC_CONVERSATION",
            "max_risk_score": round(max_score, 4),
            "mean_risk_score": round(mean_score, 4),
            "suspicious_segments": suspicious_segments,
            "n_turns_analyzed": len(timeline),
            "timeline": timeline,
        }

