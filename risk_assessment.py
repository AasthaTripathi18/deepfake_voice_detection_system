"""
Pure decision logic. See src/decision_logic/__init__.py for the hard rules
this module enforces. No file I/O, no audio processing, no model calls —
it only combines numbers and statuses that some other module already
computed (src.inference.scorer.Scorer, src.streaming.quality_gate,
src.challenge_response.decision), which is what makes every rule here
inspectable and testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

LOW_RISK = "Low risk"
NEEDS_VERIFICATION = "Needs verification"
HIGH_RISK = "High risk"
INSUFFICIENT_EVIDENCE = "Insufficient evidence"

BAND_VERY_LOW = "Very Low Risk"
BAND_LOW = "Low Risk"
BAND_NEEDS_VERIFICATION = "Needs Verification"
BAND_CRITICAL = "Critical Threat"
BAND_INSUFFICIENT = "Insufficient Evidence"

ACTION_FAST_PASS = "FAST_PASS: Genuine human voice verified with high confidence."
ACTION_STANDARD_PASS = "STANDARD_PASS: Genuine voice verified."
ACTION_STEP_UP = "STEP_UP_AUTH: Borderline risk score. Prompt secondary authentication (challenge-response or OTP) rather than rejecting the user."
ACTION_HARD_BLOCK = "HARD_BLOCK: High probability of synthetic deepfake. Block or escalate to fraud prevention."
ACTION_RETRY = "RETRY: Inconclusive audio. Prompt speaker to re-record in a quiet environment."


@dataclass
class BankingFraudPolicy:
    """
    Enterprise Banking and Call Center Fraud Policy Matrix.
    Maps biometric risk assessments and monetary transaction tiers into operational controls.
    """
    low_value_threshold_usd: float = 1000.0
    high_value_threshold_usd: float = 10000.0

    def evaluate_transaction(self, assessment: RiskAssessment, amount_usd: float = 0.0) -> dict:
        """
        Determines the transaction disposition:
        - CLEARED_APPROVE: Instant transaction approval.
        - STEP_UP_REQUIRED: Prompt for secondary authentication (SMS/App OTP, challenge phrase).
        - FRAUD_ALERT_BLOCK: Transaction blocked; security ops alerted.
        - REVIEW_RETRY: Voice sample unclear; prompt caller to re-authenticate.
        """
        raw_score = assessment.raw_score if assessment.raw_score is not None else 0.0

        if assessment.status == HIGH_RISK:
            return {
                "disposition": "FRAUD_ALERT_BLOCK",
                "action": "Block transaction immediately and route caller to fraud investigations.",
                "reason": f"Biometric risk score ({raw_score:.4f}) indicates synthetic speech (Critical Threat).",
                "step_up_method": None,
                "audit_log_level": "CRITICAL",
            }

        if assessment.status == NEEDS_VERIFICATION:
            return {
                "disposition": "STEP_UP_REQUIRED",
                "action": "Prompt for out-of-band step-up authentication (SMS OTP or dynamic challenge).",
                "reason": f"Borderline biometric score ({raw_score:.4f}); step-up authentication required to prevent false rejection.",
                "step_up_method": "SMS_OTP_OR_VOICE_CHALLENGE",
                "audit_log_level": "WARNING",
            }

        if assessment.status == INSUFFICIENT_EVIDENCE:
            return {
                "disposition": "REVIEW_RETRY",
                "action": "Voice sample insufficient; prompt caller to speak into the microphone in a quiet room.",
                "reason": "Quality gate failed or insufficient voiced duration.",
                "step_up_method": "RETRY_AUDIO",
                "audit_log_level": "INFO",
            }

        # Status is LOW_RISK
        if amount_usd >= self.high_value_threshold_usd:
            return {
                "disposition": "STEP_UP_REQUIRED",
                "action": "High-value monetary threshold reached ($10,000+); standard secondary compliance check required.",
                "reason": "Voice verified as authentic, but transaction value mandates dual-factor authorization.",
                "step_up_method": "SMS_OTP",
                "audit_log_level": "NOTICE",
            }

        return {
            "disposition": "CLEARED_APPROVE",
            "action": "Transaction approved.",
            "reason": "Authentic human voice verified with low risk score.",
            "step_up_method": None,
            "audit_log_level": "INFO",
        }


_TIER_SEVERITY = {
    INSUFFICIENT_EVIDENCE: 0,   # a distinct axis (see module docstring), not "between" the others
    LOW_RISK: 1,
    NEEDS_VERIFICATION: 2,
    HIGH_RISK: 3,
}

# quality_gate statuses that mean "no usable audio was ever available."
_NO_USABLE_AUDIO_STATUSES = {"insufficient_audio", "silence", "insufficient_speech"}


@dataclass
class ThresholdConfig:
    """
    Every number that drives the tiering below, in one place, with an
    explicit note on where it came from.

    operating_threshold: the model's own spoof/bonafide decision boundary.
        THIS is the one number in this whole assessment that IS selected
        using validation data: see scripts/train_model.py, which picks it
        as the EER point on the official DEV split (never on eval/test —
        see src/evaluation/report.py's select_threshold_from_validation).
        Pass it in from scorer.threshold; do not hand-tune it here.

    low_risk_score_ratio: a configurable FRACTION of operating_threshold
        below which a score is shown as "Low risk" rather than "Needs
        verification". Default 0.5 (i.e. a score has to be below half the
        model's own spoof threshold to read as clearly low). THIS ratio is
        NOT independently validated the same way operating_threshold is —
        this project has no labeled data distinguishing "clearly low
        concern" from "borderline" ambient audio to tune it against. It is
        a reasonable, inspectable default. If you later collect such
        labels (e.g. known-genuine calls under a range of conditions vs.
        known-spoofed ones), you could tune this the same way
        operating_threshold was tuned — sweep candidate ratios and check
        which best separates your labeled "should read as Low risk" cases
        from "should read as Needs verification" cases on a held-out
        validation split, the same EER-style methodology already used for
        operating_threshold itself, not by inspection of a handful of
        examples.

    min_reliable_speech_duration_sec: below this much analyzed (voiced)
        speech, an otherwise-low-or-middling score is NOT treated as
        reassuring — see risk_assessment.assess()'s duration handling.
        Default 2.0s is a judgment call (roughly how much speech the
        MFCC-stats features are computed from per window in this
        project's own pipeline — see src/preprocessing/features.py),
        not a value selected via any labeled validation procedure either.
    """
    operating_threshold: float
    low_risk_score_ratio: float = 0.5
    min_reliable_speech_duration_sec: float = 2.0

    @property
    def low_risk_ceiling(self) -> float:
        return self.operating_threshold * self.low_risk_score_ratio

    def explain(self) -> str:
        return (
            f"Operating threshold ({self.operating_threshold:.4f}) was selected on the model's "
            f"validation split by finding the equal-error-rate point (see "
            f"scripts/train_model.py / src/evaluation/report.py) — never on the eval/test "
            f"split used for reporting. Scores below {self.low_risk_ceiling:.4f} "
            f"({self.low_risk_score_ratio:.0%} of the operating threshold) display as 'Low risk'; "
            f"scores from there up to the operating threshold display as 'Needs verification'; "
            f"scores at or above the operating threshold display as 'High risk'. The "
            f"{self.low_risk_score_ratio:.0%} ratio is a configurable, inspectable default, NOT "
            f"itself validated against labeled data the way the operating threshold is — see "
            f"ThresholdConfig's docstring for how you could validate it if you collect the "
            f"right labels. At least {self.min_reliable_speech_duration_sec:.1f}s of analyzed "
            f"speech is required before a low-or-middling score is treated as reassuring; less "
            f"than that reports 'Insufficient evidence' instead, unless the score is already "
            f"high enough to warrant caution regardless."
        )


class SpeechAccumulator:
    """
    Running total of UNIQUE analyzed speech, in seconds, across a sequence
    of (possibly overlapping) analysis windows.

    Why this exists: with a sliding window whose hop is shorter than its
    length, consecutive windows share audio. Naively summing
    `window_duration * voiced_fraction` over every window counts the
    overlapping audio more than once, so 6 seconds of audio could appear
    to contain 9+ seconds of speech and wrongly clear the
    minimum-speech-duration requirement. This class credits each window
    only for the audio it adds beyond what earlier windows already
    covered (its "new" region), using that window's voiced fraction as an
    estimate of how much of the new region was speech.

    Windows that were not scored (silence, too short, etc.) still advance
    the covered-audio pointer but contribute zero speech, which can
    slightly undercount; that errs toward asking for more evidence, not
    less.
    """

    def __init__(self):
        self._covered_until_sec = 0.0
        self.speech_sec = 0.0

    def add(self, start_sec: float, end_sec: float, voiced_fraction: float, scored: bool) -> float:
        new_start = max(start_sec, self._covered_until_sec)
        new_duration = max(0.0, end_sec - new_start)
        if scored:
            self.speech_sec += new_duration * max(0.0, min(1.0, voiced_fraction))
        self._covered_until_sec = max(self._covered_until_sec, end_sec)
        return self.speech_sec


@dataclass
class DetectorReading:
    """spoof_score: the model's raw P(spoof) output, or None if it could
    not be computed for this audio (e.g. quality gate withheld scoring).
    This is intentionally NOT called a probability of fraud anywhere in
    this module or its output — see RiskAssessment.is_calibrated."""
    spoof_score: Optional[float]


@dataclass
class AudioQualityReading:
    status: str                          # one of quality_gate's status strings
    analyzed_speech_duration_sec: float    # cumulative voiced duration actually analyzed


@dataclass
class ChallengeReading:
    """Minimal wrapper so this module doesn't need to import
    src.challenge_response directly (keeping decision_logic usable even in
    a deployment that never enables the challenge-response feature)."""
    overall_tier: str   # "elevated_concern" | "inconclusive" | "low_signal_of_concern"


@dataclass
class RiskAssessment:
    status: str                      # LOW_RISK | NEEDS_VERIFICATION | HIGH_RISK | INSUFFICIENT_EVIDENCE
    reasons: list = field(default_factory=list)   # short, plain-language bullet points
    raw_score: Optional[float] = None
    operating_threshold: Optional[float] = None
    risk_band: str = BAND_INSUFFICIENT
    recommended_action: str = ACTION_RETRY
    is_calibrated: bool = field(default=False, init=False)  # ALWAYS False; see module/package docstrings
    calibration_note: str = field(default=(
        "This score is the model's raw output, not a calibrated probability of fraud. "
        "Turning it into one would require running and reporting a calibration evaluation "
        "(e.g. a reliability diagram, Brier score, or expected calibration error on held-out "
        "data) — this project has not done that."
    ), init=False)


def assess(
    detector: DetectorReading,
    audio_quality: AudioQualityReading,
    thresholds: ThresholdConfig,
    challenge: Optional[ChallengeReading] = None,
) -> RiskAssessment:
    reasons = []
    challenge_elevated = challenge is not None and challenge.overall_tier == "elevated_concern"

    # --- Step 1: no score at all -> Insufficient evidence, unless the
    # challenge-response check itself raised concern (a failed challenge is
    # real evidence even when the passive detector had nothing to score;
    # hiding it behind "Insufficient evidence" would under-warn). ---
    if detector.spoof_score is None:
        reasons.append(
            f"No spoof-risk score is available (audio quality status: {audio_quality.status})."
        )
        if challenge_elevated:
            reasons.append("Challenge-response check itself indicated elevated concern.")
            return RiskAssessment(
                status=HIGH_RISK, reasons=reasons,
                raw_score=None, operating_threshold=thresholds.operating_threshold,
                risk_band=BAND_CRITICAL, recommended_action=ACTION_HARD_BLOCK,
            )
        return RiskAssessment(
            status=INSUFFICIENT_EVIDENCE, reasons=reasons,
            raw_score=None, operating_threshold=thresholds.operating_threshold,
            risk_band=BAND_INSUFFICIENT, recommended_action=ACTION_RETRY,
        )

    score = detector.spoof_score
    is_high = score >= thresholds.operating_threshold

    # --- Step 2: too little analyzed speech to trust a non-high reading ---
    if audio_quality.analyzed_speech_duration_sec < thresholds.min_reliable_speech_duration_sec and not is_high:
        reasons.append(
            f"Only {audio_quality.analyzed_speech_duration_sec:.1f}s of speech has been analyzed "
            f"so far (minimum for a confident low/middling reading: "
            f"{thresholds.min_reliable_speech_duration_sec:.1f}s)."
        )
        if challenge_elevated:
            reasons.append("Challenge-response check itself indicated elevated concern.")
            return RiskAssessment(
                status=HIGH_RISK, reasons=reasons,
                raw_score=score, operating_threshold=thresholds.operating_threshold,
                risk_band=BAND_CRITICAL, recommended_action=ACTION_HARD_BLOCK,
            )
        return RiskAssessment(
            status=INSUFFICIENT_EVIDENCE, reasons=reasons,
            raw_score=score, operating_threshold=thresholds.operating_threshold,
            risk_band=BAND_INSUFFICIENT, recommended_action=ACTION_RETRY,
        )

    # --- Step 3: base tier and fine-grained 4-tier risk band from score ---
    if is_high:
        status = HIGH_RISK
        risk_band = BAND_CRITICAL
        recommended_action = ACTION_HARD_BLOCK
        reasons.append(
            f"Detector score ({score:.4f}) is at or above the operating threshold "
            f"({thresholds.operating_threshold:.4f})."
        )
        if audio_quality.analyzed_speech_duration_sec < thresholds.min_reliable_speech_duration_sec:
            reasons.append(
                f"Note: based on only {audio_quality.analyzed_speech_duration_sec:.1f}s of "
                f"speech — treat with extra caution either way."
            )
    elif score < thresholds.operating_threshold * 0.20:
        status = LOW_RISK
        risk_band = BAND_VERY_LOW
        recommended_action = ACTION_FAST_PASS
        reasons.append(
            f"Detector score ({score:.4f}) is well below the operating threshold "
            f"({thresholds.operating_threshold:.4f}) — confident genuine human voice."
        )
    elif score < thresholds.low_risk_ceiling:
        status = LOW_RISK
        risk_band = BAND_LOW
        recommended_action = ACTION_STANDARD_PASS
        reasons.append(
            f"Detector score ({score:.4f}) is below the operating threshold "
            f"({thresholds.operating_threshold:.4f})."
        )
    else:
        status = NEEDS_VERIFICATION
        risk_band = BAND_NEEDS_VERIFICATION
        recommended_action = ACTION_STEP_UP
        reasons.append(
            f"Detector score ({score:.4f}) is below the operating threshold "
            f"({thresholds.operating_threshold:.4f}) but not clearly low."
        )

    # --- Step 4: audio-quality cap (can only raise the tier, never lower) ---
    if audio_quality.status == "high_noise_low_confidence" and status == LOW_RISK:
        status = NEEDS_VERIFICATION
        risk_band = BAND_NEEDS_VERIFICATION
        recommended_action = ACTION_STEP_UP
        reasons.append("Elevated background noise or clipping reduces confidence in a 'Low risk' reading.")

    # --- Step 5: optional challenge-response signal (can only raise, never lower) ---
    if challenge is not None:
        if challenge.overall_tier == "elevated_concern" and status != HIGH_RISK:
            status = HIGH_RISK
            risk_band = BAND_CRITICAL
            recommended_action = ACTION_HARD_BLOCK
            reasons.append("Challenge-response check itself indicated elevated concern.")
        elif challenge.overall_tier == "inconclusive" and status == LOW_RISK:
            status = NEEDS_VERIFICATION
            risk_band = BAND_NEEDS_VERIFICATION
            recommended_action = ACTION_STEP_UP
            reasons.append("Challenge-response result was inconclusive.")
        elif challenge.overall_tier == "low_signal_of_concern":
            reasons.append(
                "Challenge-response check found no elevated concern (one supporting signal, "
                "not confirmation)."
            )

    return RiskAssessment(
        status=status, reasons=reasons,
        raw_score=score, operating_threshold=thresholds.operating_threshold,
        risk_band=risk_band, recommended_action=recommended_action,
    )
