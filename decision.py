"""
Combines a ChallengeResult (and, optionally, an ambient/passive spoof-risk
signal from src.streaming, if this app is also running that mode) into one
hedged assessment.

STRUCTURAL SAFEGUARD: CombinedAssessment.identity_verified is a hardcoded
constant, always False. No code path in this module can set it True — see
the class definition below. This exists specifically so "do not
automatically classify a caller as genuine" cannot be silently violated by
a future edit to the tier logic; someone would have to deliberately change
this dataclass's default, not just adjust a threshold.

TIER LOGIC (simple, explainable, NOT a trained/validated fusion model —
this project has no labeled data to train or validate a compound "trust
score" on, and inventing one would misrepresent how reliable it actually
is):

  "elevated_concern"   — the response's transcribed content did NOT match
                          the challenge (match_status == "no_match"), OR
                          any available spoof-risk score is AT OR ABOVE
                          the model's own operating threshold (i.e. the
                          model itself flags that clip as likely spoof).
                          This tier wins regardless of any other signal.

  "inconclusive"        — no elevated signal was found, but at least one
                          important input is missing or ambiguous:
                          audio quality blocked analysis, the ASR backend
                          wasn't available or failed, or the match was
                          only "partial_match".

  "low_signal_of_concern" — content matched AND every available spoof-risk
                          score is below the operating threshold AND
                          nothing else triggered "elevated_concern" or
                          "inconclusive". This is the BEST case this
                          module can report — it is deliberately never
                          named "verified", "genuine", "safe", or similar.

A recommendation to independently call back on a known, trusted number is
ALWAYS included (never omitted), with wording that escalates at higher
concern tiers — see RECOMMENDATION_TEXT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .session import ChallengeResult, LIVE_CLONING_CAVEAT

ELEVATED_CONCERN = "elevated_concern"
INCONCLUSIVE = "inconclusive"
LOW_SIGNAL_OF_CONCERN = "low_signal_of_concern"

RECOMMENDATION_TEXT = {
    ELEVATED_CONCERN: (
        "Recommended: do not act on this call's requests. Hang up and independently call back "
        "using a phone number you already trust (e.g. from a bank statement, official website, or "
        "your own contacts) — never a number given to you during this call."
    ),
    INCONCLUSIVE: (
        "This result is inconclusive, not reassuring. If anything sensitive is being requested, "
        "hang up and independently call back using a trusted, known number before proceeding."
    ),
    LOW_SIGNAL_OF_CONCERN: (
        "No elevated signal was found, but this is not identity verification. For anything "
        "sensitive (money, credentials, personal data), it remains good practice to hang up and "
        "independently call back using a trusted, known number."
    ),
}


@dataclass(frozen=True)
class CombinedAssessment:
    overall_tier: str
    identity_verified: bool = field(default=False, init=False)  # ALWAYS False; see module docstring
    explanation: str = ""
    recommendation: str = ""
    disclaimers: list = field(default_factory=list)


def combine_assessment(
    challenge_result: ChallengeResult,
    ambient_spoof_score: Optional[float] = None,
    ambient_operating_threshold: Optional[float] = None,
    ambient_status: Optional[str] = None,
) -> CombinedAssessment:
    """
    ambient_spoof_score / ambient_operating_threshold / ambient_status:
    optional — the most recent passive streaming result (src.streaming),
    if this application is running that mode alongside the challenge. Pass
    None for any/all of these if only the challenge-response result is
    available; the tiering logic below treats a missing ambient signal the
    same as "not available", never as "passed."
    """
    reasons = []

    # --- Check for anything that forces ELEVATED_CONCERN ---
    if challenge_result.match_status == "no_match" and challenge_result.transcription_status == "ok":
        reasons.append(
            f"The response's transcribed content did not match the issued challenge "
            f"(similarity ratio={challenge_result.match_ratio})."
        )
        return _build(ELEVATED_CONCERN, challenge_result, reasons)

    spoof_flagged, spoof_reason = _any_score_flags_spoof(
        challenge_result.spoof_risk_score, challenge_result.spoof_operating_threshold,
        ambient_spoof_score, ambient_operating_threshold,
    )
    if spoof_flagged:
        reasons.append(spoof_reason)
        return _build(ELEVATED_CONCERN, challenge_result, reasons)

    # --- Otherwise, check for anything that makes it INCONCLUSIVE ---
    if challenge_result.audio_quality_status in {"insufficient_audio", "silence", "insufficient_speech"}:
        reasons.append(
            f"Captured response audio was not usable for analysis "
            f"(status={challenge_result.audio_quality_status})."
        )
        return _build(INCONCLUSIVE, challenge_result, reasons)

    if challenge_result.transcription_status != "ok":
        reasons.append(
            f"Speech-to-text could not produce a usable transcript "
            f"(status={challenge_result.transcription_status})."
        )
        return _build(INCONCLUSIVE, challenge_result, reasons)

    if challenge_result.match_status == "partial_match":
        reasons.append(
            f"Response content only partially matched the challenge "
            f"(similarity ratio={challenge_result.match_ratio})."
        )
        return _build(INCONCLUSIVE, challenge_result, reasons)

    if challenge_result.spoof_risk_score is None:
        reasons.append("Spoof-risk score for the response clip could not be computed.")
        return _build(INCONCLUSIVE, challenge_result, reasons)

    if ambient_status is not None and ambient_spoof_score is None:
        reasons.append(f"Ambient/passive analysis was inconclusive (status={ambient_status}).")
        return _build(INCONCLUSIVE, challenge_result, reasons)

    # --- Best available case ---
    reasons.append(
        f"Response content matched the challenge (ratio={challenge_result.match_ratio}) and "
        f"available spoof-risk score(s) were below the model's operating threshold."
    )
    return _build(LOW_SIGNAL_OF_CONCERN, challenge_result, reasons)


def _any_score_flags_spoof(
    challenge_score, challenge_threshold, ambient_score, ambient_threshold,
) -> tuple:
    if challenge_score is not None and challenge_threshold is not None and challenge_score >= challenge_threshold:
        return True, (
            f"The challenge response clip's spoof-risk score ({challenge_score:.4f}) is at or above "
            f"the model's operating threshold ({challenge_threshold:.4f})."
        )
    if ambient_score is not None and ambient_threshold is not None and ambient_score >= ambient_threshold:
        return True, (
            f"The ambient/passive analysis's spoof-risk score ({ambient_score:.4f}) is at or above "
            f"the model's operating threshold ({ambient_threshold:.4f})."
        )
    return False, ""


def _build(tier: str, challenge_result: ChallengeResult, reasons: list) -> CombinedAssessment:
    disclaimers = [LIVE_CLONING_CAVEAT] + list(challenge_result.warnings)
    # de-duplicate while preserving order (LIVE_CLONING_CAVEAT is already
    # in challenge_result.warnings too, since session.py always adds it)
    seen = set()
    deduped = []
    for d in disclaimers:
        if d not in seen:
            deduped.append(d)
            seen.add(d)

    return CombinedAssessment(
        overall_tier=tier,
        explanation=" ".join(reasons),
        recommendation=RECOMMENDATION_TEXT[tier],
        disclaimers=deduped,
    )
