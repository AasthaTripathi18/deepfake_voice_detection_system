"""
Turns a RiskAssessment into what a person actually sees/hears. Contains
NO decision logic — it only formats what risk_assessment.py already
decided, and never disconnects a call or takes any action on the user's
behalf.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .risk_assessment import RiskAssessment, HIGH_RISK, NEEDS_VERIFICATION, INSUFFICIENT_EVIDENCE

DISCLAIMER = (
    "This is an automated indicator based on audio analysis, not proof of who is calling. "
    "It cannot and does not disconnect calls or block anyone, and does not by itself prevent fraud."
)

CALLBACK_RECOMMENDATION = (
    "If anything sensitive is being requested (money, passwords, personal data), hang up and "
    "independently call back using a phone number you already know and trust — never a number "
    "given to you during this call."
)

_STATUS_ICON = {
    "Low risk": "🟢",
    "Needs verification": "🟡",
    "High risk": "🔴",
    "Insufficient evidence": "⚪",
}


@dataclass
class DisplayPayload:
    status_label: str
    icon: str
    short_reasons: list = field(default_factory=list)
    score_line: str = ""             # e.g. "Model output: 0.9421 (uncalibrated, see note)"
    calibration_note: str = ""
    warning_banner: str = ""          # populated ONLY for High risk
    disclaimer: str = DISCLAIMER
    callback_recommendation: str = CALLBACK_RECOMMENDATION
    show_alert: bool = False


def build_display(assessment: RiskAssessment) -> DisplayPayload:
    if assessment.raw_score is not None:
        score_line = (
            f"Model output: {assessment.raw_score:.4f} "
            f"(uncalibrated — see note; operating threshold: {assessment.operating_threshold:.4f})"
        )
    else:
        score_line = "Model output: not available for this audio."

    warning_banner = ""
    if assessment.status == HIGH_RISK:
        warning_banner = (
            "⚠ HIGH RISK: This audio shows strong indicators consistent with synthetic/spoofed "
            "speech. " + CALLBACK_RECOMMENDATION
        )

    return DisplayPayload(
        status_label=assessment.status,
        icon=_STATUS_ICON.get(assessment.status, ""),
        short_reasons=list(assessment.reasons),
        score_line=score_line,
        calibration_note=assessment.calibration_note,
        warning_banner=warning_banner,
        show_alert=(assessment.status == HIGH_RISK),
    )


def trigger_alert(display: DisplayPayload, sound_backend=None) -> bool:
    """
    Plays an alert sound for a High-risk display, IF the calling
    application supplies one — same pattern as
    src/challenge_response/challenge_generator.py's speak_challenge():
    no sound is bundled with this project. sound_backend is a callable
    taking no arguments, expected to play whatever alert sound/tone the
    application wants.

    Returns whether an alert was (or would have been) triggered, so a
    caller can distinguish "no alert needed" from "alert needed but no
    sound_backend was configured" for their own logging/UI purposes.
    """
    if not display.show_alert:
        return False
    if sound_backend is not None:
        sound_backend()
    return True


def render_plain_text(display: DisplayPayload) -> str:
    """A simple, complete text rendering for a console/log/basic UI —
    real applications will likely build their own richer UI from the
    DisplayPayload fields directly instead of using this."""
    lines = [f"{display.icon} {display.status_label}"]
    if display.warning_banner:
        lines.append(display.warning_banner)
    lines.append(display.score_line)
    lines.append(f"(Note: {display.calibration_note})")
    if display.short_reasons:
        lines.append("Reasons:")
        lines.extend(f"  - {r}" for r in display.short_reasons)
    if display.status_label != HIGH_RISK:
        # avoid repeating the callback line twice when the warning banner already includes it
        lines.append(display.callback_recommendation)
    lines.append(display.disclaimer)
    return "\n".join(lines)
