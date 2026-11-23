"""
Streamlit rendering helpers. This module only DRAWS things: it receives
plain dataclasses from streamlit_app.services and never imports or calls
the model, the streaming detector, the preprocessing code, or the
challenge-response session directly. Keeping it that way is what makes
the model side testable without a UI and the UI side replaceable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.decision_logic.presentation import trigger_alert
from streamlit_app import services
from streamlit_app.beep import generate_beep_wav_bytes

RESEARCH_NOTICE = (
    "**Experimental research tool.** It estimates whether a piece of audio *sounds synthetic* and "
    "shows how much evidence it had. It **cannot prove who is speaking**, and it **does not monitor "
    "phone calls**: it only analyzes audio you upload or record on this page. It never ends or "
    "blocks calls, and a warning from it does not by itself prevent fraud."
)


def render_research_notice() -> None:
    st.info(RESEARCH_NOTICE, icon="🧪")


# ------------------------------------------------------------------- status

def render_status_card(report, challenge_outcome, min_speech_sec: float) -> None:
    st.subheader("Current status")
    if report is None:
        st.info("⚪ **No audio analyzed yet.** Upload a file or record a short clip in the "
                "“Analyze audio” tab.")
        return

    overall, display = report.overall, report.display
    headline = f"{display.icon} {display.status_label}"
    box = {
        services.LOW_RISK: st.success,
        services.NEEDS_VERIFICATION: st.warning,
        services.HIGH_RISK: st.error,
        services.INSUFFICIENT_EVIDENCE: st.info,
    }.get(overall.status, st.info)
    box(f"**{headline}**")

    if overall.status == services.HIGH_RISK:
        st.error(
            "### 🚨 ALARM: HIGH SPOOF RISK DETECTED\n"
            "The model has detected strong indicators of synthetic or AI-generated speech. "
            "Exercise caution: do not trust sensitive requests (funds transfer, password reset, personal verification) on this voice channel.",
            icon="🚨",
        )
    elif overall.status == services.NEEDS_VERIFICATION:
        st.warning(
            "### ⚠️ BORDERLINE READING: LIVENESS VERIFICATION REQUIRED\n"
            f"The voice sample scored in the review margin (Risk Band: **{getattr(overall, 'risk_band', 'Needs Verification')}**).\n\n"
            "**Operational Action:** To prevent falsely rejecting a legitimate caller, switch to **Tab 2 · Challenge-response (optional)** above and issue a fresh challenge phrase to verify live presence and stop replay attacks.",
            icon="⚠️",
        )
    elif display.warning_banner:
        st.error(display.warning_banner, icon="🚨")

    st.markdown("**Why:**")
    for reason in display.short_reasons:
        st.markdown(f"- {reason}")

    if overall.status == services.INSUFFICIENT_EVIDENCE:
        _render_insufficient_evidence(report, min_speech_sec)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Model output (uncalibrated)",
        f"{overall.raw_score:.4f}" if overall.raw_score is not None else "n/a",
        help="Raw model output, higher = more indicative of synthetic audio. It is NOT a "
             "calibrated probability of fraud.",
    )
    c2.metric("Operating threshold", f"{overall.operating_threshold:.4f}",
              help="Chosen on the model's validation split, not on test data.")
    c3.metric(
        "Uncertainty (heuristic)",
        (f"{report.uncertainty_label} ({report.mean_uncertainty:.2f})"
         if report.mean_uncertainty is not None else "unknown"),
        help="A rough display aid combining closeness to the threshold, how much of each window was "
             "speech, and a noise proxy. It has not been validated as a statistical uncertainty.",
    )
    c4.metric("Speech analyzed", f"{report.unique_speech_sec:.1f} s",
              help=f"Unique speech across windows (minimum {min_speech_sec:.1f}s needed before a "
                   f"low or middling reading is treated as reassuring).")

    if report.uncertainty_label == "high":
        st.warning("**High uncertainty:** the windows behind this result were noisy, short on speech, "
                   "or scored close to the threshold. Treat the status with extra caution.")

    if challenge_outcome is not None:
        tier = challenge_outcome.assessment.overall_tier
        if tier == "low_signal_of_concern":
            st.success("✅ **Liveness Challenge Passed:** The speaker correctly repeated the dynamic challenge prompt without synthetic acoustic artifacts. Verification cleared.")
        elif tier == "elevated_concern":
            st.error("🚨 **Liveness Challenge Failed:** The response failed content matching or exhibited synthetic traits. Verification blocked.")
        else:
            st.info(f"ℹ️ **Liveness Challenge Result:** `{tier}`")

    if hasattr(overall, "recommended_action") and overall.recommended_action:
        st.info(f"**Recommended Action:** `{overall.recommended_action}`")

    st.markdown(f"**What to do:** {display.callback_recommendation}")
    st.caption(display.calibration_note)
    st.caption(display.disclaimer)


def _render_insufficient_evidence(report, min_speech_sec: float) -> None:
    lines = []
    if report.n_windows == 0:
        lines.append("No usable analysis windows were produced.")
    if report.blocked_counts:
        parts = ", ".join(f"{n} × {name.replace('_', ' ')}" for name, n in report.blocked_counts.items())
        lines.append(f"Windows withheld by the quality gate: {parts}.")
    if report.unique_speech_sec < min_speech_sec:
        lines.append(f"Only {report.unique_speech_sec:.1f}s of speech was analyzed; at least "
                     f"{min_speech_sec:.1f}s is needed for a confident reading.")
    lines.append("The tool is deliberately **not guessing**. Try a longer, clearer clip with more "
                 "speech and less background noise.")
    st.markdown("**What's missing:**")
    for line in lines:
        st.markdown(f"- {line}")


# ------------------------------------------------------------------- source

def render_source_box(source_label: str, notes: list, duration_sec: float,
                      window_sec: float, hop_sec: float) -> None:
    st.markdown("**Audio source**")
    icon = "🎙️" if "microphone" in source_label.lower() else "📁"
    st.markdown(f"{icon} {source_label}")
    st.caption(f"Length analyzed: {duration_sec:.1f}s · Analysis window: {window_sec:.1f}s · "
               f"Hop: {hop_sec:.2f}s (windows overlap when hop < window)")
    for note in notes:
        st.caption(f"• {note}")


def render_window_details(report, window_sec: float) -> None:
    if report is None or not report.per_window:
        return
    rows = []
    for w, a in report.per_window:
        rows.append({
            "Window (s)": f"{w.window_start_sec:.1f}–{w.window_end_sec:.1f}",
            "Quality gate": w.status.replace("_", " "),
            "Model output": w.risk_score,
            "Uncertainty": w.uncertainty,
            "Speech in window": w.voiced_fraction,
            "SNR proxy (dB)": w.estimated_snr_db,
            "Delay (ms)": round(w.processing_delay_sec * 1000.0, 1),
            "Window status": a.status,
        })
    df = pd.DataFrame(rows)

    m1, m2, m3 = st.columns(3)
    m1.metric("Windows scored", f"{report.n_scored} of {report.n_windows}")
    m2.metric("Median delay per window",
              f"{report.median_delay_ms:.1f} ms" if report.median_delay_ms is not None else "n/a",
              help="Time from a window being ready to its result, on this machine. Excludes "
                   "recording and upload time.")
    m3.metric("Slowest window",
              f"{report.max_delay_ms:.1f} ms" if report.max_delay_ms is not None else "n/a")

    st.dataframe(
        df, hide_index=True,
        column_config={
            "Model output": st.column_config.ProgressColumn(
                "Model output", min_value=0.0, max_value=1.0, format="%.3f",
                help="Uncalibrated. Empty = the quality gate withheld scoring for this window."),
            "Uncertainty": st.column_config.ProgressColumn(
                "Uncertainty", min_value=0.0, max_value=1.0, format="%.2f",
                help="Heuristic; higher = trust this window less."),
            "Speech in window": st.column_config.ProgressColumn(
                "Speech in window", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )
    st.caption("Empty score = the window was withheld (silence, too little speech, or too short) and is "
               "reported as insufficient evidence instead of being guessed.")

    chart = pd.DataFrame({
        "Model output": [w.risk_score for w, _ in report.per_window],
        "Operating threshold": [report.overall.operating_threshold] * len(report.per_window),
    }, index=[round(w.window_start_sec, 2) for w, _ in report.per_window])
    st.line_chart(chart)
    st.caption("Model output per window against the operating threshold (x-axis: window start, seconds). "
               "Gaps are withheld windows.")
    st.caption(report.policy_note)


# ---------------------------------------------------------------- challenge

def render_challenge_outcome(outcome) -> None:
    r, a = outcome.result, outcome.assessment
    st.markdown("**Challenge result**")
    box = {"elevated_concern": st.error, "inconclusive": st.warning}.get(a.overall_tier, st.info)
    box(f"**{a.overall_tier.replace('_', ' ').capitalize()}**  ·  identity verified: "
        f"**{a.identity_verified}** (this tool never verifies identity)")
    st.markdown(a.explanation)

    c1, c2, c3 = st.columns(3)
    c1.metric("Content match", r.match_status.replace("_", " "),
              help="Whether the transcript matched the phrase that was issued.")
    c2.metric("Response audio quality", r.audio_quality_status.replace("_", " "))
    c3.metric("Response spoof score (uncalibrated)",
              f"{r.spoof_risk_score:.3f}" if r.spoof_risk_score is not None else "n/a")
    st.caption(f"Source: {r.source_label} · captured {r.captured_duration_sec:.1f}s · processed in "
               f"{r.processing_delay_sec * 1000:.0f} ms · transcription: {r.transcription_status}"
               + (f" → “{r.transcribed_text}”" if r.transcribed_text else ""))
    st.markdown(f"**What to do:** {a.recommendation}")
    with st.expander("Caveats for this result", expanded=True):
        for d in a.disclaimers:
            st.markdown(f"- {d}")


# -------------------------------------------------------------------- alert

def maybe_play_alert(display, token: str, enabled: bool) -> None:
    """Plays the alert beep once per new High-risk result (not on every rerun)."""
    seen = st.session_state.setdefault("alerts_played", set())
    if not enabled or not display.show_alert or token in seen:
        return
    seen.add(token)
    fired = []
    trigger_alert(display, sound_backend=lambda: fired.append(True))
    if not fired:
        return
    wav = generate_beep_wav_bytes()
    try:
        st.audio(wav, format="audio/wav", autoplay=True)
    except TypeError:   # older Streamlit without autoplay
        st.audio(wav, format="audio/wav")
        st.caption("Press play to hear the alert.")
