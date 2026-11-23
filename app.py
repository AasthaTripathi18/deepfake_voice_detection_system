"""
Streamlit front end for the deepfake-voice-detection research prototype.

Run from the repository root:
    streamlit run streamlit_app/app.py

This file only arranges widgets and moves data between them. All model,
streaming and decision work happens in streamlit_app/services.py; all
drawing helpers are in streamlit_app/ui_components.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Pre-load onnxruntime before sklearn/scipy to prevent Windows C++ runtime DLL crash
    import onnxruntime  # noqa: F401
except ImportError:
    pass

import streamlit as st  # noqa: E402

from streamlit_app import services, ui_components as ui  # noqa: E402
from streamlit_app.audio_io import decode_uploaded_audio, probe_audio, describe_conversion  # noqa: E402

st.set_page_config(page_title="Voice-spoof research prototype", page_icon="🧪", layout="wide")

AUDIO_TYPES = ["wav", "flac", "ogg", "mp3"]
SUPPORTS_MIC = hasattr(st, "audio_input")

for key, default in {
    "clip": None, "clip_id": 0, "challenge_service": None, "challenge_service_model": None,
    "challenge_text": None, "challenge_outcome": None, "challenge_id": 0,
}.items():
    st.session_state.setdefault(key, default)


@st.cache_resource(show_spinner="Loading model…")
def cached_model(model_dir: str):
    return services.load_model(model_dir)


# ------------------------------------------------------------------ header
st.title("Voice-spoof detection — research prototype")
ui.render_research_notice()

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    if (ROOT / "models/lcnn_clean").exists():
        default_model = "models/lcnn_clean"
    elif (ROOT / "models/mlp_lfcc240_clean").exists():
        default_model = "models/mlp_lfcc240_clean"
    elif (ROOT / "models/mlp_fused480_clean").exists():
        default_model = "models/mlp_fused480_clean"
    elif (ROOT / "models/mlp_mfcc240_clean").exists():
        default_model = "models/mlp_mfcc240_clean"
    elif (ROOT / "models/ensemble_mfcc240_clean").exists():
        default_model = "models/ensemble_mfcc240_clean"
    elif (ROOT / "models/svm_mfcc240_clean").exists():
        default_model = "models/svm_mfcc240_clean"
    else:
        default_model = "models/svm_baseline_clean"
    model_dir = st.text_input("Trained model folder", value=default_model,
                              help="Output folder of trained model (e.g. models/lcnn_clean, models/mlp_lfcc240_clean, models/mlp_fused480_clean)")
    handle = None
    try:
        handle = cached_model(model_dir)
        st.caption(f"`{handle.model_type}` · features: `{handle.feature_key}` · "
                   f"operating threshold: **{handle.operating_threshold:.4f}**")
    except services.ModelLoadError as exc:
        st.error(str(exc))

    st.header("Analysis settings")
    is_2d_spec = handle is not None and handle.feature_key == "lfcc_spectrogram_2d"
    default_window = 4.0 if is_2d_spec else 2.0
    default_thresh = float(handle.operating_threshold) if handle is not None else 0.5
    default_thresh = float(min(0.9999, max(0.0001, default_thresh)))

    agg_options = ["mean", "median", "p75", "worst_window"]
    agg_labels = {
        "mean": "Mean score (Recommended — lowest false alarms on genuine voice)",
        "median": "Median score (Robust against brief noise spikes)",
        "p75": "75th percentile (Security-focused)",
        "worst_window": "Worst window (Strict alarm-centric — high false alarms)",
    }
    agg_rule = st.selectbox(
        "Window aggregation rule",
        options=agg_options,
        index=0,
        format_func=lambda x: agg_labels.get(x, x),
        help="How scores from overlapping windows are combined into a clip-level decision. "
             "Mean and Median dramatically reduce false alarms on genuine human speech.",
    )
    window_sec = st.slider(
        "Analysis window (s)",
        1.0,
        4.0,
        default_window,
        0.5,
        key=f"window_sec_{handle.feature_key if handle else 'default'}",
        help="LCNN models are trained on 4.0s (400-frame) spectrograms; vector models use 2.0s windows.",
    )
    hop_sec = st.slider("Hop between windows (s)", 0.25, float(window_sec), min(1.0, float(window_sec)), 0.25)
    operating_threshold = st.slider(
        "Operating threshold (Sensitivity)",
        min_value=0.0001,
        max_value=0.9999,
        value=default_thresh,
        step=0.0001,
        format="%.4f",
        key=f"op_threshold_{model_dir}",
        help="Decision boundary for High Risk. Lower = more sensitive (catches subtle synthetic spoofs). Higher = strict (avoids false alarms on real-world laptop microphones).",
    )
    low_ratio = st.slider("‘Low risk’ ceiling (fraction of operating threshold)", 0.1, 0.9, 0.5, 0.05,
                          help="Configurable display boundary. Not independently validated; see the "
                               "About tab.")
    min_speech = st.slider("Minimum speech for a confident reading (s)", 0.5, 6.0, 2.0, 0.5)
    match_training = st.checkbox(
        "Match training preprocessing (silence trim + -3 dBFS normalization)",
        value=True,
        help="Aligns live microphone audio energy with the training distribution.",
    )
    alert_enabled = st.checkbox("Play an alert sound on High risk", value=True)

settings = services.AnalysisSettings(
    window_sec=window_sec, hop_sec=hop_sec,
    low_risk_score_ratio=low_ratio, min_reliable_speech_duration_sec=min_speech,
    aggregation_rule=agg_rule,
)

if handle is None:
    st.error("No model is loaded, so nothing can be analyzed yet. Train one with "
             "`scripts/train_model.py`, then enter its folder in the sidebar.")
    st.stop()

thresholds = settings.thresholds(handle, operating_threshold=operating_threshold)
with st.sidebar:
    with st.expander("How the thresholds were chosen"):
        st.write(thresholds.explain())


def current_report():
    clip = st.session_state.clip
    if clip is None:
        return None
    outcome = st.session_state.challenge_outcome
    tier = outcome.assessment.overall_tier if outcome is not None else None
    return services.assess_clip(clip, thresholds, challenge_tier=tier, aggregation_rule=settings.aggregation_rule)


status_slot = st.container()

tab_analyze, tab_challenge, tab_about = st.tabs(["1 · Analyze audio", "2 · Challenge-response (optional)", "About & limits"])


# ------------------------------------------------------------ analyze tab
def run_analysis(uploaded, source_label: str) -> None:
    with st.status("Analyzing audio…", expanded=True) as status:
        try:
            st.write("Decoding audio…")
            waveform, sr = decode_uploaded_audio(uploaded)
            notes = [describe_conversion(probe_audio(uploaded))]
            waveform, more = services.prepare_waveform(waveform, sr, match_training=match_training)
            notes += more
            duration = len(waveform) / sr
            total = services.estimate_window_count(duration, settings.window_sec, settings.hop_sec)

            bar = st.progress(0.0, text="Scoring windows…")
            windows = []
            for i, wr in enumerate(services.iter_analysis(
                    handle, waveform, sr, source_label, settings.window_sec, settings.hop_sec)):
                windows.append(wr)
                bar.progress(min(1.0, (i + 1) / total), text=f"Window {i + 1} of about {total}: "
                                                            f"{wr.status.replace('_', ' ')}")
            bar.progress(1.0, text="Done")

            st.session_state.clip = services.ClipAnalysis(
                source_label=source_label, duration_sec=duration, window_sec=settings.window_sec,
                hop_sec=settings.hop_sec, windows=windows, notes=notes,
            )
            st.session_state.clip_id += 1
            st.session_state.challenge_outcome = None   # new audio = new session
            status.update(label=f"Analysis complete: {len(windows)} window(s)", state="complete")
        except Exception as exc:  # noqa: BLE001 - show any failure readably instead of a stack trace
            status.update(label="Analysis failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            st.caption("If this mentions librosa or soundfile, run `pip install -r requirements.txt` "
                       "in the environment that launched Streamlit.")


with tab_analyze:
    st.subheader("Choose the audio")
    methods = ["Upload an audio file"]
    if SUPPORTS_MIC:
        methods.append("Record with the browser microphone (analyzed after recording)")
    method = st.radio("Audio input", methods, horizontal=True, label_visibility="collapsed")

    uploaded, source_label = None, None
    if method.startswith("Upload"):
        uploaded = st.file_uploader("WAV, FLAC, OGG or MP3 (the first 60 s are analyzed)", type=AUDIO_TYPES)
        if uploaded is not None:
            source_label = f"Uploaded file: {uploaded.name}"
    else:
        st.caption("Your browser will ask for microphone permission. The clip is recorded first and then "
                   "analyzed. This is a controlled demo, not continuous monitoring of anything.")
        uploaded = st.audio_input("Record a few seconds of speech")
        if uploaded is not None:
            source_label = "Browser microphone recording (recorded first, then analyzed)"
    if not SUPPORTS_MIC:
        st.caption("Browser recording needs Streamlit 1.40 or newer; only file upload is available in this "
                   "installation.")

    if uploaded is not None:
        st.audio(uploaded)
        st.caption("Analyzing new audio starts a new session and clears any earlier challenge result.")
        if st.button("Analyze this audio", type="primary"):
            run_analysis(uploaded, source_label)

    clip = st.session_state.clip
    if clip is not None:
        st.divider()
        st.subheader("Details")
        ui.render_source_box(clip.source_label, clip.notes, clip.duration_sec, clip.window_sec, clip.hop_sec)
        ui.render_window_details(current_report(), clip.window_sec)


# ---------------------------------------------------------- challenge tab
with tab_challenge:
    st.subheader("Challenge-response")
    st.markdown(
        "Ask the speaker to say a **fresh random phrase**, record their reply, and check whether the right "
        "words were said and whether the reply itself looks synthetic. Passing is **one supporting signal**: "
        "it can raise concern but never lowers it, and it never verifies identity."
    )
    st.warning(services.LIVE_CLONING_CAVEAT)

    if (st.session_state.challenge_service is None
            or st.session_state.challenge_service_model != handle.model_dir):
        st.session_state.challenge_service = services.ChallengeService(handle)
        st.session_state.challenge_service_model = handle.model_dir
        st.session_state.challenge_text = None
    svc = st.session_state.challenge_service

    mode_labels = {
        services.TRANSCRIBER_NONE: "No transcription (content check unavailable)",
        services.TRANSCRIBER_MANUAL: "Manual transcript (demo: you type what was said)",
        services.TRANSCRIBER_SPEECH_RECOGNITION: "SpeechRecognition (needs the package and an internet connection)",
    }
    mode = st.radio("How should the reply's words be checked?", list(mode_labels),
                    format_func=mode_labels.get, index=1)
    if mode == services.TRANSCRIBER_SPEECH_RECOGNITION:
        st.caption("This option sends the response audio to an online service, so it leaves your computer."
                   + ("" if services.speech_recognition_available()
                      else " The `SpeechRecognition` package is not installed, so it will be skipped."))
    if mode == services.TRANSCRIBER_MANUAL:
        st.caption("Demo only: this is not speech recognition. It exercises the matching and decision "
                   "logic so you can see how each outcome is handled.")

    if st.button("Issue a new challenge"):
        st.session_state.challenge_text = svc.issue()
        st.session_state.challenge_id += 1
        st.session_state.challenge_outcome = None

    if st.session_state.challenge_text and svc.has_pending_challenge:
        st.markdown(f"### Please say: `{st.session_state.challenge_text}`")
        st.caption("The phrase is shown on screen only; this app does not read it aloud. Each challenge "
                   "can be answered once.")

        cid = st.session_state.challenge_id
        reply_methods = ["Upload a recording"] + (["Record with the browser microphone"] if SUPPORTS_MIC else [])
        reply_method = st.radio("Reply input", reply_methods, horizontal=True, key=f"reply_method_{cid}")
        reply, reply_label = None, None
        if reply_method.startswith("Upload"):
            reply = st.file_uploader("Reply recording", type=AUDIO_TYPES, key=f"reply_up_{cid}")
            if reply is not None:
                reply_label = f"Uploaded reply file: {reply.name}"
        else:
            reply = st.audio_input("Record the reply", key=f"reply_mic_{cid}")
            if reply is not None:
                reply_label = "Browser microphone recording (challenge reply)"

        manual_text = ""
        if mode == services.TRANSCRIBER_MANUAL:
            manual_text = st.text_input("What was said (demo transcript)", key=f"manual_{cid}")

        if reply is not None and st.button("Check the reply", type="primary"):
            with st.spinner("Checking the reply…"):
                try:
                    waveform, sr = decode_uploaded_audio(reply)
                    st.session_state.challenge_outcome = svc.run(
                        waveform, sr, reply_label, mode=mode, manual_text=manual_text)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{type(exc).__name__}: {exc}")
    elif st.session_state.challenge_outcome is None:
        st.caption("No challenge is active. Press “Issue a new challenge” to begin.")

    if st.session_state.challenge_outcome is not None:
        st.divider()
        ui.render_challenge_outcome(st.session_state.challenge_outcome)
        if st.session_state.clip is None:
            st.caption("Analyze some audio in the first tab to see an overall status that combines both signals.")


# -------------------------------------------------------------- about tab
with tab_about:
    st.subheader("What this is")
    st.markdown(
        "- A **research prototype** built on a published deepfake-voice-detection paper, extended to test "
        "how detection behaves under phone-style audio degradation and to explore a challenge-response step.\n"
        "- It scores **audio you give it** (an uploaded file, or a clip recorded in this page). Recording is "
        "*record, then analyze*; nothing is captured continuously.\n"
        "- The optional challenge-response step uses the same two input routes."
    )
    st.subheader("What it is not")
    st.markdown(
        "- It **cannot prove who is speaking** and never labels anyone genuine or verified.\n"
        "- It **does not monitor cellular calls** or any other call; ordinary apps cannot access that audio.\n"
        "- It **does not disconnect or block calls**, and a warning does not by itself prevent fraud.\n"
        "- The score is a raw model output, **not a calibrated probability of fraud**."
    )
    st.subheader("Known limits")
    st.markdown(
        "- A live voice-cloning system could repeat a freshly issued phrase in the cloned voice, so passing "
        "the challenge is not reassuring on its own.\n"
        "- The model was trained on one benchmark corpus; short streaming windows and phone-channel audio "
        "have not been validated by this app. Treat any status as a research observation.\n"
        "- The uncertainty figure is a heuristic display aid, not a statistical confidence interval.\n"
        "- The clip-level status uses the most cautious window; that aggregation is a design choice."
    )
    st.subheader("How thresholds work")
    st.write(thresholds.explain())


# ------------------------------------------------------- status (top slot)
report = current_report()
outcome = st.session_state.challenge_outcome
with status_slot:
    ui.render_status_card(report, outcome, min_speech_sec=settings.min_reliable_speech_duration_sec)
    if report is not None:
        token = f"{st.session_state.clip_id}:{outcome.result.challenge_id if outcome else 'none'}:" \
                f"{report.overall.status}"
        ui.maybe_play_alert(report.display, token, alert_enabled)
    st.divider()
