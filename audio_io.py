"""
Turns whatever st.file_uploader / st.audio_input hand back (both expose
.getvalue() -> bytes, and usually a .name) into a (waveform, sample_rate)
pair — by writing to a temp file and calling
src.preprocessing.cleaning.load_audio, so the actual decoding logic lives
in exactly ONE place in this project rather than being duplicated here.
This module is a thin I/O adapter, not a decoder in its own right.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.preprocessing.cleaning import load_audio, DEFAULT_TARGET_SR


def decode_uploaded_audio(uploaded_obj, target_sr: int = DEFAULT_TARGET_SR):
    """
    uploaded_obj: the object returned by st.file_uploader(...) or
    st.audio_input(...) — both are file-like with .getvalue(); a `.name`
    attribute is used if present (to preserve the file extension for
    format sniffing) and defaults to a generic ".wav" name otherwise,
    since st.audio_input's recordings are WAV regardless of what name (if
    any) is attached.
    """
    data = uploaded_obj.getvalue()
    suffix = Path(getattr(uploaded_obj, "name", None) or "audio.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        waveform, sr = load_audio(tmp_path, target_sr=target_sr)
    finally:
        os.unlink(tmp_path)

    return waveform, sr


def probe_audio(uploaded_obj):
    """
    Best-effort peek at the ORIGINAL file's sample rate / channel count, so
    the UI can tell the user what conversion was applied before analysis.
    Returns None if the format can't be probed (e.g. some MP3 builds) —
    the audio may still decode fine via load_audio; the UI then just omits
    the "converted from ..." note rather than guessing.
    """
    import io

    try:
        import soundfile as sf
        info = sf.info(io.BytesIO(uploaded_obj.getvalue()))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_sec": float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
        }
    except Exception:  # noqa: BLE001 - probing is optional
        return None


def describe_conversion(probe, target_sr: int = DEFAULT_TARGET_SR) -> str:
    """One-line, plain-language description of resampling/downmixing applied."""
    if not probe:
        return f"Converted to mono, {target_sr} Hz for analysis (original format could not be probed)."
    parts = []
    if probe["channels"] != 1:
        parts.append(f"{probe['channels']} channels -> mono")
    if probe["sample_rate"] != target_sr:
        parts.append(f"{probe['sample_rate']} Hz -> {target_sr} Hz")
    if not parts:
        return f"Already mono, {target_sr} Hz; no conversion needed."
    return "Converted for analysis: " + ", ".join(parts) + "."
