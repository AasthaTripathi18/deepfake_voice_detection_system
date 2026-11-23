#!/usr/bin/env python
"""
Runs the streaming detector over an audio source and, for each window,
prints the transparent risk assessment: status, reasons, uncalibrated
score, and (for High risk) the on-screen warning + an optional alert sound.

Optionally folds in a challenge-response result (run separately via
scripts/run_challenge_response_demo.py, or programmatically) — pass
--challenge-tier to simulate one for this demo without wiring up a live
challenge round.

Example:
    python scripts/run_risk_assessment_demo.py \
        --model-dir models/svm_baseline_clean \
        --source file --audio-file some_clip.wav \
        --window-sec 2.0 --hop-sec 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.scorer import Scorer  # noqa: E402
from src.streaming.audio_source import FileReplaySource, MicrophoneSource  # noqa: E402
from src.streaming.stream_inference import StreamingSpoofDetector  # noqa: E402
from src.decision_logic.risk_assessment import (  # noqa: E402
    assess, DetectorReading, AudioQualityReading, ChallengeReading, ThresholdConfig, SpeechAccumulator,
)
from src.decision_logic.presentation import build_display, trigger_alert, render_plain_text  # noqa: E402


def console_beep():
    """Minimal default alert sound backend: terminal bell. Purely a
    convenience for this CLI demo -- a real application would pass its own
    sound_backend to trigger_alert()."""
    print("\a", end="", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source", choices=["file", "mic"], required=True)
    parser.add_argument("--audio-file")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--low-risk-score-ratio", type=float, default=0.5)
    parser.add_argument("--min-reliable-speech-duration-sec", type=float, default=2.0)
    parser.add_argument(
        "--challenge-tier", choices=["elevated_concern", "inconclusive", "low_signal_of_concern"], default=None,
        help="Optionally simulate a challenge-response result for this demo.",
    )
    parser.add_argument("--alert-sound", action="store_true", help="Play a terminal bell on High risk windows")
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    scorer = Scorer(args.model_dir)
    thresholds = ThresholdConfig(
        operating_threshold=scorer.threshold,
        low_risk_score_ratio=args.low_risk_score_ratio,
        min_reliable_speech_duration_sec=args.min_reliable_speech_duration_sec,
    )
    print("Threshold configuration:")
    print(" ", thresholds.explain())
    print("-" * 100)

    if args.source == "file":
        if not args.audio_file:
            parser.error("--audio-file is required when --source file")
        source = FileReplaySource(path=args.audio_file, chunk_sec=0.25)
    else:
        source = MicrophoneSource()
    print(f"Audio source: {source.label}")

    detector = StreamingSpoofDetector(scorer=scorer, source=source, window_sec=args.window_sec, hop_sec=args.hop_sec)
    challenge_reading = ChallengeReading(overall_tier=args.challenge_tier) if args.challenge_tier else None

    accumulator = SpeechAccumulator()  # counts overlapping windows' audio only once
    sound_backend = console_beep if args.alert_sound else None

    try:
        for window_result in detector.run(max_windows=args.max_windows):
            cumulative_speech_sec = accumulator.add(
                window_result.window_start_sec, window_result.window_end_sec,
                window_result.voiced_fraction,
                scored=window_result.status in ("ok", "high_noise_low_confidence"),
            )

            assessment = assess(
                detector=DetectorReading(spoof_score=window_result.risk_score),
                audio_quality=AudioQualityReading(
                    status=window_result.status, analyzed_speech_duration_sec=cumulative_speech_sec,
                ),
                thresholds=thresholds,
                challenge=challenge_reading,
            )
            display = build_display(assessment)
            trigger_alert(display, sound_backend=sound_backend)

            print(f"[window {window_result.window_start_sec:.2f}-{window_result.window_end_sec:.2f}s] "
                  f"source={window_result.source_label} delay={window_result.processing_delay_sec * 1000:.1f}ms")
            print(render_plain_text(display))
            print("-" * 100)
    finally:
        source.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
