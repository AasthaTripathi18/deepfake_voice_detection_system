#!/usr/bin/env python
"""
Runs one challenge-response round end to end: issue a fresh phrase,
display it, capture a response from a chosen audio source, check content,
score spoof-risk on the response clip, and combine into a hedged
assessment — printing every step, including the disclaimers, rather than
just a final yes/no.

No transcription backend is bundled by default (see
src/challenge_response/transcription.py) — this script uses
FixedResponseTranscriber for demonstration unless --use-speech-recognition
is passed (which requires `pip install SpeechRecognition` and was not
verified in this project's own development environment).

Examples:
    # Demo with a scripted transcript (no real ASR needed):
    python scripts/run_challenge_response_demo.py \
        --model-dir models/svm_baseline_clean \
        --source file --audio-file some_response_clip.wav \
        --demo-transcript "TIGER HARBOR 7 3 1"

    # Real microphone + real (unverified-in-this-project) ASR backend:
    python scripts/run_challenge_response_demo.py \
        --model-dir models/svm_baseline_clean \
        --source mic --use-speech-recognition
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.scorer import Scorer  # noqa: E402
from src.streaming.audio_source import FileReplaySource, MicrophoneSource  # noqa: E402
from src.challenge_response.session import ChallengeResponseSession  # noqa: E402
from src.challenge_response.decision import combine_assessment  # noqa: E402
from src.challenge_response.transcription import FixedResponseTranscriber, NullTranscriber  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source", choices=["file", "mic"], required=True)
    parser.add_argument("--audio-file", help="Required if --source file")
    parser.add_argument("--mic-device", default=None)
    parser.add_argument("--max-duration-sec", type=float, default=6.0)
    parser.add_argument("--num-words", type=int, default=2)
    parser.add_argument("--digit-count", type=int, default=3)
    parser.add_argument(
        "--demo-transcript", default=None,
        help="If set, uses FixedResponseTranscriber returning exactly this text, "
             "regardless of captured audio — for demonstrating the pipeline without "
             "a real ASR backend. If omitted and --use-speech-recognition is not "
             "passed, no content check is possible (NullTranscriber).",
    )
    parser.add_argument(
        "--use-speech-recognition", action="store_true",
        help="Use SpeechRecognitionTranscriber (requires `pip install SpeechRecognition`; "
             "not verified in this project's own development environment).",
    )
    args = parser.parse_args()

    scorer = Scorer(args.model_dir)
    print(f"Loaded model: model_type={scorer.train_config['model_type']} "
          f"feature_key={scorer.feature_key} operating_threshold={scorer.threshold:.4f}")

    if args.use_speech_recognition:
        from src.challenge_response.transcription import SpeechRecognitionTranscriber
        transcriber = SpeechRecognitionTranscriber()
    elif args.demo_transcript is not None:
        transcriber = FixedResponseTranscriber(fixed_text=args.demo_transcript)
    else:
        transcriber = NullTranscriber()
        print("No transcriber configured — content check will report 'backend_unavailable'.")

    session = ChallengeResponseSession(
        scorer=scorer, transcriber=transcriber,
        num_words=args.num_words, digit_count=args.digit_count,
    )

    challenge = session.issue_challenge()
    print("-" * 100)
    print(session.present_challenge())
    print("-" * 100)

    if args.source == "file":
        if not args.audio_file:
            parser.error("--audio-file is required when --source file")
        source = FileReplaySource(path=args.audio_file, chunk_sec=0.25)
    else:
        source = MicrophoneSource(device=args.mic_device)

    print(f"Listening for response via: {source.label} (up to {args.max_duration_sec:.1f}s)...")
    result = session.run_challenge(source, max_duration_sec=args.max_duration_sec)
    source.close()

    print("-" * 100)
    print(f"Captured {result.captured_duration_sec:.2f}s of audio (quality status: {result.audio_quality_status})")
    print(f"Transcription status: {result.transcription_status} -> \"{result.transcribed_text}\"")
    print(f"Content match: {result.match_status} (ratio={result.match_ratio})")
    print(f"Response clip spoof-risk score: {result.spoof_risk_score} "
          f"(operating threshold: {result.spoof_operating_threshold})")
    print(f"Processing delay: {result.processing_delay_sec * 1000:.1f}ms")

    assessment = combine_assessment(result)
    print("-" * 100)
    print(f"OVERALL TIER: {assessment.overall_tier}")
    print(f"identity_verified: {assessment.identity_verified}  (this is ALWAYS False — see decision.py)")
    print(f"Explanation: {assessment.explanation}")
    print(f"Recommendation: {assessment.recommendation}")
    print("Disclaimers:")
    for d in assessment.disclaimers:
        print(f"  - {d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
