#!/usr/bin/env python
"""
Runs the near-real-time streaming pipeline and prints each analysis
window's result as it's produced — this is the "show the analysis window
and measured delay" + "make the audio source visible" requirement made
concrete and runnable, ahead of any graphical UI.

Two source options:
  --source file --audio-file path/to/clip.wav   (file replay; no mic needed)
  --source mic                                    (requires `pip install sounddevice`)

Neither option is, or claims to be, a cellular-call capture — see
src/streaming/__init__.py's module docstring for the hard scope limit.

Example:
    python scripts/run_streaming_demo.py \
        --model-dir models/svm_baseline_clean \
        --source file --audio-file data/manifests/some_eval_clip.flac \
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


def print_result(result) -> None:
    score_str = f"{result.risk_score:.4f}" if result.risk_score is not None else "  --  "
    tier_str = f" ({result.risk_tier})" if result.risk_tier else ""
    uncertainty_str = f"{result.uncertainty:.2f}" if result.uncertainty is not None else " -- "
    print(
        f"[{result.source_label}] "
        f"window=[{result.window_start_sec:6.2f}s - {result.window_end_sec:6.2f}s] "
        f"(full={result.is_full_window}) "
        f"status={result.status:<24s} "
        f"risk_score={score_str}{tier_str} "
        f"uncertainty={uncertainty_str} "
        f"delay={result.processing_delay_sec * 1000:6.1f}ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source", choices=["file", "mic"], required=True)
    parser.add_argument("--audio-file", help="Required if --source file")
    parser.add_argument("--mic-device", default=None, help="Optional input device name/index if --source mic")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--min-voiced-fraction", type=float, default=0.15)
    parser.add_argument("--min-snr-db", type=float, default=6.0)
    parser.add_argument("--risk-tier-bounds", nargs=2, type=float, default=None,
                         metavar=("LOW", "HIGH"), help="Optional; enables the low/elevated/high risk_tier label")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--real-time-replay", action="store_true",
                         help="For --source file: pace chunk delivery to simulate real capture timing")
    args = parser.parse_args()

    scorer = Scorer(args.model_dir)
    print(f"Loaded model: model_type={scorer.train_config['model_type']} "
          f"feature_key={scorer.feature_key} operating_threshold={scorer.threshold:.4f}")

    if args.source == "file":
        if not args.audio_file:
            parser.error("--audio-file is required when --source file")
        source = FileReplaySource(path=args.audio_file, chunk_sec=0.25, real_time=args.real_time_replay)
    else:
        source = MicrophoneSource(chunk_sec=0.25, device=args.mic_device)

    print(f"Audio source: {source.label}")
    print("-" * 100)

    detector = StreamingSpoofDetector(
        scorer=scorer,
        source=source,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        min_duration_sec=args.min_duration_sec,
        min_voiced_fraction=args.min_voiced_fraction,
        min_snr_db=args.min_snr_db,
        risk_tier_bounds=tuple(args.risk_tier_bounds) if args.risk_tier_bounds else None,
    )

    try:
        for result in detector.run(max_windows=args.max_windows):
            print_result(result)
    finally:
        source.close()

    print("-" * 100)
    print(
        "Reminder: risk_score/risk_tier above are EXPERIMENTAL spoof-risk indicators, "
        "not a verdict on caller identity, and this demo only ever analyzed audio from "
        "the source labeled above — never a cellular call."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
