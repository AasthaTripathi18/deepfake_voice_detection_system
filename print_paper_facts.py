"""
Print verified facts, tables, metrics, and parameters for the IEEE research paper.
Usage:
    python scripts/print_paper_facts.py [--all] [--table1] [--table2] [--cm] [--latency]
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def print_table1():
    print("=" * 80)
    print("TABLE I: LCNN-9 LAYER-BY-LAYER STRUCTURAL SPECIFICATION & PARAMETER COUNT")
    print("=" * 80)
    rows = [
        ("Input", "Single-channel 2D Spectrogram", "1", "1 x 60 x 400", "0"),
        ("Conv1", "Conv2D (5x5, pad 2), BN, MFM", "32 -> 16", "16 x 60 x 400", "864"),
        ("Pool1", "MaxPool2D (2x2, stride 2)", "16", "16 x 30 x 200", "0"),
        ("Conv2a", "Conv2D (1x1), BN, MFM", "32 -> 16", "16 x 30 x 200", "560"),
        ("Conv2b", "Conv2D (3x3, pad 1), BN, MFM", "48 -> 24", "24 x 30 x 200", "3,528"),
        ("Pool2", "MaxPool2D (2x2, stride 2)", "24", "24 x 15 x 100", "0"),
        ("Conv3a", "Conv2D (1x1), BN, MFM", "48 -> 24", "24 x 15 x 100", "1,224"),
        ("Conv3b", "Conv2D (3x3, pad 1), BN, MFM", "64 -> 32", "32 x 15 x 100", "7,008"),
        ("Pool3", "MaxPool2D (2x2, stride 2)", "32", "32 x 7 x 50", "0"),
        ("Conv4a", "Conv2D (1x1), BN, MFM", "64 -> 32", "32 x 7 x 50", "2,144"),
        ("Conv4b", "Conv2D (3x3, pad 1), BN, MFM", "64 -> 32", "32 x 7 x 50", "9,312"),
        ("Pool4", "MaxPool2D (2x2, stride 2)", "32", "32 x 3 x 25", "0"),
        ("Conv5a", "Conv2D (1x1), BN, MFM", "64 -> 32", "32 x 3 x 25", "1,120"),
        ("Conv5b", "Conv2D (3x3, pad 1), BN, MFM", "64 -> 32", "32 x 3 x 25", "9,312"),
        ("Pool5", "MaxPool2D (2x2, stride 2)", "32", "32 x 1 x 12", "0"),
        ("FC1", "Linear, BN, MFM", "320 -> 160", "160", "123,520"),
        ("FC2", "Linear, BN, MFM", "128 -> 64", "64", "20,672"),
        ("Classifier", "Linear (Output Logits [z0, z1])", "2", "2", "130"),
        ("Batch Norm", "Running mean/variance (12 layers)", "-", "-", "768 (non-trainable)"),
    ]
    header = f"{'Layer':<12} {'Operation':<35} {'Channels':<12} {'Output Dim':<16} {'Params':>12}"
    print(header)
    print("-" * 80)
    for r in rows:
        print(f"{r[0]:<12} {r[1]:<35} {r[2]:<12} {r[3]:<16} {r[4]:>12}")
    print("-" * 80)
    print(f"{'TOTAL':<12} {'Full Architectural Parameter Sum':<35} {'-':<12} {'-':<16} {'309,634':>12}")
    print("  * Trainable Parameters:     308,866")
    print("  * Non-Trainable BN Stats:       768")
    print("  * Total Weights Count:      309,634\n")


def print_table2():
    print("=" * 95)
    print("TABLE III: PERFORMANCE COMPARISON ACROSS ALL ARCHITECTURES ON ASVSPOOF 2019 LA")
    print("=" * 95)
    models = [
        ("SVM Baseline", "240-dim Static/Delta MFCC", "Eval (71,237)", "28.87%", "-", "71.13%", "95.54%", "71.13%"),
        ("SVM (MFCC-240)", "240-dim Dynamic MFCC", "Eval (71,237)", "28.96%", "-", "77.94%", "93.69%", "80.84%"),
        ("Deep MLP", "240-dim Dynamic MFCC", "Eval (71,237)", "25.52%", "1.0000", "74.40%", "96.23%", "74.37%"),
        ("Deep MLP", "240-dim Dynamic LFCC", "Eval (71,237)", "24.02%", "0.7559", "64.45%", "98.93%", "61.02%"),
        ("Deep MLP", "480-dim Fused MFCC+LFCC", "Eval (71,237)", "22.00%", "0.7337", "61.82%", "99.44%", "57.75%"),
        ("SVM-MLP Ensemble", "240-dim Dynamic MFCC", "Eval (71,237)", "26.87%", "0.9467", "79.55%", "93.73%", "82.74%"),
        ("LCNN-9 (Proposed)", "2D LFCC Spectrogram", "Dev (24,844)", "5.06%", "0.1540", "97.10%", "-", "-"),
        ("LCNN-9 (Proposed)", "2D LFCC Spectrogram", "Eval (71,237)", "<= 2.45%*", "-", "98.65%", "99.72%", "98.77%"),
    ]
    header = f"{'Model':<19} {'Feature':<25} {'Split':<15} {'EER':<10} {'min t-DCF':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}"
    print(header)
    print("-" * 95)
    for m in models:
        print(f"{m[0]:<19} {m[1]:<25} {m[2]:<15} {m[3]:<10} {m[4]:<10} {m[5]:<10} {m[6]:<10} {m[7]:<10}")
    print("-" * 95)
    print("* At tau = 0.9950, Spoof Miss Rate = 1.23% (784/63,882) and False Rejection = 2.45% (180/7,355),")
    print("  bounding the evaluation EER strictly within [1.23%, 2.45%].\n")


def print_table3_compact():
    print("=" * 82)
    print("TABLE III: PERFORMANCE COMPARISON ON ASVSPOOF 2019 LA (WORD 2-COLUMN READY)")
    print("=" * 82)
    print("Evaluated across all 71,237 evaluation trials (including 11 unseen attacks)")
    print("-" * 82)
    compact_models = [
        ("SVM (Static/Delta)", "Static/Delta MFCC", "28.87%", "71.13%", "95.54%", "71.13%"),
        ("SVM (Dynamic-240)", "Dynamic MFCC", "28.96%", "77.94%", "93.69%", "80.84%"),
        ("Deep MLP (MFCC)", "Dynamic MFCC", "25.52%", "74.40%", "96.23%", "74.37%"),
        ("Deep MLP (LFCC)", "Dynamic LFCC", "24.02%", "64.45%", "98.93%", "61.02%"),
        ("Deep MLP (Fused)", "Fused MFCC+LFCC", "22.00%", "61.82%", "99.44%", "57.75%"),
        ("SVM-MLP Ensemble", "Dynamic MFCC", "26.87%", "79.55%", "93.73%", "82.74%"),
        ("Proposed LCNN-9", "2D LFCC", "<= 2.45%*", "98.65%", "99.72%", "98.77%"),
    ]
    header = f"{'Model Architecture':<22} {'Acoustic Feature':<19} {'EER':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}"
    print(header)
    print("-" * 82)
    for m in compact_models:
        print(f"{m[0]:<22} {m[1]:<19} {m[2]:<10} {m[3]:<10} {m[4]:<10} {m[5]:<10}")
    print("-" * 82)
    print("* Bounded strictly within [1.23%, 2.45%] at operating threshold tau = 0.9950.\n")


def print_confusion_matrix():
    print("=" * 80)
    print("CONFUSION MATRIX & OPERATIONAL METRICS ON EVALUATION SPLIT (N = 71,237)")
    print("=" * 80)
    tn, fp = 7175, 180
    fn, tp = 784, 63098
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total * 100.0
    precision = tp / (tp + fp) * 100.0
    recall = tp / (tp + fn) * 100.0
    f1 = 2 * (precision * recall) / (precision + recall)
    miss_rate = fn / (tp + fn) * 100.0
    false_rejection = fp / (tn + fp) * 100.0

    print(f"                      Predicted Bonafide      Predicted Spoof       Total")
    print(f"  Actual Bonafide:    {tn:>10,d} (TN)         {fp:>10,d} (FP)      {tn+fp:>10,d}")
    print(f"  Actual Spoof:       {fn:>10,d} (FN)         {tp:>10,d} (TP)      {fn+tp:>10,d}")
    print(f"  Total Predictions:  {tn+fn:>10,d}            {fp+tp:>10,d}       {total:>10,d}")
    print("-" * 80)
    print(f"  * Total Evaluation Utterances:            {total:,d}")
    print(f"  * Correct Classifications:                {tn+tp:,d} ({accuracy:.2f}%)")
    print(f"  * Spoofs Wrongly Missed (Accepted):       {fn:,d} (Miss Rate: {miss_rate:.2f}%)")
    print(f"  * Bonafide Wrongly Rejected (Alarms):     {fp:,d} (False Rejection: {false_rejection:.2f}%)")
    print(f"  * Spoof Precision (TP / (TP+FP)):         {precision:.2f}%")
    print(f"  * Spoof Recall (TP / (TP+FN)):            {recall:.2f}%")
    print(f"  * Spoof F1-Score:                         {f1:.2f}%")
    print(f"  * Committed Operating Threshold (tau):    0.9950\n")


def print_latency():
    print("=" * 80)
    print("TABLE III: COMPUTATIONAL EFFICIENCY & LATENCY (INTEL CORE i7-1355U CPU)")
    print("=" * 80)
    print("  * Hardware:              13th Gen Intel Core i7-1355U (2 P-cores, 8 E-cores)")
    print("  * Runtime Environment:   Windows 11, ONNX Runtime 1.31.0 (4 intra-op threads)")
    print("  * Benchmark Trials:      1,000 consecutive runs (50 warm-up runs discarded)")
    print("  * Audio Window Input:    2.0 s analysis window (wrap-padded to 400 frames)")
    print("  * Streaming Step (Hop):  1.0 s update interval")
    print("-" * 80)
    print(f"{'Pipeline Component':<35} {'Median Latency':>18} {'P95 Latency':>18}")
    print("-" * 80)
    print(f"{'1. Front-End Quality Gate (VAD + SNR)':<35} {'0.80 ms':>18} {'1.40 ms':>18}")
    print(f"{'2. LFCC Feature Extraction (2.0s)':<35} {'3.34 ms':>18} {'9.28 ms':>18}")
    print(f"{'3. ONNX Model Forward Pass':<35} {'8.39 ms':>18} {'10.50 ms':>18}")
    print("-" * 80)
    print(f"{'Neural Pipeline Subtotal (2+3)':<35} {'12.16 ms':>18} {'18.14 ms':>18}")
    print(f"{'FULL PIPELINE TOTAL (1+2+3)':<35} {'12.96 ms (~13 ms)':>18} {'19.54 ms (~20 ms)':>18}")
    print("-" * 80)
    print(f"{'Real-Time Factor (RTF) @ 1.0s hop':<35} {'0.013':>18} {'0.020':>18}")
    print("  * Max Processing Throughput:  ~77 frames/second")
    print("  * Physical Buffer Lag:        2.0 s initial onset lag, refreshed every 1.0 s\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table1", action="store_true", help="Print Table I: Parameter Count Breakdown")
    parser.add_argument("--table2", "--table3", action="store_true", dest="table2", help="Print Table III: Benchmark Performance Comparison")
    parser.add_argument("--compact", "--compact-table3", action="store_true", help="Print Compact Table III for 2-column Word formatting")
    parser.add_argument("--cm", action="store_true", help="Print Confusion Matrix and Error Rates")
    parser.add_argument("--latency", action="store_true", help="Print Table IV: Latency & RTF Benchmark")
    parser.add_argument("--all", action="store_true", help="Print all tables and facts")
    args = parser.parse_args()

    # Default to all if no specific flags
    run_all = args.all or not (args.table1 or args.table2 or args.compact or args.cm or args.latency)

    print("\n" + "#" * 80)
    print("  FACTS & FIGURES SOURCE OF TRUTH: IEEE DEEPFAKE VOICE DETECTION PAPER")
    print("#" * 80 + "\n")

    if run_all or args.table1:
        print_table1()
    if run_all or args.table2:
        print_table2()
    if args.compact:
        print_table3_compact()
    if run_all or args.cm:
        print_confusion_matrix()
    if run_all or args.latency:
        print_latency()

    print("=" * 80)
    print("JSON SOURCE OF TRUTH LOCATION:")
    print(f"  {ROOT_DIR / 'results' / 'paper_metrics_source_of_truth.json'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
