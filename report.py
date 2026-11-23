"""
Metric definitions and the FAR/FRR convention used throughout this project.

Convention (must match src/modeling/config.py's LABEL_TO_INT):
    label 0 = bonafide (genuine)   "positive" in the authentication sense:
                                     the system should ACCEPT this speaker.
    label 1 = spoof (synthetic)     the system should REJECT this speaker.
    score   = P(spoof), higher = more suspicious.

A sample is classified "spoof" (rejected) when score >= threshold.

  False Accept Rate (FAR): proportion of SPOOF samples with score below
    threshold, i.e. wrongly accepted as genuine. This is the costly error
    for the fraud-prevention use case (a missed deepfake).
  False Reject Rate (FRR): proportion of BONAFIDE samples with score at or
    above threshold, i.e. wrongly rejected as spoof. This is the costly
    error for usability (a genuine caller/user flagged as suspicious).
  EER: the point on the FAR/FRR tradeoff curve where FAR == FRR, a standard
    single-number summary in the anti-spoofing literature (used by
    ASVspoof and cited by the base paper, though the base paper never
    actually reports a numeric EER value despite naming it as a metric).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_curve, precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score,
)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Returns (eer, eer_threshold).

    Implementation: sklearn's roc_curve with pos_label=1 (spoof) gives, at
    each threshold, the false-positive rate (= FAR here, spoof-as-genuine
    error... note sklearn's "positive class" framing differs from
    biometric FAR/FRR naming, so we derive FAR/FRR explicitly below rather
    than trusting roc_curve's fpr/tpr names directly):

        fpr(t) = P(score >= t | label == 0)   -> this IS our FRR(t)
                  (bonafide wrongly scored high enough to be rejected)
        tpr(t) = P(score >= t | label == 1)   -> 1 - FAR(t)
                  (spoof correctly scored high enough to be rejected)

    So FAR(t) = 1 - tpr(t), FRR(t) = fpr(t). EER is where these two curves
    cross (found via linear interpolation to the nearest sign change,
    since thresholds are discrete).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    frr = fpr             # bonafide wrongly rejected
    far = 1.0 - tpr        # spoof wrongly accepted

    diffs = far - frr
    idx = np.argmin(np.abs(diffs))

    # Linear interpolation between idx and its neighbor with an opposite-
    # sign diff, for a slightly more precise EER than "nearest point" alone
    # — this matters because thresholds from roc_curve are exactly the
    # observed score values, which can be sparse for small eval sets.
    if idx > 0 and np.sign(diffs[idx]) != np.sign(diffs[idx - 1]):
        neighbor = idx - 1
    elif idx < len(diffs) - 1 and np.sign(diffs[idx]) != np.sign(diffs[idx + 1]):
        neighbor = idx + 1
    else:
        neighbor = idx

    if neighbor == idx:
        eer = (far[idx] + frr[idx]) / 2.0
        eer_threshold = float(thresholds[idx])
    else:
        lo, hi = sorted([idx, neighbor])
        # interpolate where far - frr crosses zero between lo and hi
        x0, x1 = diffs[lo], diffs[hi]
        t = 0.0 if (x1 - x0) == 0 else -x0 / (x1 - x0)
        eer = far[lo] + t * (far[hi] - far[lo])
        t0 = thresholds[lo]
        t1 = thresholds[hi]
        if not np.isfinite(t0):
            eer_threshold = float(t1)
        elif not np.isfinite(t1):
            eer_threshold = float(t0)
        else:
            eer_threshold = float(t0 + t * (t1 - t0))

    return float(eer), eer_threshold


def select_threshold_from_validation(val_scores: np.ndarray, val_labels: np.ndarray, method: str = "eer") -> dict:
    """
    Selects the OPERATING threshold using validation data only. This
    function must never be called with test/eval-split data — that
    separation is enforced at the call-site level (scripts/train_model.py
    calls this; scripts/evaluate_model.py only ever loads the threshold
    that was already saved, never recomputes it from eval data).

    Returns a dict (not just a float) so the validation EER at the chosen
    threshold is recorded alongside it, for later comparison against the
    eval-set EER computed separately during evaluation.
    """
    if method != "eer":
        raise ValueError(f"Unsupported threshold-selection method: {method!r}")
    eer, threshold = compute_eer(val_scores, val_labels)
    return {"method": "eer", "threshold": threshold, "validation_eer": eer}


def far_frr_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    spoof_mask = labels == 1
    bonafide_mask = labels == 0

    n_spoof = int(spoof_mask.sum())
    n_bonafide = int(bonafide_mask.sum())

    far = float(np.mean(scores[spoof_mask] < threshold)) if n_spoof else float("nan")
    frr = float(np.mean(scores[bonafide_mask] >= threshold)) if n_bonafide else float("nan")

    return {"far": far, "frr": frr, "n_spoof": n_spoof, "n_bonafide": n_bonafide}


def classification_metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """
    Precision/recall/F1/accuracy/confusion-matrix at a FIXED, already-
    chosen threshold (never re-derived from the data passed in here — the
    threshold argument must come from select_threshold_from_validation,
    applied once at training time and reused unchanged for every
    evaluation condition).

    Positive class = spoof (1), consistent with FAR/FRR above: precision/
    recall describe how well the model finds spoof audio, which is the
    quantity the base paper's own Table 1 reports (their "Precision"/
    "Recall"/"F1-Score" columns), for comparability.
    """
    preds = (np.asarray(scores) >= threshold).astype(int)
    labels = np.asarray(labels)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    # cm layout: rows=true [bonafide, spoof], cols=pred [bonafide, spoof]
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_spoof": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_spoof": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1_spoof": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "confusion_matrix": {
            "true_bonafide_pred_bonafide": int(tn),
            "true_bonafide_pred_spoof": int(fp),
            "true_spoof_pred_bonafide": int(fn),
            "true_spoof_pred_spoof": int(tp),
        },
    }


def latency_summary(latencies_sec: np.ndarray) -> dict:
    """Summary stats for a batch of per-sample latency measurements.
    Reports percentiles, not just a mean, because a single mean (as in the
    base paper's unqualified "~85 ms") hides worst-case behavior that
    matters for an interactive challenge-response flow."""
    latencies_sec = np.asarray(latencies_sec)
    return {
        "n_samples": int(latencies_sec.size),
        "mean_ms": float(np.mean(latencies_sec) * 1000),
        "median_ms": float(np.median(latencies_sec) * 1000),
        "p95_ms": float(np.percentile(latencies_sec, 95) * 1000),
        "p99_ms": float(np.percentile(latencies_sec, 99) * 1000),
        "max_ms": float(np.max(latencies_sec) * 1000),
    }


def compute_min_tdcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_spoof: float = 0.05,
) -> tuple[float, float]:
    """
    Compute normalized minimum tandem Detection Cost Function (min t-DCF)
    proxy conforming to the ASVspoof challenge metric specification.

    Normalized cost:
      tDCF(t) = (C_miss * P_miss(t) + C_fa * P_fa(t)) / default_cost
    where:
      P_miss(t) = FAR(t) (spoof wrongly accepted, score < t)
      P_fa(t)   = FRR(t) (bonafide wrongly rejected, score >= t)
      C_miss    = c_miss * p_spoof
      C_fa      = c_fa * (1 - p_spoof)
      default_cost = min(C_miss, C_fa)

    Returns (min_tdcf, optimal_threshold).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    p_fa = fpr             # FRR (bonafide rejected)
    p_miss = 1.0 - tpr     # FAR (spoof accepted)

    cost_miss = c_miss * p_spoof
    cost_fa = c_fa * (1.0 - p_spoof)
    default_cost = min(cost_miss, cost_fa)

    tdcf_curve = (cost_miss * p_miss + cost_fa * p_fa) / default_cost
    min_idx = np.argmin(tdcf_curve)
    min_tdcf = float(tdcf_curve[min_idx])
    opt_threshold = float(thresholds[min_idx])
    return min_tdcf, opt_threshold


def bootstrap_metric_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_name: str = "eer",
    threshold: float | None = None,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence intervals for standardized evaluation.
    metric_name can be 'eer', 'accuracy', 'far', 'frr', or 'f1'.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(labels)
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    bootstrapped_values = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        s_sample = scores[idx]
        l_sample = labels[idx]

        # Ensure both classes exist in resample
        if len(np.unique(l_sample)) < 2:
            continue

        if metric_name == "eer":
            val, _ = compute_eer(s_sample, l_sample)
        elif metric_name == "accuracy":
            th = threshold if threshold is not None else 0.5
            preds = (s_sample >= th).astype(int)
            val = float(accuracy_score(l_sample, preds))
        elif metric_name == "far":
            th = threshold if threshold is not None else 0.5
            val = far_frr_at_threshold(s_sample, l_sample, th)["far"]
        elif metric_name == "frr":
            th = threshold if threshold is not None else 0.5
            val = far_frr_at_threshold(s_sample, l_sample, th)["frr"]
        elif metric_name == "f1":
            th = threshold if threshold is not None else 0.5
            preds = (s_sample >= th).astype(int)
            val = float(f1_score(l_sample, preds, pos_label=1, zero_division=0))
        else:
            raise ValueError(f"Unknown metric_name: {metric_name}")

        bootstrapped_values.append(val)

    if not bootstrapped_values:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrapped_values, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(bootstrapped_values, (1.0 - alpha / 2.0) * 100))
    mean_val = float(np.mean(bootstrapped_values))

    return {
        "metric": metric_name,
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "n_bootstraps": len(bootstrapped_values),
    }

