"""
OPTIONAL extension, NOT used by session.py or decision.py by default: a
sketch of a voice-similarity comparison against an explicitly-enrolled
reference profile ("does this response sound like the same person as
this earlier enrollment"), for anyone extending this project who wants
that additional signal.

======================= WHY THIS IS OFF BY DEFAULT ==========================
1. No validated speaker-embedding model is used here. A real system
   would use a purpose-trained model (e.g. an x-vector/ECAPA-TDNN
   architecture trained on a large speaker-verification corpus). This
   project's development environment had no network access to obtain
   such a model, so this module instead reuses the existing MFCC mean/std
   feature vector (src/preprocessing/features.py) as a crude, UNVALIDATED
   stand-in "voice-print" — good enough to demonstrate the enrollment/
   comparison/limits structure, not a claim that it reliably distinguishes
   speakers or resists cloning.
2. A voice-cloning system built specifically to target similarity (i.e.
   trained or prompted on samples of the person being impersonated) may
   score similarly to the genuine speaker on ANY embedding-based measure,
   crude or sophisticated — this is a known open problem in speaker
   verification generally, not something this placeholder solves or
   claims to solve.
3. Reliability depends heavily on enrollment quality: few enrollment
   samples, enrollment recorded on a different device/channel than the
   comparison audio, or enrollment audio itself of poor quality, all
   degrade genuine-speaker similarity scores (raising false-rejection
   risk) without necessarily doing anything to reduce false-acceptance
   risk against a targeted clone.
4. Enrollment is a biometric-adjacent action and requires clear, explicit,
   revocable consent — see enroll()'s required consent_confirmed
   parameter, which is NOT optional and NOT defaulted to True.

If you choose to wire this in, treat its output (SimilarityResult) exactly
like the spoof-risk score: one more probabilistic signal to feed into
decision.py's tiering, never a standalone "verified" determination, and
never a substitute for an independent callback recommendation.
=============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EnrollmentProfile:
    speaker_label: str            # a human-chosen label, NOT a claim of verified real-world identity
    reference_vector: np.ndarray    # mean MFCC-stats vector across enrollment samples
    n_enrollment_samples: int
    consent_recorded: bool
    created_at: float = field(default_factory=time.time)


@dataclass
class SimilarityResult:
    similarity: float          # cosine similarity in [-1, 1] between reference and query vectors
    status: str                 # "ok" | "insufficient_enrollment_samples"
    caveats: list = field(default_factory=list)


_MIN_ENROLLMENT_SAMPLES = 3  # a bare minimum for even a crude average; still not "validated"


def enroll(
    waveforms: list,
    sample_rate: int,
    speaker_label: str,
    consent_confirmed: bool,
) -> EnrollmentProfile:
    """
    Builds an EnrollmentProfile from a list of waveforms of the SAME
    person, recorded with their explicit knowledge and agreement.

    consent_confirmed must be passed as True explicitly by the calling
    application's own consent flow — this function does not default it
    and does not infer consent from the mere presence of audio. Enrolling
    someone without their knowledge is exactly the kind of use this
    project does not support.
    """
    if not consent_confirmed:
        raise ValueError(
            "consent_confirmed must be True: enrollment requires the person's explicit, "
            "informed agreement, recorded by your application's own consent flow before "
            "calling this function — it is not assumed or inferred here."
        )
    if len(waveforms) < _MIN_ENROLLMENT_SAMPLES:
        raise ValueError(
            f"At least {_MIN_ENROLLMENT_SAMPLES} enrollment samples are required for even "
            f"this crude averaging approach; fewer makes the reference vector noisier and "
            f"less representative of the person's typical voice."
        )

    from src.preprocessing import features as feat_mod  # lazy, needs librosa

    vectors = [feat_mod.extract_mfcc_stats(np.asarray(w, dtype=np.float32), sample_rate) for w in waveforms]
    reference_vector = np.mean(np.stack(vectors), axis=0)

    return EnrollmentProfile(
        speaker_label=speaker_label,
        reference_vector=reference_vector,
        n_enrollment_samples=len(waveforms),
        consent_recorded=True,
    )


def compare(profile: EnrollmentProfile, waveform: np.ndarray, sample_rate: int) -> SimilarityResult:
    """
    Cosine similarity between the enrollment's reference vector and a new
    clip's MFCC-stats vector. See module docstring for everything this
    does NOT establish. Always attaches the same caveats so a caller can't
    accidentally strip them when displaying just the numeric score.
    """
    from src.preprocessing import features as feat_mod  # lazy, needs librosa

    query_vector = feat_mod.extract_mfcc_stats(np.asarray(waveform, dtype=np.float32), sample_rate)

    ref = profile.reference_vector
    denom = np.linalg.norm(ref) * np.linalg.norm(query_vector)
    similarity = float(np.dot(ref, query_vector) / denom) if denom > 1e-9 else 0.0

    status = "ok" if profile.n_enrollment_samples >= _MIN_ENROLLMENT_SAMPLES else "insufficient_enrollment_samples"

    caveats = [
        "This similarity score uses an unvalidated MFCC-statistics proxy, not a trained "
        "speaker-verification model, and has not been evaluated for how well it resists a "
        "voice clone specifically targeting this speaker's voice.",
        f"Enrollment was based on {profile.n_enrollment_samples} sample(s); more and more "
        f"varied enrollment audio (different sessions, phrases, background conditions) would "
        f"generally make this comparison more reliable, not less.",
    ]

    return SimilarityResult(similarity=similarity, status=status, caveats=caveats)
