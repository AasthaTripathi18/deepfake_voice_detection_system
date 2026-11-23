"""
Single source of truth for the manifest table schema.

Every corpus adapter (src/dataset/asvspoof2019.py, and any future adapter)
must emit rows conforming to this schema. Downstream modules (preprocessing,
modeling, inference, evaluation) read only from a built manifest and must
never re-derive labels or splits from raw corpus files directly — this keeps
"what data exists and how it's labeled" in exactly one place.
"""

from dataclasses import dataclass, fields

# --- Controlled vocabularies -------------------------------------------------

# We deliberately keep to the corpus's *official* partitions. Using the
# official ASVspoof 2019 LA train/dev/eval split (rather than inventing our
# own re-split) is the leakage-prevention decision for this project:
#   - The official split is speaker-disjoint across train/dev/eval.
#   - The eval partition additionally contains spoofing systems (attack IDs)
#     that do NOT appear in train/dev, i.e. it already tests generalization
#     to unseen attacks, which is exactly the property we want preserved.
# Re-splitting by ourselves (e.g. a random shuffle) would risk mixing
# speakers or attack systems across train/test and silently overstating
# performance, so we do not do that for the primary experiment.
SPLITS = ("train", "dev", "eval")

LABELS = ("bonafide", "spoof")

# Condition tag for untouched, corpus-native audio. Degraded variants get
# their own tag (e.g. "telephony_bandpass", "codec_gsm", "noise_snr10") from
# src/preprocessing/degradations.py — this module only ever writes "clean".
CLEAN_CONDITION = "clean"

STATUS_VALID = "valid"
STATUS_SKIPPED = "skipped"

# Plausible bounds used by the integrity checker. These are corpus-appropriate
# defaults for ASVspoof 2019 LA (16 kHz, mono, short utterances), not universal
# constants — a new corpus adapter may need to pass different bounds.
DEFAULT_EXPECTED_SAMPLE_RATE = 16000
DEFAULT_EXPECTED_CHANNELS = 1
DEFAULT_MIN_DURATION_SEC = 0.4
DEFAULT_MAX_DURATION_SEC = 15.0


@dataclass
class ManifestRow:
    file_id: str            # corpus-native utterance id, stable across reruns
    file_path: str           # absolute or repo-relative path to the audio file
    corpus: str               # e.g. "asvspoof2019_la"
    split: str                # one of SPLITS — the corpus's OFFICIAL partition
    label: str                 # one of LABELS
    speaker_id: str            # corpus-native speaker id, "" if unknown
    system_id: str              # attack/system id for spoof audio, "-" for bonafide
    condition: str = CLEAN_CONDITION
    sample_rate: int = 0
    duration_sec: float = 0.0
    channels: int = 0
    status: str = STATUS_VALID
    skip_reason: str = ""

    @staticmethod
    def columns():
        return [f.name for f in fields(ManifestRow)]
