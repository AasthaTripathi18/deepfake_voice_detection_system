"""
Compares a transcribed response to the challenge's expected tokens.

Uses order-SENSITIVE sequence matching (difflib.SequenceMatcher over the
token lists, not a bag-of-words overlap), since the challenge is a
specific ordered phrase ("TIGER HARBOR 7 3 1") — a transcript containing
the right words in the wrong order is weaker evidence than one that
actually matches the sequence, and a bag-of-words score would treat
"1 3 7 HARBOR TIGER" identically to a correct answer.

This is intentionally simple and explainable (no learned model, no fuzzy
phonetic matching beyond what SequenceMatcher gives for free via partial
token edits), matching this project's general preference for transparent,
auditable decision logic over an unvalidated learned fusion step.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

MATCHED = "matched"
PARTIAL_MATCH = "partial_match"
NO_MATCH = "no_match"


@dataclass
class MatchResult:
    status: str            # MATCHED | PARTIAL_MATCH | NO_MATCH
    ratio: float             # difflib ratio in [0, 1], for transparency/logging
    expected_tokens: list
    transcribed_tokens: list


def normalize_tokens(text: str) -> list:
    """Uppercase, strip punctuation, collapse whitespace, split on
    whitespace. Deliberately simple — no stemming/lemmatization, since the
    challenge tokens are short common words and digits where this is
    unnecessary and would add opacity."""
    cleaned = re.sub(r"[^\w\s]", " ", text.upper())
    return [tok for tok in cleaned.split() if tok]


def match_challenge(
    expected_tokens: list,
    transcribed_text: str,
    matched_threshold: float = 0.75,
    partial_threshold: float = 0.4,
) -> MatchResult:
    transcribed_tokens = normalize_tokens(transcribed_text)

    if not transcribed_tokens:
        return MatchResult(
            status=NO_MATCH, ratio=0.0,
            expected_tokens=expected_tokens, transcribed_tokens=transcribed_tokens,
        )

    ratio = difflib.SequenceMatcher(a=expected_tokens, b=transcribed_tokens).ratio()

    if ratio >= matched_threshold:
        status = MATCHED
    elif ratio >= partial_threshold:
        status = PARTIAL_MATCH
    else:
        status = NO_MATCH

    return MatchResult(
        status=status, ratio=round(ratio, 4),
        expected_tokens=expected_tokens, transcribed_tokens=transcribed_tokens,
    )
