"""
Generates a short, unpredictable challenge phrase: a couple of common,
easy-to-say, hard-to-confuse-with-each-other words plus a random digit
string. Uses `secrets` (a CSPRNG), not `random`, because "unpredictable"
is a real requirement here, not just "looks varied" — a naive PRNG seeded
from the clock could in principle be predicted or replayed against by an
attacker who has other information about timing.

The wordlist is deliberately small and phonetically distinct (no two
words that sound alike), so a reasonably clear spoken response should be
transcribable even by an imperfect ASR backend — accuracy of the content
check downstream depends on this list being easy to say and easy to hear
apart, not just "large enough to be unpredictable."
"""

from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass, field

# Deliberately short, phonetically distinct, common words across languages.
# Entropy comes mainly from the digit string; the words mainly help a
# listener/ASR system anchor on where the phrase starts and ends.
MULTILINGUAL_WORDLISTS: dict[str, list[str]] = {
    "en": [
        "APPLE", "TIGER", "RIVER", "PURPLE", "GOLDEN", "SILVER", "GARDEN",
        "WINTER", "SUMMER", "ISLAND", "BRIDGE", "MARKET", "VALLEY", "HARBOR",
        "CASTLE", "FOREST", "DESERT", "CANYON", "METEOR", "ROCKET",
    ],
    "es": [
        "MANZANA", "TIGRE", "RIO", "VERDE", "DORADO", "PLATA", "JARDIN",
        "INVIERNO", "VERANO", "ISLA", "PUENTE", "MERCADO", "VALLE", "PUERTO",
        "CASTILLO", "BOSQUE", "DESIERTO", "CANON", "METEORO", "COHETE",
    ],
    "fr": [
        "POMME", "TIGRE", "RIVIERE", "VIOLET", "DORE", "ARGENT", "JARDIN",
        "HIVER", "ETE", "ILE", "PONT", "MARCHE", "VALLEE", "PORT",
        "CHATEAU", "FORET", "DESERT", "METEORE", "FUSEE", "ETOILE",
    ],
    "de": [
        "APFEL", "TIGER", "FLUSS", "LILA", "GOLDEN", "SILBER", "GARTEN",
        "WINTER", "SOMMER", "INSEL", "BRUECKE", "MARKT", "TAL", "HAFEN",
        "SCHLOSS", "WALD", "WUESTE", "METEOR", "RAKETE", "STERN",
    ],
    "digits": [],  # Pure language-agnostic digit sequence
}
_WORDLIST = MULTILINGUAL_WORDLISTS["en"]


@dataclass
class Challenge:
    challenge_id: str
    text: str                 # the human-readable phrase to display/speak, e.g. "TIGER HARBOR 7 3 1"
    expected_tokens: list      # normalized tokens the response should match, e.g. ["TIGER","HARBOR","7","3","1"]
    language: str = "en"
    created_at: float = field(default_factory=time.time)
    consumed: bool = False     # set True by session.py once a response has been checked against it


class ChallengeGenerator:
    """
    Stateful only in that it remembers which challenge_ids it has already
    issued in THIS process, so a caller who (by bug or by attack) tries to
    reuse an old Challenge object is caught by session.py rather than
    silently re-scored — see session.py's ChallengeResponseSession, which
    refuses to run a challenge whose `consumed` flag is already True.
    """

    def __init__(self, num_words: int = 2, digit_count: int = 3, language: str = "en"):
        if num_words < 1 and language != "digits":
            raise ValueError("num_words must be >= 1")
        if digit_count < 1:
            raise ValueError("digit_count must be >= 1")
        if language not in MULTILINGUAL_WORDLISTS:
            raise ValueError(f"Unsupported language: {language!r}. Choose from {list(MULTILINGUAL_WORDLISTS)}")
        self.num_words = num_words
        self.digit_count = digit_count
        self.language = language

    def generate(self) -> Challenge:
        wordlist = MULTILINGUAL_WORDLISTS[self.language]
        if self.language == "digits" or not wordlist:
            digits = [str(secrets.randbelow(10)) for _ in range(max(self.digit_count, 5))]
            expected_tokens = digits
            text = " ".join(digits)
        else:
            words = [secrets.choice(wordlist) for _ in range(self.num_words)]
            digits = [str(secrets.randbelow(10)) for _ in range(self.digit_count)]
            expected_tokens = words + digits
            text = " ".join(words) + " " + " ".join(digits)

        return Challenge(
            challenge_id=str(uuid.uuid4()),
            text=text,
            expected_tokens=expected_tokens,
            language=self.language,
        )


def render_for_display(challenge: Challenge) -> str:
    """Plain text meant for an on-screen prompt. Kept as its own function
    (rather than just using challenge.text directly) so a caller has one
    obvious place to localize/restyle the prompt later without touching
    generation logic."""
    prompts = {
        "en": f"Please say: {challenge.text}",
        "es": f"Por favor diga: {challenge.text}",
        "fr": f"Veuillez dire : {challenge.text}",
        "de": f"Bitte sprechen Sie: {challenge.text}",
        "digits": f"Please read the numbers: {challenge.text}",
    }
    return prompts.get(challenge.language, f"Please say: {challenge.text}")


def speak_challenge(challenge: Challenge, tts_backend=None) -> str:
    """
    "Display or play the phrase" — displaying is always available (see
    render_for_display). Playing it as audio requires a text-to-speech
    engine, which is NOT bundled with this project (no TTS dependency is
    in requirements.txt). If the calling application has its own TTS
    backend, pass it as `tts_backend`: a callable taking the challenge
    text and doing whatever synthesis/playback it needs. If no backend is
    supplied, this function is a no-op beyond returning the display text —
    callers relying on audio playback must supply tts_backend explicitly
    rather than this function silently pretending to speak the phrase.
    """
    if tts_backend is not None:
        tts_backend(challenge.text)
    return render_for_display(challenge)
