"""Name generation for agents, families and tribes.

With hundreds of births over a long run, a small syllable pool produces the
same handful of names over and over. This module builds names from several
patterns and a much larger pool, and adds two touches that make a population
read like a society rather than a list:

* **Family names.** A child inherits a parent's family name, so lineages are
  visible in the event log and in ``agent`` inspections.
* **Naming styles.** Each tribe draws from its own slice of the syllable pool,
  so members of a tribe sound related to each other and distinct from their
  neighbours -- culture you can hear.

Every function takes a seeded ``random.Random`` and is therefore reproducible.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

# Syllable pools. Each naming style uses one onset group and one coda group,
# which is what gives tribes their distinct sound.
_ONSETS: Tuple[Tuple[str, ...], ...] = (
    ("Ka", "Ta", "Ma", "Na", "Ra", "Va", "Sa", "Da", "Ha", "Ba"),
    ("El", "Il", "Ael", "Ys", "Ei", "Ol", "Ur", "Ith", "Eth", "Ael"),
    ("Grim", "Skal", "Thor", "Bran", "Dun", "Kor", "Vor", "Har", "Bor", "Gar"),
    ("Sil", "Mir", "Lyr", "Ny", "Cel", "Fae", "Wyn", "Ala", "Eir", "Sha"),
    ("Zan", "Xor", "Qua", "Vek", "Zir", "Jak", "Kes", "Nyx", "Vas", "Rho"),
    ("Os", "Fen", "Ler", "Cal", "Mor", "Ash", "Ost", "Ura", "Vel", "Tar"),
)
_MIDDLES: Tuple[str, ...] = (
    "a", "e", "i", "o", "u", "ae", "ei", "ia", "or", "an", "el", "ir", "un", "yl",
)
_CODAS: Tuple[Tuple[str, ...], ...] = (
    ("n", "ra", "ka", "ta", "li", "sa", "ma", "na", "va", "da"),
    ("wen", "iel", "ith", "eth", "aen", "ael", "ien", "yth", "eon", "ion"),
    ("dor", "gar", "mund", "ric", "gan", "thar", "grim", "vek", "born", "helm"),
    ("wyn", "lyn", "sha", "mira", "vell", "seth", "rien", "lia", "nara", "esse"),
    ("ax", "ux", "ez", "ix", "oz", "yr", "esk", "ask", "urn", "orn"),
    ("us", "or", "is", "ar", "en", "an", "ir", "on", "ur", "el"),
)

# Family names are built from their own pools so they do not collide with
# given names.
_FAMILY_ROOTS: Tuple[str, ...] = (
    "Stone", "River", "Ash", "Oak", "Iron", "Storm", "Frost", "Ember", "Thorn",
    "Wolf", "Raven", "Bear", "Elk", "Hawk", "Boar", "Fox", "Owl", "Stag",
    "High", "Deep", "Long", "Grey", "Red", "White", "Black", "Green", "Gold",
    "North", "South", "East", "West", "Under", "Over", "Far", "Wind", "Sun",
)
_FAMILY_TAILS: Tuple[str, ...] = (
    "born", "walker", "wood", "field", "hill", "vale", "brook", "ford", "watch",
    "hand", "heart", "spear", "shield", "song", "wright", "smith", "runner",
    "seeker", "warden", "kin", "blood", "crest", "mane", "claw", "fang",
)

STYLE_COUNT = len(_ONSETS)


def given_name(rng: random.Random, style: Optional[int] = None) -> str:
    """Build a given name, optionally in a particular tribe's style.

    Three patterns are used -- short, medium and long -- so a population's
    names vary in shape as well as in sound.
    """
    index = rng.randrange(STYLE_COUNT) if style is None else style % STYLE_COUNT
    onset = rng.choice(_ONSETS[index])
    coda = rng.choice(_CODAS[index])

    roll = rng.random()
    if roll < 0.35:  # Short: Kara
        return onset + coda
    if roll < 0.85:  # Medium: Kaelira
        return onset + rng.choice(_MIDDLES) + coda
    # Long: Kaelithwen
    return onset + rng.choice(_MIDDLES) + rng.choice(_CODAS[index]) + coda


def family_name(rng: random.Random) -> str:
    """Build a family name such as ``"Stonewarden"``."""
    return rng.choice(_FAMILY_ROOTS) + rng.choice(_FAMILY_TAILS)


def full_name(
    rng: random.Random,
    style: Optional[int] = None,
    inherited_family: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(given, family)``, inheriting the family name when given.

    Children keep a parent's family name; founders get a fresh one.
    """
    family = inherited_family if inherited_family else family_name(rng)
    return given_name(rng, style), family


# --- Tribe names -----------------------------------------------------------

_TRIBE_KINDS: Tuple[str, ...] = (
    "Tribe", "Clan", "Folk", "Kin", "Host", "Company", "League", "Band",
)


def tribe_name(rng: random.Random, style: Optional[int] = None) -> str:
    """Build a tribe name such as ``"Kaelira Clan"``, in the tribe's own style."""
    index = rng.randrange(STYLE_COUNT) if style is None else style % STYLE_COUNT
    root = rng.choice(_ONSETS[index]) + rng.choice(_MIDDLES) + rng.choice(_CODAS[index])
    return f"{root} {rng.choice(_TRIBE_KINDS)}"


def settlement_name(rng: random.Random, style: Optional[int] = None) -> str:
    """Build a settlement name such as ``"Kaelira's Rest"``."""
    index = rng.randrange(STYLE_COUNT) if style is None else style % STYLE_COUNT
    root = rng.choice(_ONSETS[index]) + rng.choice(_MIDDLES) + rng.choice(_CODAS[index])
    suffix = rng.choice(
        ("Rest", "Hold", "Watch", "Crossing", "Haven", "Reach", "Gate", "Stead",
         "Bridge", "Landing", "Hearth", "Refuge", "Post", "Camp", "Rise")
    )
    return f"{root} {suffix}"


def estimate_unique_given_names() -> int:
    """Roughly how many distinct given names the generator can produce.

    Used by the test suite to check that the pool is large enough for a long
    run's worth of births.
    """
    total = 0
    for index in range(STYLE_COUNT):
        onsets = len(_ONSETS[index])
        codas = len(_CODAS[index])
        middles = len(_MIDDLES)
        total += onsets * codas  # Short.
        total += onsets * middles * codas  # Medium.
        total += onsets * middles * codas * codas  # Long.
    return total
