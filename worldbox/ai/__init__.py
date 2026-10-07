"""AI: DeepSeek Flash as historian and as the minds of tribes.

Narration is optional colour. Tribe minds are the point -- they change how
people work, fight and treat their neighbours.
"""

from .client import DEFAULT_MODEL, ENV_KEY, OPENAI_ENV_KEY, is_available
from .minds import MindDirector, TribePolicy
from . import narrator

__all__ = [
    "DEFAULT_MODEL",
    "ENV_KEY",
    "OPENAI_ENV_KEY",
    "MindDirector",
    "TribePolicy",
    "is_available",
    "narrator",
]
