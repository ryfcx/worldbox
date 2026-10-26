"""Optional AI narration of a world's history.

The simulation never depends on this. It reads the :class:`Chronicle` -- which
is already a list of turning points -- and asks DeepSeek Flash to write them
up as a historian would. If no API key is configured, or the network call fails,
narration is simply unavailable and everything else carries on.

Failures are values, not exceptions: :func:`narrate` returns a :class:`Narration`
carrying either text or an error message, so a frontend can show the problem
without crashing the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..simulation.chronicle import ChronicleEntry
from .client import DEFAULT_MODEL, ENV_KEY, OPENAI_ENV_KEY, ChatResult, api_key, chat, is_available

# Re-exported so existing imports of narrator.ENV_KEY / is_available keep working.
__all__ = [
    "DEFAULT_MODEL",
    "ENV_KEY",
    "Narration",
    "api_key",
    "format_chronicle",
    "is_available",
    "answer",
    "narrate",
    "wrap",
]

SYSTEM_PROMPT = """\
You are a historian writing about a civilisation that actually existed. You are \
given a chronicle of dated events from its history.

Write a concise, readable history in flowing prose. Requirements:
- Use ONLY the events given. Never invent wars, people, inventions or places.
- Refer to years as given (e.g. "by Year 82").
- Group related events into a narrative arc rather than listing them.
- Be plain and factual, like a good encyclopaedia entry. No purple prose.
- If the chronicle is sparse, keep the history short. Do not pad it.
"""


@dataclass
class Narration:
    """The result of a narration attempt: text, or the reason there is none."""

    text: str = ""
    error: str = ""
    model: str = DEFAULT_MODEL

    @property
    def ok(self) -> bool:
        """True if narration succeeded."""
        return bool(self.text) and not self.error

    @classmethod
    def from_chat(cls, result: ChatResult) -> "Narration":
        """Copy a chat result into the narration shape the frontends expect."""
        return cls(text=result.text, error=result.error, model=result.model)


def format_chronicle(entries: Sequence[ChronicleEntry], limit: int = 120) -> str:
    """Render chronicle entries as the prompt's source material."""
    selected = list(entries)[-limit:]
    return "\n".join(f"Year {entry.year}: {entry.message}" for entry in selected)


def narrate(
    entries: Sequence[ChronicleEntry],
    key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    max_entries: int = 120,
    retries: int = 2,
    backoff: float = 1.5,
) -> Narration:
    """Ask the model to write up a chronicle as a history.

    Returns a :class:`Narration`; check ``.ok`` before using ``.text``.
    """
    if api_key(key) is None:
        return Narration(
            error=f"No API key. Put {ENV_KEY} in .env or export it.",
            model=model,
        )
    if not entries:
        return Narration(
            error="Nothing has happened yet -- run the simulation first.",
            model=model,
        )

    source = format_chronicle(entries, max_entries)
    result = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source},
        ],
        key=key,
        model=model,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        temperature=0.7,
        max_tokens=1200,
        json_mode=False,
        thinking=False,
    )
    return Narration.from_chat(result)


ANSWER_PROMPT = """\
You answer questions about a civilisation simulation that is running right now. \
You are given a status briefing and a chronicle of what has actually happened.

Rules:
- Answer the question directly, in a few short paragraphs or a tight list.
- Use only the briefing and the chronicle. Name tribes, places, and years that appear there.
- If the material does not contain the answer, say so in one sentence. Do not guess.
- Do not invent people, wars, inventions, or places.
"""


def answer(
    question: str,
    entries: Sequence[ChronicleEntry],
    briefing: str,
    key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    max_entries: int = 80,
) -> Narration:
    """Answer one question about the running world.

    The model only sees the briefing and the chronicle, so a question it cannot
    support comes back as an admission rather than a fabricated history.
    """
    if api_key(key) is None and not is_available():
        return Narration(
            error=f"No API key. Put {ENV_KEY} or {OPENAI_ENV_KEY} in .env.",
            model=model,
        )
    cleaned = " ".join((question or "").split())
    if not cleaned:
        return Narration(error="Type a question first.", model=model)

    source = format_chronicle(entries, max_entries) if entries else "(no chronicle yet)"
    result = chat(
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {
                "role": "user",
                "content": f"Question: {cleaned}\n\nBriefing:\n{briefing}\n\nChronicle:\n{source}",
            },
        ],
        key=key,
        model=model,
        timeout=timeout,
        retries=1,
        backoff=1.0,
        temperature=0.4,
        max_tokens=700,
        json_mode=False,
        thinking=False,
    )
    return Narration.from_chat(result)


def wrap(text: str, width: int = 76) -> List[str]:
    """Wrap narration to a readable width for the terminal."""
    import textwrap

    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines
