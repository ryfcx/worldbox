"""Optional AI narration of a world's history.

The simulation never depends on this. It reads the :class:`Chronicle` -- which
is already a list of turning points -- and asks a language model to write them
up as a historian would. If no API key is configured, or the network call fails,
narration is simply unavailable and everything else carries on.

Design rules:

* **The key comes from the environment** (``GEMINI_API_KEY``), never from the
  repository. Committing a key is how keys leak.
* **Standard library only** (``urllib``), so Worldbox still has no dependencies.
* **Failures are values, not exceptions.** :func:`narrate` returns a
  :class:`Narration` carrying either text or an error message, so a frontend can
  show the problem without crashing the simulation.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..simulation.chronicle import ChronicleEntry

# Google retires specific model versions; the "-latest" alias keeps working
# without needing a code change every time one is deprecated.
DEFAULT_MODEL = "gemini-flash-latest"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
ENV_KEY = "GEMINI_API_KEY"

# Statuses worth a retry: rate limiting and transient server faults.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

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


def ssl_context() -> ssl.SSLContext:
    """A verifying SSL context that works on stock macOS Python installs.

    Python installed from python.org ships without a CA bundle until you run
    ``Install Certificates.command``, so the default context trusts nothing and
    every HTTPS call fails. When the system store is empty we fall back to the
    ``certifi`` bundle if it happens to be installed.

    Certificate verification is never disabled -- an unverified connection to a
    third party carrying an API key is not a trade worth making.
    """
    context = ssl.create_default_context()
    if not context.get_ca_certs():
        try:
            import certifi  # Optional; not a declared dependency.

            context.load_verify_locations(certifi.where())
        except (ImportError, OSError):
            pass
    return context


def api_key(explicit: Optional[str] = None) -> Optional[str]:
    """The API key to use, from the argument or the environment."""
    return explicit or os.environ.get(ENV_KEY) or None


def is_available(explicit: Optional[str] = None) -> bool:
    """True if a key is configured, so a frontend can hide the feature."""
    return api_key(explicit) is not None


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
    resolved = api_key(key)
    if resolved is None:
        return Narration(
            error=f"No API key. Set {ENV_KEY} in your environment to enable narration.",
            model=model,
        )
    if not entries:
        return Narration(error="Nothing has happened yet -- run the simulation first.", model=model)

    source = format_chronicle(entries, max_entries)
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": source}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1200},
    }

    request = urllib.request.Request(
        f"{API_ROOT}/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": resolved,
        },
        method="POST",
    )

    body = None
    last_error = ""
    # Rate limits and transient server faults are worth retrying; everything
    # else fails on the first attempt so the user sees the real problem fast.
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=ssl_context()
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")[:300]
            if error.code in RETRYABLE_STATUS and attempt < retries:
                last_error = f"API returned {error.code}"
                time.sleep(backoff * (2**attempt))
                continue
            if error.code in (401, 403):
                return Narration(
                    error=(
                        f"API rejected the key ({error.code}). Check that {ENV_KEY} is set "
                        f"to a valid key with the Generative Language API enabled. {detail}"
                    ),
                    model=model,
                )
            if error.code == 404:
                return Narration(
                    error=(
                        f"Model '{model}' is not available to this key ({error.code}). "
                        "Google retires model versions; try model='gemini-flash-latest'."
                    ),
                    model=model,
                )
            return Narration(error=f"API returned {error.code}: {detail}", model=model)
        except urllib.error.URLError as error:
            # urlopen wraps TLS failures in URLError, so a certificate problem
            # has to be recognised from the wrapped reason rather than by
            # catching ssl.SSLError (which would be unreachable here).
            reason = error.reason
            if isinstance(reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
                return Narration(
                    error=(
                        f"TLS certificate verification failed ({reason}). Python installed "
                        "from python.org ships without a CA bundle -- run "
                        "'/Applications/Python 3.x/Install Certificates.command' once, "
                        "or 'pip install certifi'."
                    ),
                    model=model,
                )
            if attempt < retries:
                last_error = f"Could not reach the API: {reason}"
                time.sleep(backoff * (2**attempt))
                continue
            return Narration(error=f"Could not reach the API: {reason}", model=model)
        except (TimeoutError, OSError) as error:
            if attempt < retries:
                last_error = f"Network error: {error}"
                time.sleep(backoff * (2**attempt))
                continue
            return Narration(error=f"Network error: {error}", model=model)
        except json.JSONDecodeError:
            return Narration(error="The API returned a response that was not JSON.", model=model)

    if body is None:
        return Narration(error=last_error or "The request failed.", model=model)

    try:
        parts = body["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError, TypeError):
        blocked = body.get("promptFeedback", {}).get("blockReason")
        if blocked:
            return Narration(error=f"The request was blocked: {blocked}", model=model)
        return Narration(error="The API response contained no text.", model=model)

    if not text:
        reason = ""
        try:
            reason = body["candidates"][0].get("finishReason", "")
        except (KeyError, IndexError, TypeError):
            pass
        if reason == "MAX_TOKENS":
            return Narration(
                error="The model hit its output limit before writing anything. "
                      "Try a shorter chronicle (lower max_entries).",
                model=model,
            )
        return Narration(error="The model returned an empty history.", model=model)
    return Narration(text=text, model=model)


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
