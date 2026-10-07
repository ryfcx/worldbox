"""Shared DeepSeek client. Standard library only.

Worldbox talks to DeepSeek-V4.1-Flash over HTTPS. Narration and tribe minds
both go through here so there is one place for auth, retries and TLS.

Keys come from ``.env`` or the environment: ``DEEPSEEK_API_KEY`` and
``OPENAI_API_KEY``. Either is enough; both together means tribes and
citizens get mixed houses, so the world is not one model talking to itself.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# DeepSeek-V4.1-Flash. The `deepseek-flash` id always points at the current
# Flash model; older aliases (deepseek-v4-flash, …) are routed here too.
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
API_URL = "https://api.deepseek.com/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ENV_KEY = "DEEPSEEK_API_KEY"
OPENAI_ENV_KEY = "OPENAI_API_KEY"

# Each house is one LLM the simulation can trap as a people.
HOUSES: tuple[dict[str, str], ...] = (
    {
        "house": "deepseek",
        "model": DEFAULT_MODEL,
        "label": "DeepSeek Flash",
        "env": ENV_KEY,
        "url": API_URL,
    },
    {
        "house": "openai",
        "model": DEFAULT_OPENAI_MODEL,
        "label": "GPT-5 Mini",
        "env": OPENAI_ENV_KEY,
        "url": OPENAI_URL,
    },
)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_DOTENV_LOADED = False


def load_dotenv() -> None:
    """Load KEY=VALUE lines from a local ``.env`` without overwriting the real env.

    Looks at the project root first, then the current directory and its
    parents. Existing environment variables win, so ``export DEEPSEEK_API_KEY``
    still overrides the file. The file is never required; a missing ``.env``
    is silence, not an error.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    seen: set[Path] = set()
    candidates = [Path(__file__).resolve().parents[2] / ".env"]
    cwd = Path.cwd()
    candidates.extend(directory / ".env" for directory in (cwd, *cwd.parents))
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if name and value and name not in os.environ:
                os.environ[name] = value


def api_key(explicit: Optional[str] = None, env_name: str = ENV_KEY) -> Optional[str]:
    """The API key to use, from the argument, the environment, or ``.env``."""
    if explicit:
        return explicit
    load_dotenv()
    return os.environ.get(env_name) or None


def house_spec(house_or_model: str) -> Dict[str, str]:
    """Look up a house by its id or model name, defaulting to DeepSeek."""
    token = (house_or_model or "").strip().lower()
    for spec in HOUSES:
        if token in {spec["house"], spec["model"], spec["label"].lower()}:
            return spec
    return dict(HOUSES[0])


def available_houses() -> List[Dict[str, str]]:
    """Houses whose keys are actually configured."""
    load_dotenv()
    return [spec for spec in HOUSES if os.environ.get(spec["env"])]


def house_for_index(index: int) -> str:
    """Stable house assignment for a tribe or citizen id."""
    roster = available_houses()
    if not roster:
        return ""
    return roster[index % len(roster)]["house"]


@dataclass
class ChatResult:
    """Text from the model, or the reason there is none."""

    text: str = ""
    error: str = ""
    model: str = DEFAULT_MODEL

    @property
    def ok(self) -> bool:
        """True if the model returned usable text."""
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


def is_available(explicit: Optional[str] = None) -> bool:
    """True if any model key is configured."""
    if explicit:
        return True
    return bool(available_houses())


def chat(
    messages: List[Dict[str, str]],
    key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    retries: int = 2,
    backoff: float = 1.5,
    temperature: float = 0.7,
    max_tokens: int = 1200,
    json_mode: bool = False,
    thinking: bool = False,
) -> ChatResult:
    """Send a chat-completions request to DeepSeek Flash.

    Thinking mode is off by default: tribe decisions have to come back in
    seconds or the live map stalls, and Flash without thinking is the whole
    point of using this model. JSON mode is for structured policies.
    """
    spec = house_spec(model)
    resolved = key or api_key(env_name=spec["env"])
    if resolved is None and spec["house"] != "deepseek":
        # Fall back to whatever key exists so a missing second house does not
        # silence a tribe that was assigned to it.
        roster = available_houses()
        if roster:
            spec = roster[0]
            resolved = api_key(env_name=spec["env"])
            model = spec["model"]
    if resolved is None:
        return ChatResult(
            error=f"No API key. Put {ENV_KEY} or {OPENAI_ENV_KEY} in .env.",
            model=model,
        )

    payload: Dict[str, Any] = {
        "model": spec["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if spec["house"] == "deepseek":
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error = ""
    body = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            spec["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {resolved}",
            },
            method="POST",
        )
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
                return ChatResult(
                    error=(
                        f"API rejected the key ({error.code}). Check that {spec['env']} "
                        f"is valid. {detail}"
                    ),
                    model=model,
                )
            if error.code == 404:
                return ChatResult(
                    error=(
                        f"Model '{model}' is not available ({error.code}). "
                        "Try model='deepseek-flash'."
                    ),
                    model=model,
                )
            return ChatResult(error=f"API returned {error.code}: {detail}", model=model)
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
                return ChatResult(
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
            return ChatResult(error=f"Could not reach the API: {reason}", model=model)
        except (TimeoutError, OSError) as error:
            if attempt < retries:
                last_error = f"Network error: {error}"
                time.sleep(backoff * (2**attempt))
                continue
            return ChatResult(error=f"Network error: {error}", model=model)
        except json.JSONDecodeError:
            return ChatResult(error="The API returned a response that was not JSON.", model=model)

    if body is None:
        return ChatResult(error=last_error or "The request failed.", model=model)

    try:
        message = body["choices"][0]["message"]
        text = (message.get("content") or "").strip()
    except (KeyError, IndexError, TypeError):
        return ChatResult(error="The API response contained no text.", model=model)

    if not text:
        reason = ""
        try:
            reason = body["choices"][0].get("finish_reason", "")
        except (KeyError, IndexError, TypeError):
            pass
        if reason == "length":
            return ChatResult(
                error="The model hit its output limit before writing anything.",
                model=model,
            )
        return ChatResult(error="The model returned an empty reply.", model=model)
    return ChatResult(text=text, model=model)
