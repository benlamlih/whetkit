"""Opt-in anonymous usage telemetry.

STRICTLY opt-in: nothing is ever sent unless the user sets
``WHETKIT_TELEMETRY=1`` or enables it with ``whetkit telemetry on``. One
event per CLI command, containing ONLY: the command name, the whetkit
version, the Python major.minor, ``sys.platform``, and a random anonymous
UUID persisted in ``~/.whetkit/telemetry.json``. Never collected: arguments,
file paths, server names, prompts, or results.

Delivery is fire-and-forget — a single PostHog capture POST from a daemon
thread with a 2s timeout, every exception swallowed — so telemetry can never
block, slow down, or fail the CLI.
"""

import contextlib
import json
import os
import sys
import threading
import uuid
from pathlib import Path

# PostHog PROJECT API key. This is a public, client-side key: it can only
# submit events, never read anything back. Embedding it in source is intended
# and is exactly how PostHog client SDKs ship.
POSTHOG_PROJECT_KEY = "phc_wLDBuW7Y73TxZQ9pEKKrhZYhd6j2Ydv2ngJBYQFdPmtt"
POSTHOG_CAPTURE_URL = "https://us.i.posthog.com/capture/"
EVENT_NAME = "cli_command"

COLLECTED = (
    "command name, whetkit version, Python major.minor, sys.platform, and a random anonymous id"
)
NEVER_COLLECTED = "arguments, file paths, server names, prompts, or results"


def config_path() -> Path:
    return Path.home() / ".whetkit" / "telemetry.json"


def load_config() -> dict:
    try:
        data = json.loads(config_path().read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(config: dict) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def is_enabled() -> bool:
    """Opt-in check: the WHETKIT_TELEMETRY env var wins; else the config file."""
    env = os.environ.get("WHETKIT_TELEMETRY")
    if env is not None:
        return env == "1"
    return load_config().get("enabled") is True


def anonymous_id() -> str:
    """The persisted random UUID (created on first use). It identifies an
    installation, never a person — and an unwritable home directory simply
    yields a fresh id per invocation."""
    config = load_config()
    existing = config.get("anonymous_id")
    if isinstance(existing, str) and existing:
        return existing
    new_id = str(uuid.uuid4())
    with contextlib.suppress(Exception):
        save_config({**config, "anonymous_id": new_id})
    return new_id


def build_payload(command: str) -> dict:
    """The complete event: these fields and NOTHING else."""
    from importlib.metadata import version

    return {
        "api_key": POSTHOG_PROJECT_KEY,
        "event": EVENT_NAME,
        "distinct_id": anonymous_id(),
        "properties": {
            "command": command,
            "whetkit_version": version("whetkit"),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": sys.platform,
        },
    }


def _post(payload: dict) -> None:
    """One capture POST; swallows every failure — telemetry may never break
    the CLI (or even print about it)."""
    try:
        import httpx

        httpx.post(POSTHOG_CAPTURE_URL, json=payload, timeout=2.0)
    except Exception:
        pass


def record_event(command: str) -> threading.Thread | None:
    """Record one command event iff the user opted in. Fire-and-forget: the
    POST happens on a daemon thread; the returned handle exists for tests."""
    if not command or not is_enabled():
        return None
    try:
        payload = build_payload(command)
    except Exception:
        return None
    thread = threading.Thread(target=_post, args=(payload,), daemon=True)
    thread.start()
    return thread
