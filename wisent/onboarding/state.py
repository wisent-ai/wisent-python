"""Where this machine keeps its attempts: the file's shape, reading, writing.

A state file written by another schema, truncated, or unreadable is treated as
absent rather than half-trusted, because a first-use record is not worth
failing an SDK call over. Writes go to a temporary file in the same directory
and are renamed onto the destination, so a killed process leaves either the
previous state or the new one.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

#: The shape of the state file this build writes and accepts.
STATE_SCHEMA_VERSION = 1


def empty_state() -> Dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "current_attempt_id": None,
        "attempts": {},
        "pending_events": [],
    }


def load_state(path: Path) -> Dict[str, Any]:
    """The state on disk, or an empty one when there is nothing usable."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(loaded, dict)
            or not isinstance(loaded.get("attempts"), dict)
            or not isinstance(loaded.get("pending_events"), list)
        ):
            raise ValueError("invalid onboarding state")
        current = loaded.get("current_attempt_id")
        if current is not None and current not in loaded["attempts"]:
            loaded["current_attempt_id"] = None
        return loaded
    except FileNotFoundError:
        return empty_state()
    except (OSError, ValueError, json.JSONDecodeError):
        return empty_state()


def persist(path: Path, state: Dict[str, Any]) -> None:
    """Writes the state whole, or leaves what was there before."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def now() -> str:
    """The timestamp every event and attempt is stamped with."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
