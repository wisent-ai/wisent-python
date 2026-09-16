"""One machine's attempt at the SDK's first-use journey.

The runtime owns the position in the three-screen journey and the queue of
events still owed to Stado; the file those live in is `state.py` and the
operations they are sent through are `transport.py`.

First use completes on a parsed authenticated result and on nothing else:
configuring credentials and dispatching a request do not finish the journey,
which is why :meth:`FirstUseRuntime._observe_api_result` is the only path that
can complete it, and why it restores the attempt exactly as it was if any part
of recording that success fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .definition import (
    CANONICAL_EVENTS,
    CLIENT_ID,
    FIRST_SUCCESS_FACT,
    JOURNEY_ID,
    JOURNEY_VERSION,
    JOURNEY_VERSION_ID,
    PRODUCT_ID,
    SUPPORTED_OPERATION,
    OnboardingError,
)
from .state import empty_state, load_state, now, persist
from .transport import StadoOnboarding

#: What an operation returns when the attempt it names is not there, and when
#: the journey does not allow what was asked for.
ATTEMPT_NOT_FOUND_STATUS = 404
INVALID_TRANSITION_STATUS = 409

#: An attempt identifier is chosen by the caller, so it is bounded.
MAX_ATTEMPT_ID_CHARACTERS = 128


class FirstUseRuntime:
    """Product-owned adapter for Echo first-use bundles and Stado operations."""

    def __init__(self) -> None:
        default_path = Path.home() / ".wisent" / "onboarding-state.json"
        self._state_path = Path(os.environ.get("WISENT_ONBOARDING_STATE_PATH", str(default_path)))
        self._stado = StadoOnboarding()
        self._lock = threading.RLock()
        self._state = load_state(self._state_path)
        if self._state["pending_events"]:
            self._schedule_flush()

    def _persist_locked(self) -> None:
        persist(self._state_path, self._state)

    def _event_locked(self, attempt: Dict[str, Any], event_name: str, screen_id: str, properties: Optional[Dict[str, Any]] = None) -> None:
        if event_name not in CANONICAL_EVENTS:
            raise ValueError("unsupported onboarding event: {}".format(event_name))
        assignment = attempt["assignment"]
        self._state["pending_events"].append({
            "event_id": str(uuid.uuid4()), "event_name": event_name, "occurred_at": now(),
            "client_id": CLIENT_ID, "product_id": PRODUCT_ID, "journey_id": JOURNEY_ID,
            "journey_version": JOURNEY_VERSION, "journey_version_id": JOURNEY_VERSION_ID,
            "attempt_id": attempt["attempt_id"], "screen_id": screen_id,
            "experiment_id": assignment["experiment_id"], "variant_id": assignment["variant_id"],
            "properties": properties or {},
        })

    def _flush_pending(self) -> None:
        with self._lock:
            pending = list(self._state["pending_events"])
        if not pending:
            return
        try:
            self._stado.collect(pending)
        except RuntimeError:
            return
        delivered = {event["event_id"] for event in pending}
        with self._lock:
            self._state["pending_events"] = [event for event in self._state["pending_events"] if event.get("event_id") not in delivered]
            self._persist_locked()

    def _schedule_flush(self) -> None:
        threading.Thread(target=self._flush_pending, name="wisent-python-onboarding-events", daemon=True).start()

    @staticmethod
    def _public_attempt(attempt: Dict[str, Any]) -> Dict[str, Any]:
        return {"attempt_id": attempt["attempt_id"], "journey_version_id": JOURNEY_VERSION_ID, "current_screen_id": attempt["current_screen_id"], "completed": attempt["completed"], "evidence": dict(attempt["evidence"]), "assignment": dict(attempt["assignment"]), "updated_at": attempt["updated_at"]}

    def start(self, attempt_id: Optional[str] = None) -> Dict[str, Any]:
        bundle, bundle_source = self._stado.read_bundle()
        with self._lock:
            selected_id = attempt_id or self._state.get("current_attempt_id") or str(uuid.uuid4())
            self._validate_attempt_id(selected_id)
            attempt = self._state["attempts"].get(selected_id)
            if attempt is None:
                attempt = self._new_attempt_locked(selected_id, bundle_source)
            else:
                self._event_locked(attempt, "onboarding_resumed", attempt["current_screen_id"])
            self._state["current_attempt_id"] = selected_id
            self._event_locked(attempt, "onboarding_step_viewed", attempt["current_screen_id"])
            attempt["updated_at"] = now()
            self._persist_locked()
            public = self._public_attempt(attempt)
        self._stado.read_state(selected_id)
        self._schedule_flush()
        return {"journey": bundle, "bundle_source": bundle_source, "attempt": public}

    def _new_attempt_locked(self, attempt_id: str, bundle_source: str) -> Dict[str, Any]:
        stamp = now()
        attempt = {"attempt_id": attempt_id, "current_screen_id": "inspect-journey", "completed": False, "evidence": {}, "assignment": self._stado.assign(attempt_id), "created_at": stamp, "updated_at": stamp}
        self._state["attempts"][attempt_id] = attempt
        self._event_locked(attempt, "onboarding_started", "inspect-journey", {"bundle_source": bundle_source})
        return attempt

    def inspect(self, attempt_id: Optional[str] = None) -> Dict[str, Any]:
        """Acknowledge the pinned journey before issuing the supported request."""
        with self._lock:
            attempt = self._attempt_locked(attempt_id)
            if attempt["completed"]:
                raise OnboardingError("journey_completed", "the first-use journey is already complete", INVALID_TRANSITION_STATUS)
            if attempt["current_screen_id"] == "inspect-journey":
                attempt["evidence"]["journey_inspected"] = True
                self._event_locked(attempt, "onboarding_step_completed", "inspect-journey", {"evidence": "journey_inspected"})
                attempt["current_screen_id"] = "run-inference"
                self._event_locked(attempt, "onboarding_step_viewed", "run-inference")
                attempt["updated_at"] = now()
                self._persist_locked()
            elif attempt["current_screen_id"] != "run-inference":
                raise OnboardingError("invalid_transition", "journey inspection is unavailable from the current screen", INVALID_TRANSITION_STATUS)
            public = self._public_attempt(attempt)
        self._schedule_flush()
        return {"attempt": public, "supported_operation": SUPPORTED_OPERATION}

    def _observe_api_result(self, operation: str, response: Dict[str, Any]) -> None:
        """Record success only after a parsed authenticated SDK result exists."""
        if operation != SUPPORTED_OPERATION or not isinstance(response, dict):
            return
        with self._lock:
            attempt = self._awaiting_result_locked()
            if attempt is None:
                return
            restore = {
                "evidence": dict(attempt["evidence"]),
                "current_screen_id": attempt["current_screen_id"],
                "completed": attempt["completed"],
                "updated_at": attempt["updated_at"],
                "event_count": len(self._state["pending_events"]),
            }
            try:
                self._complete_locked(attempt, operation, self._result_digest(response))
            except Exception:
                self._restore_locked(attempt, restore)
                raise
        self._schedule_flush()

    def _awaiting_result_locked(self) -> Optional[Dict[str, Any]]:
        """The attempt a result could complete, if there is one."""
        current_id = self._state.get("current_attempt_id")
        if not isinstance(current_id, str):
            return None
        attempt = self._state["attempts"].get(current_id)
        if (
            not isinstance(attempt, dict)
            or attempt.get("completed")
            or attempt.get("current_screen_id") != "run-inference"
        ):
            return None
        return attempt

    @staticmethod
    def _result_digest(response: Dict[str, Any]) -> str:
        """What identifies the result that completed first use."""
        return hashlib.sha256(
            json.dumps(response, sort_keys=True, default=str, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    def _complete_locked(self, attempt: Dict[str, Any], operation: str, digest: str) -> None:
        attempt["evidence"][FIRST_SUCCESS_FACT] = True
        attempt["evidence"]["operation"] = operation
        attempt["evidence"]["result_sha256"] = digest
        self._event_locked(
            attempt,
            "onboarding_step_completed",
            "run-inference",
            {"evidence": FIRST_SUCCESS_FACT, "operation": operation},
        )
        self._event_locked(
            attempt,
            "onboarding_first_success_observed",
            "run-inference",
            {"fact": FIRST_SUCCESS_FACT, "operation": operation, "result_sha256": digest},
        )
        attempt["current_screen_id"] = "keep-result"
        attempt["completed"] = True
        attempt["updated_at"] = now()
        self._event_locked(attempt, "onboarding_step_viewed", "keep-result")
        self._event_locked(attempt, "onboarding_completed", "keep-result", {"fact": FIRST_SUCCESS_FACT})
        self._persist_locked()

    def _restore_locked(self, attempt: Dict[str, Any], restore: Dict[str, Any]) -> None:
        """The attempt as it was, with the events this failure had queued dropped."""
        attempt["evidence"] = restore["evidence"]
        attempt["current_screen_id"] = restore["current_screen_id"]
        attempt["completed"] = restore["completed"]
        attempt["updated_at"] = restore["updated_at"]
        del self._state["pending_events"][restore["event_count"]:]

    def state(self, attempt_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            return {"attempt": self._public_attempt(self._attempt_locked(attempt_id))}

    def abandon(self, attempt_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            attempt = self._attempt_locked(attempt_id)
            if not attempt["completed"]:
                self._event_locked(attempt, "onboarding_abandoned", attempt["current_screen_id"])
                attempt["updated_at"] = now()
                self._persist_locked()
            public = self._public_attempt(attempt)
        self._schedule_flush()
        return {"attempt": public}

    def reset(self, attempt_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            attempt = self._attempt_locked(attempt_id)
            self._event_locked(attempt, "onboarding_reset", attempt["current_screen_id"])
            stamp = now()
            replacement = {"attempt_id": attempt["attempt_id"], "current_screen_id": "inspect-journey", "completed": False, "evidence": {}, "assignment": attempt["assignment"], "created_at": stamp, "updated_at": stamp}
            self._state["attempts"][attempt["attempt_id"]] = replacement
            self._state["current_attempt_id"] = attempt["attempt_id"]
            self._event_locked(replacement, "onboarding_started", "inspect-journey", {"reason_code": "reset"})
            self._event_locked(replacement, "onboarding_step_viewed", "inspect-journey")
            self._persist_locked()
            public = self._public_attempt(replacement)
        self._schedule_flush()
        return {"attempt": public}

    def _attempt_locked(self, requested_attempt_id: Optional[str]) -> Dict[str, Any]:
        attempt_id = requested_attempt_id or self._state.get("current_attempt_id")
        if not isinstance(attempt_id, str):
            raise OnboardingError("attempt_not_found", "start first use before calling this operation", ATTEMPT_NOT_FOUND_STATUS)
        self._validate_attempt_id(attempt_id)
        attempt = self._state["attempts"].get(attempt_id)
        if not isinstance(attempt, dict):
            raise OnboardingError("attempt_not_found", "start first use before calling this operation", ATTEMPT_NOT_FOUND_STATUS)
        return attempt

    @staticmethod
    def _validate_attempt_id(attempt_id: Any) -> None:
        if not isinstance(attempt_id, str) or not attempt_id or len(attempt_id) > MAX_ATTEMPT_ID_CHARACTERS:
            raise OnboardingError("invalid_attempt_id", "attempt_id must be a non-empty string of at most 128 characters")
