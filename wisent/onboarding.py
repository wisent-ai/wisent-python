"""Durable first-use journey for the public Wisent Python SDK."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PRODUCT_ID = "wisent-python"
CLIENT_ID = "wisent-python"
JOURNEY_ID = "first-use"
JOURNEY_VERSION = "2026-08-04.1"
JOURNEY_VERSION_ID = "12000000-0000-4000-8000-000000000012"
SOURCE_REVISION = "wisent-python-first-use-2026-08-04"
FIRST_SUCCESS_FACT = "api_result_observed"
SUPPORTED_OPERATION = "inference.generate"

CANONICAL_EVENTS = (
    "onboarding_started",
    "onboarding_resumed",
    "onboarding_step_viewed",
    "onboarding_step_completed",
    "onboarding_step_skipped",
    "onboarding_abandoned",
    "onboarding_reset",
    "onboarding_first_success_observed",
    "onboarding_completed",
)

FALLBACK_JOURNEY: Dict[str, Any] = {
    "schema_version": 1,
    "product_id": PRODUCT_ID,
    "journey_id": JOURNEY_ID,
    "journey_version": JOURNEY_VERSION,
    "entry_screen_id": "inspect-journey",
    "first_success_fact": FIRST_SUCCESS_FACT,
    "published_at": "2026-08-04T00:00:00Z",
    "source_revision": SOURCE_REVISION,
    "screens": [
        {
            "screen_id": "inspect-journey",
            "screen_kind": "machine_discovery",
            "title_key": "wisent-python.onboarding.inspect-journey.title",
            "body_key": "wisent-python.onboarding.inspect-journey.body",
            "presentation": {
                "renderer": "machine_discovery",
                "title": "Inspect the pinned first-use journey",
                "body": "Read the pinned journey identity and the supported authenticated inference operation before sending work.",
            },
            "actions": ["inspect"],
            "required": True,
            "completion_evidence": None,
            "entry_conditions": None,
            "fallback_screen_id": None,
            "transitions": [
                {
                    "next_screen_id": "run-inference",
                    "priority": 10,
                    "reason_code": "canonical_progression",
                }
            ],
        },
        {
            "screen_id": "run-inference",
            "screen_kind": "machine_action",
            "title_key": "wisent-python.onboarding.run-inference.title",
            "body_key": "wisent-python.onboarding.run-inference.body",
            "presentation": {
                "renderer": "machine_action",
                "title": "Run one authenticated inference",
                "body": "Call InferenceClient.generate with an explicit model and prompt through the normal authenticated SDK path.",
            },
            "actions": ["run"],
            "required": True,
            "completion_evidence": None,
            "entry_conditions": None,
            "fallback_screen_id": None,
            "transitions": [
                {
                    "next_screen_id": "keep-result",
                    "priority": 10,
                    "reason_code": "canonical_progression",
                }
            ],
        },
        {
            "screen_id": "keep-result",
            "screen_kind": "machine_result",
            "title_key": "wisent-python.onboarding.keep-result.title",
            "body_key": "wisent-python.onboarding.keep-result.body",
            "presentation": {
                "renderer": "machine_result",
                "title": "Keep the structured API result",
                "body": "Inspect the parsed InferenceResponse returned by the API. Authentication configuration and request dispatch alone never complete.",
            },
            "actions": ["inspect_result"],
            "required": True,
            "completion_evidence": {
                "fact": FIRST_SUCCESS_FACT,
                "kind": "fact",
                "operator": "eq",
                "value": True,
            },
            "entry_conditions": None,
            "fallback_screen_id": None,
            "transitions": [],
        },
    ],
    "analytics_contract": {
        "contract_version": "1",
        "surface": "python_sdk_first_use",
        "exposure_event": "onboarding_step_viewed",
        "primary_action_event": "onboarding_step_completed",
        "first_success_event": "onboarding_first_success_observed",
        "completion_event": "onboarding_completed",
    },
    "experiment_contract": None,
}

_EXPECTED_SCREENS = {
    "inspect-journey": ("machine_discovery", {"inspect"}),
    "run-inference": ("machine_action", {"run"}),
    "keep-result": ("machine_result", {"inspect_result"}),
}
_ALLOWED_RENDERERS = {"machine_discovery", "machine_action", "machine_result"}


class OnboardingError(Exception):
    """A stable machine-readable first-use operation error."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


class FirstUseRuntime:
    """Product-owned adapter for Echo first-use bundles and Stado operations."""

    def __init__(self) -> None:
        default_path = Path.home() / ".wisent" / "onboarding-state.json"
        self._state_path = Path(os.environ.get("WISENT_ONBOARDING_STATE_PATH", str(default_path)))
        self._stado_url = os.environ.get("STADO_URL", "https://stado.wisent.ai").rstrip("/")
        self._stado_token = os.environ.get("STADO_ONBOARDING_TOKEN", "")
        self._timeout = float(os.environ.get("STADO_ONBOARDING_TIMEOUT_SECONDS", "2"))
        self._lock = threading.RLock()
        self._state = self._load_state()
        if self._state["pending_events"]:
            self._schedule_flush()

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {"schema_version": 1, "current_attempt_id": None, "attempts": {}, "pending_events": []}

    def _load_state(self) -> Dict[str, Any]:
        try:
            loaded = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or not isinstance(loaded.get("attempts"), dict) or not isinstance(loaded.get("pending_events"), list):
                raise ValueError("invalid onboarding state")
            current = loaded.get("current_attempt_id")
            if current is not None and current not in loaded["attempts"]:
                loaded["current_attempt_id"] = None
            return loaded
        except FileNotFoundError:
            return self._empty_state()
        except (OSError, ValueError, json.JSONDecodeError):
            return self._empty_state()

    def _persist_locked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_name(".{}.{}.tmp".format(self._state_path.name, uuid.uuid4().hex))
        encoded = json.dumps(self._state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(self._state_path))
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _request_stado(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._stado_token:
            raise RuntimeError("STADO_ONBOARDING_TOKEN is not configured")
        request = urllib.request.Request(
            "{}/api/integration/onboarding/{}".format(self._stado_url, operation),
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Authorization": "Bearer {}".format(self._stado_token), "Content-Type": "application/json", "X-Stado-Client-Id": CLIENT_ID},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
                return decoded if isinstance(decoded, dict) else {}
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("Stado {} unavailable".format(operation)) from error

    @staticmethod
    def _validate_bundle(bundle: Dict[str, Any]) -> bool:
        if (bundle.get("product_id") != PRODUCT_ID or bundle.get("journey_id") != JOURNEY_ID or bundle.get("journey_version") != JOURNEY_VERSION or bundle.get("first_success_fact") != FIRST_SUCCESS_FACT or bundle.get("source_revision") != SOURCE_REVISION or bundle.get("entry_screen_id") != "inspect-journey"):
            return False
        if bundle.get("journey_version_id") not in (None, JOURNEY_VERSION_ID):
            return False
        screens = bundle.get("screens")
        if not isinstance(screens, list) or len(screens) != 3:
            return False
        found = set()
        for screen in screens:
            if not isinstance(screen, dict):
                return False
            screen_id = screen.get("screen_id")
            if screen_id not in _EXPECTED_SCREENS or screen_id in found:
                return False
            expected_kind, allowed_actions = _EXPECTED_SCREENS[screen_id]
            if screen.get("screen_kind") != expected_kind:
                return False
            actions = screen.get("actions")
            if (
                not isinstance(actions, list)
                or len(actions) != len(allowed_actions)
                or set(actions) != allowed_actions
            ):
                return False
            presentation = screen.get("presentation")
            if not isinstance(presentation, dict) or presentation.get("renderer") not in _ALLOWED_RENDERERS:
                return False
            if not isinstance(presentation.get("title"), str) or not isinstance(presentation.get("body"), str):
                return False
            if len(presentation["title"]) > 200 or len(presentation["body"]) > 2000:
                return False
            found.add(screen_id)
        return found == set(_EXPECTED_SCREENS)

    def _load_bundle(self) -> Tuple[Dict[str, Any], str]:
        try:
            response = self._request_stado("bundle.read", {"client_id": CLIENT_ID, "product_id": PRODUCT_ID, "journey_id": JOURNEY_ID, "journey_version": JOURNEY_VERSION})
            candidate = response.get("bundle", response)
            if isinstance(candidate, dict) and self._validate_bundle(candidate):
                candidate = dict(candidate)
                candidate["journey_version_id"] = JOURNEY_VERSION_ID
                return candidate, "stado"
        except RuntimeError:
            pass
        return FALLBACK_JOURNEY, "bundled"

    def _assign(self, attempt_id: str) -> Dict[str, str]:
        fallback = {"experiment_id": "wisent-python-first-use-sequence", "variant_id": "control"}
        try:
            response = self._request_stado("experiments.assign", {"client_id": CLIENT_ID, "product_id": PRODUCT_ID, "journey_id": JOURNEY_ID, "journey_version_id": JOURNEY_VERSION_ID, "attempt_id": attempt_id})
            assignment = response.get("assignment", response)
            experiment_id = assignment.get("experiment_id")
            variant_id = assignment.get("variant_id")
            if isinstance(experiment_id, str) and isinstance(variant_id, str):
                return {"experiment_id": experiment_id, "variant_id": variant_id}
        except RuntimeError:
            pass
        return fallback

    def _read_central_state(self, attempt_id: str) -> None:
        try:
            self._request_stado("state.read", {"client_id": CLIENT_ID, "product_id": PRODUCT_ID, "journey_id": JOURNEY_ID, "journey_version_id": JOURNEY_VERSION_ID, "attempt_id": attempt_id})
        except RuntimeError:
            pass

    def _event_locked(self, attempt: Dict[str, Any], event_name: str, screen_id: str, properties: Optional[Dict[str, Any]] = None) -> None:
        if event_name not in CANONICAL_EVENTS:
            raise ValueError("unsupported onboarding event: {}".format(event_name))
        assignment = attempt["assignment"]
        self._state["pending_events"].append({
            "event_id": str(uuid.uuid4()), "event_name": event_name, "occurred_at": self._now(),
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
            self._request_stado("events.collect", {"client_id": CLIENT_ID, "product_id": PRODUCT_ID, "events": pending})
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
        bundle, bundle_source = self._load_bundle()
        with self._lock:
            selected_id = attempt_id or self._state.get("current_attempt_id") or str(uuid.uuid4())
            self._validate_attempt_id(selected_id)
            attempt = self._state["attempts"].get(selected_id)
            if attempt is None:
                now = self._now()
                attempt = {"attempt_id": selected_id, "current_screen_id": "inspect-journey", "completed": False, "evidence": {}, "assignment": self._assign(selected_id), "created_at": now, "updated_at": now}
                self._state["attempts"][selected_id] = attempt
                self._event_locked(attempt, "onboarding_started", "inspect-journey", {"bundle_source": bundle_source})
            else:
                self._event_locked(attempt, "onboarding_resumed", attempt["current_screen_id"])
            self._state["current_attempt_id"] = selected_id
            self._event_locked(attempt, "onboarding_step_viewed", attempt["current_screen_id"])
            attempt["updated_at"] = self._now()
            self._persist_locked()
            public = self._public_attempt(attempt)
        self._read_central_state(selected_id)
        self._schedule_flush()
        return {"journey": bundle, "bundle_source": bundle_source, "attempt": public}

    def inspect(self, attempt_id: Optional[str] = None) -> Dict[str, Any]:
        """Acknowledge the pinned journey before issuing the supported request."""
        with self._lock:
            attempt = self._attempt_locked(attempt_id)
            if attempt["completed"]:
                raise OnboardingError("journey_completed", "the first-use journey is already complete", 409)
            if attempt["current_screen_id"] == "inspect-journey":
                attempt["evidence"]["journey_inspected"] = True
                self._event_locked(attempt, "onboarding_step_completed", "inspect-journey", {"evidence": "journey_inspected"})
                attempt["current_screen_id"] = "run-inference"
                self._event_locked(attempt, "onboarding_step_viewed", "run-inference")
                attempt["updated_at"] = self._now()
                self._persist_locked()
            elif attempt["current_screen_id"] != "run-inference":
                raise OnboardingError("invalid_transition", "journey inspection is unavailable from the current screen", 409)
            public = self._public_attempt(attempt)
        self._schedule_flush()
        return {"attempt": public, "supported_operation": SUPPORTED_OPERATION}

    def _observe_api_result(self, operation: str, response: Dict[str, Any]) -> None:
        """Record success only after a parsed authenticated SDK result exists."""
        if operation != SUPPORTED_OPERATION or not isinstance(response, dict):
            return
        with self._lock:
            current_id = self._state.get("current_attempt_id")
            if not isinstance(current_id, str):
                return
            attempt = self._state["attempts"].get(current_id)
            if (
                not isinstance(attempt, dict)
                or attempt.get("completed")
                or attempt.get("current_screen_id") != "run-inference"
            ):
                return
            digest = hashlib.sha256(
                json.dumps(
                    response,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
            prior_evidence = dict(attempt["evidence"])
            prior_screen = attempt["current_screen_id"]
            prior_completed = attempt["completed"]
            prior_updated_at = attempt["updated_at"]
            prior_event_count = len(self._state["pending_events"])
            try:
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
                    {
                        "fact": FIRST_SUCCESS_FACT,
                        "operation": operation,
                        "result_sha256": digest,
                    },
                )
                attempt["current_screen_id"] = "keep-result"
                attempt["completed"] = True
                attempt["updated_at"] = self._now()
                self._event_locked(
                    attempt, "onboarding_step_viewed", "keep-result"
                )
                self._event_locked(
                    attempt,
                    "onboarding_completed",
                    "keep-result",
                    {"fact": FIRST_SUCCESS_FACT},
                )
                self._persist_locked()
            except Exception:
                attempt["evidence"] = prior_evidence
                attempt["current_screen_id"] = prior_screen
                attempt["completed"] = prior_completed
                attempt["updated_at"] = prior_updated_at
                del self._state["pending_events"][prior_event_count:]
                raise
        self._schedule_flush()

    def state(self, attempt_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            return {"attempt": self._public_attempt(self._attempt_locked(attempt_id))}

    def abandon(self, attempt_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            attempt = self._attempt_locked(attempt_id)
            if not attempt["completed"]:
                self._event_locked(attempt, "onboarding_abandoned", attempt["current_screen_id"])
                attempt["updated_at"] = self._now()
                self._persist_locked()
            public = self._public_attempt(attempt)
        self._schedule_flush()
        return {"attempt": public}

    def reset(self, attempt_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            attempt = self._attempt_locked(attempt_id)
            self._event_locked(attempt, "onboarding_reset", attempt["current_screen_id"])
            now = self._now()
            replacement = {"attempt_id": attempt["attempt_id"], "current_screen_id": "inspect-journey", "completed": False, "evidence": {}, "assignment": attempt["assignment"], "created_at": now, "updated_at": now}
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
            raise OnboardingError("attempt_not_found", "start first use before calling this operation", 404)
        self._validate_attempt_id(attempt_id)
        attempt = self._state["attempts"].get(attempt_id)
        if not isinstance(attempt, dict):
            raise OnboardingError("attempt_not_found", "start first use before calling this operation", 404)
        return attempt

    @staticmethod
    def _validate_attempt_id(attempt_id: Any) -> None:
        if not isinstance(attempt_id, str) or not attempt_id or len(attempt_id) > 128:
            raise OnboardingError("invalid_attempt_id", "attempt_id must be a non-empty string of at most 128 characters")
