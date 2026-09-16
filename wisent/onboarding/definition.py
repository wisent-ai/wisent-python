"""What the Wisent Python SDK published as its first-use journey.

The identity, the pinned definition, the screens a bundle must carry and the
error the operations raise. Nothing here reaches a network or holds an
attempt: Stado is in transport.py and the attempt is in runtime.py.
"""

from __future__ import annotations

from typing import Any, Dict

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

PINNED_JOURNEY: Dict[str, Any] = {
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

EXPECTED_SCREENS = {
    "inspect-journey": ("machine_discovery", {"inspect"}),
    "run-inference": ("machine_action", {"run"}),
    "keep-result": ("machine_result", {"inspect_result"}),
}
ALLOWED_RENDERERS = {"machine_discovery", "machine_action", "machine_result"}


class OnboardingError(Exception):
    """A stable machine-readable first-use operation error."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status
