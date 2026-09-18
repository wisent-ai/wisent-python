"""Stado's onboarding operations, and the check a served bundle must pass.

Every call is one POST to Stado's integration API. Without
``STADO_ONBOARDING_TOKEN`` the SDK is in its documented local mode: the pinned
journey, the control variant, and an event queue that stays on disk. With the
token configured, an unreachable Stado, a non-object answer, an answer without
the operation's object, or a served definition this release does not render
raises ``RuntimeError`` naming the operation, and the caller sees it.

``validate_bundle`` is what makes a served definition usable. A bundle whose
identity, screens, actions or renderers are not the ones this release renders
is refused, so the SDK never drives its journey from someone else's
definition.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

from .definition import (
    ALLOWED_RENDERERS,
    CLIENT_ID,
    EXPECTED_SCREENS,
    FIRST_SUCCESS_FACT,
    JOURNEY_ID,
    JOURNEY_VERSION,
    JOURNEY_VERSION_ID,
    PINNED_JOURNEY,
    PRODUCT_ID,
    SOURCE_REVISION,
)

#: The journey a screen count, a title and a body must fit for this release to
#: render it. The limits are the published contract's, not a preference.
EXPECTED_SCREEN_COUNT = 3
MAX_TITLE_CHARACTERS = 200
MAX_BODY_CHARACTERS = 2000

#: The experiment this journey belongs to, and the variant a machine gets when
#: Stado cannot be asked.
CONTROL_ASSIGNMENT = {
    "experiment_id": "wisent-python-first-use-sequence",
    "variant_id": "control",
}


def validate_bundle(bundle: Dict[str, Any]) -> bool:
    """Whether this served definition is the journey this release renders."""
    if (
        bundle.get("product_id") != PRODUCT_ID
        or bundle.get("journey_id") != JOURNEY_ID
        or bundle.get("journey_version") != JOURNEY_VERSION
        or bundle.get("first_success_fact") != FIRST_SUCCESS_FACT
        or bundle.get("source_revision") != SOURCE_REVISION
        or bundle.get("entry_screen_id") != "inspect-journey"
    ):
        return False
    if bundle.get("journey_version_id") not in (None, JOURNEY_VERSION_ID):
        return False
    screens = bundle.get("screens")
    if not isinstance(screens, list) or len(screens) != EXPECTED_SCREEN_COUNT:
        return False
    found = set()
    for screen in screens:
        if not _valid_screen(screen, found):
            return False
        found.add(screen["screen_id"])
    return found == set(EXPECTED_SCREENS)


def _valid_screen(screen: Any, found: set) -> bool:
    """One screen of a served definition, against the screen it claims to be."""
    if not isinstance(screen, dict):
        return False
    screen_id = screen.get("screen_id")
    if screen_id not in EXPECTED_SCREENS or screen_id in found:
        return False
    expected_kind, allowed_actions = EXPECTED_SCREENS[screen_id]
    if screen.get("screen_kind") != expected_kind:
        return False
    actions = screen.get("actions")
    if (
        not isinstance(actions, list)
        or len(actions) != len(allowed_actions)
        or set(actions) != allowed_actions
    ):
        return False
    return _valid_presentation(screen.get("presentation"))


def _valid_presentation(presentation: Any) -> bool:
    if not isinstance(presentation, dict) or presentation.get("renderer") not in ALLOWED_RENDERERS:
        return False
    if not isinstance(presentation.get("title"), str) or not isinstance(presentation.get("body"), str):
        return False
    return (
        len(presentation["title"]) <= MAX_TITLE_CHARACTERS
        and len(presentation["body"]) <= MAX_BODY_CHARACTERS
    )


class StadoOnboarding:
    """The four onboarding operations Stado publishes, over HTTPS.

    The timeout default is short on purpose: first-use telemetry must never
    hold up an SDK call. The operator sets STADO_ONBOARDING_TIMEOUT_SECONDS to
    change it, and STADO_ONBOARDING_TOKEN is what makes any of this reachable
    at all.
    """

    def __init__(self) -> None:
        self.url = os.environ.get("STADO_URL", "https://stado.wisent.ai").rstrip("/")
        self.token = os.environ.get("STADO_ONBOARDING_TOKEN", "")
        self.timeout = float(os.environ.get("STADO_ONBOARDING_TIMEOUT_SECONDS", "2"))

    @property
    def configured(self) -> bool:
        """Whether Stado is to be asked at all."""
        return bool(self.token)

    def request(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.token:
            raise RuntimeError("STADO_ONBOARDING_TOKEN is not configured")
        request = urllib.request.Request(
            "{}/api/integration/onboarding/{}".format(self.url, operation),
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": "Bearer {}".format(self.token),
                "Content-Type": "application/json",
                "X-Stado-Client-Id": CLIENT_ID,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("Stado {} unavailable".format(operation)) from error
        if not isinstance(decoded, dict):
            raise RuntimeError("Stado {} answered with {} instead of an object".format(operation, type(decoded).__name__))
        return decoded

    def read_bundle(self) -> Tuple[Dict[str, Any], str]:
        """The served definition, or the pinned one when Stado is not configured.

        The second element names which of the two was used, and it is reported
        with the attempt, so a machine can say where its journey came from.
        """
        if not self.configured:
            return PINNED_JOURNEY, "bundled"
        response = self.request(
            "bundle.read",
            {
                "client_id": CLIENT_ID,
                "product_id": PRODUCT_ID,
                "journey_id": JOURNEY_ID,
                "journey_version": JOURNEY_VERSION,
            },
        )
        candidate = response.get("bundle")
        if not isinstance(candidate, dict):
            raise RuntimeError("Stado bundle.read answered without a bundle object")
        if not validate_bundle(candidate):
            raise RuntimeError(
                "Stado bundle.read served a definition this release does not render; "
                "expected journey {} version {}".format(JOURNEY_ID, JOURNEY_VERSION)
            )
        candidate = dict(candidate)
        candidate["journey_version_id"] = JOURNEY_VERSION_ID
        return candidate, "stado"

    def assign(self, attempt_id: str) -> Dict[str, str]:
        """The experiment variant for this attempt, or the control variant when Stado is not configured."""
        if not self.configured:
            return dict(CONTROL_ASSIGNMENT)
        response = self.request(
            "experiments.assign",
            {
                "client_id": CLIENT_ID,
                "product_id": PRODUCT_ID,
                "journey_id": JOURNEY_ID,
                "journey_version_id": JOURNEY_VERSION_ID,
                "attempt_id": attempt_id,
            },
        )
        assignment = response.get("assignment")
        if not isinstance(assignment, dict):
            raise RuntimeError("Stado experiments.assign answered without an assignment object")
        experiment_id = assignment.get("experiment_id")
        variant_id = assignment.get("variant_id")
        if not isinstance(experiment_id, str) or not isinstance(variant_id, str):
            raise RuntimeError("Stado experiments.assign answered without experiment_id and variant_id strings")
        return {"experiment_id": experiment_id, "variant_id": variant_id}

    def read_state(self, attempt_id: str) -> None:
        """Tells Stado this machine resumed; nothing is sent when Stado is not configured."""
        if not self.configured:
            return
        self.request(
            "state.read",
            {
                "client_id": CLIENT_ID,
                "product_id": PRODUCT_ID,
                "journey_id": JOURNEY_ID,
                "journey_version_id": JOURNEY_VERSION_ID,
                "attempt_id": attempt_id,
            },
        )

    def collect(self, events: list) -> None:
        """Hands over queued events; the caller keeps them if this raises."""
        self.request(
            "events.collect",
            {"client_id": CLIENT_ID, "product_id": PRODUCT_ID, "events": events},
        )
