"""The SDK's first-use journey, run for real with Stado unreachable.

No token is configured, so every Stado call raises inside the transport and
the runtime takes the path a developer's machine takes: the pinned journey,
events queued on disk, and the journey completing on a parsed result anyway.
What the tests read back is the state file the product wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wisent.onboarding import (
    FIRST_SUCCESS_FACT,
    PINNED_JOURNEY,
    SUPPORTED_OPERATION,
    FirstUseRuntime,
    OnboardingError,
    validate_bundle,
)

BUILD_ROOT = Path(__file__).resolve().parents[2] / "build" / "first-use-tests"


@pytest.fixture
def runtime(request, monkeypatch) -> FirstUseRuntime:
    """A runtime with a state file of its own and no Stado to reach.

    The path is inside the repository's ignored build directory, named after
    the test, so a failing run leaves the state file it failed on and the next
    run does not inherit it.
    """
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    path = BUILD_ROOT / f"{request.node.name}.json"
    if path.exists():
        path.unlink()
    monkeypatch.setenv("WISENT_ONBOARDING_STATE_PATH", str(path))
    monkeypatch.delenv("STADO_ONBOARDING_TOKEN", raising=False)
    created = FirstUseRuntime()
    created.state_file = path
    return created


def _written(runtime: FirstUseRuntime) -> dict:
    return json.loads(runtime.state_file.read_text(encoding="utf-8"))


def test_start_uses_the_pinned_journey_when_stado_cannot_be_reached(runtime) -> None:
    started = runtime.start()

    assert started["bundle_source"] == "bundled"
    assert started["journey"] == PINNED_JOURNEY
    assert started["attempt"]["current_screen_id"] == "inspect-journey"
    assert started["attempt"]["assignment"]["variant_id"] == "control"


def test_the_three_screens_complete_on_a_parsed_result(runtime) -> None:
    attempt_id = runtime.start()["attempt"]["attempt_id"]

    inspected = runtime.inspect()
    assert inspected["attempt"]["current_screen_id"] == "run-inference"
    assert inspected["supported_operation"] == SUPPORTED_OPERATION
    assert inspected["attempt"]["evidence"]["journey_inspected"] is True

    runtime._observe_api_result(SUPPORTED_OPERATION, {"text": "hello", "model": "wisent-1b"})

    completed = runtime.state()["attempt"]
    assert completed["attempt_id"] == attempt_id
    assert completed["completed"] is True
    assert completed["current_screen_id"] == "keep-result"
    assert completed["evidence"][FIRST_SUCCESS_FACT] is True
    assert len(completed["evidence"]["result_sha256"]) == 64


def test_the_events_the_journey_owes_are_kept_on_disk(runtime) -> None:
    runtime.start()
    runtime.inspect()

    names = [event["event_name"] for event in _written(runtime)["pending_events"]]

    assert names[:2] == ["onboarding_started", "onboarding_step_viewed"]
    assert "onboarding_step_completed" in names
    assert all(event["product_id"] == "wisent-python" for event in _written(runtime)["pending_events"])


def test_a_result_for_another_operation_does_not_complete_the_journey(runtime) -> None:
    runtime.start()
    runtime.inspect()

    runtime._observe_api_result("activations.extract", {"vectors": [1, 2, 3]})

    attempt = runtime.state()["attempt"]
    assert attempt["completed"] is False
    assert attempt["current_screen_id"] == "run-inference"
    assert FIRST_SUCCESS_FACT not in attempt["evidence"]


def test_an_operation_before_the_journey_started_is_refused(runtime) -> None:
    with pytest.raises(OnboardingError) as raised:
        runtime.state()

    assert raised.value.code == "attempt_not_found"
    assert raised.value.status == 404


def test_inspecting_a_completed_journey_is_refused(runtime) -> None:
    runtime.start()
    runtime.inspect()
    runtime._observe_api_result(SUPPORTED_OPERATION, {"text": "hello"})

    with pytest.raises(OnboardingError) as raised:
        runtime.inspect()

    assert raised.value.code == "journey_completed"
    assert raised.value.status == 409


def test_an_oversized_attempt_identifier_is_refused(runtime) -> None:
    with pytest.raises(OnboardingError) as raised:
        runtime.start("a" * 129)

    assert raised.value.code == "invalid_attempt_id"


def test_reset_opens_a_new_attempt_under_the_same_identifier(runtime) -> None:
    attempt_id = runtime.start()["attempt"]["attempt_id"]
    runtime.inspect()

    after = runtime.reset()["attempt"]

    assert after["attempt_id"] == attempt_id
    assert after["current_screen_id"] == "inspect-journey"
    assert after["completed"] is False
    assert after["evidence"] == {}


def test_abandon_records_it_and_keeps_the_attempt_readable(runtime) -> None:
    runtime.start()

    abandoned = runtime.abandon()["attempt"]

    assert abandoned["completed"] is False
    names = [event["event_name"] for event in _written(runtime)["pending_events"]]
    assert "onboarding_abandoned" in names


def test_a_served_definition_from_another_journey_is_refused() -> None:
    assert validate_bundle(PINNED_JOURNEY) is True

    foreign = dict(PINNED_JOURNEY)
    foreign["journey_version"] = "2099-01-01.1"

    assert validate_bundle(foreign) is False


def test_a_served_definition_missing_a_screen_is_refused() -> None:
    truncated = dict(PINNED_JOURNEY)
    truncated["screens"] = PINNED_JOURNEY["screens"][:2]

    assert validate_bundle(truncated) is False
