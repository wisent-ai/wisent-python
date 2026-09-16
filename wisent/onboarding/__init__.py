"""Durable first-use journey for the public Wisent Python SDK.

Four parts, in the order they depend on each other:

* ``definition`` - what the SDK published: the identity, the pinned journey
  and the error its operations raise.
* ``state`` - the file this machine keeps its attempts in.
* ``transport`` - Stado's onboarding operations, and the check a served
  definition must pass to be used instead of the pinned one.
* ``runtime`` - one machine's attempt, and the only path that can complete it.

``from wisent.onboarding import FirstUseRuntime, OnboardingError`` is
unchanged: this was one 468-line module.
"""

from .definition import (
    CANONICAL_EVENTS,
    FIRST_SUCCESS_FACT,
    JOURNEY_ID,
    JOURNEY_VERSION,
    JOURNEY_VERSION_ID,
    PINNED_JOURNEY,
    PRODUCT_ID,
    SOURCE_REVISION,
    SUPPORTED_OPERATION,
    OnboardingError,
)
from .runtime import FirstUseRuntime
from .transport import StadoOnboarding, validate_bundle

__all__ = [
    "CANONICAL_EVENTS",
    "FIRST_SUCCESS_FACT",
    "FirstUseRuntime",
    "JOURNEY_ID",
    "JOURNEY_VERSION",
    "JOURNEY_VERSION_ID",
    "OnboardingError",
    "PINNED_JOURNEY",
    "PRODUCT_ID",
    "SOURCE_REVISION",
    "SUPPORTED_OPERATION",
    "StadoOnboarding",
    "validate_bundle",
]
