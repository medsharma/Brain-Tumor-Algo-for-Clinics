"""Adapter for session B's input validation (Contract 3).

Session B owns ``src/input_validation.py`` and its ``validate_image()``
function. That module did not exist when this app was built, so this adapter
imports it if present and falls back to an obvious stub if not.

The stub always answers "in scope". That is exactly what makes it dangerous:
with the stub in place the CANNOT READ THIS IMAGE call can never fire from the
rejector, so a chest X-ray or a photograph of a wall would be handed to the
classifier and get a confident tumour answer. So the stub is treated the same
way as the stub config. The app refuses to start on it outside development.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

STUB_METHOD = "STUB_always_in_scope"
STUB_REASON = (
    "STUB VALIDATOR. Session B's input check is not installed, so nothing is "
    "being rejected. Not for clinical use."
)


@dataclass(frozen=True)
class ValidationResult:
    """Local mirror of Contract 3's ``InputValidationResult``.

    Copied rather than imported so the rest of the app has one stable type
    whether B's module is present or not.
    """

    is_in_scope: bool
    reason: str
    score: float
    method: str


class _ValidateImage(Protocol):
    def __call__(self, image_path_or_array: Any) -> Any: ...


def _coerce(raw: Any) -> ValidationResult:
    """Convert B's result object into ours, checking the contract as we go."""
    missing = [
        field
        for field in ("is_in_scope", "reason", "score", "method")
        if not hasattr(raw, field)
    ]
    if missing:
        raise AttributeError(
            f"src/input_validation.validate_image() returned an object missing "
            f"{', '.join(missing)}. Contract 3 requires is_in_scope, reason, "
            f"score and method."
        )
    return ValidationResult(
        is_in_scope=bool(raw.is_in_scope),
        reason=str(raw.reason),
        score=float(raw.score),
        method=str(raw.method),
    )


class InputValidator:
    """Wraps session B's validator, or stands in for it.

    Attributes:
        is_stub: True when B's module was not importable.
        detail: Why, in words. Shown at startup and in the readiness check.
    """

    def __init__(self, force_stub: bool = False) -> None:
        self._validate: _ValidateImage | None = None
        self.is_stub = True
        self.detail = ""

        if force_stub:
            self.detail = "stub forced by caller"
            return

        try:
            module = importlib.import_module("src.input_validation")
        except ImportError as exc:
            self.detail = f"src/input_validation.py not importable ({exc})"
            return

        candidate = getattr(module, "validate_image", None)
        if not callable(candidate):
            self.detail = "src/input_validation.py has no callable validate_image()"
            return

        self._validate = candidate
        self.is_stub = False
        self.detail = "session B's validate_image() is loaded"

    def validate(self, image_path_or_array: Any) -> ValidationResult:
        """Run the check. Never raises through to the caller.

        A crash inside the rejector must not become a silent pass. If B's code
        fails on an input, that input is treated as out of scope, because an
        input that breaks the checker is exactly the kind we should not judge.
        """
        if self._validate is None:
            return ValidationResult(
                is_in_scope=True,
                reason=STUB_REASON,
                score=0.0,
                method=STUB_METHOD,
            )

        try:
            return _coerce(self._validate(image_path_or_array))
        except Exception as exc:  # noqa: BLE001 - any failure means do not judge
            return ValidationResult(
                is_in_scope=False,
                reason=(
                    "The image check could not complete on this file, so the tool "
                    "will not judge it. Send it to a human reader."
                ),
                score=float("nan"),
                method=f"error:{type(exc).__name__}",
            )


_validator: InputValidator | None = None


def get_validator(force_stub: bool = False) -> InputValidator:
    """Process-wide validator. Loaded once, reused."""
    global _validator
    if _validator is None or force_stub:
        _validator = InputValidator(force_stub=force_stub)
    return _validator


def reset_validator() -> None:
    """Test hook, and the way the app picks up B's module after a rebase."""
    global _validator
    _validator = None
