"""Session B's real input check, once it is installed.

Every test here skips while ``src/input_validation.py`` is absent and starts
checking the real thing the moment B publishes. That way the exit criterion
"session B's input rejection wired in and firing on a non-brain image"
verifies itself rather than needing someone to remember to check.

``app/tests/test_four_calls.py`` already proves the *wiring* using a test
double that behaves the way Contract 3 says B's validator will. What is left,
and what is here, is proving it fires on genuinely out-of-scope input.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.core.validation_adapter import get_validator, reset_validator


@pytest.fixture(scope="module")
def real_validator():
    reset_validator()
    validator = get_validator()
    if validator.is_stub:
        pytest.skip(f"session B's validator is not installed: {validator.detail}")
    return validator


def _save(array: np.ndarray, path) -> str:
    Image.fromarray(array).save(path)
    return str(path)


@pytest.fixture
def colour_noise(tmp_path):
    """Unambiguously not a brain MRI: saturated random colour."""
    rng = np.random.default_rng(11)
    return _save((rng.random((512, 512, 3)) * 255).astype("uint8"), tmp_path / "noise.png")


@pytest.fixture
def flat_grey(tmp_path):
    """A blank frame. No anatomy at all."""
    return _save(np.full((512, 512, 3), 128, dtype="uint8"), tmp_path / "blank.png")


@pytest.fixture
def synthetic_shapes(tmp_path):
    """Geometric shapes on a coloured field. Structured, but not anatomy."""
    canvas = np.zeros((512, 512, 3), dtype="uint8")
    canvas[:, :, 1] = 200
    canvas[100:250, 100:250] = (255, 0, 0)
    canvas[300:450, 260:460] = (0, 0, 255)
    return _save(canvas, tmp_path / "shapes.png")


def test_the_adapter_finds_sessions_b_module(real_validator):
    assert real_validator.is_stub is False
    assert "session B" in real_validator.detail


def test_the_result_satisfies_contract_3(real_validator, brisc_glioma):
    result = real_validator.validate(str(brisc_glioma[0]))
    assert isinstance(result.is_in_scope, bool)
    assert isinstance(result.reason, str) and result.reason
    assert isinstance(result.score, float)
    assert isinstance(result.method, str) and result.method


@pytest.mark.parametrize(
    "fixture_name", ["colour_noise", "flat_grey", "synthetic_shapes"]
)
def test_out_of_scope_images_are_rejected(real_validator, request, fixture_name):
    path = request.getfixturevalue(fixture_name)
    result = real_validator.validate(path)
    assert result.is_in_scope is False, (
        f"{fixture_name} was accepted as a brain MRI. The out-of-scope check "
        f"is not firing, so a non-brain image would get a tumour answer."
    )


def test_a_real_brain_mri_is_not_rejected(real_validator, brisc_mixed):
    """The check must not reject the scans it exists to protect.

    A rejector that refuses everything is trivially safe and completely
    useless, so this is as important as the rejection tests.
    """
    rejected = [
        path.name
        for path in brisc_mixed
        if not real_validator.validate(str(path)).is_in_scope
    ]
    assert not rejected, (
        f"{len(rejected)} of {len(brisc_mixed)} real brain MRI scans were "
        f"rejected: {rejected}"
    )


@pytest.mark.parametrize(
    "fixture_name", ["colour_noise", "flat_grey", "synthetic_shapes"]
)
def test_the_fourth_call_reaches_the_operator(engine, request, fixture_name, monkeypatch):
    """End to end: an out-of-scope image produces CANNOT READ, and the model
    is never consulted."""
    reset_validator()
    validator = get_validator()
    if validator.is_stub:
        pytest.skip("session B's validator is not installed")
    monkeypatch.setattr(engine, "validator", validator)

    result = engine.analyze_path(request.getfixturevalue(fixture_name))

    assert result.call_key == "cannot_read"
    assert result.mc_passes == 0
    assert result.p_tumor is None
    assert result.reason, "the operator must be told why"


def test_the_rejection_reason_is_written_for_a_clinic_worker(
    real_validator, colour_noise
):
    """B's reason string goes straight onto the screen. It has to read."""
    result = real_validator.validate(colour_noise)
    assert result.is_in_scope is False
    assert len(result.reason) > 15
    assert result.reason[0].isupper() or result.reason.startswith('"')


def test_the_validator_makes_no_network_call(monkeypatch, real_validator, brisc_glioma):
    import socket
    import urllib.request

    def blocked(*args, **kwargs):
        raise AssertionError("the input check attempted a network connection")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)

    real_validator.validate(str(brisc_glioma[0]))


def test_the_validator_accepts_an_array_as_contract_3_allows(real_validator, brisc_glioma):
    """The web upload path has no file on disk, so it passes an array.

    Writing patient images to a temp file just to produce a path would be the
    wrong trade, so this route has to work.
    """
    with Image.open(brisc_glioma[0]) as image:
        array = np.asarray(image.convert("RGB"))

    result = real_validator.validate(array)
    assert isinstance(result.is_in_scope, bool)
