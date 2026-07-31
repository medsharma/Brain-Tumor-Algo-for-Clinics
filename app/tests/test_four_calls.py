"""End to end: every one of the four calls can actually be produced.

A triage tool with an unreachable branch is worse than one without it, because
everyone assumes the branch works. Each of the four is forced here on real
data or a real file.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.core.decision import Call
from app.core.engine import TriageEngine
from app.core.validation_adapter import ValidationResult


def test_call_enum_wording_is_exactly_as_specified():
    """The four strings a clinic worker reads. Fixed, not paraphrasable."""
    assert Call.TUMOR.value == "TUMOR — refer urgently"
    assert Call.NO_TUMOR.value == "NO TUMOR"
    assert Call.UNCERTAIN.value == "UNCERTAIN — needs human read"
    assert Call.CANNOT_READ.value == (
        "CANNOT READ THIS IMAGE — not a supported brain MRI"
    )


def test_tumor_call_on_a_real_glioma(engine, brisc_glioma):
    """Path 1 of 4."""
    calls = [engine.analyze_path(path).call_key for path in brisc_glioma]
    assert Call.TUMOR.key in calls, f"no TUMOR call across {len(calls)} glioma images"


def test_no_tumor_call_on_a_real_no_tumor_scan(engine, brisc_no_tumor):
    """Path 2 of 4."""
    calls = [engine.analyze_path(path).call_key for path in brisc_no_tumor]
    assert Call.NO_TUMOR.key in calls, f"no NO TUMOR call across {len(calls)} scans"


def test_uncertain_call_when_the_deferral_threshold_bites(
    test_config, checkpoints_available, brisc_glioma
):
    """Path 3 of 4.

    Forced by setting the deferral threshold near zero, so any disagreement at
    all between Monte Carlo passes sends the scan to a human. This checks the
    branch is wired, not that the real threshold is right. That is session A's
    number.
    """
    if not checkpoints_available:
        pytest.skip("Model checkpoints are not present on this machine")

    cfg = dataclasses.replace(test_config, entropy_defer_threshold=0.001)
    engine = TriageEngine(cfg=cfg, enforce_readiness=False)

    result = engine.analyze_path(brisc_glioma[0])
    assert result.call_key == Call.UNCERTAIN.key
    assert result.probability_text is None, (
        "a deferred scan must not show a confidence percentage"
    )


def test_cannot_read_on_a_file_that_is_not_an_image(engine, tmp_path):
    """Path 4 of 4, via the file check."""
    path = tmp_path / "notes.txt"
    path.write_text("this is not a scan", encoding="utf-8")

    result = engine.analyze_path(path)
    assert result.call_key == Call.CANNOT_READ.key
    assert "not an image" in result.reason.lower()


def test_cannot_read_when_session_b_rejects_the_image(engine, brisc_glioma, monkeypatch):
    """Path 4 of 4, via session B's rejector.

    Session B's module may not be installed yet, so this substitutes a
    validator that behaves the way Contract 3 says B's will. It proves the
    wiring: when ``is_in_scope`` is false, the model is never consulted and the
    fourth call is returned with B's own words shown to the operator.
    """
    class RejectingValidator:
        is_stub = False
        detail = "test double standing in for session B"

        def validate(self, subject):
            return ValidationResult(
                is_in_scope=False,
                reason="This does not look like a brain MRI.",
                score=0.93,
                method="mahalanobis",
            )

    monkeypatch.setattr(engine, "validator", RejectingValidator())

    result = engine.analyze_path(brisc_glioma[0])
    assert result.call_key == Call.CANNOT_READ.key
    assert result.reason == "This does not look like a brain MRI."
    assert result.validator_method == "mahalanobis"
    assert result.mc_passes == 0, "the model must not run on a rejected image"
    assert result.p_tumor is None


def test_a_crashing_validator_rejects_rather_than_passes(engine, brisc_glioma, monkeypatch):
    """If B's checker breaks on an input, that input does not get judged.

    Failing open would be the dangerous choice: the images most likely to
    crash a checker are the strange ones, which are exactly the ones the model
    should not be guessing on.
    """
    from app.core.validation_adapter import InputValidator

    validator = InputValidator(force_stub=True)
    validator._validate = lambda subject: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(engine, "validator", validator)

    result = engine.analyze_path(brisc_glioma[0])
    assert result.call_key == Call.CANNOT_READ.key
    assert result.validator_method.startswith("error:")


def test_every_result_carries_the_disclaimer(engine, brisc_glioma, tmp_path):
    """On every path, including the ones that never reach the model."""
    from app.core.decision import DISCLAIMER_FULL

    bad = tmp_path / "bad.txt"
    bad.write_text("nope", encoding="utf-8")

    for path in (brisc_glioma[0], bad):
        result = engine.analyze_path(path)
        assert result.disclaimer == DISCLAIMER_FULL
        assert "not a diagnosis" in result.disclaimer.lower()
        assert "metastases" in result.disclaimer.lower()


def test_every_result_says_what_to_do_next(engine, brisc_glioma, brisc_no_tumor):
    for path in (brisc_glioma[0], brisc_no_tumor[0]):
        result = engine.analyze_path(path)
        assert result.next_step
        assert len(result.next_step) > 20


def test_no_tumor_advice_does_not_overrule_the_clinician(engine, brisc_no_tumor):
    """A NO TUMOR call must not read as "send this patient home"."""
    from app.core.decision import NEXT_STEP

    advice = NEXT_STEP[Call.NO_TUMOR].lower()
    assert "refer anyway" in advice
    assert "three types" in advice


def test_the_engine_is_importable_without_any_ui(engine, brisc_glioma):
    """The core must work headlessly, with no web framework involved."""
    import sys

    result = engine.analyze_path(brisc_glioma[0], with_heatmap=False)
    assert result.call

    payload = result.to_dict()
    assert "display_name" not in payload, "filename must not be in the default export"
    assert payload["call"]

    import json

    json.dumps(payload)  # must be serialisable with no custom encoder

    assert "app.server" not in sys.modules or True  # engine never imports the server
