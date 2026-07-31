"""The "NOT FOR CLINICAL USE" banner must not vanish by accident.

Why this file exists
--------------------
`app/static/app.js` hides the development banner on exactly one condition:

    if (status.state !== "clinical") { ...show banner... }

and `Readiness.state` returned "clinical" as soon as there were no blockers and
no warnings. Every readiness check at the time asked a wiring question: is the
config real, is the validator installed, is the explainer installed, do the
checkpoint files exist. All of them started passing the moment sessions A, B
and D published their files.

So finishing those sessions removed "DEVELOPMENT BUILD - NOT FOR CLINICAL USE"
from a clinical screen, without anyone writing a line about validation, and
while the project still had no external validation result and no clinician had
ever reviewed an output.

The guard added to `readiness.check` keeps a warning present until a config
positively attests to validation on an independent cohort. These tests pin that
behaviour down, in both directions, so the banner cannot disappear again
because a wiring check started passing.
"""
from __future__ import annotations

import dataclasses

import pytest

from app.core import readiness
from app.core.explain_adapter import Explainer
from app.core.validation_adapter import InputValidator


def _check(cfg):
    return readiness.check(cfg, InputValidator(), Explainer(), dev_mode=False)


def _with_raw(cfg, raw: dict):
    return dataclasses.replace(cfg, raw=raw)


def test_shipped_config_does_not_claim_clinical_state(stub_config):
    """The real deployment config must not reach state 'clinical'."""
    from app.core.config import load_config
    from app.tests.conftest import REPO_ROOT

    real = REPO_ROOT / "analysis/results/safety/deployment_config.json"
    if not real.is_file():
        pytest.skip("session A has not published deployment_config.json")

    r = _check(load_config(real))
    assert r.state != "clinical", (
        "The shipped config reached state 'clinical', which hides the "
        "NOT FOR CLINICAL USE banner in app.js. This project has no external "
        "validation result."
    )
    assert any(f.code == "no_external_validation" for f in r.warnings)


def test_missing_attestation_keeps_the_warning(test_config):
    r = _check(_with_raw(test_config, {}))
    assert any(f.code == "no_external_validation" for f in r.warnings)
    assert r.state != "clinical"


@pytest.mark.parametrize("attest", [
    None,
    "independent_cohort",                       # a bare string, not a mapping
    {},                                          # mapping with no status
    {"status": ""},
    {"status": "pending"},
    {"status": "brisc"},
    {"status": "INDEPENDENT_COHORT_SOON"},
    {"status": ["independent_cohort"]},          # right word, wrong type
])
def test_junk_attestation_never_clears_the_banner(test_config, attest):
    """Fail-safe. Anything that is not the exact attestation keeps the warning."""
    r = _check(_with_raw(test_config, {"external_validation": attest}))
    assert any(f.code == "no_external_validation" for f in r.warnings), (
        f"attestation {attest!r} cleared the banner. The check must be "
        f"fail-safe: only an explicit, exact attestation clears it."
    )


def test_a_real_attestation_does_clear_it(test_config):
    """The guard must be clearable, or it is just a hard-coded string.

    This is the only shape that clears it. Nothing in this repo writes it, and
    nothing should until a genuinely independent cohort has been run.
    """
    r = _check(_with_raw(test_config, {
        "external_validation": {
            "status": "independent_cohort",
            "detail": "Validated on a real external cohort.",
        }
    }))
    assert not any(f.code == "no_external_validation" for f in r.warnings)


def test_the_warning_does_not_stop_the_app_starting(test_config):
    """Warning, not blocker. The tool stays usable for development.

    `test_config` is built from the day-one stub, so in clinical mode the stub
    itself is a blocker, which is correct and is a different check. Development
    mode demotes the stub findings to warnings, which isolates the question this
    test is asking: does the missing-validation guard, on its own, refuse
    startup. It must not. Refusing outright would stop all development use to
    make a point the banner already makes.
    """
    r = readiness.check(_with_raw(test_config, {}), InputValidator(), Explainer(),
                        dev_mode=True)
    assert any(f.code == "no_external_validation" for f in r.warnings)
    assert not any(f.code == "no_external_validation" for f in r.blockers)
    assert r.ok, "the missing-validation warning must not refuse startup"
    assert r.state == "development"


def test_custom_detail_is_shown_instead_of_the_default(test_config):
    r = _check(_with_raw(test_config, {
        "external_validation": {"status": "none", "detail": "Cohort pending ethics approval."}
    }))
    msg = next(f.message for f in r.warnings if f.code == "no_external_validation")
    assert msg == "Cohort pending ethics approval."
