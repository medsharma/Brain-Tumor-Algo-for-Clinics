"""The app must refuse to give clinical answers on invented numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core import config as config_module
from app.core.config import (
    ConfigError,
    StubConfigError,
    detect_stub,
    load_config,
    resolve_entropy_units,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STUB_PATH = REPO_ROOT / "analysis/results/safety/deployment_config.SCHEMA.json"


def test_stub_file_exists():
    assert STUB_PATH.is_file(), "the day-one stub must be checked in"


def test_stub_values_are_obviously_fake():
    """A stub must never be mistakable for real config in a screenshot."""
    raw = json.loads(STUB_PATH.read_text(encoding="utf-8"))
    assert raw["tumor_threshold"] == 0.123456
    assert "STUB" in raw["schema_version"]
    assert raw["_THIS_IS_A_STUB"] is True
    assert raw["expected_performance"]["dataset"] == "NOT_MEASURED_STUB"


def test_loading_the_stub_is_refused_by_default():
    with pytest.raises(StubConfigError) as excinfo:
        load_config(STUB_PATH, allow_stub=False)
    message = str(excinfo.value)
    assert "REFUSING TO START" in message
    assert "worse than no tool" in message


def test_dev_mode_env_var_permits_the_stub(monkeypatch):
    monkeypatch.setenv(config_module.DEV_MODE_ENV, "1")
    cfg = load_config(STUB_PATH)
    assert cfg.is_stub is True


def test_dev_mode_must_be_exactly_one(monkeypatch):
    """A truthy-looking value is not enough. Only "1" opens the gate."""
    for value in ("true", "yes", "0", "", "TRUE", "on"):
        monkeypatch.setenv(config_module.DEV_MODE_ENV, value)
        with pytest.raises(StubConfigError):
            load_config(STUB_PATH)


@pytest.mark.parametrize(
    "mutation,description",
    [
        ({"_THIS_IS_A_STUB": False}, "marker removed"),
        ({"schema_version": "1.0"}, "version made to look real"),
        ({"thresholds_fitted_on": "internal_val"}, "fitted_on made to look real"),
    ],
)
def test_stub_detection_survives_single_edits(tmp_path, mutation, description):
    """Three independent signals, so blanking one is not enough.

    The filename itself is the third signal, and it still catches these.
    """
    raw = json.loads(STUB_PATH.read_text(encoding="utf-8"))
    raw.update(mutation)
    target = tmp_path / "deployment_config.SCHEMA.json"
    target.write_text(json.dumps(raw), encoding="utf-8")

    is_stub, reason = detect_stub(raw, target)
    assert is_stub, f"stub not detected after {description}"
    with pytest.raises(StubConfigError):
        load_config(target, allow_stub=False)


def test_a_real_looking_config_loads(tmp_path):
    """A config with no stub markers and a supported schema version loads."""
    raw = json.loads(STUB_PATH.read_text(encoding="utf-8"))
    raw.pop("_THIS_IS_A_STUB")
    raw["schema_version"] = "1.0"
    raw["thresholds_fitted_on"] = "internal_val"
    raw["temperature"] = 1.0
    raw["tumor_threshold"] = 0.5
    raw["entropy_defer_threshold"] = 0.8
    target = tmp_path / "deployment_config.json"
    target.write_text(json.dumps(raw), encoding="utf-8")

    cfg = load_config(target, allow_stub=False)
    assert cfg.is_stub is False
    assert cfg.tumor_threshold == 0.5


def _real_config(tmp_path, **overrides) -> Path:
    raw = json.loads(STUB_PATH.read_text(encoding="utf-8"))
    raw.pop("_THIS_IS_A_STUB")
    raw["schema_version"] = "1.0"
    raw["thresholds_fitted_on"] = "internal_val"
    raw["temperature"] = 1.0
    raw["tumor_threshold"] = 0.5
    raw["entropy_defer_threshold"] = 0.8
    raw.update(overrides)
    target = tmp_path / "deployment_config.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    return target


@pytest.mark.parametrize("missing", list(config_module.REQUIRED_KEYS))
def test_every_required_key_is_enforced(tmp_path, missing):
    """Fail loudly rather than defaulting, as Contract 2 requires."""
    import json as _json

    raw = _json.loads(_real_config(tmp_path).read_text(encoding="utf-8"))
    raw.pop(missing)
    target = tmp_path / "deployment_config.json"
    target.write_text(_json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="missing required keys"):
        load_config(target, allow_stub=False)


def test_reordered_class_names_are_refused(tmp_path):
    """Reordering class_names silently relabels every prediction."""
    target = _real_config(
        tmp_path, class_names=["notumor", "glioma", "meningioma", "pituitary"]
    )
    with pytest.raises(ConfigError, match="class_names must be exactly"):
        load_config(target, allow_stub=False)


def test_thresholds_fitted_on_brisc_is_refused(tmp_path, stub_config):
    """Fitting on BRISC and reporting on BRISC is circular. Blocked at startup."""
    from app.core import readiness
    from app.core.explain_adapter import Explainer
    from app.core.validation_adapter import InputValidator
    import dataclasses

    cfg = dataclasses.replace(stub_config, is_stub=False, thresholds_fitted_on="brisc_test")
    result = readiness.check(
        cfg, InputValidator(force_stub=True), Explainer(force_stub=True), dev_mode=True
    )
    codes = {finding.code for finding in result.blockers}
    assert "contaminated_thresholds" in codes
    assert not result.ok


def test_unsupported_schema_version_is_refused(tmp_path):
    target = _real_config(tmp_path, schema_version="2.0")
    with pytest.raises(ConfigError, match="not supported by this build"):
        load_config(target, allow_stub=False)


def test_ensemble_false_with_many_seeds_is_refused(tmp_path):
    target = _real_config(tmp_path, ensemble=False, chosen_seeds=[42, 123])
    with pytest.raises(ConfigError, match="ensemble is false"):
        load_config(target, allow_stub=False)


def test_seed_and_checkpoint_counts_must_agree(tmp_path):
    target = _real_config(tmp_path, ensemble=True, chosen_seeds=[42, 123])
    with pytest.raises(ConfigError, match="Every chosen seed needs"):
        load_config(target, allow_stub=False)


def test_entropy_threshold_above_the_maximum_is_refused(tmp_path):
    """A threshold nothing could ever exceed means deferral never fires."""
    target = _real_config(tmp_path, entropy_defer_threshold=2.5)
    with pytest.raises(ConfigError, match="exceeds the maximum possible"):
        load_config(target, allow_stub=False)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="No deployment config"):
        load_config(tmp_path / "nope.json")


# ------------------------------------------------------------------ entropy

def test_entropy_units_declared_wins():
    units, source = resolve_entropy_units(
        {"entropy_units": "nats", "entropy_defer_threshold": 0.8}
    )
    assert units == "nats"
    assert source == "declared in config"


def test_entropy_units_inferred_when_above_nat_maximum():
    units, source = resolve_entropy_units({"entropy_defer_threshold": 1.9})
    assert units == "bits"
    assert "inferred" in source


def test_entropy_units_default_to_bits_and_say_so():
    """Ambiguous values default to bits, which is the safer of the two.

    Bits are the larger number for the same scan, so comparing a bits-valued
    entropy against the threshold defers more images to a human than nats
    would. When nobody has said which unit was meant, deferring more is the
    error worth making.
    """
    units, source = resolve_entropy_units({"entropy_defer_threshold": 0.8})
    assert units == "bits"
    assert source.startswith("ASSUMED")
