"""Startup refusal, stub adapters, and how a folder of slices is handled."""

from __future__ import annotations

import dataclasses
import shutil

import pytest

from app.core import readiness
from app.core.engine import TriageEngine
from app.core.explain_adapter import Explainer
from app.core.readiness import NotReadyError
from app.core.validation_adapter import InputValidator


# ------------------------------------------------------------------ startup

def test_the_engine_refuses_to_start_on_a_stub_outside_dev_mode(
    stub_config, checkpoints_available, monkeypatch
):
    if not checkpoints_available:
        pytest.skip("Model checkpoints are not present on this machine")

    monkeypatch.delenv("MRI_CLINIC_DEV_MODE", raising=False)
    with pytest.raises(NotReadyError) as excinfo:
        TriageEngine(cfg=stub_config)

    message = str(excinfo.value)
    assert "REFUSING TO START" in message
    assert "stub_config" in message


def test_a_stub_input_check_blocks_startup(stub_config, monkeypatch):
    """A validator that rejects nothing means the fourth call can never fire."""
    monkeypatch.delenv("MRI_CLINIC_DEV_MODE", raising=False)
    cfg = dataclasses.replace(stub_config, is_stub=False)

    result = readiness.check(
        cfg, InputValidator(force_stub=True), Explainer(force_stub=False), dev_mode=False
    )
    assert "stub_validator" in {finding.code for finding in result.blockers}


def test_a_stub_heatmap_blocks_startup(stub_config, monkeypatch):
    """A convincing fake heatmap destroys the trust it exists to build."""
    monkeypatch.delenv("MRI_CLINIC_DEV_MODE", raising=False)
    cfg = dataclasses.replace(stub_config, is_stub=False)

    result = readiness.check(
        cfg, InputValidator(force_stub=False), Explainer(force_stub=True), dev_mode=False
    )
    assert "stub_explainer" in {finding.code for finding in result.blockers}


def test_dev_mode_turns_blockers_into_warnings(stub_config):
    result = readiness.check(
        stub_config, InputValidator(force_stub=True), Explainer(force_stub=True), dev_mode=True
    )
    assert result.ok
    assert result.state == "development"
    assert len(result.warnings) >= 3


def test_preprocessing_drift_blocks_startup_even_in_dev_mode(stub_config):
    """This one is never a warning. Drift makes every number meaningless."""
    cfg = dataclasses.replace(stub_config, resize=(256, 256))
    result = readiness.check(
        cfg, InputValidator(force_stub=True), Explainer(force_stub=True), dev_mode=True
    )
    assert "preprocessing_drift" in {finding.code for finding in result.blockers}
    assert not result.ok


def test_a_missing_checkpoint_blocks_startup(stub_config):
    from app.core.config import Checkpoint

    cfg = dataclasses.replace(
        stub_config,
        checkpoints=(Checkpoint("resnet50", 42, "C:/nowhere/missing.pth", "0" * 64),),
    )
    result = readiness.check(
        cfg, InputValidator(force_stub=True), Explainer(force_stub=True), dev_mode=True
    )
    assert "missing_checkpoint" in {finding.code for finding in result.blockers}


def test_a_high_miss_rate_raises_a_warning(stub_config):
    cfg = dataclasses.replace(
        stub_config,
        is_stub=False,
        expected_performance={**stub_config.expected_performance, "tumor_miss_rate": 0.12},
    )
    result = readiness.check(
        cfg, InputValidator(force_stub=False), Explainer(force_stub=False), dev_mode=False
    )
    assert "high_miss_rate" in {finding.code for finding in result.warnings}


def test_stub_results_carry_a_visible_warning(stub_config, checkpoints_available, brisc_glioma):
    """A development result must announce itself, not look like a real one."""
    if not checkpoints_available:
        pytest.skip("Model checkpoints are not present on this machine")

    engine = TriageEngine(cfg=stub_config, enforce_readiness=False)
    result = engine.analyze_path(brisc_glioma[0])

    joined = " ".join(result.notes)
    assert "DEVELOPMENT BUILD" in joined
    assert "fake" in joined.lower()


# ------------------------------------------------------------------- adapters

def test_the_stub_heatmap_is_obviously_not_saliency():
    """Stripes, not a blob. Nobody should mistake it for a real overlay."""
    import numpy as np

    explainer = Explainer(force_stub=True)
    heatmap = explainer.generate_heatmap(None, np.zeros((1, 3, 224, 224)), "resnet50")

    assert heatmap.shape == (224, 224)
    assert set(np.unique(heatmap)) <= {0.0, 1.0}, "a real heatmap is not two-valued"


def test_a_heatmap_of_the_wrong_shape_is_rejected():
    """If session D returns a transposed or mis-sized map, fail loudly.

    A heatmap that quietly points at the wrong part of the brain is worse than
    no heatmap.
    """
    import numpy as np

    with pytest.raises(ValueError, match="does not match"):
        Explainer._check_heatmap(np.zeros((100, 100), dtype=np.float32), np.zeros((1, 3, 224, 224)))

    with pytest.raises(ValueError, match="HxW"):
        Explainer._check_heatmap(np.zeros((224, 224, 3), dtype=np.float32), np.zeros((1, 3, 224, 224)))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Explainer._check_heatmap(np.full((224, 224), 5.0, dtype=np.float32), np.zeros((1, 3, 224, 224)))

    with pytest.raises(ValueError, match="NaN"):
        Explainer._check_heatmap(np.full((224, 224), np.nan, dtype=np.float32), np.zeros((1, 3, 224, 224)))


def test_the_stub_validator_passes_everything_and_says_so():
    validator = InputValidator(force_stub=True)
    result = validator.validate("anything at all")
    assert result.is_in_scope is True
    assert "STUB" in result.reason
    assert result.method == "STUB_always_in_scope"


def test_a_failing_heatmap_does_not_break_the_call(engine, brisc_glioma, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("saliency blew up")

    monkeypatch.setattr(engine.explainer, "generate_heatmap", explode)
    result = engine.analyze_path(brisc_glioma[0], with_heatmap=True)

    assert result.call_key in {"tumor", "no_tumor", "uncertain"}
    assert result.overlay_rgb is None
    assert any("heatmap could not be produced" in note for note in result.notes)


# --------------------------------------------------------------------- series

def test_a_folder_gives_per_slice_results_and_no_study_call(engine, tmp_path, brisc_glioma):
    """A study-level answer needs its own threshold, which nobody has fitted."""
    for index, source in enumerate(brisc_glioma):
        shutil.copy(source, tmp_path / f"slice_{index:03d}.jpg")

    series = engine.analyze_folder(tmp_path)

    assert series.n_files_seen == len(brisc_glioma)
    assert len(series.slices) == len(brisc_glioma)
    assert series.study_level_call is None
    assert "threshold" in series.study_level_note
    assert series.disclaimer


def test_a_study_threshold_in_the_config_enables_a_study_call(
    test_config, checkpoints_available, tmp_path, brisc_glioma
):
    """If session A ever fits one, the app uses it. Until then, nothing."""
    if not checkpoints_available:
        pytest.skip("Model checkpoints are not present on this machine")

    raw = dict(test_config.raw)
    raw["series_tumor_threshold"] = 0.5
    cfg = dataclasses.replace(test_config, raw=raw)
    assert cfg.series_tumor_threshold == 0.5

    for index, source in enumerate(brisc_glioma):
        shutil.copy(source, tmp_path / f"slice_{index:03d}.jpg")

    engine = TriageEngine(cfg=cfg, enforce_readiness=False)
    series = engine.analyze_folder(tmp_path)

    assert series.study_level_call in {"TUMOR — refer urgently", "NO TUMOR"}


def test_the_stub_config_has_no_study_threshold(stub_config):
    assert stub_config.series_tumor_threshold is None
