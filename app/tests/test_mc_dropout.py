"""Monte Carlo Dropout: the speed trick must not change the answer.

Running T passes over the whole network recomputes an identical trunk T times,
because the only dropout layers in either backbone are in the classification
head. The app computes the trunk once and runs the head T times.

That is an exact identity rather than an approximation, but "exact by
argument" is not good enough for something that moves a clinical threshold.
These tests check it numerically on the real checkpoints.
"""

from __future__ import annotations

import pytest
import torch

from app.core import model as model_module


@pytest.fixture(scope="module")
def resnet_module(stub_config):
    from pathlib import Path

    checkpoint = stub_config.checkpoints[0]
    if not Path(checkpoint.path).is_file():
        pytest.skip("Model checkpoints are not present on this machine")
    loaded = model_module.load_checkpoint(
        checkpoint.model, checkpoint.seed, checkpoint.path, checkpoint.sha256
    )
    return loaded.module


def test_the_trunk_is_deterministic_with_dropout_active(resnet_module):
    """If the trunk moved between calls, caching it would be wrong outright."""
    x = torch.randn(1, 3, 224, 224)
    resnet_module.eval()
    model_module.activate_dropout(resnet_module)

    first = resnet_module.trunk(x)
    second = resnet_module.trunk(x)
    assert torch.equal(first, second)


def test_the_head_is_stochastic_with_dropout_active(resnet_module):
    """The whole point of MC Dropout. If the head were deterministic, entropy
    would always be zero and deferral would never fire."""
    x = torch.randn(1, 3, 224, 224)
    resnet_module.eval()
    model_module.activate_dropout(resnet_module)

    features = resnet_module.trunk(x)
    first = resnet_module.head(features)
    second = resnet_module.head(features)
    assert not torch.equal(first, second)


def test_the_fast_path_equals_the_reference_path_exactly(resnet_module):
    x = torch.randn(1, 3, 224, 224)
    passed, delta, note = model_module.verify_mc_equivalence(resnet_module, x, T=20)
    assert passed, note
    assert delta == 0.0, f"expected bit-identical output, got {delta:.3e}"


def test_the_fast_path_holds_on_a_real_scan(resnet_module, brisc_glioma, stub_config):
    from PIL import Image

    from app.core import preprocess

    image = Image.open(brisc_glioma[0]).convert("RGB")
    x = preprocess.preprocess_pil(image, preprocess.build_transform())

    passed, delta, note = model_module.verify_mc_equivalence(resnet_module, x, T=20)
    assert passed, note


def test_batchnorm_never_enters_training_mode(resnet_module):
    """``model.train()`` would let running statistics drift during inference.

    On BRISC that would be a form of contamination: the model would be
    adapting to the exam data. Only dropout layers may be switched.
    """
    resnet_module.eval()
    model_module.activate_dropout(resnet_module)

    for module in resnet_module.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert not module.training, "a BatchNorm layer is in training mode"
        if isinstance(module, torch.nn.Dropout):
            assert module.training


def test_inference_runs_without_gradients(resnet_module):
    x = torch.randn(1, 3, 224, 224)
    probs = model_module.mc_forward_cached(resnet_module, x, T=3)
    assert not probs.requires_grad


def test_probabilities_sum_to_one(resnet_module):
    x = torch.randn(1, 3, 224, 224)
    result = model_module.run_mc_dropout(
        [resnet_module], x, T=10, temperature=1.0, entropy_units="bits"
    )
    assert sum(result.mean_probs) == pytest.approx(1.0, abs=1e-5)
    assert result.n_passes == 10


def test_entropy_units_differ_by_the_expected_factor(resnet_module):
    import math

    x = torch.randn(1, 3, 224, 224)
    result = model_module.run_mc_dropout(
        [resnet_module], x, T=10, temperature=1.0, entropy_units="bits"
    )
    assert result.entropy_bits == pytest.approx(result.entropy_nats / math.log(2), rel=1e-5)
    assert result.entropy == result.entropy_bits


def test_a_uniform_distribution_hits_the_maximum_entropy():
    uniform = torch.full((1, 4), 0.25)
    assert model_module.entropy_from_probs(uniform, "bits").item() == pytest.approx(2.0, abs=1e-4)
    import math

    assert model_module.entropy_from_probs(uniform, "nats").item() == pytest.approx(
        math.log(4), abs=1e-4
    )


def test_temperature_flattens_the_distribution(resnet_module, brisc_glioma):
    """A high temperature must move probabilities towards uniform.

    This is what makes the stub config's absurd temperature of 9.876543
    produce visibly nonsensical output rather than something plausible.
    """
    from PIL import Image

    from app.core import preprocess

    image = Image.open(brisc_glioma[0]).convert("RGB")
    x = preprocess.preprocess_pil(image, preprocess.build_transform())

    sharp = model_module.run_mc_dropout(
        [resnet_module], x, T=20, temperature=1.0, entropy_units="bits"
    )
    flat = model_module.run_mc_dropout(
        [resnet_module], x, T=20, temperature=9.876543, entropy_units="bits"
    )
    assert flat.entropy > sharp.entropy


def test_p_tumor_is_the_sum_of_the_three_tumour_classes():
    result = model_module.MCResult(
        mean_probs=(0.2, 0.3, 0.1, 0.4),
        std_probs=(0.0, 0.0, 0.0, 0.0),
        entropy=1.0,
        entropy_units="bits",
        entropy_bits=1.0,
        entropy_nats=0.69,
        mutual_information=0.1,
        n_passes=20,
        n_models=1,
    )
    assert result.p_tumor == pytest.approx(0.6)
    assert result.pred_index == 3


def test_a_checkpoint_that_does_not_match_its_hash_is_refused(stub_config, tmp_path):
    """Every safety number was measured on one exact file. Refuse any other."""
    from pathlib import Path

    checkpoint = stub_config.checkpoints[0]
    if not Path(checkpoint.path).is_file():
        pytest.skip("Model checkpoints are not present on this machine")

    with pytest.raises(model_module.ModelLoadError, match="does not match"):
        model_module.load_checkpoint(
            checkpoint.model, checkpoint.seed, checkpoint.path, "a" * 64
        )


def test_a_missing_checkpoint_gives_a_readable_error():
    with pytest.raises(model_module.ModelLoadError, match="not found"):
        model_module.load_checkpoint("resnet50", 42, "C:/nowhere/missing.pth", None)


def test_loading_a_checkpoint_into_the_wrong_architecture_is_refused(stub_config):
    from pathlib import Path

    checkpoint = stub_config.checkpoints[0]
    if not Path(checkpoint.path).is_file():
        pytest.skip("Model checkpoints are not present on this machine")

    with pytest.raises(model_module.ModelLoadError, match="does not fit"):
        model_module.load_checkpoint("vit", 42, checkpoint.path, None)


def test_the_app_architecture_matches_the_training_architecture():
    """State dict keys and shapes must be identical to ``src/code.py``'s classes.

    Built with ``weights=None`` here and pretrained weights there, so only the
    structure is compared, which is what determines whether a checkpoint loads.
    """
    try:
        from src.code import BrainTumorResNet50 as TrainingResNet
        from src.code import BrainTumorViT as TrainingViT
    except ImportError:  # pragma: no cover
        pytest.skip("src/code.py is not importable here")

    for app_class, training_class in (
        (model_module.BrainTumorResNet50, TrainingResNet),
        (model_module.BrainTumorViT, TrainingViT),
    ):
        app_state = app_class().state_dict()
        training_state = training_class().state_dict()

        assert set(app_state) == set(training_state), (
            f"{app_class.__name__} has different parameter names than the "
            f"training class, so checkpoints would not load"
        )
        for key in app_state:
            assert app_state[key].shape == training_state[key].shape, (
                f"{key} has shape {tuple(app_state[key].shape)} in the app and "
                f"{tuple(training_state[key].shape)} in training"
            )
