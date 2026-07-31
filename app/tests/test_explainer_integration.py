"""Session D's real explainer, once it is installed.

Every test here skips when ``src/explain_runtime.py`` is absent, so the suite
stays green before D publishes and starts checking the real thing the moment
it lands. Contract 4 promises a specific shape and range; these check the
promise rather than trusting it, because a heatmap that quietly points at the
wrong part of the brain is worse than no heatmap at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.core import preprocess
from app.core.explain_adapter import Explainer, get_explainer, reset_explainer


@pytest.fixture(scope="module")
def real_explainer():
    reset_explainer()
    explainer = get_explainer()
    if explainer.is_stub:
        pytest.skip(f"session D's explainer is not installed: {explainer.detail}")
    return explainer


def test_the_adapter_finds_sessions_d_module(real_explainer):
    assert real_explainer.is_stub is False
    assert "session D" in real_explainer.detail


def test_the_published_caveat_is_used_verbatim(real_explainer):
    """D owns this wording and measured what backs it. Do not paraphrase it."""
    from src.explain_runtime import HEATMAP_CAVEAT

    assert real_explainer.caveat == HEATMAP_CAVEAT
    assert "not where the tumour is" in real_explainer.caveat


def test_the_caveat_reaches_every_result(engine, brisc_glioma):
    result = engine.analyze_path(brisc_glioma[0], with_heatmap=True)
    assert result.heatmap_caveat
    assert len(result.heatmap_caveat) > 20


def test_a_real_heatmap_meets_the_contract_shape_and_range(
    real_explainer, engine, brisc_glioma
):
    from PIL import Image

    with Image.open(brisc_glioma[0]) as image:
        tensor = preprocess.preprocess_pil(image, preprocess.build_transform())

    heatmap = real_explainer.generate_heatmap(
        engine.models[0].module, tensor, engine.models[0].kind, None
    )

    assert heatmap.ndim == 2
    assert heatmap.shape == (224, 224)
    assert np.isfinite(heatmap).all()
    assert 0.0 - 1e-6 <= float(heatmap.min())
    assert float(heatmap.max()) <= 1.0 + 1e-6


def test_a_real_heatmap_is_not_flat(real_explainer, engine, brisc_glioma):
    """A constant map would satisfy the contract and mean nothing."""
    from PIL import Image

    with Image.open(brisc_glioma[0]) as image:
        tensor = preprocess.preprocess_pil(image, preprocess.build_transform())

    heatmap = real_explainer.generate_heatmap(
        engine.models[0].module, tensor, engine.models[0].kind, None
    )
    assert float(heatmap.std()) > 0.01, "the heatmap carries no spatial information"
    assert len(np.unique(heatmap)) > 10


def test_the_overlay_is_a_displayable_image(real_explainer, engine, brisc_glioma):
    from PIL import Image

    with Image.open(brisc_glioma[0]) as image:
        tensor = preprocess.preprocess_pil(image, preprocess.build_transform())

    original = (
        (tensor[0].permute(1, 2, 0).numpy() * 0 + 0.5) * 255
    ).astype(np.uint8)

    heatmap = real_explainer.generate_heatmap(
        engine.models[0].module, tensor, engine.models[0].kind, None
    )
    overlay = real_explainer.overlay_heatmap(original, heatmap, alpha=0.4)

    assert overlay.dtype == np.uint8
    assert overlay.shape == (224, 224, 3)


def test_the_engine_produces_an_overlay_end_to_end(engine, brisc_glioma):
    result = engine.analyze_path(brisc_glioma[0], with_heatmap=True)

    assert result.original_rgb is not None
    assert result.overlay_rgb is not None
    assert result.overlay_rgb.shape == result.original_rgb.shape
    assert result.overlay_rgb.dtype == np.uint8
    assert not np.array_equal(result.overlay_rgb, result.original_rgb), (
        "the overlay is identical to the original, so nothing was drawn"
    )


def test_generating_a_heatmap_does_not_change_the_call(engine, brisc_glioma):
    """Explaining a prediction must not alter it.

    Grad-CAM needs gradients, which means turning autograd back on around a
    model the rest of the app runs under ``no_grad``. If that leaked state,
    the answer could differ depending on whether the heatmap was requested.
    """
    with_heatmap = engine.analyze_path(brisc_glioma[0], with_heatmap=True)
    without = engine.analyze_path(brisc_glioma[0], with_heatmap=False)

    assert with_heatmap.call_key == without.call_key


def test_the_model_is_left_in_eval_mode_afterwards(engine, brisc_glioma):
    """A model left in training mode would let BatchNorm statistics drift."""
    engine.analyze_path(brisc_glioma[0], with_heatmap=True)

    module = engine.models[0].module
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            assert not submodule.training, "a BatchNorm layer was left in training mode"


def test_no_parameter_gained_a_gradient(engine, brisc_glioma):
    """Grad-CAM backpropagates. Nothing may accumulate into the weights."""
    module = engine.models[0].module
    for parameter in module.parameters():
        parameter.grad = None

    engine.analyze_path(brisc_glioma[0], with_heatmap=True)

    with_grad = [
        name for name, parameter in module.named_parameters() if parameter.grad is not None
    ]
    assert not with_grad, f"gradients were left on {len(with_grad)} parameters"


def test_the_explainer_makes_no_network_call(monkeypatch, real_explainer, engine, brisc_glioma):
    import socket
    import urllib.request

    def blocked(*args, **kwargs):
        raise AssertionError("the explainer attempted a network connection")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)

    result = engine.analyze_path(brisc_glioma[0], with_heatmap=True)
    assert result.overlay_rgb is not None
