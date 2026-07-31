"""The test that stops everything else being quietly meaningless.

If the app prepares an image even slightly differently from
``get_transforms("test")`` in ``src/code.py``, then every number session A
measured describes a different program than the one a clinic runs, nothing
raises an error, and nobody finds out.

So this asserts the two pipelines produce *identical* tensors, not similar
ones. The tolerance is exact equality, because both paths call the same
torchvision operations in the same order and there is no legitimate reason for
a single bit to differ. A loose tolerance here would hide exactly the drift
this test exists to catch.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from app.core import preprocess


@pytest.fixture(scope="module")
def analysis_transform():
    """``get_transforms("test")`` from the research pipeline itself."""
    try:
        from src.code import get_transforms
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"src/code.py is not importable here: {exc}")
    return get_transforms("test")


def _load_rgb(path) -> Image.Image:
    """Exactly what ``ManifestDataset.__getitem__`` does."""
    return Image.open(path).convert("RGB")


def test_transform_pipelines_are_structurally_identical(analysis_transform):
    """Same operations, same order, same parameters."""
    app_transform = preprocess.build_transform()

    app_steps = [type(step).__name__ for step in app_transform.transforms]
    analysis_steps = [type(step).__name__ for step in analysis_transform.transforms]
    assert app_steps == analysis_steps, (
        f"The app's transform steps {app_steps} differ from the analysis "
        f"pipeline's {analysis_steps}."
    )

    app_resize, _, app_norm = app_transform.transforms
    analysis_resize, _, analysis_norm = analysis_transform.transforms

    assert tuple(app_resize.size) == tuple(analysis_resize.size)
    assert app_resize.interpolation == analysis_resize.interpolation
    assert getattr(app_resize, "antialias", None) == getattr(analysis_resize, "antialias", None)
    assert tuple(app_norm.mean) == tuple(analysis_norm.mean)
    assert tuple(app_norm.std) == tuple(analysis_norm.std)


def test_frozen_constants_match_the_training_code(analysis_transform):
    """The app's copied constants still equal the ones in src/code.py."""
    from src.code import _IMAGENET_MEAN, _IMAGENET_STD

    assert preprocess.TRAINING_MEAN == tuple(_IMAGENET_MEAN)
    assert preprocess.TRAINING_STD == tuple(_IMAGENET_STD)

    resize_step = analysis_transform.transforms[0]
    assert tuple(resize_step.size) == preprocess.TRAINING_RESIZE


def test_identical_tensors_on_real_brisc_images(analysis_transform, brisc_mixed):
    """The real check: same file in, bit-identical tensor out.

    Runs across all four classes so a class-specific quirk, such as an image
    saved in a different mode, cannot slip past.
    """
    app_transform = preprocess.build_transform()

    for path in brisc_mixed:
        image = _load_rgb(path)

        expected = analysis_transform(image).unsqueeze(0)
        actual = preprocess.preprocess_pil(image, app_transform)

        assert actual.shape == expected.shape, f"shape differs on {path.name}"
        assert actual.dtype == expected.dtype, f"dtype differs on {path.name}"
        assert torch.equal(actual, expected), (
            f"PREPROCESSING DRIFT on {path.name}. "
            f"Maximum difference {(actual - expected).abs().max().item():.3e}. "
            f"The app is not feeding the model what the analysis fed it, so "
            f"session A's numbers do not describe this app."
        )


@pytest.mark.parametrize(
    "size,mode",
    [
        ((512, 512), "L"),      # the usual BRISC shape, greyscale
        ((256, 300), "L"),      # non-square, so resize squashes aspect ratio
        ((640, 480), "RGB"),    # already colour
        ((64, 64), "L"),        # small
        ((1024, 512), "I;16"),  # 16-bit, the kind of thing a converter emits
    ],
)
def test_identical_tensors_on_awkward_shapes(analysis_transform, tmp_path, size, mode):
    """Parity must hold for shapes and modes BRISC does not happen to contain.

    Aspect ratio is the interesting one. ``Resize((224, 224))`` squashes a
    non-square image; ``Resize(224)`` would preserve the ratio. Those produce
    different pixels, and a drift between them would be invisible on the
    square BRISC images.
    """
    import numpy as np

    rng = np.random.default_rng(abs(hash((size, mode))) % (2**32))
    if mode == "I;16":
        array = (rng.random(size[::-1]) * 65535).astype("uint16")
        image = Image.fromarray(array, mode="I;16")
    elif mode == "RGB":
        array = (rng.random((size[1], size[0], 3)) * 255).astype("uint8")
        image = Image.fromarray(array, mode="RGB")
    else:
        array = (rng.random(size[::-1]) * 255).astype("uint8")
        image = Image.fromarray(array, mode="L")

    path = tmp_path / f"awkward_{size[0]}x{size[1]}_{mode.replace(';', '')}.png"
    image.save(path)

    loaded = _load_rgb(path)
    expected = analysis_transform(loaded).unsqueeze(0)
    actual = preprocess.preprocess_pil(loaded, preprocess.build_transform())

    assert torch.equal(actual, expected), (
        f"PREPROCESSING DRIFT at size {size} mode {mode}."
    )


def test_no_exif_rotation_is_applied(analysis_transform, tmp_path):
    """The analysis pipeline ignores EXIF orientation, so the app must too.

    Silently rotating a scan the analysis did not rotate would be drift that
    only shows up on images from certain cameras or converters.
    """
    import numpy as np

    array = np.zeros((100, 200), dtype="uint8")
    array[:30, :] = 255  # asymmetric, so a rotation would be visible
    image = Image.fromarray(array, mode="L")

    path = tmp_path / "exif.jpg"
    exif = Image.Exif()
    exif[274] = 6  # orientation: rotate 90 degrees clockwise
    image.save(path, exif=exif)

    loaded = _load_rgb(path)
    expected = analysis_transform(loaded).unsqueeze(0)
    actual = preprocess.preprocess_pil(loaded, preprocess.build_transform())

    assert torch.equal(actual, expected)


def test_config_preprocessing_matches_training(stub_config):
    """The deployment config's preprocessing block agrees with the model."""
    matches, detail = preprocess.transform_matches_training(
        stub_config.resize, stub_config.normalize_mean, stub_config.normalize_std
    )
    assert matches, detail


def test_drift_is_actually_detected():
    """Confirm the guard has teeth.

    A test asserting equality is worthless if the comparison could never fail.
    This deliberately introduces the two most likely real drifts and checks
    both are caught.
    """
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(7)
    image = Image.fromarray((rng.random((300, 200)) * 255).astype("uint8"), mode="L")

    reference = preprocess.preprocess_pil(image, preprocess.build_transform())

    # Drift 1: aspect-ratio-preserving resize instead of a squash.
    from torchvision import transforms

    ratio_preserving = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(preprocess.TRAINING_MEAN, preprocess.TRAINING_STD),
    ])
    drifted = ratio_preserving(image.convert("RGB")).unsqueeze(0)
    assert not torch.equal(reference, drifted)

    # Drift 2: normalisation constants half a percent off.
    nudged = preprocess.build_transform(
        preprocess.TRAINING_RESIZE,
        tuple(v + 0.005 for v in preprocess.TRAINING_MEAN),
        preprocess.TRAINING_STD,
    )
    assert not torch.equal(reference, preprocess.preprocess_pil(image, nudged))

    matches, _ = preprocess.transform_matches_training(
        (256, 256), preprocess.TRAINING_MEAN, preprocess.TRAINING_STD
    )
    assert not matches
