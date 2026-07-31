"""Preprocessing, held identical to the analysis pipeline.

This is the module most likely to silently invalidate everything. If the app
resizes or normalises even slightly differently from ``get_transforms("test")``
in ``src/code.py``, then every number session A measured describes a different
program than the one the clinic runs, and nothing visible breaks.

Two defences:

1. This module uses the same ``torchvision.transforms`` calls in the same
   order, so parity holds by construction rather than by coincidence.
2. ``app/tests/test_preprocessing_parity.py`` imports the real
   ``get_transforms("test")`` and asserts the two produce bit-identical
   tensors on real images. That test is the contract.

Things deliberately *not* done here, because the analysis pipeline does not do
them either:

- No EXIF auto-rotation. ``Image.open`` does not apply it and neither do we.
- No histogram equalisation, denoising, or skull stripping.
- No centre crop. Test-time resize is a straight squash to 224x224, which
  changes aspect ratio. That is what the model was trained to see.
"""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from torchvision import transforms

# Frozen copies of the constants in src/code.py. Duplicated on purpose: the app
# must not import the analysis code at runtime, so packaging stays small and
# the app cannot be broken by an unrelated edit to the research script. The
# parity test asserts these still match.
TRAINING_RESIZE: tuple[int, int] = (224, 224)
TRAINING_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
TRAINING_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_transform(
    resize: Sequence[int] = TRAINING_RESIZE,
    mean: Sequence[float] = TRAINING_MEAN,
    std: Sequence[float] = TRAINING_STD,
) -> transforms.Compose:
    """The test-time transform.

    Mirrors the ``split != "train"`` branch of ``get_transforms`` exactly:
    ``Resize((H, W))`` then ``ToTensor()`` then ``Normalize(mean, std)``.

    Note ``Resize`` receives a two-element tuple, not an int. Passing an int
    would resize the short side and preserve aspect ratio, which is a
    different image. This distinction is the classic way preprocessing drifts.
    """
    height, width = int(resize[0]), int(resize[1])
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(float(v) for v in mean), tuple(float(v) for v in std)),
    ])


def preprocess_pil(
    image: Image.Image,
    transform: transforms.Compose | None = None,
) -> torch.Tensor:
    """PIL image to a normalised ``(1, 3, H, W)`` batch of one.

    The ``convert("RGB")`` matches ``ManifestDataset.__getitem__``. MRI is
    greyscale, so this replicates one channel three times, which is what the
    ImageNet-pretrained backbones expect.
    """
    if transform is None:
        transform = build_transform()
    tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0)


def transform_matches_training(
    resize: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    tolerance: float = 1e-9,
) -> tuple[bool, str]:
    """Check config preprocessing against the values the model was trained on.

    Returns ``(matches, explanation)``. The app logs loudly on a mismatch
    rather than silently honouring a config that would move every input off
    the distribution the checkpoint expects.
    """
    problems: list[str] = []

    if tuple(int(v) for v in resize) != TRAINING_RESIZE:
        problems.append(f"resize {list(resize)} != training {list(TRAINING_RESIZE)}")

    for label, got, want in (
        ("normalize_mean", mean, TRAINING_MEAN),
        ("normalize_std", std, TRAINING_STD),
    ):
        if any(abs(float(a) - b) > tolerance for a, b in zip(got, want)):
            problems.append(f"{label} {list(got)} != training {list(want)}")

    if problems:
        return False, "; ".join(problems)
    return True, "matches the training pipeline"
