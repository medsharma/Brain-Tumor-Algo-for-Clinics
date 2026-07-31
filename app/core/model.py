"""Model construction, checkpoint loading, and MC-Dropout inference on CPU.

Architectures mirror ``BrainTumorViT`` and ``BrainTumorResNet50`` in
``src/code.py`` exactly, because the checkpoints are state dicts of those
classes. Two deliberate differences:

- Backbones are built with ``weights=None``. ``src/code.py`` requests ImageNet
  weights, which torchvision downloads over the network. The app must work on
  a laptop that has never been online, and the checkpoint overwrites every
  parameter anyway, so the download is both fatal and pointless here.
- Loading is strict. A missing or unexpected key means the checkpoint does not
  match the architecture, and that must be an error, never a warning.

Speed note. Both networks put their only dropout layers in the classification
head. Everything before the head is deterministic at eval time, so running T
Monte Carlo passes over the whole network recomputes an identical trunk T
times. This module computes the trunk once and runs only the head T times.
That is an exact algebraic identity, not an approximation, and
``verify_mc_equivalence`` proves it at runtime rather than asking you to trust
the argument.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torchvision import models

from . import hashing

log = logging.getLogger(__name__)

NUM_CLASSES = 4
EMBED_DIM = 768
DROPOUT_P = 0.3

#: Checkpoint sha256 of all zeros means "the stub did not know". Skip the check.
_UNKNOWN_SHA = "0" * 64


class ModelLoadError(RuntimeError):
    """A checkpoint is missing, corrupt, or does not match the architecture."""


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------

def _make_head(in_features: int, dropout_p: float = DROPOUT_P) -> nn.Sequential:
    """The shared classification head. Identical in both backbones."""
    return nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(256, NUM_CLASSES),
    )


class BrainTumorViT(nn.Module):
    """ViT-B/16, structurally identical to the training-time class."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_p: float = DROPOUT_P) -> None:
        super().__init__()
        backbone = models.vit_b_16(weights=None)
        backbone.heads = _make_head(EMBED_DIM, dropout_p)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Everything up to, but not including, the head.

        Copied from ``torchvision.models.VisionTransformer.forward``. If
        torchvision ever changes that method, ``verify_mc_equivalence``
        catches the divergence.
        """
        backbone = self.backbone
        x = backbone._process_input(x)
        batch_class_token = backbone.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = backbone.encoder(x)
        return x[:, 0]

    def head(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.heads(features)


class BrainTumorResNet50(nn.Module):
    """ResNet-50, structurally identical to the training-time class."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_p: float = DROPOUT_P) -> None:
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.fc = _make_head(backbone.fc.in_features, dropout_p)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)

    def head(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(features)


BACKBONES: dict[str, Callable[[], nn.Module]] = {
    "vit": BrainTumorViT,
    "resnet50": BrainTumorResNet50,
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def activate_dropout(module: nn.Module) -> None:
    """Put only the dropout layers back into training mode.

    Never ``model.train()``. That would let BatchNorm running statistics
    update during inference, which is both wrong and, on BRISC, a form of
    contamination.
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.train()


@dataclass
class LoadedModel:
    module: nn.Module
    kind: str
    seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    trained_epoch: int | None
    val_acc: float | None


def load_checkpoint(
    kind: str,
    seed: int,
    checkpoint_path: str | Path,
    expected_sha256: str | None = None,
    *,
    verify_hash: bool = True,
) -> LoadedModel:
    """Build the architecture and load one checkpoint into it.

    Raises ``ModelLoadError`` on anything unexpected. There is no lenient path:
    a partially loaded model produces confident nonsense.
    """
    if kind not in BACKBONES:
        raise ModelLoadError(f"Unknown backbone {kind!r}. Expected one of {sorted(BACKBONES)}.")

    path = Path(checkpoint_path)
    if not path.is_file():
        raise ModelLoadError(
            f"Model file not found:\n  {path}\n"
            f"Checkpoints are not stored in git because each is over 200 MB. "
            f"Check the path in the deployment config points at the real "
            f"results folder on this machine."
        )

    if verify_hash and expected_sha256 and expected_sha256 != _UNKNOWN_SHA:
        actual = hashing.sha256_file(path)
        if actual != expected_sha256:
            raise ModelLoadError(
                f"Model file does not match the deployment config.\n"
                f"  file     : {path}\n"
                f"  expected : {expected_sha256}\n"
                f"  found    : {actual}\n"
                f"Refusing to run. Every safety number in the config was measured "
                f"on a different file than this one."
            )

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - torch raises a wide variety here
        raise ModelLoadError(f"Could not read the model file {path}: {exc}") from exc

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        trained_epoch = payload.get("epoch")
        val_acc = payload.get("val_acc")
    else:
        state_dict = payload
        trained_epoch = None
        val_acc = None

    module = BACKBONES[kind]()
    try:
        module.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(
            f"The model file {path.name} does not fit the {kind} architecture "
            f"this app builds. This usually means the config names the wrong "
            f"backbone for that checkpoint.\n\n{exc}"
        ) from exc

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    return LoadedModel(
        module=module,
        kind=kind,
        seed=seed,
        checkpoint_path=path,
        checkpoint_sha256=expected_sha256 or "",
        trained_epoch=int(trained_epoch) if isinstance(trained_epoch, (int, float)) else None,
        val_acc=float(val_acc) if isinstance(val_acc, (int, float)) else None,
    )


# --------------------------------------------------------------------------
# MC Dropout
# --------------------------------------------------------------------------

@torch.no_grad()
def mc_forward_full(
    module: nn.Module,
    x: torch.Tensor,
    T: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """The straightforward T passes over the whole network.

    Returns stacked probabilities, shape ``(T, B, C)``. Kept as the reference
    implementation that the fast path is checked against.
    """
    module.eval()
    activate_dropout(module)
    samples = [torch.softmax(module(x) / temperature, dim=-1) for _ in range(T)]
    return torch.stack(samples, dim=0)


@torch.no_grad()
def mc_forward_cached(
    module: nn.Module,
    x: torch.Tensor,
    T: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Trunk once, head T times. Same distribution, a fraction of the work.

    Falls back to the full path for any module that does not expose a
    trunk/head split.
    """
    if not (hasattr(module, "trunk") and hasattr(module, "head")):
        return mc_forward_full(module, x, T, temperature)

    module.eval()
    activate_dropout(module)
    features = module.trunk(x)
    samples = [torch.softmax(module.head(features) / temperature, dim=-1) for _ in range(T)]
    return torch.stack(samples, dim=0)


@torch.no_grad()
def verify_mc_equivalence(
    module: nn.Module,
    x: torch.Tensor,
    T: int = 20,
    seed: int = 12345,
    tolerance: float = 1e-6,
) -> tuple[bool, float, str]:
    """Prove the fast path equals the reference path on this exact input.

    Both paths consume randomness in the same order, so seeding identically
    makes them bit-comparable rather than merely similar in distribution.

    Returns ``(passed, max_abs_difference, note)``.
    """
    if not (hasattr(module, "trunk") and hasattr(module, "head")):
        return True, 0.0, "no trunk/head split, fast path is the reference path"

    # The trunk must be deterministic, otherwise caching it is wrong outright.
    module.eval()
    activate_dropout(module)
    trunk_a = module.trunk(x)
    trunk_b = module.trunk(x)
    trunk_delta = (trunk_a - trunk_b).abs().max().item()
    if trunk_delta != 0.0:
        return False, trunk_delta, (
            "the trunk is not deterministic across calls, so caching it changes "
            "the sampling distribution"
        )

    torch.manual_seed(seed)
    reference = mc_forward_full(module, x, T)
    torch.manual_seed(seed)
    fast = mc_forward_cached(module, x, T)

    delta = (reference - fast).abs().max().item()
    passed = delta <= tolerance
    note = (
        f"max |full - cached| = {delta:.3e} over T={T}"
        if passed
        else f"MISMATCH: max |full - cached| = {delta:.3e} exceeds {tolerance:.0e}"
    )
    return passed, delta, note


def entropy_from_probs(probs: torch.Tensor, units: str, epsilon: float = 1e-10) -> torch.Tensor:
    """Predictive entropy of a probability vector.

    ``src/code.py`` uses log2, so ``units="bits"`` reproduces its numbers
    exactly. ``units="nats"`` matches the Contract 1 column description. The
    two differ by a factor of ln 2, which is why the app resolves the unit
    explicitly instead of assuming.
    """
    if units == "bits":
        return -(probs * torch.log2(probs + epsilon)).sum(dim=-1)
    if units == "nats":
        return -(probs * torch.log(probs + epsilon)).sum(dim=-1)
    raise ValueError(f"entropy units must be 'bits' or 'nats', got {units!r}")


@dataclass(frozen=True)
class MCResult:
    """Aggregated Monte Carlo output for a single image."""

    mean_probs: tuple[float, ...]
    std_probs: tuple[float, ...]
    entropy: float
    entropy_units: str
    entropy_bits: float
    entropy_nats: float
    mutual_information: float
    n_passes: int
    n_models: int

    @property
    def pred_index(self) -> int:
        return max(range(len(self.mean_probs)), key=lambda i: self.mean_probs[i])

    @property
    def p_tumor(self) -> float:
        """Probability of any tumour class. Classes 0-2 are the tumour families."""
        return float(sum(self.mean_probs[:3]))


@torch.no_grad()
def run_mc_dropout(
    modules: list[nn.Module],
    x: torch.Tensor,
    T: int,
    temperature: float,
    entropy_units: str,
    *,
    fast: bool = True,
) -> MCResult:
    """Run MC Dropout across one or more models and aggregate.

    With several models the posterior samples from every model are pooled, so
    an ensemble of S seeds at T passes contributes S*T samples. That keeps the
    predictive mean the same quantity regardless of ensemble size.
    """
    if not modules:
        raise ValueError("run_mc_dropout needs at least one model")

    forward = mc_forward_cached if fast else mc_forward_full
    per_model = [forward(module, x, T, temperature) for module in modules]
    stacked = torch.cat(per_model, dim=0)          # (S*T, B, C)

    mean_probs = stacked.mean(dim=0)               # (B, C)
    std_probs = stacked.std(dim=0) if stacked.shape[0] > 1 else torch.zeros_like(mean_probs)

    entropy_bits = entropy_from_probs(mean_probs, "bits")
    entropy_nats = entropy_from_probs(mean_probs, "nats")
    chosen = entropy_bits if entropy_units == "bits" else entropy_nats

    # Mutual information (BALD): total uncertainty minus the average
    # uncertainty of each individual pass. High values mean the passes
    # disagree with each other, which is the model-uncertainty part.
    per_pass_entropy = entropy_from_probs(stacked, entropy_units)   # (S*T, B)
    mutual_information = chosen - per_pass_entropy.mean(dim=0)

    return MCResult(
        mean_probs=tuple(mean_probs[0].tolist()),
        std_probs=tuple(std_probs[0].tolist()),
        entropy=float(chosen[0].item()),
        entropy_units=entropy_units,
        entropy_bits=float(entropy_bits[0].item()),
        entropy_nats=float(entropy_nats[0].item()),
        mutual_information=float(mutual_information[0].item()),
        n_passes=int(stacked.shape[0]),
        n_models=len(modules),
    )


def configure_cpu_threads(threads: int | None = None) -> int:
    """Set the torch thread count for a mid-range laptop.

    Defaults to leaving one core free so the machine stays usable while a scan
    is being read.
    """
    import os

    if threads is None:
        cpu_count = os.cpu_count() or 4
        threads = max(1, min(8, cpu_count - 1))
    torch.set_num_threads(int(threads))
    return int(threads)
