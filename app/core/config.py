"""Deployment configuration: loading, validation, and the stub guard.

Contract 2 (see ``prompts/CONTRACTS.md``) says session A owns
``analysis/results/safety/deployment_config.json``. Session C built the app
before that file existed, against a deliberately fake stub at
``deployment_config.SCHEMA.json``.

The guard in this module is the reason that is not dangerous. A made-up
threshold in a medical triage tool is worse than no tool at all, so the app
refuses to start on the stub unless the operator has explicitly turned on
development mode. That refusal is code, not a comment.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------
# Where things live
# --------------------------------------------------------------------------

#: Repository root, resolved from this file: app/core/config.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

REAL_CONFIG_RELPATH = Path("analysis/results/safety/deployment_config.json")
STUB_CONFIG_RELPATH = Path("analysis/results/safety/deployment_config.SCHEMA.json")

#: Point the app at a specific config file.
CONFIG_PATH_ENV = "MRI_CLINIC_CONFIG"
#: Set to "1" to permit the stub config. Development only. Never in a clinic.
DEV_MODE_ENV = "MRI_CLINIC_DEV_MODE"

SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})

# --------------------------------------------------------------------------
# Contract 2 shape
# --------------------------------------------------------------------------

REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "created_utc",
    "chosen_backbone",
    "chosen_seeds",
    "ensemble",
    "mc_T",
    "temperature",
    "tumor_threshold",
    "entropy_defer_threshold",
    "thresholds_fitted_on",
    "class_names",
    "preprocessing",
    "checkpoints",
    "expected_performance",
)

REQUIRED_PREPROCESSING_KEYS: tuple[str, ...] = (
    "resize",
    "normalize_mean",
    "normalize_std",
)

REQUIRED_PERFORMANCE_KEYS: tuple[str, ...] = (
    "dataset",
    "n",
    "tumor_miss_rate",
    "tumor_miss_rate_ci",
    "binary_sensitivity",
    "binary_specificity",
    "four_way_accuracy",
    "defer_rate",
    "miss_rate_after_defer",
)

#: Frozen by src/code.py. Order matters: it is the model's output order.
EXPECTED_CLASS_NAMES: tuple[str, ...] = ("glioma", "meningioma", "pituitary", "notumor")

SUPPORTED_BACKBONES = frozenset({"vit", "resnet50"})

# Predictive entropy over 4 classes is bounded. The bound differs by unit, and
# that is how we can sometimes tell which unit a threshold was written in.
MAX_ENTROPY_NATS = math.log(len(EXPECTED_CLASS_NAMES))   # 1.3863
MAX_ENTROPY_BITS = math.log2(len(EXPECTED_CLASS_NAMES))  # 2.0


class ConfigError(RuntimeError):
    """The deployment config is missing, malformed, or internally inconsistent."""


class StubConfigError(ConfigError):
    """The app was asked to run clinically on fake numbers. Hard stop."""


# --------------------------------------------------------------------------
# Stub detection
# --------------------------------------------------------------------------

def detect_stub(raw: Mapping[str, Any], path: Path) -> tuple[bool, str]:
    """Decide whether ``raw`` is the day-one stub rather than real config.

    Three independent signals, so removing any one of them by hand is not
    enough to sneak fake numbers past the guard.

    Returns ``(is_stub, reason)``.
    """
    reasons: list[str] = []

    if raw.get("_THIS_IS_A_STUB") is True:
        reasons.append("the file carries the _THIS_IS_A_STUB marker")

    version = str(raw.get("schema_version", ""))
    if "stub" in version.lower():
        reasons.append(f"schema_version is {version!r}")

    if path.name == STUB_CONFIG_RELPATH.name:
        reasons.append(f"the filename is {path.name}")

    fitted = str(raw.get("thresholds_fitted_on", ""))
    if "stub" in fitted.lower():
        reasons.append(f"thresholds_fitted_on is {fitted!r}")

    return bool(reasons), "; ".join(reasons)


def dev_mode_enabled() -> bool:
    """True when the operator has explicitly opted into development mode."""
    return os.environ.get(DEV_MODE_ENV, "").strip() == "1"


# --------------------------------------------------------------------------
# Entropy units
# --------------------------------------------------------------------------

def resolve_entropy_units(raw: Mapping[str, Any]) -> tuple[str, str]:
    """Work out whether ``entropy_defer_threshold`` is in bits or nats.

    This matters more than it looks. ``src/code.py`` computes predictive
    entropy with ``log2``, so its numbers are in bits. Contract 1 describes
    the cached entropy column as nats. Those differ by a factor of about 1.44.
    Read a bits threshold as nats and the app defers far less than session A
    intended, which means confident "NO TUMOR" calls on scans a human was
    supposed to see.

    Order of preference:

    1. An explicit ``entropy_units`` key, if session A adds one.
    2. Inference: a threshold above ln(4) cannot be nats, so it is bits.
    3. Fall back to bits, matching ``src/code.py``. Bits are the larger
       number for the same scan, so this is the choice that defers more.

    Returns ``(units, how_it_was_determined)``.
    """
    declared = raw.get("entropy_units")
    if isinstance(declared, str) and declared.lower() in {"bits", "nats"}:
        return declared.lower(), "declared in config"

    threshold = raw.get("entropy_defer_threshold")
    if isinstance(threshold, (int, float)) and threshold > MAX_ENTROPY_NATS:
        return "bits", (
            f"inferred: threshold {threshold} exceeds the maximum possible "
            f"entropy in nats ({MAX_ENTROPY_NATS:.4f}), so it must be bits"
        )

    return "bits", (
        "ASSUMED: config does not say, and the value is valid in either unit. "
        "Defaulting to bits to match src/code.py, which is also the assumption "
        "that defers more scans to a human. See handoff/ISSUES.md entry C-1."
    )


# --------------------------------------------------------------------------
# The config object
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Checkpoint:
    model: str
    seed: int
    path: str
    sha256: str


@dataclass(frozen=True)
class DeploymentConfig:
    """Validated Contract 2 config. Frozen; nothing mutates it after load."""

    schema_version: str
    created_utc: str
    chosen_backbone: str
    chosen_seeds: tuple[int, ...]
    ensemble: bool
    mc_T: int
    temperature: float
    tumor_threshold: float
    entropy_defer_threshold: float
    entropy_units: str
    entropy_units_source: str
    thresholds_fitted_on: str
    class_names: tuple[str, ...]
    resize: tuple[int, int]
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    checkpoints: tuple[Checkpoint, ...]
    expected_performance: Mapping[str, Any]
    is_stub: bool
    stub_reason: str
    source_path: Path
    raw: Mapping[str, Any]

    # -- optional, additive keys -------------------------------------------

    @property
    def series_tumor_threshold(self) -> float | None:
        """Study-level threshold for a multi-slice folder, if A ever sets one.

        Absent by design. Taking the maximum tumour probability across slices
        moves the operating point, so it needs its own threshold fitted on the
        internal validation split. Session C will not invent one. See
        handoff/ISSUES.md entry C-2.
        """
        value = self.raw.get("series_tumor_threshold")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def config_version(self) -> str:
        """Short identifier written into the audit log."""
        marker = "STUB" if self.is_stub else self.schema_version
        return f"{marker}@{self.created_utc}"

    def max_entropy(self) -> float:
        return MAX_ENTROPY_BITS if self.entropy_units == "bits" else MAX_ENTROPY_NATS


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _as_float_triple(value: Any, name: str) -> tuple[float, float, float]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 3,
        f"{name} must be a list of three numbers, got {value!r}",
    )
    return (float(value[0]), float(value[1]), float(value[2]))


def validate_raw(raw: Mapping[str, Any], path: Path) -> None:
    """Structural validation. Raises ``ConfigError`` on the first problem.

    Fails loudly rather than defaulting, as Contract 2 requires.
    """
    _require(isinstance(raw, Mapping), f"{path} does not contain a JSON object")

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    _require(not missing, f"{path} is missing required keys: {', '.join(missing)}")

    preprocessing = raw["preprocessing"]
    _require(isinstance(preprocessing, Mapping), "preprocessing must be an object")
    missing_pre = [k for k in REQUIRED_PREPROCESSING_KEYS if k not in preprocessing]
    _require(not missing_pre, f"preprocessing is missing: {', '.join(missing_pre)}")

    performance = raw["expected_performance"]
    _require(isinstance(performance, Mapping), "expected_performance must be an object")
    missing_perf = [k for k in REQUIRED_PERFORMANCE_KEYS if k not in performance]
    _require(
        not missing_perf,
        f"expected_performance is missing: {', '.join(missing_perf)}",
    )

    class_names = raw["class_names"]
    _require(
        tuple(class_names) == EXPECTED_CLASS_NAMES,
        f"class_names must be exactly {list(EXPECTED_CLASS_NAMES)} in that order, "
        f"got {list(class_names)!r}. The order is the model's output order; "
        f"reordering it silently relabels every prediction.",
    )

    backbone = raw["chosen_backbone"]
    _require(
        backbone in SUPPORTED_BACKBONES,
        f"chosen_backbone must be one of {sorted(SUPPORTED_BACKBONES)}, got {backbone!r}",
    )

    seeds = raw["chosen_seeds"]
    _require(
        isinstance(seeds, Sequence) and not isinstance(seeds, str) and len(seeds) >= 1,
        f"chosen_seeds must be a non-empty list, got {seeds!r}",
    )

    mc_T = raw["mc_T"]
    _require(
        isinstance(mc_T, int) and mc_T >= 1,
        f"mc_T must be an integer >= 1, got {mc_T!r}",
    )

    temperature = raw["temperature"]
    _require(
        isinstance(temperature, (int, float)) and temperature > 0,
        f"temperature must be a positive number, got {temperature!r}",
    )

    tumor_threshold = raw["tumor_threshold"]
    _require(
        isinstance(tumor_threshold, (int, float)) and 0.0 <= tumor_threshold <= 1.0,
        f"tumor_threshold must be between 0 and 1, got {tumor_threshold!r}",
    )

    entropy_threshold = raw["entropy_defer_threshold"]
    _require(
        isinstance(entropy_threshold, (int, float)) and entropy_threshold >= 0.0,
        f"entropy_defer_threshold must be >= 0, got {entropy_threshold!r}",
    )
    _require(
        entropy_threshold <= MAX_ENTROPY_BITS + 1e-9,
        f"entropy_defer_threshold {entropy_threshold} exceeds the maximum possible "
        f"4-class predictive entropy in either unit ({MAX_ENTROPY_BITS} bits). "
        f"Nothing would ever be deferred.",
    )

    resize = preprocessing["resize"]
    _require(
        isinstance(resize, Sequence)
        and not isinstance(resize, str)
        and len(resize) == 2
        and all(isinstance(v, int) and v > 0 for v in resize),
        f"preprocessing.resize must be two positive integers, got {resize!r}",
    )

    _as_float_triple(preprocessing["normalize_mean"], "preprocessing.normalize_mean")
    _as_float_triple(preprocessing["normalize_std"], "preprocessing.normalize_std")

    checkpoints = raw["checkpoints"]
    _require(
        isinstance(checkpoints, Sequence)
        and not isinstance(checkpoints, str)
        and len(checkpoints) >= 1,
        "checkpoints must be a non-empty list",
    )
    for index, entry in enumerate(checkpoints):
        _require(
            isinstance(entry, Mapping)
            and {"model", "seed", "path", "sha256"} <= set(entry),
            f"checkpoints[{index}] needs model, seed, path and sha256",
        )
        _require(
            entry["model"] == backbone,
            f"checkpoints[{index}].model is {entry['model']!r} but chosen_backbone "
            f"is {backbone!r}. The app will not mix backbones.",
        )

    ensemble = raw["ensemble"]
    _require(isinstance(ensemble, bool), f"ensemble must be true or false, got {ensemble!r}")
    if not ensemble:
        _require(
            len(seeds) == 1,
            f"ensemble is false but chosen_seeds has {len(seeds)} entries. "
            f"Ambiguous: say which single seed ships.",
        )
    _require(
        len(checkpoints) == len(seeds),
        f"chosen_seeds has {len(seeds)} entries but checkpoints has "
        f"{len(checkpoints)}. Every chosen seed needs exactly one checkpoint.",
    )


def _build(raw: Mapping[str, Any], path: Path, is_stub: bool, stub_reason: str) -> DeploymentConfig:
    preprocessing = raw["preprocessing"]
    units, units_source = resolve_entropy_units(raw)
    resize = tuple(int(v) for v in preprocessing["resize"])
    return DeploymentConfig(
        schema_version=str(raw["schema_version"]),
        created_utc=str(raw["created_utc"]),
        chosen_backbone=str(raw["chosen_backbone"]),
        chosen_seeds=tuple(int(s) for s in raw["chosen_seeds"]),
        ensemble=bool(raw["ensemble"]),
        mc_T=int(raw["mc_T"]),
        temperature=float(raw["temperature"]),
        tumor_threshold=float(raw["tumor_threshold"]),
        entropy_defer_threshold=float(raw["entropy_defer_threshold"]),
        entropy_units=units,
        entropy_units_source=units_source,
        thresholds_fitted_on=str(raw["thresholds_fitted_on"]),
        class_names=tuple(str(c) for c in raw["class_names"]),
        resize=(resize[0], resize[1]),
        normalize_mean=_as_float_triple(preprocessing["normalize_mean"], "normalize_mean"),
        normalize_std=_as_float_triple(preprocessing["normalize_std"], "normalize_std"),
        checkpoints=tuple(
            Checkpoint(
                model=str(entry["model"]),
                seed=int(entry["seed"]),
                path=str(entry["path"]),
                sha256=str(entry["sha256"]),
            )
            for entry in raw["checkpoints"]
        ),
        expected_performance=dict(raw["expected_performance"]),
        is_stub=is_stub,
        stub_reason=stub_reason,
        source_path=path,
        raw=dict(raw),
    )


def default_config_path() -> Path:
    """The real config if session A has published it, otherwise the stub."""
    override = os.environ.get(CONFIG_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()

    real = REPO_ROOT / REAL_CONFIG_RELPATH
    if real.is_file():
        return real
    return REPO_ROOT / STUB_CONFIG_RELPATH


def load_config(
    path: str | Path | None = None,
    *,
    allow_stub: bool | None = None,
) -> DeploymentConfig:
    """Load and validate the deployment config.

    Args:
        path: Config file. Defaults to the real file if present, else the stub.
        allow_stub: Permit fake numbers. Defaults to whether ``MRI_CLINIC_DEV_MODE``
            is set to ``"1"``. Passing ``True`` explicitly is for tests.

    Raises:
        ConfigError: The file is missing, unparseable, or violates Contract 2.
        StubConfigError: It is the stub and stubs are not allowed here.
    """
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()

    if not config_path.is_file():
        raise ConfigError(
            f"No deployment config at {config_path}. Session A publishes "
            f"{REAL_CONFIG_RELPATH.as_posix()}; until then the stub lives at "
            f"{STUB_CONFIG_RELPATH.as_posix()} and needs {DEV_MODE_ENV}=1."
        )

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid JSON: {exc}") from exc

    is_stub, stub_reason = detect_stub(raw, config_path)

    if allow_stub is None:
        allow_stub = dev_mode_enabled()

    if is_stub and not allow_stub:
        raise StubConfigError(
            "REFUSING TO START.\n"
            f"  Config file : {config_path}\n"
            f"  Why refused : this is the placeholder stub ({stub_reason}).\n"
            "  Every threshold in it is fake. A made-up threshold in a triage\n"
            "  tool is worse than no tool. The app will not produce a clinical\n"
            "  call on invented numbers.\n"
            "\n"
            f"  Fix: wait for session A to publish {REAL_CONFIG_RELPATH.as_posix()},\n"
            f"  or point {CONFIG_PATH_ENV} at a real config.\n"
            f"  For development only, set {DEV_MODE_ENV}=1. Never do that in a clinic."
        )

    # Structural validation runs after the stub check so the operator sees the
    # useful message first, not a complaint about a fake field.
    validate_raw(raw, config_path)

    if not is_stub:
        version = str(raw["schema_version"])
        _require(
            version in SUPPORTED_SCHEMA_VERSIONS,
            f"schema_version {version!r} is not supported by this build "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}). Refusing to guess "
            f"what changed.",
        )

    return _build(raw, config_path, is_stub, stub_reason)
