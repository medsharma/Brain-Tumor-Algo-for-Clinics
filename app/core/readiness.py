"""The startup gate.

The app can be in three states, and the difference matters more than usual
because the failure mode is a confident wrong answer rather than a crash.

- **Clinical**: real config from session A, session B's input check installed,
  session D's explainer installed, preprocessing matching what the model was
  trained on. This is the only state that may see a patient.
- **Development**: one or more of those is a stub. Allowed only when
  ``MRI_CLINIC_DEV_MODE=1``. Every screen carries a banner saying so.
- **Refused**: something is a stub and development mode is off. The app does
  not start.

Blockers stop startup. Warnings do not, but they are shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config as config_module
from . import preprocess
from .config import DeploymentConfig
from .explain_adapter import Explainer
from .validation_adapter import InputValidator


@dataclass(frozen=True)
class Finding:
    code: str
    message: str


@dataclass(frozen=True)
class Readiness:
    blockers: tuple[Finding, ...]
    warnings: tuple[Finding, ...]
    dev_mode: bool

    @property
    def ok(self) -> bool:
        return not self.blockers

    @property
    def state(self) -> str:
        if self.blockers:
            return "refused"
        if self.warnings or self.dev_mode:
            return "development"
        return "clinical"

    def summary(self) -> str:
        lines: list[str] = []
        for finding in self.blockers:
            lines.append(f"  BLOCKER [{finding.code}] {finding.message}")
        for finding in self.warnings:
            lines.append(f"  WARNING [{finding.code}] {finding.message}")
        return "\n".join(lines)


def check(
    cfg: DeploymentConfig,
    validator: InputValidator,
    explainer: Explainer,
    *,
    dev_mode: bool | None = None,
) -> Readiness:
    """Decide whether this installation may produce clinical calls."""
    if dev_mode is None:
        dev_mode = config_module.dev_mode_enabled()

    stub_findings: list[Finding] = []
    warnings: list[Finding] = []
    blockers: list[Finding] = []

    if cfg.is_stub:
        stub_findings.append(Finding(
            "stub_config",
            f"The deployment config is the placeholder stub ({cfg.stub_reason}). "
            f"Every threshold in it is invented.",
        ))

    if validator.is_stub:
        stub_findings.append(Finding(
            "stub_validator",
            f"Session B's input check is not installed ({validator.detail}). "
            f"Nothing will be rejected as out of scope, so a non-brain image "
            f"would still get a tumour answer.",
        ))

    if explainer.is_stub:
        stub_findings.append(Finding(
            "stub_explainer",
            f"Session D's heatmap is not installed ({explainer.detail}). "
            f"The overlay shown is a test pattern, not where the model looked.",
        ))

    # Stubs block startup unless development mode is explicitly on.
    if dev_mode:
        warnings.extend(stub_findings)
    else:
        blockers.extend(stub_findings)

    matches, detail = preprocess.transform_matches_training(
        cfg.resize, cfg.normalize_mean, cfg.normalize_std
    )
    if not matches:
        blockers.append(Finding(
            "preprocessing_drift",
            f"The config's image preparation does not match what the model was "
            f"trained on: {detail}. Every safety number would describe a "
            f"different program than the one running.",
        ))

    for checkpoint in cfg.checkpoints:
        if not Path(checkpoint.path).is_file():
            blockers.append(Finding(
                "missing_checkpoint",
                f"Model file not found: {checkpoint.path}",
            ))

    if cfg.entropy_units_source.startswith("ASSUMED"):
        warnings.append(Finding(
            "entropy_units_assumed",
            f"The config does not say whether entropy_defer_threshold is in bits "
            f"or nats. {cfg.entropy_units_source}",
        ))

    if not cfg.is_stub:
        miss_rate = cfg.expected_performance.get("tumor_miss_rate")
        if isinstance(miss_rate, (int, float)) and miss_rate > 0.05:
            warnings.append(Finding(
                "high_miss_rate",
                f"Session A measured a tumour miss rate of {miss_rate:.1%} on "
                f"external data. More than one real tumour in twenty is called "
                f"no-tumour. Read app/DEPLOYED_CONFIG_VALIDATION.md before any "
                f"clinical use.",
            ))

        if str(cfg.thresholds_fitted_on).lower().find("brisc") >= 0:
            blockers.append(Finding(
                "contaminated_thresholds",
                f"thresholds_fitted_on is {cfg.thresholds_fitted_on!r}. Thresholds "
                f"must be fitted on the internal validation split, never on BRISC. "
                f"A threshold picked on BRISC and reported on BRISC is circular.",
            ))

    return Readiness(
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        dev_mode=bool(dev_mode),
    )


class NotReadyError(RuntimeError):
    """Startup refused. The message is written for the person at the laptop."""

    def __init__(self, readiness: Readiness) -> None:
        detail = readiness.summary()
        super().__init__(
            "REFUSING TO START.\n\n"
            f"{detail}\n\n"
            "This tool will not produce a clinical call it cannot stand behind.\n"
            "For development only, set MRI_CLINIC_DEV_MODE=1. Never do that in a clinic."
        )
        self.readiness = readiness
