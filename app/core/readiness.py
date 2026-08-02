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

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
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

    # A clinic laptop is offline by design, so nothing corrects its clock. When
    # the CMOS battery goes, it comes back as 2009 and stays there, and every
    # result sheet and every audit line is then dated wrongly with no warning.
    # A printed record with the wrong date is worse than one with no date: it
    # can be filed against the wrong visit and nobody has a reason to doubt it.
    #
    # There is no reference to check against on a machine with no network. There
    # is one fact: this software cannot have been run before it was built. A
    # clock reading earlier than the config it is loading is wrong, definitely,
    # and that catches the dead-battery case, which is the common one.
    clock = _clock_finding(cfg.created_utc)
    if clock is not None:
        warnings.append(clock)

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

    # ------------------------------------------------------------------
    # Having real config files is not the same as being fit for patients.
    #
    # Every other check here asks "is this installation wired up correctly".
    # They all pass the moment sessions A, B and D publish real files. Nothing
    # was asking the separate and more important question: has this tool been
    # shown to work on data it did not train on, and has a clinician ever
    # looked at it.
    #
    # That gap had a direct consequence. `state` returns "clinical" when there
    # are no blockers and no warnings, and app.js hides the
    # "DEVELOPMENT BUILD - NOT FOR CLINICAL USE" banner on exactly that
    # condition. So finishing A, B and D's work silently removed the warning
    # from the screen, without one line being written about validation.
    #
    # This check is deliberately fail-safe. The banner stays up until someone
    # records positive evidence that it should come down. Absent, unreadable or
    # unrecognised evidence all keep the warning. You cannot clear this by
    # forgetting to fill in a field.
    # ------------------------------------------------------------------
    attest = cfg.raw.get("external_validation") if isinstance(cfg.raw, Mapping) else None
    status = str((attest or {}).get("status", "")).strip().lower() if isinstance(attest, Mapping) else ""
    if status != "independent_cohort":
        detail = str((attest or {}).get("detail", "")).strip() if isinstance(attest, Mapping) else ""
        warnings.append(Finding(
            "no_external_validation",
            # Every fact here is the same as it was. What changed is the order:
            # what it means for the person reading a scan comes first, and the
            # history of how we know it comes after. The old version opened with
            # the BRISC contamination story, which is the most important thing
            # to a developer and close to the least important thing to a nurse
            # deciding whether to act on the answer in front of them.
            detail or (
                "What this tool has been measured on comes from the same "
                "sources, scanners and preparation as the data it learned from. "
                "It has not yet been tested on scans from a different hospital "
                "or a different scanner, and no clinician has reviewed its "
                "output. Expect it to do worse on your scans than the figures "
                "here suggest, by an amount nobody has measured yet. Treat "
                "every answer as a prompt to look, not as a finding. "
                "(The intended external test set, BRISC 2025, turned out to be "
                "roughly 80% the training data republished. The figures shown "
                "are what survived removing that overlap.)"
            ),
        ))

    return Readiness(
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        dev_mode=bool(dev_mode),
    )


def _clock_finding(created_utc: str) -> Finding | None:
    """Warn when this computer's clock is provably wrong.

    Returns None when the clock is plausible, or when the config carries no
    readable creation date, because an unreadable date is not evidence of a
    broken clock and this must not cry wolf on every startup.
    """
    stamp = str(created_utc or "").strip()
    if not stamp:
        return None
    try:
        built = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if built.tzinfo is None:
        built = built.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    if now >= built:
        return None

    return Finding(
        "clock_is_wrong",
        f"This computer's clock says {now.strftime('%Y-%m-%d')}, which is before "
        f"this software was built ({built.strftime('%Y-%m-%d')}). The clock is "
        f"wrong. Every result sheet and every audit entry will carry the wrong "
        f"date until it is corrected, and a report filed under the wrong date "
        f"can end up against the wrong visit.",
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
