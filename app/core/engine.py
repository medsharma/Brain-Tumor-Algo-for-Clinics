"""The inference core. No UI, no web framework, importable and testable alone.

All the logic lives here. ``app/server.py`` is a thin shell over it, so
anything the app decides can be reproduced from a script or a test without
starting a server.

One engine per process. Models are loaded once at startup and kept in memory.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from . import audit, decision, hashing, image_loading, model as model_module
from . import paths, preprocess, readiness
from . import version as version_module
from .config import DeploymentConfig, load_config
from .decision import Call, Decision
from .explain_adapter import Explainer, get_explainer
from .image_loading import ImageLoadError, LoadedImage
from .validation_adapter import InputValidator, ValidationResult, get_validator

log = logging.getLogger(__name__)

SUPPORTED_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


@dataclass
class TriageResult:
    """Everything one image produced. Serialisable, and safe to export.

    ``display_name`` is the only field carrying the original filename. It is
    kept in memory for the operator, excluded from the audit log, and excluded
    from exports unless the operator asks for it.
    """

    call: str
    call_key: str
    confidence: str
    confidence_key: str
    reason: str
    next_step: str
    probability_text: str | None

    p_tumor: float | None
    entropy: float | None
    entropy_units: str | None
    mutual_information: float | None
    class_probabilities: dict[str, float] | None
    tumor_type: str | None
    tumor_type_probability: float | None

    image_sha256: str
    filename_hash: str
    display_name: str
    image_width: int
    image_height: int

    validator_in_scope: bool
    validator_reason: str
    validator_method: str
    validator_is_stub: bool

    app_version: str
    model_backbone: str
    model_seeds: list[int]
    config_version: str
    mc_passes: int
    latency_ms: float

    disclaimer: str
    heatmap_caveat: str
    dev_mode: bool
    config_is_stub: bool
    explainer_is_stub: bool
    notes: list[str] = field(default_factory=list)

    #: "image" for a file that arrived as a picture, "dicom" for one this app
    #: converted itself. Every published accuracy figure was measured on the
    #: first kind, so the difference has to reach the screen and the export.
    source_format: str = "image"

    #: When this reading was taken, UTC, ISO 8601.
    #:
    #: A result sheet in a patient's file with no date on it is not a record.
    #: Nobody can tell later which visit it belongs to, which scan it describes,
    #: or whether it came before or after the referral. The audit log has always
    #: carried this; the thing a clinic actually prints and files did not.
    #:
    #: Recorded when the image is read, not when somebody presses Save, because
    #: those can be days apart and it is the reading that is being reported.
    read_at_utc: str = ""

    # Not serialised into JSON. Held for the UI to encode as PNG.
    original_rgb: np.ndarray | None = None
    heatmap: np.ndarray | None = None
    overlay_rgb: np.ndarray | None = None

    def to_dict(self, include_display_name: bool = False) -> dict[str, Any]:
        """JSON-safe dictionary. Drops image arrays and, by default, the filename."""
        skip = {"original_rgb", "heatmap", "overlay_rgb"}
        if not include_display_name:
            skip.add("display_name")
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in skip
        }


@dataclass
class SeriesResult:
    """A folder of slices. Deliberately does not produce a study-level call."""

    slices: list[TriageResult]
    n_files_seen: int
    n_unreadable: int
    study_level_call: str | None
    study_level_note: str
    disclaimer: str
    heatmap_caveat: str


class TriageEngine:
    """Loads the config and models once, then answers questions about images."""

    def __init__(
        self,
        cfg: DeploymentConfig | None = None,
        *,
        allow_stub: bool | None = None,
        load_models: bool = True,
        force_stub_validator: bool = False,
        force_stub_explainer: bool = False,
        enforce_readiness: bool = True,
        verify_checkpoint_hash: bool = True,
    ) -> None:
        self.config = cfg if cfg is not None else load_config(allow_stub=allow_stub)
        self.validator: InputValidator = get_validator(force_stub=force_stub_validator)
        self.explainer: Explainer = get_explainer(force_stub=force_stub_explainer)

        self.readiness = readiness.check(self.config, self.validator, self.explainer)
        if enforce_readiness and not self.readiness.ok:
            raise readiness.NotReadyError(self.readiness)

        self.threads = model_module.configure_cpu_threads()
        self.transform = preprocess.build_transform(
            self.config.resize, self.config.normalize_mean, self.config.normalize_std
        )
        self.audit_dir = paths.audit_dir()

        self.models: list[model_module.LoadedModel] = []
        if load_models:
            self.load_models(verify_hash=verify_checkpoint_hash)

        self.mc_equivalence_verified = False
        self.mc_equivalence_note = "not checked yet"

    # -- setup --------------------------------------------------------------

    def load_models(self, verify_hash: bool = True) -> None:
        started = time.perf_counter()
        self.models = [
            model_module.load_checkpoint(
                kind=checkpoint.model,
                seed=checkpoint.seed,
                checkpoint_path=checkpoint.path,
                expected_sha256=checkpoint.sha256,
                verify_hash=verify_hash,
            )
            for checkpoint in self.config.checkpoints
        ]
        log.info(
            "Loaded %d model(s) in %.1f s: %s",
            len(self.models),
            time.perf_counter() - started,
            ", ".join(f"{m.kind}/seed{m.seed}" for m in self.models),
        )

    def verify_fast_path(self, tolerance: float = 1e-6) -> tuple[bool, str]:
        """Confirm the cached-trunk optimisation matches the reference pass.

        Run once at startup on a synthetic input. If it ever fails, the engine
        falls back to the slow path rather than shipping a different sampling
        distribution than session A measured.
        """
        if not self.models:
            return False, "no models loaded"

        dummy = torch.zeros(1, 3, self.config.resize[0], self.config.resize[1])
        notes: list[str] = []
        all_passed = True
        for loaded in self.models:
            passed, _, note = model_module.verify_mc_equivalence(
                loaded.module, dummy, T=min(self.config.mc_T, 8), tolerance=tolerance
            )
            all_passed = all_passed and passed
            notes.append(f"{loaded.kind}/seed{loaded.seed}: {note}")

        self.mc_equivalence_verified = all_passed
        self.mc_equivalence_note = "; ".join(notes)
        return all_passed, self.mc_equivalence_note

    # -- helpers ------------------------------------------------------------

    @property
    def _ece(self) -> float | None:
        """Expected calibration error, if session A reported one.

        Contract 2 has no field for it, so this reads an optional additive key.
        While it is absent the app shows confidence in words only and no
        percentages at all, which is the honest position when nobody has
        measured how well the numbers track reality.
        """
        value = self.config.expected_performance.get("ece")
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
        return None

    def _model_identity(self) -> tuple[str, list[int], list[str]]:
        backbone = self.config.chosen_backbone
        seeds = [checkpoint.seed for checkpoint in self.config.checkpoints]
        prefixes = [checkpoint.sha256[:12] for checkpoint in self.config.checkpoints]
        return backbone, seeds, prefixes

    def _run_validator(self, loaded: LoadedImage, source_path: Path | None) -> ValidationResult:
        """Ask session B whether this image is in scope.

        Passes the path when we have one, since B may want file-level signals.
        For an upload there is no path, and writing patient images to a temp
        file to create one would be the wrong trade, so an array goes instead.
        """
        subject: Any = source_path if source_path is not None else np.asarray(
            loaded.image.convert("RGB")
        )
        return self.validator.validate(subject)

    def _build_heatmap(
        self,
        tensor: torch.Tensor,
        target_class: int | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
        """Original, heatmap and overlay at the model's input resolution."""
        notes: list[str] = []

        denormalised = self._denormalise(tensor)
        original_rgb = (denormalised * 255.0).clip(0, 255).astype(np.uint8)

        if not self.models:
            return original_rgb, None, None, notes

        try:
            heatmap = self.explainer.generate_heatmap(
                self.models[0].module, tensor, self.models[0].kind, target_class
            )
            overlay = self.explainer.overlay_heatmap(original_rgb, heatmap, alpha=0.4)
        except Exception as exc:  # noqa: BLE001 - a broken heatmap must not kill the call
            log.warning("Heatmap generation failed: %s", exc)
            notes.append(
                "The heatmap could not be produced for this scan. The call above "
                "is unaffected, but there is nothing to sanity check it against."
            )
            return original_rgb, None, None, notes

        return original_rgb, heatmap, overlay, notes

    def _denormalise(self, tensor: torch.Tensor) -> np.ndarray:
        """Undo Normalize, giving an HxWx3 float array in [0, 1]."""
        mean = torch.tensor(self.config.normalize_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.config.normalize_std).view(1, 3, 1, 1)
        restored = (tensor * std + mean).clamp(0.0, 1.0)
        return restored[0].permute(1, 2, 0).numpy()

    # -- the main entry points ---------------------------------------------

    def analyze_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        with_heatmap: bool = True,
        source_path: Path | None = None,
        write_audit: bool = True,
    ) -> TriageResult:
        """Judge one image. The only path that produces a clinical call."""
        started = time.perf_counter()
        notes: list[str] = []

        try:
            loaded = image_loading.load_image_bytes(data, filename, self.audit_dir)
        except ImageLoadError as exc:
            return self._unreadable_result(
                reason=exc.message,
                filename=filename,
                data=data,
                started=started,
                write_audit=write_audit,
                kind=exc.kind,
            )

        # How the picture was made, when the app made it. A DICOM converted
        # here is not the image any published figure was measured on, and the
        # operator has to be told that on the result rather than in a manual.
        notes.extend(loaded.conversion_notes)

        validation = self._run_validator(loaded, source_path)

        if not validation.is_in_scope:
            decided = decision.cannot_read(validation.reason)
            return self._finalise(
                decided, loaded, validation, mc=None, tensor=None,
                started=started, notes=notes, with_heatmap=False,
                write_audit=write_audit,
            )

        tensor = preprocess.preprocess_pil(loaded.image, self.transform)

        mc = model_module.run_mc_dropout(
            [loaded_model.module for loaded_model in self.models],
            tensor,
            T=self.config.mc_T,
            temperature=self.config.temperature,
            entropy_units=self.config.entropy_units,
            fast=True,
        )

        decided = decision.decide(
            mean_probs=mc.mean_probs,
            p_tumor=mc.p_tumor,
            entropy=mc.entropy,
            class_names=self.config.class_names,
            tumor_threshold=self.config.tumor_threshold,
            entropy_defer_threshold=self.config.entropy_defer_threshold,
            max_entropy=self.config.max_entropy(),
            ece=self._ece,
            mutual_information=mc.mutual_information,
            defer_signal=str(self.config.raw.get("defer_signal", "entropy")),
        )

        return self._finalise(
            decided, loaded, validation, mc=mc, tensor=tensor,
            started=started, notes=notes, with_heatmap=with_heatmap,
            write_audit=write_audit,
        )

    def analyze_path(
        self,
        path: str | Path,
        *,
        with_heatmap: bool = True,
        write_audit: bool = True,
    ) -> TriageResult:
        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            return self._unreadable_result(
                reason=f"That file could not be opened. ({exc.strerror or exc})",
                filename=path.name,
                data=b"",
                started=time.perf_counter(),
                write_audit=write_audit,
                kind="unreadable",
            )
        return self.analyze_bytes(
            data,
            path.name,
            with_heatmap=with_heatmap,
            source_path=path,
            write_audit=write_audit,
        )

    def analyze_folder(
        self,
        folder: str | Path,
        *,
        with_heatmap: bool = False,
        write_audit: bool = True,
    ) -> SeriesResult:
        """Run every image in a folder, one slice at a time.

        This deliberately stops short of a study-level answer. A real MRI study
        is many slices and the model judges one. Taking the maximum tumour
        probability across slices is a reasonable rule, but it moves the
        operating point: the more slices you look at, the more chances there
        are for one to cross the threshold, so a per-slice threshold applied to
        a study gives a different false-alarm rate than the one that was
        measured. That needs its own threshold fitted on the internal
        validation split. Session C will not invent one. See
        ``handoff/ISSUES.md`` entry C-2.
        """
        folder = Path(folder)
        candidates: list[Path] = []
        for pattern in SUPPORTED_IMAGE_GLOBS:
            candidates.extend(sorted(folder.glob(pattern)))
        candidates = sorted(set(candidates))

        results: list[TriageResult] = []
        unreadable = 0
        for candidate in candidates:
            result = self.analyze_path(
                candidate, with_heatmap=with_heatmap, write_audit=write_audit
            )
            if result.call_key == Call.CANNOT_READ.key:
                unreadable += 1
            results.append(result)

        threshold = self.config.series_tumor_threshold
        if threshold is None:
            study_call = None
            note = (
                "No study-level call. This tool judges one slice at a time. "
                "Combining slices into a single answer needs a separate "
                "threshold that has not been measured, and guessing one would "
                "change how often the tool raises a false alarm without anyone "
                "noticing. Read the per-slice results below."
            )
        else:
            tumour_probabilities = [
                result.p_tumor for result in results if result.p_tumor is not None
            ]
            highest = max(tumour_probabilities, default=None)
            if highest is None:
                study_call = None
                note = "No slice in this folder could be read."
            elif highest >= threshold:
                study_call = Call.TUMOR.value
                note = (
                    f"Study-level call from the highest-scoring slice, using the "
                    f"study threshold of {threshold:.3f} from the deployment config."
                )
            else:
                study_call = Call.NO_TUMOR.value
                note = (
                    f"Study-level call from the highest-scoring slice, using the "
                    f"study threshold of {threshold:.3f} from the deployment config."
                )

        return SeriesResult(
            slices=results,
            n_files_seen=len(candidates),
            n_unreadable=unreadable,
            study_level_call=study_call,
            study_level_note=note,
            disclaimer=decision.DISCLAIMER_FULL,
            heatmap_caveat=self.explainer.caveat,
        )

    # -- result assembly ----------------------------------------------------

    def _finalise(
        self,
        decided: Decision,
        loaded: LoadedImage,
        validation: ValidationResult,
        *,
        mc: model_module.MCResult | None,
        tensor: torch.Tensor | None,
        started: float,
        notes: list[str],
        with_heatmap: bool,
        write_audit: bool,
    ) -> TriageResult:
        original_rgb = heatmap = overlay = None
        if with_heatmap and tensor is not None:
            target = mc.pred_index if mc is not None else None
            original_rgb, heatmap, overlay, heatmap_notes = self._build_heatmap(tensor, target)
            notes = notes + heatmap_notes

        backbone, seeds, prefixes = self._model_identity()
        latency_ms = (time.perf_counter() - started) * 1000.0

        class_probabilities = None
        if mc is not None:
            class_probabilities = {
                name: float(probability)
                for name, probability in zip(self.config.class_names, mc.mean_probs)
            }

        if self.config.is_stub:
            notes.append(
                "DEVELOPMENT BUILD. The thresholds behind this answer are fake "
                "placeholder numbers. This result means nothing clinically."
            )
        if self.explainer.is_stub:
            notes.append(
                "The heatmap is a test pattern, not where the model looked."
            )
        if self.validator.is_stub:
            notes.append(
                "The out-of-scope check is not installed. Nothing is being rejected."
            )

        result = TriageResult(
            call=decided.call.value,
            call_key=decided.call.key,
            confidence=decided.confidence.value,
            confidence_key=decided.confidence.key,
            reason=decided.reason,
            next_step=decided.next_step,
            probability_text=decided.probability_text,
            p_tumor=decided.p_tumor,
            entropy=decided.entropy,
            entropy_units=self.config.entropy_units if mc is not None else None,
            mutual_information=mc.mutual_information if mc is not None else None,
            class_probabilities=class_probabilities,
            tumor_type=decided.tumor_type,
            tumor_type_probability=decided.tumor_type_probability,
            image_sha256=loaded.sha256,
            filename_hash=loaded.filename_hash,
            display_name=loaded.display_name,
            image_width=loaded.width,
            image_height=loaded.height,
            validator_in_scope=validation.is_in_scope,
            validator_reason=validation.reason,
            validator_method=validation.method,
            validator_is_stub=self.validator.is_stub,
            app_version=version_module.APP_VERSION,
            model_backbone=backbone,
            model_seeds=seeds,
            config_version=self.config.config_version,
            mc_passes=mc.n_passes if mc is not None else 0,
            latency_ms=latency_ms,
            disclaimer=decision.DISCLAIMER_FULL,
            heatmap_caveat=self.explainer.caveat,
            dev_mode=self.readiness.dev_mode,
            config_is_stub=self.config.is_stub,
            explainer_is_stub=self.explainer.is_stub,
            notes=notes,
            source_format=loaded.source_format,
            read_at_utc=audit.utc_now_iso(),
            original_rgb=original_rgb,
            heatmap=heatmap,
            overlay_rgb=overlay,
        )

        if write_audit:
            self._write_audit(result, prefixes)
        return result

    def _unreadable_result(
        self,
        *,
        reason: str,
        filename: str,
        data: bytes,
        started: float,
        write_audit: bool,
        kind: str,
    ) -> TriageResult:
        """The CANNOT READ path for files that never became an image."""
        decided = decision.cannot_read(reason)
        backbone, seeds, prefixes = self._model_identity()

        result = TriageResult(
            call=decided.call.value,
            call_key=decided.call.key,
            confidence=decided.confidence.value,
            confidence_key=decided.confidence.key,
            reason=decided.reason,
            next_step=decided.next_step,
            probability_text=None,
            p_tumor=None,
            entropy=None,
            entropy_units=None,
            mutual_information=None,
            class_probabilities=None,
            tumor_type=None,
            tumor_type_probability=None,
            image_sha256=hashing.sha256_bytes(data),
            filename_hash=hashing.hash_filename(filename, self.audit_dir),
            display_name=Path(filename).name,
            image_width=0,
            image_height=0,
            validator_in_scope=False,
            validator_reason=reason,
            validator_method=f"file_check:{kind}",
            validator_is_stub=self.validator.is_stub,
            app_version=version_module.APP_VERSION,
            model_backbone=backbone,
            model_seeds=seeds,
            config_version=self.config.config_version,
            mc_passes=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            disclaimer=decision.DISCLAIMER_FULL,
            heatmap_caveat=self.explainer.caveat,
            dev_mode=self.readiness.dev_mode,
            config_is_stub=self.config.is_stub,
            explainer_is_stub=self.explainer.is_stub,
            notes=[],
            read_at_utc=audit.utc_now_iso(),
        )

        if write_audit:
            self._write_audit(result, prefixes)
        return result

    def _write_audit(self, result: TriageResult, checkpoint_prefixes: list[str]) -> None:
        try:
            audit.write(audit.AuditRecord(
                timestamp_utc=audit.utc_now_iso(),
                audit_schema_version=audit.AUDIT_SCHEMA_VERSION,
                image_sha256=result.image_sha256,
                filename_hash=result.filename_hash,
                file_extension=hashing.safe_extension(result.display_name),
                image_width=result.image_width,
                image_height=result.image_height,
                app_version=result.app_version,
                model_backbone=result.model_backbone,
                model_seeds=list(result.model_seeds),
                checkpoint_sha256_prefixes=checkpoint_prefixes,
                config_version=result.config_version,
                config_schema_version=self.config.schema_version,
                mc_passes=result.mc_passes,
                temperature=self.config.temperature,
                call=result.call_key,
                confidence=result.confidence_key,
                p_tumor=result.p_tumor,
                entropy=result.entropy,
                entropy_units=result.entropy_units,
                mutual_information=result.mutual_information,
                class_probabilities=result.class_probabilities,
                tumor_type=result.tumor_type,
                validator_method=result.validator_method,
                validator_in_scope=result.validator_in_scope,
                validator_reason=result.validator_reason,
                latency_ms=result.latency_ms,
                explainer_is_stub=result.explainer_is_stub,
                validator_is_stub=result.validator_is_stub,
                config_is_stub=result.config_is_stub,
                dev_mode=result.dev_mode,
                notes=list(result.notes),
            ))
        except Exception as exc:  # noqa: BLE001 - a failed log must not lose the call
            log.error("Could not write the audit record: %s", exc)
