"""The local audit trail.

Every prediction gets one line. A clinic needs this to answer "what did the
tool say about this scan, and which version said it", and any future
regulatory conversation will start here.

Local file only. Nothing is sent anywhere, ever.

What is deliberately absent: filenames, patient identifiers, and image data.
Filenames in clinics routinely carry patient names, so only a salted
install-local hash is written. The image content hash is enough to prove which
pixels produced which call, and it reveals nothing on its own.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import paths

_write_lock = threading.Lock()

AUDIT_SCHEMA_VERSION = "1.0"

#: Keys that must never appear in an audit record. Enforced, not just intended.
FORBIDDEN_KEYS = frozenset({
    "filename",
    "file_name",
    "display_name",
    "path",
    "filepath",
    "file_path",
    "patient",
    "patient_id",
    "patient_name",
    "image",
    "image_bytes",
    "image_data",
    "pixels",
})


class AuditIntegrityError(RuntimeError):
    """An audit record tried to carry something that could identify a patient."""


@dataclass
class AuditRecord:
    """One prediction. Field names are stable; downstream tools read them."""

    timestamp_utc: str
    audit_schema_version: str

    # What was read
    image_sha256: str
    filename_hash: str
    file_extension: str
    image_width: int
    image_height: int

    # What read it
    app_version: str
    model_backbone: str
    model_seeds: list[int]
    checkpoint_sha256_prefixes: list[str]
    config_version: str
    config_schema_version: str
    mc_passes: int
    temperature: float

    # What it said
    call: str
    confidence: str
    p_tumor: float | None
    entropy: float | None
    entropy_units: str | None
    mutual_information: float | None
    class_probabilities: dict[str, float] | None
    tumor_type: str | None

    # How the input check went
    validator_method: str
    validator_in_scope: bool
    validator_reason: str

    # Operating conditions, so a bad batch can be traced later
    latency_ms: float
    explainer_is_stub: bool
    validator_is_stub: bool
    config_is_stub: bool
    dev_mode: bool
    notes: list[str] = field(default_factory=list)


def _check_no_phi(record: dict[str, Any]) -> None:
    offenders = FORBIDDEN_KEYS.intersection(record)
    if offenders:
        raise AuditIntegrityError(
            f"Refusing to write an audit record containing {sorted(offenders)}. "
            f"Those fields can carry patient identifiers."
        )


def log_path(when: datetime | None = None) -> Path:
    """One file per month. Keeps files small enough to open in a text editor."""
    when = when or datetime.now(timezone.utc)
    return paths.audit_dir() / f"predictions-{when:%Y-%m}.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def write(record: AuditRecord) -> Path:
    """Append one record. Returns the file written to.

    Append-only, flushed and fsynced before returning, so a crash or a pulled
    power cord cannot lose the line that has already been reported on screen.
    """
    payload = asdict(record)
    _check_no_phi(payload)

    target = log_path()
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    with _write_lock:
        with open(target, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return target


def read_all(path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Iterate records from one log file. Skips lines that will not parse."""
    target = Path(path) if path is not None else log_path()
    if not target.is_file():
        return
    with open(target, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
