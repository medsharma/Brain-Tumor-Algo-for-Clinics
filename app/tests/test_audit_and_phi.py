"""The audit trail, and keeping patient names out of it.

Clinic filenames routinely carry patient names and dates of birth. A log that
records them is a patient data leak sitting in a plain text file on a shared
laptop.
"""

from __future__ import annotations

import json

import pytest

from app.core import audit, hashing, paths


def test_a_prediction_writes_exactly_one_audit_line(engine, brisc_glioma):
    before = sum(1 for _ in audit.read_all())
    engine.analyze_path(brisc_glioma[0])
    after = sum(1 for _ in audit.read_all())
    assert after == before + 1


def test_the_audit_record_holds_what_a_regulator_would_ask_for(engine, brisc_glioma):
    engine.analyze_path(brisc_glioma[0])
    record = list(audit.read_all())[-1]

    for field in (
        "timestamp_utc",
        "image_sha256",
        "app_version",
        "model_backbone",
        "model_seeds",
        "config_version",
        "call",
        "confidence",
    ):
        assert record.get(field) not in (None, ""), f"audit record is missing {field}"

    assert len(record["image_sha256"]) == 64
    assert record["timestamp_utc"].endswith("Z")


def test_the_filename_never_reaches_the_log(engine, tmp_path, brisc_glioma):
    """The scenario this exists for: a real patient name in a real filename."""
    import shutil

    name = "SMITH_JOHN_1962-04-11_axial_T1.jpg"
    path = tmp_path / name
    shutil.copy(brisc_glioma[0], path)

    engine.analyze_path(path)

    log_file = audit.log_path()
    text = log_file.read_text(encoding="utf-8")

    assert "SMITH" not in text
    assert "JOHN" not in text
    # The full dated form, not a bare "1962". A four-digit run appears inside
    # ordinary probability floats, so asserting on it would fail at random.
    assert "1962-04-11" not in text
    assert name not in text
    assert "axial_T1" not in text

    record = list(audit.read_all())[-1]
    assert record["filename_hash"]
    assert len(record["filename_hash"]) == hashing.FILENAME_HASH_LENGTH
    assert record["file_extension"] == ".jpg"


def test_the_same_filename_hashes_the_same_way_twice():
    """Repeat scans of one file must line up in the log."""
    key_dir = paths.audit_dir()
    first = hashing.hash_filename("SMITH_JOHN.jpg", key_dir)
    second = hashing.hash_filename("SMITH_JOHN.jpg", key_dir)
    assert first == second
    assert first != hashing.hash_filename("SMITH_JANE.jpg", key_dir)


def test_the_filename_hash_is_not_a_plain_sha256():
    """A plain hash is reversible by guessing names. This one is not.

    Anyone with the log and a list of candidate patient names could hash each
    name and look for a match. The HMAC key never leaves the laptop, so the
    log alone gives an attacker nothing.
    """
    import hashlib

    key_dir = paths.audit_dir()
    name = "SMITH_JOHN.jpg"
    plain = hashlib.sha256(name.encode()).hexdigest()[: hashing.FILENAME_HASH_LENGTH]
    assert hashing.hash_filename(name, key_dir) != plain


def test_an_audit_record_carrying_a_filename_is_rejected():
    """A guard, not a convention. Adding a filename field must fail loudly."""
    with pytest.raises(audit.AuditIntegrityError):
        audit._check_no_phi({"timestamp_utc": "x", "filename": "SMITH_JOHN.jpg"})

    with pytest.raises(audit.AuditIntegrityError):
        audit._check_no_phi({"patient_id": "12345"})


def test_no_audit_field_name_could_hold_an_identifier():
    """Every field of the record type is checked against the forbidden list."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(audit.AuditRecord)}
    assert not field_names & audit.FORBIDDEN_KEYS


def test_the_log_is_append_only_across_runs(engine, brisc_glioma):
    """A second prediction adds a line rather than replacing the file."""
    engine.analyze_path(brisc_glioma[0])
    first_count = sum(1 for _ in audit.read_all())
    engine.analyze_path(brisc_glioma[0])
    assert sum(1 for _ in audit.read_all()) == first_count + 1


def test_the_log_is_valid_jsonl(engine, brisc_glioma):
    engine.analyze_path(brisc_glioma[0])
    for line in audit.log_path().read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


def test_rejected_images_are_logged_too(engine, tmp_path):
    """A refusal is a decision and belongs in the trail."""
    path = tmp_path / "junk.txt"
    path.write_text("not a scan", encoding="utf-8")

    before = sum(1 for _ in audit.read_all())
    engine.analyze_path(path)
    records = list(audit.read_all())

    assert len(records) == before + 1
    assert records[-1]["call"] == "cannot_read"


def test_a_stub_config_is_flagged_in_the_log(stub_config, checkpoints_available, brisc_glioma):
    """So nobody later mistakes a development run for a clinical one."""
    if not checkpoints_available:
        pytest.skip("Model checkpoints are not present on this machine")

    from app.core.engine import TriageEngine

    engine = TriageEngine(cfg=stub_config, enforce_readiness=False)
    engine.analyze_path(brisc_glioma[0])

    record = list(audit.read_all())[-1]
    assert record["config_is_stub"] is True
    assert record["config_version"].startswith("STUB@")


def test_the_install_key_is_local_and_not_in_the_repo():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    key_path = paths.audit_dir() / "install_key.bin"
    assert key_path.is_file()
    assert repo_root not in key_path.parents, (
        "the per-install key must live in the user data directory, not the repo"
    )
