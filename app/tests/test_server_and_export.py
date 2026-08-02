"""The web shell: status, analysis, and exports that keep the disclaimer."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core.decision import DISCLAIMER_FULL
from app.server import create_app, set_engine


@pytest.fixture
def client(engine):
    """A client that presents a loopback address.

    ``TestClient`` defaults to the literal host ``"testclient"``, which the
    app's middleware rejects because it is not an IP address at all. That
    rejection is correct and stays; the test supplies a real loopback address
    instead of the app being loosened to accommodate a test harness.
    """
    set_engine(engine)
    with TestClient(create_app(), client=("127.0.0.1", 50001)) as test_client:
        yield test_client


def test_a_request_from_another_machine_is_refused(engine):
    """Belt and braces behind the loopback bind."""
    set_engine(engine)
    with TestClient(create_app(), client=("192.168.1.50", 40000)) as remote:
        response = remote.get("/api/status")
    assert response.status_code == 403
    assert "local machine only" in response.text


def test_a_client_with_no_valid_address_is_refused(engine):
    set_engine(engine)
    with TestClient(create_app()) as odd:
        assert odd.get("/api/status").status_code == 403


def test_the_page_loads_and_references_only_local_assets(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "/static/app.css" in body
    assert "/static/app.js" in body
    assert "http://" not in body
    assert "https://" not in body


def test_static_assets_are_served(client):
    for path in ("/static/app.css", "/static/app.js"):
        assert client.get(path).status_code == 200


def test_status_reports_scope_honestly(client):
    payload = client.get("/api/status").json()
    assert payload["supports_dicom"] is False
    assert payload["disclaimer"] == DISCLAIMER_FULL
    assert payload["config"]["thresholds_fitted_on"]
    assert "state" in payload


def test_analysing_a_real_scan_returns_a_call_and_images(client, brisc_glioma):
    with open(brisc_glioma[0], "rb") as handle:
        response = client.post(
            "/api/analyze", files={"file": ("scan.jpg", handle, "image/jpeg")}
        )
    assert response.status_code == 200
    payload = response.json()

    assert payload["call"]
    assert payload["disclaimer"] == DISCLAIMER_FULL
    assert payload["original_png"].startswith("data:image/png;base64,")
    assert payload["overlay_png"].startswith("data:image/png;base64,")


def test_images_are_data_uris_not_fetchable_urls(client, brisc_glioma):
    """A patient scan must never have an address something could request."""
    with open(brisc_glioma[0], "rb") as handle:
        payload = client.post(
            "/api/analyze", files={"file": ("scan.jpg", handle, "image/jpeg")}
        ).json()

    assert not payload["original_png"].startswith("http")
    assert not payload["original_png"].startswith("/")


def test_a_non_image_upload_returns_the_fourth_call(client):
    response = client.post(
        "/api/analyze", files={"file": ("notes.txt", b"not a scan", "text/plain")}
    )
    assert response.status_code == 200
    assert response.json()["call_key"] == "cannot_read"


def test_responses_are_not_cached(client):
    response = client.get("/api/status")
    assert "no-store" in response.headers["Cache-Control"]


def test_the_content_security_policy_forbids_outside_sources(client):
    policy = client.get("/").headers["Content-Security-Policy"]
    assert "default-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy


# ------------------------------------------------------------------ exports

def _analyse(client, path):
    with open(path, "rb") as handle:
        return client.post(
            "/api/analyze", files={"file": ("scan.jpg", handle, "image/jpeg")}
        ).json()


def test_an_export_always_carries_the_disclaimer(client, brisc_glioma):
    payload = _analyse(client, brisc_glioma[0])
    response = client.post("/api/export", json=payload)

    assert response.status_code == 200
    assert DISCLAIMER_FULL.splitlines()[0] in response.text
    assert "metastases" in response.text


def test_the_disclaimer_cannot_be_stripped_from_an_export(client, brisc_glioma):
    """The server writes it, so editing the page cannot remove it."""
    payload = _analyse(client, brisc_glioma[0])
    payload["disclaimer"] = ""

    response = client.post("/api/export", json=payload)
    assert "not a diagnosis" in response.text.lower()
    assert "metastases" in response.text


def test_an_export_withholds_the_filename_by_default(client, brisc_glioma):
    payload = _analyse(client, brisc_glioma[0])
    payload["display_name"] = "SMITH_JOHN_1962.jpg"

    response = client.post("/api/export", json=payload)
    assert "SMITH" not in response.text
    assert "filename withheld" in response.text


def test_an_export_can_include_the_filename_when_asked(client, brisc_glioma):
    payload = _analyse(client, brisc_glioma[0])
    payload["display_name"] = "SMITH_JOHN_1962.jpg"
    payload["include_filename"] = True

    response = client.post("/api/export", json=payload)
    assert "SMITH_JOHN_1962.jpg" in response.text


def test_an_export_is_self_contained_and_offline(client, brisc_glioma):
    payload = _analyse(client, brisc_glioma[0])
    text = client.post("/api/export", json=payload).text

    assert "http://" not in text
    assert "https://" not in text
    assert "data:image/png;base64," in text


def test_an_export_records_where_the_picture_came_from(client, brisc_glioma):
    """A converted DICOM is not the image the published figures were measured on.

    The sheet that goes into a patient's file has to say which of the two it
    was, because it changes how much weight the answer deserves.
    """
    payload = _analyse(client, brisc_glioma[0])
    text = client.post("/api/export", json=payload).text
    assert "Image came from" in text
    assert "used as supplied" in text

    payload["source_format"] = "dicom"
    converted = client.post("/api/export", json=payload).text
    assert "converted by this app" in converted
    assert "measured on slices exported as JPEG" in converted


def test_export_escapes_injected_markup(client, brisc_glioma):
    """A filename is attacker-influenced text. It must not become markup."""
    payload = _analyse(client, brisc_glioma[0])
    payload["display_name"] = "<script>alert(1)</script>.jpg"
    payload["include_filename"] = True

    text = client.post("/api/export", json=payload).text
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_the_export_records_which_model_and_config_produced_it(client, brisc_glioma):
    payload = _analyse(client, brisc_glioma[0])
    text = client.post("/api/export", json=payload).text

    assert payload["image_sha256"] in text
    assert payload["config_version"] in text
    assert payload["app_version"] in text


# ------------------------------------------------------- when it was read

def test_a_result_records_when_the_scan_was_read(client, brisc_glioma):
    payload = _analyse(client, brisc_glioma[0])

    assert payload["read_at_utc"], "the result carries no reading time"
    assert payload["read_at_utc"].endswith("Z"), "the reading time is not UTC"


def test_an_export_is_dated(client, brisc_glioma):
    """A result sheet with no date on it is not a record.

    Filed in a patient's notes it cannot be tied to a visit, ordered against a
    referral, or checked afterwards. This shipped undated.
    """
    payload = _analyse(client, brisc_glioma[0])
    text = client.post("/api/export", json=payload).text

    assert "Scan read at" in text
    assert payload["read_at_utc"] in text
    assert "This sheet made at" in text


def test_an_older_export_says_the_time_is_missing_rather_than_inventing_one(
        client, brisc_glioma):
    """Stamping the export time as the reading time would be a quiet lie.

    They can be days apart. A sheet that says a scan was read at the moment
    somebody pressed Save is worse than one that admits it does not know.
    """
    payload = _analyse(client, brisc_glioma[0])
    payload.pop("read_at_utc")

    text = client.post("/api/export", json=payload).text
    assert "not recorded by this version" in text


def test_a_case_reference_reaches_the_printed_sheet(client, brisc_glioma):
    """Without it the sheet cannot be filed against a patient at all."""
    payload = _analyse(client, brisc_glioma[0])
    payload["case_reference"] = "CLINIC-2291 / 01-08-2026"

    text = client.post("/api/export", json=payload).text
    assert "Case reference" in text
    assert "CLINIC-2291 / 01-08-2026" in text


def test_no_case_reference_means_no_empty_row(client, brisc_glioma):
    payload = _analyse(client, brisc_glioma[0])
    assert "Case reference" not in client.post("/api/export", json=payload).text


def test_a_case_reference_is_escaped_and_capped(client, brisc_glioma):
    """Operator-typed text is still attacker-influenced text."""
    payload = _analyse(client, brisc_glioma[0])
    payload["case_reference"] = "<script>alert(1)</script>" + "x" * 200

    text = client.post("/api/export", json=payload).text
    assert "<script>alert(1)</script>" not in text
    assert "x" * 100 not in text, "the reference was not length-capped"


def test_the_case_reference_is_never_written_to_the_audit_log(client, brisc_glioma,
                                                              tmp_path, monkeypatch):
    """It exists so a sheet can be filed, not so the app can keep it.

    The audit log is the one thing this app does write to disk, and it is
    deliberately free of anything identifying a patient. A clinic number is
    exactly that.
    """
    from app.core import paths

    payload = _analyse(client, brisc_glioma[0])
    payload["case_reference"] = "CLINIC-2291"
    client.post("/api/export", json=payload)

    log_dir = paths.audit_dir()
    written = "".join(
        path.read_text(encoding="utf-8") for path in sorted(log_dir.glob("*.jsonl"))
    )
    assert "CLINIC-2291" not in written
