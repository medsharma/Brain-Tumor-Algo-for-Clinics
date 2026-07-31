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


def test_an_export_says_dicom_is_unsupported(client, brisc_glioma):
    payload = _analyse(client, brisc_glioma[0])
    assert "Not supported by this version" in client.post("/api/export", json=payload).text


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
