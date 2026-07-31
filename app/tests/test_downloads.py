"""The download list must describe the model that is actually answering.

The promise the download page makes is narrow and worth protecting: what you
download reproduces this server, rather than resembling it. That holds only if
the catalogue is derived from the live config. A hardcoded list, or a directory
listing, would drift the first time the operating point changed, and the failure
would be silent: someone downloads five files, runs them, and gets different
answers from the page they downloaded them from.

These tests pin that, plus the two things that make a file server dangerous:
serving a path someone asked for, and serving a file that is not what it claims.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import downloads


@pytest.fixture
def client(engine):
    """A client the loopback middleware will accept.

    ``TestClient`` reports its host as the literal string ``"testclient"``,
    which is not an IP address, so the local-only middleware answers 403 to
    everything. Same convention as ``test_server_and_export.py``.
    """
    import app.server as server

    server.set_engine(engine)
    with TestClient(server.create_app(), client=("127.0.0.1", 50002)) as c:
        yield c
    server.set_engine(None)


def test_catalogue_matches_the_running_config(engine):
    """Every checkpoint the engine loaded is offered, and nothing else is."""
    items = downloads.build_catalogue(engine.config)
    models = [i for i in items if i.kind == "model"]

    on_disk = [c for c in engine.config.checkpoints if Path(c.path).is_file()]
    assert len(models) == len(on_disk)

    offered = {i.filename for i in models}
    expected = {Path(c.path).name for c in on_disk}
    assert offered == expected


def test_the_settings_file_is_offered_too(engine):
    """Weights alone are useless: the app will not invent the thresholds."""
    items = downloads.build_catalogue(engine.config)
    assert any(i.filename == "deployment_config.json" for i in items)


def test_manifest_reports_the_live_operating_point(engine):
    items = downloads.build_catalogue(engine.config)
    m = downloads.manifest(engine.config, items)
    cfg = m["app_config"]

    assert cfg["backbone"] == engine.config.chosen_backbone
    assert cfg["tumor_threshold"] == engine.config.tumor_threshold
    assert cfg["defer_threshold"] == engine.config.entropy_defer_threshold
    assert cfg["thresholds_fitted_on"] == engine.config.thresholds_fitted_on


def test_manifest_says_it_is_not_validated(engine):
    items = downloads.build_catalogue(engine.config)
    m = downloads.manifest(engine.config, items)
    assert "not for clinical use" in m["not_validated"].lower()


def test_api_lists_downloads(client):
    r = client.get("/api/downloads")
    assert r.status_code == 200
    body = r.json()
    assert body["files"], "nothing offered"
    for f in body["files"]:
        assert len(f["sha256"]) == 64
        assert f["bytes"] > 0
        assert f["description"]


def test_downloading_a_file_returns_its_real_bytes(client):
    """Serve the actual file, and let the client verify it without asking twice."""
    listing = client.get("/api/downloads").json()
    smallest = min(listing["files"], key=lambda f: f["bytes"])

    r = client.get(f"/api/downloads/{smallest['key']}")
    assert r.status_code == 200
    assert hashlib.sha256(r.content).hexdigest() == smallest["sha256"]
    assert r.headers.get("X-Content-SHA256") == smallest["sha256"]
    assert smallest["filename"] in r.headers.get("Content-Disposition", "")


@pytest.mark.parametrize("key", [
    "../../../../etc/passwd",
    "..%2F..%2Fsecrets",
    "....//....//app/core/config.py",
    "/absolute/path",
    "does-not-exist",
])
def test_only_catalogued_keys_are_served(client, key):
    """The key is matched against a list, never joined onto a directory.

    There is no traversal to defend against because no path arithmetic happens:
    a key that is not in the catalogue simply matches nothing.
    """
    r = client.get(f"/api/downloads/{key}")
    assert r.status_code != 200, f"{key!r} was served"
    assert r.status_code in (404, 405, 307), f"{key!r} returned {r.status_code}"


def test_an_empty_key_falls_through_to_the_listing(client):
    """`/api/downloads/` is the list endpoint after a slash redirect.

    Worth pinning rather than leaving to chance: it must return the catalogue,
    which is public information, and never a file.
    """
    r = client.get("/api/downloads/")
    assert r.status_code == 200
    body = r.json()
    assert "files" in body, "an empty key returned something other than the listing"


def test_hashes_are_stable_across_calls(engine):
    """Cached by size and mtime, so a repeat call is cheap and still correct."""
    a = downloads.build_catalogue(engine.config)
    b = downloads.build_catalogue(engine.config)
    assert [i.sha256 for i in a] == [i.sha256 for i in b]


def test_hash_cache_notices_a_changed_file(tmp_path):
    """A replaced file must not be served under the old hash."""
    f = tmp_path / "thing.bin"
    f.write_bytes(b"first")
    first = downloads.file_sha256(f)

    import os
    import time
    time.sleep(0.01)
    f.write_bytes(b"second, different length")
    os.utime(f, None)

    assert downloads.file_sha256(f) != first
    assert downloads.file_sha256(f) == hashlib.sha256(b"second, different length").hexdigest()


def test_human_size_reads_naturally():
    assert downloads.human_size(344_056_471) == "344 MB"
    assert downloads.human_size(1_720_000_000) == "1.72 GB"
    assert downloads.human_size(900) == "900 bytes"
