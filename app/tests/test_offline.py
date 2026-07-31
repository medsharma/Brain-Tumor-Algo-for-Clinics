"""Proof that the app never touches the network.

Two levels of evidence.

The strong one is here: every way Python can open a network connection is
replaced with something that raises, then the whole pipeline runs. That is
stricter than pulling the ethernet cable, because a disconnected machine only
proves nothing *succeeded*, while this proves nothing was even *attempted*.
A silent retry loop or a swallowed timeout would pass the cable test and fail
this one.

The weaker but still worthwhile one is documented in ``app/README.md``: run
the app with the network adapter disabled and confirm it behaves identically.

The specific hazard this catches is torchvision. ``models.resnet50(weights=
ResNet50_Weights.IMAGENET1K_V2)``, which is what ``src/code.py`` calls, will
download about 100 MB the first time it runs. On a clinic laptop that has
never been online it raises and the app never starts. ``app/core/model.py``
builds with ``weights=None`` for exactly that reason, and this test is what
keeps it that way.
"""

from __future__ import annotations

import socket
import urllib.request

import pytest


class NetworkAccessAttempted(AssertionError):
    """Something tried to reach the network. That is a hard failure."""


@pytest.fixture
def no_network(monkeypatch):
    """Make every outbound network primitive raise."""

    def blocked(*args, **kwargs):
        raise NetworkAccessAttempted(
            "The app attempted to use the network. It must run on a laptop "
            "that has never been online."
        )

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "gethostbyname", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(urllib.request, "urlretrieve", blocked)
    yield


def test_building_both_architectures_makes_no_network_call(no_network):
    """Model construction must not fetch pretrained weights."""
    from app.core.model import BrainTumorResNet50, BrainTumorViT

    resnet = BrainTumorResNet50()
    vit = BrainTumorViT()
    assert resnet is not None and vit is not None


def test_full_pipeline_makes_no_network_call(
    no_network, test_config, checkpoints_available, brisc_glioma
):
    """Load config, load a checkpoint, read an image, produce a call."""
    if not checkpoints_available:
        pytest.skip("Model checkpoints are not present on this machine")

    from app.core.engine import TriageEngine

    engine = TriageEngine(cfg=test_config, enforce_readiness=False)
    result = engine.analyze_path(brisc_glioma[0], with_heatmap=True)

    assert result.call_key in {"tumor", "no_tumor", "uncertain", "cannot_read"}
    assert result.mc_passes > 0


def test_creating_the_web_app_makes_no_network_call(no_network, engine):
    from app.server import create_app, set_engine

    set_engine(engine)
    assert create_app() is not None


def test_the_guard_itself_works(no_network):
    """A blocking fixture that blocks nothing would make this file a lie."""
    with pytest.raises(NetworkAccessAttempted):
        socket.socket()
    with pytest.raises(NetworkAccessAttempted):
        urllib.request.urlopen("http://example.invalid")


def test_the_app_never_requests_pretrained_weights():
    """``weights=None`` in both constructors, checked in the source.

    The runtime test above only proves no download happened on a machine where
    the torch cache may already be warm. This checks the code itself.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "model.py").read_text(
        encoding="utf-8"
    )
    assert "models.resnet50(weights=None)" in source
    assert "models.vit_b_16(weights=None)" in source
    assert "IMAGENET1K" not in source


def test_no_static_asset_references_the_internet():
    """No CDN stylesheet, no web font, no analytics beacon."""
    import re
    from pathlib import Path

    static_dir = Path(__file__).resolve().parents[1] / "static"
    pattern = re.compile(r"https?://|//cdn\.|googleapis|jsdelivr|unpkg|cloudflare", re.I)

    offenders: list[str] = []
    for path in sorted(static_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")

    assert not offenders, (
        "Static files reference the internet. On an offline laptop these fail "
        "silently and the page renders wrong:\n  " + "\n  ".join(offenders)
    )


def test_no_telemetry_or_crash_reporting_imports():
    """Nothing in the app pulls in a package whose job is phoning home."""
    from pathlib import Path

    forbidden = ("sentry_sdk", "requests", "httpx", "urllib3", "posthog", "analytics")
    app_dir = Path(__file__).resolve().parents[1]

    offenders: list[str] = []
    for path in sorted(app_dir.rglob("*.py")):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            if any(name in stripped for name in forbidden):
                offenders.append(f"{path.name}:{number}: {stripped}")

    assert not offenders, "Network-capable libraries imported:\n  " + "\n  ".join(offenders)
