"""The app listens on this laptop only.

A clinic laptop sits on a shared network with reception machines, other
clinics' devices, and whatever else is plugged in. Binding ``0.0.0.0`` would
serve patient scans to all of it with no authentication. Binding loopback
means the socket is not reachable from another machine at all.
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

from app.server import LOOPBACK_HOST, BindingRefused, assert_loopback

APP_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.5", "::1"])
def test_loopback_addresses_are_accepted(host):
    assert_loopback(host)


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.10", "10.0.0.4", "8.8.8.8", "localhost", "example.com", ""],
)
def test_everything_else_is_refused(host):
    """Hostnames are refused too, since they can resolve anywhere."""
    with pytest.raises(BindingRefused):
        assert_loopback(host)


def test_the_default_host_is_loopback():
    assert LOOPBACK_HOST == "127.0.0.1"


def test_run_refuses_a_network_host_before_loading_anything(monkeypatch):
    """The refusal happens first, so no model is loaded and no port is opened."""
    import app.server as server

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("run() got past the binding check")

    monkeypatch.setattr(server, "TriageEngine", explode)

    with pytest.raises(BindingRefused):
        server.run(host="0.0.0.0", open_browser=False)


def test_run_passes_loopback_through_to_uvicorn(monkeypatch, engine):
    """Whatever else run() does, uvicorn is told to bind loopback."""
    import uvicorn

    import app.server as server

    captured: dict[str, object] = {}

    def fake_run(app, host, port, **kwargs):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(uvicorn, "run", fake_run)
    server.run(host=LOOPBACK_HOST, port=0, open_browser=False, engine=engine)

    assert captured["host"] == "127.0.0.1"


def test_source_contains_no_wildcard_bind():
    """Nobody has quietly reintroduced 0.0.0.0 anywhere in the app.

    Parsed rather than grepped. Documentation that names ``0.0.0.0`` to
    explain why it is refused is not a bind address, and a text search cannot
    tell the difference. Only real string literals count, and docstrings are
    excluded.
    """
    import ast

    wildcards = {"0.0.0.0", "::", "0::0"}
    offenders: list[str] = []

    for path in sorted(APP_DIR.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), str(path))

        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    docstrings.add(id(body[0].value))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or id(node) in docstrings:
                continue
            if isinstance(node.value, str) and node.value in wildcards:
                offenders.append(f"{path.relative_to(APP_DIR)}:{node.lineno}: {node.value!r}")

    assert not offenders, "Wildcard bind address literal found:\n  " + "\n  ".join(offenders)


def test_a_real_loopback_socket_is_not_reachable_from_the_lan_address():
    """Bind loopback for real and confirm the LAN address does not answer.

    Skipped on a machine with no LAN address, since there would be nothing to
    check against.
    """
    lan_ip = _primary_lan_address()
    if lan_ip is None:
        pytest.skip("no non-loopback IPv4 address on this machine")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK_HOST, 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as loopback_client:
            loopback_client.settimeout(2.0)
            loopback_client.connect((LOOPBACK_HOST, port))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as lan_client:
            lan_client.settimeout(2.0)
            with pytest.raises((ConnectionRefusedError, socket.timeout, OSError)):
                lan_client.connect((lan_ip, port))


def _primary_lan_address() -> str | None:
    """This machine's outward-facing IPv4 address, without sending anything.

    ``connect`` on a UDP socket only sets the destination locally; no packet
    leaves the machine.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.2)
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1, reserved and unroutable
            address = probe.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") else address
