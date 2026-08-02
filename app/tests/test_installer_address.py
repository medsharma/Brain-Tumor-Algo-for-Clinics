"""The setup program must download from the server that handed it out.

The bug this pins is not subtle and it shipped. ``BrainMRITriageSetup.exe`` is
compiled once, on one machine, with a server address baked into it at build
time. The build that went out carried ``http://192.168.1.83:8765`` -- a private
address on the developer's home network.

Give that file to a clinic on another network and it does exactly what it was
told: it tries to fetch 2.4 GB from 192.168.1.83, finds nothing there, and stops.
To the person at the laptop the download is simply broken, with no clue that the
address is the problem and no way to correct it from the interface.

So the running app now stamps the address the browser actually reached it on
onto the end of the exe, and the installer reads it back out of its own file.
These tests hold that contract from both ends, because the two halves are in
different files and a rename on one side would otherwise fail silently at a
clinic rather than here.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import downloads

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_GUI = REPO_ROOT / "app" / "packaging" / "setup_gui.py"


@pytest.fixture
def client(engine):
    import app.server as server

    server.set_engine(engine)
    with TestClient(server.create_app(), client=("127.0.0.1", 50002)) as c:
        yield c
    server.set_engine(None)


# ------------------------------------------------------------------ the stamp

def test_stamp_appends_a_readable_address():
    out = downloads.stamp_installer(b"PRETEND EXE BYTES", "https://clinic.example")
    assert out.startswith(b"PRETEND EXE BYTES"), "the program itself must be untouched"
    assert b"https://clinic.example" in out


def test_stamping_twice_does_not_stack_footers():
    """Re-serving an already-stamped file must not leave two addresses in it.

    ``footer_server`` searches backwards and would find the newest, so a stack
    would still work. It is rejected anyway: a file carrying two different
    server addresses is a thing nobody can reason about while debugging a failed
    install in a clinic.
    """
    once = downloads.stamp_installer(b"EXE", "http://one.example")
    twice = downloads.stamp_installer(once, "http://two.example")

    assert twice.count(downloads.INSTALLER_FOOTER_MAGIC) == 1
    assert b"one.example" not in twice
    assert b"two.example" in twice
    assert twice.startswith(b"EXE")


def test_trailing_slash_is_not_carried_into_the_footer():
    """The installer joins paths onto this with an f-string, not urljoin.

    A trailing slash produces ``.../api//downloads``, which some servers accept
    and some do not. Cheaper to never emit it.
    """
    out = downloads.stamp_installer(b"EXE", "http://example.test:8765/")
    assert b"http://example.test:8765:MRITRIAGE-END" in out


# -------------------------------------------------------- the two halves agree

def test_markers_match_between_server_and_installer():
    """The stamp is written in one file and read in another.

    Nothing at import time connects them, so a rename would break installs in
    the field while every test still passed. This is that connection.
    """
    source = SETUP_GUI.read_text(encoding="utf-8")

    magic = re.search(r'FOOTER_MAGIC\s*=\s*b"([^"]+)"', source)
    end = re.search(r'FOOTER_END\s*=\s*b"([^"]+)"', source)
    assert magic and end, "setup_gui.py no longer declares its footer markers"

    assert magic.group(1).encode() == downloads.INSTALLER_FOOTER_MAGIC
    assert end.group(1).encode() == downloads.INSTALLER_FOOTER_END


def test_installer_still_prefers_an_explicit_address():
    """A URL on the command line is the escape hatch and must outrank the stamp.

    When the stamped address is wrong -- served through a proxy, say, or behind
    a name the clinic cannot resolve -- typing the right one has to work. That
    is the only recovery path that does not involve a rebuild.
    """
    namespace = _load_setup_gui_without_tk()
    assert namespace["server_url"](["setup.exe", "https://typed.example"]) == "https://typed.example"


def test_installer_reads_the_stamp_from_its_own_file(tmp_path):
    fake_exe = tmp_path / "BrainMRITriageSetup.exe"
    fake_exe.write_bytes(downloads.stamp_installer(b"X" * 5000, "http://stamped.example:8765"))

    namespace = _load_setup_gui_without_tk(own_file=str(fake_exe))
    assert namespace["footer_server"]() == "http://stamped.example:8765"
    assert namespace["server_url"](["setup.exe"]) == "http://stamped.example:8765"


def test_a_corrupted_footer_is_ignored_rather_than_fetched(tmp_path):
    """Fail safe. A damaged tail must not become an address this thing downloads from."""
    fake_exe = tmp_path / "BrainMRITriageSetup.exe"
    fake_exe.write_bytes(
        b"X" * 5000
        + downloads.INSTALLER_FOOTER_MAGIC
        + b"file:///C:/Windows/System32"
        + downloads.INSTALLER_FOOTER_END
    )

    namespace = _load_setup_gui_without_tk(own_file=str(fake_exe))
    assert namespace["footer_server"]() == ""


def test_no_footer_falls_back_instead_of_crashing(tmp_path):
    """Running the exe straight out of the release folder has to still work."""
    fake_exe = tmp_path / "BrainMRITriageSetup.exe"
    fake_exe.write_bytes(b"X" * 5000)

    namespace = _load_setup_gui_without_tk(own_file=str(fake_exe))
    assert namespace["footer_server"]() == ""
    assert namespace["server_url"](["setup.exe"]) == "http://built-in.example"


# --------------------------------------------------------------- over the wire

def _installer_present() -> bool:
    return downloads.find_windows_installer() is not None


needs_installer = pytest.mark.skipif(
    not _installer_present(),
    reason="no BrainMRITriageSetup.exe built on this machine",
)


@needs_installer
def test_served_installer_carries_the_callers_address(client):
    r = client.get("/api/downloads/BrainMRITriageSetup.exe",
                   headers={"host": "192.168.7.7:8765"})
    assert r.status_code == 200

    body = r.content
    start = body.rfind(downloads.INSTALLER_FOOTER_MAGIC)
    assert start != -1, "the served installer has no address stamped into it"
    stop = body.find(downloads.INSTALLER_FOOTER_END, start)
    address = body[start + len(downloads.INSTALLER_FOOTER_MAGIC):stop].decode()

    assert address == "http://192.168.7.7:8765"
    assert b"192.168.1.83" not in body[start:], "a build-time address leaked into the footer"


@needs_installer
def test_two_callers_get_their_own_address(client):
    def address_for(host: str) -> str:
        body = client.get("/api/downloads/BrainMRITriageSetup.exe",
                          headers={"host": host}).content
        start = body.rfind(downloads.INSTALLER_FOOTER_MAGIC)
        stop = body.find(downloads.INSTALLER_FOOTER_END, start)
        return body[start + len(downloads.INSTALLER_FOOTER_MAGIC):stop].decode()

    assert address_for("10.0.0.5:8765") == "http://10.0.0.5:8765"
    assert address_for("127.0.0.1:8765") == "http://127.0.0.1:8765"


@needs_installer
def test_listed_checksum_matches_the_bytes_that_caller_receives(client):
    """The page tells people to verify the download. That has to be checkable.

    Stamping changes the file, so a checksum computed over the unstamped exe
    would fail for every single person who followed the instructions. Worse than
    useless: it teaches clinics that a failed checksum is normal.
    """
    headers = {"host": "10.1.2.3:8765"}

    listed = client.get("/api/downloads", headers=headers).json()["windows_installer"]
    body = client.get("/api/downloads/BrainMRITriageSetup.exe", headers=headers).content

    assert hashlib.sha256(body).hexdigest() == listed["sha256"]
    assert len(body) == listed["bytes"]


# ------------------------------------------------------------------- utilities

def _load_setup_gui_without_tk(own_file: str = "setup.exe") -> dict:
    """Execute setup_gui.py with tkinter stubbed and the placeholder resolved.

    The module imports tkinter at the top and builds a window in ``main``. The
    address logic is plain functions with no window behind them, and it is worth
    testing on machines with no display, so tkinter is replaced with an empty
    module rather than required.
    """
    import sys
    import types

    for name in ("tkinter", "tkinter.ttk"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["tkinter"].ttk = sys.modules["tkinter.ttk"]

    source = SETUP_GUI.read_text(encoding="utf-8").replace(
        "@@SERVER_URL@@", "http://built-in.example"
    )
    namespace: dict = {"__file__": own_file, "__name__": "setup_gui_under_test"}
    exec(compile(source, str(SETUP_GUI), "exec"), namespace)
    return namespace
