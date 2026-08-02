"""Installing a second time, on top of a copy that is already running.

This is not an edge case, it is the normal path. The setup program ends by
starting the app, so from the first install onwards there is always a copy of
the app running when somebody runs the setup again. Windows keeps the files of
a running program locked, so unpacking on top of it stopped partway through
with::

    PermissionError: [Errno 13] Permission denied:
    C:\\Users\\...\\BrainMRITriage\\BrainMRITriage\\_internal\\_asyncio.pyd

Two separate failures in that. The person is told nothing they can act on, and
it happens *after* the whole download, leaving the install as a mix of two
versions of an app that reads brain scans.

These tests pin the fix from the outside: close it first, refuse to write if it
will not close, and say so in words a clinic can act on.
"""
from __future__ import annotations

import hashlib
import os
import queue
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_GUI = REPO_ROOT / "app" / "packaging" / "setup_gui.py"


# ------------------------------------------------------------------- fixtures

@pytest.fixture
def gui() -> dict:
    """setup_gui.py executed with tkinter stubbed out.

    The module builds a window in ``main``; everything tested here is plain
    methods with no window behind them, and it must be testable on a machine
    with no display.
    """
    import sys
    import types

    for name in ("tkinter", "tkinter.ttk"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["tkinter"].ttk = sys.modules["tkinter.ttk"]

    source = SETUP_GUI.read_text(encoding="utf-8").replace(
        "@@SERVER_URL@@", "http://built-in.example")
    namespace: dict = {"__file__": "setup.exe", "__name__": "setup_gui_under_test"}
    exec(compile(source, str(SETUP_GUI), "exec"), namespace)
    return namespace


@pytest.fixture
def setup(gui, monkeypatch):
    """A Setup instance with no window, and no real waiting."""
    instance = object.__new__(gui["Setup"])
    instance.messages = queue.Queue()
    instance.server = "http://example.test"
    instance.failed = False

    monkeypatch.setattr(gui["time"], "sleep", lambda _seconds: None)
    monkeypatch.setattr(gui["os"], "name", "nt")
    return instance


def _zip_of(path: Path, names: tuple[str, ...]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, b"contents")
    return path


# ----------------------------------------------------- closing the running app

def test_a_running_copy_is_closed_before_any_file_is_written(gui, setup, monkeypatch):
    calls: list[list[str]] = []
    running = {"yes": True}

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        running["yes"] = False          # taskkill worked
        return _completed()

    monkeypatch.setattr(gui["subprocess"], "run", fake_run)
    monkeypatch.setattr(type(setup), "_app_is_running", lambda _self: running["yes"])

    setup._stop_running_app()

    assert calls, "a running app was found and nothing was done about it"
    assert calls[0][0] == "taskkill"
    assert gui["EXE_NAME"] in calls[0]
    # Asked before forced. /F on the first attempt would be needless.
    assert "/F" not in calls[0]


def test_nothing_is_killed_when_nothing_is_running(gui, setup, monkeypatch):
    """Every first install takes this path. It must not shell out at all."""
    calls: list[list[str]] = []
    monkeypatch.setattr(gui["subprocess"], "run",
                        lambda command, **_k: calls.append(list(command)) or _completed())
    monkeypatch.setattr(type(setup), "_app_is_running", lambda _self: False)

    setup._stop_running_app()

    assert calls == []


def test_it_refuses_to_write_when_the_app_will_not_close(gui, setup, monkeypatch):
    """Stop before the first file, not halfway through the install.

    A half-replaced install is one version's code against another version's
    weights, and it starts up looking perfectly normal.
    """
    monkeypatch.setattr(gui["subprocess"], "run", lambda _c, **_k: _completed())
    monkeypatch.setattr(type(setup), "_app_is_running", lambda _self: True)

    with pytest.raises(RuntimeError) as raised:
        setup._stop_running_app()

    message = str(raised.value)
    assert "Close it" in message
    assert "run this setup again" in message
    assert "Nothing has been changed" in message


def test_the_check_survives_a_machine_without_tasklist(gui, setup, monkeypatch):
    """A locked-down laptop may not let this run. It must not become the error."""
    def explode(*_args, **_kwargs):
        raise OSError("tasklist is not available")

    monkeypatch.setattr(gui["subprocess"], "run", explode)
    assert setup._app_is_running() is False
    setup._stop_running_app()           # must not raise


# --------------------------------------------------------- the locked-file message

def test_a_locked_file_is_explained_in_words_a_clinic_can_act_on(gui, setup, tmp_path,
                                                                 monkeypatch):
    archive = _zip_of(tmp_path / "app.zip", ("BrainMRITriage/_internal/_asyncio.pyd",))

    def refuse(_self, _member, _target):
        raise PermissionError(
            13, "Permission denied",
            str(tmp_path / "BrainMRITriage" / "_internal" / "_asyncio.pyd"))

    monkeypatch.setattr(zipfile.ZipFile, "extract", refuse)

    with pytest.raises(RuntimeError) as raised:
        setup._unpack(archive, tmp_path / "install")

    message = str(raised.value)
    assert "_asyncio.pyd" in message
    assert "Brain MRI Triage" in message
    assert "Close it" in message or "Close it (" in message
    # The install is now a mixture of two versions. Say so.
    assert "incomplete" in message
    assert "Do not use the app" in message
    # Whatever else it says, it must not say this.
    assert "Errno" not in message
    assert "Traceback" not in message


def test_an_ordinary_unpack_still_works(gui, setup, tmp_path):
    archive = _zip_of(tmp_path / "app.zip",
                      ("BrainMRITriage/BrainMRITriage.exe", "BrainMRITriage/data.bin"))
    target = tmp_path / "install"

    setup._unpack(archive, target)

    assert (target / "BrainMRITriage" / "BrainMRITriage.exe").is_file()
    assert setup._find_exe(target) == target / "BrainMRITriage" / "BrainMRITriage.exe"


# ------------------------------------------------------- not downloading it twice

def test_a_download_already_here_and_verified_is_not_fetched_again(gui, setup, tmp_path,
                                                                   monkeypatch):
    """The retry after a failed install must not re-pull 600 MB over a clinic link."""
    body = b"the windows package" * 100
    already = tmp_path / "_download.zip"
    already.write_bytes(body)

    def must_not_be_called(*_args, **_kwargs):
        raise AssertionError("it downloaded a file it already had")

    monkeypatch.setattr(gui["urllib"].request, "urlopen", must_not_be_called)

    setup._download("http://example.test/pkg", already,
                    expected=hashlib.sha256(body).hexdigest(), total=len(body))

    assert already.read_bytes() == body


def test_a_leftover_file_that_does_not_match_is_thrown_away(gui, setup, tmp_path,
                                                            monkeypatch):
    """A truncated leftover must never be unpacked, and never be trusted."""
    stale = tmp_path / "_download.zip"
    stale.write_bytes(b"half a download")

    opened: list[str] = []

    class FakeResponse:
        headers = {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self, _size):
            if opened.count("read"):
                return b""
            opened.append("read")
            return b"good"

    monkeypatch.setattr(gui["urllib"].request, "urlopen",
                        lambda url, **_k: opened.append(url) or FakeResponse())

    setup._download("http://example.test/pkg", stale,
                    expected=hashlib.sha256(b"good").hexdigest(), total=4)

    assert stale.read_bytes() == b"good"


# ------------------------------------------------------------------- utilities

class _completed:
    """Stand-in for subprocess.CompletedProcess."""

    returncode = 0
    stdout = b""
    stderr = b""


# --------------------------------------------------------- removing it again

def test_the_uninstall_script_removes_the_app_and_its_shortcuts(gui, tmp_path):
    """A clinic laptop that cannot uninstall software keeps it forever."""
    script = gui["_UNINSTALL_SCRIPT"].format(
        exe_name_stem="BrainMRITriage",
        app_dir=str(tmp_path / "BrainMRITriage"),
        zip_path=str(tmp_path / "_download.zip"),
        script_path=str(tmp_path / "uninstall.ps1"),
        audit_dir=str(tmp_path / "audit"),
        app_name="Brain MRI Triage",
        registry_key=gui["UNINSTALL_KEY"],
    )

    assert "Stop-Process" in script, "a running app locks its own files"
    assert str(tmp_path / "BrainMRITriage") in script
    assert "Brain MRI Triage.lnk" in script
    assert "HKCU:" in script


def test_the_uninstaller_never_deletes_the_audit_log(gui, tmp_path):
    """It is the clinic's record of what this tool said about which scan.

    It may be the only trace that a patient was ever triaged by this app. No
    uninstaller should quietly delete a clinical record, so the script names
    where the log is and leaves it there.
    """
    audit = tmp_path / "audit"
    script = gui["_UNINSTALL_SCRIPT"].format(
        exe_name_stem="BrainMRITriage",
        app_dir=str(tmp_path / "BrainMRITriage"),
        zip_path=str(tmp_path / "_download.zip"),
        script_path=str(tmp_path / "uninstall.ps1"),
        audit_dir=str(audit),
        app_name="Brain MRI Triage",
        registry_key=gui["UNINSTALL_KEY"],
    )

    removals = [line for line in script.splitlines()
                if line.strip().startswith("Remove-Item")]
    assert removals, "the script removes nothing at all"
    for line in removals:
        assert str(audit) not in line, (
            f"the uninstaller deletes the audit log: {line}")
    assert str(audit) in script, "it must at least say where the log was left"


def test_the_uninstall_entry_is_per_user(gui):
    """No administrator to install, so none to remove it either."""
    assert gui["UNINSTALL_KEY"].startswith(r"Software\Microsoft\Windows")
    assert "WOW6432Node" not in gui["UNINSTALL_KEY"]
