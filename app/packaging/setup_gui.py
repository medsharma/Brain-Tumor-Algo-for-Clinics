"""The setup program a clinic actually double-clicks.

One small .exe. Run it and it downloads the application, installs it, makes a
desktop shortcut and opens the app. Nothing is asked of the person running it.

Why this exists at all
----------------------
The previous best route was: download a 2.4 GB zip, right-click, Extract All,
find the folder, double-click the exe inside it. Every one of those steps is a
place where somebody who does not use computers for a living stops and asks a
colleague. "Extract All" in particular gets skipped constantly, and Windows will
happily let you run an exe from *inside* a zip preview, where it fails in a
confusing way because none of its data files are there.

So this replaces all of it with: download one file, double-click, wait, use.

Design notes
------------
**It is deliberately tiny.** It contains no model and no PyTorch. It is a few
megabytes and downloads the rest, which means the person sees a progress bar
early instead of staring at a browser download for ten minutes.

**It installs per-user, into LOCALAPPDATA.** No administrator rights, no UAC
prompt, no "ask your IT department". A clinic laptop is often a locked-down
machine and this has to work on one.

**It is resumable in the way that matters.** If the download dies halfway, the
part-file is discarded and the next run starts again cleanly rather than
unpacking a truncated zip. A corrupt install that half-works is worse than one
that obviously failed.

**The window stays open on failure**, with the reason in plain words. The
failure mode this replaces is a console flashing shut with an unread traceback.

**It closes the running copy before it installs over it.** The end of every
successful install starts the app, so the second time anybody runs this setup
there is a copy of the app already running. Windows keeps the files of a running
program locked, so unpacking on top of it died with

    PermissionError: [Errno 13] Permission denied:
    ...BrainMRITriage\\_internal\\_asyncio.pyd

which tells the person nothing they can act on and, worse, only happened after
the whole download had finished. See ``_stop_running_app``.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from tkinter import ttk

#: Fallback address, written in at build time by
#: app/packaging/build_installer.py, which reads MRI_TRIAGE_SERVER_URL.
#:
#: This is only a fallback. The address actually used normally comes from the
#: footer the server appends when it hands this file out. See ``server_url``.
SERVER_URL = "@@SERVER_URL@@"

APP_NAME = "Brain MRI Triage"
FOLDER_NAME = "BrainMRITriage"
EXE_NAME = "BrainMRITriage.exe"

#: How long to wait for Windows to let go of a closed program's files. Killing
#: a process returns before its file handles are released, and unpacking into
#: that gap fails exactly the same way as not killing it at all.
RELEASE_TIMEOUT_SECONDS = 12

#: Anything that runs a Windows command line here does so without flashing a
#: console window over the setup window.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

#: Marker for the address the serving app appends to this exe on download.
#:
#: A build-time address cannot be right for everybody. The exe is built once, on
#: one machine, and then handed to clinics on other networks entirely. Whatever
#: address was correct at build time ("http://192.168.1.83:8765", say) is not an
#: address that resolves to anything on a clinic laptop in another country, so
#: the installer would sit there trying to download 2.4 GB from a machine that
#: is not there. That shipped once and it is the reason this exists.
#:
#: So the server appends the address the browser actually reached it on, and the
#: installer reads it back out of its own file. A compiled binary cannot be
#: rewritten per request, but bytes can be stuck on the end of one: PyInstaller
#: finds its archive by searching backwards from the end of the file, so a
#: trailing block is ignored by the bootloader and the exe still runs.
FOOTER_MAGIC = b"MRITRIAGE-SERVER-V1:"
FOOTER_END = b":MRITRIAGE-END"

#: How much of the tail to read looking for the footer. The footer is written
#: last, so it is within a few hundred bytes of the end.
FOOTER_SEARCH_BYTES = 4096


def footer_server() -> str:
    """The address this file was downloaded from, if the server wrote one in.

    Returns "" when there is no footer, which is the case when somebody runs
    the exe straight out of the ``release`` folder rather than downloading it.
    """
    try:
        own = Path(sys.executable if getattr(sys, "frozen", False) else __file__)
        with open(own, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - FOOTER_SEARCH_BYTES))
            tail = fh.read()
    except OSError:
        return ""

    start = tail.rfind(FOOTER_MAGIC)
    if start == -1:
        return ""
    stop = tail.find(FOOTER_END, start)
    if stop == -1:
        return ""
    url = tail[start + len(FOOTER_MAGIC):stop].decode("utf-8", "replace").strip()
    # Only ever a plain http(s) address. Anything else is a corrupted tail, and
    # a corrupted tail must not become something this program tries to fetch.
    if not url.startswith(("http://", "https://")):
        return ""
    return url


def server_url(argv: list[str]) -> str:
    """Where to download from, most trustworthy source first.

    1. An address typed on the command line. The explicit escape hatch, so it
       always wins.
    2. The footer the serving app wrote in. Correct by construction: it is the
       address the browser just used to fetch this file.
    3. The build-time address. Right only on the network it was built for.
    4. Localhost, for a developer running both halves on one machine.
    """
    if len(argv) > 1 and argv[1].strip():
        return argv[1].strip()

    from_footer = footer_server()
    if from_footer:
        return from_footer

    if SERVER_URL and not SERVER_URL.startswith("@@"):
        return SERVER_URL

    return "http://127.0.0.1:8765"


#: Where Windows looks for per-user installed programs. Per-user, so removing
#: the app needs no administrator either.
UNINSTALL_KEY = (r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
                 r"\BrainMRITriage")

#: Run by "Add or remove programs". Deletes the application and its shortcuts,
#: and deliberately leaves the audit log alone: it is the clinic's record of
#: what this tool said about which scan, and an uninstaller must not quietly
#: delete a clinical record.
_UNINSTALL_SCRIPT = r"""# Removes {app_name}. Written by the setup program.
$ErrorActionPreference = 'SilentlyContinue'

# The app locks its own files while it runs.
Stop-Process -Name '{exe_name_stem}' -Force
Start-Sleep -Seconds 2

foreach ($place in @(
    [Environment]::GetFolderPath('Desktop'),
    (Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs'))) {{
  if (-not [string]::IsNullOrWhiteSpace($place)) {{
    Remove-Item -LiteralPath (Join-Path $place '{app_name}.lnk') -Force
  }}
}}

Remove-Item -LiteralPath '{app_dir}' -Recurse -Force
Remove-Item -LiteralPath '{zip_path}' -Force
Remove-Item -LiteralPath 'HKCU:\{registry_key}' -Recurse -Force

# NOT removed: the audit log. One line per scan this tool ever read, which may
# be the only record that a patient was triaged by it. Delete it yourself, on
# purpose, when your clinic's retention policy says to:
#
#   {audit_dir}

Remove-Item -LiteralPath '{script_path}' -Force
"""


def _installed_version(exe: Path) -> str:
    """Version string for the uninstall entry, best effort."""
    config = exe.parent / "deployment_config.json"
    try:
        raw = json.loads(config.read_text(encoding="utf-8"))
        return str(raw.get("config_version") or raw.get("created_utc") or "")[:32] or "1.0"
    except Exception:  # noqa: BLE001 - a missing version is not a failure
        return "1.0"


def install_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / FOLDER_NAME


class Setup:
    """Downloads, installs, then launches. All of it on a worker thread."""

    def __init__(self, root: tk.Tk, server: str) -> None:
        self.root = root
        self.server = server.rstrip("/")
        self.messages: "queue.Queue[tuple]" = queue.Queue()
        self.failed = False

        root.title(f"{APP_NAME} Setup")
        root.geometry("560x260")
        root.resizable(False, False)

        frame = ttk.Frame(root, padding=22)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=APP_NAME, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Setting up. This takes a few minutes and needs the internet once.",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 16))

        self.status = ttk.Label(frame, text="Starting", font=("Segoe UI", 10))
        self.status.pack(anchor="w")

        self.bar = ttk.Progressbar(frame, length=500, mode="determinate", maximum=100)
        self.bar.pack(anchor="w", pady=(8, 4))

        self.detail = ttk.Label(frame, text="", font=("Segoe UI", 8), foreground="#555")
        self.detail.pack(anchor="w")

        self.button = ttk.Button(frame, text="Close", command=root.destroy)
        self.button.pack(anchor="e", pady=(14, 0))
        self.button.state(["disabled"])

        # Not a formality. Somebody will use this on a real scan on day one.
        ttk.Label(
            frame,
            text=("Development build. A second opinion, not a diagnosis. Never "
                  "the only basis for a decision about a patient."),
            font=("Segoe UI", 8),
            foreground="#b3261e",
            wraplength=500,
        ).pack(anchor="w", pady=(12, 0))

        threading.Thread(target=self._work_safely, daemon=True).start()
        self.root.after(80, self._drain)

    # -- talking to the UI from the worker thread -------------------------
    def say(self, text: str, percent: float | None = None, detail: str = "") -> None:
        self.messages.put(("status", text, percent, detail))

    def _drain(self) -> None:
        try:
            while True:
                kind, *rest = self.messages.get_nowait()
                if kind == "status":
                    text, percent, detail = rest
                    self.status.config(text=text)
                    if percent is not None:
                        self.bar.config(value=percent)
                    self.detail.config(text=detail)
                elif kind == "done":
                    self.status.config(text=rest[0])
                    self.bar.config(value=100)
                    self.button.state(["!disabled"])
                elif kind == "failed":
                    self.failed = True
                    self.status.config(text="Setup did not finish", foreground="#b3261e")
                    self.detail.config(text=rest[0], wraplength=500)
                    self.button.state(["!disabled"])
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    # -- the actual work --------------------------------------------------
    def _work_safely(self) -> None:
        try:
            self._work()
        except Exception as exc:  # noqa: BLE001 - the window must show anything
            self.messages.put(("failed", f"{type(exc).__name__}: {exc}"))

    def _work(self) -> None:
        target = install_root()
        target.mkdir(parents=True, exist_ok=True)

        self.say("Checking what to download", 2)
        manifest = self._json(f"{self.server}/api/downloads")
        package = manifest.get("windows_package")
        if not package:
            raise RuntimeError(
                "That server is not offering a ready-built Windows app. Ask "
                "whoever runs it to build one, or use the Python setup instead.")

        zip_path = target / "_download.zip"
        self._download(
            f"{self.server}/api/downloads/{package['key']}",
            zip_path,
            expected=package.get("sha256"),
            total=int(package.get("bytes") or 0),
        )

        # Before a single file is written, not after. A half-unpacked install is
        # a mix of two versions of the app, and this one has model weights in it.
        self._stop_running_app()

        self.say("Installing", 88, "Unpacking the application")
        self._unpack(zip_path, target)

        # Only on success. If the unpack failed, the zip is worth keeping: it is
        # hundreds of megabytes over whatever link this clinic has, and the next
        # run verifies it and skips the download.
        zip_path.unlink(missing_ok=True)

        exe = self._find_exe(target)
        self.say("Making a shortcut", 95)
        self._make_shortcut(exe)
        self._register_uninstaller(target, exe)

        self.say("Starting the app", 99, "The first start takes about 20 seconds")
        self._launch(exe)

        self.messages.put(("done", "Done. The app is opening in your browser."))

    # -- steps ------------------------------------------------------------
    def _json(self, url: str) -> dict:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach {self.server}.\n\n"
                f"({exc.reason})\n\n"
                "The computer serving the app may be switched off, or this "
                "machine may not be able to see it. If you were given a "
                "different address, run this setup from a command prompt with "
                "that address after it, for example:\n\n"
                "    BrainMRITriageSetup.exe https://example.org\n\n"
                "Otherwise ask whoever sent you this file for the address."
            ) from exc

    def _download(self, url: str, destination: Path, expected: str | None,
                  total: int) -> None:
        # A finished download left over from a run that failed later on. Hashing
        # it costs a few seconds; downloading it again costs a clinic an hour.
        if destination.is_file() and expected:
            self.say("Checking the file already downloaded", 3,
                     "This takes a few seconds")
            if self._sha256_of(destination) == expected:
                self.say("Already downloaded and verified", 85)
                return
            destination.unlink(missing_ok=True)

        digest = hashlib.sha256()
        done = 0
        partial = destination.with_suffix(".part")
        partial.unlink(missing_ok=True)

        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or total or 0)
            with open(partial, "wb") as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
                    done += len(block)
                    percent = (85.0 * done / total) if total else 40.0
                    self.say(
                        "Downloading the application",
                        max(3.0, percent),
                        f"{done / 1e6:,.0f} MB of {total / 1e6:,.0f} MB",
                    )

        if expected and digest.hexdigest() != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                "The download was damaged in transit and has been deleted. "
                "Run this setup again.")

        partial.replace(destination)

    @staticmethod
    def _sha256_of(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def _app_is_running(self) -> bool:
        """Is a copy of the app running right now?"""
        if os.name != "nt":
            return False
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {EXE_NAME}", "/NH"],
                capture_output=True, timeout=30, creationflags=_NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            # If the question cannot be asked, do not claim the answer is no.
            # The unpack below will report the real problem in plain words.
            return False
        return EXE_NAME.lower() in result.stdout.decode("utf-8", "replace").lower()

    def _stop_running_app(self) -> None:
        """Close a running copy so its files can be replaced.

        Every successful install ends by starting the app, so by the time
        anybody runs this setup a second time there is a copy already running.
        Windows keeps the files of a running program locked, and unpacking on
        top of one dies partway through with a "Permission denied" on some
        ``.pyd`` nobody has heard of. The person did nothing wrong and there is
        nothing in that message telling them what to do.

        Asked politely first, then forced. The app holds no unsaved work: a
        result lives in the browser until it is exported, and the person doing
        this has a setup window in front of them, not a scan.

        If it will not close, this stops before writing anything rather than
        leaving half of one version of the app on top of another.
        """
        if os.name != "nt" or not self._app_is_running():
            return

        self.say("Installing", 86, f"Closing the copy of {APP_NAME} already running")
        for command in (["taskkill", "/IM", EXE_NAME],
                        ["taskkill", "/F", "/IM", EXE_NAME]):
            try:
                subprocess.run(command, capture_output=True, timeout=30,
                               creationflags=_NO_WINDOW)
            except (OSError, subprocess.SubprocessError):
                break

            # Killing a process returns before Windows releases its file
            # handles. Unpacking into that gap fails just the same.
            deadline = time.monotonic() + RELEASE_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if not self._app_is_running():
                    time.sleep(1.0)
                    return
                time.sleep(0.4)

        if self._app_is_running():
            raise RuntimeError(
                f"{APP_NAME} is running on this computer and will not close, so "
                "the setup cannot replace its files.\n\n"
                "Close it yourself: look on the taskbar and in the notification "
                "area by the clock, or restart the computer. Then run this setup "
                "again. Nothing has been changed and the download is kept, so "
                "the next run is quick.")

    def _unpack(self, zip_path: Path, target: Path) -> None:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            for i, member in enumerate(members, 1):
                try:
                    zf.extract(member, target)
                except PermissionError as exc:
                    raise RuntimeError(self._locked_file_message(exc)) from exc
                if i % 200 == 0:
                    self.say("Installing", 88 + 6.0 * i / len(members),
                             f"{i:,} of {len(members):,} files")

    @staticmethod
    def _locked_file_message(exc: PermissionError) -> str:
        """Plain words for the one error this step actually produces."""
        name = Path(exc.filename).name if exc.filename else "one of its files"
        return (
            f"Windows would not let the setup replace {name}, which means "
            f"{APP_NAME} is still open on this computer.\n\n"
            "Close it (check the taskbar and the notification area by the "
            "clock), or restart the computer, then run this setup again. The "
            "download is kept, so the next run is quick.\n\n"
            "Until then this install is incomplete. Do not use the app."
        )

    def _find_exe(self, target: Path) -> Path:
        direct = target / FOLDER_NAME / EXE_NAME
        if direct.is_file():
            return direct
        found = next(target.rglob(EXE_NAME), None)
        if found is None:
            raise RuntimeError(
                "The application was downloaded but BrainMRITriage.exe is not in "
                "it. The download may be incomplete.")
        return found

    def _make_shortcut(self, exe: Path) -> None:
        """Desktop and Start Menu shortcuts, via PowerShell so nothing is needed.

        **Windows is asked where the Desktop is rather than being told.** The
        obvious version of this builds ``%USERPROFILE%\\Desktop`` and it is
        wrong on a very large share of real machines: with OneDrive backup
        turned on, the Desktop is redirected to ``%USERPROFILE%\\OneDrive\\
        Desktop`` and the old path does not exist at all. The first version of
        this installer did exactly that, found no such folder, and silently made
        no shortcut. ``[Environment]::GetFolderPath('Desktop')`` returns wherever
        it actually is.

        A Start Menu entry goes in too, because it survives someone tidying
        their desktop and it is where people look for an installed program.

        Best effort throughout. A missing shortcut is an inconvenience, not a
        reason to fail an install that otherwise worked, so this never raises.
        """
        try:
            target = str(exe).replace("'", "''")
            workdir = str(exe.parent).replace("'", "''")
            description = f"{APP_NAME} - research prototype, not for clinical use"
            script = "\n".join([
                "$ErrorActionPreference = 'SilentlyContinue'",
                "$shell = New-Object -COM WScript.Shell",
                "$places = @(",
                "  [Environment]::GetFolderPath('Desktop'),",
                "  (Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs')",
                ")",
                "foreach ($place in $places) {",
                "  if ([string]::IsNullOrWhiteSpace($place)) { continue }",
                "  if (-not (Test-Path $place)) { continue }",
                f"  $link = Join-Path $place '{APP_NAME}.lnk'",
                "  $s = $shell.CreateShortcut($link)",
                f"  $s.TargetPath = '{target}'",
                f"  $s.WorkingDirectory = '{workdir}'",
                f"  $s.Description = '{description}'",
                "  $s.Save()",
                "}",
            ])

            # Written to a file and run with -File, not passed to -Command.
            #
            # A multi-line script handed to `-Command` through subprocess gets
            # its quoting mangled on the way in: the single quotes around the
            # paths are eaten and PowerShell then tries to execute the first
            # word of a string as a cmdlet. That is not a hypothetical, it is
            # what the first two versions of this did, and because the failure
            # is swallowed below the only symptom was a missing shortcut.
            script_path = Path(os.environ.get("TEMP", ".")) / "_mri_shortcut.ps1"
            script_path.write_text(script, encoding="utf-8")
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive",
                     "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
                    capture_output=True, timeout=90,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            finally:
                script_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - never fail the install over a shortcut
            pass

    # -- removing it again -------------------------------------------------

    def _register_uninstaller(self, target: Path, exe: Path) -> None:
        """Put this in Add or remove programs, with something that works.

        A clinic laptop that cannot uninstall software is a laptop with 2.4 GB
        of a research prototype on it forever, and "delete the folder" is not an
        instruction anyone should have to be given for an app that installed
        itself with one click.

        **The audit log is deliberately not removed.** It is the clinic's record
        of what this tool said about which scan, it may be the only trace that a
        patient was ever triaged by it, and no uninstaller should quietly delete
        a clinical record. The script says where it is left.

        Best effort throughout. Failing to register an uninstaller is not a
        reason to fail an install that otherwise worked.
        """
        if os.name != "nt":
            return
        try:
            script = target / "uninstall.ps1"
            script.write_text(_UNINSTALL_SCRIPT.format(
                exe_name_stem=Path(EXE_NAME).stem,
                app_dir=str(target / FOLDER_NAME),
                zip_path=str(target / "_download.zip"),
                script_path=str(script),
                audit_dir=str(target / "audit"),
                app_name=APP_NAME,
                registry_key=UNINSTALL_KEY,
            ), encoding="utf-8")

            import winreg

            command = (f'powershell -NoProfile -ExecutionPolicy Bypass '
                       f'-File "{script}"')
            size_kb = sum(
                f.stat().st_size for f in (target / FOLDER_NAME).rglob("*")
                if f.is_file()) // 1024

            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
                for name, value in (
                    ("DisplayName", f"{APP_NAME} (research prototype)"),
                    ("DisplayVersion", _installed_version(exe)),
                    ("Publisher", "Brain MRI Triage project"),
                    ("InstallLocation", str(target)),
                    ("UninstallString", command),
                    ("QuietUninstallString", command),
                    ("DisplayIcon", str(exe)),
                    ("URLInfoAbout", ""),
                ):
                    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
                winreg.SetValueEx(key, "EstimatedSize", 0, winreg.REG_DWORD, size_kb)
                winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
        except Exception:  # noqa: BLE001 - never fail an install over this
            pass

    def _launch(self, exe: Path) -> None:
        subprocess.Popen([str(exe)], cwd=str(exe.parent),
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def main() -> int:
    server = server_url(sys.argv)

    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.3)
    except tk.TclError:
        pass
    Setup(root, server)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
