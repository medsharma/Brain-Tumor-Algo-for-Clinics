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
import tkinter as tk
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from tkinter import ttk

#: Where the payload comes from. Written in at build time by
#: app/packaging/build_installer.py, which reads MRI_TRIAGE_SERVER_URL.
SERVER_URL = "@@SERVER_URL@@"

APP_NAME = "Brain MRI Triage"
FOLDER_NAME = "BrainMRITriage"


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
            text=("Research prototype. Not a diagnosis and not approved for "
                  "clinical use anywhere."),
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

        self.say("Installing", 88, "Unpacking the application")
        self._unpack(zip_path, target)
        zip_path.unlink(missing_ok=True)

        exe = self._find_exe(target)
        self.say("Making a shortcut", 95)
        self._make_shortcut(exe)

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
                f"Could not reach {self.server}. It may be switched off, or this "
                f"machine may not be on the same network. ({exc.reason})") from exc

    def _download(self, url: str, destination: Path, expected: str | None,
                  total: int) -> None:
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

    def _unpack(self, zip_path: Path, target: Path) -> None:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            for i, member in enumerate(members, 1):
                zf.extract(member, target)
                if i % 200 == 0:
                    self.say("Installing", 88 + 6.0 * i / len(members),
                             f"{i:,} of {len(members):,} files")

    def _find_exe(self, target: Path) -> Path:
        direct = target / FOLDER_NAME / "BrainMRITriage.exe"
        if direct.is_file():
            return direct
        found = next(target.rglob("BrainMRITriage.exe"), None)
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

    def _launch(self, exe: Path) -> None:
        subprocess.Popen([str(exe)], cwd=str(exe.parent),
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def main() -> int:
    server = SERVER_URL
    if len(sys.argv) > 1:
        server = sys.argv[1]
    if server.startswith("@@"):
        server = "http://127.0.0.1:8765"

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
