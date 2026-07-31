"""Start the app. This is what the desktop shortcut runs.

Everything printed here is aimed at whoever is standing at the laptop, not at
a developer. If the app cannot start, the reason has to be readable and the
window has to stay open long enough to read it.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .core.config import ConfigError, StubConfigError
from .core.model import ModelLoadError
from .core.readiness import NotReadyError
from .core.version import APP_NAME, APP_VERSION
from .server import LOOPBACK_HOST, BindingRefused, run


def _safe_print(text: str) -> None:
    """Print without dying on an old Windows console codepage.

    The four call strings contain an em dash. On a cp1252 console that raises
    ``UnicodeEncodeError``, which would turn a readable error into a crash.
    """
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        stream.write(text.encode(encoding, errors="replace").decode(encoding) + "\n")
    stream.flush()


def _hold_window_open() -> None:
    """Stop a double-clicked shortcut from vanishing before the error is read."""
    if sys.platform == "win32" and sys.stdin is not None and sys.stdin.isatty():
        try:
            input("\nPress Enter to close this window. ")
        except (EOFError, KeyboardInterrupt):
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="brain-mri-triage",
        description=f"{APP_NAME} {APP_VERSION}. Offline brain MRI triage.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port to listen on. Defaults to the first free port from 8765.",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not open a browser window automatically.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print more detail while running.",
    )
    parser.add_argument(
        "--host", default=LOOPBACK_HOST,
        help=(
            "Address to listen on. Defaults to 127.0.0.1, which is this "
            "machine only. Anything else serves patient scans over the network "
            "and is refused unless MRI_TRIAGE_ALLOW_PUBLIC_BIND=1 is set. Use "
            "0.0.0.0 to accept connections from anywhere."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    _safe_print(f"\n  {APP_NAME} {APP_VERSION}")
    _safe_print("  Starting up. Loading the model takes a few seconds.\n")

    try:
        run(host=args.host, port=args.port, open_browser=not args.no_browser)
    except KeyboardInterrupt:
        _safe_print("\n  Stopped.")
        return 0
    except (NotReadyError, StubConfigError) as exc:
        _safe_print("\n" + str(exc))
        _safe_print(
            "\n  Nothing is wrong with your computer. The app is refusing to "
            "\n  give clinical answers it cannot stand behind. Show this message"
            "\n  to whoever installed the app.\n"
        )
        _hold_window_open()
        return 2
    except ModelLoadError as exc:
        _safe_print(f"\n  The app could not load its model.\n\n{exc}\n")
        _hold_window_open()
        return 3
    except ConfigError as exc:
        _safe_print(f"\n  The app's settings file is not usable.\n\n{exc}\n")
        _hold_window_open()
        return 4
    except BindingRefused as exc:
        _safe_print(f"\n  {exc}\n")
        _hold_window_open()
        return 5
    except OSError as exc:
        _safe_print(
            f"\n  The app could not start: {exc}\n"
            "\n  If it says the address is already in use, the app may already be"
            "\n  running. Check your browser tabs before starting it again.\n"
        )
        _hold_window_open()
        return 6

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
