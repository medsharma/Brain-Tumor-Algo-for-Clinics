#!/usr/bin/env bash
# ===========================================================================
#  Brain MRI Triage - install and run  (macOS and Linux)
#
#  Run it:   bash app/install_and_run.sh
#
#  It installs what the app needs the first time, then starts the app and
#  opens it in your browser. After the first run it just starts the app.
#
#  It needs the internet ONCE, for the first install only. After that the app
#  never uses the internet again.
# ===========================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
VENV="$REPO_ROOT/.venv"

echo
echo "  Brain MRI Triage"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "  Python 3 is not installed."
  echo "  Install Python 3.10 or newer, then run this again."
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "  First run. Setting up. This takes a few minutes and needs the internet."
  echo "  You only have to do this once."
  echo
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip --quiet
  "$VENV/bin/python" -m pip install -r app/requirements.txt
  echo
  echo "  Setup finished. The app will not need the internet again."
  echo
fi

exec "$VENV/bin/python" -m app "$@"
