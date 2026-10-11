#!/usr/bin/env bash
# MedHarness desktop launcher — Linux/macOS.
# First run: creates a private .venv, installs the 5 requirements, starts the app.
# Later runs: starts instantly. No Docker, no global pip changes.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || {
  echo "[MedHarness] Python not found. Install Python 3.10+ from https://python.org"
  exit 1
}

if [ ! -x ".venv/bin/python" ]; then
  echo "[MedHarness] First run — creating a private Python environment (.venv)..."
  if ! "$PY" -m venv .venv 2>/dev/null; then
    echo "[MedHarness] Could not create a venv (Debian/Ubuntu: sudo apt install python3-venv python3-pip)"
    echo "[MedHarness] Or run manually:  pip install --user -r requirements.txt && python3 -m app.main"
    exit 1
  fi
  .venv/bin/pip install --quiet --upgrade pip
fi

if ! .venv/bin/python -c "import fastapi, uvicorn, httpx, pydantic, h2" >/dev/null 2>&1; then
  echo "[MedHarness] Installing requirements (one time only)..."
  .venv/bin/pip install --quiet -r requirements.txt
fi

echo "[MedHarness] Starting the desktop app at http://127.0.0.1:8787 ..."
exec .venv/bin/python -m app.main
