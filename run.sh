#!/usr/bin/env bash
# MedHarness desktop launcher — Linux/macOS/POSIX. No Docker, no venv required.
# End-user experience: double-click / run once. The browser window opens itself.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m app.main
