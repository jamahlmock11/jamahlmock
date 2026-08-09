#!/usr/bin/env bash
# Start the V6 bot in a continuous scan loop (survives until killed).
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 run.py --loop --execute "$@"
