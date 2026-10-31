#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/rival_lineup.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/rival_lineup/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.candidates.rival_lineup.run --mode smoke
