#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/_baseline_availability.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/_baseline_availability/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.availability --key _baseline_availability --mode smoke \
    --candidate bench.harness.tracks.availability:Baseline --variant baseline \
    --notes "core sim availability: 1 - P_OUT (Q 0.25, D 0.8) this week, flat 2%/wk hazard, geometric mean-3 absences ahead"
