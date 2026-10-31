#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/availability_hazard.smoke.json and the track's smoke
# baseline bench/results/_baseline_availability.smoke.json. Run it through the slot queue:
#   bench/cpuq -n 1 -- bash bench/candidates/availability_hazard/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=availability_hazard
.venv/bin/python -m bench.harness.tracks.availability --key _baseline_availability --mode smoke \
    --candidate bench.harness.tracks.availability:Baseline --variant baseline \
    --notes "core sim availability: 1 - P_OUT (Q 0.25, D 0.8) this week, flat 2%/wk hazard, geometric mean-3 absences ahead"
.venv/bin/python -m bench.harness.tracks.availability --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.models:Candidate" --variant cells --variant hazard --variant gbm \
    --notes "availability_hazard: EB cells / discrete-time onset+return hazard recursion / direct LightGBM"
exec .venv/bin/python -m bench.harness.compare availability --mode smoke
