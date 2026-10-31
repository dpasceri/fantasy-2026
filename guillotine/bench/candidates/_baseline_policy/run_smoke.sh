#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/_baseline_policy.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/_baseline_policy/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.policy --key _baseline_policy --mode smoke \
    --candidate bench.candidates._baseline_policy.multiplier:Multiplier \
    --variant baseline --variant m0.75 --variant x1.0 \
    --notes "dynamic.py rule: m x league-typical bid; baseline 1.0/0.75 (run.py's pick), m0.75 0.75/0.75, x1.0 1.0/1.0"
