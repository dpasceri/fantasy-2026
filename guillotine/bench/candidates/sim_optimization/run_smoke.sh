#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/sim_optimization.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/sim_optimization/run_smoke.sh
# Sanity checks on tuning seed 1000 (check.py), then the frozen variants in tuned.json on smoke seeds 100-104.
# It does not retune (tune.py does, on seeds >= 1000).
set -euo pipefail
cd "$(dirname "$0")/../../.."
.venv/bin/python -m bench.candidates.sim_optimization.check
exec .venv/bin/python -m bench.candidates.sim_optimization.evaluate --mode smoke
