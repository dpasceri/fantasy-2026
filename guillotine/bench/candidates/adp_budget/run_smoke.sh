#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): small tuning-seed fit, then the 1,000 smoke paths.
# Writes bench/results/adp_budget.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/adp_budget/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.candidates.adp_budget.run --mode smoke
