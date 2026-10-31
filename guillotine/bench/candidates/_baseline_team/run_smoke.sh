#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/_baseline_team.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/_baseline_team/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.team --key _baseline_team --mode smoke \
    --candidate bench.candidates._baseline_team.copula:CopulaSim --variant baseline \
    --notes "model.py marginal + one-factor Gaussian copula (team/game), refit per origin; 96000 CRN draws per week, p_out=0"
