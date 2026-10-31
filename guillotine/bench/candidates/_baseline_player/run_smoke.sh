#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/_baseline_player.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/_baseline_player/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.player --key _baseline_player --mode smoke \
    --candidate bench.candidates._baseline_player.ratio:EmpiricalRatio --variant baseline \
    --notes "model.py marginal (bucketed actual/proj quantile tables, log-proj interpolation), refit per origin"
