#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/projection_ensemble.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/projection_ensemble/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=projection_ensemble
exec .venv/bin/python -m bench.harness.tracks.player --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.ensemble:ProjectionEnsemble" --variant lpool --variant recal --variant stack \
    --notes "forecast combination: linear-pool bucket blend; probit recalibration; stacked point-in-time signals"
