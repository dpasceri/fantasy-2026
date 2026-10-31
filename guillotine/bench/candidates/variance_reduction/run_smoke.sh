#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core, ~30 s): writes bench/results/variance_reduction.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/variance_reduction/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=variance_reduction
exec .venv/bin/python -m bench.harness.tracks.mc --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.candidate:VR" --variant cmc --variant cmc_strat --variant cmc_mix \
    --notes "conditional MC given copula factors: product-form team combinations (+ stratified inner draws, + player-level recombination)"
