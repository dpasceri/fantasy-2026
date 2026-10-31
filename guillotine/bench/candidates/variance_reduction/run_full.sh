#!/usr/bin/env bash
# Full run (queued only; <= 15 min on 6 cores, ~7-8 min, ~2500 core-s): writes bench/results/variance_reduction.json. Enqueue:
#   bench/queue/pending/variance_reduction.job = {"key": "variance_reduction", "cmd": ["bash", "bench/candidates/variance_reduction/run_full.sh"], "timeout_s": 1200}
# Workers = CPUQ_SLOTS (the harness default); every call is single-threaded.
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=variance_reduction
exec .venv/bin/python -m bench.harness.tracks.mc --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.candidate:VR" --variant cmc --variant cmc_strat --variant cmc_mix \
    --notes "conditional MC given copula factors: product-form team combinations (+ stratified inner draws, + player-level recombination)"
