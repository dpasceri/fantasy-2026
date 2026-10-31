#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core; ~35 s of it builds the smoke reference the first time):
# writes bench/results/_baseline_mc.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/_baseline_mc/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.mc --key _baseline_mc --mode smoke \
    --candidate bench.candidates._baseline_mc.plain:PlainMC --variant baseline \
    --notes "plain MC (sim.py sampler, chunks of 8000 from rng([seed, ci]))"
