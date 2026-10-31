#!/usr/bin/env bash
# Full run (queued only, <= 15 min on 6 cores; ~7 min): builds the cached full reference
# (64M week paths + 16M season paths) if missing, then 160 baseline replications of each target.
# Writes bench/results/_baseline_mc.json. Enqueue:
#   bench/queue/pending/_baseline_mc.job = {"key": "_baseline_mc", "cmd": ["bash", "bench/candidates/_baseline_mc/run_full.sh"], "timeout_s": 1200}
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.mc --key _baseline_mc --mode full --make-ref \
    --candidate bench.candidates._baseline_mc.plain:PlainMC --variant baseline \
    --notes "plain MC (sim.py sampler, chunks of 8000 from rng([seed, ci]))"
