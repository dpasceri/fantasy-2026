#!/usr/bin/env bash
# Full run (<= 15 min on 6 cores, queued only): writes bench/results/_baseline_policy.json
# enqueue bench/queue/pending/_baseline_policy.job = {"key": "_baseline_policy", "cmd": ["bash", "bench/candidates/_baseline_policy/run_full.sh"], "timeout_s": 1200}
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.policy --key _baseline_policy --mode full \
    --candidate bench.candidates._baseline_policy.multiplier:Multiplier \
    --variant baseline --variant m0.75 --variant x1.0 \
    --notes "dynamic.py rule: m x league-typical bid; baseline 1.0/0.75 (run.py's pick), m0.75 0.75/0.75, x1.0 1.0/1.0"
