#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/availability_hazard.json and the track
# baseline bench/results/_baseline_availability.json. Never run it yourself; enqueue
#   bench/queue/pending/availability_hazard.job = {"key": "availability_hazard", "cmd": ["bash", "bench/candidates/availability_hazard/run_full.sh"], "timeout_s": 1200}
# The queue runs it with cwd = the guillotine folder under cpuq; CPUQ_SLOTS = cores granted (origins run in parallel).
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=availability_hazard
.venv/bin/python -m bench.harness.tracks.availability --key _baseline_availability --mode full \
    --candidate bench.harness.tracks.availability:Baseline --variant baseline \
    --notes "core sim availability: 1 - P_OUT (Q 0.25, D 0.8) this week, flat 2%/wk hazard, geometric mean-3 absences ahead"
.venv/bin/python -m bench.harness.tracks.availability --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.models:Candidate" --variant cells --variant hazard --variant gbm \
    --notes "availability_hazard: EB cells / discrete-time onset+return hazard recursion / direct LightGBM"
exec .venv/bin/python -m bench.harness.compare availability --mode full
