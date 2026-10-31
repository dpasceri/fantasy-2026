#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/_baseline_availability.json
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.availability --key _baseline_availability --mode full \
    --candidate bench.harness.tracks.availability:Baseline --variant baseline \
    --notes "core sim availability: 1 - P_OUT (Q 0.25, D 0.8) this week, flat 2%/wk hazard, geometric mean-3 absences ahead"
