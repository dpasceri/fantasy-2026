#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/_baseline_player.json
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.player --key _baseline_player --mode full \
    --candidate bench.candidates._baseline_player.ratio:EmpiricalRatio --variant baseline \
    --notes "model.py marginal (bucketed actual/proj quantile tables, log-proj interpolation), refit per origin"
