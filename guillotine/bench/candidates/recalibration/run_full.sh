#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/recalibration.json
# bench/queue/pending/recalibration.job = {"key": "recalibration", "cmd": ["bash", "bench/candidates/recalibration/run_full.sh"], "timeout_s": 1200}
# Calibration set: every earlier virtual week (leave-one-week-out refits), all 48 replicates.
set -euo pipefail
cd "$(dirname "$0")/../../.."
rm -rf bench/results/recalibration/params_full      # per-task map dumps of this run only
export RECAL_CACHE="$PWD/bench/candidates/recalibration/.calcache_full"   # per-origin calibration sets, this run only
rm -rf "$RECAL_CACHE"
trap 'rm -rf "$RECAL_CACHE"' EXIT
export RECAL_K=0 RECAL_R=48 RECAL_MODE=full
.venv/bin/python -m bench.harness.tracks.team --key recalibration --mode full \
    --candidate bench.candidates.recalibration.recal:Recal \
    --variant temp --variant tmix_qq --variant clogit_dcp \
    --notes "baseline CopulaSim + post-hoc maps fit on leave-one-week-out forecasts of every earlier virtual week (48 replicates)"
.venv/bin/python -m bench.candidates.recalibration.params full
