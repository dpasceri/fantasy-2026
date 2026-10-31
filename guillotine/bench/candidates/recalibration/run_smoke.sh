#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/recalibration.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/recalibration/run_smoke.sh
# Light calibration set for speed: the 2 most recent weeks, virtual replicates r < 16 (full: every week, r < 48).
set -euo pipefail
cd "$(dirname "$0")/../../.."
rm -rf bench/results/recalibration/params_smoke      # per-task map dumps of this run only
export RECAL_CACHE="$PWD/bench/candidates/recalibration/.calcache_smoke"   # per-origin calibration sets, this run only
rm -rf "$RECAL_CACHE"
trap 'rm -rf "$RECAL_CACHE"' EXIT
export RECAL_K=2 RECAL_R=16 RECAL_MODE=smoke
.venv/bin/python -m bench.harness.tracks.team --key recalibration --mode smoke \
    --candidate bench.candidates.recalibration.recal:Recal \
    --variant temp --variant tmix_qq --variant clogit_dcp \
    --notes "baseline CopulaSim + post-hoc maps fit on leave-one-week-out forecasts of earlier virtual weeks (smoke: K=2 weeks, r<16)"
.venv/bin/python -m bench.candidates.recalibration.params smoke
