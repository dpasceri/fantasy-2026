#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/projection_ensemble.json
#   bench/queue/pending/projection_ensemble.job =
#   {"key": "projection_ensemble", "cmd": ["bash", "bench/candidates/projection_ensemble/run_full.sh"], "timeout_s": 1200}
# Origins x variants run in parallel over CPUQ_SLOTS forked workers (harness default).
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=projection_ensemble
exec .venv/bin/python -m bench.harness.tracks.player --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.ensemble:ProjectionEnsemble" --variant lpool --variant recal --variant stack \
    --notes "forecast combination: linear-pool bucket blend; probit recalibration; stacked point-in-time signals"
