#!/usr/bin/env bash
# Full backtest: <= 15 min on 6 cores (estimated 1-2 min), writes bench/results/gbm_quantile.json. Never run it yourself;
# enqueue bench/queue/pending/gbm_quantile.job = {"key": "gbm_quantile", "cmd": ["bash", "bench/candidates/gbm_quantile/run_full.sh"], "timeout_s": 1200}
# The queue runs it with cwd = the guillotine folder under cpuq; CPUQ_SLOTS = cores granted (origins run in parallel).
# Afterwards (reviewer add-on, non-fatal, seconds): clean_subset.py re-scores vs the baseline on the 5 test weeks design
# selection never scored (2025-05..08, 2025-14) -> bench/candidates/gbm_quantile/clean_subset.json.
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=gbm_quantile
.venv/bin/python -W ignore::UserWarning -m bench.harness.tracks.player --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.candidate:GBMQuantile" --variant qrf --variant qrf_hurdle --variant lgbq \
    --notes "ratio-space QRF (week-subsampled, Meinshausen weights), QRF + LightGBM P0 hurdle, LightGBM quantile regression; fixed hyperparameters were chosen on inner-validation weeks 2025-09..13, 2025-15..18, 2026-01..06 plus smoke weeks (16 of 21 test blocks), see bench/candidates/$KEY/clean_subset.json for the comparison on the other 5"
.venv/bin/python -m "bench.candidates.$KEY.clean_subset" --mode full || echo "[run_full] clean_subset failed (results above are unaffected)"
