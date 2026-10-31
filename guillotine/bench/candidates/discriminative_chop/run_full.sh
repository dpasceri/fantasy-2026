#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/discriminative_chop.json
# bench/queue/pending/discriminative_chop.job = {"key": "discriminative_chop", "cmd": ["bash", "bench/candidates/discriminative_chop/run_full.sh"], "timeout_s": 1200}
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=discriminative_chop
exec .venv/bin/python -m bench.harness.tracks.team --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.candidate:DiscriminativeChop" --variant clogit --variant benter --variant gbm \
    --notes "discriminative chop: ridge rank-ordered logit (pure), Benter two-stage stack of log p_sim and the logit, LightGBM softmax boosting from log p_sim; total_q/line_q = baseline sim"
