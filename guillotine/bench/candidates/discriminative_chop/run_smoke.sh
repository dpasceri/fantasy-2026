#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/discriminative_chop.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/discriminative_chop/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=discriminative_chop
exec .venv/bin/python -m bench.harness.tracks.team --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.candidate:DiscriminativeChop" --variant clogit --variant benter --variant gbm \
    --notes "discriminative chop: ridge rank-ordered logit (pure), Benter two-stage stack of log p_sim and the logit, LightGBM softmax boosting from log p_sim; total_q/line_q = baseline sim"
