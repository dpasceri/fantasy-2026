#!/usr/bin/env bash
# Smoke test: <= 90 s on 1 core, writes bench/results/mixture_hurdle.smoke.json. Run it through the slot queue:
#   bench/cpuq -n 1 -- bash bench/candidates/mixture_hurdle/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=mixture_hurdle
exec .venv/bin/python -m bench.harness.tracks.player --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.hurdle:MixtureHurdle" --variant zaga --variant zabct --variant zabct_cov \
    --notes "GAMLSS hurdle: multinomial neg/zero/pos (P-spline zero logit, AIC penalty) + gamma or Box-Cox t positive part"
