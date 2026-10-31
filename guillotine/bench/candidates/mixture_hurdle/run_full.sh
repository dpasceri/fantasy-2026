#!/usr/bin/env bash
# Full backtest: <= 15 min on 6 cores (about 1-2 min expected), writes bench/results/mixture_hurdle.json. Never run it
# yourself; enqueue bench/queue/pending/mixture_hurdle.job =
#   {"key": "mixture_hurdle", "cmd": ["bash", "bench/candidates/mixture_hurdle/run_full.sh"], "timeout_s": 1200}
# The queue runs it with cwd = the guillotine folder under cpuq; CPUQ_SLOTS = cores granted (origins run in parallel).
# Afterwards (reviewer add-on, non-fatal, seconds): clean_subset.py re-scores vs the baseline without the
# development weeks 2025-05..09 -> bench/candidates/mixture_hurdle/clean_subset.json.
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=mixture_hurdle
.venv/bin/python -m bench.harness.tracks.player --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.hurdle:MixtureHurdle" --variant zaga --variant zabct --variant zabct_cov \
    --notes "GAMLSS hurdle: multinomial neg/zero/pos (P-spline zero logit, AIC penalty) + gamma or Box-Cox t positive part; dev checks touched 2025-05..09, see bench/candidates/$KEY/clean_subset.json for the comparison without them"
.venv/bin/python -m "bench.candidates.$KEY.clean_subset" --mode full || echo "[run_full] clean_subset failed (results above are unaffected)"
