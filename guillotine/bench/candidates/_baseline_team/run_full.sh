#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/_baseline_team.json
# bench/queue/pending/_baseline_team.job = {"key": "_baseline_team", "cmd": ["bash", "bench/candidates/_baseline_team/run_full.sh"], "timeout_s": 1200}
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.team --key _baseline_team --mode full \
    --candidate bench.candidates._baseline_team.copula:CopulaSim --variant baseline \
    --notes "model.py marginal + one-factor Gaussian copula (team/game), refit per origin; 96000 CRN draws per week, p_out=0"
