#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/equilibrium_bidding.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/equilibrium_bidding/run_smoke.sh
# Parameters come from tuned.json (tune.py, tuning seeds 1000-1039 only); the eval seeds are never used to pick them.
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=equilibrium_bidding
exec .venv/bin/python -m bench.harness.tracks.policy --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.eqbid:EqBid" --variant shade --variant bayes --variant fp \
    --notes "first-price best response argmax_b P(win|b)(k*typical(g)*pacing - b); shade: point BidModel, bayes: bootstrap posterior predictive, fp: belief learning over rival aggression"
