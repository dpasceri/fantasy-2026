#!/usr/bin/env bash
# Full run (<= 15 min on 6 cores, queued only): writes bench/results/equilibrium_bidding.json
# enqueue bench/queue/pending/equilibrium_bidding.job = {"key": "equilibrium_bidding", "cmd": ["bash", "bench/candidates/equilibrium_bidding/run_full.sh"], "timeout_s": 1200}
# Seeds run in parallel over CPUQ_SLOTS workers (the track's default). Parameters come from tuned.json.
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=equilibrium_bidding
exec .venv/bin/python -m bench.harness.tracks.policy --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.eqbid:EqBid" --variant shade --variant bayes --variant fp \
    --notes "first-price best response argmax_b P(win|b)(k*typical(g)*pacing - b); shade: point BidModel, bayes: bootstrap posterior predictive, fp: belief learning over rival aggression"
