#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/_baseline_bids.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/_baseline_bids/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.bids --key _baseline_bids --mode smoke \
    --candidate bench.candidates._baseline_bids.bidmodel:BidModelRefit --variant baseline --variant logit --variant clean \
    --notes "auction.BidModel hurdle model (logit participation + lognormal size, manager effects), refit per origin"
