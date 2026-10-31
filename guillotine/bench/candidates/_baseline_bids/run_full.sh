#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/_baseline_bids.json
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.harness.tracks.bids --key _baseline_bids --mode full \
    --candidate bench.candidates._baseline_bids.bidmodel:BidModelRefit --variant baseline --variant logit --variant clean \
    --notes "auction.BidModel hurdle model (logit participation + lognormal size, manager effects), refit per origin"
