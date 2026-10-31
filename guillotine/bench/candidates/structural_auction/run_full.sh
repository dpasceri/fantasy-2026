#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/structural_auction.json
# enqueue bench/queue/pending/structural_auction.job =
#   {"key": "structural_auction", "cmd": ["bash", "bench/candidates/structural_auction/run_full.sh"], "timeout_s": 1200}
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=structural_auction
exec .venv/bin/python -m bench.harness.tracks.bids --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.structural:StructuralAuction" --variant hb --variant ah --variant gpv --variant hb0 \
    --draws 20000 \
    --notes "hierarchical hurdle on market-value covariates (hb), + Krasnokutskaya auction shocks with joint draws (ah), + GPV-style structural QRE first-price bids (gpv); hb0 = reviewer pre-peek control (first-smoke covariates), not a contender"
