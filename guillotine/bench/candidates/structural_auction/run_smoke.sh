#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/structural_auction.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/structural_auction/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=structural_auction
exec .venv/bin/python -m bench.harness.tracks.bids --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.structural:StructuralAuction" --variant hb --variant ah --variant gpv --variant hb0 \
    --draws 20000 \
    --notes "hierarchical hurdle on market-value covariates (hb), + Krasnokutskaya auction shocks with joint draws (ah), + GPV-style structural QRE first-price bids (gpv); hb0 = reviewer pre-peek control (first-smoke covariates), not a contender"
