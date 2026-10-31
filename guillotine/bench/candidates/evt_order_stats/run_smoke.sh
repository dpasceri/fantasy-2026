#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/evt_order_stats.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/evt_order_stats/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=evt_order_stats
exec .venv/bin/python -m bench.harness.tracks.team --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.analytic:OrderStats" --variant cf --variant conv --variant cal \
    --notes "analytic order statistics: cf = Cornish-Fisher team totals (Mehler covariances), independent teams; conv = exact lattice pmf under the baseline copula + one-factor cross-team copula, Gauss-Hermite; cal = conv + rolling (beta, kappa) recalibration"
