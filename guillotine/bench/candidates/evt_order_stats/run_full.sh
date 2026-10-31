#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores; about 3-4 min expected), queued only: writes bench/results/evt_order_stats.json
# bench/queue/pending/evt_order_stats.job = {"key": "evt_order_stats", "cmd": ["bash", "bench/candidates/evt_order_stats/run_full.sh"], "timeout_s": 1200}
# The queue runs it with cwd = the guillotine folder under cpuq; the harness sizes its pool from CPUQ_SLOTS.
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=evt_order_stats
exec .venv/bin/python -m bench.harness.tracks.team --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.analytic:OrderStats" --variant cf --variant conv --variant cal \
    --notes "analytic order statistics: cf = Cornish-Fisher team totals (Mehler covariances), independent teams; conv = exact lattice pmf under the baseline copula + one-factor cross-team copula, Gauss-Hermite; cal = conv + rolling (beta, kappa) recalibration"
