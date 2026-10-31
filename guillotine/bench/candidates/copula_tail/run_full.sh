#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/copula_tail.json
# bench/queue/pending/copula_tail.job = {"key": "copula_tail", "cmd": ["bash", "bench/candidates/copula_tail/run_full.sh"], "timeout_s": 1200}
# The queue runs it under cpuq with cwd = the guillotine folder; origins x variants run on CPUQ_SLOTS workers.
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=copula_tail
exec .venv/bin/python -m bench.harness.tracks.team --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.tailcop:TailCopula" --variant gt --variant slot_t --variant shuffle \
    --notes "baseline marginals, tail-dependent copulas: grouped t (game/slate, nu by composite likelihood), depth-chart slot t copula (EB-shrunk slot correlations), empirical game-slot templates (Schaake/ECC reordering)"
