#!/usr/bin/env bash
# Smoke test (<= 90 s on 1 core): writes bench/results/copula_tail.smoke.json
#   bench/cpuq -n 1 -- bash bench/candidates/copula_tail/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=copula_tail
exec .venv/bin/python -m bench.harness.tracks.team --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.tailcop:TailCopula" --variant gt --variant slot_t --variant shuffle \
    --notes "baseline marginals, tail-dependent copulas: grouped t (game/slate, nu by composite likelihood), depth-chart slot t copula (EB-shrunk slot correlations), empirical game-slot templates (Schaake/ECC reordering)"
