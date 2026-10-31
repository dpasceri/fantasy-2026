#!/usr/bin/env bash
# Full run (<= 15 min on 6 cores, queued only): tuning-seed fit (seeds >= 1000), then eval seeds 100-399.
# Writes bench/results/adp_budget.json
# enqueue bench/queue/pending/adp_budget.job = {"key": "adp_budget", "cmd": ["bash", "bench/candidates/adp_budget/run_full.sh"], "timeout_s": 1200}
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.candidates.adp_budget.run --mode full
