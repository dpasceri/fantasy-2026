#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/rival_lineup.json
# bench/queue/pending/rival_lineup.job = {"key": "rival_lineup", "cmd": ["bash", "bench/candidates/rival_lineup/run_full.sh"], "timeout_s": 1200}
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec .venv/bin/python -m bench.candidates.rival_lineup.run --mode full
