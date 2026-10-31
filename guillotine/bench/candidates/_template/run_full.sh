#!/usr/bin/env bash
# Full backtest: <= 15 min on 6 cores, writes bench/results/<KEY>.json. Never run it yourself;
# enqueue bench/queue/pending/<KEY>.job = {"key": "<KEY>", "cmd": ["bash", "bench/candidates/<KEY>/run_full.sh"], "timeout_s": 1200}
# The queue runs it with cwd = the guillotine folder under cpuq; CPUQ_SLOTS = cores granted (origins run in parallel).
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=_template
exec .venv/bin/python -m bench.harness.tracks.player --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.candidate:Candidate" --variant additive --variant recent \
    --notes "template: additive residual quantiles by position"
