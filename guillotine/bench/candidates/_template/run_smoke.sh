#!/usr/bin/env bash
# Smoke test: <= 90 s on 1 core, writes bench/results/<KEY>.smoke.json. Run it through the slot queue:
#   bench/cpuq -n 1 -- bash bench/candidates/<KEY>/run_smoke.sh
# Copy this folder to bench/candidates/<KEY>/, then set KEY and the candidate class below.
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=_template
exec .venv/bin/python -m bench.harness.tracks.player --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.candidate:Candidate" --variant additive --variant recent \
    --notes "template: additive residual quantiles by position"
