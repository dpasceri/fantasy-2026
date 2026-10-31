#!/usr/bin/env bash
# Full run (<= 15 min on 6 cores, queued only): writes bench/results/sim_optimization.json
# enqueue bench/queue/pending/sim_optimization.job = {"key": "sim_optimization", "cmd": ["bash", "bench/candidates/sim_optimization/run_full.sh"], "timeout_s": 1200}
# Evaluates the frozen variants in tuned.json on seeds 100-399 (60,000 paths x 3 conditions).
# Seed workers = min(CPUQ_SLOTS, 6): the protocol's 6-core budget even if the runner hands out more (results do not depend on it).
set -euo pipefail
cd "$(dirname "$0")/../../.."
JOBS=${CPUQ_SLOTS:-1}
if [ "$JOBS" -gt 6 ]; then JOBS=6; fi
exec .venv/bin/python -m bench.candidates.sim_optimization.evaluate --mode full --jobs "$JOBS"
