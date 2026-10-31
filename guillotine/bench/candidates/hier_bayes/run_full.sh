#!/usr/bin/env bash
# Full backtest (<= 15 min on 6 cores, queued only): writes bench/results/hier_bayes.json
# enqueue bench/queue/pending/hier_bayes.job = {"key": "hier_bayes", "cmd": ["bash", "bench/candidates/hier_bayes/run_full.sh"], "timeout_s": 1200}
# The queue runs it with cwd = the guillotine folder under cpuq; origins x variants run in parallel (CPUQ_SLOTS).
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=hier_bayes
.venv/bin/python -m bench.harness.tracks.player --key "$KEY" --mode full \
    --candidate "bench.candidates.$KEY.hb:HierBayes" --variant hurdle --variant eb --variant dlm \
    --notes "hurdle base (baseline buckets split at the atom: bucket zero share + positive ratio tables) + EB player/team/opp effects (logistic-normal player zero propensity, crossed normal-normal location on probit PIT, per-position z scale); dlm = eb + limma player-scale squeeze + prequential half-lives. CAVEAT: design choices were made on 2025 wk5-18 rolling origins (= full-run test blocks); 2026 blocks are the cleaner holdout (see holdout field)"
# 2025 (design) vs 2026 (holdout) split of the paired comparison; never fails the run
.venv/bin/python -m "bench.candidates.$KEY.holdout" full || echo "[holdout] summary failed (results unaffected)"
