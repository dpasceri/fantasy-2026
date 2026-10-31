#!/usr/bin/env bash
# Smoke test: <= 90 s on 1 core (about 40 s), writes bench/results/gbm_quantile.smoke.json. Run it through the slot queue:
#   bench/cpuq -n 1 -- bash bench/candidates/gbm_quantile/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
KEY=gbm_quantile
exec .venv/bin/python -W ignore::UserWarning -m bench.harness.tracks.player --key "$KEY" --mode smoke \
    --candidate "bench.candidates.$KEY.candidate:GBMQuantile" --variant qrf --variant qrf_hurdle --variant lgbq \
    --notes "ratio-space QRF (week-subsampled, Meinshausen weights), QRF + LightGBM P0 hurdle, LightGBM quantile regression"
