"""Score the frozen sim_optimization variants (tuned.json) on the policy track's evaluation seeds.

    .venv/bin/python -m bench.candidates.sim_optimization.evaluate --mode smoke|full [--jobs N]

Thin wrapper over bench.harness.tracks.policy.run() that also stores the tuning summary (picked
parameters, tuning-seed estimates, compute) and the R&S efficiency study in the result JSON.
"""
import argparse
import json

from bench import harness  # noqa: F401
from bench.harness.tracks import policy as T
from bench.harness.tracks.player import factory
from bench.candidates.sim_optimization.policy import SimOpt, TUNED

KEY = "sim_optimization"
VARIANTS = ("ocba", "cma", "bo")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args(argv)
    with open(TUNED) as f:
        tv = json.load(f)
    sig = T.env()["sig"]
    if tv["state_sig"] != sig:
        print(f"[{KEY}] WARNING: tuned on state {tv['state_sig']}, evaluating on {sig}; rerun tune.py to retune",
              flush=True)
    extra = {"tuning": {v: tv["variants"][v] for v in VARIANTS}, "tuned_state_sig": tv["state_sig"],
             "tune_compute": tv["compute"], "rs_efficiency": tv.get("pcs")}
    notes = ("frozen policies from tune.py on seeds >= 1000: ocba = OCBA-CRN over a 42-point step-multiplier "
             "grid; cma = CMA-ES, bo = batch BO (noisy GP, Thompson sampling) over the 6-D rich family "
             "(log-linear multiplier by week, tier exponent, min gain, strength term, competition cap), each "
             "finished by an OCBA final selection; objective = mean P(win) over the three conditions")
    return T.run(KEY, [(v, factory(SimOpt, v)) for v in VARIANTS], a.mode, a.jobs, notes, extra)


if __name__ == "__main__":
    main()
