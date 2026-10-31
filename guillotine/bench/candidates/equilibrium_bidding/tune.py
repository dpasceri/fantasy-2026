"""Tuning for the equilibrium_bidding candidate, on TUNING seeds only (>= 1000, through policy.tune()).

    bench/cpuq -n 2 -- .venv/bin/python -m bench.candidates.equilibrium_bidding.tune --grid   # writes tuned.json
    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates.equilibrium_bidding.tune --bind   # prints eqbid.BIND

--grid  k in {2, 2.5, 3} x gamma in {0, 0.5, 1} for the shade variant on seeds 1000-1039 (8,000 paths), each
        condition (main, _stress, _bmboot), paired against the baseline's m0.75 on the same paths. The
        objective is the mean P(win) over the three conditions (robust to rivals unlike state.bm). The argmax
        is written to tuned.json for all three variants (bayes and fp only change the rival model).
--bind  the binding function of fp's partial-likelihood aggression estimator: the pooled MLE of s on seeds
        1000-1007 when rivals bid at s = 0 (main) and s = 1 (_stress), solved for eqbid.BIND.
"""
import argparse
import itertools
import json
import os
import time

from bench import harness  # noqa: F401
from bench.harness.tracks import policy as T
from bench.candidates._baseline_policy.multiplier import Multiplier
from . import eqbid as EB

import numpy as np

GRID_SEEDS = range(1000, 1040)
BIND_SEEDS = range(1000, 1008)
KS, GAMMAS = (2.0, 2.5, 3.0), (0.0, 0.5, 1.0)
CONDS = ("", "_stress", "_bmboot")


def grid():
    T.env()
    cfg = list(itertools.product(KS, GAMMAS))
    pols = [Multiplier("m0.75")] + [EB.EqBid("shade", k=k, gamma=g) for k, g in cfg]
    t0 = time.time()
    R = T.tune(pols, seeds=GRID_SEEDS, conditions=CONDS)
    blocks = np.repeat(np.array(list(GRID_SEEDS)), T.C)
    ub, inv = np.unique(blocks, return_inverse=True)
    ref = R[0]["arrays"]
    rows = []
    for (k, g), r in zip([(None, None)] + cfg, R):
        a = r["arrays"]
        d = np.mean([a["win" + s] - ref["win" + s] for s in CONDS], 0)
        bm = np.bincount(inv, d) / np.bincount(inv)
        rows.append({"k": k, "gamma": g, **{f"p_win{s}": round(float(a["win" + s].mean()), 4) for s in CONDS},
                     "avg3": round(float(np.mean([a["win" + s].mean() for s in CONDS])), 4),
                     "d_avg3_vs_m0.75": round(float(d.mean()), 4), "se": round(float(bm.std(ddof=1) / np.sqrt(len(ub))), 4),
                     "avg_spend": round(float(a["spend"].mean()), 1)})
        print(rows[-1], flush=True)
    best = max(rows[1:], key=lambda r: r["avg3"])
    par = {"k": best["k"], "gamma": best["gamma"]}
    out = {"params": {v: dict(par) for v in EB.VARIANTS}, "objective": "mean p_win over main, _stress, _bmboot",
           "seeds": [GRID_SEEDS[0], GRID_SEEDS[-1]], "paths": len(GRID_SEEDS) * T.C, "grid": rows,
           "bind": list(EB.BIND), "state_sig": T.env()["sig"], "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(EB.HERE, "tuned.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("best", par, "->", os.path.join(EB.HERE, "tuned.json"))


def bind():
    T.env()
    S = np.linspace(-1.5, 2.5, 41)
    tot = _TOT
    for cond in ("", "_stress"):
        T.tune([_Recorder(S, cond)], seeds=BIND_SEEDS, conditions=(cond,), jobs=1)  # serial: _TOT is per process
    peak = {}
    for c, v in tot.items():
        v = v - v.max()
        i = int(np.clip(np.argmax(v), 1, len(S) - 2))
        a, b, cc = v[i - 1], v[i], v[i + 1]
        peak[c] = float(S[i] + 0.5 * (a - cc) / (a - 2 * b + cc) * (S[1] - S[0]))
    print(f"pooled MLE: s=0 -> {peak['']:.3f}, s=1 -> {peak['_stress']:.3f}; "
          f"BIND = ({peak['']:.2f}, {peak['_stress'] - peak['']:.2f})")


_TOT = {}  # module level: the track deep-copies policy objects, so an attribute would not collect


class _Recorder(EB.EqBid):
    """fp-style play (posterior bids) that also accumulates the raw partial log-likelihood on a fine grid."""

    def __init__(self, S, cond):
        super().__init__("fp")
        self.S, self.cond = S, cond
        self.raw = [tuple(x * EB.STRESS) for x in S]

    def _learn(self, state):
        if self.last is not None:
            keep, keep_lik = self.logw, self.shifts_lik
            self.logw, self.shifts_lik = np.zeros(len(self.S)), self.raw
            last = self.last
            super()._learn(state)
            _TOT[self.cond] = _TOT.get(self.cond, 0) + (self.logw - self.logw.max())
            self.logw, self.shifts_lik, self.last = keep, keep_lik, last
        super()._learn(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--bind", action="store_true")
    a = ap.parse_args()
    if a.bind:
        bind()
    if a.grid:
        grid()


if __name__ == "__main__":
    main()
