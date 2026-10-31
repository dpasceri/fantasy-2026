"""Exactness checks on the dev target (never the scored state).

    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates.variance_reduction.verify [reps]

1. The re-implemented totals (NaN-coded inactivity, pruned week-0 subs, component sums, virtual draws with zero
   shift) equal sim.team_scores on the same draws.
2. chop_vstat equals brute-force enumeration of all K^T team combinations on small random inputs.
3. Bias: every variant, reps calls at the full-mode budget, against a large plain-MC dev reference
   (z = mean error / sqrt(var/reps + ref se^2), per team).
"""
import itertools
import sys
import time

from bench.harness.tracks import mc

import numpy as np
import sim

from . import candidate, vr


def check_totals(tg, P):
    U = tg.U
    rng = np.random.default_rng(3)
    NF, K = 10, 8
    sns = vr.season_draws(P, NF, rng, vr.Noise(rng, K, True), "anti") if tg.name == "season" else \
        [vr.week_draws(P, vr.factors(rng, NF, P.nf[0], "iid"), vr.Noise(rng, K, True))]
    X = vr.season_totals(P, sns) if tg.name == "season" else vr.team_sum(P, vr.week_slots(P, sns[0]))[..., None]
    mx = 0.0
    for wi, sn in enumerate(sns):
        nr = sn.shape[2] - 1
        S = np.zeros((NF * K, U.n + 1), np.float32)
        A = np.zeros((NF * K, U.n + 1), bool)
        S[:, P.cols[P.rel[wi]]] = np.nan_to_num(sn[:, :, :nr].reshape(NF * K, nr))
        A[:, P.cols[P.rel[wi]]] = ~np.isnan(sn[:, :, :nr].reshape(NF * K, nr))
        ref = sim.team_scores(S, A, tg.idx[wi]).reshape(NF, K, -1)
        mx = max(mx, float(np.abs(ref - X[:, :, :, wi]).max()))
    return mx


def check_vstat():
    rng = np.random.default_rng(0)
    NF, M, T = 3, 4, 4
    X = (rng.normal(size=(NF, M, T)) * 10 + 50).astype(np.float32)
    br = np.zeros(T)
    for g in range(NF):
        for combo in itertools.product(range(M), repeat=T):
            br[int(np.argmin([X[g, combo[t], t] for t in range(T)]))] += 1
    return float(np.abs(br / M ** T - vr.chop_vstat(X)).max())


def main(reps=12):
    tgs = mc.build_targets(dev=True)
    ok = True
    for name, tg in tgs.items():
        d = check_totals(tg, vr.Prep(tg))
        print(f"[verify] {name}: totals vs sim.team_scores max |diff| {d:.3g}")
        ok &= d < 1e-3
    d = check_vstat()
    print(f"[verify] chop_vstat vs brute force max |diff| {d:.3g}")
    ok &= d < 1e-9
    nref = {"week": 3_200_000, "season": 320_000}
    for name, tg in tgs.items():
        c0 = time.process_time()
        pr = mc.plain(tg, nref[name], 777)
        se = np.sqrt(pr * (1 - pr) / nref[name])
        print(f"[verify] dev {name} plain reference N={nref[name]} ({time.process_time() - c0:.0f} CPU s)")
        n = {"week": 40_000, "season": 32_000}[name]
        for v in candidate.VARIANTS:
            E = np.array([candidate.VR(v).estimate(tg, n, 900_000 + r) for r in range(reps)])
            z = (E.mean(0) - pr) / np.sqrt(E.var(0, ddof=1) / reps + se ** 2)
            print(f"[verify]   {v:12s} max|z| {np.abs(z).max():.2f}  z {np.round(z, 1).tolist()}", flush=True)
            ok &= np.abs(z).max() < 4
    print("[verify] OK" if ok else "[verify] FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main(*(int(a) for a in sys.argv[1:])) else 1)
