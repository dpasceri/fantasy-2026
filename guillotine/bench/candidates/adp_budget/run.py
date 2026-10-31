"""adp_budget driver: fit post-decision value functions on tuning seeds, then score the ADP policies.

    python -m bench.candidates.adp_budget.run --mode smoke|full [--variant NAME ...] [--jobs N]

1. sigma: pooled sd of a team's weekly score around its projected lineup (own draws, seed 4242).
2. Round 0: continuations of Explore(m0.75 multiplier, eps) with budget interventions on tuning seeds
   TRAIN[mode]["r0"], all three rival conditions (main / stress / lot-bootstrap BidModel refit per
   tuning seed), so V is not tied to the exact point rivals. V0 = VF fit by TD(lambda) backward induction;
   V0c = the same rows fit with traces cut at the budget interventions (review fix, vfa.VF.fit cut=True).
3. Round 1 (api_root only): continuations of Explore(ADP(V0, root), eps) on disjoint tuning seeds
   TRAIN[mode]["r1"]; V1 refit on them (one approximate-policy-iteration step: V of the improved
   policy fvi_root, then greedy on V1 again, same root rule).
4. bench.harness.tracks.policy.run() on the eval seeds (smoke 100-104, full 100-399).
Every seed used before step 4 is >= 1000 (vfa.generate asserts it); nothing reads eval-seed paths.
"""
import argparse
import time

import numpy as np

from bench import harness  # noqa: F401
from bench.harness.tracks import policy as T
from bench.candidates._baseline_policy.multiplier import Multiplier

from . import vfa
from .policy import ADP, Explore

KEY = "adp_budget"
ALL = ("", "_stress", "_bmboot")
TRAIN = {"smoke": {"r0": range(1000, 1002), "r1": range(1500, 1501)},
         "full": {"r0": range(1000, 1110), "r1": range(1500, 1570)}}
EPS, PERTURB, LAM, ROOT = 0.3, 0.25, 0.5, 7
VARIANTS = {          # name: (value function, ADP kwargs)
    "fvi": ("v0", {}),             # not in the default run (replaced by fvi_cut, see README "Review")
    "fvi_cut": ("v0c", {}),
    "fvi_root": ("v0", {"root": ROOT}),
    "api_root": ("v1", {"root": ROOT}),
}
DEFAULT = ("fvi_cut", "fvi_root", "api_root")


def build(mode, names, jobs=None):
    t0 = time.time()
    E = T.env()
    W = len(E["U"].weeks)
    sig = vfa.team_sigma()
    tr = TRAIN[mode]
    D0 = vfa.generate(Explore(Multiplier("m0.75"), eps=EPS), tr["r0"], ALL, PERTURB, sig, jobs, tag="round 0")
    vfs = {"v0": vfa.VF(W, sig).fit(D0, lam=LAM)}
    info = {"sigma": round(sig, 3), "round0": {"rows": int(len(D0["y"])), "seeds": [tr["r0"][0], tr["r0"][-1]],
                                                "behaviour_p_win": round(D0["p_win_behaviour"], 4)},
            "theta_v0": vfs["v0"].summary()}
    if any(VARIANTS[n][0] == "v0c" for n in names):
        vfs["v0c"] = vfa.VF(W, sig).fit(D0, lam=LAM, cut=True)
        info["theta_v0c"] = vfs["v0c"].summary()
        info["round0"]["perturbed_share"] = round(float(D0["pert"].mean()), 4)
    if any(VARIANTS[n][0] == "v1" for n in names):
        D1 = vfa.generate(Explore(ADP("v0", vfs["v0"], root=ROOT), eps=EPS), tr["r1"], ALL, PERTURB, sig, jobs,
                          tag="round 1")
        vfs["v1"] = vfa.VF(W, sig).fit(D1, lam=LAM)
        info["round1"] = {"rows": int(len(D1["y"])), "seeds": [tr["r1"][0], tr["r1"][-1]],
                          "behaviour_p_win": round(D1["p_win_behaviour"], 4)}
        info["theta_v1"] = vfs["v1"].summary()
    info["fit_s"] = round(time.time() - t0, 1)
    for k, v in vfs.items():
        print(f"[adp] {k} theta per weeks-left [1, ln1, lsr, sat, lb, lin, lrs, lrm], n:", flush=True)
        for L, row in v.summary().items():
            print(f"   L={L}: {np.round(row[:-1], 3).tolist()} n={row[-1]}", flush=True)
    variants = [(n, (lambda n=n: ADP(n, vfs[VARIANTS[n][0]], **VARIANTS[n][1]))) for n in names]
    return variants, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="adp_budget: ADP FAAB policy (policy track)")
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--variant", action="append", default=None, choices=list(VARIANTS))
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    names = args.variant or list(DEFAULT)
    variants, info = build(args.mode, names, args.jobs)
    notes = (f"ADP: one-step lookahead on a post-decision logistic V fitted by TD(lambda={LAM}) backward induction "
             f"on tuning-seed continuations (seeds >= 1000, all 3 rival conditions); variants {', '.join(names)}")
    return T.run(KEY, variants, args.mode, args.jobs, notes, extra={"adp": info})


if __name__ == "__main__":
    main()
