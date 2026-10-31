"""mc-track candidate: conditional (product-form) Monte Carlo given the copula factors. See vr.py and README.md.

    VR(variant).estimate(target, n, seed) -> (T,) estimates, unbiased for the plain-MC targets.

A call draws N = (0.95 n - a) / b inner paths: a (fixed overhead) and b (CPU per inner path), both in units of one
plain-MC path, are calibrated on the dev target (calib.py), so a call costs about 0.95 plain-MC calls at the same n
(the harness bills max(cost, 1), and overspending is billed in full). Deterministic given seed.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)

import numpy as np

from . import vr

TARGET_COST = 0.95

# per variant and target: sampler settings, and the cost model cpu = a + b N in plain-path units
# (calib.py on the dev target at the smoke and full n, CALIB_REPS 20 / 6)
VARIANTS = {
    # product-form given the factors: exact all-combination chop statistic / team recombination, iid inner draws
    "cmc": {"week": dict(outer="sobol", K=16, lhs=False, Rp=1, a=580.1, b=0.4247),
            "season": dict(outer="anti", K=8, lhs=False, R=16, Rp=1, a=-56.4, b=0.8396)},
    # + stratified inner draws: week LHS on eps and availability; season LHS on week-0 availability and
    #   antithetic eps / level steps within the group (cheaper than LHS there, equally good on dev)
    "cmc_strat": {"week": dict(outer="sobol", K=16, lhs=True, Rp=1, a=185.9, b=0.5703),
                  "season": dict(outer="anti", K=8, lhs=("av",), anti=("eps", "lvl"), R=16, Rp=1, a=-12.5, b=0.729)},
    # + player-level recombination: (Rp - 1) * K virtual team draws per group scored alongside the real ones
    "cmc_mix": {"week": dict(outer="sobol", K=16, lhs=True, Rp=6, a=273.1, b=0.8577),
                "season": dict(outer="anti", K=8, lhs=("eps", "av"), anti=("lvl",), R=6, Rp=4, chunk=1024, a=48.5, b=1.3282)},
}


class VR:
    def __init__(self, variant="cmc_mix"):
        assert variant in VARIANTS, f"unknown variant {variant!r}: {sorted(VARIANTS)}"
        self.variant = variant

    def estimate(self, target, n, seed):
        cfg = dict(VARIANTS[self.variant][target.name])
        a, b = cfg.pop("a"), cfg.pop("b")
        N = max(2 * cfg["K"], int(round((TARGET_COST * n - a) / b)))
        rng = np.random.default_rng([seed, 0x5652])
        P = vr.Prep(target)
        if target.name == "week":
            return vr.week_estimate(P, N, rng, **cfg)
        return vr.season_estimate(P, N, rng, **cfg)
