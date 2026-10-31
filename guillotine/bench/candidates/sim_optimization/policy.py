"""sim_optimization policy family: a parametric FAAB bid rule whose parameters are picked by simulation
optimization (tune.py: OCBA, CMA-ES, batch Bayesian optimization) on tuning seeds >= 1000.

    bid = min(m(week) x (g / 6)^gam x exp(kap * z) x typical(g),  cap_alpha)      (0 if g < gmin)

  m(week)   step (m_early before `switch`, m_late from it on; the baseline family) or log-linear,
            exp(lm0 + lm1 * t) with t = 0 at the auction 8 weeks before the final week, 1 at the last one
  gam       player-tier exponent: > 0 spends relatively more on studs, < 0 on depth
  kap, z    team strength: z in [-1, 1] = 2 x (share of alive rivals whose projected lineup for the next
            week is below mine) - 1, computed lazily once per auction (only when a bid needs it)
  cap_alpha competition-aware shading: the smallest bid b with P(every rival claim <= b - 1) >= alpha under
            the point BidModel in state.bm (same formula as State.rival_cdf, scalar math + bisection). It
            never exceeds the richest rival's budget + 1, so money above what can be contested is not spent.
Parameters live in tuned.json (written by tune.py); SimOpt(theta=...) builds any point of the family.
"""
import json
import math
import os

from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)
from common import J, best_lineup

HERE = os.path.dirname(os.path.abspath(__file__))
TUNED = os.path.join(HERE, "tuned.json")
G_REF = 6.0
SQ2 = math.sqrt(2.0)
SLOTS = [s for s in J("league.json")["roster_positions"] if s not in ("BN", "IR", "TAXI")]
DEFAULT = {"m_early": 1.0, "m_late": 0.75, "switch": 12, "lm0": None, "lm1": 0.0,
           "gam": 0.0, "gmin": 2.0, "kap": 0.0, "alpha": None}


def load(variant):
    """Frozen parameters of a tuned variant (tuned.json["variants"][variant]["theta"])."""
    with open(TUNED) as f:
        tv = json.load(f)["variants"]
    if variant not in tv:
        raise KeyError(f"unknown sim_optimization variant {variant!r} (tuned.json has {sorted(tv)})")
    return tv[variant]["theta"]


class SimOpt:
    def __init__(self, variant="cma", theta=None):
        self.variant = variant
        th = dict(DEFAULT)
        th.update(load(variant) if theta is None else theta)
        unknown = set(th) - set(DEFAULT)
        assert not unknown, f"unknown parameters {unknown}"
        self.th = th
        self.z = None

    def reset(self):
        self.z = None

    def begin(self, state):
        self.z = None  # strength is computed lazily, on the first bid that needs it

    def strength(self, state):
        """z in [-1, 1]: 2 x (share of alive rivals whose projected lineup next week is below mine) - 1."""
        wi = state.fut[0]
        tot = {}
        for t, r in state.rosters.items():
            val = {p: state.proj(p, wi) for p in r}
            lu = best_lineup(SLOTS, [(v, p, state.pos(p)) for p, v in val.items()])
            tot[t] = sum(val[p] for _, p in lu if p)
        mine = tot[state.me]
        others = [v for t, v in tot.items() if t != state.me]
        return 2.0 * sum(v < mine for v in others) / len(others) - 1.0 if others else 0.0

    def mult(self, gain, week, state):
        th = self.th
        if th["lm0"] is None:
            m = th["m_early"] if week < th["switch"] else th["m_late"]
        else:
            t = min(max((8 - (state.weeks[-1] - week)) / 7.0, 0.0), 1.0)
            m = math.exp(th["lm0"] + th["lm1"] * t)
        if th["gam"]:
            m *= (gain / G_REF) ** th["gam"]
        if th["kap"]:
            if self.z is None:
                self.z = self.strength(state)
            m *= math.exp(th["kap"] * self.z)
        return m

    def bid(self, player, gain, week, state):
        th = self.th
        if gain < th["gmin"]:
            return 0
        b = self.mult(gain, week, state) * state.typical(gain)
        if th["alpha"] is not None and b > 1.5:
            b = min(b, cap(player, state, th["alpha"]))
        return b


def cap(pid, state, alpha):
    """Smallest whole-dollar bid b >= 1 with P(top rival claim <= b - 1) >= alpha (point model, rivals
    independent, as State.rival_cdf); at most max rival budget + 1."""
    bm, wk = state.bm, state.week
    a0, a1, a2 = bm.a
    b0, b1, b2, b3 = bm.beta
    den = bm.sigma * SQ2
    riv, top = [], 0
    for t, g in state.rival_gains(pid).items():
        B = state.budgets[t]
        lb = math.log(max(B, 1))
        q = 1.0 / (1.0 + math.exp(-(a0 + a1 * math.log1p(g) + a2 * lb + bm.v.get(t, 0.0))))
        riv.append((q, b0 + b1 * g + b2 * lb + b3 * wk + bm.u.get(t, 0.0), B))
        top = max(top, B)

    def F(x):
        lx, p = math.log(x + 0.5), 1.0
        for q, mu, B in riv:
            if x < B:
                p *= 1.0 - q + q * 0.5 * (1.0 + math.erf((lx - mu) / den))
        return p

    if F(0) >= alpha:
        return 1
    lo, hi = 0, top  # F(lo) < alpha <= F(hi) = 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if F(mid) >= alpha:
            hi = mid
        else:
            lo = mid
    return hi + 1
