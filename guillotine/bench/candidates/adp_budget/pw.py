"""Vectorised P(my bid b beats every rival claim) for b = 0..bmax under a BidModel (state.p_win's formula).

Same model as dyn_policy.State.rival_cdf / p_win (independent rivals, claims capped at budget, ties
split), but one numpy pass over every rival and every b. boost = (d_logit, d_log_size) gives the
stressed rival curve used by the robust variants.
"""
import math

import numpy as np
from scipy.special import ndtr


def _rivals(s, pid, boost):
    """(P(claim), log-size mean, budget) arrays over the rivals with money, BidModel.p_bid / mu inlined."""
    rg = s.rival_gains(pid)
    if not rg:
        return None
    bm, wk = s.bm, s.week
    ts = list(rg)
    g = np.array([rg[t] for t in ts], float)
    B = np.array([s.budgets[t] for t in ts], float)
    lB = np.log(np.maximum(B, 1))
    a, be = bm.a, bm.beta
    eta = a[0] + a[1] * np.log1p(g) + a[2] * lB + np.array([bm.v.get(t, 0.0) for t in ts]) + boost[0]
    mu = be[0] + be[1] * g + be[2] * lB + be[3] * wk + np.array([bm.u.get(t, 0.0) for t in ts]) + boost[1]
    return 1 / (1 + np.exp(-eta)), mu, B


def pwin_curve(s, pid, bmax, boost=(0.0, 0.0)):
    """p[b] = P(a bid of b wins pid), b = 0..bmax; p[0] = 0. Matches state.p_win (checked)."""
    bmax = int(bmax)
    xs = np.arange(-1, bmax + 1, dtype=float)                 # F at x = -1..bmax
    rv = _rivals(s, pid, boost)
    if rv is not None:
        q, mu, cap = rv
        lx = np.log(np.maximum(xs, 0) + 0.5)
        f = 1 - q[:, None] + q[:, None] * ndtr((lx[None, :] - mu[:, None]) / s.bm.sigma)
        F = np.where(xs[None, :] >= cap[:, None], 1.0, f).prod(0)
    else:
        F = np.ones(len(xs))
    F[0] = 0.0                                                 # x = -1
    p = 0.5 * (F[:-1] + F[1:])
    p[0] = 0.0
    return p


def claim_odds(s, pid):
    """{rid: (P(claims), typical claim)} for rivals with money: who takes pid if I don't."""
    out = {}
    bm, wk = s.bm, s.week
    for t, g in s.rival_gains(pid).items():
        B = s.budgets[t]
        out[t] = (float(bm.p_bid(t, g, B)), min(math.exp(float(bm.mu(t, g, B, wk))), B), g)
    return out
