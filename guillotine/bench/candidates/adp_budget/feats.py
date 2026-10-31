"""State features for the ADP value function: fast lineup sums and a static-elimination survival basis.

A post-decision state (right after an auction, before the next week is played) at sim week index wi
is summarised by every alive team's projected best-lineup points in each remaining week
w = wi+1 .. W-1 (byes and known injuries included) and the budgets, turned into the basis
  ln1    log P(I am not the lowest next week), weekly scores N(lineup proj, sigma^2), every rival alive
  lsr    sum over the later weeks of log P(I am not the lowest), with the weakest rivals (by mean
         remaining strength) already chopped: one rival fewer each week (static elimination model)
  sat    1 - exp(-B / KAPPA): the first few dollars (enough for $1 claims once rivals are broke) are
         worth far more than the rest; lb = log1p(B) and lin = B / 100 carry the rest (a log alone
         makes $100 of a $165 budget look nearly free at the first auction)
  lrs    log1p(sum of rival budgets), lrm = log1p(max rival budget)
Projections come from the track's Universe tables (projections only, never the sim's draws).
"""
import math

import numpy as np
from scipy.special import log_ndtr

PC = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
GH_Z, GH_W = np.polynomial.hermite_e.hermegauss(10)
GH_W = GH_W / GH_W.sum()
NF = 8  # [1, ln1, lsr, sat, lb, lin, lrs, lrm]
MONEY = slice(3, 6)
KAPPA = 8.0


class Tables:
    """Per-week projection lists and position codes for every Universe player."""

    def __init__(self, U):
        self.W = len(U.weeks)
        self.ix = dict(U.ix)
        self.proj = [list(map(float, U.proj[wi])) for wi in range(self.W)]
        self.pc = [PC.get(p, -1) for p in U.pos]

    def __deepcopy__(self, memo):  # read-only: policy copies share it
        return self


def lineup(T, roster, wi):
    """(projected best-lineup points, (weakest eligible starter for QB, RB, WR, TE)) for
    QB RB RB WR WR TE FLEX FLEX; equals common.best_lineup / dyn_policy._minelig (checked)."""
    pw, ix, pc = T.proj[wi], T.ix, T.pc
    q, r, w, t = [], [], [], []
    for p in roster:
        i = ix.get(p)
        if i is None:
            continue
        v = pw[i]
        if v <= 0:
            continue
        c = pc[i]
        if c == 1:
            r.append(v)
        elif c == 2:
            w.append(v)
        elif c == 3:
            t.append(v)
        elif c == 0:
            q.append(v)
    q.sort(reverse=True)
    r.sort(reverse=True)
    w.sort(reverse=True)
    t.sort(reverse=True)
    fl = sorted(r[2:] + w[2:] + t[1:], reverse=True)[:2]
    qs, rs, ws, ts = (q[:1] + [0.0])[:1], (r[:2] + [0.0, 0.0])[:2], (w[:2] + [0.0, 0.0])[:2], (t[:1] + [0.0])[:1]
    fs = (fl + [0.0, 0.0])[:2]
    tot = qs[0] + rs[0] + rs[1] + ws[0] + ws[1] + ts[0] + fs[0] + fs[1]
    mf = min(fs)
    return tot, (qs[0], min(rs[1], mf), min(ws[1], mf), min(ts[0], mf))


class Strength:
    """Memoised lineup sums for one path (clear() at every path start)."""

    def __init__(self, T):
        self.T = T
        self.memo = {}

    def clear(self):
        self.memo.clear()

    def weeks(self, roster, wi):
        """[(lineup points, weakest starters)] for every remaining week wi+1 .. W-1."""
        fr = frozenset(roster)
        out = []
        for w2 in range(wi + 1, self.T.W):
            k = (fr, w2)
            v = self.memo.get(k)
            if v is None:
                v = self.memo[k] = lineup(self.T, roster, w2)
            out.append(v)
        return out


def gains_by_week(T, pid, wi, mine):
    """My per-week lineup gain from adding pid: max(0, proj - weakest eligible starter), weeks wi+1..W-1."""
    i = T.ix.get(pid)
    if i is None or T.pc[i] < 0:
        return np.zeros(len(mine))
    c = T.pc[i]
    return np.array([max(0.0, T.proj[wi + 1 + k][i] - m[1][c]) for k, m in enumerate(mine)])


def surv_terms(Sm, R, sigma):
    """(ln1, lsr), each (k,), for my per-week strengths Sm (k, L) against rival per-week strengths
    R (n, L) or (k, n, L). Rivals are ordered by mean remaining strength (strongest first); in week w
    the n - w strongest are still alive (the static elimination model)."""
    Sm = np.atleast_2d(np.asarray(Sm, float))
    R = np.asarray(R, float)
    if R.ndim == 2:
        R = np.broadcast_to(R, (Sm.shape[0],) + R.shape)
    k, n, L = R.shape
    o = np.argsort(-R.mean(-1), axis=-1, kind="stable")
    R = np.take_along_axis(R, o[..., None], 1)
    d = (np.swapaxes(R, 1, 2) - Sm[:, :, None]) / sigma                               # (k, L, n)
    A = np.cumsum(log_ndtr(d[:, :, None, :] - GH_Z[None, None, :, None]), -1)          # (k, L, J, n)
    m = np.clip(n - np.arange(L), 1, n) - 1                                            # rivals alive - 1
    pl = np.einsum("j,lkj->kl", GH_W, np.exp(A[:, np.arange(L), :, m]))                # (k, L)
    ls = np.log1p(-np.minimum(pl, 1 - 1e-12))
    return ls[:, 0], ls[:, 1:].sum(-1)


def my_money(B):
    """(sat, lb, lin) for my budget(s) B (scalar or array)."""
    B = np.maximum(B, 0)
    return 1 - np.exp(-B / KAPPA), np.log1p(B), B / 100.0


def rival_money(R):
    """(lrs, lrm) for the rivals' budgets R."""
    R = list(R)
    return math.log1p(sum(R)), math.log1p(max(R) if R else 0)


def team_matrix(ST, wi, me, alive, rosters):
    """(my per-week strengths (L,), my per-week lineups, rival ids, rival strengths (n, L))."""
    mine = ST.weeks(rosters[me], wi)
    riv = [t for t in alive if t != me]
    R = np.array([[x[0] for x in ST.weeks(rosters[t], wi)] for t in riv]).reshape(len(riv), len(mine))
    return np.array([x[0] for x in mine]), mine, riv, R


def state_row(ST, wi, me, alive, rosters, budget, sigma):
    """Feature row (NF,) of a post-decision state at sim week index wi (None if nothing is left to play)."""
    S, _, riv, R = team_matrix(ST, wi, me, alive, rosters)
    if not riv or len(S) == 0:
        return None
    ln1, lsr = surv_terms(S[None, :], R, sigma)
    lrs, lrm = rival_money(budget[t] for t in riv)
    return np.array([1.0, ln1[0], lsr[0], *my_money(budget[me]), lrs, lrm])
