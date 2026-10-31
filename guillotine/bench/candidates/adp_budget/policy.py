"""ADP FAAB bid policy: one-step lookahead on a fitted post-decision value function (policy track).

At every auction (begin), for each player p on offer with lineup gain g >= gmin, in decreasing gain:
    Q(b) = pwin_p(b) * V(B - b, S + g_w) + (1 - pwin_p(b)) * V_lose - V(B, S)
and bid b* = argmax_{0 <= b <= B} Q(b) (no claim when no b > 0 beats b = 0). V is vfa.VF (logistic
in the feats basis; S = my projected lineup in every remaining week, g_w = p's per-week lineup gain),
pwin_p is the vectorised state.p_win curve under the point BidModel (robust variants mix in the
stressed rival curve), and V_lose = V(B, S) unless `denial` credits the rival who would take p.
With `root`, auctions with more than `root` weeks left use the base multiplier rule instead (the
fitted V misprices the 8-week first auction: see README).
Players are planned in turn on the expected budget left by the ones before
(budget -= pwin b*; each player's gain is against my current roster). bid() returns the plan.

Explore is the behaviour policy behind the training continuations (vfa.generate): its base policy
(the m0.75 multiplier rule in round 0, a fitted ADP policy in the policy-iteration round) with
probability 1 - eps, otherwise a random rule for that auction (skip / random multiple of the typical
bid / random P(win) target / the base plan with log-normal noise).
"""
import math

import numpy as np
from scipy.special import expit

from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)
from bench.harness.tracks import policy as T
from bench.candidates._baseline_policy.multiplier import Multiplier

from . import feats as F
from .pw import claim_odds, pwin_curve

STRESS = T.STRESS
_TABLES = {}


def tables():
    U = T.env()["U"]
    if id(U) not in _TABLES:
        _TABLES[id(U)] = F.Tables(U)
    return _TABLES[id(U)]


class _Shared:
    """Read-only holder that deepcopy passes through (the fitted value function)."""

    def __init__(self, obj):
        self.obj = obj

    def __deepcopy__(self, memo):
        return self


class ADP:
    """root: with more than `root` weeks left (the 8-week first auction), bid the base multiplier rule
    (the track's reference, 1.0 x typical before week 12) instead of the lookahead."""

    def __init__(self, variant="fvi", vf=None, gmin=1.0, robust=0.0, denial=False, root=None):
        if vf is None:
            raise ValueError("ADP needs a fitted value function: run bench.candidates.adp_budget.run")
        self.variant = variant
        self.vf = _Shared(vf)
        self.gmin, self.robust, self.denial, self.root = gmin, robust, denial, root
        self.base = Multiplier("baseline")
        self.ST = F.Strength(tables())   # Tables is shared by deepcopy; the memo is per copy
        self.plan = {}

    def reset(self):
        self.ST.clear()
        self.plan = {}

    def _curve(self, s, p, bmax):
        pw = pwin_curve(s, p, bmax)
        if self.robust:
            pw = (1 - self.robust) * pw + self.robust * pwin_curve(s, p, bmax, STRESS)
        return pw

    @staticmethod
    def _mval(th, B):
        sat, lb, lin = F.my_money(B)
        return th[3] * sat + th[4] * lb + th[5] * lin

    def begin(self, s):
        self.plan = plan = {}
        if self.root is not None and s.weeks_left > self.root:
            for p in s.pool:
                plan[p] = self.base.bid(p, s.gains[p], s.week, s)
            return
        vf = self.vf.obj
        L = min(s.weeks_left, vf.W - 1)
        if L <= 0 or s.budget <= 0:
            return
        cands = sorted((p for p in s.pool if s.gains[p] >= self.gmin), key=lambda p: -s.gains[p])
        if not cands:
            return
        th, sig, ST, wi, tb = vf.theta[L], vf.sigma, self.ST, s.wi, self.ST.T
        S, mine, riv, R = F.team_matrix(ST, wi, s.me, s.alive, s.rosters)
        if not riv or len(S) == 0:
            return
        lrs, lrm = F.rival_money(s.budgets[t] for t in riv)
        const = th[0] + th[6] * lrs + th[7] * lrm
        gws = [F.gains_by_week(tb, p, wi, mine) for p in cands]
        ln1, lsr = F.surv_terms(np.stack([S] + [S + g for g in gws]), R, sig)   # one pass: base + each player
        h = const + th[1] * ln1 + th[2] * lsr
        Bc = float(s.budget)
        for j, p in enumerate(cands):
            bmax = int(math.floor(Bc))
            if bmax < 1:
                break
            v0 = expit(h[0] + self._mval(th, Bc))
            v1 = expit(h[1 + j] + self._mval(th, Bc - np.arange(bmax + 1)))
            pw = self._curve(s, p, bmax)
            Q = pw * (v1 - v0)
            if self.denial:
                Q = Q + (1 - pw) * (self._v_lose(s, p, riv, R, S, th, sig, const, Bc) - v0)
            k = int(np.argmax(Q))
            if k < 1 or Q[k] <= Q[0]:
                continue
            plan[p] = k
            Bc -= pw[k] * k                 # plan the next player on the expected budget left

    def _v_lose(self, s, p, riv, R, Sc, th, sig, const, Bc):
        """E[V | I don't get p]: a rival may claim him (weights ~ P(claim) x typical claim)."""
        odds = claim_odds(s, p)
        mv = self._mval(th, Bc)
        ln1, lsr = F.surv_terms(Sc[None, :], R, sig)
        v_none = expit(const + th[1] * ln1[0] + th[2] * lsr[0] + mv)
        ts = [t for t in odds if t in riv]
        if not ts:
            return v_none
        q = np.array([odds[t][0] for t in ts])
        w = q * np.array([odds[t][1] for t in ts])
        if w.sum() <= 0:
            return v_none
        pi = (1 - np.prod(1 - q)) * w / w.sum()
        pos = {t: k for k, t in enumerate(riv)}
        Rk = np.repeat(R[None, :, :], len(ts), 0)
        for k, t in enumerate(ts):
            Rk[k, pos[t]] += F.gains_by_week(self.ST.T, p, s.wi, self.ST.weeks(s.rosters[t], s.wi))
        ln1, lsr = F.surv_terms(np.repeat(Sc[None, :], len(ts), 0), Rk, sig)
        vt = expit(const + th[1] * ln1 + th[2] * lsr + mv)
        return float((1 - pi.sum()) * v_none + pi @ vt)

    def bid(self, player, gain, week, state):
        return self.plan.get(player, 0)


class Explore:
    """Behaviour policy for training continuations; npath counts reset() calls (path ids for vfa)."""

    def __init__(self, base, eps=0.3, noise=0.4):
        self.base, self.eps, self.noise = base, eps, noise
        self.npath = 0
        self.plan = {}

    def reset(self):
        self.npath += 1
        if hasattr(self.base, "reset"):
            self.base.reset()

    def _base_plan(self, s):
        if hasattr(self.base, "begin"):
            self.base.begin(s)
        return {p: self.base.bid(p, s.gains[p], s.week, s) for p in s.pool}

    def begin(self, s):
        r = s.rng
        plan = self._base_plan(s)
        if r.random() >= self.eps:
            self.plan = plan
            return
        self.plan = out = {}
        u = r.random()
        if u < 0.1:                                                    # skip the auction
            return
        if u < 0.4:                                                    # noisy base plan
            for p, b in plan.items():
                out[p] = round(b * math.exp(r.gauss(0, self.noise)))
            return
        gmin = r.choice((1.0, 2.0, 3.0))
        cands = [p for p in s.pool if s.gains[p] >= gmin]
        if u < 0.7:                                                    # random multiple of typical
            m = math.exp(r.uniform(math.log(0.35), math.log(1.8)))
            for p in cands:
                out[p] = round(m * s.typical(s.gains[p]))
            return
        tgt, cap = r.uniform(0.3, 0.97), math.exp(r.uniform(0.0, math.log(3.0)))
        for p in cands:                                                # random P(win) target
            pw = pwin_curve(s, p, s.budget)
            k = int(np.searchsorted(pw, tgt)) if pw[-1] >= tgt else s.budget
            out[p] = min(k, round(cap * s.typical(s.gains[p])))

    def bid(self, player, gain, week, state):
        return self.plan.get(player, 0)
