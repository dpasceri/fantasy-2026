"""Equilibrium bidding (policy track): first-price bid shading against the rival top-bid distribution.

Each auction is a sealed first-price auction, so the optimal bid for a player worth V to me is the
best response to the distribution F of the highest rival claim (Lagrangian form, budget priced at 1/k):
    b* = argmax_b  P(win | b) * (V - b),   P(win | b) = F(b-1) + (F(b) - F(b-1)) / 2   (ties split)
    V  = k * pace * typical(g)             g >= gmin (my lineup gain), typical = league median bid
    pace = (rate / rate_0) ** gamma        rate = budget / (auctions_left + 1), rate_0 = this path's first rate
typical(g) is the market's willingness to pay for gain g at my budget, so V is my reservation price with
money priced at 1/k of a market dollar, and pace (adaptive pacing) cheapens money when I am ahead of plan.
The baseline bids m * typical whatever the competition; a best responder bids $1-3 when nobody can or will
bid and up to V when the field is hot. Variants (the rival model F):
  shade    point BidModel (state.bm), rivals independent
  bayes    posterior predictive: mixture over K lot-bootstrap refits of the bid history (own tag, never the
           eval condition's), i.e. Bayesian decision under rival-model uncertainty
  fp       belief learning ("fictitious play" in the Brown/Robinson sense): a posterior over rival aggression
           (d_logit, d_logsize) = s * (0.7, 0.3), s in S_GRID, updated after every auction from the prices
           rivals actually paid (their budget drops and roster adds), and the bid is the best response to it
k and gamma are tuned on tuning seeds >= 1000 only (tune.py -> tuned.json), on all three conditions.
"""
import json
import math
import os

from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)
from bench.harness.dyn_policy import MIN_GAIN, refit_bm

import numpy as np
from scipy.special import expit, ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
STRESS = np.array([0.7, 0.3])           # unit aggression step (= run.py's STRESS direction)
S_GRID = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
S_PRIOR = np.array([0.1, 0.4, 0.25, 0.15, 0.1])
# indirect inference: the partial likelihood in _learn (rivals with unidentified buys dropped) peaks at
# s_hat = BIND[0] + BIND[1] * s when rivals truly bid at aggression s (pooled over tuning seeds 1000-1007:
# s = 0 -> -0.46, s = 1 -> +0.71; tune.py --bind). The likelihood is evaluated at binding(s), not at s.
BIND = (-0.46, 1.17)
BOOT_TAG = 777_001                      # != dyn_policy.boot_bm's eval tag 20261029
K_BOOT = 8

# variant -> (rival model, default k, gmin)
VARIANTS = {"shade": ("point", 1.5, MIN_GAIN), "bayes": ("boot", 1.5, MIN_GAIN), "fp": ("learn", 1.5, MIN_GAIN)}


def _tuned():
    p = os.path.join(HERE, "tuned.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh).get("params", {})
    return {}


_BOOT = {}  # (id(bm), k, tag) -> refits; module level, so deep copies share it and forks inherit it


def boot_models(bm, k=K_BOOT, tag=BOOT_TAG):
    """bm plus k BidModel refits on lot (waiver run x player) bootstraps of its rows."""
    key = (id(bm), k, tag)
    if key not in _BOOT:
        lots = {}
        for r in sorted(bm.rows, key=lambda r: (r["leg"], r["run"], r["rid"], r["pid"])):
            lots.setdefault((r["run"], r["pid"]), []).append(r)
        keys = sorted(lots)
        out = [bm]
        for i in range(k):
            pick = np.random.default_rng([tag, i]).integers(0, len(keys), len(keys))
            out.append(refit_bm(bm, [r for j in pick for r in lots[keys[j]]]))
        _BOOT[key] = out
    return _BOOT[key]


def _coef(m, rivals):
    return (np.asarray(m.a, float), np.asarray(m.beta, float), float(m.sigma),
            np.array([m.v.get(t, 0.0) for t in rivals]), np.array([m.u.get(t, 0.0) for t in rivals]))


def rival_cdfs(models, shifts, rivals, B, Gr, week, x):
    """F[m, s, p, t, j] = P(rival t's claim on player p <= x[p, j]) per model m and aggression shift s.

    rivals: rival ids, B: their budgets (T,), Gr: their gains (P, T), x: (P, J) or per rival (P, T, J) integer
    bids. Mirrors the sim's rival draw: claim w.p. expit(eta + dl), size round(exp(mu + ds + sigma z)), capped
    at the budget."""
    x = np.asarray(x, float)
    x = x[:, None, :] if x.ndim == 2 else x                          # (P, 1 or T, J)
    lx = np.log(np.maximum(x, 0) + 0.5)
    lB = np.log(np.maximum(B, 1.0))
    cap = x >= B[None, :, None]                                      # (P, T, J): claim <= x surely
    out = np.empty((len(models), len(shifts), x.shape[0], len(rivals), x.shape[2]))
    for i, m in enumerate(models):
        a, b, sig, v, u = _coef(m, rivals)
        eta = a[0] + a[1] * np.log1p(Gr) + a[2] * lB + v              # (P, T)
        mu = b[0] + b[1] * Gr + b[2] * lB + b[3] * week + u
        for j, (dl, ds) in enumerate(shifts):
            q = expit(eta + dl)[..., None]
            f = 1 - q + q * ndtr((lx - (mu + ds)[..., None]) / sig)
            out[i, j] = np.where(cap, 1.0, f)
    return np.where(x < 0, 0.0, out)


def top_cdf(models, shifts, rivals, B, Gr, week, x):
    """F[m, s, p, j] = P(highest rival claim on p <= x[p, j]), rivals independent (= state.rival_cdf)."""
    return rival_cdfs(models, shifts, rivals, B, Gr, week, x).prod(3)


def best_bids(V, F, beta=None):
    """argmax_b pw(b) (V - cost(b)), b = 1..X, per player, with F[p, x] = P(top rival claim <= x), x = 0..X.
    cost(b) = b (linear shadow price) or beta (exp(b / beta) - 1) (concave money curve, scale beta).
    Returns whole-dollar bids (0 = no claim)."""
    X = F.shape[1] - 1
    b = np.arange(1, X + 1, dtype=float)
    pw = 0.5 * (F[:, :-1] + F[:, 1:])                                 # P(win | b), ties split
    c = b if beta is None else beta * np.expm1(b / beta)
    S = pw * (V[:, None] - c[None, :])
    j = S.argmax(1)
    return np.where(S[np.arange(len(V)), j] > 0, j + 1, 0).astype(int)


class EqBid:
    def __init__(self, variant="shade", k=None, gmin=None, gamma=None, beta=None, k_boot=K_BOOT):
        model, k0, g0 = VARIANTS[variant]
        t = _tuned().get(variant, {})
        self.variant, self.model = variant, model
        self.k = float(k if k is not None else t.get("k", k0))
        self.gmin = float(gmin if gmin is not None else t.get("gmin", g0))
        self.gamma = float(gamma if gamma is not None else t.get("gamma", 0.0))
        self.beta = beta if beta is not None else t.get("beta")
        self.k_boot = k_boot
        self.shifts = [tuple(x * STRESS) for x in S_GRID]
        self.shifts_lik = [tuple((BIND[0] + BIND[1] * x) * STRESS) for x in S_GRID]
        self.bids = {}
        self.reset()
        # bayes' refits are fit lazily from state.bm on the first begin() in each process (~0.5 s, cached at
        # module level, deterministic). Not here via policy.env(): under `python -m bench.harness.tracks.policy`
        # that imports a second copy of the track module, whose env() rebuilds the League/Universe/BidModel and
        # re-points dyn_policy's globals at them (numerically identical, but it is the harness's state, not ours).

    def reset(self):
        self.bids = {}
        self.logw = np.log(S_PRIOR)       # fp: belief over rival aggression s, reset every path
        self.last = None                  # fp: the previous auction (pool, rivals, budgets, gains, week)
        self.plan = None                  # pacing: $ per auction at this path's first auction

    # ------------------------------------------------------------ fp: belief update from the last auction
    def _learn(self, state):
        """Bayes update of the aggression belief from what the last auction revealed (public in Sleeper:
        budgets and rosters). Claims run from the highest down, a claim is skipped once its bidder can no
        longer afford it, and a win past the roster cap drops the bidder's weakest rest-of-season player
        (possibly one bought earlier in the same run). Per player on offer, y = the price that cleared it:
          rival t won it        t's claim was y (t's budget drop; t made no other buy)     f_t(y)
          I won it at b         y = b
          nobody took it        y = 0; only used when no buyer can have bought and then dropped it
                                (its ROS is above every buyer's weakest kept player)
          every other rival j   its claim lost: c_j < y (ties split), or j had bought once at x_j > y
                                and c_j fell in the skip window (max(y, B_j - x_j), x_j)   pw_j(y) + skip_j
        Rivals with unidentified buys (two or more, or a possible buy-and-drop) carry no evidence."""
        last, self.last = self.last, None
        if last is None:
            return
        pids, rivals, B, Gr, week, mine, fut0 = last
        ros = {p: state.ros(p, fut0) for p in pids}                  # the value the drop rule used
        teams = dict(state.rosters)
        if state.chopped is not None:
            teams[state.chopped] = tuple(state.pool)                  # zero-ROS players are not on offer: approximate
        owner = {p: t for t, r in teams.items() for p in r}
        pset = set(pids)
        adds = {t: [p for p in r if p in pset] for t, r in teams.items()}
        spent = {t: (B[j] - state.budgets[t]) if t in state.budgets else np.nan for j, t in enumerate(rivals)}
        buyers = [t for t in teams if t == state.chopped or adds[t] or spent.get(t, 0) != 0]
        floor = {t: min((ros.get(p, state.ros(p, fut0)) for p in teams[t] if p not in pset), default=np.inf)
                 for t in buyers}
        vanished = [p for p in pids if p not in owner]
        amb = {p for p in vanished if any(ros[p] <= floor[t] for t in buyers)}   # maybe bought, then dropped
        nb = np.array([len(adds.get(t, ())) for t in rivals])
        x = np.array([spent[t] for t in rivals], float)
        maybe = np.array([any(ros[p] <= floor.get(t, -np.inf) for p in amb) for t in rivals])
        known = ~np.isnan(x) & ~maybe & (((nb == 0) & (x == 0)) | ((nb == 1) & (x >= 1)))
        x = np.where(known, x, 0.0)
        rows, obs = [], []                                            # obs: (y, winner index or -1, kind)
        for i, p in enumerate(pids):
            t = owner.get(p)
            if t is None:
                if p not in amb:
                    rows.append(i), obs.append((0, -1, 0))
            elif t == state.me:
                if mine.get(p, 0) >= 1:
                    rows.append(i), obs.append((mine[p], -1, 1))
            elif t in rivals and known[rivals.index(t)]:
                rows.append(i), obs.append((int(x[rivals.index(t)]), rivals.index(t), 1))
        if not rows:
            return
        o = np.array(obs)
        y = o[:, 0].astype(float)
        R, T = len(rows), len(rivals)
        lo = np.maximum(y[:, None], B[None, :] - x[None, :])          # skip window (lo, x_j - 1]
        pts = np.stack([np.broadcast_to(y[:, None] - 1, (R, T)), np.broadcast_to(y[:, None], (R, T)),
                        lo, np.broadcast_to(x[None, :] - 1, (R, T))], 2)  # (R, T, 4)
        F = rival_cdfs([state.bm], self.shifts_lik, rivals, B, Gr[rows], week, pts)[0]  # (S, R, T, 4)
        lost = np.where(o[:, 2, None] == 0, F[..., 1], 0.5 * (F[..., 0] + F[..., 1]))
        skip = np.where((x[None, :] > y[:, None]) & (nb[None, :] == 1), np.maximum(F[..., 3] - F[..., 2], 0.0), 0.0)
        use = np.broadcast_to(known[None, :], (R, T)).copy()
        r = np.nonzero(o[:, 1] >= 0)[0]
        use[r, o[r, 1]] = False
        ll = (np.log(np.maximum(np.minimum(lost + skip, 1.0), 1e-9)) * use).sum(2)
        if len(r):
            ll[:, r] += np.log(np.maximum(F[:, r, o[r, 1], 1] - F[:, r, o[r, 1], 0], 1e-9))
        self.logw = self.logw + ll.sum(1)
        self.logw -= self.logw.max()

    # ------------------------------------------------------------ policy API
    def begin(self, state):
        self.bids = {}
        learn = self.model == "learn"
        if learn:
            self._learn(state)
        rivals = [t for t in sorted(state.rival_budgets) if state.rival_budgets[t] > 0]
        pool = list(state.pool)
        B = np.array([state.rival_budgets[t] for t in rivals], float)
        Gr = np.array([[state.rival_gains(p).get(t, 0.0) for t in rivals] for p in pool]).reshape(len(pool), len(rivals))
        rate = state.budget / (state.auctions_left + 1)
        self.plan = self.plan or rate
        live = [i for i, p in enumerate(pool) if state.gains[p] >= self.gmin]
        if learn and rivals:
            self.last = (pool, rivals, B, Gr, state.week, self.bids, state.fut[0])  # self.bids is filled below
        if not live:
            return
        k = self.k
        if self.gamma:  # adaptive pacing: value money less when my budget per remaining auction runs ahead of plan
            k *= (rate / self.plan) ** self.gamma
        V = np.array([k * state.typical(state.gains[pool[i]]) for i in live])
        if not rivals:
            self.bids.update({pool[i]: 1 for i, v in zip(live, V) if v > 1})
            return
        X = max(1, int(min(state.budget, math.ceil(V.max()))))
        x = np.broadcast_to(np.arange(X + 1), (len(live), X + 1))
        if self.model == "point":
            Fb = top_cdf([state.bm], [(0.0, 0.0)], rivals, B, Gr[live], state.week, x)[0, 0]
        elif self.model == "boot":
            Fb = top_cdf(boot_models(state.bm, self.k_boot), [(0.0, 0.0)], rivals, B, Gr[live], state.week, x)[:, 0].mean(0)
        else:
            w = np.exp(self.logw)
            F = top_cdf([state.bm], self.shifts, rivals, B, Gr[live], state.week, x)[0]
            Fb = np.tensordot(w / w.sum(), F, 1)
        bb = best_bids(V, Fb, self.beta)
        self.bids.update({pool[i]: int(b) for i, b in zip(live, bb) if b > 0})

    def bid(self, player, gain, week, state):
        return self.bids.get(player, 0)
