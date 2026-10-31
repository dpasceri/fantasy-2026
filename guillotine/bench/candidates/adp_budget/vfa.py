"""Value-function approximation for FAAB bidding: simulated continuations on tuning seeds -> V(post-decision state).

Training data (tuning seeds >= 1000 only). The track's own sim (dyn_policy._paths_chunk, exactly the
evaluation dynamics) is run with an exploring behaviour policy; inside the generating worker, the
auction routine is wrapped so that right after every auction the post-decision state (rosters and
budgets of every alive team, week index) is recorded, and with probability `perturb` my budget is
reset to a uniform draw on [0, my starting budget] (a randomised intervention, so the money
coefficient is identified from exogenous variation, not from who happened to buy). Rows are fitted
by backward induction over weeks left on TD(lambda) targets (VF.fit): policy evaluation of the
behaviour policy, the Longstaff-Schwartz / Powell post-decision regression. With cut=True (review fix)
a row whose next row was intervened on bootstraps on V at the pre-intervention state instead of
tracing through the reset (rows carry Bpre, the natural budget, and pert, the intervention flag).

Model. One constrained logistic regression per weeks-left L on feats.NF basis functions:
  logit V = th0 + th1 ln1 + th2 lsr + th3 sat(B) + th4 log1p(B) + th5 B/100 + th6 log1p(sum R) + th7 log1p(max R)
with th1..th5 >= 0 (monotone: more strength and more money never hurt; Jiang & Powell 2015)
and th2..th7 = 0 at L = 1 (no auction follows the last one, so money is worth nothing; lsr is
empty there too).
"""
import copy
import multiprocessing as mp
import os
import random
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from bench.harness import slots
from bench.harness import dyn_policy as DP
from bench.harness.tracks import policy as T

from . import feats as F

TUNE_MIN = T.TUNE_MIN_SEED


# ---------------------------------------------------------------- team-score scale for the basis
def team_sigma(seed=4242, C=1500):
    """Pooled sd of a team's weekly score around its projected best lineup (own draws, own seed)."""
    from sim import team_scores
    E = T.env()
    U, base = E["U"], E["base"]
    teams = base["teams"]
    rng = np.random.default_rng([seed])
    res = []
    for wi, score, avail in U.paths(C, rng, 0):
        idx = np.stack([U.lineup_idx(wi, base["rosters"][t]) for t in teams], 1)
        tot = team_scores(score, avail, idx)
        mu = np.array([U.proj[wi, idx[0, k][idx[0, k] < U.n]].sum() for k in range(len(teams))])
        res.append(tot - mu[None, :])
    r = np.concatenate(res)
    return float(np.sqrt((r ** 2).mean()))


# ---------------------------------------------------------------- data generation
_GW = {}


def fresh(pol):
    """Deep copy of a policy object that shares the track's heavy read-only objects (as the track does)."""
    E = T.env()
    return copy.deepcopy(pol, {id(o): o for o in (E["lg"], E["pm"], E["bm"], E["U"], E["base"])})


def _gen_init(threads):
    os.environ["HARNESS_THREADS"] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
        _GW["limit"] = threadpool_limits(threads)
    except ImportError:
        pass


def _gen_task(job):
    seed, sfx, perturb = job
    assert seed >= TUNE_MIN, "training data only on tuning seeds"
    proto, sigma, Tb = _GW["proto"], _GW["sigma"], _GW["tables"]
    E = T.env()
    me = E["base"]["me"]
    B0 = E["base"]["budget"][me]
    pol = fresh(proto)
    st = {"policy": pol, "budget": B0}
    for k, v in dict(T.CONDITIONS)[sfx].items():
        st[k] = DP.boot_bm(E["bm"], seed) if (k == "bm_rival" and v == "boot") else v
    random.seed(f"adp-gen:{seed}:{sfx}")
    np.random.seed([seed, 7, T.SFX.index(sfx)])
    prng = np.random.default_rng([seed, 99, T.SFX.index(sfx)])
    ST = F.Strength(Tb)
    rows, last = [], [-1]
    orig = DP._auction

    def wrapped(ctx, pool, alive, rosters, budget, *a):
        out = orig(ctx, pool, alive, rosters, budget, *a)
        if pol.npath != last[0]:
            ST.clear()
            last[0] = pol.npath
        wi = a[1][0] - 1
        bpre = budget[me]                       # the natural post-decision budget (before any intervention)
        pert = bool(perturb) and prng.random() < perturb
        if pert:
            budget[me] = int(prng.integers(0, B0 + 1))
        x = F.state_row(ST, wi, me, alive, rosters, budget, sigma)
        if x is not None:
            rows.append((pol.npath - 1, wi, x, bpre, pert))
        return out

    DP._auction = wrapped
    try:
        res = DP._paths_chunk((0, T.C, seed, [st]))[0]["path"]
    finally:
        DP._auction = orig
    win = np.asarray(res["win"])
    if not rows:
        return (seed, sfx, np.zeros((0, F.NF)), np.zeros(0, int), np.zeros(0), np.zeros(0, int), float(win.mean()),
                np.zeros(0), np.zeros(0, bool))
    pth = np.array([r[0] for r in rows])
    return (seed, sfx, np.stack([r[2] for r in rows]), np.array([r[1] for r in rows]), win[pth], pth,
            float(win.mean()), np.array([r[3] for r in rows], float), np.array([r[4] for r in rows], bool))


def generate(proto, seeds, sfxs=("",), perturb=0.3, sigma=None, jobs=None, tag="gen"):
    """Rows {X, wi, y, seed, sfx, path} from simulated continuations of behaviour policy `proto`."""
    E = T.env()
    seeds = [int(s) for s in seeds]
    assert min(seeds) >= TUNE_MIN, f"tuning seeds only (>= {TUNE_MIN})"
    tb = F.Tables(E["U"])
    _GW.update(proto=proto, sigma=sigma, tables=tb)
    jobsl = [(s, x, perturb) for s in seeds for x in sfxs]
    n_jobs = max(1, min(jobs or slots(), slots(), len(jobsl)))
    t0 = time.time()
    if n_jobs == 1:
        out = [_gen_task(j) for j in jobsl]
    else:
        with mp.get_context("fork").Pool(n_jobs, initializer=_gen_init, initargs=(1,)) as pool:
            out = list(pool.imap(_gen_task, jobsl, chunksize=1))
    keep = [o for o in out if len(o[3])]
    D = {"X": np.concatenate([o[2] for o in keep]), "wi": np.concatenate([o[3] for o in keep]),
         "y": np.concatenate([o[4] for o in keep]),
         "seed": np.concatenate([np.full(len(o[3]), o[0]) for o in keep]),
         "sfx": np.concatenate([np.full(len(o[3]), T.SFX.index(o[1])) for o in keep]),
         "path": np.concatenate([o[5] for o in keep]),
         "Bpre": np.concatenate([o[7] for o in keep]), "pert": np.concatenate([o[8] for o in keep]),
         "p_win_behaviour": float(np.mean([o[6] for o in out]))}
    print(f"[adp] {tag}: {len(jobsl)} seed-conditions, {len(D['y'])} rows, behaviour p_win "
          f"{D['p_win_behaviour']:.4f}, {time.time() - t0:.1f}s on {n_jobs} worker(s)", flush=True)
    return D


# ---------------------------------------------------------------- value function
def _fit_logit(X, y, lo, hi, l2=1e-3, x0=None):
    n = len(y)
    pen = np.r_[0.0, np.ones(X.shape[1] - 1)]

    def f(th):
        eta = X @ th
        ll = np.logaddexp(0, eta) - y * eta
        g = X.T @ (expit(eta) - y)
        return ll.sum() / n + l2 * (pen * th) @ th, g / n + 2 * l2 * pen * th

    x0 = np.zeros(X.shape[1]) if x0 is None else x0
    return minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi))).x


def next_row(D):
    """Index of the same path's next auction row (-1 if none) for every row."""
    key = (D["seed"].astype(np.int64) * 8 + D["sfx"]) * 100000 + D["path"]
    o = np.lexsort((D["wi"], key))
    nxt = np.full(len(key), -1)
    same = key[o[1:]] == key[o[:-1]]
    nxt[o[:-1][same]] = o[1:][same]
    return nxt


class VF:
    """Per-weeks-left logistic value function on feats basis. theta[L] (NF,), L = 1..W-1."""

    def __init__(self, W, sigma):
        self.W, self.sigma = W, sigma
        self.theta = np.zeros((W + 1, F.NF))
        self.n = np.zeros(W + 1, int)

    def fit(self, D, l2=1e-3, lam=1.0, cut=False):
        """Backward induction over weeks-left L = 1, 2, ...: rows at L are regressed on the lambda-return
        G_L = (1 - lam) * V_{L-1}(next post-decision state) + lam * G_{L-1} (G = the path's outcome at L = 1,
        and 0 when there is no next auction because I was chopped; lam = 1: Monte Carlo returns; lam < 1:
        TD(lambda) targets, Longstaff-Schwartz backward induction). Fractional targets are fine for the
        logistic log-loss.
        cut (review fix): the budget interventions are not part of the real dynamics, so a row whose next
        row was intervened on must not see that reset in its target. Its trace is cut there and it
        bootstraps fully on V_{L-1} at the natural (pre-intervention) next state, as in off-policy
        trace cutting. Without the cut, every later reset to U{0..B0} shrinks the regression's money
        signal at long horizons (about 0.75 per later auction), so V underprices money early."""
        L = self.W - 1 - D["wi"]
        y = D["y"].astype(float)
        nxt = next_row(D)
        inf = np.inf
        X = D["X"]
        if cut:
            Xn = X.copy()                               # natural post-decision states: pre-intervention budget
            Xn[:, F.MONEY] = np.column_stack(F.my_money(D["Bpre"]))
            pert = np.asarray(D["pert"], bool)
        tgt = y.copy()                                  # lambda-return, built backwards in L
        for l in range(1, self.W):
            m = L == l
            self.n[l] = int(m.sum())
            if self.n[l] < 50:
                continue
            if l > 1 and (lam < 1 or cut):
                j = nxt[m]
                has = j >= 0
                jj = j[has]
                boot = np.zeros(int(m.sum()))
                boot[has] = (1 - lam) * expit(self.logit(l - 1, X[jj])) + lam * tgt[jj]
                if cut:
                    pj = pert[jj]
                    boot[np.flatnonzero(has)[pj]] = expit(self.logit(l - 1, Xn[jj[pj]]))
                tgt[m] = boot                          # no next auction (and l > 1): I was chopped -> 0
            lo = [-inf, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf]
            hi = [inf] * F.NF
            if l == 1:                       # no auction after the last one: money is worthless
                lo[2:] = hi[2:] = [0.0] * 6
            self.theta[l] = _fit_logit(D["X"][m], tgt[m], lo, hi, l2)
        return self

    def logit(self, L, X):
        return np.asarray(X) @ self.theta[L]

    def logloss(self, D):
        L = self.W - 1 - D["wi"]
        eta = np.einsum("ij,ij->i", D["X"], self.theta[L])
        return float(np.mean(np.logaddexp(0, eta) - D["y"] * eta))

    def summary(self):
        return {int(l): [round(float(v), 4) for v in self.theta[l]] + [int(self.n[l])] for l in range(1, self.W)}
