"""structural_auction: rival FAAB bids as a sealed first-price auction with common values and budgets.

Covariates (all known at the run):
  lg    log1p(own lineup gain)                                     private value (need)
  lpace log(budget / auctions left), auctions left = final_week + 1 - w0
        budget pacing (Balseiro-Besbes-Weintraub 2015): a budget-constrained bidder shades by the
        shadow price of money, which falls as the remaining horizon shrinks; this replaces the core's
        free b3 * leg trend with one budget-per-auction coefficient
  lgm   log1p(mean gain over every team that could bid in the run)   common (market) value
  lpq   log1p(the player's mean projection over his last 4 projected weeks among the 8 completed weeks
        up to the run's info_week, from data.table() under the harness horizon)  reputation value that
        does not depend on the look-ahead projections behind gain (an injured star has gain 0)
  pool  player came from the chopped roster (participation only)
Variants (simplest first):
  hb   hierarchical hurdle. P(bid) = logistic(X a + v_i), log bid = X b + u_i + e, manager effects
       v_i, u_i with sds by Laplace / REML (mixed.py), Gamma(2) sd priors. The predictive integrates
       fixed-effect and BLUP uncertainty. The bid is the rounded lognormal, heaped to a multiple of $5
       with probability pi (ML on training bids), capped at the budget. No leg trend (the core's b3 * leg
       extrapolates badly). Price = exact independent product.
  ah   hb + auction-level unobserved heterogeneity (Krasnokutskaya 2011): a player-run shock eta_g
       multiplies every bid on that player (crossed random effect in the size LMM) and a player-run
       shock zeta_g shifts every team's participation (crossed effect in the GLMM). Marginals integrate
       both; sample_joint shares them across a group's chances, so the top bid comes from affiliated
       bids.
  gpv  structural first-price model on top of ah, GPV two-step logic (Guerre-Perrigne-Vuong 2000): the
       stage-1 bid distribution (ah) is each manager's belief about rivals, W_i(b | eta) = P(win with b);
       values log v = eta_g + g'x + u_i + s_v z; the bid is a logit quantal response to expected surplus
       (Goeree-Holt-Palfrey 2002): P(b | v) ~ [softplus(v - b) W_i(b | eta)]^lam on 1..budget. (g, s_v,
       lam) by ML on training bids with eta_g at its stage-1 BLUP. Prediction integrates eta over 15
       equal-probability nodes and v by Gauss-Hermite; rivals in the test group (their budgets, needs,
       participation) enter through W. Participation is ah's.
  hb0  reviewer's pre-peek control, not a contender: hb on the first smoke's covariates [lg, log(budget),
       lgm] (no lpq, no pacing). lpq and lpace were added after the smoke legs 5 / 7 were scored, so
       hb vs hb0 on the unseen legs 3, 4, 6 measures what those choices buy out of sample.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import norm

from bench.harness import data as D

from .mixed import GLMM, LMM, codes

VARIANTS = ("hb", "ah", "gpv", "hb0")
HEAP = 5            # round-number heaping unit ($)
J_ETA = 15          # equal-probability nodes for the auction shock (gpv)
_QX, _QW = np.polynomial.hermite_e.hermegauss(24)
_LQW = np.log(_QW / _QW.sum())


def quality(d):
    """Point-in-time player value: mean proj over the player's last 4 weeks with proj >= 0.5 among the 8
    completed weeks up to each row's info_week (data.table(), so the harness horizon applies); 0 if none."""
    T = D.table()
    t0 = D.t_of(D.league().season, 1) - 1                  # global index of info_week 0 (last week of 2025)
    ti = t0 + d["info_week"].to_numpy(int)
    pids = d["pid"].astype(str).to_numpy()
    h = T[T["pid"].isin(set(pids)) & (T["proj"].to_numpy(float) >= 0.5) & (T["t"].to_numpy() <= ti.max())]
    hist = {p: (g["t"].to_numpy(), g["proj"].to_numpy(float)) for p, g in h.sort_values("t").groupby("pid")}
    out, memo = np.zeros(len(d)), {}
    for i, (p, t) in enumerate(zip(pids, ti)):
        if (p, t) not in memo:
            tt, pp = hist.get(p, (np.zeros(0), np.zeros(0)))
            w = pp[(tt <= t) & (tt > t - 8)][-4:]
            memo[p, t] = float(w.mean()) if len(w) else 0.0
        out[i] = memo[p, t]
    return out


def feats(d, pre=False):
    """(n, 4) [lg, lpace, lgm, lpq] and pool for a frame of chances (lgm over the frame's (run, pid) groups).
    pre: the first smoke's (n, 3) [lg, log(budget), lgm] (hb0 control)."""
    gm = d.groupby(["run", "pid"])["gain"].transform("mean").to_numpy(float)
    if pre:
        X = np.column_stack([np.log1p(d["gain"].to_numpy(float)), np.log(d["budget"].to_numpy(float)), np.log1p(gm)])
        return X, d["pool"].to_numpy(float)
    left = D.settings()["final_week"] + 1 - d["w0"].to_numpy(float)
    X = np.column_stack([np.log1p(d["gain"].to_numpy(float)), np.log(d["budget"].to_numpy(float) / np.maximum(left, 1)),
                         np.log1p(gm), np.log1p(quality(d))])
    return X, d["pool"].to_numpy(float)


def group_labels(d):
    return (d["run"].astype(str) + "-" + d["pid"].astype(str)).to_numpy()


def size_pmf(m, s, B, K, pi):
    """(n, K+1) pmf over dollars 0..K of min(report, B): log x ~ N(m, s^2), report = round(x) (x < 1.5 -> $1)
    or, with probability pi, x heaped to the nearest multiple of $5 (x >= 2.5)."""
    m, s = np.asarray(m, float)[:, None], np.asarray(s, float)[:, None]
    B = np.asarray(B, int)
    n = len(B)
    k = np.arange(K + 1)[None, :]

    def cdf(x):
        return norm.cdf((np.log(x) - m) / s)
    Fu = cdf(k + 0.5)
    Pc = np.diff(Fu, axis=1, prepend=0.0)
    Pc[:, 1] += Pc[:, 0]
    Pc[:, 0] = 0.0
    tail = 1.0 - Fu[:, -1]
    if pi > 0:
        Ph = np.zeros((n, K + 1))
        Ph[:, 1] = cdf(1.5)[:, 0]
        Ph[:, 2] = (cdf(2.5) - cdf(1.5))[:, 0]
        j = np.arange(HEAP, K + 1, HEAP)
        Ph[:, j] = cdf(j[None, :] + HEAP / 2) - cdf(j[None, :] - HEAP / 2)
        th = 1.0 - cdf(np.array([[j[-1] + HEAP / 2]]))[:, 0]
        Pc = (1 - pi) * Pc + pi * Ph
        tail = (1 - pi) * tail + pi * th
    over = k >= B[:, None]
    cap = (Pc * over).sum(1) + tail
    Pc[np.broadcast_to(over, Pc.shape)] = 0.0
    Pc[np.arange(n), B] = cap
    return Pc


def log_win(p, S, gidx, ng):
    """(n, K) log P(a bid of b = 1..K beats every other chance of its group): rivals independent given the
    group's shocks, ties split (waiver order), rival k bids 0 w.p. 1 - p_k, else from S_k."""
    H = p[:, None] * S
    H[:, 0] = 1.0 - p
    F = np.cumsum(H, 1)
    L = np.log(np.maximum(F[:, :-1] + 0.5 * H[:, 1:], 1e-300))
    G = np.zeros((ng, L.shape[1]))
    np.add.at(G, gidx, L)
    return G[gidx] - L


def qre_logp(LW, logv, B, lam):
    """(n, Q, K) log P(bid = b | value node) for b = 1..K: logit response to log expected surplus."""
    K = LW.shape[1]
    b = np.arange(1, K + 1, dtype=float)
    x = np.exp(logv)[:, :, None] - b[None, None, :]
    lsp = np.where(x > -30, np.log(np.logaddexp(0.0, np.maximum(x, -30))), x)
    U = lam * (lsp + LW[:, None, :])
    U = np.where(b[None, None, :] <= np.asarray(B)[:, None, None], U, -np.inf)
    return U - logsumexp(U, axis=2, keepdims=True)


class StructuralAuction:
    def __init__(self, variant="hb"):
        assert variant in VARIANTS, variant
        self.variant = variant
        self.pre = variant == "hb0"
        self.shock = variant in ("ah", "gpv")
        if self.shock:                       # only ah / gpv draw jointly; hb keeps the exact product
            self.sample_joint = self._sample_joint

    # ------------------------------------------------------------ design
    def _design(self, d):
        X3, pool = feats(d, self.pre)
        X3 = X3 - self.ctr
        one = np.ones((len(d), 1))
        return np.hstack([one, X3, pool[:, None]]), np.hstack([one, X3])

    def _codes(self, d, glev=None):
        r, _ = codes(d["rid"].astype(int).astype(str), self.rlev)
        g, _ = codes(group_labels(d), glev if glev is not None else np.array([], str))
        return r, g

    # ------------------------------------------------------------ fit
    def fit(self, train):
        d = train[train["budget"].to_numpy() > 0].reset_index(drop=True)
        X3, _ = feats(d, self.pre)
        self.ctr = X3.mean(0)
        self.rlev = np.unique(d["rid"].astype(int).astype(str))
        Xp, Xs = self._design(d)
        y = (d["bid"].to_numpy() > 0).astype(float)
        r, _ = codes(d["rid"].astype(int).astype(str), self.rlev)
        gl = group_labels(d)
        g, self.glev = codes(gl)
        fac = [r, g] if self.shock else [r]
        nl = [len(self.rlev), len(self.glev)] if self.shock else [len(self.rlev)]
        self.part = GLMM(y, Xp, fac, nl).fit()

        pos = y > 0
        bids = np.minimum(d["bid"].to_numpy(int), d["budget"].to_numpy(int))[pos]
        gb, self.gblev = codes(gl[pos])
        fac_s = [r[pos], gb] if self.shock else [r[pos]]
        nl_s = [len(self.rlev), len(self.gblev)] if self.shock else [len(self.rlev)]
        self.size = LMM(np.log(bids), Xs[pos], fac_s, nl_s).fit()

        # heaping share: ML on training bids under the out-of-group predictive (as at test time)
        newg = [np.full(pos.sum(), -1)] if self.shock else []
        m, v = self.size.predict(Xs[pos], [r[pos]] + newg)
        Bb = d["budget"].to_numpy(int)[pos]
        K = int(Bb.max())
        P0, P1 = size_pmf(m, np.sqrt(v), Bb, K, 0.0), size_pmf(m, np.sqrt(v), Bb, K, 1.0)
        a, b1 = P0[np.arange(len(bids)), bids], P1[np.arange(len(bids)), bids]
        self.pi = float(minimize_scalar(lambda t: -np.log((1 - t) * a + t * b1 + 1e-300).sum(),
                                        bounds=(0.0, 0.6), method="bounded").x)
        if self.variant == "gpv":
            self._fit_qre(d, Xp, Xs, r, gl, pos, bids)
        return self

    def _stage1(self, Xp, Xs, r, n):
        """Ex-ante stage-1 beliefs for chances of a new (or held-out) group: marginal P(bid) with the
        participation shock integrated, and the size mean / sd without the auction shock eta."""
        ng = np.full(n, -1)
        p = self.part.predict(Xp, [r, ng])
        m, v = self.size.predict(Xs, [r, ng], drop=(1,))
        return p, m, np.sqrt(v)

    def _fit_qre(self, d, Xp, Xs, r, gl, pos, bids):
        # training groups with at least one bid; every chance of those groups is a rival
        gb_lab = set(gl[pos])
        keep = np.array([x in gb_lab for x in gl])
        dk = d[keep].reset_index(drop=True)
        gk, _ = codes(gl[keep], self.gblev)
        p, m, s = self._stage1(Xp[keep], Xs[keep], r[keep], int(keep.sum()))
        eta = self.size.u[1][gk]
        Bk = dk["budget"].to_numpy(int)
        K = int(Bk.max())
        S = size_pmf(m + eta, s, Bk, K, self.pi)
        LW = log_win(p, S, gk, len(self.gblev))
        yk = dk["bid"].to_numpy(int) > 0
        LW, Bb, eta_b = LW[yk], Bk[yk], eta[yk]
        bb = np.minimum(dk["bid"].to_numpy(int), Bk)[yk]
        Xv = Xs[keep][yk]
        ub = self.size.u[0][r[keep][yk]]
        nb = len(bb)
        off = eta_b + ub

        def nll(th):
            logv = (off + Xv @ th[:nx])[:, None] + np.exp(th[nx]) * _QX[None, :]
            lp = qre_logp(LW, logv, Bb, np.exp(th[nx + 1]))
            lpb = lp[np.arange(nb), :, bb - 1]
            return -logsumexp(lpb + _LQW[None, :], axis=1).sum()

        nx = Xv.shape[1]
        th0 = np.r_[self.size.beta + np.r_[0.3, np.zeros(nx - 1)], np.log(self.size.s_e), np.log(2.0)]
        bnds = [(None, None)] * nx + [(np.log(0.05), np.log(3.0)), (np.log(0.1), np.log(50.0))]
        res = minimize(nll, th0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 300})
        self.qre = res.x
        self.qre_nll = float(res.fun) / nb

    # ------------------------------------------------------------ predict
    def predict(self, test):
        Xp, Xs = self._design(test)
        r, _ = self._codes(test)
        n = len(test)
        ng = [np.full(n, -1)] if self.shock else []
        p = self.part.predict(Xp, [r] + ng)
        B = test["budget"].to_numpy(int)
        K = int(B.max())
        m, v = self.size.predict(Xs, [r] + ng)
        if self.variant != "gpv":
            return {"p_bid": p, "size_pmf": size_pmf(m, np.sqrt(v), B, K, self.pi)}
        return {"p_bid": p, "size_pmf": self._qre_nodes(test, Xp, Xs, r).mean(1)}

    def _qre_nodes(self, test, Xp, Xs, r):
        """(n, J_ETA, K+1) structural bid-size pmf of every chance given each equal-probability eta node."""
        n = len(test)
        B = test["budget"].to_numpy(int)
        K = int(B.max())
        gidx, glev = codes(group_labels(test))
        p, m, s = self._stage1(Xp, Xs, r, n)
        th = self.qre
        # value: stage-1 manager effect + structural coefficients; parameter / BLUP uncertainty of the
        # stage-1 size mean is added to the value sd
        par_v = np.maximum(s ** 2 - self.size.s_e ** 2, 0.0)
        ur = np.where(r >= 0, self.size.u[0][np.maximum(r, 0)], 0.0)
        nx = Xs.shape[1]
        muv = ur + Xs @ th[:nx]
        sv = np.sqrt(np.exp(th[nx]) ** 2 + par_v)
        se = self.size.s_re[1]
        out = np.zeros((n, J_ETA, K + 1))
        for j, z in enumerate(norm.ppf((np.arange(J_ETA) + 0.5) / J_ETA)):
            eta = se * z
            S = size_pmf(m + eta, s, B, K, self.pi)
            LW = log_win(p, S, gidx, len(glev))
            logv = (eta + muv)[:, None] + sv[:, None] * _QX[None, :]
            lp = qre_logp(LW, logv, B, np.exp(th[nx + 1]))
            out[:, j, 1:] = np.exp(logsumexp(lp + _LQW[None, :, None], axis=1))
        return out

    # ------------------------------------------------------------ joint draws (ah, gpv)
    def _sample_joint(self, test, rng, n):
        Xp, Xs = self._design(test)
        r, _ = self._codes(test)
        nk = len(test)
        B = test["budget"].to_numpy(int)
        gidx, glev = codes(group_labels(test))
        ng = len(glev)
        # stratified group shocks: each group's n draws cover N(0, 1) evenly, in random order
        def strat():
            u = (rng.permuted(np.tile(np.arange(n), (ng, 1)), axis=1) + rng.random((ng, n))) / n
            return u
        uz, ue = strat(), strat()
        lp_m, lp_v = self.part.linpred(Xp, [r, np.full(nk, -1)], drop=(1,))
        zeta = self.part.s_re[1] * norm.ppf(uz)[gidx]
        pb = expit(lp_m[:, None] + np.sqrt(lp_v)[:, None] * rng.standard_normal((nk, n)) + zeta)
        bid = rng.random((nk, n)) < pb
        if self.variant == "gpv":
            nodes = self._qre_nodes(test, Xp, Xs, r)
            jx = np.minimum((ue * J_ETA).astype(int), J_ETA - 1)[gidx]          # (nk, n) node per draw
            C = np.cumsum(nodes, 2)
            C /= C[:, :, -1:]
            u = rng.random((nk, n))
            amt = np.empty((nk, n))
            for k in range(nk):
                for j in range(J_ETA):
                    sel = jx[k] == j
                    amt[k, sel] = np.searchsorted(C[k, j], u[k, sel], side="left")
            amt = np.clip(amt, 1, B[:, None])
        else:
            m, v = self.size.predict(Xs, [r, np.full(nk, -1)], drop=(1,))
            eta = self.size.s_re[1] * norm.ppf(ue)[gidx]
            x = np.exp(m[:, None] + eta + np.sqrt(v)[:, None] * rng.standard_normal((nk, n)))
            amt = self._report(x, rng)
            amt = np.clip(amt, 1, B[:, None])
        return np.where(bid, amt, 0.0)

    def _report(self, x, rng):
        heap = rng.random(x.shape) < self.pi
        h = np.where(x >= HEAP / 2, HEAP * np.round(x / HEAP), np.round(x))
        return np.where(heap, h, np.round(x))
