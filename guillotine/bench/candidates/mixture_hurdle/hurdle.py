"""Player-track candidate: GAMLSS-style hurdle distributional regression, refit per origin.

Three-part mixture per position, fit by penalized maximum likelihood:
  P(y < 0), P(y = 0), P(y > 0)  multinomial logit: zero logit a P-spline in x = log proj, negative
                                logit linear in x (rare: a few dozen rows)
  y | y < 0                     -Exponential(m), m pooled over positions with a weak prior
  y | y > 0                     a continuous skewed law with parameters smooth in x:
    "zaga"       gamma (GAMLSS GA): log mu, log sigma P-splines
    "zabct"      Box-Cox t (GAMLSS BCT): log mu, log sigma P-splines, nu linear in x, tau constant
    "zabct_cov"  zabct plus ridge-penalized linear terms on the zero logit only: log team_pos_rank,
                 TD / reception / rushing shares of the projection, log team projection
Quantiles are exact: levels inside the atom are exactly 0.0 (so the harness PIT sees the tie); the
rest map through the conditional part's quantile function.
P-splines (Eilers & Marx 1996): cubic B-splines on fixed knots in log proj with a 2nd-difference
penalty whose weight is picked per fit and per block by AIC (edf = tr((H + P)^-1 H)), so nothing is
tuned outside the training rows. Gradients: analytic for the hurdle; for the continuous part, central
differences on each row's linear predictors (the likelihood is separable by row), chained through the
design matrices. Deterministic (no random numbers).
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)

import numpy as np
from scipy import special as sp
from scipy.interpolate import BSpline
from scipy.optimize import minimize

POS = ("QB", "RB", "WR", "TE")
XLO, XHI = np.log(0.5), np.log(30.0)       # basis support in log proj (items have proj >= 0.5)
NSEG, DEG = 4, 3                           # 4 segments, cubic -> 7 basis functions
LAM_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # 2nd-difference spline penalties, chosen per fit and block by AIC
NU_RIDGE = (1e-2, 10.0)                    # ridge on BCT nu (intercept, slope in x): shape kept parsimonious
LAM_COV = 25.0                             # ridge on standardized covariate coefs (zabct_cov, zero logit)
NEG_M0, NEG_K0 = 1.0, 5.0                  # prior mean |y| of negatives and its pseudo-count
DEBUG = False
COVS = ("lrank", "td", "rec", "rush", "ltp")


# ---------------------------------------------------------------- basis
def _knots():
    h = (XHI - XLO) / NSEG
    return XLO + h * np.arange(-DEG, NSEG + DEG + 1)


def bbasis(x):
    """(n, NSEG + DEG) cubic B-spline basis of clamped x (partition of unity)."""
    x = np.clip(np.asarray(x, float), XLO, XHI - 1e-9)
    return BSpline.design_matrix(x, _knots(), DEG).toarray()


def greville():
    t = _knots()
    return np.array([t[i + 1:i + DEG + 1].mean() for i in range(NSEG + DEG)])


def d2pen(k):
    D = np.diff(np.eye(k), 2, axis=0)
    return D.T @ D


def features(df):
    """Covariates for zabct_cov (pre-game columns only)."""
    p = df["proj"].to_numpy(float)
    g = lambda c: df[c].to_numpy(float) if c in df else np.zeros(len(df))  # noqa: E731
    rank = df["team_pos_rank"].to_numpy(float)
    tp = df["team_proj"].to_numpy(float)
    return {"lrank": np.log(np.where(np.isfinite(rank) & (rank > 0), rank, 1.0)),
            "td": np.clip((6 * (g("ps_rec_td") + g("ps_rush_td")) + 4 * g("ps_pass_td")) / p, 0, 1.5),
            "rec": np.clip(0.5 * g("ps_rec") / p, 0, 1.5),
            "rush": np.clip(0.1 * g("ps_rush_yd") / p, 0, 1.5),
            "ltp": np.log(np.where(np.isfinite(tp) & (tp > 1), tp, 1.0))}


# ---------------------------------------------------------------- generic penalized fit
class Spec:
    """A block of linear predictors: eta_j = X_j @ theta_j; quadratic penalty 0.5 theta' P theta."""

    def __init__(self, Xs, Ps, bounds, theta0):
        self.Xs, self.Ps = Xs, Ps
        self.sizes = [X.shape[1] for X in Xs]
        self.cut = np.cumsum([0] + self.sizes)
        self.bounds, self.theta0 = bounds, theta0

    def split(self, th):
        return [th[a:b] for a, b in zip(self.cut[:-1], self.cut[1:])]

    def etas(self, th):
        return [X @ t for X, t in zip(self.Xs, self.split(th))]


def row_terms(spec, th, row_nll, analytic=None, h=1e-5):
    """Per-row nll and its derivatives wrt each linear predictor (analytic, or central differences:
    the likelihood is separable by row, so one shift per predictor gives every row's derivative)."""
    E = spec.etas(th)
    if analytic is not None:
        return analytic(E)
    v, G = row_nll(E), []
    for j in range(len(E)):
        Ep, Em = list(E), list(E)
        Ep[j], Em[j] = E[j] + h, E[j] - h
        G.append((row_nll(Ep) - row_nll(Em)) / (2 * h))
    return v, G


def pen_fit(spec, row_nll, analytic=None, maxiter=500):
    """Minimize sum(row_nll(etas)) + sum_j 0.5 theta_j' P_j theta_j by L-BFGS-B (chain rule X_j' dnll/deta_j)."""
    def f(th):
        v, G = row_terms(spec, th, row_nll, analytic)
        th_s = spec.split(th)
        val = v.sum() + sum(0.5 * t @ P @ t for t, P in zip(th_s, spec.Ps))
        if not np.isfinite(val):
            return 1e300, np.zeros_like(th)
        return val, np.concatenate([X.T @ g + P @ t for X, g, P, t in zip(spec.Xs, G, spec.Ps, th_s)])
    r = minimize(f, spec.theta0, jac=True, method="L-BFGS-B", bounds=spec.bounds,
                 options={"maxiter": maxiter, "maxcor": 20})
    if DEBUG:
        print("   fit", r.nit, r.nfev, round(float(r.fun), 2), r.message, flush=True)
    return r.x


def aic_fit(make, row_nll, analytic=None, info=None, grid=None):
    """Fit make(lam) -> Spec over the penalty grid (warm starts) and keep the lowest AIC = 2 nll + 2 edf,
    edf = tr((H + P)^-1 H). H = info(spec, theta) if given (expected information), else the outer product
    of the per-row scores (BHHH). Returns (spec, theta, lam)."""
    best, th0 = None, None
    for lam in (LAM_GRID if grid is None else grid):
        spec = make(lam)
        if th0 is not None:
            spec.theta0 = th0
        th = pen_fit(spec, row_nll, analytic)
        v, G = row_terms(spec, th, row_nll, analytic)
        if info is not None:
            H = info(spec, th)
        else:
            Sc = np.hstack([X * g[:, None] for X, g in zip(spec.Xs, G)])
            H = Sc.T @ Sc
        P = _blkn(spec.Ps)
        try:
            edf = float(np.trace(np.linalg.solve(H + P + 1e-9 * np.eye(len(P)), H)))
        except np.linalg.LinAlgError:
            edf = float(len(th))
        aic = 2 * v.sum() + 2 * edf
        if DEBUG:
            print(f"   lam {lam:g} nll {v.sum():.2f} edf {edf:.2f} aic {aic:.2f}", flush=True)
        if best is None or aic < best[0]:
            best = (aic, spec, th, lam)
        th0 = th
    return best[1], best[2], best[3]


# ---------------------------------------------------------------- hurdle (multinomial: neg, zero, pos)
def hurdle_probs(en, ez):
    m = np.maximum(np.maximum(en, ez), 0.0)
    a, b, c = np.exp(en - m), np.exp(ez - m), np.exp(-m)
    s = a + b + c
    return a / s, b / s, c / s


def hurdle_analytic(cls):
    def f(E):
        pn, pz, pp = hurdle_probs(E[0], E[1])
        P = np.stack([pn, pz, pp], 1)
        nll = -np.log(np.maximum(P[np.arange(len(cls)), cls], 1e-300))
        return nll, [pn - (cls == 0), pz - (cls == 1)]
    return f


# ---------------------------------------------------------------- continuous parts (y > 0)
LOG2PI = np.log(2 * np.pi)


def ga_nll(y, lmu, lsig):
    """GAMLSS GA: mean mu, Var = sigma^2 mu^2 (shape 1/sigma^2)."""
    lsig = np.clip(lsig, -4, 2)
    k = np.exp(-2 * lsig)
    lsc = lmu + 2 * lsig                     # log scale
    return -((k - 1) * np.log(y) - y * np.exp(-lsc) - k * lsc - sp.gammaln(k))


def ga_ppf(u, lmu, lsig):
    lsig = np.clip(lsig, -4, 2)
    k = np.exp(-2 * lsig)[:, None]
    return sp.gammaincinv(k, u) * np.exp(lmu + 2 * lsig)[:, None]


def _bc_z(ly, lmu, sig, nu):
    l = ly - lmu
    small = np.abs(nu) < 1e-6
    nus = np.where(small, 1.0, nu)
    return np.where(small, l / sig, np.expm1(nus * l) / (nus * sig))


def bct_nll(y, lmu, lsig, nu, ltau):
    """GAMLSS BCT (Rigby & Stasinopoulos 2006): z = ((y/mu)^nu - 1)/(nu sigma) ~ t_tau truncated."""
    lsig = np.clip(lsig, -4, 2)
    sig, tau = np.exp(lsig), np.exp(ltau)
    nu = np.clip(nu, -2.0, 3.0)
    ly = np.log(y)
    z = _bc_z(ly, lmu, sig, nu)
    lt = (sp.gammaln((tau + 1) / 2) - sp.gammaln(tau / 2) - 0.5 * np.log(np.pi * tau)
          - (tau + 1) / 2 * np.log1p(z * z / tau))
    trunc = np.log(sp.stdtr(tau, np.minimum(1.0 / (sig * np.maximum(np.abs(nu), 1e-12)), 1e12)))
    return -((nu - 1) * ly - nu * lmu - lsig + lt - trunc)


def bct_ppf(u, lmu, lsig, nu, ltau):
    """(n, L) quantiles of BCT at probabilities u (n, L)."""
    lsig = np.clip(lsig, -4, 2)
    sig, tau = np.exp(lsig)[:, None], np.exp(np.asarray(ltau, float))[..., None]
    nu = np.clip(nu, -2.0, 3.0)[:, None]
    lmu = lmu[:, None]
    a = np.minimum(1.0 / (sig * np.maximum(np.abs(nu), 1e-12)), 1e12)
    Fa = sp.stdtr(tau, a)
    p = np.where(nu > 0, sp.stdtr(tau, -a) + u * Fa, u * Fa)
    z = sp.stdtrit(tau, np.clip(p, 1e-12, 1 - 1e-12))
    small = np.abs(nu) < 1e-6
    nus = np.where(small, 1.0, nu)
    w = np.maximum(1.0 + sig * nus * z, 1e-12)
    ly = np.where(small, lmu + sig * z, lmu + np.log(w) / nus)
    return np.exp(np.minimum(ly, np.log(200.0)))


# ---------------------------------------------------------------- candidate
def hurdle_info(spec, th):
    """Expected information of the multinomial hurdle (neg, zero vs pos reference)."""
    L, Xz = spec.Xs
    pn, pz, _ = hurdle_probs(*spec.etas(th))
    Hnz = -L.T @ (Xz * (pn * pz)[:, None])
    return np.block([[L.T @ (L * (pn * (1 - pn))[:, None]), Hnz],
                     [Hnz.T, Xz.T @ (Xz * (pz * (1 - pz))[:, None])]])


def hurdle_fit(L, Xz, cls, Sz, nz, k):
    """Penalized multinomial hurdle: neg logit linear in x, zero logit P-spline (+ ridge covariates);
    the zero-logit spline penalty is picked by AIC over LAM_GRID. Returns (spec, theta, lam)."""
    p0 = np.clip((cls == 1).mean(), 0.02, 0.5)
    th0 = np.concatenate([[-4.5, 0.0], np.full(k, np.log(p0 / (1 - p0))), np.zeros(nz)])
    bounds = [(-12, 4), (-3, 3)] + [(-10, 6)] * k + [(None, None)] * nz
    Pn = np.diag([1e-2, 1.0])
    make = lambda lam: Spec([L, Xz], [Pn, _blkn([lam * Sz, LAM_COV * np.eye(nz)])], bounds, th0.copy())  # noqa: E731
    return aic_fit(make, None, analytic=hurdle_analytic(cls), info=hurdle_info)


class MixtureHurdle:
    def __init__(self, variant="zabct"):
        assert variant in ("zaga", "zabct", "zabct_cov"), variant
        self.variant = variant
        self.cov = variant == "zabct_cov"             # covariates enter the zero logit only

    def _design(self, df, P, fit=False):
        """Spline basis B, linear-in-x design L, hurdle design Xz (= B, or B + standardized covariates)."""
        x = np.clip(np.log(df["proj"].to_numpy(float)), XLO, XHI)
        B = bbasis(x)
        L = np.stack([np.ones(len(x)), x - 2.0], 1)
        if not self.cov:
            return B, L, B
        F = features(df)
        Z = np.stack([F[c] for c in COVS], 1)
        if fit:
            mu, sd = Z.mean(0), Z.std(0)
            self.zstd[P] = (mu, np.where(sd > 1e-6, sd, np.inf))
        mu, sd = self.zstd[P]
        return B, L, np.hstack([B, (Z - mu) / sd])

    def fit(self, train):
        d = train[train["item"].to_numpy(bool)]
        y_all = d["actual"].to_numpy(float)
        neg = -y_all[y_all < 0]
        self.neg_m = (neg.sum() + NEG_M0 * NEG_K0) / (len(neg) + NEG_K0)
        self.zstd, self.par, self.lam, self.lam_c = {}, {}, {}, {}
        k = NSEG + DEG
        Sk = d2pen(k)
        g = greville()
        for P in POS:
            dp = d[d["pos"].to_numpy() == P]
            if len(dp) < 20:
                continue
            y = dp["actual"].to_numpy(float)
            B, L, Xz = self._design(dp, P, fit=True)
            # hurdle: neg logit linear in x; zero logit P-spline (+ covariates), AIC-chosen penalty
            cls = np.where(y < 0, 0, np.where(y == 0, 1, 2))
            spec, th_h, self.lam[P] = hurdle_fit(L, Xz, cls, Sk, Xz.shape[1] - k, k)
            # continuous part on y > 0
            m = y > 0
            yp, Bp, Lp = y[m], B[m], L[m]
            bm = [(None, None)] * k
            th0 = np.concatenate([g, np.full(k, np.log(0.6))])
            if self.variant == "zaga":
                make = lambda lam: Spec([Bp, Bp], [lam * Sk, lam * Sk], bm + bm, th0.copy())  # noqa: E731
                nll = lambda E: ga_nll(yp, E[0], E[1])  # noqa: E731
            else:
                one = np.ones((len(yp), 1))
                make = lambda lam: Spec(  # noqa: E731
                    [Bp, Bp, Lp, one], [lam * Sk, lam * Sk, np.diag(NU_RIDGE), np.zeros((1, 1))],
                    bm + bm + [(-1.5, 2.5), (-1.5, 1.5), (np.log(3.0), np.log(100.0))],
                    np.concatenate([th0, [0.3, 0.0], [np.log(10.0)]]))
                nll = lambda E: bct_nll(yp, E[0], E[1], E[2], E[3])  # noqa: E731
            spec_c, th_c, self.lam_c[P] = aic_fit(make, nll)
            self.par[P] = (spec, th_h, spec_c, th_c)
        return self

    def _parts(self, df, P):
        """(pn, pz, eta list of the continuous part) for one position's rows."""
        spec, th_h, spec_c, th_c = self.par[P]
        B, L, Xz = self._design(df, P)
        th_n, th_z = spec.split(th_h)
        pn, pz, _ = hurdle_probs(L @ th_n, Xz @ th_z)
        tc = spec_c.split(th_c)
        E = [B @ tc[0], B @ tc[1]]
        if self.variant != "zaga":
            E += [L @ tc[2], np.full(len(B), tc[3][0])]
        return pn, pz, E

    def predict_quantiles(self, test, levels):
        lv = np.asarray(levels, float)
        q = np.zeros((len(test), len(lv)))
        pos = test["pos"].to_numpy()
        for P in self.par:
            idx = np.flatnonzero(pos == P)
            if not len(idx):
                continue
            pn, pz, E = self._parts(test.iloc[idx], P)
            T = np.broadcast_to(lv, (len(idx), len(lv)))
            pn_, pz_ = pn[:, None], pz[:, None]
            u = np.clip((T - pn_ - pz_) / (1 - pn_ - pz_), 1e-9, 1 - 1e-9)
            qp = ga_ppf(u, *E) if self.variant == "zaga" else bct_ppf(u, *E)
            qn = self.neg_m * np.log(np.clip(T / np.maximum(pn_, 1e-300), 1e-12, 1.0))
            q[idx] = np.where(T <= pn_, qn, np.where(T <= pn_ + pz_, 0.0, qp))
        return q


def _blkn(Ms):
    """Block-diagonal matrix of square blocks (empty blocks allowed)."""
    n = sum(M.shape[0] for M in Ms)
    out, i = np.zeros((n, n)), 0
    for M in Ms:
        out[i:i + M.shape[0], i:i + M.shape[0]] = M
        i += M.shape[0]
    return out
