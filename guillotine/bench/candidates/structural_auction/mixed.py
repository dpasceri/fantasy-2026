"""Small mixed models with crossed random intercepts (structural_auction candidate).

LMM:  y = X b + sum_k Z_k u_k + e,  u_k ~ N(0, s_k^2 I), e ~ N(0, s_e^2). REML on the dense marginal
      covariance (n is ~150 bids), with a Gamma(2, RATE) boundary-avoiding prior on every random-effect
      sd (Chung et al. 2013). The posterior of (b, u) comes from Henderson's mixed-model equations
      (flat prior on b), so predictions carry fixed-effect and BLUP uncertainty, and a factor level
      not seen in training adds its variance s_k^2.
GLMM: logistic with the same random-effect structure. Laplace approximation of the marginal likelihood
      with the fixed effects integrated too (N(0, PRIOR_B^2) prior), mode by damped Newton on sparse
      designs, the same Gamma(2) prior on the sds. Predictive P(y = 1) integrates the Gaussian posterior
      of the linear predictor (plus unseen-level variance) by Gauss-Hermite quadrature.
No random numbers; deterministic given the row order.
"""
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

RATE = 0.5           # Gamma(2, RATE) prior on each random-effect sd (mode-avoiding; weak at sd ~ 0.3-2)
PRIOR_B = 10.0       # N(0, PRIOR_B^2) on GLMM fixed effects (numerical safety only)
_GX, _GW = np.polynomial.hermite_e.hermegauss(24)
_GW = _GW / _GW.sum()


def codes(labels, levels=None):
    """Labels -> integer codes against `levels` (-1 = unseen); levels default to the sorted uniques."""
    lab = np.asarray(labels).astype(str)
    if levels is None:
        levels = np.unique(lab)
    pos = {v: i for i, v in enumerate(levels)}
    return np.array([pos.get(v, -1) for v in lab], int), levels


def _onehot(c, q):
    Z = np.zeros((len(c), q))
    ok = c >= 0
    Z[np.flatnonzero(ok), c[ok]] = 1.0
    return Z


def _sd_prior(s):
    """-log Gamma(2, RATE) density of the sds (up to a constant)."""
    s = np.asarray(s, float)
    return float(np.sum(-np.log(s) + RATE * s))


def gh_logistic(m, v):
    """E[expit(m + sqrt(v) Z)], Z ~ N(0, 1), by Gauss-Hermite."""
    return expit(np.asarray(m)[:, None] + np.sqrt(np.maximum(v, 0.0))[:, None] * _GX[None, :]) @ _GW


class LMM:
    def __init__(self, y, X, factors, nlev):
        self.y, self.X = np.asarray(y, float), np.asarray(X, float)
        self.f, self.q = [np.asarray(c, int) for c in factors], list(nlev)

    def fit(self):
        y, X = self.y, self.X
        n, p = X.shape
        Zs = [_onehot(c, q) for c, q in zip(self.f, self.q)]
        ZZ = [Z @ Z.T for Z in Zs]
        K = len(Zs)
        I = np.eye(n)

        def obj(th):
            s = np.exp(th)
            V = s[-1] ** 2 * I + sum(s[k] ** 2 * ZZ[k] for k in range(K))
            c = cho_factor(V, lower=True)
            ViX, Viy = cho_solve(c, X), cho_solve(c, y)
            A = X.T @ ViX
            ca = cho_factor(A, lower=True)
            b = cho_solve(ca, X.T @ Viy)
            r = y - X @ b
            ld = 2 * np.log(np.diag(c[0])).sum() + 2 * np.log(np.diag(ca[0])).sum()
            return 0.5 * (ld + r @ cho_solve(c, r)) + _sd_prior(s[:K])

        th0 = np.r_[np.full(K, np.log(0.5)), np.log(max(y.std(), 0.1))]
        th = minimize(obj, th0, method="L-BFGS-B", bounds=[(np.log(1e-3), np.log(10.0))] * (K + 1)).x
        s = np.exp(th)
        self.s_re, self.s_e = s[:K], float(s[-1])
        # Henderson: posterior of (b, u) given the variance components, flat prior on b
        Z = np.hstack(Zs)
        gi = np.concatenate([np.full(q, self.s_e ** 2 / self.s_re[k] ** 2) for k, q in enumerate(self.q)])
        Mm = np.block([[X.T @ X, X.T @ Z], [Z.T @ X, Z.T @ Z + np.diag(gi)]])
        Mi = np.linalg.inv(Mm)
        self.sol = Mi @ np.r_[X.T @ y, Z.T @ y]
        self.C = self.s_e ** 2 * Mi
        self.beta = self.sol[:p]
        off = np.cumsum([p] + self.q)
        self.u = [self.sol[off[k]:off[k + 1]] for k in range(K)]
        return self

    def predict(self, X, factors, drop=()):
        """Predictive mean and variance of a new y (e included). factors: code arrays (-1 = unseen level,
        which adds that factor's s^2). drop: factor indices left out of the mean and variance entirely."""
        X = np.asarray(X, float)
        A = [X] + [_onehot(np.where(np.asarray(c) >= 0, c, -1), q) for c, q in zip(factors, self.q)]
        A = np.hstack(A)
        m = A @ self.sol
        v = np.einsum("ij,jk,ik->i", A, self.C, A) + self.s_e ** 2
        for k, c in enumerate(factors):
            if k not in drop:
                v = v + (np.asarray(c) < 0) * self.s_re[k] ** 2
        return m, v


class GLMM:
    def __init__(self, y, X, factors, nlev):
        self.y, self.X = np.asarray(y, float), np.asarray(X, float)
        self.f, self.q = [np.asarray(c, int) for c in factors], list(nlev)
        n, p = self.X.shape
        cols = [sparse.csr_matrix(self.X)]
        for c, q in zip(self.f, self.q):
            cols.append(sparse.csr_matrix((np.ones(n), (np.arange(n), c)), shape=(n, q)))
        self.A = sparse.hstack(cols).tocsr()
        self.p = p
        self._th = np.zeros(self.A.shape[1])

    def _pdiag(self, s):
        return np.concatenate([np.full(self.p, 1 / PRIOR_B ** 2)] + [np.full(q, 1 / s[k] ** 2) for k, q in enumerate(self.q)])

    def _f(self, th, P):
        eta = self.A @ th
        return np.sum(np.logaddexp(0, eta) - self.y * eta) + 0.5 * np.sum(P * th * th)

    def _mode(self, s):
        A, y, P = self.A, self.y, self._pdiag(s)
        th = self._th.copy()
        f = self._f(th, P)
        for _ in range(100):
            mu = expit(A @ th)
            g = A.T @ (mu - y) + P * th
            H = (A.T @ sparse.diags(mu * (1 - mu)) @ A).toarray() + np.diag(P)
            c = cho_factor(H, lower=True)
            step = cho_solve(c, g)
            t = 1.0
            while True:
                th2 = th - t * step
                f2 = self._f(th2, P)
                if f2 <= f + 1e-10 or t < 1e-4:
                    break
                t *= 0.5
            th, df, f = th2, f - f2, f2
            if np.max(np.abs(t * step)) < 1e-8 or df < 1e-12:
                break
        mu = expit(A @ th)
        H = (A.T @ sparse.diags(mu * (1 - mu)) @ A).toarray() + np.diag(P)
        self._th = th
        return th, cho_factor(H, lower=True), f

    def _neg_marginal(self, logs):
        s = np.exp(np.atleast_1d(logs))
        _, c, f = self._mode(s)
        return f + np.log(np.diag(c[0])).sum() + sum(q * np.log(s[k]) for k, q in enumerate(self.q)) + _sd_prior(s)

    def fit(self):
        K = len(self.q)
        lo, hi = np.log(0.05), np.log(5.0)
        if K == 1:
            r = minimize_scalar(self._neg_marginal, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
            logs = np.array([r.x])
        else:
            logs = minimize(self._neg_marginal, np.full(K, np.log(0.7)), method="L-BFGS-B",
                            bounds=[(lo, hi)] * K, options={"eps": 1e-4}).x
        self.s_re = np.exp(logs)
        th, c, _ = self._mode(self.s_re)
        self.th = th
        self.C = cho_solve(c, np.eye(len(th)))
        self.beta = th[:self.p]
        off = np.cumsum([self.p] + self.q)
        self.u = [th[off[k]:off[k + 1]] for k in range(K)]
        return self

    def linpred(self, X, factors, drop=()):
        """Posterior mean and variance of the linear predictor for new rows (unseen levels add s^2;
        factors in `drop` are left out, e.g. to draw them jointly elsewhere)."""
        X = np.asarray(X, float)
        A = np.hstack([X] + [_onehot(np.asarray(c), q) for c, q in zip(factors, self.q)])
        m = A @ self.th
        v = np.einsum("ij,jk,ik->i", A, self.C, A)
        for k, c in enumerate(factors):
            if k not in drop:
                v = v + (np.asarray(c) < 0) * self.s_re[k] ** 2
        return m, v

    def predict(self, X, factors):
        m, v = self.linpred(X, factors)
        return gh_logistic(m, v)
