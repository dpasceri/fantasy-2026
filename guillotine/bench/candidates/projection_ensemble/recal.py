"""Covariate-dependent probit recalibration of a base predictive (a one-component, probit-normal
cousin of the beta-transformed linear pool).

The base PIT u of each outcome is mapped to z = probit(u); an outcome on an atom gives an interval
[probit F(y-), probit F(y)]. We model z ~ N(mu(x), s(x)^2) with mu = Xm b, log s = Xs g, fitted by
weighted interval-censored maximum likelihood with a ridge prior centred on the identity (b = g = 0
reproduces the base exactly). The combined quantile at level tau is Q_base(Phi(mu + s * probit(tau))).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr

LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _logphi(x):
    return np.where(np.isfinite(x), -0.5 * np.where(np.isfinite(x), x, 0) ** 2 - LOG_SQRT_2PI, -np.inf)


def _log_mass(A, B):
    """log(Phi(B) - Phi(A)) for A < B, stable in both tails."""
    up = A > 0
    hi = np.where(up, log_ndtr(-A), log_ndtr(B))
    lo = np.where(up, log_ndtr(-B), log_ndtr(A))
    return hi + np.log1p(-np.exp(np.minimum(lo - hi, -1e-300)))


def nll(theta, Xm, Xs, a, b, w, pen_m, pen_s):
    km = Xm.shape[1]
    beta, gam = theta[:km], theta[km:]
    mu, eta = Xm @ beta, Xs @ gam
    s = np.exp(eta)
    pt = a == b
    dmu, deta = np.zeros_like(mu), np.zeros_like(mu)
    ll = np.zeros_like(mu)
    if pt.any():
        r = (a[pt] - mu[pt]) / s[pt]
        ll[pt] = -eta[pt] - 0.5 * r * r
        dmu[pt] = r / s[pt]
        deta[pt] = r * r - 1.0
    iv = ~pt
    if iv.any():
        A, B = (a[iv] - mu[iv]) / s[iv], (b[iv] - mu[iv]) / s[iv]
        lP = _log_mass(A, B)
        ll[iv] = lP
        ra = np.exp(_logphi(A) - lP)                       # phi(A) / P
        rb = np.exp(_logphi(B) - lP)
        Af, Bf = np.where(np.isfinite(A), A, 0.0), np.where(np.isfinite(B), B, 0.0)
        dmu[iv] = (ra - rb) / s[iv]
        deta[iv] = Af * ra - Bf * rb
    f = -np.sum(w * ll) + 0.5 * np.sum(pen_m * beta ** 2) + 0.5 * np.sum(pen_s * gam ** 2)
    g = np.concatenate([-(Xm.T @ (w * dmu)) + pen_m * beta, -(Xs.T @ (w * deta)) + pen_s * gam])
    return f, g


class ProbitRecal:
    def fit(self, Xm, Xs, a, b, w, pen_m, pen_s):
        th0 = np.zeros(Xm.shape[1] + Xs.shape[1])
        r = minimize(nll, th0, args=(Xm, Xs, a, b, w, pen_m, pen_s), jac=True, method="L-BFGS-B",
                     bounds=[(None, None)] * Xm.shape[1] + [(-3, 3)] * Xs.shape[1],
                     options={"maxiter": 2000, "gtol": 1e-7})
        self.km = Xm.shape[1]
        self.beta, self.gam = r.x[:self.km], r.x[self.km:]
        self.ok, self.nit = bool(r.success), int(r.nit)
        return self

    def levels(self, Xm, Xs, tau):
        """(n, L) base levels u = Phi(mu + s * probit(tau)) for target levels tau."""
        from scipy.special import ndtri
        mu, s = Xm @ self.beta, np.exp(Xs @ self.gam)
        return ndtr(mu[:, None] + s[:, None] * ndtri(np.asarray(tau, float))[None, :])


def logit_nll(d, X, off, e, w, pen):
    eta = off + X @ d
    lp1, lp0 = -np.logaddexp(0, -eta), -np.logaddexp(0, eta)     # log p, log(1 - p)
    p = np.exp(lp1)
    f = -np.sum(w * (e * lp1 + (1 - e) * lp0)) + 0.5 * np.sum(pen * d ** 2)
    return f, X.T @ (w * (p - e)) + pen * d


class LogitOffset:
    """Ridge logistic regression with an offset: logit P(event) = off + X d (d = 0 keeps the base)."""

    def fit(self, X, off, e, w, pen):
        r = minimize(logit_nll, np.zeros(X.shape[1]), args=(X, off, e, w, pen), jac=True, method="L-BFGS-B",
                     options={"maxiter": 2000, "gtol": 1e-7})
        self.d, self.ok = r.x, bool(r.success)
        return self

    def prob(self, X, off):
        return 1.0 / (1.0 + np.exp(-(off + X @ self.d)))
