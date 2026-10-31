"""Bids-track baseline: auction.BidModel's hurdle model, refit on the training legs only.

    P(bid > 0)          = logistic(a0 + a1 log(1 + gain) + a2 log(budget) + v_j),  v_j ~ N(0, 1) penalty
    log(bid) | bid > 0  = b0 + b1 gain + b2 log(budget) + b3 leg + u_j + e,  e ~ N(0, s^2), u_j EB-shrunk
Same optimizer, start point and shrinkage as BidModel.__init__. The bid distribution is the one
BidModel.sample draws: min(round(exp(mu + s z)), budget), so a draw rounding to $0 is no bid.
Variants:
  baseline  the core pipeline as deployed: fit on BidModel's own rows (bid_history, repaired == False)
            with its own feature (gain_core, end-of-week matchup rosters), i.e. exactly auction.BidModel
            on a league cut off before the test leg (verify.py: max |diff| = 0); predicts from the
            point-in-time gain, as run.py does from today's rosters. P(bid > 0) = p_bid x P(round >= 1).
  logit     the same fit, with BidModel.p_bid (the raw logistic) as the participation probability.
  clean     the same spec fit on the harness training frame as a candidate sees it (every chance,
            point-in-time gain).
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

VARIANTS = ("baseline", "logit", "clean")


class BidModelRefit:
    def __init__(self, variant="baseline"):
        assert variant in VARIANTS, variant
        self.variant = variant

    def fit(self, train):
        rows = train[train["budget"].to_numpy() > 0]
        if self.variant == "clean":
            g = rows["gain"]
        else:
            rows = rows[~rows["repaired"].to_numpy(bool)]
            g = rows["gain_core"]
        return self.fit_arrays(rows["rid"].to_numpy(), g.to_numpy(float), rows["budget"].to_numpy(),
                               rows["bid"].to_numpy(), rows["leg"].to_numpy())

    def fit_arrays(self, rid_, gain_, bud_, bid_, leg_):
        """BidModel.__init__ on row arrays (in the caller's row order)."""
        rids = sorted({int(r) for r in rid_})
        ri = {rid: k for k, rid in enumerate(rids)}
        Ri = np.array([ri[int(r)] for r in rid_])
        x1 = np.log1p(gain_)
        x2 = np.log(bud_.astype(float))
        y = (bid_ > 0).astype(float)

        def nll(th):
            v = th[3:]
            eta = th[0] + th[1] * x1 + th[2] * x2 + v[Ri]
            return np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * np.sum(v ** 2)

        th = minimize(nll, np.r_[-6.0, 1.5, 0.3, np.zeros(len(rids))], method="L-BFGS-B").x
        self.a = th[:3]
        self.v = {rid: float(th[3 + ri[rid]]) for rid in rids}

        pos = bid_ > 0
        X = np.column_stack([np.ones(pos.sum()), gain_[pos], np.log(bud_[pos]), leg_[pos]])
        Y = np.log(np.minimum(bid_[pos], bud_[pos]))
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        prid = np.array([int(r) for r in rid_[pos]])
        s2 = res.var()
        means = {rid: (res[prid == rid].mean(), int((prid == rid).sum())) for rid in set(prid)}
        tau2 = max(np.var([m for m, n in means.values()]) - s2 / np.mean([n for m, n in means.values()]), 0.0)
        self.u = {rid: (m * n / (n + s2 / tau2) if tau2 > 0 else 0.0) for rid, (m, n) in means.items()}
        self.beta = beta
        self.sigma = float(np.sqrt(s2))
        self.tau = float(np.sqrt(tau2))
        self.r2 = float(1 - res.var() / Y.var())
        return self

    # BidModel.p_bid / BidModel.mu, vectorized over a frame
    def p_logit(self, d):
        b = np.maximum(d["budget"].to_numpy(float), 1)
        v = np.array([self.v.get(int(r), 0.0) for r in d["rid"]])
        eta = self.a[0] + self.a[1] * np.log1p(d["gain"].to_numpy(float)) + self.a[2] * np.log(b) + v
        return 1 / (1 + np.exp(-eta))

    def mu(self, d):
        b = np.maximum(d["budget"].to_numpy(float), 1)
        u = np.array([self.u.get(int(r), 0.0) for r in d["rid"]])
        return (self.beta[0] + self.beta[1] * d["gain"].to_numpy(float) + self.beta[2] * np.log(b)
                + self.beta[3] * d["leg"].to_numpy(float) + u)

    def amount_pmf(self, d, K=None):
        """(n, K+1) pmf of min(round(exp(mu + s z)), budget) on dollars 0..K (mass at 0 = rounds to $0)."""
        b = d["budget"].to_numpy(int)
        K = int(b.max()) if K is None else K
        mu = self.mu(d)[:, None]
        k = np.arange(K + 1)[None, :]
        hi = norm.cdf((np.log(k + 0.5) - mu) / self.sigma)
        lo = np.where(k > 0, norm.cdf((np.log(np.maximum(k, 1) - 0.5) - mu) / self.sigma), 0.0)
        P = hi - lo
        cap = k >= b[:, None]
        P = np.where(cap, 0.0, P)
        P[np.arange(len(b)), b] = 1 - lo[np.arange(len(b)), b]   # everything from budget - 0.5 up is capped
        return P

    def predict(self, test):
        A = self.amount_pmf(test)
        p0 = A[:, 0].copy()
        S = A.copy()
        S[:, 0] = 0.0
        S /= S.sum(1, keepdims=True)
        p = self.p_logit(test)
        return {"p_bid": p if self.variant == "logit" else p * (1 - p0), "size_pmf": S}
