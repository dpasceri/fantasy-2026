"""Player track: hierarchical / empirical-Bayes player effects on a hurdle base.

Base (the position x projection prior; no player identity), variant "hurdle":
  the baseline's buckets (position, equal-count projection buckets, nb = clip(n // 300, 5, 14)),
  each split into its zero share P(y <= 0) and a quantile table of actual/proj among positive
  scores. Both are interpolated in log proj between bucket medians, so the atom is interpolated
  explicitly (the baseline's interpolated tables keep only the smaller neighbouring atom).
  y | y <= 0 is the position's pooled empirical non-positive scores (mostly an exact 0.0).
Effects, all empirical Bayes on decay-weighted history (variants eb, dlm):
  zero propensity: random-intercept logistic, logit p = logit p0_base + b_player, b ~ N(0, sig^2);
    sig by Gauss-Hermite marginal likelihood, posterior predictive by quadrature;
  location: crossed normal-normal random effects on the probit PIT z of positive scores,
    z = player + team-season + opponent-season-position + eps; each tau^2 by marginal likelihood
    on the groups' weighted means (James-Stein-type shrinkage), fitted by backfitting;
  z-scale level: residual variance of z by position (the interpolated base is a little too narrow);
  player scale (dlm): per-player residual variance squeezed toward an inverse-gamma prior
    (Smyth 2004 limma squeezeVar: prior df and scale by log-moment matching).
  Predictive z ~ N(m, s^2): q(tau) = F_pos^-1(Phi(m + s Phi^-1(v))), v = (tau - p0) / (1 - p0).
Decay: weight 2^(-age/h) with an offseason gap (team / opponent effects reset each season).
  eb: fixed a-priori half-lives (8 weeks). dlm: half-lives and the variance-prior strength chosen per
  origin by one-step-ahead (prequential) likelihood over the training weeks, the discount-factor
  reading of a local-level state-space model.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)

import numpy as np
import pandas as pd
from scipy.special import digamma, ndtr, ndtri, polygamma
from common import POS
from model import LEVELS as GRID, Q
from bench.candidates._baseline_player.ratio import ratio_grid

VARIANTS = ("hurdle", "eb", "dlm")
INF = np.inf
EB_H = {"player": 8.0, "team": 8.0, "zero": 8.0}     # a-priori half-lives (weeks)
GAP = {"player": 8.0, "team": INF, "zero": 8.0}      # extra age across a season boundary
H_PLAYER, H_TEAM, H_ZERO = (4.0, 8.0, 16.0, INF), (4.0, 8.0, INF), (2.0, 4.0, 8.0, 16.0, INF)
D0_MULT = (1.0, 3.0, 10.0, INF)                      # inf = no player scale
PREQ_WEEKS, PREQ_MIN_PRIOR = 16, 3
TAU2 = np.r_[0.0, np.geomspace(1e-4, 1.0, 80)]
SIG = np.r_[0.0, np.geomspace(0.05, 2.0, 29)]
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(16)
GH_W = GH_W / GH_W.sum()
V_EPS = 1e-6
LOC = ("player", "team", "opp")                       # crossed location effects
CFG = {"hurdle": dict(effects=()),
       "eb": dict(effects=("zero", "loc")),
       "dlm": dict(effects=("zero", "loc", "scale"), select=True)}


# ---------------------------------------------------------------- base: hurdle prior
def interp_rows(centers, tables, proj):
    """Linear interpolation of per-bucket rows in log proj between bucket medians (flat outside)."""
    if len(centers) == 1:
        return np.repeat(tables[:1], len(proj), 0)
    x, lc = np.log(proj), np.log(centers)
    k = np.clip(np.searchsorted(lc, x) - 1, 0, len(lc) - 2)
    den = lc[k + 1] - lc[k]
    w = np.clip(np.divide(x - lc[k], den, out=np.zeros_like(x), where=den > 0), 0.0, 1.0)
    return (1 - w)[:, None] * tables[k] + w[:, None] * tables[k + 1]


class HurdleBase:
    """Baseline buckets split at the atom: zero share + positive ratio table per bucket."""

    def fit(self, pos, proj, y):
        self.tables, self.atoms, self.neg = {}, {}, {}
        ratio = y / proj
        for P in POS:
            m = np.flatnonzero(pos == P)
            if not len(m):
                continue
            nb = int(np.clip(len(m) // 300, 5, 14))
            edges = np.quantile(proj[m], np.linspace(0, 1, nb + 1))
            b = np.clip(np.searchsorted(edges, proj[m], side="right") - 1, 0, nb - 1)
            cs, ts, ps = [], [], []
            for k in range(nb):
                idx = m[b == k]
                if not len(idx):            # tied edges on a tiny training set
                    continue
                r = ratio[idx]
                rp = np.sort(r[r > 0])
                cs.append(np.median(proj[idx]))
                ps.append((r <= 0).mean())
                ts.append(np.quantile(rp, GRID) if len(rp) else np.zeros(Q))
            self.tables[P] = (np.array(cs), np.array(ts))
            self.atoms[P] = (np.array(cs), np.array(ps)[:, None])
            nv = np.sort(y[m][y[m] <= 0])
            self.neg[P] = nv if len(nv) else np.zeros(1)
        return self

    def p0(self, pos, proj):
        out = np.full(len(proj), 0.1)
        for P, a in self.atoms.items():
            m = pos == P
            if m.any():
                out[m] = interp_rows(*a, proj[m])[:, 0]
        return np.clip(out, 1e-5, 0.97)

    def ratio(self, pos, proj):
        return ratio_grid(self.tables, pos, proj)

    def z_pos(self, pos, proj, y):
        """Probit PIT of positive scores under the positive-part table (NaN where y <= 0)."""
        z = np.full(len(y), np.nan)
        k = np.flatnonzero(y > 0)
        if not len(k):
            return z
        R = self.ratio(pos[k], proj[k])
        r = y[k] / proj[k]
        j = np.clip((R < r[:, None]).sum(1) - 1, 0, Q - 2)
        lo, hi = R[np.arange(len(k)), j], R[np.arange(len(k)), j + 1]
        fr = np.clip(np.divide(r - lo, hi - lo, out=np.full(len(k), 0.5), where=hi > lo), 0, 1)
        z[k] = ndtri(np.clip((j + fr) / (Q - 1), 0.5 / Q, 1 - 0.5 / Q))
        return z

    def quantiles(self, pos, proj, p0, m, s, levels):
        """(n, L) quantiles: atom / negatives below p0, the shifted positive part above."""
        n, L = len(proj), len(levels)
        tau = np.broadcast_to(np.asarray(levels, float), (n, L))
        p0 = p0[:, None]
        v = np.clip((tau - p0) / (1 - p0), V_EPS, 1 - V_EPS)
        u = ndtr(m[:, None] + s[:, None] * ndtri(v)) * (Q - 1)   # probit location-scale move
        i0 = np.clip(u.astype(np.int64), 0, Q - 2)
        fr = u - i0
        R = self.ratio(pos, proj)
        qp = proj[:, None] * (np.take_along_axis(R, i0, 1) * (1 - fr) + np.take_along_axis(R, i0 + 1, 1) * fr)
        qn = np.zeros((n, L))
        w = np.clip(tau / p0, 0, 1)
        for P, nv in self.neg.items():
            k = pos == P
            if k.any():
                qn[k] = np.interp(w[k], np.linspace(0, 1, len(nv)), nv) if len(nv) > 1 else nv[0]
        return np.where(tau <= p0, qn, qp)


# ---------------------------------------------------------------- empirical-Bayes pieces
def decay(t, season, T, sT, h, gap):
    age = (T - t) + np.where(season != sT, gap, 0.0)
    if not np.isfinite(h):
        return np.where(np.isfinite(age), 1.0, 0.0)
    return np.exp2(-age / h)


def eb_normal(g, r, w, ng, s2):
    """Normal-normal EB over groups: tau^2 by marginal likelihood on the weighted group means
    (sampling variance s2 * sum w^2 / (sum w)^2); returns (shrunk effects, posterior vars, tau^2)."""
    W, S, W2 = (np.bincount(g, x, ng) for x in (w, w * r, w * w))
    ok = W > 1e-12
    eff, pv = np.zeros(ng), np.zeros(ng)
    if not ok.any():
        return eff, pv, 0.0
    mu = S[ok] / W[ok]
    v = s2 * W2[ok] / W[ok] ** 2
    tot = TAU2[:, None] + v[None]
    tau2 = TAU2[np.argmax(-0.5 * (np.log(tot) + mu ** 2 / tot).sum(1))]
    B = tau2 / (tau2 + v)
    eff[ok], pv[ok] = B * mu, B * v
    return eff, pv, float(tau2)


def loc_effects(z, codes, ngs, ws, on=None, iters=3):
    """Crossed random effects by backfitting: per-group effects, posterior vars, row fit, s2, tau2s."""
    k_ = len(codes)
    on = on or (True,) * k_
    row = [np.zeros(len(z)) for _ in range(k_)]
    eff, pv, tau2 = [np.zeros(n) for n in ngs], [np.zeros(n) for n in ngs], [0.0] * k_
    s2 = float(np.average(z ** 2, weights=ws[0])) if ws[0].sum() > 0 else 1.0
    for _ in range(iters):
        for k in range(k_):
            if on[k]:
                r = z - sum(row[j] for j in range(k_) if j != k)
                eff[k], pv[k], tau2[k] = eb_normal(codes[k], r, ws[k], ngs[k], s2)
                row[k] = eff[k][codes[k]]
        if ws[0].sum() > 0:
            s2 = float(np.average((z - sum(row)) ** 2, weights=ws[0]))
    return eff, pv, sum(row), s2, tau2


def trigamma_inv(x):
    """Inverse of the trigamma function (Newton, as in limma)."""
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = polygamma(1, y)
        d = tri * (1 - tri / x) / polygamma(2, y)
        y += d
        if -d / y < 1e-8:
            break
    return y


def squeeze_var(s2, df, d0_mult=1.0):
    """Smyth (2004) fitFDist + squeezeVar: posterior variances and the prior (d0, s0^2).
    d0_mult scales the fitted prior df (inf = all players at s0^2)."""
    ok = (df >= 1) & (s2 > 0)
    if ok.sum() < 5:
        s0 = float(np.average(s2[ok], weights=df[ok])) if ok.any() else 1.0
        return np.full_like(s2, s0), INF, s0
    e = np.log(s2[ok]) - digamma(df[ok] / 2) + np.log(df[ok] / 2)
    em = e.mean()
    ev = ((e - em) ** 2).sum() / (ok.sum() - 1) - polygamma(1, df[ok] / 2).mean()
    if ev > 0 and np.isfinite(d0_mult):
        d0 = 2 * trigamma_inv(ev)
        s0 = float(np.exp(em + digamma(d0 / 2) - np.log(d0 / 2)))
        d0 *= d0_mult
        return (d0 * s0 + df * s2) / (d0 + df), d0, s0
    s0 = float(np.exp(em))
    return np.full_like(s2, s0), INF, s0


def _node_ll(g, ng, y, lp0, w, sg):
    """(ng, nodes) weighted Bernoulli log-likelihood of each player's rows at b = sg * node."""
    LL = np.empty((ng, len(GH_X)))
    for k, x in enumerate(GH_X):
        eta = lp0 + sg * x
        LL[:, k] = np.bincount(g, w * (y * eta - np.logaddexp(0, eta)), ng)
    return LL


def logit_normal_sig(g, ng, y, lp0, w):
    """ML random-intercept sd for logit p = logit p0_base + b_player (Gauss-Hermite marginal)."""
    has = np.bincount(g, w, ng) > 1e-12
    best = (-INF, 0.0)
    for sg in SIG:
        LL = _node_ll(g, ng, y, lp0, w, sg)
        mx = LL.max(1, keepdims=True)
        ml = float((np.log(np.exp(LL - mx) @ GH_W) + mx[:, 0])[has].sum())
        if ml > best[0] + 1e-9:
            best = (ml, sg)
    return best[1]


def logit_normal_post(g, ng, y, lp0, w, sg):
    """Posterior node weights (ng, nodes) of each player's intercept under the discounted likelihood."""
    LL = _node_ll(g, ng, y, lp0, w, sg)
    post = GH_W * np.exp(LL - LL.max(1, keepdims=True))
    return post / post.sum(1, keepdims=True)


# ---------------------------------------------------------------- candidate
class HierBayes:
    def __init__(self, variant="eb", **kw):
        assert variant in VARIANTS, variant
        self.variant = variant
        cfg = dict(effects=(), select=False, h=EB_H, d0_mult=1.0, loc_on=None, anchor="pos")
        cfg.update(CFG[variant])
        cfg.update(kw)                         # dev ablations only
        self.cfg = cfg

    def fit(self, train):
        d = train[train["item"].to_numpy(bool)]
        pos, proj, y = d["pos"].to_numpy(str), d["proj"].to_numpy(float), d["actual"].to_numpy(float)
        self.base = HurdleBase().fit(pos, proj, y)
        if not self.cfg["effects"]:
            return self
        self.t, self.season, self.pos = d["t"].to_numpy(float), d["season"].to_numpy(int), pos
        self.zero = (y <= 0).astype(float)
        p0 = self.base.p0(pos, proj)
        self.lp0 = np.log(p0 / (1 - p0))
        self.z = self.base.z_pos(pos, proj, y)
        keys = self._keys(d)
        self.idx = {k: pd.Index(pd.unique(v)) for k, v in keys.items()}
        self.codes = {k: self.idx[k].get_indexer(v) for k, v in keys.items()}
        self.h, self.d0_mult = dict(self.cfg["h"]), self.cfg["d0_mult"]
        self._sig = {}
        if self.cfg["select"]:
            self._select()
        return self

    @staticmethod
    def _keys(d):
        s = d["season"].astype(str)
        return {"player": d["pid"].astype(str).to_numpy(),
                "team": (s + "|" + d["team"].astype(str)).to_numpy(),
                "opp": (s + "|" + d["opp"].astype(str) + "|" + d["pos"].astype(str)).to_numpy()}

    # ---- effects as of week T (season sT) from the training rows in mask `use`
    def _p_zero(self, T, sT, h, use, pid, lp0):
        """Posterior predictive P(y <= 0) for rows (player codes pid, base logits lp0)."""
        g, ng = self.codes["player"], len(self.idx["player"])
        if T not in self._sig:                 # hyperparameter on the undiscounted history before T
            self._sig[T] = logit_normal_sig(g, ng, self.zero, self.lp0, use.astype(float))
        sg = self._sig[T]
        w = decay(self.t, self.season, T, sT, h, GAP["zero"]) * use
        post = logit_normal_post(g, ng, self.zero, self.lp0, w, sg)
        P = np.where(pid[:, None] >= 0, post[np.maximum(pid, 0)], GH_W[None])
        p = (P / (1 + np.exp(-(lp0[:, None] + sg * GH_X[None])))).sum(1)
        return np.clip(p, 1e-5, 0.97)

    def _loc(self, T, sT, hp, ht, use, d0_mult=None):
        pz = use & np.isfinite(self.z)
        zz = np.where(pz, self.z, 0.0)
        wp = decay(self.t, self.season, T, sT, hp, GAP["player"]) * pz
        wt = decay(self.t, self.season, T, sT, ht, GAP["team"]) * pz
        names = LOC
        codes = [self.codes[k] for k in names]
        ngs = [len(self.idx[k]) for k in names]
        eff, pv, fitted, s2, tau2 = loc_effects(zz, codes, ngs, [wp] + [wt] * (len(names) - 1), self.cfg["loc_on"])
        r2 = (zz - fitted) ** 2
        L = {"eff": dict(zip(names, eff)), "pv": dict(zip(names, pv)), "s2": s2, "tau2": dict(zip(names, tau2)),
             "s2pos": {P: float(np.average(r2[k], weights=wp[k])) if wp[k].sum() > 0 else s2
                       for P in POS for k in [self.pos == P]}}
        if d0_mult is not None and np.isfinite(d0_mult):
            # residuals relative to the position level (s2pos already carries it), squeezed, then
            # normalized so the row-weighted mean of rel is 1 within each position (review fix: post / s0
            # double-counted the position level and averaged > 1, inflating every seen player's variance)
            g, ng = codes[0], ngs[0]
            lev = pd.Series(self.pos).map(L["s2pos"]).fillna(s2).to_numpy(float)
            W, W2 = np.bincount(g, wp, ng), np.bincount(g, wp * wp, ng)
            s2p = np.divide(np.bincount(g, wp * r2 / lev, ng), W, out=np.zeros(ng), where=W > 0)
            dfp = np.divide(W ** 2, W2, out=np.zeros(ng), where=W2 > 0)
            post, d0, s0 = squeeze_var(s2p, dfp, d0_mult)
            rel = np.where(W > 0, post, 1.0)
            for P in POS:
                k = (self.pos == P) & (wp > 0)
                if k.any():
                    hit = np.zeros(ng, bool)
                    hit[g[k]] = True
                    rel[hit] /= np.average(rel[g[k]], weights=wp[k])
            L["rel"] = np.where(W > 0, rel, 1.0)            # relative player scale (variance ratio)
            L["d0"] = d0
        return L

    def _zmoments(self, L, c, pos):
        """Predictive mean and variance of z for rows with group codes c (dict) and positions pos."""
        n = len(pos)
        anc = self.cfg["anchor"]
        var = (np.array([L["s2pos"].get(P, L["s2"]) for P in pos]) if anc == "pos"
               else np.full(n, L["s2"] if anc else 1.0))
        if "rel" in L:
            var = var * np.where(c["player"] >= 0, L["rel"][np.maximum(c["player"], 0)], 1.0)
        m = np.zeros(n)
        if "loc" in self.cfg["effects"]:
            for k in L["eff"]:
                hit = c[k] >= 0
                m[hit] += L["eff"][k][c[k][hit]]
                var[hit] += L["pv"][k][c[k][hit]]
        return m, var

    def _select(self):
        """Prequential choice over the last PREQ_WEEKS training weeks (each predicted from the rows
        before it): player/team half-lives by one-step-ahead z MSE, the variance-prior multiplier by
        Gaussian predictive log-likelihood of z, the zero half-life by Bernoulli log-likelihood."""
        ts = np.unique(self.t)
        evals = [T for i, T in enumerate(ts) if i >= PREQ_MIN_PRIOR][-PREQ_WEEKS:]
        if len(evals) < 2:
            return
        ev = []
        for T in evals:
            at = np.flatnonzero(self.t == T)
            k = at[np.isfinite(self.z[at])]
            ev.append((T, self.season[at][0], at, k, {n: self.codes[n][k] for n in self.codes}))
        best = (INF, None)
        for hp in H_PLAYER:
            for ht in H_TEAM:
                sse = 0.0
                for T, sT, at, k, c in ev:
                    L = self._loc(T, sT, hp, ht, self.t < T)
                    sse += float(((self.z[k] - self._zmoments(L, c, self.pos[k])[0]) ** 2).sum())
                if sse < best[0] - 1e-9:
                    best = (sse, (hp, ht))
        self.h["player"], self.h["team"] = best[1]
        if "scale" in self.cfg["effects"]:
            best = (-INF, None)
            for dm in D0_MULT:
                ll = 0.0
                for T, sT, at, k, c in ev:
                    L = self._loc(T, sT, self.h["player"], self.h["team"], self.t < T, dm)
                    m, var = self._zmoments(L, c, self.pos[k])
                    ll -= 0.5 * float((np.log(var) + (self.z[k] - m) ** 2 / var).sum())
                if ll > best[0] + 1e-9:
                    best = (ll, dm)
            self.d0_mult = best[1]
        best = (-INF, None)
        for hz in H_ZERO:
            ll = 0.0
            for T, sT, at, k, c in ev:
                p = self._p_zero(T, sT, hz, self.t < T, self.codes["player"][at], self.lp0[at])
                ll += float((self.zero[at] * np.log(p) + (1 - self.zero[at]) * np.log(1 - p)).sum())
            if ll > best[0] + 1e-9:
                best = (ll, hz)
        self.h["zero"] = best[1]

    def predict_quantiles(self, test, levels):
        pos, proj = test["pos"].to_numpy(str), test["proj"].to_numpy(float)
        n = len(test)
        p0 = self.base.p0(pos, proj)
        m, s = np.zeros(n), np.ones(n)
        eff = self.cfg["effects"]
        if eff:
            T, sT = float(test["t"].iloc[0]), int(test["season"].iloc[0])
            assert (test["t"].to_numpy() == T).all()
            use = np.ones(len(self.t), bool)
            c = {k: self.idx[k].get_indexer(v) for k, v in self._keys(test).items()}
            if "zero" in eff:
                p0 = self._p_zero(T, sT, self.h["zero"], use, c["player"], np.log(p0 / (1 - p0)))
            L = self._loc(T, sT, self.h["player"], self.h["team"], use,
                          self.d0_mult if "scale" in eff else None)
            m, var = self._zmoments(L, c, pos)
            s = np.sqrt(var)
            self.diag = {"h": dict(self.h), "d0_mult": self.d0_mult, "d0": L.get("d0"), "s2pos": L["s2pos"],
                         "tau2": L["tau2"], "zero_sig": self._sig.get(T)}
        return self.base.quantiles(pos, proj, p0, m, s, levels)
