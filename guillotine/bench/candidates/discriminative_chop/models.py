"""Chop-choice models on team features: ridge conditional logit / rank-ordered (Plackett-Luce) logit and a
LightGBM booster with a softmax-per-league objective. All hyperparameters are picked by week-grouped CV
on the training leagues only (score = held-out top-1 chop log-loss).

Choice sets: league l's stage j (j < K) is the bottom-up Plackett-Luce event "the j-th lowest realized
team is the minimum of the teams not yet eliminated", weight 0.5^j (deeper ranks are noisier: Hausman &
Ruud 1987); K = 1 is the plain conditional logit on who got chopped. Offsets are the sim's
log P(team = min of that stage's set), recomputed per stage from the draws (no Harville shortcut).
"""
from bench import harness  # noqa: F401

import numpy as np
from bench.harness import task_seed, task_threads

from .features import FI

LAMS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, np.inf)   # inf = the prior exactly
KS = (1, 3)
STAGE_W = 0.5
NFOLD = 5
EPS = 1e-6
STRUCT = ("td_share", "rec_share", "n_low", "stack", "opp", "env", "h_bias", "h_vol")
SETS = {"clogit": ("gap", "gap2", "lsd", "gq10") + STRUCT,
        "gbm": ("lp", "gap", "lsd", "gq10") + STRUCT + ("n", "rank_mu")}


def derived(X, lg):
    """Team-level model features, relative within each league (rows of a league are contiguous)."""
    lg = np.asarray(lg)
    nl = int(lg.max()) + 1
    cnt = np.bincount(lg, minlength=nl).astype(float)

    def lmin(v):
        m = np.full(nl, np.inf)
        np.minimum.at(m, lg, v)
        return m[lg]

    def lmean(v):
        return (np.bincount(lg, v, minlength=nl) / cnt)[lg]
    mu, sd, q10 = X[:, FI["mu"]], np.maximum(X[:, FI["sd"]], 1e-6), X[:, FI["q10"]]
    sdb = lmean(sd)
    gap = (mu - lmin(mu)) / sdb
    out = {"lp": X[:, FI["lp"]], "gap": gap, "gap2": gap ** 2, "lsd": np.log(sd / sdb),
           "gq10": (q10 - lmin(q10)) / sdb, "n": cnt[lg]}
    for k in STRUCT:
        v = X[:, FI[k]] / (sdb if k == "h_bias" else 1.0)
        out[k] = v - lmean(v)
    rk = np.zeros(len(mu))                                   # rank of mu in league / (N - 1)
    for l in np.unique(lg):
        ix = np.flatnonzero(lg == l)
        rk[ix[np.argsort(mu[ix], kind="stable")]] = np.arange(len(ix)) / max(len(ix) - 1, 1)
    out["rank_mu"] = rk
    return out


def _groups(grp):
    ptr = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    return ptr, np.diff(np.r_[ptr, len(grp)])


def _softmax(u, ptr, sizes):
    mx = np.maximum.reduceat(u, ptr)
    e = np.exp(u - np.repeat(mx, sizes))
    se = np.add.reduceat(e, ptr)
    return e / np.repeat(se, sizes), mx + np.log(se)


def heldout_ll(u, ptr, sizes):
    """Sum of top-1 log-loss (chosen = first row of each group), p clipped at EPS and renormalized."""
    p, _ = _softmax(u, ptr, sizes)
    p = np.maximum(p, EPS)
    p /= np.repeat(np.add.reduceat(p, ptr), sizes)
    return float(-np.log(p[ptr]).sum())


class Stages:
    """Stage rows of a training set (or a subset of its groups) in contiguous group order."""

    def __init__(self, tr, Zt, gmask):
        rm = gmask[tr["st_grp"]]
        self.Z = Zt[tr["st_team"][rm]]
        self.off = tr["st_off"][rm]
        self.ptr, self.sizes = _groups(tr["st_grp"][rm])
        self.w = STAGE_W ** tr["g_stage"][gmask]
        self.y = np.zeros(len(self.off))
        self.y[self.ptr] = 1.0


# ---------------------------------------------------------------- ridge conditional logit (Newton)
def _obj(th, Z, S, lam_s, th0, hess=True):
    u = Z @ th
    p, lse = _softmax(u, S.ptr, S.sizes)
    f = -(S.w * (u[S.ptr] - lse)).sum() + 0.5 * (lam_s * (th - th0) ** 2).sum()
    if not hess:
        return f
    m = np.add.reduceat(Z * p[:, None], S.ptr)
    gr = -(S.w[:, None] * (Z[S.ptr] - m)).sum(0) + lam_s * (th - th0)
    wr = np.repeat(S.w, S.sizes) * p
    H = (Z * wr[:, None]).T @ Z - (m * S.w[:, None]).T @ m + np.diag(lam_s)
    return f, gr, H


def newton(Z, S, lam_s, th0, th=None, iters=50):
    th = th0.copy() if th is None else th.copy()
    f, gr, H = _obj(th, Z, S, lam_s, th0)
    for _ in range(iters):
        step = np.linalg.solve(H, gr)
        dec = gr @ step
        if dec < 1e-10:
            break
        a = 1.0
        while True:
            th1 = th - a * step
            f1 = _obj(th1, Z, S, lam_s, th0, hess=False)
            if f1 <= f - 1e-4 * a * dec or a < 1e-8:
                break
            a *= 0.5
        th = th1
        f, gr, H = _obj(th, Z, S, lam_s, th0)
    return th


def folds(tr):
    """Week -> fold (round-robin over the training weeks)."""
    wk = np.unique(tr["g_t"])
    nf = min(NFOLD, len(wk))
    return dict(zip(wk.tolist(), (np.arange(len(wk)) % nf).tolist())), nf


def ridge_cv(tr, Zt, th0, lam_s, ks, use_off):
    """Week-grouped CV of a ridge conditional logit over (K, lambda); the chosen pair refit on all leagues.

    Zt: (R, F) team-level design; use_off puts the stage offset (log P_sim) first. Returns (K, lambda,
    cv {K/lambda: held-out top-1 log-loss per league}, theta, oof) where oof holds each stage-0 row's
    out-of-fold utility at the chosen (K, lambda) (index = position in the st_* arrays)."""
    fo, nf = folds(tr)
    gf = np.array([fo[t] for t in tr["g_t"]])
    full = (lambda S: np.column_stack([S.off, S.Z])) if use_off else (lambda S: S.Z)
    score, oof = {}, {}
    for K in ks:
        for f in range(nf):
            S = Stages(tr, Zt, (gf != f) & (tr["g_stage"] < K))
            vm = (gf == f) & (tr["g_stage"] == 0)
            V = Stages(tr, Zt, vm)
            vpos = np.flatnonzero(vm[tr["st_grp"]])
            Z, Zv = full(S), full(V)
            th = None
            for lam in sorted(LAMS, reverse=True):              # warm start along the path
                th = th0.copy() if np.isinf(lam) else newton(Z, S, lam * lam_s, th0, th)
                uv = Zv @ th
                score[K, lam] = score.get((K, lam), 0.0) + heldout_ll(uv, V.ptr, V.sizes)
                oof.setdefault((K, lam), np.zeros(len(tr["st_grp"])))[vpos] = uv
    nv = (tr["g_stage"] == 0).sum()
    K, lam = min(score, key=lambda k: (round(score[k] / nv, 6), -k[1], k[0]))   # ties: more shrinkage, smaller K
    cv = {f"{k[0]}/{k[1]:g}": round(float(v / nv), 5) for k, v in score.items()}
    S = Stages(tr, Zt, tr["g_stage"] < K)
    th = th0.copy() if np.isinf(lam) else newton(full(S), S, lam * lam_s, th0)
    return K, lam, cv, th, oof[K, lam]


class CLogit:
    """u = [alpha * offset] + beta . z (z standardized on the training teams). Ridge toward the prior
    (alpha = 1, beta = 0; penalty on alpha scaled by var(offset)): with no evidence an offset model is the
    baseline sim, a pure one is uniform. (lambda, K) by week-grouped CV."""

    def __init__(self, cols, offset):
        self.cols, self.offset = cols, offset

    def _design(self, d):
        return np.column_stack([(d[c] - self.mu[i]) / self.sd[i] for i, c in enumerate(self.cols)])

    def fit(self, tr):
        d = derived(tr["X"], tr["lg"])
        self.mu = np.array([d[c].mean() for c in self.cols])
        self.sd = np.array([max(d[c].std(), 1e-9) for c in self.cols])
        F = len(self.cols) + self.offset
        th0, lam_s = np.zeros(F), np.ones(F)
        if self.offset:
            th0[0] = 1.0
            lam_s[0] = tr["st_off"][tr["st_stage"] == 0].var()
        self.K, self.lam, self.cv, self.th, self.oof = ridge_cv(tr, self._design(d), th0, lam_s, KS, self.offset)
        return self

    def info(self):
        return {"K": self.K, "lam": self.lam, "cv_best": self.cv[f"{self.K}/{self.lam:g}"],
                "cv_prior": self.cv.get(f"1/{np.inf:g}"), "theta": np.round(self.th, 4).tolist(),
                "cols": (["offset"] if self.offset else []) + list(self.cols)}

    def utility(self, X):
        Z = self._design(derived(X, np.zeros(len(X), np.int64)))
        return (np.column_stack([X[:, FI["lp"]], Z]) if self.offset else Z) @ self.th

    def predict(self, X):
        u = self.utility(X)
        p = np.exp(u - u.max())
        return p / p.sum()


class Benter:
    """Benter (1994) second stage: u = a * log p_sim + b * u_clogit. Stage 1 is the pure CLogit; its
    out-of-fold utilities on the training leagues feed a 2-coefficient ridge conditional logit toward
    (a, b) = (1, 0), lambda by the same week-grouped CV (inf = the sim alone)."""

    def fit(self, tr):
        self.c1 = CLogit(SETS["clogit"], offset=False).fit(tr)
        rm0 = (tr["g_stage"] == 0)[tr["st_grp"]]
        Zt = np.zeros((len(tr["X"]), 2))                    # stage 0 holds every team exactly once
        Zt[tr["st_team"][rm0]] = np.column_stack([tr["st_off"][rm0], self.c1.oof[rm0]])
        self.K, self.lam, self.cv, self.th, _ = ridge_cv(tr, Zt, np.array([1.0, 0.0]), np.maximum(Zt.var(0), 1e-6), (1,), False)
        return self

    def info(self):
        return {"stage1": {k: v for k, v in self.c1.info().items() if k not in ("theta", "cols")},
                "lam": self.lam, "cv_best": self.cv[f"1/{self.lam:g}"], "cv_sim": self.cv.get(f"1/{np.inf:g}"),
                "a_b": np.round(self.th, 4).tolist()}

    def predict(self, X):
        u = self.th[0] * X[:, FI["lp"]] + self.th[1] * self.c1.utility(X)
        p = np.exp(u - u.max())
        return p / p.sum()


# ---------------------------------------------------------------- LightGBM softmax-per-league booster
GBM = {"learning_rate": 0.05, "num_leaves": 4, "max_depth": 2, "min_data_in_leaf": 100,
       "min_sum_hessian_in_leaf": 1.0, "lambda_l2": 10.0, "max_bin": 63, "feature_fraction": 1.0,
       "verbose": -1, "deterministic": True, "force_row_wise": True}
ROUNDS, STEP = 100, 5


def _fobj(S):
    wr = np.repeat(S.w, S.sizes)

    def f(preds, ds):
        p, _ = _softmax(preds, S.ptr, S.sizes)
        return wr * (p - S.y), np.maximum(wr * p * (1 - p), 1e-9)
    return f


class Boost:
    """Boosting from the sim: init score = stage offset log P_sim, trees on relative team features;
    (K, rounds) by week-grouped CV, rounds = 0 (the baseline) allowed."""

    def __init__(self, cols):
        self.cols = cols

    def _rows(self, d, S_team, off):
        return np.column_stack([off if c == "lp" else d[c][S_team] for c in self.cols])

    def _train(self, tr, d, gmask, rounds):
        import lightgbm as lgb
        S = Stages(tr, np.arange(len(tr["X"]))[:, None], gmask)
        Z = self._rows(d, S.Z[:, 0], S.off)
        ds = lgb.Dataset(Z, label=S.y, init_score=S.off, free_raw_data=False)
        prm = dict(GBM, objective=_fobj(S), num_threads=task_threads(), seed=task_seed())
        return lgb.train(prm, ds, num_boost_round=rounds)

    def fit(self, tr):
        d = derived(tr["X"], tr["lg"])
        fo, nf = folds(tr)
        gf = np.array([fo[t] for t in tr["g_t"]])
        rs = np.arange(0, ROUNDS + 1, STEP)
        score = {}
        for K in KS:
            for f in range(nf):
                b = self._train(tr, d, (gf != f) & (tr["g_stage"] < K), ROUNDS)
                V = Stages(tr, np.arange(len(tr["X"]))[:, None], (gf == f) & (tr["g_stage"] == 0))
                Zv = self._rows(d, V.Z[:, 0], V.off)
                for r in rs:
                    u = V.off + (b.predict(Zv, num_iteration=int(r), raw_score=True) if r > 0 else 0.0)
                    score[K, int(r)] = score.get((K, int(r)), 0.0) + heldout_ll(u, V.ptr, V.sizes)
        nv = (tr["g_stage"] == 0).sum()
        self.K, self.rounds = min(score, key=lambda k: (round(score[k] / nv, 6), k[1], k[0]))
        self.cv0 = round(float(score[1, 0] / nv), 5)
        self.cvbest = round(float(score[self.K, self.rounds] / nv), 5)
        self.b = self._train(tr, d, tr["g_stage"] < self.K, self.rounds) if self.rounds > 0 else None
        return self

    def info(self):
        return {"K": self.K, "rounds": self.rounds, "cv_base": self.cv0, "cv_best": self.cvbest}

    def predict(self, X):
        d = derived(X, np.zeros(len(X), np.int64))
        u = X[:, FI["lp"]].copy()
        if self.b is not None:
            u += self.b.predict(self._rows(d, np.arange(len(X)), u), raw_score=True)
        p = np.exp(u - u.max())
        return p / p.sum()
