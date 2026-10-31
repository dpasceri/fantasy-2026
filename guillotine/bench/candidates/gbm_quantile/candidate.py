"""Tree-ensemble distributional regression for player-week scores, on the actual/projected ratio
r = y / proj like the baseline: quantile regression forest and LightGBM.

Variants:
  qrf         quantile regression forest (Meinshausen 2006). CART trees, each grown on a random half
              of the training weeks (subagging by whole weeks: rows of a week share games), split on
              [y / 8, 1{y <= 0}] (mean and zero mass), max_features 1/3, min leaf n/40. A test row's
              Meinshausen weights over all training rows give a weighted ECDF of r, and
              q_y(tau) = proj x Q_r(tau). Zeros among the neighbours make an exact atom at 0, so the
              zero mass is estimated directly instead of interpolated between projection buckets.
  qrf_hurdle  the same forest as a hurdle mixture: P(y <= 0) = mean of the forest's neighbour share
              and a LightGBM binary classifier (rounds by early stopping on the last training weeks);
              y <= 0 part and y > 0 part from the forest weights restricted to each side.
  lgbq        LightGBM quantile regression of r at 15 levels, started from the training r quantile
              (init_score), one shared round count (median of per-level early-stopping rounds on the
              last training weeks), monotone rearrangement (Chernozhukov et al. 2010), linear
              interpolation to the 99 levels.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)

import numpy as np
import scipy.sparse as sp

from bench.harness import task_seed, task_threads

from . import features

VARIANTS = ("qrf", "qrf_hurdle", "lgbq")
TAUS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
N_TREES = 500
LEAF_DIV = 40        # min leaf = n_train / 40 (each tree sees about half of them)
SNAP = 0.05          # |q| below this is the zero atom (ties must be exact for the PIT)


def wquant(vals, W, levels):
    """(m, L) quantiles of the weighted ECDF of vals (n,) under row weights W (m, n).
    levels: (L,) shared, or (m, L) per row."""
    o = np.argsort(vals, kind="stable")
    v = vals[o]
    C = np.cumsum(W[:, o], 1)
    C /= np.maximum(C[:, -1:], 1e-300)
    lv = np.broadcast_to(np.asarray(levels, float), (len(W), np.shape(levels)[-1]))
    out = np.empty(lv.shape)
    top = len(v) - 1
    for i in range(len(W)):
        out[i] = v[np.minimum(np.searchsorted(C[i], lv[i] - 1e-12), top)]
    return out


def inner_split(week):
    """Validation mask = the last 20% of training weeks (1 to 4 weeks), for early stopping."""
    ws = np.unique(week)
    nv = int(np.clip(round(0.2 * len(ws)), 1, 4))
    return week >= ws[-nv] if len(ws) > 1 else np.zeros(len(week), bool)


# ---------------------------------------------------------------- quantile regression forest
class WeekForest:
    """CART trees, each grown on a random half of the training weeks; Meinshausen weights over all
    training rows (leaf co-membership / leaf size, averaged over trees)."""

    def __init__(self, n_trees=N_TREES, leaf=50, max_features=1 / 3, frac=0.5, seed=0):
        self.n_trees, self.leaf, self.max_features, self.frac, self.seed = n_trees, leaf, max_features, frac, seed

    def fit(self, X, Y, week):
        from sklearn.tree import DecisionTreeRegressor
        rng = np.random.default_rng(self.seed)
        ws = np.unique(week)
        wi = np.searchsorted(ws, week)
        k = max(1, int(round(self.frac * len(ws))))
        self.trees = []
        for _ in range(self.n_trees):
            m = np.isin(wi, rng.choice(len(ws), k, replace=False))
            tr = DecisionTreeRegressor(min_samples_leaf=self.leaf, max_features=self.max_features,
                                       random_state=int(rng.integers(2 ** 31 - 1)))
            self.trees.append(tr.fit(X[m], Y[m]))
        L = self.leaves(X)
        nc = np.array([t.tree_.node_count for t in self.trees])
        self.offs = np.r_[0, np.cumsum(nc)[:-1]]
        cols = (L + self.offs[None, :]).ravel()
        size = np.bincount(cols, minlength=int(nc.sum())).astype(float)
        n, T = L.shape
        self.B = sp.csr_matrix((1.0 / size[cols], (np.repeat(np.arange(n), T), cols)), shape=(n, len(size)))
        return self

    def leaves(self, X):
        return np.stack([t.apply(X) for t in self.trees], 1)

    def weights(self, X):
        """(m, n_train) weights: mean over trees of 1{same leaf} / leaf size (rows sum to 1)."""
        L = self.leaves(X)
        m, T = L.shape
        A = sp.csr_matrix((np.full(m * T, 1.0 / T), (np.repeat(np.arange(m), T), (L + self.offs[None, :]).ravel())),
                          shape=(m, self.B.shape[1]))
        return (A @ self.B.T).toarray()


def forest_weights(X, y, week, Xq, seed):
    Y = np.stack([y / 8.0, (y <= 0).astype(float)], 1)
    f = WeekForest(N_TREES, max(15, int(len(y) / LEAF_DIV)), seed=seed).fit(X, Y, week)
    return f.weights(Xq)


def qrf_quantiles(X, y, proj, week, Xq, projq, levels, seed, **_):
    """Ratio-space QRF: (m, L) score quantiles for query rows."""
    return projq[:, None] * wquant(y / proj, forest_weights(X, y, week, Xq, seed), levels)


def p0_classifier(X, z, week, Xq, seed, threads=1, info=None):
    """LightGBM P(y <= 0): rounds by early stopping on the last training weeks, then refit on all."""
    import lightgbm as lgb
    P = dict(objective="binary", learning_rate=0.03, num_leaves=7, max_depth=3, min_data_in_leaf=40,
             feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=seed,
             deterministic=True, force_col_wise=True, num_threads=threads)
    va = inner_split(week)
    rounds = 0
    if va.any() and (~va).sum() > 200 and 0 < z[~va].mean() < 1:
        b = lgb.train(P, lgb.Dataset(X[~va], z[~va]), 1000, valid_sets=[lgb.Dataset(X[va], z[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        rounds = int(b.best_iteration or 0)
    if info is not None:
        info["p0_rounds"] = rounds
    if rounds < 1:
        return np.full(len(Xq), z.mean())
    return lgb.train(P, lgb.Dataset(X, z), rounds).predict(Xq)


def qrf_hurdle_quantiles(X, y, proj, week, Xq, projq, levels, seed, threads=1, info=None):
    """Hurdle mixture on the forest: P0 = mean(forest share, LightGBM); each side from the weights."""
    W = forest_weights(X, y, week, Xq, seed)
    z = (y <= 0).astype(float)
    p0 = np.clip(0.5 * (W @ z + p0_classifier(X, z, week, Xq, seed, threads, info)), 1e-3, 0.98)
    r = y / proj
    neg = z > 0
    Wn, Wp = W * neg, W * ~neg
    has_n, has_p = Wn.sum(1) > 0, Wp.sum(1) > 0
    Wn[~has_n] = neg / max(neg.sum(), 1)                         # no zero neighbour: pooled y <= 0 rows
    Wp[~has_p] = ~neg / max((~neg).sum(), 1)
    lo = levels[None, :] < p0[:, None]
    qn = wquant(r, Wn, np.clip(levels[None, :] / p0[:, None], 0, 1))
    qp = wquant(r, Wp, np.clip((levels[None, :] - p0[:, None]) / (1 - p0[:, None]), 0, 1))
    return projq[:, None] * np.where(lo, qn, qp)


# ---------------------------------------------------------------- LightGBM quantiles
def _levels_from_taus(out, levels):
    """Rearranged (m, len(TAUS)) quantiles -> (m, L): linear in tau, linear extrapolation past the ends."""
    out = np.sort(out, axis=1)
    xs = np.r_[0.0, TAUS, 1.0]
    lo = out[:, :1] - (out[:, 1:2] - out[:, :1]) * TAUS[0] / (TAUS[1] - TAUS[0])
    hi = out[:, -1:] + (out[:, -1:] - out[:, -2:-1]) * (1 - TAUS[-1]) / (TAUS[-1] - TAUS[-2])
    ys = np.concatenate([lo, out, hi], 1)
    k = np.clip(np.searchsorted(xs, levels, side="right") - 1, 0, len(xs) - 2)
    w = (levels - xs[k]) / (xs[k + 1] - xs[k])
    return ys[:, k] * (1 - w) + ys[:, k + 1] * w


def lgb_quantiles(X, y, proj, week, Xq, projq, levels, seed, threads=1, info=None):
    """Ratio-space LightGBM quantile regression: (m, L) score quantiles for query rows."""
    import lightgbm as lgb
    r = y / proj
    va = inner_split(week)
    tr = ~va
    P = dict(objective="quantile", learning_rate=0.03, num_leaves=7, max_depth=3,
             min_data_in_leaf=max(40, len(y) // 80), feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
             verbose=-1, seed=seed, deterministic=True, force_col_wise=True, num_threads=threads)
    best = []
    if va.any() and tr.sum() > 200:
        for tau in TAUS:
            init = float(np.quantile(r[tr], tau))
            dt = lgb.Dataset(X[tr], r[tr], init_score=np.full(tr.sum(), init), free_raw_data=False)
            dv = lgb.Dataset(X[va], r[va], init_score=np.full(va.sum(), init), reference=dt)
            b = lgb.train(dict(P, alpha=float(tau)), dt, num_boost_round=400, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
            best.append(int(b.best_iteration or 0))
    rounds = int(np.median(best)) if best else 0          # one round count for every level
    out = np.zeros((len(Xq), len(TAUS)))
    for j, tau in enumerate(TAUS):
        init = float(np.quantile(r, tau))
        out[:, j] = init
        if rounds > 0:
            b = lgb.train(dict(P, alpha=float(tau)), lgb.Dataset(X, r, init_score=np.full(len(r), init)),
                          num_boost_round=rounds)
            out[:, j] += b.predict(Xq)
    if info is not None:
        info.update(best=best, rounds=rounds)
    qr = np.clip(_levels_from_taus(out, levels), r.min(), r.max())
    return projq[:, None] * qr


METHODS = {"qrf": qrf_quantiles, "qrf_hurdle": qrf_hurdle_quantiles, "lgbq": lgb_quantiles}


# ---------------------------------------------------------------- candidate
class GBMQuantile:
    def __init__(self, variant="qrf"):
        assert variant in VARIANTS, variant
        self.variant = variant
        self.info = {}

    def fit(self, train):
        self.train = train          # features need the test week's frame too, so the work is in predict
        return self

    def predict_quantiles(self, test, levels):
        levels = np.asarray(levels, float)
        X, Xq, m, _ = features.build(self.train, test)
        X, Xq = np.nan_to_num(X, nan=-1.0), np.nan_to_num(Xq, nan=-1.0)
        d = self.train[m]
        y, proj, week = d["actual"].to_numpy(float), d["proj"].to_numpy(float), d["t"].to_numpy(np.int64)
        q = METHODS[self.variant](X, y, proj, week, Xq, test["proj"].to_numpy(float), levels, task_seed(),
                                  threads=task_threads(), info=self.info)
        q = np.sort(q, axis=1)
        q[np.abs(q) < SNAP] = 0.0
        return q
