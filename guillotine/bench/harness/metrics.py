"""Scoring rules: quantile forecasts, probabilities, and the block bootstrap.

Quantile forecasts live on LEVELS = 0.01..0.99 (99 levels).
- CRPS is approximated as 2 x mean pinball over LEVELS. CRPS(F, y) = 2 * int_0^1 pinball_tau(F^-1(tau), y) dtau;
  the mean over 99 levels weights each by 1/99 instead of 1/100 (+1%) and leaves out the outer 1% tails
  (-), so it reads about 1% high on a normal (checked: 1.0099 x the closed form) and a little less on heavy
  tails. Same items + same levels for everyone, so paired differences are fair (a near-constant scale).
- PIT is randomized: the outcome's position among the 99 quantiles picks one of 100 equal-mass bands, a tie
  with several quantiles (a point mass, e.g. the zero for inactive players) spreads it uniformly over every
  band it touches, and the uniform draw is a fixed hash of the item id (identical for every candidate, so it
  cancels in paired comparisons). A calibrated forecast then hits the nominal rates (checked: atom of 17.7%
  -> coverage 0.099/0.500/0.900/0.802). If the atom's mass sits exactly on a level (e.g. 20%) the tie also
  covers that level's band, which spreads the atom over one band too many (cov10 0.096 there).
- cov10/cov50/cov90 = share with PIT <= 0.1/0.5/0.9 (= y <= q_level when there is no tie);
  cov80 = share inside the central 80% interval (0.1 < PIT <= 0.9); width80 = q90 - q10 (sharpness).
- pit_maxdev = max |share in bin - 0.1| over a 10-bin PIT histogram.

Every leaderboard metric is a Metric: per-item sufficient statistics (n, k) that sum over items, and a
final() map from the sums to the value. Sums per block make the paired block bootstrap a matrix product.
"""
import hashlib

import numpy as np

LEVELS = np.linspace(0.01, 0.99, 99)
EPS = 1e-6
POS = ("QB", "RB", "WR", "TE")


def level_index(p, levels=LEVELS):
    i = int(np.argmin(np.abs(np.asarray(levels) - p)))
    assert abs(levels[i] - p) < 1e-9, f"level {p} not on the grid"
    return i


# ---------------------------------------------------------------- quantile forecasts
def pinball(q, y, levels=LEVELS):
    """(n,) mean pinball loss over levels; q (n, L) sorted, y (n,)."""
    d = np.asarray(y, float)[:, None] - q
    lv = np.asarray(levels, float)
    return np.mean(np.maximum(lv * d, (lv - 1.0) * d), axis=1)


def crps(q, y, levels=LEVELS):
    """(n,) CRPS ~= 2 x mean pinball over levels (see module docstring)."""
    return 2.0 * pinball(q, y, levels)


def item_uniform(item_ids):
    """Deterministic U(0,1) per item id (blake2b hash), shared by every candidate."""
    return np.array([int.from_bytes(hashlib.blake2b(str(i).encode(), digest_size=8).digest(), "little")
                     for i in item_ids], float) / 2.0 ** 64


def pit(q, y, u):
    """Randomized PIT for quantiles on equispaced levels: (n,) in [0, 1]."""
    L = q.shape[1]
    yy = np.asarray(y, float)[:, None]
    lo = (q < yy).sum(1)
    hi = (q <= yy).sum(1)
    return (lo + u * (hi - lo + 1)) / (L + 1)


def pit_hist(p, bins=10):
    p = np.asarray(p, float)
    return np.bincount(np.minimum((p * bins).astype(int), bins - 1), minlength=bins) / max(len(p), 1)


def quantile_items(q, y, item_ids, levels=LEVELS):
    """Per-item arrays for a quantile forecast (q is sorted row-wise here)."""
    q = np.sort(np.asarray(q, float), axis=1)
    y = np.asarray(y, float)
    i10, i90 = level_index(0.1, levels), level_index(0.9, levels)
    p = pit(q, y, item_uniform(item_ids))
    return {"crps": crps(q, y, levels), "pinball": pinball(q, y, levels), "pit": p,
            "cov10": (p <= 0.1).astype(float), "cov50": (p <= 0.5).astype(float),
            "cov90": (p <= 0.9).astype(float), "cov80": ((p > 0.1) & (p <= 0.9)).astype(float),
            "width80": q[:, i90] - q[:, i10]}


# ---------------------------------------------------------------- probability forecasts
def logloss(p, y, eps=EPS):
    """(n,) log-loss. Binary: p (n,) = P(y=1), y in {0,1}. Multiclass: p (n, K), y (n,) class index.
    Probabilities are clipped to [eps, 1-eps] (multiclass rows renormalized after clipping)."""
    p = np.asarray(p, float)
    y = np.asarray(y)
    if p.ndim == 1:
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(p)), y.astype(int)])


def brier(p, y):
    """(n,) Brier score; multiclass = sum over classes of squared error vs the one-hot outcome."""
    p = np.asarray(p, float)
    y = np.asarray(y)
    if p.ndim == 1:
        return (p - y) ** 2
    oh = np.zeros_like(p)
    oh[np.arange(len(p)), y.astype(int)] = 1.0
    return ((p - oh) ** 2).sum(1)


def calibration_bins(p, y, bins=10):
    """Reliability table for binary forecasts: per bin n, mean p, observed rate; plus ECE."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for k in range(bins):
        m = b == k
        if m.any():
            out.append({"bin": [k / bins, (k + 1) / bins], "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "rate": float(y[m].mean())})
    ece = sum(r["n"] * abs(r["mean_p"] - r["rate"]) for r in out) / max(len(p), 1)
    return {"bins": out, "ece": float(ece)}


def prob_items(p, y):
    """Per-item arrays for probability forecasts (binary or multiclass)."""
    p, y = np.asarray(p, float), np.asarray(y)
    a = {"logloss": logloss(p, y), "brier": brier(p, y)}
    if p.ndim == 1:
        a["p"], a["y"] = p, y.astype(float)
    return a


# ---------------------------------------------------------------- metric specs
class Metric:
    """name; stat(arrays) -> (n, k) per-item sums; final(S (..., k)) -> value; lower: True/False/None."""

    def __init__(self, name, stat, final, lower=True, needs=()):
        self.name, self.stat, self.final, self.lower, self.needs = name, stat, final, lower, tuple(needs)

    def value(self, a):
        return float(self.final(self.stat(a).sum(0)))

    def ok(self, a):
        return all(k in a for k in self.needs)


def mean_of(name, arr=None, lower=True):
    arr = arr or name
    return Metric(name, lambda a: np.stack([np.asarray(a[arr], float), np.ones(len(a[arr]))], 1),
                  lambda S: S[..., 0] / S[..., 1], lower, (arr,))


def group_mean(name, arr, gcol, g, lower=True):
    def stat(a):
        m = (np.asarray(a[gcol]) == g).astype(float)
        return np.stack([np.asarray(a[arr], float) * m, m], 1)
    return Metric(name, stat, lambda S: S[..., 0] / np.maximum(S[..., 1], 1e-12), lower, (arr, gcol))


def mean_abs_err(name, targets):
    """Mean over {arr: nominal} of |share - nominal| (coverage error)."""
    keys = list(targets)

    def stat(a):
        n = len(a[keys[0]])
        return np.stack([np.asarray(a[k], float) for k in keys] + [np.ones(n)], 1)

    def final(S):
        return np.mean([np.abs(S[..., i] / S[..., -1] - targets[k]) for i, k in enumerate(keys)], axis=0)
    return Metric(name, stat, final, True, keys)


def pit_dev(name, arr="pit", bins=10, gcol=None, g=None):
    def stat(a):
        p = np.asarray(a[arr], float)
        oh = np.zeros((len(p), bins))
        oh[np.arange(len(p)), np.minimum((p * bins).astype(int), bins - 1)] = 1.0
        if gcol is not None:
            oh *= (np.asarray(a[gcol]) == g)[:, None]
        return oh

    def final(S):
        tot = np.maximum(S.sum(-1, keepdims=True), 1e-12)
        return np.abs(S / tot - 1.0 / bins).max(-1)
    return Metric(name, stat, final, True, (arr,) + ((gcol,) if gcol else ()))


def ece_metric(name="ece", bins=10):
    def stat(a):
        p, y = np.asarray(a["p"], float), np.asarray(a["y"], float)
        b = np.minimum((p * bins).astype(int), bins - 1)
        S = np.zeros((len(p), 3 * bins))
        r = np.arange(len(p))
        S[r, b], S[r, bins + b], S[r, 2 * bins + b] = p, y, 1.0
        return S

    def final(S):
        sp, sy, n = S[..., :bins], S[..., bins:2 * bins], S[..., 2 * bins:]
        return np.abs(sp - sy).sum(-1) / np.maximum(n.sum(-1), 1e-12)
    return Metric(name, stat, final, True, ("p", "y"))


def quantile_metrics(groups=POS, gcol="pos"):
    ms = [mean_of("crps"), mean_of("pinball"), pit_dev("pit_maxdev"),
          mean_abs_err("cov_err", {"cov10": 0.1, "cov50": 0.5, "cov90": 0.9, "cov80": 0.8}),
          mean_of("cov10", lower=None), mean_of("cov50", lower=None), mean_of("cov90", lower=None),
          mean_of("cov80", lower=None), mean_of("width80", lower=None)]
    for g in groups:
        ms += [group_mean(f"crps_{g}", "crps", gcol, g), pit_dev(f"pit_maxdev_{g}", gcol=gcol, g=g)]
    return ms


def prob_metrics(binary=True):
    return [mean_of("logloss"), mean_of("brier")] + ([ece_metric()] if binary else [])


def summarize(metric_list, a):
    """{metric: value} over all items, plus n."""
    out = {m.name: m.value(a) for m in metric_list if m.ok(a)}
    out["n"] = int(len(a["item_id"])) if "item_id" in a else int(len(next(iter(a.values()))))
    return out


# ---------------------------------------------------------------- block bootstrap
def block_index(blocks):
    """(unique blocks, inverse index)."""
    return np.unique(np.asarray(blocks), return_inverse=True)


def block_sums(S, inv, nb):
    out = np.zeros((nb, S.shape[1]))
    np.add.at(out, inv, S)
    return out


def bootstrap_counts(nb, B=2000, seed=0):
    """(B, nb) multinomial resample counts over blocks."""
    return np.random.default_rng(seed).multinomial(nb, np.full(nb, 1.0 / nb), size=B).astype(float)


def block_bootstrap(metric, a, blocks, B=2000, seed=0, ci=0.95):
    """Point value and percentile CI of one metric, resampling whole blocks."""
    ub, inv = block_index(blocks)
    bs = block_sums(metric.stat(a), inv, len(ub))
    C = bootstrap_counts(len(ub), B, seed)
    v = metric.final(C @ bs)
    lo, hi = np.nanpercentile(v, [50 * (1 - ci), 50 * (1 + ci)])
    return {"value": float(metric.final(bs.sum(0))), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap(metric, a_cand, a_base, blocks, C=None, B=2000, seed=0, ci=0.95):
    """Candidate minus baseline on the same items, CI from resampling blocks (same resample for both)."""
    ub, inv = block_index(blocks)
    if C is None:
        C = bootstrap_counts(len(ub), B, seed)
    bc, bb = block_sums(metric.stat(a_cand), inv, len(ub)), block_sums(metric.stat(a_base), inv, len(ub))
    vc, vb = float(metric.final(bc.sum(0))), float(metric.final(bb.sum(0)))
    d = metric.final(C @ bc) - metric.final(C @ bb)
    lo, hi = np.nanpercentile(d, [50 * (1 - ci), 50 * (1 + ci)])
    worse = None if metric.lower is None else float(np.mean(d > 0) if metric.lower else np.mean(d < 0))
    return {"value": vc, "base": vb, "diff": vc - vb, "lo": float(lo), "hi": float(hi), "p_worse": worse}
