"""Player-track baseline: model.py's marginal, refit on the training rows only.

actual/projected ratio, empirical quantiles on model.py's 256-level grid by position and
projection bucket (nb = clip(n // 300, 5, 14) equal-count buckets), interpolated in log
projection between bucket medians exactly like PlayerModel.quantiles; then read off at the
harness levels by the linear interpolation sim.Universe.paths applies to the 256-level grid
(u = level * 255), so these are the quantiles the sim actually samples from.
Fit on model.py's rows (2025 weeks 1-18 + 2026 weeks 1-7) it reproduces data/model.npz (verify.py).
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import numpy as np
from common import POS
from model import LEVELS as GRID, Q


def fit_tables(pos, proj, act):
    """{pos: (centers, tables (nb, Q))}, the same computation as model.fit's marginal."""
    pos, proj, act = np.asarray(pos), np.asarray(proj, float), np.asarray(act, float)
    ratio = act / proj
    out = {}
    for P in POS:
        m = np.flatnonzero(pos == P)
        if not len(m):
            continue
        nb = int(np.clip(len(m) // 300, 5, 14))
        edges = np.quantile(proj[m], np.linspace(0, 1, nb + 1))
        b = np.clip(np.searchsorted(edges, proj[m], side="right") - 1, 0, nb - 1)
        centers, tables = [], []
        for k in range(nb):
            idx = m[b == k]
            if not len(idx):        # tied edges on a tiny training set; never happens on model.py's rows
                continue
            centers.append(np.median(proj[idx]))
            tables.append(np.quantile(np.sort(ratio[idx]), GRID))
        out[P] = (np.array(centers), np.array(tables))
    return out


def ratio_grid(tables, pos, proj):
    """(n, Q) ratio quantiles, vectorized PlayerModel.quantiles / proj (0 for proj <= 0 or unknown pos)."""
    pos, proj = np.asarray(pos), np.asarray(proj, float)
    R = np.zeros((len(proj), Q))
    for P, (c, t) in tables.items():
        m = np.flatnonzero((pos == P) & (proj > 0))
        if not len(m):
            continue
        if len(c) == 1:
            R[m] = t[0]
            continue
        x, lc = np.log(proj[m]), np.log(c)
        k = np.clip(np.searchsorted(lc, x) - 1, 0, len(lc) - 2)
        den = lc[k + 1] - lc[k]
        w = np.clip(np.divide(x - lc[k], den, out=np.zeros_like(x), where=den > 0), 0.0, 1.0)
        R[m] = (1 - w)[:, None] * t[k] + w[:, None] * t[k + 1]
    return R


def grid_to_levels(q, levels):
    """Read (n, Q) grid quantiles at arbitrary levels the way sim.Universe.paths does."""
    u = np.asarray(levels, float) * (Q - 1)
    i0 = np.minimum(u.astype(np.int64), Q - 2)
    fr = u - i0
    return q[:, i0] * (1 - fr) + q[:, i0 + 1] * fr


class EmpiricalRatio:
    def __init__(self, variant="baseline"):
        self.variant = variant

    def fit(self, train):
        d = train[train["item"].to_numpy(bool)]   # = model.calibration_rows: proj >= 0.5 and a game
        self.tables = fit_tables(d["pos"].to_numpy(), d["proj"].to_numpy(), d["actual"].to_numpy())
        return self

    def predict_quantiles(self, test, levels):
        proj = test["proj"].to_numpy(float)
        q = proj[:, None] * ratio_grid(self.tables, test["pos"].to_numpy(), proj)
        return grid_to_levels(q, levels)
