"""Base predictive: the baseline's bucket ratio tables, with neighbouring buckets combined by a
LINEAR POOL (CDF averaging) instead of quantile averaging, plus PIT intervals and per-row reads.

The baseline interpolates two neighbouring bucket tables in quantile space (Vincentization), so its
atom at 0 is the smaller of the two buckets' atoms (Lichtendahl et al. 2013: quantile averaging is
sharper than the linear pool; with atoms it takes the min). Averaging the two CDFs with the same
log-projection weight averages the atoms instead. Everything lives on model.py's 256-level grid.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import numpy as np
from model import Q
from scipy.special import ndtri

from bench.candidates._baseline_player.ratio import fit_tables, grid_to_levels  # noqa: F401

GL = np.linspace(0.0, 1.0, Q)
U_EDGE = 0.5 / (Q - 1)        # PIT of an outcome outside the table range is censored to this edge


def _cdf_lr(t, x):
    """Left/right CDF limits F(x-), F(x) of the piecewise-linear quantile function t (Q,) on GL."""
    def F(k):                  # k = knots < x (left) or <= x (right)
        j = np.clip(k - 1, 0, Q - 2)
        den = t[j + 1] - t[j]
        fr = np.divide(x - t[j], den, out=np.zeros_like(x), where=den > 0)
        v = (j + np.clip(fr, 0.0, 1.0)) / (Q - 1)
        return np.where(k <= 0, 0.0, np.where(k >= Q, 1.0, v))
    return F(np.searchsorted(t, x, "left")), F(np.searchsorted(t, x, "right"))


def _pool_pair(ta, tb, w):
    """(n, Q) quantiles of the mixture (1 - w) A + w B of two tables, one row per weight."""
    x = np.unique(np.concatenate([ta, tb]))
    al, ar = _cdf_lr(ta, x)
    bl, br = _cdf_lr(tb, x)
    xs = np.repeat(x, 2)
    fa = np.stack([al, ar], 1).ravel()
    fb = np.stack([bl, br], 1).ravel()
    out = np.empty((len(w), Q))
    for i, wi in enumerate(w):
        if wi <= 0 or wi >= 1:                 # one table only (np.interp would take the pooled max/min)
            out[i] = ta if wi <= 0 else tb
            continue
        g = np.maximum.accumulate((1 - wi) * fa + wi * fb)
        out[i] = np.interp(GL, g, xs)
    return out


def ratio_grid_pool(tables, pos, proj):
    """(n, Q) ratio quantiles, baseline bucket weights (log projection) but CDF-averaged."""
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
        for kk in np.unique(k):
            s = np.flatnonzero(k == kk)
            R[m[s]] = _pool_pair(t[kk], t[kk + 1], w[s])
    return R


def base_grid(tables, pos, proj, pool=True):
    """(n, Q) score quantiles on the grid: proj x ratio quantiles (pooled or the baseline's)."""
    proj = np.asarray(proj, float)
    if pool:
        R = ratio_grid_pool(tables, pos, proj)
    else:
        from bench.candidates._baseline_player.ratio import ratio_grid
        R = ratio_grid(tables, pos, proj)
    return proj[:, None] * R


def pit_interval(q, y):
    """PIT interval [F(y-), F(y)] of y under each grid row q (n, Q) (an atom gives a wide interval)."""
    y = np.asarray(y, float)[:, None]
    n = len(q)
    lo = (q < y).sum(1)
    hi = (q <= y).sum(1)
    r = np.arange(n)

    def F(k):
        j = np.clip(k - 1, 0, Q - 2)
        a, b = q[r, j], q[r, j + 1]
        den = b - a
        fr = np.divide(y[:, 0] - a, den, out=np.zeros(n), where=den > 0)
        v = (j + np.clip(fr, 0.0, 1.0)) / (Q - 1)
        return np.where(k <= 0, 0.0, np.where(k >= Q, 1.0, v))
    return F(lo), F(hi)


def z_interval(ulo, uhi):
    """Probit interval (a, b) for PIT intervals; a == b marks a point. Outcomes beyond the table
    range are censored to the outer half-cell."""
    ulo, uhi = np.asarray(ulo, float), np.asarray(uhi, float)
    pt = (uhi - ulo) < 1e-9
    a = np.where(ulo <= 0, -np.inf, ndtri(np.clip(ulo, 1e-12, 1 - 1e-12)))
    b = np.where(uhi >= 1, np.inf, ndtri(np.clip(uhi, 1e-12, 1 - 1e-12)))
    up = np.clip(ulo, U_EDGE, 1 - U_EDGE)                    # points, kept off the edges
    zp = ndtri(up)
    a = np.where(pt, zp, a)
    b = np.where(pt, zp, b)
    below, above = pt & (uhi <= 0), pt & (ulo >= 1)           # outside the table: censored tails
    a = np.where(below, -np.inf, a)
    b = np.where(above, np.inf, b)
    return a, b


def read_rows(q, u):
    """Read grid rows q (n, Q) at per-row levels u (n, L) by linear interpolation (as the sim does)."""
    x = np.clip(np.asarray(u, float), 0.0, 1.0) * (Q - 1)
    i0 = np.minimum(x.astype(np.int64), Q - 2)
    fr = x - i0
    a = np.take_along_axis(q, i0, 1)
    b = np.take_along_axis(q, i0 + 1, 1)
    return a * (1 - fr) + b * fr
