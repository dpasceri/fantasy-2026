"""PLAYER-track candidate: forecast combination of Sleeper's projection with point-in-time signals.

Variants (each nests the previous one):
  lpool  the baseline's bucket tables, neighbouring buckets combined by a linear pool (CDF average)
         instead of quantile averaging, which averages their zero atoms (no fitted parameters).
  recal  lpool + a two-part recalibration by position and projection level only (no information
         beyond Sleeper): P(y <= 0) by a logistic with the base's own mass as offset, and the base's
         positive part re-shaped by a probit location-scale map of its PIT.
  stack  the same two parts, but both are stacked on point-in-time signals: the player's past PIT
         residuals and excess zeros (EWMA), usage (snap share), recent projection change and missed
         games, opponent x position points allowed (PIT units), team environment, depth-chart rank
         and the stat mix of the projection. Their ridge strength is chosen per origin by an inner
         rolling validation on the last training weeks (stacking weights from out-of-sample scores).
Both recalibrations are fitted on cross-fitted PITs: each training week is scored by a base fitted on
every OTHER training week (blocked leave-one-week-out, the level-1 data of stacking), so they learn
how a base of nearly the final sample size misses out of sample. A forward-only inner origin was the
first design, but its early bases (2-5 weeks of data) are much noisier than the final one: on 2026-07's
training weeks the QB probit sd of its PITs is 1.076 (1.013 over the last 10 weeks) against 0.998
cross-fitted, so it taught the recalibration to over-widen. Everything stays strictly before the test
week (train_df only); the penalty selection below is still forward-only.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import numpy as np
import pandas as pd
from scipy.special import logit, ndtr, ndtri

from . import base as B
from . import signals as S
from .recal import LogitOffset, ProbitRecal

POS = S.POS
HL_ROWS = 20.0       # recency half-life (weeks) of the recalibration's training rows
PEN_INT = 1.0                                      # ridge precisions (standardized signals):
PEN_SHARED, PEN_POS = 20.0, 60.0                   #   positive part location: shared / per-position slopes
PEN_SIG, PEN_SIG_F = 20.0, 60.0                    #   positive part log-scale: pos x lp / shared signal slopes
PEN_L_SHARED, PEN_L_POS = 10.0, 30.0               #   zero part (logit units): shared / per-position slopes
PEN_MULT = (1.0, 3.0, 10.0, 30.0, 100.0, 1e4)      # stack: signal-penalty multipliers tried by inner validation
                                                   # (1e4 ~ no signals; the first grid 0.3..10 hit its upper edge)
N_VAL = 4                                          # ... on the last N_VAL training weeks (refit before each)
F0_CLIP = 1e-4
STACK_FEATS = S.NAMES
RECAL_FEATS = ("lp",)


class ProjectionEnsemble:
    def __init__(self, variant="stack"):
        assert variant in ("lpool", "recal", "stack"), variant
        self.variant = variant

    # ------------------------------------------------------------ fit
    def fit(self, train):
        it = train[train["item"].to_numpy(bool)].sort_values(["t", "pid"], kind="stable").reset_index(drop=True)
        pos, proj, act = it["pos"].to_numpy(), it["proj"].to_numpy(float), it["actual"].to_numpy(float)
        self.tables = B.fit_tables(pos, proj, act)
        if self.variant == "lpool":
            return self
        # cross-fitting: PIT of every training week under a base fitted on all the other training weeks
        t = it["t"].to_numpy(int)
        weeks = np.unique(t)
        n = len(it)
        zb, ez, F0 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        va, vb = np.full(n, np.nan), np.full(n, np.nan)
        grids = {}
        for s in weeks if len(weeks) > 1 else []:
            other, cur = t != s, np.flatnonzero(t == s)
            q = B.base_grid(B.fit_tables(pos[other], proj[other], act[other]), pos[cur], proj[cur])
            y = act[cur]
            lo, hi = B.pit_interval(q, y)
            zb[cur] = S.zbar(*B.z_interval(lo, hi))
            l0, h0 = B.pit_interval(q, np.zeros(len(cur)))
            F0[cur] = h0
            ez[cur] = (y == 0).astype(float) - (h0 - l0)
            den = np.maximum(1.0 - h0, 1e-12)              # positive part: PIT within (F(0), 1]
            a, b = B.z_interval(np.clip((lo - h0) / den, 0, 1), np.clip((hi - h0) / den, 0, 1))
            pz = y > 0
            va[cur[pz]], vb[cur[pz]] = a[pz], b[pz]
            grids[s] = (cur, q, y, h0)
        it["zb"], it["ez"] = zb, ez
        self.items = it
        self.rows = train[["t", "season", "pid", "pos", "team", "opp", "proj", "as_off_snp"]]
        X = S.build(it, self.rows)
        m = np.flatnonzero(~np.isnan(F0))                 # rows with a cross-fitted PIT
        if len(m) < 100:                                  # too little to recalibrate: stay on the base
            self.model = None
            return self
        self._D = dict(zip(("Xm", "Xs"), self._design(it.iloc[m], X[m], fit=True)))
        self._D.update(t=t[m], off=self._off(F0[m]), e=(act[m] <= 0).astype(float), va=va[m], vb=vb[m],
                       row=m)
        self.mult, self.val = 1.0, {}
        if self.variant == "stack":
            self.val = self._select(grids)
            if self.val:
                self.mult = min(self.val, key=self.val.get)
        self.model = self._fit_rows(np.ones(len(m), bool), int(t.max()), self.mult)
        return self

    @staticmethod
    def _off(F0):
        return logit(np.clip(F0, F0_CLIP, 1 - F0_CLIP))

    def _design(self, rows, X, fit=False):
        """mu / logit design [pos one-hot, shared slopes, pos x slopes]; log-s design [pos one-hot,
        pos x lp, shared signal slopes (stack only)]. Signals are standardized on the training rows."""
        feats = STACK_FEATS if self.variant == "stack" else RECAL_FEATS
        F = X[:, [S.NAMES.index(f) for f in feats]]
        lp = X[:, [S.NAMES.index("lp")]]
        oh = np.stack([(rows["pos"].to_numpy() == p).astype(float) for p in POS], 1)
        if fit:
            self.mu_f, self.sd_f = F.mean(0), F.std(0) + 1e-9
            self.mu_lp, self.sd_lp = lp.mean(0), lp.std(0) + 1e-9
        F = (F - self.mu_f) / self.sd_f
        lp = (lp - self.mu_lp) / self.sd_lp
        Fp = (oh[:, :, None] * F[:, None, :]).reshape(len(F), -1)
        Xm = np.concatenate([oh, F, Fp], 1)
        Fs = F if self.variant == "stack" else F[:, :0]       # stack: signals also move the scale
        Xs = np.concatenate([oh, oh * lp, Fs], 1)
        if fit:
            k, ks = F.shape[1], Fs.shape[1]
            self.pen_m = np.r_[np.full(4, PEN_INT), np.full(k, PEN_SHARED), np.full(4 * k, PEN_POS)]
            self.pen_l = np.r_[np.full(4, PEN_INT), np.full(k, PEN_L_SHARED), np.full(4 * k, PEN_L_POS)]
            self.pen_s = np.r_[np.full(4, PEN_INT), np.full(4, PEN_SIG), np.full(ks, PEN_SIG_F)]
            self.slope_m = np.r_[np.zeros(4, bool), np.ones(5 * k, bool)]      # scaled by the multiplier
            self.slope_s = np.r_[np.zeros(8, bool), np.ones(ks, bool)]
        return Xm, Xs

    def _fit_rows(self, sel, t_end, mult):
        """(zero-part logistic, positive-part probit recalibration) fitted on the selected rows."""
        D = self._D
        w = 0.5 ** ((t_end - D["t"][sel]) / HL_ROWS)
        sm = np.where(self.slope_m, mult, 1.0)
        lg = LogitOffset().fit(D["Xm"][sel], D["off"][sel], D["e"][sel], w, self.pen_l * sm)
        pp = D["e"][sel] == 0
        pr = ProbitRecal().fit(D["Xm"][sel][pp], D["Xs"][sel][pp], D["va"][sel][pp], D["vb"][sel][pp], w[pp],
                               self.pen_m * sm, self.pen_s * np.where(self.slope_s, mult, 1.0))
        return lg, pr

    @staticmethod
    def _quantiles(model, q, F0, Xm, Xs, levels):
        """Combined quantiles: level tau < p0 maps into the base's mass at or below 0, the rest into
        its positive part through the probit map."""
        lg, pr = model
        tau = np.asarray(levels, float)[None, :]
        p0 = np.clip(lg.prob(Xm, ProjectionEnsemble._off(F0)), 1e-6, 1 - 1e-6)[:, None]
        F0 = F0[:, None]
        low = tau < p0
        v = np.clip((tau - p0) / (1 - p0), 1e-12, 1 - 1e-12)
        mu, s = (Xm @ pr.beta)[:, None], np.exp(Xs @ pr.gam)[:, None]
        u = np.where(low, F0 * tau / p0, F0 + (1 - F0) * ndtr(mu + s * ndtri(v)))
        out = B.read_rows(q, u)
        return np.where(low, np.minimum(out, 0.0), out)     # the low branch is y <= 0 even where the base has no mass there

    def _select(self, grids):
        """Mean CRPS of each penalty multiplier over the last N_VAL training weeks, each scored by
        models fitted on the rows of earlier weeks only, with that week's cross-fitted base."""
        from bench.harness.metrics import LEVELS, crps
        D = self._D
        wk = np.unique(D["t"])
        val = [v for v in wk[-N_VAL:] if (wk < v).sum() >= 2]
        if not val:
            return {}
        where = {r: i for i, r in enumerate(D["row"])}
        out = {}
        for mult in PEN_MULT:
            tot, n = 0.0, 0
            for v in val:
                mod = self._fit_rows(D["t"] < v, int(v) - 1, mult)
                cur, q, y, F0 = grids[v]
                rr = np.array([where[c] for c in cur])
                tot += crps(self._quantiles(mod, q, F0, D["Xm"][rr], D["Xs"][rr], LEVELS), y).sum()
                n += len(y)
            out[float(mult)] = round(tot / n, 6)
        return out

    # ------------------------------------------------------------ predict
    def predict_quantiles(self, test, levels):
        q = B.base_grid(self.tables, test["pos"].to_numpy(), test["proj"].to_numpy(float))
        if self.variant == "lpool" or self.model is None:
            return B.grid_to_levels(q, levels)
        te = test.copy()
        te["zb"], te["ez"] = np.nan, np.nan
        cols = [c for c in self.items.columns if c in te.columns]
        allr = pd.concat([self.items[cols], te[cols]], ignore_index=True)
        X = S.build(allr, self.rows)[len(self.items):]
        Xm, Xs = self._design(te, X)
        l0, h0 = B.pit_interval(q, np.zeros(len(te)))
        return self._quantiles(self.model, q, h0, Xm, Xs, levels)


Candidate = ProjectionEnsemble
