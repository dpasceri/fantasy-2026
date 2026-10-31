"""availability_hazard: P(plays) from the injury designation, practice report and history, and a discrete-time
hazard model for multi-week absences (AVAILABILITY track).

Variants (simplest first):
  cells   empirical-Bayes cell rates. Gameday: (designation, practice status), shrunk to the designation rate.
          Ahead: (h, origin tag, missed-last-game, starter role), shrunk to (h, tag) and then to h.
  hazard  structural two-state discrete-time model in team games (Singer & Willett person-period logits):
          week-t availability from a gameday logit (Q/D) or an untagged logit; then per team game an onset
          hazard lambda(x, r) for an available player (r = games since his return) and a return hazard
          mu(x, d) for an absent one (d = games missed so far), composed by forward recursion over (A_r, U_d).
          The same lambda / mu could replace sim.py's INJ_HAZARD and geometric durations per player.
  gbm     direct multi-horizon LightGBM on the ahead items (h, k and every origin feature incl. depth rank);
          gameday items use the hazard variant's gameday logit.
Every fit uses only the train / periods frames the harness passes (strictly before the test week).
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)
from bench.harness import task_threads

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

POS = ("QB", "RB", "WR", "TE")
DEPTH = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
RCAP, DCAP = 12, 12
EPS = 1e-4
# strongly regularized: the 8 horizons of one player-origin (and consecutive origins) are near-duplicates, so the
# effective sample is far smaller than the row count (31 leaves / 300 rounds overfit on 2019-20 dev origins)
GBM = {"n": 150, "lr": 0.03, "leaves": 7, "min_child": 1000, "l2": 10.0}


def _oh(x, levels):
    x = np.asarray(x)
    return np.stack([(x == v).astype(float) for v in levels], 1)


def _pos(df):
    return _oh(df["pos"].to_numpy(), POS[1:])            # QB is the reference


def _num(df, cols):
    return np.stack([df[c].to_numpy(float) for c in cols], 1)


class Logit:
    """L2 logistic regression on standardized columns (constant columns left alone)."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y, w=None):
        self.mu = X.mean(0)
        sd = X.std(0)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        y = np.asarray(y, int)
        if y.min() == y.max():                              # degenerate: constant rate
            self.const = float(np.clip(y.mean(), EPS, 1 - EPS))
            return self
        self.const = None
        self.m = LogisticRegression(C=self.C, max_iter=2000, tol=1e-6).fit((X - self.mu) / self.sd, y, sample_weight=w)
        return self

    def predict(self, X):
        if self.const is not None:
            return np.full(len(X), self.const)
        return self.m.predict_proba((X - self.mu) / self.sd)[:, 1]


# ---------------------------------------------------------------- features
def gameday_X(df):
    """Week-t features for a designated (Q/D) player."""
    d = df["d"].to_numpy()
    return np.hstack([
        (df["rep"].to_numpy() == 2)[:, None].astype(float),
        _oh(df["prac"].to_numpy(), (1, 2, 3)), _oh(df["injg"].to_numpy(), range(1, 11)), _pos(df),
        np.stack([d == 1, d >= 2], 1).astype(float), np.log1p(df["r"].to_numpy(float))[:, None],
        _oh(df["rep_prev"].to_numpy(), (1, 2, 3)), _oh(df["prac_prev"].to_numpy(), (2, 3)),
        _num(df, ["ir_ret", "rel", "pct_last", "n_rep4"]),
        ((df["age"].to_numpy(float) - 27) / 4)[:, None], (df["wk"].to_numpy(float) / 18)[:, None],
        _starter(df)[:, None]])


def untagged_X(df):
    """Week-t features for a player with no game designation."""
    d = df["d"].to_numpy()
    return np.hstack([
        (df["prac"].to_numpy() >= 1)[:, None].astype(float), _pos(df),
        np.stack([d == 1, d == 2, d >= 3], 1).astype(float), np.log1p(df["r"].to_numpy(float))[:, None],
        _oh(df["rep_prev"].to_numpy(), (1, 2, 3)),
        _num(df, ["rel", "pct_last", "pct_avg3", "miss_2y", "miss_s"]),
        ((df["age"].to_numpy(float) - 27) / 4)[:, None], (df["wk"].to_numpy(float) / 18)[:, None],
        _starter(df)[:, None]])


def _starter(df):
    pr = df["pos_rank"].to_numpy()
    lim = df["pos"].map(DEPTH).to_numpy()
    return ((pr >= 1) & (pr <= lim)).astype(float)


def _static(df):
    """Hazard covariates that stay fixed along a forecast path: position, age, role, history."""
    a = (df["age"].to_numpy(float) - 27) / 4
    return np.hstack([_pos(df), np.stack([a, a ** 2], 1), _num(df, ["rel", "miss_2y"])])


def _prev(rep_prev, prac_prev):
    if rep_prev is None:                                    # marginal model: the last report is unknown
        return np.zeros((0, 0))
    return np.hstack([_oh(rep_prev, (1, 2, 3)), _oh(prac_prev, (2, 3))])


def _cat(*blocks):
    return np.hstack([b for b in blocks if b.size])


def onset_X(S, r, wk, rep_prev=None, prac_prev=None):
    """lambda covariates: static block S, games since return r (>= 1), week of season, [last report]."""
    r = np.asarray(r, float)
    return _cat(S, np.stack([np.log(r), r == 1, r == 2], 1).astype(float), (np.asarray(wk, float) / 18)[:, None],
                _prev(rep_prev, prac_prev))


def return_X(S, d, wk, rep_prev=None, prac_prev=None):
    """mu covariates: static block S, games missed so far d (>= 1), week of season, [last report]."""
    d = np.asarray(d, float)
    return _cat(S, np.stack([np.log(d), d == 1, d == 2, d == 3], 1).astype(float),
                (np.asarray(wk, float) / 18)[:, None], _prev(rep_prev, prac_prev))


# ---------------------------------------------------------------- variants
class Candidate:
    def __init__(self, variant="hazard"):
        assert variant in ("cells", "hazard", "gbm"), variant
        self.variant = variant

    # ---- fit
    def fit(self, train, periods):
        self.base_gd = float(train.loc[train.kind == 0, "y"].mean())
        if self.variant == "cells":
            self._fit_cells(train)
            return self
        self._fit_week(train, periods)
        if self.variant == "hazard":
            self._fit_hazard(periods)
        else:
            self._fit_gbm(train)
        return self

    def _fit_week(self, train, periods):
        gd = train[train.kind == 0]
        self.gd = Logit(C=0.3).fit(gameday_X(gd), gd["y"])
        un = periods[(periods.rep == 0) & (periods.rstat == 1) & (periods.wk >= 2)]
        self.un = Logit(C=1.0).fit(untagged_X(un), un["y"])

    def _fit_hazard(self, periods):
        pe = _drop_backfilled(periods)
        pe = pe[pe.wk >= 2]                                 # week-1 streaks span the offseason
        S = _static(pe)
        a = pe[pe.d == 0]
        m = (pe.d == 0).to_numpy()
        ra, db = np.clip(a.r, 1, RCAP), np.clip(pe[~m].d, 1, DCAP)
        b = pe[~m]
        # step 1 knows the origin's report; later steps do not, so they use models marginal over it
        self.lam1 = Logit(C=1.0).fit(onset_X(S[m], ra, a.wk, a.rep_prev, a.prac_prev), 1 - a["y"])
        self.lam0 = Logit(C=1.0).fit(onset_X(S[m], ra, a.wk), 1 - a["y"])
        self.mu1 = Logit(C=1.0).fit(return_X(S[~m], db, b.wk, b.rep_prev, b.prac_prev), b["y"])
        self.mu0 = Logit(C=1.0).fit(return_X(S[~m], db, b.wk), b["y"])

    def _fit_gbm(self, train):
        import lightgbm as lgb
        ah = train[train.kind == 1]
        self.cols = gbm_cols()
        self.gbm = lgb.LGBMClassifier(
            n_estimators=GBM["n"], learning_rate=GBM["lr"], num_leaves=GBM["leaves"],
            min_child_samples=GBM["min_child"], subsample=1.0, colsample_bytree=0.8, reg_lambda=GBM["l2"],
            random_state=7, deterministic=True, force_col_wise=True, n_jobs=task_threads(), verbose=-1)
        self.gbm.fit(gbm_frame(ah, self.cols), ah["y"].to_numpy())

    def _gbm_predict(self, ah):
        return self.gbm.predict_proba(gbm_frame(ah, self.cols))[:, 1]

    def _fit_cells(self, train):
        gd = train[train.kind == 0]
        self.c_gd = Cells([["rep"], ["rep", "prac"]], m=20).fit(gd, gd["y"])
        ah = train[train.kind == 1]
        self.c_ah = Cells([["h"], ["h", "tag"], ["h", "tag", "missed", "starter"]], m=50).fit(_cell_cols(ah), ah["y"])

    # ---- predict
    def predict_proba(self, test):
        p = np.zeros(len(test))
        k0 = (test["kind"] == 0).to_numpy()
        if self.variant == "cells":
            if k0.any():
                p[k0] = self.c_gd.predict(test[k0])
            if (~k0).any():
                p[~k0] = self.c_ah.predict(_cell_cols(test[~k0]))
            return np.clip(p, EPS, 1 - EPS)
        if k0.any():
            p[k0] = self.gd.predict(gameday_X(test[k0]))
        if (~k0).any():
            ah = test[~k0]
            p[~k0] = self._hazard_ahead(ah) if self.variant == "hazard" else self._gbm_predict(ah)
        return np.clip(p, EPS, 1 - EPS)

    def week_play(self, df):
        """P(plays at the origin week) for each row (its tag decides the model)."""
        tag = df["rep"].to_numpy() >= 1
        p = np.zeros(len(df))
        if tag.any():
            p[tag] = self.gd.predict(gameday_X(df[tag]))
        if (~tag).any():
            p[~tag] = self.un.predict(untagged_X(df[~tag]))
        return p

    def _hazard_ahead(self, ah):
        """Forward recursion over (available, r) / (absent, d) per player, in his team's games."""
        first = ah.drop_duplicates("pidx").reset_index(drop=True)      # origin features are per player
        n = len(first)
        K = int(ah["k"].max())
        S = _static(first)
        game0 = first["game0"].to_numpy() > 0
        p0 = np.where(game0, self.week_play(first), np.nan)
        d0, r0 = first["d"].to_numpy(), first["r"].to_numpy()
        A = np.zeros((n, RCAP + 1))                         # A[:, r]: available, played last r games (r >= 1)
        U = np.zeros((n, DCAP + 1))                         # U[:, d]: absent d games (d >= 1)
        rows = np.arange(n)
        ra = np.clip(np.where(d0 == 0, r0 + 1, 1), 1, RCAP)
        du = np.clip(np.where(d0 == 0, 1, d0 + 1), 1, DCAP)
        A[rows[game0], ra[game0]] += p0[game0]
        U[rows[game0], du[game0]] += 1 - p0[game0]
        nog = ~game0                                        # bye at the origin: state as of his last game
        A[rows[nog & (d0 == 0)], np.clip(np.maximum(r0, 1), 1, RCAP)[nog & (d0 == 0)]] += 1
        U[rows[nog & (d0 > 0)], np.clip(d0, 1, DCAP)[nog & (d0 > 0)]] += 1
        # step 1's "last report" = the origin's report, or at a bye origin the one of his last game (as in periods)
        rep_t = np.where(game0, first["rep"].to_numpy(), first["rep_prev"].to_numpy())
        prac_t = np.where(game0, first["prac"].to_numpy(), first["prac_prev"].to_numpy())
        wk0 = first["wk"].to_numpy(float)
        span = ah.groupby("pidx").apply(lambda g: (g["h"] / g["k"].clip(lower=1)).max(), include_groups=False)
        step_w = span.reindex(first["pidx"]).to_numpy(float)
        play = np.zeros((n, K + 1))
        rr = np.arange(1, RCAP + 1)
        dd = np.arange(1, DCAP + 1)
        SR, SD = np.repeat(S, RCAP, 0), np.repeat(S, DCAP, 0)
        for j in range(1, K + 1):
            wk = wk0 + j * step_w
            if j == 1:
                lam = self.lam1.predict(onset_X(SR, np.tile(rr, n), np.repeat(wk, RCAP), np.repeat(rep_t, RCAP),
                                                np.repeat(prac_t, RCAP)))
                mu = self.mu1.predict(return_X(SD, np.tile(dd, n), np.repeat(wk, DCAP), np.repeat(rep_t, DCAP),
                                               np.repeat(prac_t, DCAP)))
            else:
                lam = self.lam0.predict(onset_X(SR, np.tile(rr, n), np.repeat(wk, RCAP)))
                mu = self.mu0.predict(return_X(SD, np.tile(dd, n), np.repeat(wk, DCAP)))
            lam, mu = lam.reshape(n, RCAP), mu.reshape(n, DCAP)
            stay = A[:, 1:] * (1 - lam)                     # available -> plays again
            back = U[:, 1:] * mu                            # absent -> returns
            play[:, j] = stay.sum(1) + back.sum(1)
            A2 = np.zeros_like(A)
            U2 = np.zeros_like(U)
            A2[:, 2:] += stay[:, :-1]
            A2[:, RCAP] += stay[:, -1]
            A2[:, 1] += back.sum(1)
            U2[:, 1] += (A[:, 1:] * lam).sum(1)
            U2[:, 2:] += (U[:, 1:-1] * (1 - mu[:, :-1]))
            U2[:, DCAP] += U[:, DCAP] * (1 - mu[:, -1])
            A, U = A2, U2
        ix = pd.Index(first["pidx"]).get_indexer(ah["pidx"])
        k = np.clip(ah["k"].to_numpy(), 1, K)
        return play[ix, k]


def _drop_backfilled(pe):
    """Drop no-roster-row periods (rstat 0) before his first roster row of the season: avail_data fills the team
    backwards within a season, so those rows (an unsigned player 'missing' his future team's games) exist only
    because of a roster row that may lie after the test week (~0.5% of periods, all y = 0)."""
    first = pe[pe.rstat != 0].groupby(["pidx", "season"]).g.min()
    f = first.reindex(pd.MultiIndex.from_arrays([pe.pidx, pe.season])).to_numpy(float)
    back = (pe.rstat.to_numpy() == 0) & (np.isnan(f) | (pe.g.to_numpy() < f))
    return pe[~back]


# ---------------------------------------------------------------- cells
def _cell_cols(df):
    out = pd.DataFrame({"h": df["h"].to_numpy(), "tag": np.clip(df["rep"].to_numpy(), 0, 2),
                        "missed": (df["d"].to_numpy() >= 1).astype(int), "starter": _starter(df).astype(int)},
                       index=df.index)
    return out


class Cells:
    """Hierarchical cell rates: each level is shrunk to its parent with a Beta prior of strength m."""

    def __init__(self, levels, m=20.0):
        self.levels, self.m = levels, m

    def fit(self, X, y):
        self.top = float(np.mean(y)) if len(y) else 0.5
        d = X.copy()
        d["_y"] = np.asarray(y, float)
        self.tabs = []
        for cols in self.levels:
            g = d.groupby(cols)["_y"].agg(["sum", "count"])
            self.tabs.append(g)
        return self

    def predict(self, X):
        p = np.full(len(X), self.top)
        for cols, g in zip(self.levels, self.tabs):
            key = pd.MultiIndex.from_frame(X[cols]) if len(cols) > 1 else pd.Index(X[cols[0]])
            s = g["sum"].reindex(key).to_numpy()
            c = g["count"].reindex(key).to_numpy()
            s, c = np.nan_to_num(s), np.nan_to_num(c)
            p = (s + self.m * p) / (c + self.m)
        return p


# ---------------------------------------------------------------- gbm
def gbm_cols():
    return ["h", "k", "rep", "prac", "injg", "ir_ret", "rep_prev", "prac_prev", "d", "r", "pct_last", "pct_avg3",
            "rel", "gap", "games_s", "miss_s", "games_2y", "miss_2y", "n_rep4", "age", "wk", "last_wk", "game0",
            "pos_rank", "posc", "starter", "tw"]


def gbm_frame(df, cols):
    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c == "posc":
            X[c] = df["pos"].map({p: i for i, p in enumerate(POS)}).astype(float)
        elif c == "starter":
            X[c] = _starter(df)
        elif c == "tw":
            X[c] = df["wk"].to_numpy(float) + df["h"].to_numpy(float)
        else:
            X[c] = df[c].to_numpy(float)
    return X
