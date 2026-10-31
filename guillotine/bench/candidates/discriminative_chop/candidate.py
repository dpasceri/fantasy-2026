"""Team-track candidate: learn who gets chopped directly from team features (see README.md).

fit(): cross-fitted training features (features.train_set, cached per origin), then the variant's
choice model. predict(): the test week's sim is the baseline's (copula.CopulaSim refit on all train
items, 96000 draws), so total_q / line_q are the baseline's and only p_chop changes; its summary and the
league's raw features are cached per league (pre-game only), shared by the variants of an origin.
  clogit  pure ridge conditional / rank-ordered logit on sim summaries + structural features (no sim p)
  benter  hybrid, Benter's second stage: u = a * log p_sim + b * u_clogit (out-of-fold), ridge toward (1, 0)
  gbm     hybrid, LightGBM softmax-per-league boosting from init score log p_sim (0 rounds = the sim)
"""
from bench import harness  # noqa: F401

import json

import numpy as np
from bench.candidates._baseline_team.copula import CopulaSim, summarize

from . import features as F
from . import models as M


class DiscriminativeChop:
    def __init__(self, variant="benter"):
        self.variant = variant
        self._lk = None

    def fit(self, train, train_leagues=None):
        self.key, tr = F.train_set(train, train_leagues)
        if self.variant == "gbm":
            self.model = M.Boost(M.SETS["gbm"]).fit(tr)
        elif self.variant == "benter":
            self.model = M.Benter().fit(tr)
        else:
            self.model = M.CLogit(M.SETS["clogit"], offset=False).fit(tr)
        self.train = train
        print(f"[dc] {self.variant} t<{int(train['t'].max()) + 1} leagues {tr['n_leagues']} "
              f"{json.dumps(self.model.info())}", flush=True)
        return self

    def _sim(self, league):
        """Baseline sim summary (p_chop, total_q, line_q) + raw team features X of one test league."""
        if not hasattr(self, "base"):
            self.base = CopulaSim().fit(self.train)
            self.hist = F.History(self.train)
            self.hb, self.hv = self.hist.at(np.inf)
        T = self.base.draws(league).totals(league["starters"])
        out = summarize(T)
        key = (league["season"], league["week"])
        if self._lk is None or self._lk[0] != key:
            self._lk = (key, F.week_lookup(league["week_df"]))
        out["X"] = F.team_rows(np.ascontiguousarray(T.T), league["starters"], self._lk[1], self.hist, self.hb, self.hv)
        return out

    def predict(self, league):
        out = dict(F.cached(F.test_path(self.key, league), lambda: self._sim(league), lock=False))
        out["p_chop"] = self.model.predict(out.pop("X"))     # replaces the sim's p_chop
        return out
