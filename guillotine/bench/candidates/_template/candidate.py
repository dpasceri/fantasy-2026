"""Template player-track candidate: additive residual quantiles by position.

q(level) = proj + empirical quantile of (actual - proj) over training items of the same position.
Variant "recent" uses only the last 8 training weeks. A deliberately simple example of the API.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)

import numpy as np


class Candidate:
    def __init__(self, variant="additive"):
        self.variant = variant

    def fit(self, train):
        # train: every scored player-week strictly before the test week (all players);
        # restrict to items (proj >= 0.5 and a game) like the baseline does
        d = train[train["item"].to_numpy(bool)]
        if self.variant == "recent":
            d = d[d["t"] > d["t"].max() - 8]
        self.res = {p: np.sort((g["actual"] - g["proj"]).to_numpy()) for p, g in d.groupby("pos")}
        return self

    def predict_quantiles(self, test, levels):
        # test: the test week's items, pre-game columns only; return (len(test), len(levels))
        q = np.zeros((len(test), len(levels)))
        pos, proj = test["pos"].to_numpy(), test["proj"].to_numpy(float)
        for p, r in self.res.items():
            m = pos == p
            q[m] = proj[m, None] + np.quantile(r, levels)[None, :]
        return q
