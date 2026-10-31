"""Team-track candidate rival_lineup: the baseline copula sim plus a model of how real managers set lineups.

Virtual leagues have no managers (roster = the drafted starters), so every variant returns the
baseline's prediction there, bit for bit. Variants differ only on the real league:
  zero_start  as played (the harness's actual starters, like the baseline), but starters projected 0
              get lineup.ZeroModel's P(play) and score instead of a sure 0.
  roster_max  lineup unknown: ignore the given starters and play each roster's max-projection lineup
              (common.best_lineup, what run.py assumes for every rival). Reference for roster_rum.
  roster_rum  lineup unknown: lineups drawn per score draw from lineup.LineupFit (inattention mixture:
              stale last-week lineup with partially pooled manager rate lambda_m, else a probit utility
              lineup with status-quo bonus and game-time optimism), zero-proj starters scored by
              ZeroModel. Falls back to roster_max when no earlier real lineups exist (2026 week 1).
The roster_* variants answer the live chop-sheet question (who gets chopped before lineups lock);
the baseline's as-played starters are the oracle for them, so compare roster_rum with roster_max.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import numpy as np
from bench.candidates._baseline_team.copula import CopulaSim, summarize
from bench.candidates.rival_lineup import lineup as LU
from common import POS

VARIANTS = ("zero_start", "roster_max", "roster_rum")


def real_obs(train, train_leagues, zero):
    """Training team-weeks of the real league for the lineup model, and zero-proj starter events."""
    reals = sorted((L for L in train_leagues if L["kind"] == "real"), key=lambda L: L["t"])
    if not reals:
        return [], []
    d = train[train["t"].isin([L["t"] for L in reals]).to_numpy()]
    proj = {t: dict(zip(g["pid"], g["proj"].astype(float))) for t, g in d.groupby("t")}
    pos_of = dict(zip(train["pid"], train["pos"]))
    by_t = {L["t"]: L for L in reals}
    obs, zev = [], []
    for L in reals:
        prev = prev_starters(by_t.get(L["t"] - 1))
        for k in range(L["n"]):
            rid = int(L["team_ids"][k])
            st = {p for p in L["starters"][k] if p}
            pids, pos, pr, pl, s = LU.team_feats(L["roster"][k], proj.get(L["t"], {}), pos_of, prev.get(rid, set()),
                                                 lambda p, t=L["t"]: zero.train_last(p, t))
            y = np.array([p in st for p in pids], float)
            obs.append({"mgr": rid, "week": L["week"], "pos": pos, "pr": pr, "pl": pl, "s": s, "y": y})
            for j, p in enumerate(L["starters"][k]):
                if p and L["starter_proj"][k, j] <= 0 and L["starter_pos"][k, j] in POS:
                    zev.append((p, L["t"], L["starter_pts"][k, j] != 0))
    return obs, zev


def prev_starters(L):
    """{roster id: set of starters} of a league view (empty if None)."""
    if L is None:
        return {}
    return {int(r): {p for p in row if p} for r, row in zip(L["team_ids"], L["starters"])}


class RivalLineup(CopulaSim):
    def __init__(self, variant="roster_rum"):
        assert variant in VARIANTS, variant
        super().__init__(variant)
        self.rum = None

    def fit(self, train, train_leagues=None):
        super().fit(train, train_leagues)
        tl = list(train_leagues or [])
        self.pos_of = dict(zip(train["pid"], train["pos"]))
        self.zero = LU.ZeroModel(train)
        obs, zev = real_obs(train, tl, self.zero)
        self.zero.update_starters(zev)
        self.obs = obs
        reals = [L for L in tl if L["kind"] == "real"]
        self.last_real = max(reals, key=lambda L: L["t"]) if reals else None
        return self

    def _rum(self):
        if self.rum is None and self.obs:
            self.rum = LU.LineupFit(self.obs)
        return self.rum

    # ---------------------------------------------------------------- scoring helpers
    def _cols(self, W, pids, league, wk_proj, games):
        """(S, n) float32 score columns for pids: copula draws, or ZeroModel draws for eligible zeros."""
        S = W.score.shape[0]
        X = np.zeros((S, len(pids)), np.float32)
        for j, p in enumerate(pids):
            i = W.ix.get(p)
            if i is not None:
                X[:, j] = W.score[:, i]
            elif wk_proj.get(p, 0.0) <= 0:
                lp = self.zero.test_eligible(p, league["t"], games)
                if lp is not None:
                    X[:, j] = self.zero.draws(self.tables, self.pos_of.get(p, "WR"), lp, S,
                                              LU._seed(LU.SEED_Z, league["season"], league["week"], p))
        return X

    def predict(self, league):
        if league["kind"] != "real":
            return super().predict(league)
        W = self.draws(league)
        wk = league["week_df"]
        wk_proj = dict(zip(wk["pid"], wk["proj"].astype(float)))
        wk_pos = dict(zip(wk["pid"], wk["pos"]))
        games = set(wk.loc[wk["game"].notna(), "team"].dropna())
        S = W.score.shape[0]
        tot = np.zeros((S, league["n"]))
        prev = prev_starters(self.last_real) if self.last_real is not None and self.last_real["t"] == league["t"] - 1 else {}
        for k in range(league["n"]):
            if self.variant == "zero_start":
                pids = [p for p in league["starters"][k] if p]
                tot[:, k] = self._cols(W, pids, league, wk_proj, games).sum(1, dtype=np.float64)
                continue
            rid = int(league["team_ids"][k])
            pids, pos, pr, pl, s = LU.team_feats(league["roster"][k], wk_proj, wk_pos, prev.get(rid, set()),
                                                 lambda p: self.zero.test_eligible(p, league["t"], games))
            if self.variant == "roster_max":
                lu = LU.max_lineup(pos, pr)
                tot[:, k] = W.totals([[pids[j] for j in lu if j >= 0]])[:, 0]
                continue
            lus = LU.sample_lineups(self._rum(), rid, pos, pr, pl, s, S,
                                    LU._seed(LU.SEED_PRED, league["season"], league["week"], rid))
            X = np.concatenate([self._cols(W, pids, league, wk_proj, games), np.zeros((S, 1), np.float32)], 1)
            idx = np.where(lus >= 0, lus, len(pids))
            tot[:, k] = np.take_along_axis(X, idx, 1).sum(1, dtype=np.float64)
        return summarize(tot)
