"""Team-track baseline: model.py's player marginal + one-factor Gaussian copula, simulated like sim.Universe.

Marginal: _baseline_player's bucketed actual/proj quantile tables, refit per origin on train items.
Dependence: model.fit's copula refit on the same train items: randomized-PIT z per bucket (seed 7),
same-team and same-game residual correlations of players projected >= 3, loadings a_pos by least
squares, game share g. A 1-week horizon needs no level drift or injury hazard; the leaky inj tags are
not used, so every starter is available (p_out = 0, as in backtest.py: past-week projections already
zero the players ruled out). Lineups are the league's starters, no bench subs (Sleeper doesn't sub).
Draws: S = 96000 (= backtest.py) in chunks of sim.CHUNK, rng = default_rng([SEED, season, week, chunk]),
over every QB/RB/WR/TE projected > 0 that week, so every league of a week shares the same draws
(common random numbers). sample_chunk() is Universe.paths' week-0 draw verbatim (verify.py: identical).
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import numpy as np
from bench.candidates._baseline_player.ratio import fit_tables, ratio_grid
from bench.harness.metrics import LEVELS
from common import POS
from model import CORR_MIN_PROJ, Q
from scipy.optimize import least_squares
from scipy.special import ndtr, ndtri
from sim import CHUNK

SEED = 11
S_DRAWS = 96000


def fit_copula(pos, proj, act, team_key, game_key, seed=7):
    """model.fit's copula on these rows (in this order): (loadings (4,) in POS order, game share g)."""
    pos, proj, act = np.asarray(pos), np.asarray(proj, float), np.asarray(act, float)
    rng = np.random.default_rng(seed)
    ratio = act / proj
    z = np.zeros(len(proj))
    for P in POS:
        m = np.flatnonzero(pos == P)
        if not len(m):
            continue
        nb = int(np.clip(len(m) // 300, 5, 14))
        edges = np.quantile(proj[m], np.linspace(0, 1, nb + 1))
        b = np.clip(np.searchsorted(edges, proj[m], side="right") - 1, 0, nb - 1)
        for k in range(nb):
            idx = m[b == k]
            if not len(idx):
                continue
            r = np.sort(ratio[idx])
            lo = np.searchsorted(r, ratio[idx], side="left")
            hi = np.searchsorted(r, ratio[idx], side="right")
            u = (lo + rng.random(len(idx)) * (hi - lo)) / len(r)
            z[idx] = ndtri(np.clip(u, 1e-6, 1 - 1e-6))
    same, cross, by_team, by_game = {}, {}, {}, {}
    for i in np.flatnonzero(proj >= CORR_MIN_PROJ):
        by_team.setdefault(team_key[i], []).append(i)
        by_game.setdefault(game_key[i], []).append(i)
    for grp in by_team.values():
        for x in range(len(grp)):
            for y in range(x + 1, len(grp)):
                i, j = grp[x], grp[y]
                s = same.setdefault(tuple(sorted((str(pos[i]), str(pos[j])))), [0.0, 0])
                s[0] += z[i] * z[j]
                s[1] += 1
    for grp in by_game.values():
        for x in range(len(grp)):
            for y in range(x + 1, len(grp)):
                i, j = grp[x], grp[y]
                if team_key[i] == team_key[j]:
                    continue
                s = cross.setdefault(tuple(sorted((str(pos[i]), str(pos[j])))), [0.0, 0])
                s[0] += z[i] * z[j]
                s[1] += 1
    pairs = [(k, v[0] / v[1], v[1]) for k, v in same.items() if v[1] >= 50 and k != ("QB", "QB")]
    pi = {p: n for n, p in enumerate(POS)}

    def resid(a):
        return [np.sqrt(n) * (c - a[pi[k[0]]] * a[pi[k[1]]]) for k, c, n in pairs]

    a = least_squares(resid, x0=np.full(4, 0.3), bounds=(0.0, 0.95)).x if pairs else np.full(4, 0.3)
    num = sum(n * c * a[pi[k[0]]] * a[pi[k[1]]] for k, (sc, n) in cross.items() if n >= 50 for c in [sc / n])
    den = sum(n * (a[pi[k[0]]] * a[pi[k[1]]]) ** 2 for k, (sc, n) in cross.items() if n >= 50)
    g = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0
    return np.asarray(a, float), g


def sample_chunk(qflat, load, cload, team, game, n_team, n_game, g, rng, C):
    """(C, n + 1) float32 scores, the last column an empty slot: sim.Universe.paths' week-0 draw."""
    n = len(load)
    G = rng.standard_normal((C, n_game), dtype=np.float32)
    T = rng.standard_normal((C, n_team), dtype=np.float32)
    z = load * (np.float32(np.sqrt(g)) * G[:, game] + np.float32(np.sqrt(1 - g)) * T[:, team])
    z += cload * rng.standard_normal((C, n), dtype=np.float32)
    u = ndtr(z) * np.float32(Q - 1)
    i0 = np.minimum(u.astype(np.int64), Q - 2)
    fr = u - i0
    base = np.arange(n) * Q + i0
    score = np.zeros((C, n + 1), np.float32)
    score[:, :n] = qflat[base] * (1 - fr) + qflat[base + 1] * fr
    return score


class WeekDraws:
    """Every QB/RB/WR/TE projected > 0 in a week frame, with S correlated score draws."""

    def __init__(self, wk, tables, loading, g, S, seed):
        m = (wk["proj"].to_numpy(float) > 0) & wk["pos"].isin(POS).to_numpy()
        w = wk[m]
        self.pids = w["pid"].astype(str).tolist()
        self.ix = {p: i for i, p in enumerate(self.pids)}
        n = self.n = len(self.pids)
        pos, proj = w["pos"].to_numpy(str), w["proj"].to_numpy(float)
        qflat = (proj[:, None] * ratio_grid(tables, pos, proj)).astype(np.float32).ravel()
        load = np.array([loading.get(p, 0.0) for p in pos], np.float32)
        cload = np.sqrt(1 - load ** 2).astype(np.float32)
        teams, games = {}, {}
        team = np.array([teams.setdefault(t if isinstance(t, str) and t else f"_{i}", len(teams))
                         for i, t in enumerate(w["team"])], np.int64)
        game = np.array([games.setdefault(x if isinstance(x, str) and x else f"_{i}", len(games))
                         for i, x in enumerate(w["game"])], np.int64)
        nc = max(1, S // CHUNK)
        self.score = np.empty((nc * CHUNK, n + 1), np.float32)
        for ci in range(nc):
            rng = np.random.default_rng([*seed, ci])
            self.score[ci * CHUNK:(ci + 1) * CHUNK] = sample_chunk(qflat, load, cload, team, game, len(teams),
                                                                  len(games), g, rng, CHUNK)

    def totals(self, starters):
        """(S, N) float64 team totals for an (N, slots) pid array ('' or unknown = 0)."""
        idx = np.array([[self.ix.get(p, self.n) for p in row] for row in starters], np.int64)
        return self.score[:, idx].sum(-1, dtype=np.float64)


def summarize(tot):
    """p_chop, total_q (N, 99), line_q (99,) from (S, N) simulated totals."""
    N = tot.shape[1]
    p = np.bincount(tot.argmin(1), minlength=N) / len(tot)
    return {"p_chop": p, "total_q": np.quantile(tot, LEVELS, axis=0).T, "line_q": np.quantile(tot.min(1), LEVELS)}


class CopulaSim:
    def __init__(self, variant="baseline"):
        self.variant = variant
        self.S = S_DRAWS
        self._week = None

    def fit(self, train, train_leagues=None):
        d = train[train["item"].to_numpy(bool)]       # = model.calibration_rows: proj >= 0.5 and a game
        pos, proj, act = d["pos"].to_numpy(str), d["proj"].to_numpy(float), d["actual"].to_numpy(float)
        self.tables = fit_tables(pos, proj, act)
        t = d["t"].astype(str).to_numpy()
        a, self.g = fit_copula(pos, proj, act, t + "-" + d["team"].astype(str).to_numpy(),
                               t + "-" + d["game"].astype(str).to_numpy())
        self.loading = {p: float(a[i]) for i, p in enumerate(POS)}
        return self

    def set_params(self, tables, loading, g):
        """Use given parameters instead of fit() (verify.py: data/model.npz)."""
        self.tables, self.loading, self.g = tables, dict(loading), float(g)
        return self

    def draws(self, league):
        key = (league["season"], league["week"])
        if self._week is None or self._week[0] != key:
            self._week = (key, WeekDraws(league["week_df"], self.tables, self.loading, self.g, self.S, (SEED, *key)))
        return self._week[1]

    def predict(self, league):
        return summarize(self.draws(league).totals(league["starters"]))
