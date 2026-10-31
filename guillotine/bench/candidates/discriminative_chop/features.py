"""Team features for the discriminative chop models: one row per (league, team), pre-game data only.

Training weeks s < origin (cross-fitted): the baseline's marginal tables + copula refit on every train
item with t < origin EXCEPT week s, so a week's sim features never see that week's outcomes, then
S_TRAIN correlated draws of every QB/RB/WR/TE projected > 0 (baseline sampler, own seeds). Per league:
team-total mean / sd / q05 / q10 and, for the Plackett-Luce stages of the realized bottom-K order,
log P_sim(team = min of the remaining set) (stage 0 = the sim's own log P(chop)).
Test week: the baseline itself (refit on all train items, 96000 draws, rng (11, season, week)), so its
p_chop / total_q / line_q are the baseline's bit for bit.
Structural (starters, pre-game): TD and reception share of projected points, thin and empty slots,
same-NFL-team pairs, opposing-game pairs, offense environment (proj-weighted team_proj).
History (train rows with t < s only): shrunk projection bias and excess volatility per player.
Training features are cached per origin in bench/cache/discriminative_chop/ (key = hash of every .py
of this candidate, the baseline sampler and ratio modules, the core constants they use, CFG and the full
training data), so the 3 variants of one origin share one computation; the content
depends only on data before that origin.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import fcntl
import glob
import hashlib
import inspect
import json
import os
import pickle

import numpy as np
import pandas as pd
from bench.candidates._baseline_player import ratio as BR
from bench.candidates._baseline_player.ratio import fit_tables
from bench.candidates._baseline_team import copula as BC
from bench.harness import CACHE
from bench.harness import data as D
from bench.harness.metrics import LEVELS
from common import POS
from model import CORR_MIN_PROJ, Q
from sim import CHUNK

CFG = {"s_train": 8000, "seed_train": 13, "k_max": 3, "p_floor": 1e-4, "kb": 8.0, "kv": 8.0, "low": 5.0,
       "version": 1}
RAW = ("lp", "mu", "sd", "q05", "q10", "proj", "td_share", "rec_share", "n_low", "n_zero", "stack", "opp",
       "env", "h_bias", "h_vol")
FI = {k: i for i, k in enumerate(RAW)}
PRE = ["pid", "pos", "proj", "team", "game", "team_proj", "ps_pass_td", "ps_rush_td", "ps_rec_td", "ps_rec"]
CACHE_DIR = os.path.join(CACHE, "discriminative_chop")
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- player history (t < s only)
class History:
    """Per-player shrunk bias (points) and volatility index (1 = pooled) from train items before week s."""

    def __init__(self, train):
        d = train[train["item"].to_numpy(bool)]
        pos, proj, act = d["pos"].to_numpy(str), d["proj"].to_numpy(float), d["actual"].to_numpy(float)
        r = act - proj
        self.c = {}
        for P in POS:                                  # E[r^2] ~ c_pos (proj + 2): pooled scale
            m = pos == P
            self.c[P] = float((r[m] ** 2).sum() / max((proj[m] + 2).sum(), 1e-9)) if m.any() else 1.0
        e2 = r ** 2 / (np.array([self.c.get(p, 1.0) for p in pos]) * (proj + 2))
        self.pids, pix = np.unique(d["pid"].to_numpy(str), return_inverse=True)
        self.ix = {p: i for i, p in enumerate(self.pids)}
        self.weeks = np.unique(d["t"].to_numpy())
        wix = np.searchsorted(self.weeks, d["t"].to_numpy())
        shp = (len(self.pids), len(self.weeks))
        self.cum = []
        for v in (np.ones(len(r)), r, e2):             # cumulative over weeks: n, sum r, sum e2
            M = np.zeros(shp)
            np.add.at(M, (pix, wix), v)
            self.cum.append(np.cumsum(M, axis=1))

    def at(self, s):
        """(bias, vol) arrays over self.pids using weeks t < s."""
        k = int(np.searchsorted(self.weeks, s, side="left"))      # columns < k are weeks < s
        if k == 0:
            z = np.zeros(len(self.pids))
            return z, z + 1.0
        n, sr, se = (c[:, k - 1] for c in self.cum)
        return sr / (n + CFG["kb"]), (se + CFG["kv"]) / (n + CFG["kv"])


# ---------------------------------------------------------------- per-league features
def week_lookup(wk):
    """pid -> row of the week's pre-game frame, plus the per-player columns the features need."""
    sc = D.settings()["scoring"]

    def col(c):
        return wk[c].fillna(0.0).to_numpy(float) if c in wk else np.zeros(len(wk))
    proj = col("proj")
    return {"ix": {p: i for i, p in enumerate(wk["pid"].astype(str))}, "proj": proj,
            "td": sc.get("pass_td", 0) * col("ps_pass_td") + sc.get("rush_td", 0) * col("ps_rush_td")
            + sc.get("rec_td", 0) * col("ps_rec_td"),
            "rec": sc.get("rec", 0) * col("ps_rec"), "tp": col("team_proj"),
            "team": wk["team"].fillna("").astype(str).to_numpy(), "game": wk["game"].fillna("").astype(str).to_numpy(),
            "pos": wk["pos"].fillna("").astype(str).to_numpy()}


def struct_feats(starters, lk, hist, hb, hv):
    """(N, 10) proj, td_share, rec_share, n_low, n_zero, stack, opp, env, h_bias, h_vol (env NaN-safe)."""
    N = len(starters)
    out = np.zeros((N, 10))
    for k, row in enumerate(starters):
        ii = [lk["ix"].get(p, -1) for p in row]
        ok = [i for i in ii if i >= 0 and lk["proj"][i] > 0]
        pj = np.array([lk["proj"][i] if i >= 0 else 0.0 for i in ii])
        tot = sum(lk["proj"][i] for i in ok)
        tm = [lk["team"][i] for i in ok]
        gm = [lk["game"][i] for i in ok]
        stack = opp = 0
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                if tm[a] and tm[a] == tm[b]:
                    stack += 1
                elif gm[a] and gm[a] == gm[b]:
                    opp += 1
        env = sum(lk["proj"][i] * lk["tp"][i] for i in ok) / tot if tot > 0 else np.nan
        bias = vsum = v0 = 0.0
        for i, p in zip(ii, row):
            if i < 0 or lk["proj"][i] <= 0:
                continue
            j = hist.ix.get(p)
            v = hist.c.get(lk["pos"][i], 1.0) * (lk["proj"][i] + 2)
            v0 += v
            if j is not None:
                bias += hb[j]
                vsum += v * (hv[j] - 1.0)
        out[k] = [tot, sum(lk["td"][i] for i in ok) / max(tot, 1.0), sum(lk["rec"][i] for i in ok) / max(tot, 1.0),
                  (pj < CFG["low"]).sum(), (pj <= 0).sum(), stack, opp, env, bias, vsum / max(v0, 1e-9)]
    if np.isnan(out[:, 7]).any():
        out[np.isnan(out[:, 7]), 7] = np.nanmean(out[:, 7]) if np.isfinite(out[:, 7]).any() else 0.0
    return out


def sim_feats(Tt):
    """(N, 5) lp, mu, sd, q05, q10 from (N, S) simulated totals (one row per team)."""
    N, S = Tt.shape
    p = np.bincount(Tt.argmin(0), minlength=N) / S
    q = np.quantile(Tt, [0.05, 0.10], axis=1)
    return np.column_stack([np.log(np.maximum(p, CFG["p_floor"])), Tt.mean(1), Tt.std(1), q[0], q[1]])


def totals_t(scT, ix, starters):
    """(N, S) team totals from player-major draws scT (n + 1, S); unknown / empty pid = the zero row."""
    idx = np.array([[ix.get(p, len(scT) - 1) for p in row] for row in starters], np.int64)
    return scT[idx].sum(1, dtype=np.float64)


def team_rows(Tt, starters, lk, hist, hb, hv):
    """(N, len(RAW)) features of one league; Tt = (N, S) simulated team totals."""
    X = np.zeros((len(starters), len(RAW)))
    X[:, :5] = sim_feats(Tt)
    X[:, 5:] = struct_feats(starters, lk, hist, hb, hv)
    return X


# ---------------------------------------------------------------- training set (cross-fitted, cached)
def _items(train):
    d = train[train["item"].to_numpy(bool)]
    t = d["t"].to_numpy()
    ts = t.astype(str)
    return {"t": t, "pos": d["pos"].to_numpy(str), "proj": d["proj"].to_numpy(float),
            "act": d["actual"].to_numpy(float), "tk": ts + "-" + d["team"].astype(str).to_numpy(),
            "gk": ts + "-" + d["game"].astype(str).to_numpy()}


def build_train(train, leagues):
    """Training arrays: team rows X (R, F) with league / week ids, and Plackett-Luce stage rows."""
    it = _items(train)
    hist = History(train)
    by_t = {}
    for L in leagues:
        by_t.setdefault(int(L["t"]), []).append(L)
    X, t_row, lg_row, n_row = [], [], [], []
    st_team, st_off, st_stage, st_grp, g_t, g_lg, g_stage = [], [], [], [], [], [], []
    li = rows = ng = 0
    tcol = train["t"].to_numpy()
    for s in sorted(by_t):
        m = it["t"] != s
        tables = fit_tables(it["pos"][m], it["proj"][m], it["act"][m])
        a, g = BC.fit_copula(it["pos"][m], it["proj"][m], it["act"][m], it["tk"][m], it["gk"][m])
        wk = train.loc[tcol == s, [c for c in PRE if c in train]]
        L0 = by_t[s][0]
        W = BC.WeekDraws(wk, tables, {p: float(a[i]) for i, p in enumerate(POS)}, g, CFG["s_train"],
                         (CFG["seed_train"], int(L0["season"]), int(L0["week"])))
        scT = np.ascontiguousarray(W.score.T)          # player-major: fast per-team gathers
        lk = week_lookup(wk)
        hb, hv = hist.at(s)
        for L in by_t[s]:
            T = totals_t(scT, W.ix, L["starters"])
            N = T.shape[0]
            X.append(team_rows(T, L["starters"], lk, hist, hb, hv))
            t_row += [s] * N
            lg_row += [li] * N
            n_row += [N] * N
            order = np.argsort(np.asarray(L["points"], float), kind="stable")   # ties: lower index (= harness)
            assert order[0] == L["chopped"]
            Tm = T.copy()
            for j in range(min(CFG["k_max"], N - 1)):
                mem = order[j:]
                c = np.bincount(Tm.argmin(0), minlength=N)[mem] / T.shape[1]
                Tm[order[j]] = np.inf                   # stage j + 1: min of the teams not yet out
                st_team += list(rows + mem)
                st_off += list(np.log(np.maximum(c, CFG["p_floor"])))
                st_stage += [j] * len(mem)
                st_grp += [ng] * len(mem)      # chosen = first member of each group
                g_t.append(s)
                g_lg.append(li)
                g_stage.append(j)
                ng += 1
            rows += N
            li += 1
        del W, scT
    return {"X": np.vstack(X), "t": np.array(t_row), "lg": np.array(lg_row), "n": np.array(n_row),
            "st_team": np.array(st_team), "st_off": np.array(st_off), "st_stage": np.array(st_stage),
            "st_grp": np.array(st_grp), "g_t": np.array(g_t), "g_lg": np.array(g_lg),
            "g_stage": np.array(g_stage),
            "n_leagues": li}


def _key(train, leagues):
    h = hashlib.sha1(json.dumps(CFG, sort_keys=True).encode())
    for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):   # all of this candidate (candidate._sim makes test entries)
        with open(f, "rb") as fh:
            h.update(fh.read())
    for mod in (BC, BR):                                       # baseline sampler + marginal tables (ratio_grid too)
        h.update(inspect.getsource(mod).encode())
    h.update(repr((Q, CHUNK, CORR_MIN_PROJ, np.asarray(LEVELS).tolist(),
                   sorted(D.settings()["scoring"].items()))).encode())
    h.update(pd.util.hash_pandas_object(train, index=False).to_numpy().tobytes())
    h.update(json.dumps([[L["league_id"], np.asarray(L["starters"]).tolist(),
                          np.round(np.asarray(L["points"], float), 9).tolist()] for L in leagues]).encode())
    return h.hexdigest()[:20]


def cached(path, compute, lock=True):
    """Load path, or compute() and write it atomically (under an exclusive lock unless lock=False).
    DC_NOCACHE=1 always recomputes."""
    if os.environ.get("DC_NOCACHE"):
        return compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lf = open(path + ".lock", "w") if lock else None
    try:
        if lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        out = compute()
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return out
    finally:
        if lf:
            lf.close()


def train_set(train, leagues):
    """(key, build_train()), cached per training set: the first task of an origin computes it, the other
    variants of that origin wait on the lock and load it."""
    key = _key(train, leagues)
    path = os.path.join(CACHE_DIR, f"train_t{int(train['t'].max()):02d}_{key}.pkl")
    return key, cached(path, lambda: build_train(train, leagues))


def test_path(key, league):
    """Cache file of one test league's sim summary + raw features (pre-game only): key = training key
    (code, CFG, all training data) + the league's week frame and starters."""
    h = hashlib.sha1(pd.util.hash_pandas_object(league["week_df"], index=False).to_numpy().tobytes())
    h.update(json.dumps(np.asarray(league["starters"]).tolist()).encode())
    return os.path.join(CACHE_DIR, f"test_{key}", f"{league['league_id']}_{h.hexdigest()[:12]}.pkl")
