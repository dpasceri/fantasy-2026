"""TEAM / CHOP track: P(chopped) for every alive team of a guillotine week, plus each team total's
distribution and the chop line (the week's lowest total).

Items are leagues, one guillotine week each; block = t (global week index) of the test week.
- real (kind "real", id r-2026-03): this league, 2026 weeks 1-7. Every alive team (its roster had
  players) with the starters it actually played (point-in-time, from matchups); total = sum of the
  starters' league-scoring actual points (= Sleeper's points to 3e-14); chopped = argmin
  (= data/backtest.json's chopped owner, asserted). Only 7 items: low power, real conditions.
- virtual (kind "virtual", id v-2025-05-07 = season-week-replicate): GEN["r"] deterministic synthetic
  guillotine leagues per scored week, N = 10 + r % 8 teams. Each team's starting slots are filled by a
  seeded randomized snake draft from that week's projections (players projected > 0, each at most once
  per league); team k drafts off its own noisy board log(proj) + sigma_k * eps with
  sigma_k = SIGMA0 * exp(TAU * z_k), which spreads team strength like the real league (CV of projected
  starter totals 0.096 vs 0.096 real, level 90 at N=17 to 104 at N=10 vs 92 to 103 real; calibrated on
  projections only, never on outcomes). Total = sum of the starters' actual points; chopped = argmin.
  Built for every scored week and cached in bench/cache/team_leagues_<sig>.json, so every candidate
  sees identical leagues; full mode scores r < 48 per test week, smoke r < 16.
Rolling origin: for each test week a fresh candidate gets fit(train_df, train_leagues): every scored
player-week strictly before it (data.train_before) and every league (virtual and real) of earlier weeks
with outcomes. Then predict(league) for each test league (pre-game fields + week_df, the test week's
pre-game frame) returns {"p_chop": (N,), "total_q": (N, 99) quantiles at metrics.LEVELS, optional
"line_q": (99,) quantiles of the week's minimum total}. Without line_q the chop line is derived from
total_q as if teams were independent (flagged line_src = 1), which ignores shared NFL players/games.
Guards (as in the player track): every (variant, origin) task runs in its own freshly forked process
(maxtasksperchild=1, --jobs 0 = in-process, debugging only), with HARNESS_SEED and numpy's/random's
global RNGs seeded per origin (player.origin_seed: the same for every variant), and fit/predict run
inside data.horizon(origin), where view() refuses any league after the test week and the test week's
outcomes.

    python -m bench.harness.tracks.team --candidate pkg.module:Class --key KEY \
        --variant NAME [--variant NAME2 | --variant NAME3=pkg.mod:Other] --mode smoke|full [--jobs N]
"""
import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import random
import time

from .. import CACHE, rel, slots
from .. import data as D
from .. import metrics as M
from .. import results as R
from .player import factory, load_class, origin_seed

import numpy as np
from common import FLEX_OK

TRACK = "team"
LEVELS = M.LEVELS
POS = ("QB", "RB", "WR", "TE")
ORIGINS = {"full": [(2025, w) for w in range(5, 19)] + [(2026, w) for w in range(1, 8)],
           "smoke": [(2025, 10), (2026, 3), (2026, 7)]}
N_LEAGUES = {"full": 48, "smoke": 16}          # virtual test leagues per week (replicates r < this)
SIZES = tuple(range(10, 18))                   # N = SIZES[r % 8]
GEN = {"version": 1, "seed": 61, "r": 48, "sigma0": 0.30, "tau": 0.50, "sizes": list(SIZES)}
NMAX = 17                                      # widest league (real week 1 and virtual N=17)
BINS = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])   # p_chop reliability bins
NB = len(BINS) - 1
COV = {"cov10": (0, 1, 0.1), "cov50": (0, 5, 0.5), "cov90": (0, 9, 0.9), "cov80": (1, 9, 0.8)}
OUTCOME_ARRAYS = ("n_teams", "chopped", "y_tot", "line_y")   # compare.py: must equal the baseline's
_LEAGUES = None
_LOOK = {}


def _full():
    """The unmasked shared table: league building/lookups must not depend on an active horizon()."""
    D._load()
    return D._TABLE


def slot_list():
    return list(D.league().slots)


def origins(mode):
    return list(ORIGINS[mode])


# ---------------------------------------------------------------- virtual leagues
def _draft(pos, lp, slots_, N, rng, sigma0, tau):
    """Randomized snake draft of N teams into slots_: (N, len(slots_)) pool indices (-1 = empty).

    Team k ranks players by its own board lp + sigma_k * eps (sigma_k = sigma0 * exp(tau * z_k), so
    some drafters are sharper than others), takes the best one that fits an open slot and puts him in
    the most restrictive open slot he fits (fixed position before flex)."""
    n = len(lp)
    sig = sigma0 * np.exp(tau * rng.standard_normal(N))
    board = lp[None, :] + sig[:, None] * rng.standard_normal((N, n))
    order = rng.permutation(N)
    S = len(slots_)
    elig = [np.isin(pos, [s] if s in POS else list(FLEX_OK.get(s, POS))) for s in slots_]
    pref = sorted(range(S), key=lambda i: (slots_[i] not in POS, len(FLEX_OK.get(slots_[i], POS)), i))
    open_ = np.ones((N, S), bool)
    taken = np.zeros(n, bool)
    out = np.full((N, S), -1, np.int64)
    for rd in range(S):
        for k in (order if rd % 2 == 0 else order[::-1]):
            ok = np.zeros(n, bool)
            for i in np.flatnonzero(open_[k]):
                ok |= elig[i]
            ok &= ~taken
            if not ok.any():
                continue
            j = int(np.argmax(np.where(ok, board[k], -np.inf)))
            taken[j] = True
            s = next(i for i in pref if open_[k, i] and elig[i][j])
            open_[k, s] = False
            out[k, s] = j
    return out


def _pool(t):
    """Draft pool of week t: (pid, pos, proj) of every QB/RB/WR/TE projected > 0, in table (pid) order."""
    d = _full()
    m = (d["t"].to_numpy() == t) & (d["proj"].to_numpy() > 0) & d["pos"].isin(POS).to_numpy()
    w = d[m]
    return w["pid"].to_numpy(str), w["pos"].to_numpy(str), w["proj"].to_numpy(float)


def _virtual(season, week, t, pool, slots_):
    pid, pos, proj = pool
    lp = np.log(proj)
    out = []
    for r in range(GEN["r"]):
        N = GEN["sizes"][r % len(GEN["sizes"])]
        rng = np.random.default_rng([GEN["seed"], season, week, r])
        ix = _draft(pos, lp, slots_, N, rng, GEN["sigma0"], GEN["tau"])
        out.append({"league_id": f"v-{season}-{week:02d}-{r:02d}", "kind": "virtual", "season": season,
                    "week": week, "t": t, "r": r, "n": N, "team_ids": list(range(N)),
                    "starters": [[str(pid[j]) if j >= 0 else "" for j in row] for row in ix]})
    return out


# ---------------------------------------------------------------- real leagues (2026)
def _real(slots_):
    """This league's scored weeks: alive teams (roster had players) by roster id, actual starters."""
    lg = D.league()
    out = []
    for w, ms in sorted(lg.matchups.items()):
        if w >= lg.week or (lg.season, w) not in D.T:
            continue
        alive = sorted((m for m in ms if m.get("players")), key=lambda m: m["roster_id"])
        st = []
        for m in alive:
            s = ["" if p in ("0", 0, None) else str(p) for p in (m.get("starters") or [])]
            st.append((s + [""] * len(slots_))[:len(slots_)])
        out.append({"league_id": f"r-{lg.season}-{w:02d}", "kind": "real", "season": lg.season, "week": w,
                    "t": D.t_of(lg.season, w), "r": -1, "n": len(alive),
                    "team_ids": [int(m["roster_id"]) for m in alive],
                    "owners": [lg.owner(m["roster_id"]) for m in alive], "starters": st,
                    "roster": [[str(p) for p in m["players"]] for m in alive]})
    return out


def _signature(pools, reals, slots_):
    h = hashlib.sha1(json.dumps({"gen": GEN, "slots": slots_}, sort_keys=True).encode())
    h.update(inspect.getsource(_draft).encode())
    h.update(inspect.getsource(_virtual).encode())
    for t in sorted(pools):
        pid, pos, proj = pools[t]
        h.update(f"{t}:{'|'.join(pid)}:{'|'.join(pos)}:".encode())
        h.update(np.round(proj, 9).tobytes())
    h.update(json.dumps([[L["league_id"], L["team_ids"], L["starters"]] for L in reals]).encode())
    return h.hexdigest()[:16]


def _scored_weeks():
    d = _full()
    f = d.groupby("t")["final"].all()
    return [int(t) for t in f.index[f.to_numpy(bool)]]


def leagues():
    """{"sig", "gen", "leagues": [...]} for every scored week (virtual) + this league's real weeks.

    Cached in bench/cache/team_leagues_<sig>.json; the signature covers GEN, the generator code, every
    week's draft pool (pid, pos, proj) and the real starters, so a change rebuilds (and invalidates old
    results: rerun the baseline when "leagues_sig" changes)."""
    global _LEAGUES
    if _LEAGUES is not None:
        return _LEAGUES
    slots_ = slot_list()
    pools = {t: _pool(t) for t in _scored_weeks()}
    reals = _real(slots_)
    sig = _signature(pools, reals, slots_)
    path = os.path.join(CACHE, f"team_leagues_{sig}.json")
    if os.path.exists(path):
        with open(path) as f:
            _LEAGUES = json.load(f)
    else:
        lv = []
        for t in sorted(pools):
            s, w = D.week_of(t)
            lv += _virtual(s, w, t, pools[t], slots_)
            lv += [L for L in reals if L["t"] == t]
        _LEAGUES = {"sig": sig, "gen": GEN, "slots": slots_, "leagues": lv}
        os.makedirs(CACHE, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(_LEAGUES, f, separators=(",", ":"))
        os.replace(tmp, path)
        for f in os.listdir(CACHE):              # drop leagues of older data / generator versions
            if f.startswith("team_leagues_") and f.endswith(".json") and f != os.path.basename(path):
                try:
                    os.remove(os.path.join(CACHE, f))
                except OSError:
                    pass
    _LEAGUES["by_id"] = {L["league_id"]: L for L in _LEAGUES["leagues"]}
    _check_real(_LEAGUES)
    return _LEAGUES


def _check_real(LG):
    """Real totals from table actuals equal Sleeper's points; chopped owner equals data/backtest.json."""
    bt = os.path.join(D.DATA, "backtest.json")
    ref = {r["week"]: r["chopped"] for r in json.load(open(bt))["weeks"]} if os.path.exists(bt) else {}
    lg = D.league()
    for L in LG["leagues"]:
        if L["kind"] != "real":
            continue
        v = _view(L, outcome=True)
        pts = {m["roster_id"]: float(m.get("points") or 0.0) for m in lg.matchups[L["week"]]}
        dp = np.abs(v["points"] - np.array([pts[r] for r in L["team_ids"]])).max()
        assert dp < 1e-6, f"{L['league_id']}: starter actuals differ from Sleeper points by {dp}"
        if L["week"] in ref:
            got = L["owners"][v["chopped"]]
            assert got == ref[L["week"]], f"{L['league_id']}: chopped {got} but backtest.json says {ref[L['week']]}"


# ---------------------------------------------------------------- league views
def _lookup(t):
    """pid -> (proj, actual, pos) for week t (actual NaN if not final)."""
    if t not in _LOOK:
        d = _full()
        w = d[d["t"].to_numpy() == t]
        _LOOK[t] = (dict(zip(w["pid"], w["proj"].astype(float))), dict(zip(w["pid"], w["actual"].astype(float))),
                    dict(zip(w["pid"], w["pos"])))
    return _LOOK[t]


def view(L, outcome=False):
    """What a candidate sees of a league: fresh numpy arrays (safe to mutate).

    league_id, kind, season, week, t, n, slots, team_ids (N,), starters (N, 8) pid str ('' = empty
    slot), starter_pos (N, 8), starter_proj (N, 8) league-scoring projection (0 if none), roster (list
    of N pid lists: real = the week's full roster, bench included; virtual = the starters), owners (real).
    outcome=True (training leagues only) adds starter_pts (N, 8), points (N,), chopped, line.
    Inside data.horizon(origin) a league after origin, or origin's outcomes, raise RuntimeError."""
    h = D._HORIZON
    if h is not None and (L["t"] > h or (outcome and L["t"] >= h)):
        raise RuntimeError(f"{L['league_id']}: {'outcomes' if L['t'] == h else 'a later week'} inside horizon(t={h})")
    return _view(L, outcome)


def _view(L, outcome=False):
    proj, act, pos = _lookup(L["t"])
    st = L["starters"]
    v = {"league_id": L["league_id"], "kind": L["kind"], "season": L["season"], "week": L["week"], "t": L["t"],
         "n": L["n"], "slots": list(_LEAGUES["slots"] if _LEAGUES else slot_list()),
         "team_ids": np.array(L["team_ids"], np.int64), "starters": np.array(st, dtype=str),
         "starter_pos": np.array([[pos.get(p, "") if p else "" for p in row] for row in st], dtype=str),
         "starter_proj": np.array([[proj.get(p, 0.0) if p else 0.0 for p in row] for row in st], float),
         "roster": [list(x) for x in L["roster"]] if "roster" in L else [[p for p in row if p] for row in st]}
    if "owners" in L:
        v["owners"] = list(L["owners"])
    if outcome:
        pts = np.array([[act.get(p, 0.0) if p else 0.0 for p in row] for row in st], float)
        assert np.isfinite(pts).all(), f"{L['league_id']}: week not scored"
        tot = pts.sum(1)
        v.update(starter_pts=pts, points=tot, chopped=int(np.argmin(tot)), line=float(tot.min()))
    return v


def test_ids(origin, mode):
    """Harness league ids of one test week, in harness order: virtual r < N_LEAGUES[mode], then real."""
    by = leagues()["by_id"]
    s, w = origin
    ids = [f"v-{s}-{w:02d}-{r:02d}" for r in range(N_LEAGUES[mode])]
    real = f"r-{s}-{w:02d}"
    ids += [real] if real in by else []
    missing = [i for i in ids if i not in by]
    assert not missing, f"leagues missing from the cache: {missing[:3]}"
    return ids


def train_leagues(origin):
    """Every league (virtual, all replicates, and real) of the weeks strictly before origin, with outcomes."""
    t0 = D.t_of(origin)
    return [view(L, outcome=True) for L in leagues()["leagues"] if L["t"] < t0]


def test_leagues(origin, mode):
    """The test week's leagues, pre-game only, each with week_df = the week's pre-game frame (shared)."""
    wk = D.week_frame(*origin, items_only=False, pre=True).reset_index(drop=True)
    by = leagues()["by_id"]
    out = []
    for lid in test_ids(origin, mode):
        v = view(by[lid])
        v["week_df"] = wk
        out.append(v)
    return out


def truth(mode):
    """item_id, block, kind, n_teams, chopped, y_tot (n, NMAX) NaN-padded team totals, line_y."""
    by = leagues()["by_id"]
    ids, blk, kind, nt, ch, Y, ly = [], [], [], [], [], [], []
    for o in origins(mode):
        for lid in test_ids(o, mode):
            v = _view(by[lid], outcome=True)
            ids.append(lid)
            blk.append(v["t"])
            kind.append(v["kind"])
            nt.append(v["n"])
            ch.append(v["chopped"])
            y = np.full(NMAX, np.nan)
            y[:v["n"]] = v["points"]
            Y.append(y)
            ly.append(v["line"])
    return {"item_id": np.array(ids, dtype=str), "block": np.array(blk, np.int64), "kind": np.array(kind, dtype=str),
            "n_teams": np.array(nt, np.int64), "chopped": np.array(ch, np.int64), "y_tot": np.array(Y),
            "line_y": np.array(ly, float)}


def items(mode="full"):
    return truth(mode)["item_id"]


# ---------------------------------------------------------------- predictions + scoring
def line_from_marginals(q):
    """Quantiles of min_k X_k for independent teams with quantiles q (N, L) at LEVELS (tails extended
    linearly to levels 0 and 1)."""
    q = np.sort(np.asarray(q, float), axis=1)
    lo = q[:, :1] - (q[:, 1:2] - q[:, :1])
    hi = q[:, -1:] + (q[:, -1:] - q[:, -2:-1])
    qq = np.concatenate([lo, q, hi], 1) + np.arange(q.shape[1] + 2) * 1e-9   # strictly increasing for interp
    lv = np.concatenate([[0.0], LEVELS, [1.0]])
    xs = np.unique(qq)
    F = np.stack([np.interp(xs, r, lv) for r in qq])
    Fmin = 1.0 - np.prod(1.0 - F, axis=0)
    return np.interp(LEVELS, np.maximum.accumulate(Fmin), xs)


def check_output(o, L):
    """Validated (p_chop normalized, total_q sorted, line_q sorted, line_src) for one league."""
    N, lid = L["n"], L["league_id"]
    p = np.asarray(o["p_chop"], float).ravel()
    if p.shape != (N,) or not np.isfinite(p).all() or (p < -1e-9).any() or p.sum() <= 0:
        raise ValueError(f"{lid}: p_chop must be ({N},) finite, >= 0, positive sum; got shape {p.shape}")
    p = np.maximum(p, 0.0)
    p = p / p.sum()
    q = np.asarray(o["total_q"], float)
    if q.shape != (N, len(LEVELS)) or not np.isfinite(q).all():
        raise ValueError(f"{lid}: total_q must be finite ({N}, {len(LEVELS)}); got {q.shape}")
    q = np.sort(q, axis=1)
    lq = o.get("line_q")
    if lq is None:
        return p, q, line_from_marginals(q), 1
    lq = np.asarray(lq, float).ravel()
    if lq.shape != (len(LEVELS),) or not np.isfinite(lq).all():
        raise ValueError(f"{lid}: line_q must be finite ({len(LEVELS)},); got {lq.shape}")
    return p, q, np.sort(lq), 0


def score_league(lid, p, q, lq, y):
    """Per-league losses: chop log-loss / Brier / mid-rank of the chopped team, mean team-total CRPS,
    chop-line CRPS, team-total PIT decile counts, p_chop reliability sums."""
    N = len(y)
    c = int(np.argmin(y))
    yc = np.zeros(N)
    yc[c] = 1.0
    rank = 1.0 + (p > p[c]).sum() + 0.5 * ((p == p[c]).sum() - 1)     # mid-rank: uniform p -> (N+1)/2
    pit = M.pit(q, y, M.item_uniform([f"{lid}-{k}" for k in range(N)]))
    b = np.clip(np.searchsorted(BINS, p, side="right") - 1, 0, NB - 1)
    rel_ = np.zeros(3 * NB)
    np.add.at(rel_, b, p)
    np.add.at(rel_, NB + b, yc)
    np.add.at(rel_, 2 * NB + b, 1.0)
    return {"logloss": float(M.logloss(p[None], np.array([c]))[0]), "brier": float(M.brier(p[None], np.array([c]))[0]),
            "rank": rank, "rank_norm": (rank - 1.0) / max(N - 1, 1),
            "crps_total": float(M.crps(q, y).mean()), "crps_line": float(M.crps(lq[None], np.array([y.min()]))[0]),
            "tot_hist": np.bincount(np.minimum((pit * 10).astype(int), 9), minlength=10).astype(float),
            "rel": rel_, "p_chopped": float(p[c])}


def _mask(a, kind):
    return (np.asarray(a["kind"]) == kind).astype(float)


def _gm(name, arr, kind, lower=True):
    return M.group_mean(name, arr, "kind", kind, lower)


def _hist_metrics(pre, kind):
    def stat(a):
        return np.asarray(a["tot_hist"], float) * _mask(a, kind)[:, None]

    def share(S, lo, hi):
        return S[..., lo:hi].sum(-1) / np.maximum(S.sum(-1), 1e-12)

    need = ("tot_hist", "kind")
    ms = [M.Metric(f"{pre}pit_maxdev_total", stat,
                   lambda S: np.abs(S / np.maximum(S.sum(-1, keepdims=True), 1e-12) - 0.1).max(-1), True, need),
          M.Metric(f"{pre}cov_err_total", stat,
                   lambda S: np.mean([np.abs(share(S, lo, hi) - nom) for lo, hi, nom in COV.values()], axis=0), True, need)]
    ms += [M.Metric(f"{pre}{k}_total", stat, lambda S, lo=lo, hi=hi: share(S, lo, hi), None, need)
           for k, (lo, hi, _) in COV.items()]
    return ms


def _ece(name, kind):
    def stat(a):
        return np.asarray(a["rel"], float) * _mask(a, kind)[:, None]

    def final(S):
        return np.abs(S[..., :NB] - S[..., NB:2 * NB]).sum(-1) / np.maximum(S[..., 2 * NB:].sum(-1), 1e-12)
    return M.Metric(name, stat, final, True, ("rel", "kind"))


def _metrics():
    V, Rl = "virtual", "real"
    ms = [_gm("logloss", "logloss", V), _gm("brier", "brier", V), _gm("rank_norm", "rank_norm", V),
          _gm("crps_total", "crps_total", V), _gm("crps_line", "crps_line", V), _ece("ece", V)]
    ms += _hist_metrics("", V)
    ms += [_gm("rank", "rank", V), _gm("p_chopped", "p_chopped", V, lower=False)]
    ms += [_gm("real_logloss", "logloss", Rl), _gm("real_brier", "brier", Rl), _gm("real_rank", "rank", Rl),
           _gm("real_rank_norm", "rank_norm", Rl), _gm("real_crps_total", "crps_total", Rl),
           _gm("real_crps_line", "crps_line", Rl), _ece("real_ece", Rl)]
    return ms


METRICS = _metrics()
PRIMARY = "logloss"


def reliability(rel_sum):
    """Reliability table from summed per-league rel arrays."""
    sp, sy, n = rel_sum[:NB], rel_sum[NB:2 * NB], rel_sum[2 * NB:]
    return [{"bin": [float(BINS[k]), float(BINS[k + 1])], "n": int(n[k]), "mean_p": round(float(sp[k] / n[k]), 4),
             "rate": round(float(sy[k] / n[k]), 4)} for k in range(NB) if n[k] > 0]


# ---------------------------------------------------------------- running
def predict_origin(make, origin, mode):
    """Fresh candidate: fit on everything before origin, predict each test league:
    (league ids, outs, fit_s, pred_s). Views are built first; the candidate is made, fit and run
    inside D.horizon(origin), with HARNESS_SEED and numpy's/random's global RNGs seeded per origin."""
    train = D.train_before(origin)
    tl = train_leagues(origin)
    test = test_leagues(origin, mode)
    ids = [L["league_id"] for L in test]
    seed = origin_seed(origin)
    os.environ["HARNESS_SEED"] = str(seed)
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)
    with D.horizon(origin):
        cand = make()
        t0 = time.time()
        cand.fit(train, tl)
        t1 = time.time()
        outs = [check_output(cand.predict(L), L) for L in test]
        t2 = time.time()
    return ids, outs, t1 - t0, t2 - t1


_W = {}


def _init_worker(threads):
    os.environ["HARNESS_THREADS"] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
        _W["limit"] = threadpool_limits(threads)
    except ImportError:
        pass


def _task(job):
    vi, oi = job
    try:
        return vi, oi, *predict_origin(_W["makers"][vi], _W["origins"][oi], _W["mode"])
    except StopIteration as e:          # would silently end the parent's imap loop
        raise RuntimeError(f"candidate raised StopIteration at {_W['origins'][oi]}") from e


def run(key, variants, mode="smoke", jobs=None, notes="", extra=None):
    """variants: [(name, zero-arg factory returning a fresh candidate)]. Writes results; returns the JSON path.

    Every (variant, origin) task runs in its own freshly forked process (maxtasksperchild=1), so no
    module-level state carries from one origin to another; jobs = parallel processes (default and
    cap CPUQ_SLOTS). jobs=0 runs everything in this process instead (debugging only: no isolation)."""
    t_start = time.time()
    D.table()                                      # build/load the caches before forking
    LG = leagues()
    os_ = origins(mode)
    tr = truth(mode)
    for t in _scored_weeks():                      # shared with forked workers
        _lookup(t)
    makers = [f for _, f in variants]
    jobs_list = [(vi, oi) for vi in range(len(variants)) for oi in range(len(os_))]
    n_jobs = max(1, min(jobs or slots(), slots(), len(jobs_list)))
    per = max(1, slots() // n_jobs)
    out = {}

    def log(vi, oi, r):
        print(f"[team] {variants[vi][0]} {os_[oi]} fit {r[2]:.1f}s pred {r[3]:.1f}s ({len(r[1])} leagues)", flush=True)
    if jobs == 0:
        os.environ.setdefault("HARNESS_THREADS", str(slots()))
        for vi, oi in jobs_list:
            out[vi, oi] = predict_origin(makers[vi], os_[oi], mode)
            log(vi, oi, out[vi, oi])
    else:
        _W.update(makers=makers, origins=os_, mode=mode)
        with mp.get_context("fork").Pool(n_jobs, initializer=_init_worker, initargs=(per,), maxtasksperchild=1) as pool:
            for vi, oi, *res in pool.imap_unordered(_task, jobs_list):
                out[vi, oi] = tuple(res)
                log(vi, oi, out[vi, oi])
    n = len(tr["item_id"])
    recs = []
    for vi, (name, _) in enumerate(variants):
        ids = np.array([i for oi in range(len(os_)) for i in out[vi, oi][0]], dtype=str)
        R.assert_aligned(ids, tr["item_id"], f"{key}/{name} predicted leagues vs truth")
        preds = [x for oi in range(len(os_)) for x in out[vi, oi][1]]
        assert len(preds) == n, f"{name}: {len(preds)} predictions for {n} leagues"
        P = np.full((n, NMAX), np.nan)
        Qt = np.full((n, NMAX, len(LEVELS)), np.nan, np.float32)
        LQ = np.zeros((n, len(LEVELS)), np.float32)
        src = np.zeros(n, np.int8)
        per_item = {k: [] for k in ("logloss", "brier", "rank", "rank_norm", "crps_total", "crps_line",
                                    "tot_hist", "rel", "p_chopped")}
        for i, (p, q, lq, s) in enumerate(preds):
            N = tr["n_teams"][i]
            y = tr["y_tot"][i, :N]
            sc = score_league(tr["item_id"][i], p, q, lq, y)
            for k in per_item:
                per_item[k].append(sc[k])
            P[i, :N], Qt[i, :N], LQ[i], src[i] = p, q, lq, s
        arrays = {"item_id": tr["item_id"], "block": tr["block"], "kind": tr["kind"], "n_teams": tr["n_teams"],
                  "chopped": tr["chopped"], **{k: np.asarray(x, float) for k, x in per_item.items()},
                  "log_n": np.log(tr["n_teams"]), "p": P, "y_tot": tr["y_tot"], "q_tot": Qt, "line_q": LQ,
                  "line_y": tr["line_y"], "line_src": src}
        met = R.evaluate(METRICS, arrays, tr["item_id"], f"{key}/{name}")
        by_week, relt = {}, {}
        for o in os_:
            m = tr["block"] == D.t_of(o)
            row = {}
            for kd in ("virtual", "real"):
                mk = m & (tr["kind"] == kd)
                if mk.any():
                    row[kd] = {"logloss": round(float(arrays["logloss"][mk].mean()), 4),
                               "crps_line": round(float(arrays["crps_line"][mk].mean()), 3), "n": int(mk.sum())}
            by_week[f"{o[0]}-{o[1]:02d}"] = row
        for kd in ("virtual", "real"):
            mk = tr["kind"] == kd
            if mk.any():
                relt[kd] = reliability(arrays["rel"][mk].sum(0))
        by_size = {}
        for N in np.unique(tr["n_teams"][tr["kind"] == "virtual"]):
            mk = (tr["kind"] == "virtual") & (tr["n_teams"] == N)
            by_size[str(N)] = {"logloss": round(float(arrays["logloss"][mk].mean()), 4),
                               "uniform": round(float(np.log(N)), 4), "n": int(mk.sum())}
        recs.append({"name": name, "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items()},
                     "by_week": by_week, "by_size": by_size, "reliability": relt, "line_from_marginals": int(src.sum()),
                     "fit_s": round(sum(out[vi, oi][2] for oi in range(len(os_))), 2),
                     "predict_s": round(sum(out[vi, oi][3] for oi in range(len(os_))), 2), "arrays": arrays})
    isv, isr = tr["kind"] == "virtual", tr["kind"] == "real"
    uniform = {"logloss": float(np.log(tr["n_teams"][isv]).mean()), "brier": float((1 - 1 / tr["n_teams"][isv]).mean()),
               "rank_norm": 0.5}
    if isr.any():
        uniform.update(real_logloss=float(np.log(tr["n_teams"][isr]).mean()),
                       real_brier=float((1 - 1 / tr["n_teams"][isr]).mean()), real_rank_norm=0.5)
    path = R.write(key, TRACK, mode, recs, time.time() - t_start, notes,
                   origins=[f"{s}-{w:02d}" for s, w in os_], levels=len(LEVELS), n_items=int(n),
                   n_virtual=int(isv.sum()), n_real=int(isr.sum()), leagues_sig=LG["sig"], gen=GEN,
                   uniform=uniform, jobs=n_jobs, **(extra or {}))
    bad = R.validate(R.read(path))
    assert not bad, bad
    for r in recs:
        m = r["metrics"]
        print(f"[team] {key}/{r['name']} ({mode}): virtual logloss {m['logloss']:.4f} (uniform {uniform['logloss']:.4f})  "
              f"brier {m['brier']:.4f}  rank_norm {m['rank_norm']:.3f}  crps_total {m['crps_total']:.3f}  "
              f"crps_line {m['crps_line']:.3f}  ece {m['ece']:.4f}  cov10/50/90 {m['cov10_total']:.3f}/"
              f"{m['cov50_total']:.3f}/{m['cov90_total']:.3f}  n {int(isv.sum())}", flush=True)
        if isr.any():
            print(f"[team] {key}/{r['name']} ({mode}): real logloss {m['real_logloss']:.4f} (uniform "
                  f"{uniform['real_logloss']:.4f})  brier {m['real_brier']:.4f}  rank {m['real_rank']:.2f}  "
                  f"crps_total {m['real_crps_total']:.3f}  crps_line {m['real_crps_line']:.3f}  n {int(isr.sum())}", flush=True)
    print(f"[team] wrote {rel(path)}", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="team / chop track backtest")
    ap.add_argument("--candidate", help="module:Class used by every --variant without its own =module:Class")
    ap.add_argument("--key", required=True)
    ap.add_argument("--variant", action="append", default=None, help="NAME or NAME=module:Class (repeatable)")
    ap.add_argument("--mode", choices=R.MODES, default="smoke")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel task processes (default and cap CPUQ_SLOTS); 0 = in-process, debugging only")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)
    variants = []
    for v in args.variant or ["default"]:
        name, _, spec = v.partition("=")
        spec = spec or args.candidate
        if not spec:
            ap.error(f"variant {name}: no candidate class")
        variants.append((name, factory(load_class(spec), name)))
    return run(args.key, variants, args.mode, args.jobs, args.notes)


if __name__ == "__main__":
    main()
