"""BIDS track: rival FAAB bid models (who bids on whom, how much, and what the top bid clears at).

Chances: one row per (waiver run, alive team with budget > 0, offered player it does not hold), with
rosters and budgets as they stood just before the run (rosters_at: today's rosters with every later
completed transaction rolled back). auction.bid_history uses the end-of-week matchup rosters instead,
which keep players dropped after their game and, in legs 5-7, already hold some of the next post-chop
run's adds, so it drops 12 claims and feeds the run's own outcome into gain. Its rows and feature stay
available for the core-faithful baseline (train-only columns repaired, gain_core).
Positive bids are the outcome; a $0 claim counts as no bid, as in BidModel.
Groups: one per (run, offered player). Its top bid (max over the group's chances, 0 = nobody bid) is
the winning price of a sealed first-price auction; contested = top > 0.
Rolling origin: test legs 3..7 (smoke: 5 and 7). A fresh candidate is fit on every chance from
strictly earlier legs (outcomes included), then predicts the test leg one waiver run at a time, each
call seeing only what was known at that run (guard + data.horizon). Items = the test legs' chances then
their groups (kind "chance" / "price"); block = leg.
Frame columns: item_id, leg, run, rid, pid, name, pos, budget, gain (lineup gain on the point-in-time
roster, auction.gain), proj0, proj_h (player proj at w0 / mean over the gain horizon), pool (player
from the chopped roster, known before the run), post_chop, w0, info_week (last week final at the run),
n_teams, faab_left; train frames add bid (0 = none), claim, won, repaired, gain_core.
LEAKS: gain / proj use final pre-game projections of weeks after the run (look-ahead, unfixable from
the cache); non-pool players are offered only because someone claimed them in that run.

    python -m bench.harness.tracks.bids --candidate pkg.module:Class --key KEY \
        --variant NAME [--variant NAME2 | --variant NAME3=pkg.mod:Other] --mode smoke|full [--jobs N]

Candidate: Class(variant=NAME) (or Class()), then fit(train_df) once per test leg and predict(test_df)
once per waiver run of that leg -> dict:
    p_bid (n,)            P(bid > 0)
    one of size_pmf (n, K) P(bid = k | bid > 0) for k = 0..K-1 dollars (column 0 ignored),
           size_q (n, 99) quantiles of the bid given bid > 0 at metrics.LEVELS,
           size_samples (n, S) draws of the bid given bid > 0.
Optional sample_joint(test_df, rng, n) -> (len(test_df), n) whole-dollar bids, 0 = no bid,
column s one joint draw over the run's chances; then the price distribution is the per-group max of
those draws instead of the exact independent product of the marginals.
The harness caps bid mass above the budget onto the budget and scores on dollars 0..KMAX.
"""
import argparse
import contextlib
import multiprocessing as mp
import os
import random
import time

from .. import rel, slots
from .. import data as D
from .. import metrics as M
from .. import results as R
from .player import factory, load_class

import numpy as np
import pandas as pd
from auction import HORIZON, bid_history, gain
from common import POS

TRACK = "bids"
LEVELS = M.LEVELS
ORIGINS = {"full": [3, 4, 5, 6, 7], "smoke": [5, 7]}
EPS = M.EPS
MIX = 0.01          # uniform mixed into pmfs built from size_q / size_samples (keeps log scores finite)
DRAWS = 4000        # joint draws per waiver run when the candidate has sample_joint
OUTCOME = ("bid", "claim", "won")
TRAIN_ONLY = OUTCOME + ("repaired", "gain_core")   # a repaired row exists because of a claim; gain_core sees outcomes
INTERNAL = ("raw_index",)
_CH = None
_TX = None
_ROS = {}
_GUARD = {}         # while a candidate runs: {"leg": test leg, "t_max": latest run time it may see}


class LeakError(RuntimeError):
    """A candidate asked for information from after the waiver run it is predicting."""


# ---------------------------------------------------------------- point-in-time league state
def _txlog():
    """Every completed transaction of every cached leg, oldest first."""
    global _TX
    if _TX is None:
        lg = D.league()
        _TX = sorted((t for leg in sorted(lg.tx) for t in lg.tx[leg] if t["status"] == "complete"),
                     key=lambda t: t["status_updated"])
    return _TX


def _rosters_at(T):
    lg = D.league()
    r = {rid: set(x.get("players") or []) for rid, x in lg.rosters.items()}
    for t in reversed(_txlog()):
        if t["status_updated"] < T:
            break
        for p, rid in (t.get("adds") or {}).items():
            r[rid].discard(p)
        for p, rid in (t.get("drops") or {}).items():
            r[rid].add(p)
    return {rid: tuple(sorted(s)) for rid, s in r.items()}


def rosters_at(T):
    """{roster_id: sorted pid tuple} just before time T (ms): data/rosters.json (today) with every completed
    transaction at or after T rolled back (adds removed, drops restored, so a chop restores the chopped
    roster). It equals the end-of-week matchup rosters wherever Sleeper's snapshot is clean (legs 3-4).
    Inside fit/predict, T may not exceed the waiver run being predicted (fit: the test leg's first run)."""
    t_max = _GUARD.get("t_max")
    if t_max is not None and T > t_max:
        raise LeakError(f"rosters_at({T}) is after the run being predicted ({t_max})")
    if T not in _ROS:
        _ROS[T] = _rosters_at(T)
    return dict(_ROS[T])


def _budgets(run):
    """Each roster's FAAB just before a waiver run (bid_history's rule over every leg's log: completed
    claims and traded FAAB before the run)."""
    lg = D.league()
    spent = {}
    for t in _txlog():
        if t["status_updated"] >= run:
            break
        if t["type"] == "waiver":
            r = t["roster_ids"][0]
            spent[r] = spent.get(r, 0) + t["settings"]["waiver_bid"]
        elif t["type"] == "trade":
            for wb in t.get("waiver_budget") or []:
                spent[wb["sender"]] = spent.get(wb["sender"], 0) + wb["amount"]
                spent[wb["receiver"]] = spent.get(wb["receiver"], 0) - wb["amount"]
    return {rid: lg.budget - spent.get(rid, 0) for rid in lg.rosters}


# ---------------------------------------------------------------- data
def _build():
    lg = D.league()
    core = bid_history(lg)                              # BidModel's rows: end-of-week matchup rosters
    ck = {(r["run"], r["rid"], r["pid"]): (i, r) for i, r in enumerate(core)}
    chops = {leg: next((t for t in lg.tx[leg] if t["type"] == "chopped"), None) for leg in lg.tx}
    rows = []
    for leg in range(1, lg.leg + 1):
        tx, chop = lg.tx[leg], chops.get(leg)
        for run in sorted({t["status_updated"] for t in tx if t["type"] == "waiver"}):
            claims = [t for t in tx if t["type"] == "waiver" and t["status_updated"] == run]
            post = chop is not None and run > chop["status_updated"]
            w0 = leg + 1 if post else leg
            offered = {list(t["adds"])[0] for t in claims}
            if post and chop.get("drops"):
                offered |= set(chop["drops"])
            offered = sorted(p for p in offered if lg.pos(p) in POS)
            bids, won = {}, set()
            for t in claims:
                k = (t["roster_ids"][0], list(t["adds"])[0])
                bids[k] = max(bids.get(k, 0), t["settings"]["waiver_bid"])
                if t["status"] == "complete":
                    won.add(k)
            ros, bud = _rosters_at(run), _budgets(run)
            if post:
                assert not ros[chop["roster_ids"][0]], f"chopped roster not empty at run {run}"
            for rid in sorted(ros):
                roster = list(ros[rid])
                if not roster:                          # chopped earlier (or this leg, for a post-chop run)
                    continue
                cache = {}
                for pid in offered:
                    if pid in ros[rid]:
                        continue
                    i, cr = ck.get((run, rid, pid), (-1, None))
                    if cr is not None:
                        assert cr["budget"] == bud[rid], (run, rid, cr["budget"], bud[rid])
                    rows.append({"leg": leg, "run": run, "rid": rid, "pid": pid, "budget": bud[rid],
                                 "gain": gain(lg, roster, pid, w0, w0 + HORIZON - 1, cache),
                                 "gain_core": cr["gain"] if cr is not None else np.nan, "raw_index": i,
                                 "bid": bids.get((rid, pid)), "won": (rid, pid) in won})
    d = pd.DataFrame(rows)
    have = set(zip(d["run"], d["rid"], d["pid"]))
    lost = [k for k, (_, r) in ck.items() if r["budget"] > 0 and k not in have]
    assert not lost, f"bid_history rows with no point-in-time chance: {lost[:5]}"
    d["repaired"] = d["raw_index"] < 0                 # not one of bid_history's rows
    d["claim"] = d["bid"].notna()
    d["bid"] = d["bid"].fillna(0).astype(int)
    d["won"] = d["won"].astype(bool)
    chop_t = d["leg"].map(lambda l: chops[l]["status_updated"] if chops.get(l) else np.inf)
    d["post_chop"] = d["run"].to_numpy(float) > chop_t.to_numpy(float)
    d["w0"] = d["leg"] + d["post_chop"].astype(int)
    d["info_week"] = d["w0"] - 1                       # last week whose games were final at the run
    pool = {(leg, p) for leg, c in chops.items() if c and c.get("drops") for p in c["drops"]}
    d["pool"] = [bool(pc) and (l, p) in pool for l, p, pc in zip(d["leg"], d["pid"], d["post_chop"])]
    d["pos"] = d["pid"].map(lg.pos)
    d["name"] = d["pid"].map(lambda p: lg.info.get(p, {}).get("name", p))
    d["proj0"] = [lg.proj_pts(p, w) for p, w in zip(d["pid"], d["w0"])]
    d["proj_h"] = [np.mean([lg.proj_pts(p, w) for w in range(w, min(w + HORIZON - 1, lg.final_week) + 1)])
                   for p, w in zip(d["pid"], d["w0"])]
    d = d[d["budget"] > 0].copy()                      # BidModel's rule: a $0 budget can't bid
    claims = {(t["status_updated"], t["roster_ids"][0], list(t["adds"])[0]) for leg in lg.tx for t in lg.tx[leg]
              if t["type"] == "waiver" and lg.pos(list(t["adds"])[0]) in POS}
    have = set(zip(d["run"], d["rid"], d["pid"]))
    miss = [k for k in claims if k not in have and _budgets(k[0])[k[1]] > 0]
    assert not miss, f"claims with no chance row: {miss[:5]}"
    assert (d["bid"] <= d["budget"]).all()
    g = d.groupby("run")
    d["n_teams"] = g["rid"].transform("nunique")
    d["faab_left"] = d["run"].map(d.drop_duplicates(["run", "rid"]).groupby("run")["budget"].sum())
    d = d.sort_values(["leg", "run", "pid", "rid"], kind="stable").reset_index(drop=True)
    d.insert(0, "item_id", [f"c{l:02d}-{r}-{p}-{t:02d}" for l, r, p, t in zip(d["leg"], d["run"], d["pid"], d["rid"])])
    cols = ["item_id", "leg", "run", "rid", "pid", "name", "pos", "budget", "gain", "proj0", "proj_h", "pool",
            "post_chop", "w0", "info_week", "n_teams", "faab_left", "repaired", "gain_core", "raw_index"] + list(OUTCOME)
    return d[cols]


def _all():
    global _CH
    if _CH is None:
        _CH = _build()
    return _CH


def chances():
    """Every chance (all legs with waiver runs), outcomes included. Shared: treat as read-only.
    Inside a candidate's fit/predict it shows only legs before the test leg."""
    d = _all()
    leg = _GUARD.get("leg")
    return d if leg is None else d[d["leg"].to_numpy() < leg].copy()


def kmax():
    """Top of the dollar grid every pmf is scored on: the largest budget of any chance."""
    return int(_all()["budget"].max())


def groups(d=None):
    """One row per (run, offered player): leg, top bid, cleared price (-1 = no claim completed), contested."""
    d = chances() if d is None else d
    lg = D.league()
    g = d.groupby(["leg", "run", "pid"], sort=True)
    out = g.agg(pool=("pool", "first"), pos=("pos", "first"), n_teams=("rid", "size"), top=("bid", "max"),
                n_bidders=("bid", lambda b: int((b > 0).sum()))).reset_index()
    cleared = {}
    for leg in range(1, lg.leg + 1):
        for t in lg.tx[leg]:
            if t["type"] == "waiver" and t["status"] == "complete":
                cleared[(t["status_updated"], list(t["adds"])[0])] = t["settings"]["waiver_bid"]
    out["cleared"] = [cleared.get((r, p), -1) for r, p in zip(out["run"], out["pid"])]
    out["contested"] = out["top"] > 0
    out.insert(0, "item_id", [f"p{l:02d}-{r}-{p}" for l, r, p in zip(out["leg"], out["run"], out["pid"])])
    return out


def train_before(leg):
    """Every chance of legs strictly before leg, outcomes, repaired and gain_core included (a copy)."""
    d = chances()
    return d[d["leg"].to_numpy() < leg].drop(columns=list(INTERNAL)).reset_index(drop=True)


def test_frame(leg):
    """The test leg's chances, pre-auction columns only, in harness order (with item_id)."""
    d = _all() if _GUARD.get("leg") is None else chances()
    return d[d["leg"].to_numpy() == leg].drop(columns=list(TRAIN_ONLY + INTERNAL)).reset_index(drop=True)


def origins(mode):
    return list(ORIGINS[mode])


def truth(mode):
    """Harness items of a mode: the test legs' chances then their groups, with outcomes."""
    d = chances()
    c = d[d["leg"].isin(origins(mode))]
    g = groups(c)
    return c.reset_index(drop=True), g


def items(mode="full"):
    c, g = truth(mode)
    return np.concatenate([c["item_id"].to_numpy(str), g["item_id"].to_numpy(str)])


@contextlib.contextmanager
def guard(leg, t_max, week):
    """While a candidate runs: chances() / groups() / truth() show only legs before `leg`, rosters_at()
    refuses times after t_max, and data.table() is data.horizon((season, week))."""
    prev = dict(_GUARD)
    _GUARD.update(leg=leg, t_max=t_max)
    try:
        with D.horizon((D.league().season, week)):
            yield
    finally:
        _GUARD.clear()
        _GUARD.update(prev)


# ---------------------------------------------------------------- distributions
def pmf_from_points(x, budget, K, mix=MIX):
    """(n, S) bid points (quantiles or draws, given bid > 0) -> (n, K+1) pmf over dollars: rounded, clipped
    to 1..budget, histogrammed, mixed with `mix` uniform over 1..budget."""
    x = np.asarray(x, float)
    b = np.asarray(budget, int)
    k = np.clip(np.rint(x), 1, b[:, None]).astype(int)
    P = np.zeros((len(x), K + 1))
    np.add.at(P, (np.repeat(np.arange(len(x)), x.shape[1]), k.ravel()), 1.0 / x.shape[1])
    U = (np.arange(K + 1)[None, :] >= 1) & (np.arange(K + 1)[None, :] <= b[:, None])
    return (1 - mix) * P + mix * U / U.sum(1, keepdims=True)


def size_pmf(out, budget, K):
    """Candidate output -> (n, K+1) pmf of the bid given bid > 0 on dollars 0..K, capped at the budget."""
    n, b = len(budget), np.asarray(budget, int)
    if "size_pmf" in out:
        P = np.asarray(out["size_pmf"], float)
        if P.ndim != 2 or len(P) != n or not np.isfinite(P).all() or (P < 0).any():
            raise ValueError(f"size_pmf must be a finite non-negative (n, K) array, got {P.shape}")
        Q = np.zeros((n, K + 1))
        m = min(P.shape[1], K + 1)
        Q[:, :m] = P[:, :m]
        if P.shape[1] > K + 1:
            Q[:, K] += P[:, K + 1:].sum(1)
    elif "size_q" in out or "size_samples" in out:
        x = np.asarray(out.get("size_q", out.get("size_samples")), float)
        if x.ndim != 2 or len(x) != n or not np.isfinite(x).all():
            raise ValueError(f"size_q / size_samples must be a finite (n, S) array, got {x.shape}")
        if "size_q" in out and x.shape[1] != len(LEVELS):
            raise ValueError(f"size_q needs {len(LEVELS)} columns, got {x.shape[1]}")
        Q = pmf_from_points(x, b, K)
    else:
        raise ValueError("predict() must return size_pmf, size_q or size_samples")
    Q[:, 0] = 0.0
    cols = np.arange(K + 1)[None, :]
    over = cols > b[:, None]
    Q[np.arange(n), b] += np.where(over, Q, 0.0).sum(1)     # mass above the budget piles on the budget
    Q[over] = 0.0
    s = Q.sum(1, keepdims=True)
    if (s <= 0).any():
        raise ValueError("a bid-size pmf has no mass on 1..budget")
    return Q / s


def dcrps(P, y):
    """(n,) exact CRPS of integer pmfs P (n, K+1) on 0..K against integer outcomes y."""
    F = np.cumsum(P, 1)
    I = np.arange(P.shape[1])[None, :] >= np.asarray(y)[:, None]
    return ((F - I) ** 2).sum(1)


def dq(P, tau):
    """Lower tau-quantile per row of integer pmfs: min k with F(k) >= tau."""
    F = np.cumsum(P, 1)
    return (F < tau - 1e-12).sum(1)


def dpit(P, y, u):
    """Randomized PIT of integer outcomes: F(y-1) + u * P(y)."""
    F = np.cumsum(P, 1)
    r = np.arange(len(y))
    lo = np.where(y > 0, F[r, np.maximum(y - 1, 0)], 0.0)
    return lo + u * P[r, y]


def joint_pmf(p, S):
    """(n, K+1) pmf of one chance's bid (0 = no bid): 1 - p at 0, p * size above."""
    H = p[:, None] * S
    H[:, 0] = 1 - p
    return H


def price_pmf(H, gidx, ng):
    """(ng, K+1) pmf of each group's top bid, bids independent across chances: CDF = product of CDFs."""
    L = np.zeros((ng, H.shape[1]))
    np.add.at(L, gidx, np.log(np.clip(np.cumsum(H, 1), 1e-300, 1.0)))
    F = np.minimum(np.exp(L), 1.0)
    F[:, -1] = 1.0
    return np.diff(F, axis=1, prepend=0.0)


def price_pmf_draws(D_, budget, gidx, ng, K):
    """(ng, K+1) empirical pmf of each group's top bid from joint draws (n_chances, S)."""
    D_ = np.clip(np.rint(np.asarray(D_, float)), 0, np.asarray(budget)[:, None]).astype(int)
    T = np.zeros((ng, D_.shape[1]), int)
    np.maximum.at(T, gidx, D_)
    P = np.zeros((ng, K + 1))
    np.add.at(P, (np.repeat(np.arange(ng), T.shape[1]), T.ravel()), 1.0 / T.shape[1])
    return P


# ---------------------------------------------------------------- metrics
def masked(name, arr, *masks, lower=True):
    """Mean of arr over items where every mask array is 1."""
    def stat(a):
        m = np.prod([np.asarray(a[k], float) for k in masks], axis=0)
        return np.stack([np.asarray(a[arr], float) * m, m], 1)
    return M.Metric(name, stat, lambda S: S[..., 0] / np.maximum(S[..., 1], 1e-12), lower, (arr,) + masks)


def masked_abs_err(name, arr, target, *masks):
    def stat(a):
        m = np.prod([np.asarray(a[k], float) for k in masks], axis=0)
        return np.stack([np.asarray(a[arr], float) * m, m], 1)
    return M.Metric(name, stat, lambda S: np.abs(S[..., 0] / np.maximum(S[..., 1], 1e-12) - target), True,
                    (arr,) + masks)


def masked_ece(name, mask, bins=10):
    def stat(a):
        m = np.asarray(a[mask], float)
        p, y = np.asarray(a["p"], float), np.asarray(a["y_bid"], float)
        b = np.minimum((p * bins).astype(int), bins - 1)
        S = np.zeros((len(p), 3 * bins))
        r = np.arange(len(p))
        S[r, b], S[r, bins + b], S[r, 2 * bins + b] = p * m, y * m, m
        return S

    def final(S):
        sp, sy, n = S[..., :bins], S[..., bins:2 * bins], S[..., 2 * bins:]
        return np.abs(sp - sy).sum(-1) / np.maximum(n.sum(-1), 1e-12)
    return M.Metric(name, final=final, stat=stat, lower=True, needs=("p", "y_bid", mask))


def _metrics():
    ms = []
    for sfx, extra in (("", ()), ("_pool", ("m_pool",))):
        ms += [masked("logs" + sfx, "logs", "m_chance", *extra),
               masked("logloss" + sfx, "logloss", "m_chance", *extra),
               masked("size_crps" + sfx, "size_crps", "m_bid", *extra),
               masked("size_logs" + sfx, "size_logs", "m_bid", *extra),
               masked("price_crps" + sfx, "price_crps", "m_contested", *extra),
               masked("price_logs" + sfx, "price_logs", "m_contested", *extra),
               masked_abs_err("price_cov_err" + sfx, "price_cov80", 0.8, "m_contested", *extra),
               masked("price_cov80" + sfx, "price_cov80", "m_contested", *extra, lower=None),
               masked("price_in80" + sfx, "price_in80", "m_contested", *extra, lower=None),
               masked("top_logloss" + sfx, "top_logloss", "m_group", *extra),
               masked("top_crps" + sfx, "top_crps", "m_group", *extra)]
    ms += [masked("brier", "brier", "m_chance"), masked_ece("ece", "m_chance"),
           masked("bid_crps", "bid_crps", "m_chance"), masked("size_pit", "size_pit", "m_bid", lower=None),
           masked("price_pit", "price_pit", "m_contested", lower=None),
           masked("price_width80", "price_width80", "m_contested", lower=None),
           masked("p_mean", "p", "m_chance", lower=None), masked("y_mean", "y_bid", "m_chance", lower=None)]
    return ms


METRICS = _metrics()
PRIMARY = "logs_pool"   # chopped-roster chances: what run.py / dynamic.py price; non-pool rows are selected on a claim


def score(c, g, p, S, P):
    """Per-item arrays over chances c (outcomes, rows aligned with p and S) then groups g (aligned with P)."""
    nc, ng = len(c), len(g)
    ids = np.concatenate([c["item_id"].to_numpy(str), g["item_id"].to_numpy(str)])
    u = M.item_uniform(ids)
    y = np.minimum(c["bid"].to_numpy(int), c["budget"].to_numpy(int))
    yb = (y > 0).astype(float)
    pc = np.clip(p, EPS, 1 - EPS)
    ll = M.logloss(pc, yb)
    r = np.arange(nc)
    ys = np.maximum(y, 1)
    s_logs = -np.log(np.maximum(S[r, ys], EPS))
    s_crps = dcrps(S, ys)
    s_pit = dpit(S, ys, u[:nc])
    H = joint_pmf(pc, S)
    # groups: unconditional top bid, and conditional on top > 0
    top = g["top"].to_numpy(int)
    f0 = np.clip(P[:, 0], 0.0, 1.0)
    pcon = np.clip(1 - f0, EPS, 1 - EPS)
    Cn = P.copy()
    Cn[:, 0] = 0.0
    z = Cn.sum(1)
    Cn[z <= 1e-12, 1] = 1.0                            # model says nobody bids: condition on a $1 floor
    Cn = Cn / Cn.sum(1, keepdims=True)
    tg = np.maximum(top, 1)
    q10, q50, q90 = dq(Cn, 0.1), dq(Cn, 0.5), dq(Cn, 0.9)
    ppit = dpit(Cn, tg, u[nc:])

    def cat(a, b):
        return np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    zc, zg = np.zeros(nc), np.zeros(ng)
    con = g["contested"].to_numpy(float)
    a = {"item_id": ids,
         "block": np.concatenate([c["leg"].to_numpy(np.int64), g["leg"].to_numpy(np.int64)]),
         "kind": np.array(["chance"] * nc + ["price"] * ng),
         "rid": np.concatenate([c["rid"].to_numpy(np.int64), np.full(ng, -1)]),
         "pid": np.concatenate([c["pid"].to_numpy(str), g["pid"].to_numpy(str)]),
         "pos": np.concatenate([c["pos"].to_numpy(str), g["pos"].to_numpy(str)]),
         "m_chance": cat(np.ones(nc), zg), "m_group": cat(zc, np.ones(ng)),
         "m_bid": cat(yb, zg), "m_contested": cat(zc, con),
         "m_pool": cat(c["pool"].to_numpy(float), g["pool"].to_numpy(float)),
         "y": cat(y, top), "y_bid": cat(yb, con), "p": cat(pc, pcon),
         "logloss": cat(ll, M.logloss(pcon, con)), "brier": cat((pc - yb) ** 2, (pcon - con) ** 2),
         "logs": cat(ll + yb * s_logs, zg), "bid_crps": cat(dcrps(H, y), zg),
         "size_crps": cat(yb * s_crps, zg), "size_logs": cat(yb * s_logs, zg), "size_pit": cat(yb * s_pit, zg),
         "size_q50": cat(dq(S, 0.5), zg),
         "top_logloss": cat(zc, M.logloss(pcon, con)), "top_crps": cat(zc, dcrps(P, top)),
         "price_crps": cat(zc, con * dcrps(Cn, tg)),
         "price_logs": cat(zc, con * -np.log(np.maximum(Cn[np.arange(ng), tg], EPS))),
         "price_pit": cat(zc, con * ppit), "price_cov80": cat(zc, con * ((ppit > 0.1) & (ppit <= 0.9))),
         "price_in80": cat(zc, con * ((q10 <= tg) & (tg <= q90))),
         "price_q10": cat(zc, q10), "price_q50": cat(zc, q50), "price_q90": cat(zc, q90),
         "price_width80": cat(zc, con * (q90 - q10)), "price_p0": cat(zc, f0),
         "cleared": cat(zc - 1, g["cleared"].to_numpy(float))}
    return a


# ---------------------------------------------------------------- running
def origin_seed(leg):
    """Deterministic per-leg seed (the same for every variant: common random numbers)."""
    return 104729 * (int(leg) + 1)


def predict_origin(make, leg, draws=DRAWS):
    """Fit a fresh candidate on every chance before leg, then predict the leg one waiver run at a time:
    (group item_ids, p, S, P, fit_s, pred_s). fit runs inside guard(leg, first run, leg) and each predict
    inside guard(leg, run, info_week + 1), with numpy's / random's global RNGs and HARNESS_SEED seeded."""
    train = train_before(leg)
    test = test_frame(leg)
    K = kmax()
    seed = origin_seed(leg)
    os.environ["HARNESS_SEED"] = str(seed)
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)
    runs = sorted(int(r) for r in test["run"].unique())
    n = len(test)
    t0 = time.time()
    with guard(leg, runs[0], leg):
        cand = make()
        cand.fit(train)
    t1 = time.time()
    joint = hasattr(cand, "sample_joint")
    p, S = np.zeros(n), np.zeros((n, K + 1))
    Dr = np.zeros((n, draws)) if joint else None
    for k, run in enumerate(runs):
        ix = np.flatnonzero(test["run"].to_numpy() == run)
        blk = test.iloc[ix].reset_index(drop=True)
        with guard(leg, run, int(blk["info_week"].iloc[0]) + 1):
            out = cand.predict(blk.copy())
            pb = np.asarray(out["p_bid"], float)
            if pb.shape != (len(blk),) or not np.isfinite(pb).all() or (pb < 0).any() or (pb > 1).any():
                raise ValueError(f"p_bid must be (n,) in [0, 1], got {pb.shape} at leg {leg} run {run}")
            p[ix] = pb
            S[ix] = size_pmf(out, blk["budget"].to_numpy(int), K)
            if joint:
                d = np.asarray(cand.sample_joint(blk.copy(), np.random.default_rng([1000 + leg, k]), draws))
                if d.shape != (len(blk), draws) or not np.isfinite(d).all():
                    raise ValueError(f"sample_joint returned {d.shape}, expected {(len(blk), draws)} at leg {leg}")
                Dr[ix] = d
    g = groups(test.assign(bid=0))
    gi = pd.Series(np.arange(len(g)), index=pd.MultiIndex.from_frame(g[["run", "pid"]]))
    gidx = gi.reindex(pd.MultiIndex.from_frame(test[["run", "pid"]])).to_numpy()
    if joint:
        P = price_pmf_draws(Dr, test["budget"].to_numpy(int), gidx, len(g), K)
    else:
        P = price_pmf(joint_pmf(np.clip(p, EPS, 1 - EPS), S), gidx, len(g))
    t2 = time.time()
    return test["item_id"].to_numpy(str), g["item_id"].to_numpy(str), p, S, P, t1 - t0, t2 - t1


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
    return vi, oi, predict_origin(_W["makers"][vi], _W["origins"][oi], _W["draws"])


def run(key, variants, mode="smoke", jobs=None, notes="", extra=None, draws=DRAWS):
    """variants: [(name, zero-arg factory returning a fresh candidate)]. Writes results; returns the JSON path.

    Every (variant, leg) task runs in its own freshly forked process (maxtasksperchild=1), so no
    module-level state carries from one leg to another; jobs = parallel processes (default and cap
    CPUQ_SLOTS). jobs=0 runs everything in this process instead (debugging only: no isolation)."""
    t_start = time.time()
    _all()                                         # build before forking
    D.table()
    os_ = origins(mode)
    c, g = truth(mode)
    makers = [f for _, f in variants]
    jobs_list = [(vi, oi) for vi in range(len(variants)) for oi in range(len(os_))]
    n_jobs = max(1, min(jobs or slots(), slots(), len(jobs_list)))
    per = max(1, slots() // n_jobs)
    out = {}

    def log(vi, oi, r):
        print(f"[bids] {variants[vi][0]} leg {os_[oi]} fit {r[5]:.1f}s pred {r[6]:.1f}s", flush=True)
    if jobs == 0:
        os.environ.setdefault("HARNESS_THREADS", str(slots()))
        for vi, oi in jobs_list:
            out[vi, oi] = predict_origin(makers[vi], os_[oi], draws)
            log(vi, oi, out[vi, oi])
    else:
        _W.update(makers=makers, origins=os_, draws=draws)
        with mp.get_context("fork").Pool(n_jobs, initializer=_init_worker, initargs=(per,), maxtasksperchild=1) as pool:
            for vi, oi, r in pool.imap_unordered(_task, jobs_list):
                out[vi, oi] = r
                log(vi, oi, r)
    ref = np.concatenate([c["item_id"].to_numpy(str), g["item_id"].to_numpy(str)])
    recs = []
    for vi, (name, _) in enumerate(variants):
        R.assert_aligned(np.concatenate([out[vi, oi][0] for oi in range(len(os_))]), c["item_id"], f"{key}/{name} chances")
        R.assert_aligned(np.concatenate([out[vi, oi][1] for oi in range(len(os_))]), g["item_id"], f"{key}/{name} groups")
        p = np.concatenate([out[vi, oi][2] for oi in range(len(os_))])
        S = np.concatenate([out[vi, oi][3] for oi in range(len(os_))])
        P = np.concatenate([out[vi, oi][4] for oi in range(len(os_))])
        a = score(c, g, p, S, P)
        met = R.evaluate(METRICS, a, ref, f"{key}/{name}")
        by_leg = {}
        for leg in os_:
            m = a["block"] == leg
            sub = {k: x[m] for k, x in a.items()}
            by_leg[str(leg)] = {k: round(M.summarize([x for x in METRICS if x.name == k], sub)[k], 4)
                                for k in ("logs_pool", "logs", "logloss", "size_crps", "price_crps")}
            by_leg[str(leg)] |= {"n_chances": int(sub["m_chance"].sum()), "n_bids": int(sub["m_bid"].sum()),
                                 "n_contested": int(sub["m_contested"].sum())}
        recs.append({"name": name, "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items()},
                     "by_leg": by_leg, "fit_s": round(sum(out[vi, oi][5] for oi in range(len(os_))), 2),
                     "predict_s": round(sum(out[vi, oi][6] for oi in range(len(os_))), 2), "arrays": a})
    path = R.write(key, TRACK, mode, recs, time.time() - t_start, notes,
                   origins=[int(x) for x in os_], n_items=int(len(ref)), n_chances=int(len(c)), n_groups=int(len(g)),
                   n_bids=int((c["bid"] > 0).sum()), n_contested=int(g["contested"].sum()), kmax=kmax(),
                   jobs=n_jobs, **(extra or {}))
    bad = R.validate(R.read(path))
    assert not bad, bad
    for r in recs:
        m = r["metrics"]
        print(f"[bids] {key}/{r['name']} ({mode}): logs_pool {m['logs_pool']:.4f}  logs {m['logs']:.4f}  "
              f"logloss {m['logloss']:.4f}  size_crps {m['size_crps']:.3f}  size_logs {m['size_logs']:.3f}  "
              f"price_crps {m['price_crps']:.3f}  price_cov80 {m['price_cov80']:.3f}  n {m['n']}", flush=True)
    print(f"[bids] wrote {rel(path)}", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="bids track backtest")
    ap.add_argument("--candidate", help="module:Class used by every --variant without its own =module:Class")
    ap.add_argument("--key", required=True)
    ap.add_argument("--variant", action="append", default=None, help="NAME or NAME=module:Class (repeatable)")
    ap.add_argument("--mode", choices=R.MODES, default="smoke")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel task processes (default and cap CPUQ_SLOTS); 0 = in-process, debugging only")
    ap.add_argument("--draws", type=int, default=DRAWS, help="joint draws per waiver run for candidates with sample_joint")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)
    variants = []
    for v in args.variant or ["default"]:
        name, _, spec = v.partition("=")
        spec = spec or args.candidate
        if not spec:
            ap.error(f"variant {name}: no candidate class")
        variants.append((name, factory(load_class(spec), name)))
    return run(args.key, variants, args.mode, args.jobs, args.notes, draws=args.draws)


if __name__ == "__main__":
    # run as the canonical module: a candidate that imports bench.harness.tracks.bids must share its guard
    import importlib
    importlib.import_module("bench.harness.tracks.bids").main()
