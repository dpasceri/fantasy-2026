"""AVAILABILITY track: P(a fantasy-relevant player plays) this week given his injury designation, and 1-8 weeks ahead.

Data: bench.harness.avail_data (nflverse injury reports, weekly rosters and snap counts, REG 2016-2026, on a
global week axis g). played = offense snaps >= 1. Items, at an origin week t (pre-game: the final injury report
and roster moves of week t are known, games of week t are not):
  kind 0 "gameday": in-scope players listed Questionable or Doubtful at t; target = t.
  kind 1 "ahead":   in-scope players not ruled out at t (no tag, Q or D); target = t + h for h = 1..8 in the same
                    season, on weeks his origin team plays (byes skipped, so h is in calendar weeks like sim.py).
  In scope = QB/RB/WR/TE with mean offense share >= 40% over his last 3 played games, played within 18 weeks,
  on the active roster (or designated to return from IR and listed Q/D), not Out at t. Week-1 origins excluded.
item_id = "<season>-<week:02d>-<gsis_id>-h<h>", block = season * 100 + week of the origin.
Rolling origin: for test origin t a fresh candidate gets
  fit(train, periods): train = every item of every earlier origin whose target week is < t (outcome y known),
                       periods = person-periods (relevant player x team-game week) of weeks < t with outcome y,
  predict_proba(test) -> (len(test),) P(plays): test = the origin's items, pre-game columns only, harness order.
Baseline = the core sim as of the 2026-10-30 bake-off (frozen): 1 - P_OUT (Questionable 0.25, Doubtful 0.8) for gameday items; for ahead items the
exact availability of sim.Universe.paths' flat INJ_HAZARD (2%/week) with geometric absences of mean 3 weeks.

    python -m bench.harness.tracks.availability --candidate pkg.module:Class --key KEY \
        --variant NAME [--variant NAME2 | --variant NAME3=pkg.mod:Other] --mode smoke|full [--jobs N]
"""
import argparse
import multiprocessing as mp
import os
import time

from .. import rel, slots
from .. import avail_data as AD
from .. import metrics as M
from .. import results as R
from .player import factory, load_class

import numpy as np
import pandas as pd

TRACK = "availability"
SMOKE = [(2023, 9), (2025, 6), (2026, 4)]
FULL_SEASONS = range(2021, 2027)
OUTCOME = ("y",)


def origins(mode):
    if mode == "smoke":
        return list(SMOKE)
    it = AD.items()
    o = it[it.season.isin(FULL_SEASONS)][["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    return [tuple(map(int, x)) for x in o.to_numpy()]


def block_of(season, week):
    return int(season) * 100 + int(week)


def _origin_rows(origin):
    it = AD.items()
    g = AD.g_of(origin)
    return it[it.origin.to_numpy() == g]


def test_frame(origin):
    """The origin's items in harness order, pre-game columns only."""
    return _origin_rows(origin).drop(columns=list(OUTCOME)).reset_index(drop=True)


def train_frames(origin):
    """(train items with target < t, periods with g < t): everything a candidate may learn from at origin t."""
    g = AD.g_of(origin)
    it, pe = AD.items(), AD.periods()
    return it[it.target.to_numpy() < g].copy(), pe[pe.g.to_numpy() < g].copy()


def truth(mode):
    parts = [_origin_rows(o) for o in origins(mode)]
    w = pd.concat(parts)
    rep = w.rep.to_numpy()
    h = w.h.to_numpy()
    return {"item_id": w.item_id.to_numpy(str), "block": (w.season.to_numpy() * 100 + w.week.to_numpy()).astype(np.int64),
            "kind": w.kind.to_numpy(np.int64), "h": h.astype(np.int64),
            "grp": np.where(w.kind.to_numpy() == 0, "gameday", "ahead"),
            "sgrp": np.select([rep == 1, rep == 2], ["Q", "D"], "none"),
            "hgrp": np.select([h == 0, h == 1, h <= 4], ["h0", "h1", "h2_4"], "h5_8"),
            "pos": w.pos.to_numpy(str), "starter": w.starter.to_numpy(np.int64), "y": w.y.to_numpy(float)}


def items(mode="full"):
    return truth(mode)["item_id"]


# ---------------------------------------------------------------- metrics
def balanced(name, arr):
    """0.5 x (mean over gameday items + mean over ahead items): neither part swamps the other."""
    def stat(a):
        k = np.asarray(a["kind"])
        x = np.asarray(a[arr], float)
        return np.stack([x * (k == 0), k == 0, x * (k == 1), k == 1], 1).astype(float)

    def final(S):
        return 0.5 * (S[..., 0] / np.maximum(S[..., 1], 1e-12) + S[..., 2] / np.maximum(S[..., 3], 1e-12))
    return M.Metric(name, stat, final, True, (arr, "kind"))


def group_ece(name, gcol, g, bins=10):
    def stat(a):
        p, y = np.asarray(a["p"], float), np.asarray(a["y"], float)
        m = (np.asarray(a[gcol]) == g).astype(float)
        b = np.minimum((p * bins).astype(int), bins - 1)
        S = np.zeros((len(p), 3 * bins))
        r = np.arange(len(p))
        S[r, b], S[r, bins + b], S[r, 2 * bins + b] = p * m, y * m, m
        return S

    def final(S):
        sp, sy, n = S[..., :bins], S[..., bins:2 * bins], S[..., 2 * bins:]
        return np.abs(sp - sy).sum(-1) / np.maximum(n.sum(-1), 1e-12)
    return M.Metric(name, stat, final, True, ("p", "y", gcol))


def _metrics():
    gm = M.group_mean
    ms = [balanced("logloss_bal", "logloss"), balanced("brier_bal", "brier"), M.mean_of("logloss"), M.mean_of("brier"),
          gm("logloss_gameday", "logloss", "grp", "gameday"), gm("brier_gameday", "brier", "grp", "gameday"),
          group_ece("ece_gameday", "grp", "gameday"),
          gm("logloss_ahead", "logloss", "grp", "ahead"), gm("brier_ahead", "brier", "grp", "ahead"),
          group_ece("ece_ahead", "grp", "ahead"),
          gm("logloss_gameday_Q", "logloss", "sgrp_gd", "Q"), gm("logloss_gameday_D", "logloss", "sgrp_gd", "D"),
          gm("logloss_ahead_starter", "logloss", "starter_ah", 1), gm("logloss_ahead_tagged", "logloss", "tag_ah", 1),
          gm("logloss_ahead_h1", "logloss", "hgrp", "h1"), gm("logloss_ahead_h2_4", "logloss", "hgrp", "h2_4"),
          gm("logloss_ahead_h5_8", "logloss", "hgrp", "h5_8")]
    for g in ("gameday", "ahead"):
        ms += [gm(f"mean_p_{g}", "p", "grp", g, lower=None), gm(f"rate_{g}", "y", "grp", g, lower=None)]
    return ms


METRICS = _metrics()
PRIMARY = "logloss_bal"


def item_arrays(p, tr):
    """Per-item arrays for probabilities p on the truth dict tr."""
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    a = M.prob_items(p, tr["y"])
    k = tr["kind"]
    out = {k_: tr[k_] for k_ in ("item_id", "block", "kind", "h", "grp", "sgrp", "hgrp", "pos", "starter")}
    out.update(a)
    out["sgrp_gd"] = np.where(k == 0, tr["sgrp"], "-")
    out["starter_ah"] = np.where(k == 1, tr["starter"], -1)
    out["tag_ah"] = np.where(k == 1, (tr["sgrp"] != "none").astype(int), -1)
    return out


# ---------------------------------------------------------------- baseline: the core sim's availability
def sim_ahead(hmax=AD.H_MAX):
    """P(available) h = 1..hmax weeks after the current week under sim.Universe.paths: out_left starts at 0, each
    week a player with out_left == 0 starts an absence with prob INJ_HAZARD lasting Geometric(1/INJ_MEAN_WEEKS)
    weeks (support 1, 2, ...), available iff out_left == 0 after the draw, then out_left counts down."""
    # frozen at sim.py's constants as of the 2026-10-30 bake-off (sim.py has since been refit), so the
    # track's reference and its leaderboards stay reproducible
    INJ_HAZARD, INJ_MEAN_WEEKS = 0.02, 3.0
    K = 400
    q = 1.0 / INJ_MEAN_WEEKS
    geo = q * (1 - q) ** np.arange(K)            # P(length = 1 + j)
    st = np.zeros(K + 1)
    st[0] = 1.0
    out = [np.nan]
    for _ in range(hmax):
        p0 = st[0]
        out.append(p0 * (1 - INJ_HAZARD))
        nxt = np.zeros_like(st)
        nxt[0] += p0 * (1 - INJ_HAZARD)
        nxt[:K] += p0 * INJ_HAZARD * geo          # length 1 + j -> out_left j after the countdown
        nxt[:-1] += st[1:]                        # already out: count down
        st = nxt
    return np.array(out)


class Baseline:
    """The core pipeline's availability (common.P_OUT + sim's flat hazard), used as the track reference."""

    def fit(self, train, periods):
        return self

    def predict_proba(self, test):
        P_OUT = {"Questionable": 0.25, "Doubtful": 0.8}  # common.P_OUT as of the bake-off (frozen, see sim_ahead)
        rep = test["rep"].to_numpy()
        p_out = np.select([rep == 1, rep == 2], [P_OUT["Questionable"], P_OUT["Doubtful"]], 0.0)
        ahead = sim_ahead()[np.clip(test["h"].to_numpy(), 1, AD.H_MAX)]
        return np.where(test["kind"].to_numpy() == 0, 1.0 - p_out, ahead)


# ---------------------------------------------------------------- runner
def predict_origin(make, origin):
    cand = make()
    train, periods = train_frames(origin)
    test = test_frame(origin)
    t0 = time.time()
    cand.fit(train, periods)
    t1 = time.time()
    p = np.asarray(cand.predict_proba(test.copy()), float).reshape(-1)
    t2 = time.time()
    if p.shape != (len(test),):
        raise ValueError(f"predict_proba returned {p.shape}, expected {(len(test),)} at {origin}")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"probabilities outside [0, 1] at {origin}")
    return p, t1 - t0, t2 - t1


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
    return vi, oi, *predict_origin(_W["makers"][vi], _W["origins"][oi])


def run(key, variants, mode="smoke", jobs=None, notes="", extra=None):
    """variants: [(name, zero-arg factory)]. Writes results; returns the JSON path."""
    t_start = time.time()
    AD.grid()                                    # build/load the cache before forking
    os_ = origins(mode)
    tr = truth(mode)
    makers = [f for _, f in variants]
    jobs_list = [(vi, oi) for vi in range(len(variants)) for oi in range(len(os_))]
    n_jobs = max(1, min(jobs or slots(), len(jobs_list)))
    per = max(1, slots() // n_jobs)
    out = {}
    if n_jobs == 1:
        os.environ.setdefault("HARNESS_THREADS", str(slots()))
        for vi, oi in jobs_list:
            out[vi, oi] = predict_origin(makers[vi], os_[oi])
    else:
        _W.update(makers=makers, origins=os_)
        with mp.get_context("fork").Pool(n_jobs, initializer=_init_worker, initargs=(per,)) as pool:
            for vi, oi, p, fs, ps in pool.imap_unordered(_task, jobs_list):
                out[vi, oi] = (p, fs, ps)
    recs = []
    for vi, (name, _) in enumerate(variants):
        p = np.concatenate([out[vi, oi][0] for oi in range(len(os_))])
        a = item_arrays(p, tr)
        arrays = {**a, "p": np.asarray(a["p"], float), "y": tr["y"]}
        met = R.evaluate(METRICS, arrays, tr["item_id"], f"{key}/{name}")
        by_week = {}
        for o in os_:
            m = tr["block"] == block_of(*o)
            g0, g1 = m & (tr["kind"] == 0), m & (tr["kind"] == 1)
            by_week[f"{o[0]}-{o[1]:02d}"] = {
                "logloss_gameday": round(float(a["logloss"][g0].mean()), 4) if g0.any() else None,
                "logloss_ahead": round(float(a["logloss"][g1].mean()), 4) if g1.any() else None,
                "n_gameday": int(g0.sum()), "n_ahead": int(g1.sum())}
        recs.append({"name": name, "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items()},
                     "by_week": by_week, "fit_s": round(sum(out[vi, oi][1] for oi in range(len(os_))), 2),
                     "predict_s": round(sum(out[vi, oi][2] for oi in range(len(os_))), 2), "arrays": arrays})
    path = R.write(key, TRACK, mode, recs, time.time() - t_start, notes,
                   origins=[f"{s}-{w:02d}" for s, w in os_], n_items=int(len(tr["item_id"])),
                   n_gameday=int((tr["kind"] == 0).sum()), n_ahead=int((tr["kind"] == 1).sum()), jobs=n_jobs,
                   **(extra or {}))
    bad = R.validate(R.read(path))
    assert not bad, bad
    for r in recs:
        m = r["metrics"]
        print(f"[availability] {key}/{r['name']} ({mode}): logloss_bal {m['logloss_bal']:.4f}  "
              f"gameday ll {m['logloss_gameday']:.4f} brier {m['brier_gameday']:.4f}  "
              f"ahead ll {m['logloss_ahead']:.4f} brier {m['brier_ahead']:.4f}  n {m['n']}", flush=True)
    print(f"[availability] wrote {rel(path)}", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="availability track backtest")
    ap.add_argument("--candidate", help="module:Class used by every --variant without its own =module:Class")
    ap.add_argument("--key", required=True)
    ap.add_argument("--variant", action="append", default=None, help="NAME or NAME=module:Class (repeatable)")
    ap.add_argument("--mode", choices=R.MODES, default="smoke")
    ap.add_argument("--jobs", type=int, default=None, help="parallel origin workers (default CPUQ_SLOTS)")
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
