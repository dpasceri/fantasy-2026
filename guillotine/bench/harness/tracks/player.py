"""PLAYER track: weekly league-scoring score quantiles for every projected player-week.

Items: player-weeks with proj >= 0.5 and a game id (model.calibration_rows' inclusion, = data item).
Rolling origin: for each test week a fresh candidate is fit on every scored player-week strictly
before it (all players, item flag included, actual columns allowed) and predicts 99 quantiles
(metrics.LEVELS) for the test week's items from pre-game columns only.

    python -m bench.harness.tracks.player --candidate pkg.module:Class --key KEY \
        --variant NAME [--variant NAME2 | --variant NAME3=pkg.mod:Other] --mode smoke|full [--jobs N]

Candidate: Class(variant=NAME) (or Class() if it takes no variant), then
    fit(train_df) and predict_quantiles(test_df, levels) -> (len(test_df), len(levels)).
Rows are sorted to enforce monotone quantiles. Each (variant, origin) task runs in its own forked
process, --jobs at a time (default and cap CPUQ_SLOTS); each may use HARNESS_THREADS threads
(bench.harness.task_threads()) and gets a per-origin seed (bench.harness.task_seed(); numpy's and
random's global RNGs are seeded too). fit/predict run inside data.horizon(origin), so data.table()
shows nothing after the test week's kickoff.
"""
import argparse
import importlib
import importlib.util
import inspect
import multiprocessing as mp
import os
import random
import sys
import time

from .. import rel, slots
from .. import data as D
from .. import metrics as M
from .. import results as R

import numpy as np
import pandas as pd

TRACK = "player"
LEVELS = M.LEVELS
ORIGINS = {"full": [(2025, w) for w in range(5, 19)] + [(2026, w) for w in range(1, 8)],
           "smoke": [(2025, 10), (2026, 3), (2026, 7)]}
METRICS = M.quantile_metrics()
PRIMARY = "crps"


def origins(mode):
    return list(ORIGINS[mode])


def test_frame(origin):
    """The test week's items, pre-game columns only, in harness order (with item_id)."""
    w = D.week_frame(*origin, items_only=True, pre=True).reset_index(drop=True)
    w.insert(0, "item_id", D.item_ids(w))
    return w


def truth(mode):
    """item_id, block (= global week index t), pos, proj, y over every origin of a mode."""
    d = D.table()
    parts = []
    for o in origins(mode):
        m = (d["t"].to_numpy() == D.t_of(o)) & d["item"].to_numpy()
        w = d[m]
        assert w["final"].all(), f"origin {o} has no actuals"
        parts.append(w)
    w = pd.concat(parts)
    return {"item_id": D.item_ids(w), "block": w["t"].to_numpy(np.int64), "pos": w["pos"].to_numpy(str),
            "proj": w["proj"].to_numpy(float), "y": w["actual"].to_numpy(float)}


def items(mode="full"):
    """Harness item ids for a mode (what every candidate must be scored on)."""
    return truth(mode)["item_id"]


# ---------------------------------------------------------------- candidates
def load_class(spec):
    """'pkg.module:Class' or 'path/to/file.py:Class'."""
    mod, _, cls = spec.rpartition(":")
    assert mod and cls, f"candidate spec must be module:Class, got {spec!r}"
    if mod.endswith(".py") or os.sep in mod:
        path = os.path.abspath(mod)
        name = "cand_" + os.path.splitext(os.path.basename(path))[0]
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, cls)


def factory(cls, variant):
    try:
        params = inspect.signature(cls).parameters
        takes = "variant" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        takes = False
    return (lambda: cls(variant=variant)) if takes else (lambda: cls())


def origin_seed(origin):
    """Deterministic per-origin seed (the same for every variant: common random numbers)."""
    return 7919 * (D.t_of(origin) + 1)


def predict_origin(make, origin):
    """Fit a fresh candidate on everything before origin and predict its test week:
    (test item_ids, q, fit_s, pred_s). The candidate runs inside D.horizon(origin), so D.table()
    shows nothing after origin's kickoff, with numpy's/random's global RNGs and HARNESS_SEED seeded."""
    train = D.train_before(origin)
    test = test_frame(origin)
    seed = origin_seed(origin)
    os.environ["HARNESS_SEED"] = str(seed)
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)
    with D.horizon(origin):
        cand = make()
        t0 = time.time()
        cand.fit(train)
        t1 = time.time()
        q = np.asarray(cand.predict_quantiles(test.copy(), LEVELS.copy()), float)
        t2 = time.time()
    if q.shape != (len(test), len(LEVELS)):
        raise ValueError(f"predict_quantiles returned {q.shape}, expected {(len(test), len(LEVELS))} at {origin}")
    if not np.isfinite(q).all():
        raise ValueError(f"non-finite quantiles at {origin}")
    return test["item_id"].to_numpy(str), np.sort(q, axis=1), t1 - t0, t2 - t1


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
    return vi, oi, predict_origin(_W["makers"][vi], _W["origins"][oi])


def run(key, variants, mode="smoke", jobs=None, notes="", extra=None):
    """variants: [(name, zero-arg factory returning a fresh candidate)]. Writes results; returns the JSON path.

    Every (variant, origin) task runs in its own freshly forked process (maxtasksperchild=1), so no
    module-level state carries from one origin to another; jobs = parallel processes (default and
    cap CPUQ_SLOTS). jobs=0 runs everything in this process instead (debugging only: no isolation)."""
    t_start = time.time()
    D.table()                                      # build/load the cache before forking
    os_ = origins(mode)
    tr = truth(mode)
    makers = [f for _, f in variants]
    jobs_list = [(vi, oi) for vi in range(len(variants)) for oi in range(len(os_))]
    n_jobs = max(1, min(jobs or slots(), slots(), len(jobs_list)))
    per = max(1, slots() // n_jobs)
    out = {}
    if jobs == 0:
        os.environ.setdefault("HARNESS_THREADS", str(slots()))
        for vi, oi in jobs_list:
            out[vi, oi] = predict_origin(makers[vi], os_[oi])
            print(f"[player] {variants[vi][0]} {os_[oi]} fit {out[vi, oi][2]:.1f}s pred {out[vi, oi][3]:.1f}s", flush=True)
    else:
        _W.update(makers=makers, origins=os_)
        with mp.get_context("fork").Pool(n_jobs, initializer=_init_worker, initargs=(per,), maxtasksperchild=1) as pool:
            for vi, oi, res in pool.imap_unordered(_task, jobs_list):
                out[vi, oi] = res
                print(f"[player] {variants[vi][0]} {os_[oi]} fit {res[2]:.1f}s pred {res[3]:.1f}s", flush=True)
    recs = []
    for vi, (name, _) in enumerate(variants):          # out[vi, oi] = (ids, q, fit_s, pred_s)
        ids = np.concatenate([out[vi, oi][0] for oi in range(len(os_))])
        R.assert_aligned(ids, tr["item_id"], f"{key}/{name} test frames vs truth")
        q = np.concatenate([out[vi, oi][1] for oi in range(len(os_))])
        a = M.quantile_items(q, tr["y"], tr["item_id"], LEVELS)
        arrays = {"item_id": tr["item_id"], "block": tr["block"], "pos": tr["pos"], **a,
                  "proj": tr["proj"], "y": tr["y"], "q": q.astype(np.float32)}
        met = R.evaluate(METRICS, arrays, tr["item_id"], f"{key}/{name}")
        by_week = {}
        for oi, o in enumerate(os_):
            m = tr["block"] == D.t_of(o)
            by_week[f"{o[0]}-{o[1]:02d}"] = {"crps": round(float(a["crps"][m].mean()), 4), "n": int(m.sum())}
        recs.append({"name": name, "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items()},
                     "by_week": by_week, "fit_s": round(sum(out[vi, oi][2] for oi in range(len(os_))), 2),
                     "predict_s": round(sum(out[vi, oi][3] for oi in range(len(os_))), 2), "arrays": arrays})
    path = R.write(key, TRACK, mode, recs, time.time() - t_start, notes,
                   origins=[f"{s}-{w:02d}" for s, w in os_], levels=len(LEVELS), n_items=int(len(tr["item_id"])),
                   jobs=n_jobs, **(extra or {}))
    bad = R.validate(R.read(path))
    assert not bad, bad
    for r in recs:
        m = r["metrics"]
        print(f"[player] {key}/{r['name']} ({mode}): crps {m['crps']:.4f}  pinball {m['pinball']:.4f}  "
              f"cov10/50/90 {m['cov10']:.3f}/{m['cov50']:.3f}/{m['cov90']:.3f}  cov80 {m['cov80']:.3f}  "
              f"pit_maxdev {m['pit_maxdev']:.4f}  n {m['n']}", flush=True)
    print(f"[player] wrote {rel(path)}", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="player track backtest")
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
