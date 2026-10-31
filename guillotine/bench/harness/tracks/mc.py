"""MC track: estimator efficiency for the sim's two headline numbers, on today's real league state.

Targets (alive teams, current rosters, projection-optimal lineups with two bench subs per slot,
exactly what run.py hands sim.run_season; one probability per alive team):
  week    P(chopped this week)
  season  P(win the league), rosters frozen (sim.run_season's "win")
Reference: one very large plain-MC run per target (sim.py's own sampler, seeds disjoint from every
replication), built once per state by the _baseline_mc job and cached in
bench/cache/mc_ref_<mode>_<sig>.json together with its own standard error.
Items: R independent replications (block = replication). For each target and replication r the
harness forks a fresh worker, builds a fresh candidate and calls
    Class(variant=NAME).estimate(target, n, seed) -> (T,) estimates
charging the CPU seconds of construction + the call. n is the budget in plain-MC paths. Per item:
squared error vs the reference (mean over teams), CPU seconds, and cost = CPU / the median CPU of
plain-MC calls at the same n timed in the same run (calibration calls interleaved with the
candidate's, so background load on the box cancels). log_work = log(MSE x max(cost, 1)): spending
less than the budget earns nothing (MSE x cost only extrapolates for unbiased estimators). A paired
difference d in log_work is an efficiency of exp(-d) relative to plain MC.

    python -m bench.harness.tracks.mc --candidate pkg.module:Class --key KEY \
        --variant NAME [--variant NAME2=pkg.mod:Other] --mode smoke|full [--jobs N] [--make-ref]
"""
import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import threading
import time

from .. import CACHE, ROOT, rel, slots
from .. import data as D
from .. import metrics as M
from .. import results as R
from .player import factory, load_class

import numpy as np
import sim
from common import DATA, best_lineup
from model import PlayerModel
from sim import SeasonPlan, Universe, team_scores

TRACK = "mc"
TARGETS = ("week", "season")
REF_VERSION = 1
CHUNK = sim.CHUNK                                        # 8000 paths per chunk, as sim.run_season
CFG = {  # R replications, n = budget (plain-MC paths) per call, calib = plain-MC timing calls per target,
         # ref = reference size in CHUNKs, batch = chunks per reference task
    "full": {"R": 160, "n": {"week": 40_000, "season": 32_000}, "calib": 40,
             "ref": {"week": 8000, "season": 2000}, "batch": {"week": 25, "season": 25}},
    "smoke": {"R": 8, "n": {"week": 8_000, "season": 8_000}, "calib": 4,
              "ref": {"week": 200, "season": 30}, "batch": {"week": 10, "season": 1}},
}
REF_SEED = {"week": 9_000_001, "season": 9_000_002}     # reference chunk ci uses rng([REF_SEED, ci])
SEED0 = {"week": 100_000, "season": 200_000}            # replication r gets seed SEED0 + r
CALIB_SEED = 300_000                                    # calibration call k (timing only) uses CALIB_SEED + k
COST_FLOOR = 1.0       # log_work / eff_rel charge max(mean cost, 1): an under-budget call is billed the budget
IMPORT_CPU_MAX = 5.0   # CLI refuses a candidate whose module import (untimed, in the parent) burns more CPU


def seed_of(target, r):
    return SEED0[target] + int(r)


# ---------------------------------------------------------------- state + targets
class Target:
    """What estimate() receives. Shared, built once in the parent: treat as read-only.

    name     'week' | 'season'
    U        sim.Universe over weeks lg.week..final (rostered players of alive teams only)
    plan     sim.SeasonPlan (my roster as its single variant)
    idx      {wi: (3, T, 8)} lineup indices of every team: starters, bench sub 1, bench sub 2 (U.n = empty)
    teams    roster ids in column order; owners; me_t (my column); T
    horizon  weeks the target simulates (1 for week, len(U.weeks) for season)
    lg, pm   core League and PlayerModel
    """

    def __init__(self, name, lg, pm, U, plan, teams, me_t):
        self.name, self.lg, self.pm, self.U, self.plan = name, lg, pm, U, plan
        self.teams, self.me_t, self.T = list(teams), me_t, len(teams)
        self.owners = [lg.owner(t) for t in self.teams]
        self.idx = plan.opp_idx
        self.horizon = 1 if name == "week" else len(U.weeks)

    def scores(self, score, avail, wi):
        """(C, T) team totals for one week's draws (sim.team_scores with bench subs)."""
        return team_scores(score, avail, self.idx[wi])


def build_targets(dev=False):
    """{'week': Target, 'season': Target} on today's state.

    dev=True: a nearby practice state for tuning (the alive team with the weakest projected lineup
    this week, me excepted, is removed and the sim starts next week). Never scored by the harness."""
    lg, pm = D.league(), PlayerModel()
    teams = lg.alive()
    weeks = list(range(lg.week, lg.final_week + 1))
    ros = {t: lg.players(t) for t in teams}
    if dev:
        def proj(t):
            lu = best_lineup(lg.slots, [(lg.proj_pts(p, lg.week), p, lg.pos(p)) for p in ros[t]])
            return sum(lg.proj_pts(p, lg.week) for _, p in lu if p)
        out = min((t for t in teams if t != lg.me), key=lambda t: (proj(t), t))
        teams, weeks = [t for t in teams if t != out], weeks[1:]
    assert len(teams) - len(weeks) <= 1, f"{len(teams)} teams but {len(weeks)} weeks: no single winner"
    me = lg.me if lg.me in teams else teams[0]
    U = Universe(lg, pm, [p for t in teams for p in ros[t]], weeks)
    plan = SeasonPlan(U, teams, ros, me, [ros[me]])
    return {name: Target(name, lg, pm, U, plan, teams, teams.index(me)) for name in TARGETS}


_STATE = {}


def state_sig():
    """Content hash of everything the targets depend on: state files, the fitted model, core sim code,
    and this module's own target/reference code (build_targets, Target, the plain-MC counters)."""
    if "sig" in _STATE:
        return _STATE["sig"]
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    h = hashlib.sha1(TRACK.encode())
    files = [os.path.join(DATA, f) for f in ("meta.json", "league.json", "rosters.json", "users.json",
                                             "schedule.json", "model.npz")]
    sd = os.path.join(DATA, str(meta["season"]))
    files += sorted(os.path.join(sd, f) for f in os.listdir(sd) if f.startswith("proj_"))
    files += [os.path.join(ROOT, f) for f in ("common.py", "model.py", "sim.py")]
    for f in files:
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    for fn in (Target, build_targets, week_counts, season_counts, plain, _ref_task):
        h.update(inspect.getsource(fn).encode())
    _STATE["sig"] = h.hexdigest()
    return _STATE["sig"]


def _h(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]


def signature(mode):
    """State + this mode's full config (budgets, replications, seeds, reference). Item ids carry it, so
    results from another state or config never pair up in compare."""
    return _h(state_sig(), REF_VERSION, mode, CFG[mode], REF_SEED, SEED0, CALIB_SEED)


def ref_sig(mode):
    """State + only what the reference depends on (so changing budgets or R keeps the cached reference)."""
    return _h(state_sig(), REF_VERSION, CFG[mode]["ref"], CFG[mode]["batch"], REF_SEED, CHUNK)


def item_ids(mode, sig=None):
    sig = sig or signature(mode)
    return np.array([f"{sig}-r{r:03d}" for r in range(CFG[mode]["R"])], dtype=str)


# ---------------------------------------------------------------- plain MC (baseline + reference)
def week_counts(tg, C, rng):
    """Chop counts (T,) over C paths of this week: lowest team total is chopped (sim's rule)."""
    _, score, avail = next(tg.U.paths(C, rng, 0))
    return np.bincount(tg.scores(score, avail, 0).argmin(-1), minlength=tg.T)


def season_counts(tg, C, rng):
    """Win counts (T,) over C season paths: chop the lowest alive team every week (= sim._season_chunk)."""
    alive = np.ones((C, tg.T), bool)
    rows = np.arange(C)
    for wi, score, avail in tg.U.paths(C, rng, 0):
        if alive[0].sum() <= 1:
            break
        sc = tg.scores(score, avail, wi)
        sc[~alive] = np.inf
        alive[rows, sc.argmin(-1)] = False
    return np.bincount(alive.argmax(-1), minlength=tg.T)


COUNTS = {"week": week_counts, "season": season_counts}


def plain(target, n, seed, chunk=CHUNK):
    """Plain MC: n iid paths in chunks, chunk ci drawn from rng([seed, ci]) like sim.run_season.
    With n a multiple of 8000 the season estimate equals sim.run_season(U, plan, n, seed)['win'][0] exactly."""
    f = COUNTS[target.name]
    cnt = np.zeros(target.T)
    done = ci = 0
    while done < n:
        C = min(chunk, n - done)
        cnt += f(target, C, np.random.default_rng([seed, ci]))
        done += C
        ci += 1
    return cnt / max(n, 1)


# ---------------------------------------------------------------- workers
_W = {}


def _init_worker(threads):
    os.environ["HARNESS_THREADS"] = str(threads)
    sim.THREADS = 1                                  # sim's own pools stay off inside a harness worker
    try:
        from threadpoolctl import threadpool_limits
        _W["limit"] = threadpool_limits(threads)
    except ImportError:
        pass
    for tg in (_W.get("targets") or {}).values():   # untimed warm-up: lazy per-process init stays out of the timings
        plain(tg, 1000, CALIB_SEED - 2)


def _cpu():
    t = os.times()
    return time.process_time() + t.children_user + t.children_system


def _n_children():
    """Live or unreaped child processes of this process (their CPU is not in os.times() yet)."""
    n = 0
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/children") as f:
                n += len(f.read().split())
        except OSError:
            pass
    return n


def _task(job):
    vi, name, r = job                                # vi < 0: a plain-MC calibration call (timing only)
    tg = _W["targets"][name]
    if vi < 0:
        c0 = _cpu()
        plain(tg, _W["n"][name], CALIB_SEED + r)
        return vi, name, r, None, _cpu() - c0, None
    th0 = threading.active_count()
    c0, w0 = _cpu(), time.perf_counter()
    cand = _W["makers"][vi]()                        # construction is charged too
    est = cand.estimate(tg, _W["n"][name], seed_of(name, r))
    cpu, wall = _cpu() - c0, time.perf_counter() - w0
    if _n_children() or threading.active_count() > th0:
        raise RuntimeError(f"{name} rep {r}: estimate() left child processes or threads running, whose CPU "
                           "would go uncharged (calls must be single-threaded; join anything you start)")
    est = np.asarray(est, float)
    if est.shape != (tg.T,):
        raise ValueError(f"estimate returned shape {est.shape}, expected ({tg.T},) for {name} rep {r}")
    if not np.isfinite(est).all():
        raise ValueError(f"non-finite estimate for {name} rep {r}")
    return vi, name, r, est, cpu, wall


def _ref_task(job):
    name, cis = job
    tg = _W["targets"][name]
    f = COUNTS[name]
    c0 = _cpu()
    cnt = sum(f(tg, CHUNK, np.random.default_rng([REF_SEED[name], ci])) for ci in cis)
    return name, cis[0], cnt, _cpu() - c0


def _pool(n_jobs, threads, **kw):
    return mp.get_context("fork").Pool(n_jobs, initializer=_init_worker, initargs=(threads,), **kw)


# ---------------------------------------------------------------- reference
def ref_path(mode):
    return os.path.join(CACHE, f"mc_ref_{mode}_{ref_sig(mode)}.json")


def _save_partial(part, mode, batches, cpu):
    tmp = f"{part}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({"ref_sig": ref_sig(mode), "cpu": cpu,
                   "batches": {n: {str(k): np.asarray(v).astype(int).tolist() for k, v in b.items()}
                               for n, b in batches.items()}}, f)
    os.replace(tmp, part)


def make_reference(mode, targets, jobs=None):
    """Plain MC with CFG[mode]['ref'] chunks per target (= plain(target, N, REF_SEED[target]) bit for bit),
    run in parallel batches; writes the cache JSON (p, binomial SE, batch-means SE) and returns it.
    Finished batches are checkpointed to <ref>.partial.json, so a killed build resumes where it stopped
    (same counts: each batch is keyed by its first chunk)."""
    cfg = CFG[mode]
    os.makedirs(CACHE, exist_ok=True)
    path = ref_path(mode)
    part = path[:-len(".json")] + ".partial.json"
    batches = {name: {} for name in TARGETS}
    cpu = {name: 0.0 for name in TARGETS}
    try:
        with open(part) as f:
            P = json.load(f)
        if P.get("ref_sig") == ref_sig(mode):
            for name in TARGETS:
                batches[name] = {int(k): np.array(v, np.int64) for k, v in P["batches"][name].items()}
                cpu[name] = float(P["cpu"][name])
    except (OSError, ValueError, KeyError):
        pass
    resumed = sum(len(b) for b in batches.values())
    tasks = []
    for name in ("season", "week"):                   # long tasks first
        nc, b = cfg["ref"][name], cfg["batch"][name]
        assert nc % b == 0, (name, nc, b)
        tasks += [(name, list(range(i, i + b))) for i in range(0, nc, b) if i not in batches[name]]
    n_jobs = max(1, min(jobs or slots(), len(tasks)))
    _W["targets"] = targets
    t0 = last = time.time()
    print(f"[mc] reference ({mode}): {len(tasks)} batches on {n_jobs} workers"
          + (f" ({resumed} resumed from {rel(part)})" if resumed else ""), flush=True)
    if tasks:
        with _pool(n_jobs, 1) as pool:
            for k, (name, ci0, cnt, c) in enumerate(pool.imap_unordered(_ref_task, tasks), 1):
                batches[name][ci0] = cnt
                cpu[name] += c
                if time.time() - last > 15 and k < len(tasks):
                    _save_partial(part, mode, batches, cpu)
                    last = time.time()
                if k % max(1, len(tasks) // 10) == 0 or k == len(tasks):
                    print(f"[mc] reference {k}/{len(tasks)} batches, {time.time() - t0:.0f}s", flush=True)
    for name in TARGETS:
        assert len(batches[name]) == cfg["ref"][name] // cfg["batch"][name], (name, len(batches[name]))
    out = {"ref_sig": ref_sig(mode), "state_sig": state_sig()[:10], "mode": mode, "version": REF_VERSION,
           "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": round(time.time() - t0, 1),
           "resumed_batches": resumed,
           "teams": targets["week"].teams, "owners": targets["week"].owners, "seed": REF_SEED, "targets": {}}
    for name in TARGETS:
        B = np.array([batches[name][k] for k in sorted(batches[name])], float)    # (nb, T) counts
        nb, per = len(B), cfg["batch"][name] * CHUNK
        N = nb * per
        p = B.sum(0) / N
        pb = B / per
        out["targets"][name] = {
            "N": int(N), "counts": B.sum(0).astype(int).tolist(), "p": p.tolist(),
            "se": np.sqrt(p * (1 - p) / N).tolist(),                          # exact for iid paths
            "se_batch": (pb.std(0, ddof=1) / np.sqrt(nb)).tolist(), "n_batches": nb,
            "ref_var": float(np.mean(p * (1 - p) / N)),                       # mean over teams of se^2
            "cpu_s": round(cpu[name], 1), "us_per_path": round(cpu[name] / N * 1e6, 2)}
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)
    try:
        os.remove(part)
    except FileNotFoundError:
        pass
    print(f"[mc] wrote {rel(path)} ({out['wall_s']}s)", flush=True)
    return out


def reference(mode):
    """The cached reference for a mode (current state), or None."""
    p = ref_path(mode)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- metrics
def _log_metric(parts, targets):
    """Mean over targets of log(mean of each part), e.g. parts ('mse', 'cost') -> log_work = log(MSE x cost).
    In log_work the mean cost is floored at COST_FLOOR (charged cost); log_cost stays raw."""
    keys = [f"{p}_{t}" for t in targets for p in parts]
    tag = {("mse", "cost"): "work", ("mse",): "mse", ("cost",): "cost"}[tuple(parts)]
    floor = np.array([COST_FLOOR if tag == "work" and k.startswith("cost_") else 0.0 for k in keys])

    def stat(a):
        return np.stack([np.asarray(a[k], float) for k in keys] + [np.ones(len(a[keys[0]]))], 1)

    def final(S):
        return np.sum(np.log(np.maximum(S[..., :-1] / S[..., -1:], floor)), axis=-1) / len(targets)
    name = f"log_{tag}" if len(targets) > 1 else f"log_{tag}_{targets[0]}"
    return M.Metric(name, stat, final, True, keys)


def _rmse_metric(t):
    k = f"mse_{t}"
    return M.Metric(f"rmse_{t}", lambda a: np.stack([np.asarray(a[k], float), np.ones(len(a[k]))], 1),
                    lambda S: np.sqrt(S[..., 0] / S[..., 1]), True, (k,))


def _bias2_metric(t):
    k = f"err_{t}"

    def stat(a):
        e = np.asarray(a[k], float)
        return np.concatenate([e, np.ones((len(e), 1))], 1)
    return M.Metric(f"bias2_{t}", stat, lambda S: np.mean((S[..., :-1] / S[..., -1:]) ** 2, axis=-1), True, (k,))


_W2, _M, _C = ("mse", "cost"), ("mse",), ("cost",)
METRICS = ([_log_metric(_W2, TARGETS)] + [_log_metric(_W2, (t,)) for t in TARGETS]
           + [_log_metric(_M, TARGETS), _log_metric(_C, TARGETS)]
           + [_log_metric(x, (t,)) for x in (_M, _C) for t in TARGETS]
           + [_rmse_metric(t) for t in TARGETS] + [M.mean_of(f"mse_{t}") for t in TARGETS]
           + [M.mean_of(f"cost_{t}") for t in TARGETS] + [M.mean_of(f"cpu_{t}") for t in TARGETS]
           + [_bias2_metric(t) for t in TARGETS])
PRIMARY = "log_work"


def charged(a, t):
    """Mean cost billed in log_work / eff_rel: max(mean cost, COST_FLOOR)."""
    return max(float(np.mean(a[f"cost_{t}"])), COST_FLOOR)


def _work(a, t, ref_var=0.0, cost="cost"):
    mse = float(np.mean(a[f"mse_{t}"])) - ref_var
    c = charged(a, t) if cost == "cost" else float(np.mean(a[f"{cost}_{t}"]))
    return mse * c if mse > 0 else None


def efficiency(a, base, ref):
    """{target: eff vs plain MC} = (MSE x charged cost) of the baseline over the candidate's; also with the
    reference's own variance subtracted from both MSEs (adj) and with raw CPU seconds across runs, unfloored
    (rawcpu, a diagnostic)."""
    out = {}
    for t in TARGETS:
        v = ref["targets"][t]["ref_var"]
        wc, wb, wca, wba = _work(a, t), _work(base, t), _work(a, t, v), _work(base, t, v)
        rc, rb = _work(a, t, cost="cpu"), _work(base, t, cost="cpu")
        out[t] = {"eff_rel": wb / wc if wc and wb else None, "eff_rel_adj": wba / wca if wca and wba else None,
                  "eff_rel_rawcpu": rb / rc if rc and rb else None}
    e = [out[t]["eff_rel"] for t in TARGETS]
    out["both"] = float(np.exp(np.mean(np.log(e)))) if all(e) else None
    return out


# ---------------------------------------------------------------- run
def run(key, variants, mode="smoke", jobs=None, notes="", extra=None, make_ref=False):
    """variants: [(name, zero-arg factory returning a fresh candidate)]. Writes results; returns the JSON path."""
    t_start = time.time()
    names = [n for n, _ in variants]
    assert R.NAME_RE.match(key) and all(R.NAME_RE.match(n) for n in names), f"bad key/variant name in {key} {names}"
    assert len(set(names)) == len(names), f"duplicate variant names {names}"
    cfg = CFG[mode]
    sig = signature(mode)
    targets = build_targets()
    ref = reference(mode)
    if ref is None:
        if mode == "full" and not make_ref:
            raise SystemExit(f"[mc] no full reference {rel(ref_path(mode))}: the queued _baseline_mc job "
                             "builds it (or pass --make-ref)")
        make_reference(mode, targets, jobs)
    del ref                                         # scoring re-reads it after the candidates ran
    ids = item_ids(mode, sig)
    Rn = cfg["R"]
    tasks = []
    for name in ("season", "week"):                  # long target first; calibration calls spread evenly
        cand = [(vi, name, r) for r in range(Rn) for vi in range(len(variants))]   # variants interleaved
        K = cfg["calib"]
        at = [min(len(cand), int((k + 0.5) * len(cand) / K)) for k in range(K)]
        k = 0
        for i in range(len(cand) + 1):
            while k < K and at[k] == i:
                tasks.append((-1, name, k))
                k += 1
            if i < len(cand):
                tasks.append(cand[i])
        assert k == K, (k, K)
    n_jobs = max(1, min(jobs or slots(), len(tasks)))
    per = max(1, slots() // n_jobs)
    _W.update(targets=targets, makers=[f for _, f in variants], n=cfg["n"])
    for t in TARGETS:            # warm the parent's heap so freshly forked workers don't pay first-touch page faults
        plain(targets[t], CHUNK, CALIB_SEED - 1)
    est = {(vi, t): np.zeros((Rn, targets[t].T)) for vi in range(len(variants)) for t in TARGETS}
    cpu = {k: np.zeros(Rn) for k in est}
    wall = {k: np.zeros(Rn) for k in est}
    calib = {t: np.zeros(cfg["calib"]) for t in TARGETS}
    t0, done = time.time(), 0
    print(f"[mc] {key} ({mode}, sig {sig}): {len(variants)} variant(s) x {Rn} reps x {len(TARGETS)} targets "
          f"+ {cfg['calib']} plain timing calls per target on {n_jobs} workers, n = {cfg['n']}", flush=True)
    with _pool(n_jobs, per, maxtasksperchild=1) as pool:        # a fresh process per call: no state carries over
        for vi, name, r, e, c, w in pool.imap_unordered(_task, tasks):
            if vi < 0:
                calib[name][r] = c
            else:
                est[vi, name][r], cpu[vi, name][r], wall[vi, name][r] = e, c, w
            done += 1
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                print(f"[mc] {done}/{len(tasks)} calls, {time.time() - t0:.0f}s", flush=True)
    assert all((calib[t] > 0).all() for t in TARGETS), "a calibration call did not run"
    ref = reference(mode)
    pref = {t: np.asarray(ref["targets"][t]["p"]) for t in TARGETS}
    base = None
    if key != f"_baseline_{TRACK}" and os.path.exists(R.baseline_path(TRACK, mode)):
        b = R.read(R.baseline_path(TRACK, mode))
        bv = next((v for v in b["variants"] if v["name"] == "baseline"), b["variants"][0])
        try:
            base = R.align(R.load_arrays(bv), ids, "baseline")
        except (AssertionError, OSError, KeyError):
            print("[mc] baseline result is for another state/config: no eff_rel", flush=True)
    recs = []
    for vi, (name, _) in enumerate(variants):
        a = {"item_id": ids, "block": np.arange(Rn, dtype=np.int64)}
        for t in TARGETS:
            err = est[vi, t] - pref[t][None, :]
            a.update({f"est_{t}": est[vi, t], f"err_{t}": err, f"mse_{t}": (err ** 2).mean(1),
                      f"cpu_{t}": cpu[vi, t], f"cost_{t}": cpu[vi, t] / np.median(calib[t]), f"wall_{t}": wall[vi, t],
                      f"seed_{t}": np.array([seed_of(t, r) for r in range(Rn)], np.int64)})
        met = R.evaluate(METRICS, a, ids, f"{key}/{name}")
        recs.append({"name": name, "metrics": {k: (float(f"{v:.7g}") if isinstance(v, float) else v) for k, v in met.items()},
                     "arrays": a})
    if base is None:                                # the baseline run scores against its own 'baseline' variant
        bi = next((i for i, (n, _) in enumerate(variants) if n == "baseline"), None)
        base = recs[bi]["arrays"] if key == f"_baseline_{TRACK}" and bi is not None else None
    for r in recs:
        a, by = r["arrays"], {}
        eff = efficiency(a, base, ref) if base is not None else None
        for t in TARGETS:
            e = a[f"err_{t}"]
            ae = np.abs(e).mean(0)
            # bias check: mean error per team over replications vs its SE (replication spread + reference SE)
            bz = e.mean(0) / (np.sqrt(e.var(0, ddof=1) / len(e) + np.asarray(ref["targets"][t]["se"]) ** 2) + 1e-12)
            by[t] = {"n": cfg["n"][t], "rmse": round(float(np.sqrt(a[f"mse_{t}"].mean())), 6),
                     "cpu_s": round(float(a[f"cpu_{t}"].mean()), 4), "wall_s": round(float(a[f"wall_{t}"].mean()), 4),
                     "cost": round(float(a[f"cost_{t}"].mean()), 4), "cost_charged": round(charged(a, t), 4),
                     "mean_abs_err_by_team": dict(zip(targets[t].owners, np.round(ae, 6).tolist())),
                     "bias_z_by_team": dict(zip(targets[t].owners, np.round(bz, 2).tolist())),
                     "bias_z_max": round(float(np.abs(bz).max()), 2),
                     "mse_over_ref_var": round(float(a[f"mse_{t}"].mean()) / ref["targets"][t]["ref_var"], 1)}
            if eff:
                by[t].update(eff[t])
                by[t]["cost_ratio"] = round(float(a[f"cost_{t}"].mean() / base[f"cost_{t}"].mean()), 3)
        r["by_target"] = by
        r["eff_rel"] = eff["both"] if eff else None
    tinfo = {t: {"n": cfg["n"][t], "horizon": targets[t].horizon, "weeks": targets[t].U.weeks[:targets[t].horizon],
                 "reference": [round(x, 6) for x in ref["targets"][t]["p"]],
                 "ref_se": [round(x, 7) for x in ref["targets"][t]["se"]],
                 "ref_N": ref["targets"][t]["N"], "ref_var": ref["targets"][t]["ref_var"],
                 "calib_cpu_s": round(float(np.median(calib[t])), 4), "calib_k": int(cfg["calib"]),
                 "calib_cpu_iqr": [round(float(x), 4) for x in np.quantile(calib[t], [0.25, 0.75])]} for t in TARGETS}
    path = R.write(key, TRACK, mode, recs, time.time() - t_start, notes, sig=sig, R=Rn, n_items=Rn,
                   teams=targets["week"].teams, owners=targets["week"].owners, me=targets["week"].me_t,
                   targets=tinfo, reference_file=rel(ref_path(mode)), jobs=n_jobs, **(extra or {}))
    bad = R.validate(R.read(path))
    assert not bad, bad
    for r in recs:
        m, by = r["metrics"], r["by_target"]
        cells = "  ".join(f"{t}: rmse {by[t]['rmse']:.5f} cpu {by[t]['cpu_s']:.3f}s cost {by[t]['cost']:.2f}"
                          f" |bias z|max {by[t]['bias_z_max']:.1f}"
                          + (f" eff {by[t]['eff_rel']:.2f}x" if by[t].get("eff_rel") else "") for t in TARGETS)
        print(f"[mc] {key}/{r['name']} ({mode}): log_work {m['log_work']:.4f}  {cells}"
              + (f"  eff(both) {r['eff_rel']:.2f}x" if r["eff_rel"] else ""), flush=True)
    print(f"[mc] wrote {rel(path)}", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="mc track: estimator efficiency vs plain MC")
    ap.add_argument("--candidate", help="module:Class used by every --variant without its own =module:Class")
    ap.add_argument("--key")
    ap.add_argument("--variant", action="append", default=None, help="NAME or NAME=module:Class (repeatable)")
    ap.add_argument("--mode", choices=R.MODES, default="smoke")
    ap.add_argument("--jobs", type=int, default=None, help="parallel workers (default CPUQ_SLOTS)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--make-ref", action="store_true", help="build the full reference if it is missing (baseline job)")
    ap.add_argument("--ref-only", action="store_true", help="build/check the reference and exit")
    args = ap.parse_args(argv)
    if args.ref_only:
        ref = reference(args.mode)
        if ref is None and args.mode == "full" and not args.make_ref:
            raise SystemExit(f"[mc] no full reference {rel(ref_path('full'))}: only the queued _baseline_mc job builds it")
        ref = ref or make_reference(args.mode, build_targets(), args.jobs)
        for t in TARGETS:
            x = ref["targets"][t]
            print(f"[mc] {t}: N {x['N']}  " + "  ".join(f"{o} {p:.4f}±{s:.5f}" for o, p, s in
                                                         zip(ref["owners"], x["p"], x["se"])))
        return ref_path(args.mode)
    if not args.key:
        ap.error("--key is required")
    variants, classes = [], {}
    c0 = time.process_time()                         # module imports run here, in the parent, untimed
    for v in args.variant or ["default"]:
        name, _, spec = v.partition("=")
        spec = spec or args.candidate
        if not spec:
            ap.error(f"variant {name}: no candidate class")
        if spec not in classes:
            classes[spec] = load_class(spec)
        variants.append((name, factory(classes[spec], name)))
    imp = time.process_time() - c0
    if imp > IMPORT_CPU_MAX:
        raise SystemExit(f"[mc] importing the candidate took {imp:.1f} CPU s (> {IMPORT_CPU_MAX}): import time is "
                         "not charged, so do target-dependent work inside estimate()")
    return run(args.key, variants, args.mode, args.jobs, args.notes, make_ref=args.make_ref,
               extra={"import_cpu_s": round(imp, 3)})


if __name__ == "__main__":
    main()
