"""POLICY track: my FAAB bidding policy, scored inside the season-with-auctions sim from today's state.

IN-SIMULATION ONLY. Nothing here is checked against real outcomes: every path is a draw from the
core models (player score model + copula, BidModel rivals), started from the real current league
state (League(): week, alive teams, rosters, budgets, a pending chopped roster if any). A policy
that leans on the very BidModel the rivals follow looks better here than it would in reality, so
read the two robustness conditions (rival stress, bootstrap-refit rivals) alongside the main one.

Items: season paths. Path c of CRN seed s is item "<state>-s<s:03d>-<c:03d>" (state = hash of the
data snapshot, so results from different snapshots never pair up), block = s. Seed s drives the
same random streams as dynamic.run(seed=s) chunk 0 (player scores, injuries, rival bid noise).
Every variant and condition is scored on the same paths (common random numbers).
  eval seeds   smoke 100-104, full 100-399 (fixed, < 1000; 200 paths each). run.py's own sims use
               seeds 11, 12, 13, 21, so these paths never overlap the ones its sweep was picked on.
  tune seeds   >= 1000 only: tune() refuses anything lower. Candidates tune there, never on eval seeds.
Conditions (same paths, rivals change; my policy always sees the point BidModel in state.bm):
  ""          rivals bid from the BidModel fit on all bid history (= dynamic.py)
  "_stress"   rival_boost (0.7, 0.3): rivals bid ~2x as often and ~35% higher (run.py's STRESS)
  "_bmboot"   rivals bid from a BidModel refit on a cluster bootstrap of the bid history (whole
              lots = waiver run x player), one replicate per seed, identical for every candidate
Per-item arrays (per condition suffix): win, weeks (weeks survived, 0-9), spend, buys, studs.

    python -m bench.harness.tracks.policy --candidate pkg.module:Class --key KEY \
        --variant NAME [--variant NAME2 | --variant NAME3=pkg.mod:Other] --mode smoke|full [--jobs N]

Candidate: Class(variant=NAME) (or Class()), built once in the parent; it must provide
bid(player, gain, week, state) -> dollars (rounded to whole dollars, capped at my budget, < 1 = no
claim) and may provide reset() (called at the start of every path; keep nothing across paths) and
begin(state) (once per auction). state is a dyn_policy.State. Seeds run in parallel over --jobs
forked workers (default CPUQ_SLOTS).
Isolation: every (seed, variant, condition) runs on a fresh deep copy of the built object (the
shared League / Universe / BidModel / PlayerModel are not copied) with numpy's and random's global
RNGs reseeded, so results never depend on --jobs, worker scheduling or the other variants in the run.
"""
import argparse
import copy
import hashlib
import multiprocessing as mp
import os
import random
import time

from .. import rel, slots
from .. import dyn_policy as DP
from .. import metrics as M
from .. import results as R
from .player import factory, load_class

import numpy as np

TRACK = "policy"
VERSION = 1
C = 200                                  # paths per CRN seed (one block)
SEEDS = {"smoke": range(100, 105), "full": range(100, 400)}
TUNE_MIN_SEED = 1000
TUNE_SEEDS = range(1000, 1010)
STRESS = (0.7, 0.3)                      # = run.STRESS
CONDITIONS = (("", {}), ("_stress", {"rival_boost": STRESS}), ("_bmboot", {"bm_rival": "boot"}))
OUTCOMES = DP.OUTCOMES


def _min_of(name, arrs, lower=False):
    """Worst (lowest) of several per-item means, e.g. P(win) across conditions."""
    arrs = list(arrs)

    def stat(a):
        return np.stack([np.asarray(a[k], float) for k in arrs] + [np.ones(len(a[arrs[0]]))], 1)

    def final(S):
        return np.min([S[..., i] / S[..., -1] for i in range(len(arrs))], axis=0)
    return M.Metric(name, stat, final, lower, arrs)


SFX = [s for s, _ in CONDITIONS]
METRICS = ([M.mean_of(f"p_win{s}", f"win{s}", lower=False) for s in SFX]
           + [_min_of("p_win_min", [f"win{s}" for s in SFX])]
           + [M.mean_of(f"e_weeks{s}", f"weeks{s}", lower=False) for s in SFX]
           + [M.mean_of(f"avg_spend{s}", f"spend{s}", lower=None) for s in SFX]
           + [M.mean_of(f"buys{s}", f"buys{s}", lower=None) for s in SFX]
           + [M.mean_of(f"studs{s}", f"studs{s}", lower=None) for s in SFX])
PRIMARY = "p_win"


# ---------------------------------------------------------------- environment (today's state)
_ENV = {}
SIG_FILES = ("meta.json", "league.json", "users.json", "rosters.json", "schedule.json", "model.npz")


def signature():
    """Content hash of the data snapshot the sim starts from (+ the track version)."""
    from common import DATA, J
    h = hashlib.sha1(f"policy-v{VERSION}".encode())
    meta = J("meta.json")
    names = list(SIG_FILES) + [f"tx_{w}.json" for w in range(1, meta["leg"] + 1)] \
        + [f"matchups_{w}.json" for w in range(1, meta["leg"] + 1)] \
        + [f"{meta['season']}/proj_{w}.json" for w in range(1, meta["final_week"] + 1)]
    for f in names:
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            h.update(f.encode())
            with open(p, "rb") as fh:
                h.update(hashlib.sha1(fh.read()).digest())
    return h.hexdigest()[:12]


def env():
    """League, PlayerModel, point BidModel, Universe and base state, built as run.py does (same
    Universe player list, so seed s reproduces dynamic.run's draws given the same BidModel). The
    BidModel is refit on canonically ordered rows (dyn_policy.canonical_bm): the core fit drifts
    ~1e-3 with PYTHONHASHSEED, which would break pairing across runs. Cached per process."""
    if not _ENV:
        import run as core_run
        from auction import BidModel
        from common import League
        from model import PlayerModel
        from sim import Universe
        sig0 = signature()
        lg = League()
        pm, bm = PlayerModel(), DP.canonical_bm(BidModel(lg))
        teams, me = lg.alive(), lg.me
        ros = {t: lg.players(t) for t in teams}
        pending = lg.pending
        fas = core_run.fa_candidates(lg, core_run.on_waivers(lg))
        weeks = list(range(lg.week, lg.final_week + 1))
        pend_pool = [p for p in (pending or {}).get("pool", []) if sum(lg.proj_pts(p, w) for w in weeks) > 0]
        U = Universe(lg, pm, [p for t in teams for p in ros[t]] + fas + pend_pool, weeks)
        base = {"me": me, "teams": teams, "rosters": ros, "budget": {t: lg.faab(t) for t in teams},
                "pending_pool": pend_pool if pending else None}
        DP.setup(U, bm, base)
        sig = signature()
        if sig != sig0:  # fetch.py / model.py wrote data/ while we were loading it
            raise RuntimeError(f"data/ changed while loading (state {sig0} -> {sig}); rerun once it settles")
        _ENV.update(lg=lg, pm=pm, bm=bm, U=U, base=base, sig=sig, state={
            "season": lg.season, "week": lg.week, "weeks": weeks, "me": me, "owner": lg.owner(me),
            "my_budget": lg.faab(me), "alive": teams, "budgets": base["budget"],
            "pending": {"rid": pending["rid"], "n": len(pend_pool)} if pending else None})
    return _ENV


def item_ids(seeds, sig=None):
    sig = sig or env()["sig"]
    ids = np.array([f"{sig}-s{s:03d}-{c:03d}" for s in seeds for c in range(C)], dtype=str)
    return ids, np.repeat(np.asarray(list(seeds), np.int64), C)


def items(mode="full"):
    """Harness item ids for a mode (what every candidate must be scored on)."""
    return item_ids(SEEDS[mode])[0]


# ---------------------------------------------------------------- simulation
_W = {}


def _init_worker(threads):
    os.environ["HARNESS_THREADS"] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
        _W["limit"] = threadpool_limits(threads)
    except ImportError:
        pass


_COPY_WARNED = set()


def _fresh(pol):
    """A pristine deep copy of a built policy object; the big shared read-only objects (League,
    Universe, BidModel, PlayerModel, base state) are referenced, not copied. Falls back to the
    object itself (with a warning) if it cannot be deep-copied."""
    E = env()
    memo = {id(o): o for o in (E["lg"], E["pm"], E["bm"], E["U"], E["base"])}
    try:
        return copy.deepcopy(pol, memo)
    except Exception as e:  # noqa: BLE001  (pools, file handles, ...)
        if type(pol).__name__ not in _COPY_WARNED:
            _COPY_WARNED.add(type(pol).__name__)
            print(f"[policy] WARNING: cannot deep-copy {type(pol).__name__} ({e!r}); reusing one object, so "
                  "anything it keeps across paths makes results depend on worker scheduling", flush=True)
        return pol


def _chunk(seed, protos, conds):
    """Every (policy, condition) on the C paths of one CRN seed; {(vi, sfx): {outcome: list}}.

    Each (variant, condition) runs in its own _paths_chunk call on a fresh copy of the policy, with
    the global RNGs reseeded: the same draws as one joint call (every setting re-derives its streams
    from the seed), but nothing a policy keeps or consumes can leak into another setting or seed."""
    E = env()
    seed = int(seed)
    B = E["base"]["budget"][E["base"]["me"]]
    boot = None
    out = {}
    for vi, pol in enumerate(protos):
        for sfx, cond in conds:
            st = {"policy": _fresh(pol), "budget": B}
            for k, v in cond.items():
                if k == "bm_rival" and v == "boot":
                    boot = boot or DP.boot_bm(E["bm"], seed)
                    v = boot
                st[k] = v
            # same global streams for every variant in a condition (CRN even for code that uses them)
            random.seed(f"policy-global:{seed}:{sfx}")
            np.random.seed([seed, SFX.index(sfx)])
            out[vi, sfx] = DP._paths_chunk((0, C, seed, [st]))[0]["path"]
    return out


def _task(seed):
    t0 = time.time()
    return seed, _chunk(seed, _W["protos"], _W["conds"]), time.time() - t0


def _simulate(protos, seeds, conds=CONDITIONS, jobs=None, tag="policy"):
    """{vi: {f"{outcome}{sfx}": array over paths, seed-major}} for policy objects on CRN seeds."""
    env()
    seeds = [int(s) for s in seeds]
    n_jobs = max(1, min(jobs or slots(), slots(), len(seeds)))  # never more workers than cpuq cores
    per = max(1, slots() // n_jobs)
    got = {}
    if n_jobs == 1:
        for s in seeds:
            t0 = time.time()
            got[s] = _chunk(s, protos, conds)
            print(f"[{tag}] seed {s} {time.time() - t0:.1f}s", flush=True)
    else:
        _W.update(protos=protos, conds=conds)
        with mp.get_context("fork").Pool(n_jobs, initializer=_init_worker, initargs=(per,)) as pool:
            for k, (s, r, dt) in enumerate(pool.imap_unordered(_task, seeds)):
                got[s] = r
                if k % max(1, len(seeds) // 10) == 0 or k == len(seeds) - 1:
                    print(f"[{tag}] {k + 1}/{len(seeds)} seeds (seed {s} {dt:.1f}s)", flush=True)
    out = {}
    for vi in range(len(protos)):
        out[vi] = {f"{o}{sfx}": np.concatenate([np.asarray(got[s][vi, sfx][o], float) for s in seeds])
                   for sfx, _ in conds for o in OUTCOMES}
    return out


def tune(policies, seeds=None, conditions=("",), jobs=None):
    """Score policy objects on TUNING seeds only (>= 1000); for candidates choosing parameters.

    policies: list of policy objects; conditions: suffixes from CONDITIONS. Returns one dict per
    policy: p_win, se, e_weeks, avg_spend, n and the per-path arrays (key "arrays").
    """
    seeds = list(TUNE_SEEDS if seeds is None else seeds)
    if not seeds or min(seeds) < TUNE_MIN_SEED:
        raise ValueError(f"tune() may only use CRN seeds >= {TUNE_MIN_SEED}; evaluation seeds are below that")
    conds = [c for c in CONDITIONS if c[0] in conditions]
    assert len(conds) == len(conditions), f"unknown condition in {conditions}"
    sim = _simulate(list(policies), seeds, conds, jobs, tag="tune")
    out = []
    for vi in range(len(policies)):
        a = sim[vi]
        w = a["win" + conds[0][0]]
        p = float(w.mean())
        out.append({"p_win": p, "se": float(np.sqrt(p * (1 - p) / len(w))), "e_weeks": float(a["weeks" + conds[0][0]].mean()),
                    "avg_spend": float(a["spend" + conds[0][0]].mean()), "n": int(len(w)), "arrays": a})
    return out


# ---------------------------------------------------------------- scoring
def paired(a, b, blocks):
    """Mean paired difference a - b per condition (win, weeks, spend) with a cluster (block) se."""
    ub, inv = np.unique(blocks, return_inverse=True)
    out = {}
    for sfx in SFX:
        for o, name in (("win", "p_win"), ("weeks", "e_weeks"), ("spend", "avg_spend")):
            d = np.asarray(a[o + sfx], float) - np.asarray(b[o + sfx], float)
            bm = np.bincount(inv, d) / np.bincount(inv)
            se = float(bm.std(ddof=1) / np.sqrt(len(ub))) if len(ub) > 1 else float("nan")
            out[f"d_{name}{sfx}"] = {"diff": round(float(d.mean()), 6), "se": round(se, 6),
                                     "disagree": round(float((d != 0).mean()), 4)}
    return out


def _reference(mode, ids, recs):
    """Arrays of the baseline's "baseline" variant on these items: this run's own (baseline run) or the file."""
    for r in recs:
        if r["name"] == "baseline" and r.get("_is_baseline"):
            return r["arrays"], "this run"
    p = R.baseline_path(TRACK, mode)
    if not os.path.exists(p):
        return None, None
    base = R.read(p)
    ref = next((v for v in base["variants"] if v["name"] == "baseline"), base["variants"][0])
    try:
        return R.align(R.load_arrays(ref), ids, "baseline"), rel(p)
    except (AssertionError, OSError, KeyError, ValueError) as e:
        print(f"[policy] no vs_baseline: {rel(p)} does not pair with this run ({e}); "
              f"its state is {base.get('state_sig')}, this run's {ids[0].split('-')[0]}", flush=True)
        return None, None


def run(key, variants, mode="smoke", jobs=None, notes="", extra=None):
    """variants: [(name, zero-arg factory returning a policy object)]. Writes results; returns the JSON path."""
    t_start = time.time()
    E = env()
    seeds = list(SEEDS[mode])
    assert max(seeds) < TUNE_MIN_SEED
    protos = [f() for _, f in variants]
    t_sim = time.time()
    sim = _simulate(protos, seeds, CONDITIONS, jobs, tag=TRACK)
    t_sim = time.time() - t_sim
    ids, blocks = item_ids(seeds)
    recs = []
    for vi, (name, _) in enumerate(variants):
        arrays = {"item_id": ids, "block": blocks, **sim[vi]}
        met = R.evaluate(METRICS, arrays, ids, f"{key}/{name}")
        for s in SFX:
            p, w = met[f"p_win{s}"], arrays["win" + s]
            met[f"p_win{s}_se"] = float(np.sqrt(p * (1 - p) / len(w)))
        recs.append({"name": name, "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items()},
                     "arrays": arrays, "_is_baseline": key == f"_baseline_{TRACK}"})
    ref, ref_src = _reference(mode, ids, recs)
    for r in recs:
        r.pop("_is_baseline")
        if ref is not None:
            r["vs_baseline"] = paired(r["arrays"], ref, blocks) | {"_ref": ref_src}
    path = R.write(key, TRACK, mode, recs, time.time() - t_start, notes, in_sim=True,
                   state=E["state"], state_sig=E["sig"], seeds=[seeds[0], seeds[-1]], paths_per_seed=C,
                   n_items=int(len(ids)), conditions={s or "main": c for s, c in
                                                      (("", "point BidModel"), ("_stress", f"rival_boost {STRESS}"),
                                                       ("_bmboot", "lot-bootstrap BidModel refit per seed"))},
                   sim_s=round(t_sim, 1), jobs=max(1, min(jobs or slots(), slots(), len(seeds))), **(extra or {}))
    bad = R.validate(R.read(path))
    assert not bad, bad
    for r in recs:
        m = r["metrics"]
        line = (f"[policy] {key}/{r['name']} ({mode}, in-sim): p_win {m['p_win']:.4f}±{m['p_win_se']:.4f}  "
                f"stress {m['p_win_stress']:.4f}  bmboot {m['p_win_bmboot']:.4f}  e_weeks {m['e_weeks']:.3f}  "
                f"spend ${m['avg_spend']:.1f}  n {m['n']}")
        vb = r.get("vs_baseline")
        if vb:
            d = vb["d_p_win"]
            line += f"  | vs baseline {d['diff']:+.4f}±{d['se']:.4f}"
        print(line, flush=True)
    print(f"[policy] wrote {rel(path)}", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="policy track (in-sim FAAB bid policy evaluation)")
    ap.add_argument("--candidate", help="module:Class used by every --variant without its own =module:Class")
    ap.add_argument("--key", required=True)
    ap.add_argument("--variant", action="append", default=None, help="NAME or NAME=module:Class (repeatable)")
    ap.add_argument("--mode", choices=R.MODES, default="smoke")
    ap.add_argument("--jobs", type=int, default=None, help="parallel seed workers (default CPUQ_SLOTS)")
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
