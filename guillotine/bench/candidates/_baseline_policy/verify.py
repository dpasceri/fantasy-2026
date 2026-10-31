"""Check the policy track's sim against the core code (tuning seeds only, never eval paths).

    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates._baseline_policy.verify

1. refit_bm(bm, bm.rows) reproduces BidModel(lg) exactly.
2. dyn_policy._paths_chunk without a policy equals dynamic._paths_chunk on the same job (every sum),
   for two multiplier settings and the stress setting.
3. The policy hook with Multiplier reproduces those settings bit for bit (wins, weeks, spend, studs),
   and its per-path outcomes sum to the chunk totals.
4. State.p_win (analytic) vs a Monte Carlo of auction.top_rival_bids on one captured auction state.
5. Determinism across processes: one seed's outcomes hash the same under two PYTHONHASHSEEDs
   (the core BidModel fit alone does not; the track refits it on canonically ordered rows).
6. Isolation: the track's per-(variant, condition) calls on fresh policy copies equal one joint call, a
   variant's paths do not depend on the other variants, and a policy that keeps state across paths
   (or uses numpy's global RNG) gives the same seed results whatever ran before it.
"""
import time

from bench import harness  # noqa: F401
from bench.harness import dyn_policy as DP
from bench.harness.tracks import policy as T
from bench.candidates._baseline_policy.multiplier import Multiplier

import numpy as np


def main():
    t0 = time.time()
    E = T.env()
    U, bm, base = E["U"], E["bm"], E["base"]
    B = base["budget"][base["me"]]
    print(f"state {E['sig']}: week {E['state']['week']}, {len(base['teams'])} alive, my budget {B}, "
          f"pending {E['state']['pending']}, universe {U.n} players x {len(U.weeks)} weeks")

    # 1. bid model refit: the copied fitting code reproduces the core fit on the same rows exactly;
    # the track's point model (canonical row order) differs from this process's core fit only by
    # the hash-order drift
    from auction import BidModel
    core = BidModel(E["lg"])

    def dev(x, y):
        return max(np.max(np.abs(np.asarray(x.a) - y.a)), np.max(np.abs(x.beta - y.beta)), abs(x.sigma - y.sigma),
                   max(abs(x.v[k] - y.v[k]) for k in y.v), max(abs(x.u[k] - y.u[k]) for k in y.u))
    assert dev(DP.refit_bm(core, core.rows), core) == 0
    t1 = time.time()
    bb = DP.boot_bm(bm, 1000)
    print(f"1. refit_bm == BidModel exactly; canonical-order point model vs this process's core fit: "
          f"max |diff| {dev(bm, core):.1e}; one bootstrap refit {time.time() - t1:.2f}s "
          f"(a {np.round(bm.a, 2)} -> {np.round(bb.a, 2)}, sigma {bm.sigma:.3f} -> {bb.sigma:.3f})")

    # 2 + 3. chunks vs dynamic.py
    import dynamic
    from sim import _G as SG
    SG.update(U=U, bm=bm, base=base)
    sets = [{"m_early": 0.75, "m_late": 0.75, "switch_week": 12, "budget": B},
            {"m_early": 1.0, "m_late": 0.75, "switch_week": 12, "budget": B},
            {"m_early": 0.75, "m_late": 0.75, "switch_week": 12, "budget": B, "rival_boost": T.STRESS}]
    pols = [Multiplier("m0.75"), Multiplier("baseline"), Multiplier("m0.75")]
    keys = ("wins", "weeks", "spent", "studs", "win_by", "weeks_by", "league_spend", "top_price", "burned")
    for seed in (1000, 1001):
        job = (0, 100, seed, sets)
        ref = dynamic._paths_chunk(job)
        mine = DP._paths_chunk(job)
        hook = DP._paths_chunk((0, 100, seed, [s | {"policy": p} for s, p in zip(sets, pols)]))
        expl = DP._paths_chunk((0, 100, seed, [s | {"bm_rival": bm} for s in sets]))
        for i in range(len(sets)):
            for k in keys:
                assert ref[i][k] == mine[i][k], (seed, i, k, ref[i][k], mine[i][k])
                assert ref[i][k] == expl[i][k], (seed, i, k, "explicit bm_rival")
            for k in ("wins", "weeks", "spent", "studs"):
                assert ref[i][k] == hook[i][k], (seed, i, k, ref[i][k], hook[i][k])
            pth = hook[i]["path"]
            assert sum(pth["win"]) == ref[i]["wins"] and sum(pth["weeks"]) == ref[i]["weeks"]
            assert sum(pth["spend"]) == ref[i]["spent"] and sum(pth["studs"]) == ref[i]["studs"]
        print(f"2-3. seed {seed}: dyn_policy == dynamic on {len(sets)} settings x 100 paths "
              f"(wins {[r['wins'] for r in ref]}, spent {[round(r['spent']) for r in ref]})")

    # 4. analytic p_win vs Monte Carlo of the rival top bid
    from auction import p_win_curve, top_rival_bids

    class Probe(Multiplier):
        seen = []

        def begin(self, state):
            if len(self.seen) < 3:
                self.seen.append(state)

    pr = Probe("baseline")
    DP._paths_chunk((0, 20, 1002, [{"policy": pr, "budget": B}]))
    rng = np.random.default_rng(5)
    worst = 0.0
    for st in Probe.seen:
        pid = max(st.pool, key=lambda p: max(st.rival_gains(p).values(), default=0))
        rg = st.rival_gains(pid)
        top = top_rival_bids(st.bm, {t: st.budgets[t] for t in rg}, rg, st.week, rng, 200_000)
        curve = p_win_curve(top, 200)
        grid = np.arange(1, 121)
        an = st.p_win(pid, grid)
        worst = max(worst, float(np.max(np.abs(an - curve[grid]))))
        print(f"4. week {st.week} {pid} ({st.pos(pid)}): my gain {st.gains[pid]:.1f}, top rival gain {max(rg.values()):.1f}; "
              f"p_win(10/30/60) analytic {np.round(st.p_win(pid, [10, 30, 60]), 3)} vs MC {np.round(curve[[10, 30, 60]], 3)}")
    assert worst < 0.01, worst
    print(f"   max |analytic - MC| over bids 1..120: {worst:.4f}")

    # 5. the same seed gives the same paths in fresh processes with different hash seeds
    import os
    import subprocess
    import sys
    code = ("from bench.harness.tracks import policy as T; from bench.candidates._baseline_policy.multiplier import Multiplier; "
            "import hashlib, json; r = T._chunk(1003, [Multiplier('baseline')], T.CONDITIONS); "
            "print(hashlib.sha1(json.dumps({str(k): v for k, v in r.items()}, sort_keys=True).encode()).hexdigest())")
    outs = [subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED=h), capture_output=True,
                           text=True, check=True).stdout.strip() for h in ("1", "2")]
    assert outs[0] == outs[1], outs
    print(f"5. seed 1003 identical under PYTHONHASHSEED 1 and 2 ({outs[0][:12]})")

    # 6. isolation: the track's per-setting calls on fresh policy copies equal one joint _paths_chunk call,
    # a variant's paths do not depend on the other variants in the run, and a policy that (against the
    # rules) keeps state across paths still gives the same seed results whatever ran before it
    class Leaky(Multiplier):
        def __init__(self, variant="baseline"):
            super().__init__(variant)
            self.n = 0

        def bid(self, player, gain, week, state):
            self.n += 1
            return super().bid(player, gain, week, state) + (self.n % 3 == 0) + float(np.random.random() < 0.1)

    two = [Multiplier("baseline"), Multiplier("m0.75")]
    got = T._chunk(1004, two, T.CONDITIONS)
    joint = DP._paths_chunk((0, T.C, 1004, [{"policy": p, "budget": B} | ({"rival_boost": T.STRESS} if s == "_stress" else {})
                                            | ({"bm_rival": DP.boot_bm(bm, 1004)} if s == "_bmboot" else {})
                                            for p in two for s, _ in T.CONDITIONS]))
    for k, r in zip([(vi, s) for vi in range(2) for s, _ in T.CONDITIONS], joint):
        assert got[k] == r["path"], k
    solo = T._chunk(1004, [Multiplier("m0.75")], T.CONDITIONS)
    assert all(solo[0, s] == got[1, s] for s, _ in T.CONDITIONS)
    lk = Leaky("baseline")
    a1 = T._chunk(1005, [lk], T.CONDITIONS)
    T._chunk(1006, [lk], T.CONDITIONS)
    a2 = T._chunk(1005, [lk], T.CONDITIONS)
    assert a1 == a2 and lk.n == 0, "per-seed isolation broken"
    print("6. per-setting fresh copies == one joint call; variants independent of each other; "
          "a state-carrying policy gives the same seed results whatever ran before")
    print(f"all checks passed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
