"""Tune on the dev target only (mc.build_targets(dev=True): 9 teams, weeks 9-16), never the scored state.

    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates.variance_reduction.tune week '{"K":16,"lhs":true,"Rp":4}' ...

For each config: reps independent calls at n paths, CPU per call vs interleaved plain-MC calls at the same n,
efficiency = (plain variance / config variance) x (plain CPU / config CPU), variances summed over teams.
Plain variance is p(1-p)/n with p the pooled mean of every config's (unbiased) estimates.
"""
import json
import sys
import time

from bench.harness.tracks import mc

import numpy as np

from . import vr


def main(argv):
    target, cfgs = argv[0], [json.loads(a) for a in argv[1:]]
    reps = int(next((c["reps"] for c in cfgs if "reps" in c), 16))
    n = int(next((c["n"] for c in cfgs if "n" in c), 40000 if target == "week" else 24000))
    tg = mc.build_targets(dev=True)[target]
    P = vr.Prep(tg)
    fn = vr.week_estimate if target == "week" else vr.season_estimate
    E = {i: [] for i in range(len(cfgs))}
    cpu = {i: [] for i in range(len(cfgs))}
    pc = []
    for r in range(reps):
        if r % 4 == 0:
            c0 = time.process_time()
            mc.plain(tg, n, 300 + r)
            pc.append(time.process_time() - c0)
        for i, c in enumerate(cfgs):
            kw = {k: v for k, v in c.items() if k not in ("reps", "n")}
            c0 = time.process_time()
            E[i].append(fn(P, n, np.random.default_rng([11, r]), **kw))
            cpu[i].append(time.process_time() - c0)
    p = np.mean([np.mean(E[i], 0) for i in E], 0)
    pv = (p * (1 - p)).sum() / n
    tp = float(np.median(pc))
    print(f"[tune] dev {target} n={n} reps={reps}: plain cpu {tp:.3f}s  p {np.round(p, 4).tolist()}")
    for i, c in enumerate(cfgs):
        v = np.var(E[i], 0, ddof=1).sum()
        cc = float(np.median(cpu[i])) / tp
        print(f"[tune] {json.dumps(c):60s} var x{pv / v:6.2f}  cost {cc:5.2f}  eff {pv / v / cc:6.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
