"""Budget calibration on the dev target only: CPU of each variant's sampler at two path counts vs interleaved
plain-MC calls, fitted as cpu(N) = plain_per_path x (a + b N). Prints the (a, b) to put in candidate.COST.

    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates.variance_reduction.calib [week|season] [variant ...]
"""
import os
import sys
import time

from bench.harness.tracks import mc

import numpy as np

from . import candidate, vr


def main(argv):
    which = ("week", "season")
    if argv and argv[0] in which:
        which, argv = (argv[0],), argv[1:]
    names = argv or list(candidate.VARIANTS)
    tgs = mc.build_targets(dev=True)
    for tname in which:
        tg = tgs[tname]
        Ns = (8000, 40000) if tname == "week" else (8000, 32000)
        reps = int(os.environ.get("CALIB_REPS", 20 if tname == "week" else 6))
        pl = {N: [] for N in Ns}
        cpu = {(v, N): [] for v in names for N in Ns}
        for r in range(reps):
            for N in Ns:
                c0 = time.process_time()
                mc.plain(tg, N, 500 + r)
                pl[N].append(time.process_time() - c0)
                for v in names:
                    cfg = {k: x for k, x in candidate.VARIANTS[v][tname].items() if k not in ("a", "b")}
                    c0 = time.process_time()
                    P = vr.Prep(tg)
                    (vr.week_estimate if tname == "week" else vr.season_estimate)(P, N, np.random.default_rng([r, N]), **cfg)
                    cpu[v, N].append(time.process_time() - c0)
        pp = np.mean([np.median(pl[N]) / N for N in Ns])                      # plain CPU per path
        print(f"[calib] dev {tname}: plain {pp * 1e6:.2f} us/path " +
              " ".join(f"(n={N}: {np.median(pl[N]):.3f}s)" for N in Ns))
        for v in names:
            y = np.array([np.median(cpu[v, N]) / pp for N in Ns])                # in plain-path units
            b = (y[1] - y[0]) / (Ns[1] - Ns[0])
            a = y[0] - b * Ns[0]
            print(f"[calib]   {v:14s} a {a:8.1f}  b {b:.4f}   cost at N=n: " +
                  " ".join(f"{y[i] / Ns[i]:.3f}" for i in range(2)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
