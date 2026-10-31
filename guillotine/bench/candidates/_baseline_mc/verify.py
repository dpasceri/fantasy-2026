"""Check the plain-MC baseline against the core sim, and the cached reference against run.py's last run.

    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates._baseline_mc.verify

1. mc.plain on the season target equals sim.run_season(U, plan, n, seed)["win"][0] exactly, and on
   the week target equals run_season's week-0 chop probabilities exactly (same seed, same draws).
2. If a reference is cached and out/results.json is from the same fetch, z-scores of run.py's
   standings (640k paths, a Universe that also holds free agents) against the reference.
"""
import json
import os

from bench.harness.tracks import mc

import numpy as np
import sim
from common import OUT


def main(n=16_000, seed=5):
    sim.THREADS = 1
    tg = mc.build_targets()
    S = sim.run_season(tg["season"].U, tg["season"].plan, n, seed=seed)
    dw = np.abs(mc.plain(tg["season"], n, seed) - S["win"][0]).max()
    dc = np.abs(mc.plain(tg["week"], n, seed) - S["chop"][0, :, 0]).max()
    print(f"plain vs sim.run_season ({n} paths, seed {seed}): max|d win| {dw:.3g}  max|d week-0 chop| {dc:.3g}")
    ok = dw == 0 and dc == 0
    res_p = os.path.join(OUT, "results.json")
    for mode in ("full", "smoke"):
        ref = mc.reference(mode)
        if ref is None or not os.path.exists(res_p):
            continue
        res = json.load(open(res_p))
        meta = json.load(open(os.path.join(mc.DATA, "meta.json")))
        if res.get("fetched_at") != meta["fetched_at"] or res.get("week") != tg["week"].lg.week:
            print("out/results.json is from another fetch: skipped")
            break
        st = {s["rid"]: s for s in res["standings"]}
        Nr = res["sims"]["season"]
        print(f"reference ({mode}) vs out/results.json standings ({Nr} paths):")
        for t, key in (("week", "p_chop_now"), ("season", "p_win")):
            x = ref["targets"][t]
            z = [(st[r][key] - p) / np.sqrt(se ** 2 + p * (1 - p) / Nr)
                 for r, p, se in zip(ref["teams"], x["p"], x["se"])]
            print(f"  {t:6s} N {x['N']:>9d}  z: " + " ".join(f"{v:+.1f}" for v in z) + f"  max|z| {np.abs(z).max():.2f}")
        break
    print("PLAIN == sim.run_season" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
