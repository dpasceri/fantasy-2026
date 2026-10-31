"""Reviewer add-on: paired comparison vs the track baseline on blocks the implementer never looked at.

Development checks ran on 2025 weeks 5-9 (full-run test blocks), so design choices could carry a small
selection bias there. This re-scores every variant on the remaining blocks only (same paired block
bootstrap as bench.harness.compare) and writes bench/candidates/mixture_hurdle/clean_subset[.smoke].json.
Reads result files only; no fitting.

    python -m bench.candidates.mixture_hurdle.clean_subset [--mode full|smoke]
"""
import argparse
import json
import os

from bench import harness  # noqa: F401  (core modules on sys.path)
from bench.harness import data as D
from bench.harness import metrics as M
from bench.harness import results as R
from bench.harness.tracks import player as PT

import numpy as np

KEY = "mixture_hurdle"
DEV_WEEKS = [(2025, w) for w in range(5, 10)]           # seen during development
SHOW = ("crps", "pinball", "cov_err", "pit_maxdev", "crps_QB", "crps_RB", "crps_WR", "crps_TE")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="full", choices=R.MODES)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args(argv)
    base = R.read(R.baseline_path(PT.TRACK, args.mode))
    ref = next((v for v in base["variants"] if v["name"] == "baseline"), base["variants"][0])
    ba = R.load_arrays(ref)
    dev_t = [D.t_of(o) for o in DEV_WEEKS]
    keep = ~np.isin(ba["block"], dev_t)
    sub = lambda a: {k: np.asarray(x)[keep] for k, x in a.items()}  # noqa: E731
    bs = sub(ba)
    ub, _ = M.block_index(bs["block"])
    C = M.bootstrap_counts(len(ub), args.B, 0)
    ms = [m for m in PT.METRICS if m.name in SHOW and m.ok(bs)]
    res = R.read(R.json_path(KEY, args.mode))
    out = {"key": KEY, "mode": args.mode, "excluded_blocks": [f"{s}-{w:02d}" for s, w in DEV_WEEKS],
           "n_items": int(keep.sum()), "n_blocks": int(len(ub)), "B": args.B, "variants": {}}
    print(f"[clean_subset] {KEY} ({args.mode}): {keep.sum()} items, {len(ub)} blocks (dev weeks 2025-05..09 dropped)")
    for v in res["variants"]:
        a = sub(R.align(R.load_arrays(v), ba["item_id"].astype(str), f"{KEY}/{v['name']}"))
        assert np.array_equal(a["block"], bs["block"]) and np.allclose(a["y"], bs["y"])
        row = {m.name: {k: (round(x, 6) if isinstance(x, float) else x)
                        for k, x in M.paired_bootstrap(m, a, bs, bs["block"], C).items()} for m in ms}
        out["variants"][v["name"]] = row
        print(f"  {v['name']:<10} " + "  ".join(f"{k} {row[k]['diff']:+.4f} [{row[k]['lo']:+.4f},{row[k]['hi']:+.4f}]"
                                            for k in ("crps", "cov_err", "pit_maxdev") if k in row))
    sfx = ".smoke" if args.mode == "smoke" else ""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"clean_subset{sfx}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[clean_subset] wrote {harness.rel(path)}")


if __name__ == "__main__":
    main()
