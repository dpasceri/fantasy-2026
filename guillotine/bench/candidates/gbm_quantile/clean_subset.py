"""Reviewer add-on: paired comparison vs the track baseline on the blocks design selection never scored.

The fixed choices (leaf n/40, max_features 1/3, split target, feature list, 50/50 P0 average, the
variant set) were picked on inner-validation weeks 2025 9-13, 2025 15-18 and 2026 1-6, and the smoke
weeks (2025-10, 2026-03, 2026-07) were seen. Those are full-run test blocks, so the full leaderboard
can carry selection bias there. This re-scores every variant on the remaining blocks only
(2025-05..08 and 2025-14; 2025-05 was run once as a crash check, no tuning after it), with the same
paired block bootstrap as bench.harness.compare plus per-block CRPS differences, and writes
bench/candidates/gbm_quantile/clean_subset[.smoke].json. Reads result files only; no fitting.

    python -m bench.candidates.gbm_quantile.clean_subset [--mode full|smoke] [--exclude 2026-07 ...]
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

KEY = "gbm_quantile"
DEV_WEEKS = ([(2025, w) for w in range(9, 14)] + [(2025, w) for w in range(15, 19)]
             + [(2026, w) for w in range(1, 8)])          # scored during design (inner val + smoke)
SHOW = ("crps", "pinball", "cov_err", "pit_maxdev", "crps_QB", "crps_RB", "crps_WR", "crps_TE")


def _week(s):
    a, b = s.split("-")
    return int(a), int(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="full", choices=R.MODES)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--exclude", nargs="*", default=None, help="season-week blocks to drop (default: the design weeks)")
    args = ap.parse_args(argv)
    dev = [_week(s) for s in args.exclude] if args.exclude is not None else DEV_WEEKS
    base = R.read(R.baseline_path(PT.TRACK, args.mode))
    ref = next((v for v in base["variants"] if v["name"] == "baseline"), base["variants"][0])
    ba = R.load_arrays(ref)
    keep = ~np.isin(ba["block"], [D.t_of(o) for o in dev])
    out = {"key": KEY, "mode": args.mode, "excluded_blocks": [f"{s}-{w:02d}" for s, w in dev],
           "n_items": int(keep.sum()), "n_blocks": 0, "B": args.B, "variants": {}}
    sfx = ".smoke" if args.mode == "smoke" else ""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"clean_subset{sfx}.json")
    if not keep.any():
        print(f"[clean_subset] {KEY} ({args.mode}): every block is excluded, nothing to score")
    else:
        sub = lambda a: {k: np.asarray(x)[keep] for k, x in a.items()}  # noqa: E731
        bs = sub(ba)
        ub, inv = M.block_index(bs["block"])
        out["n_blocks"] = int(len(ub))
        out["blocks"] = ["%d-%02d" % D.week_of(t) for t in ub]
        C = M.bootstrap_counts(len(ub), args.B, 0)
        ms = [m for m in PT.METRICS if m.name in SHOW and m.ok(bs)]
        res = R.read(R.json_path(KEY, args.mode))
        print(f"[clean_subset] {KEY} ({args.mode}): {keep.sum()} items, {len(ub)} blocks {out['blocks']}")
        for v in res["variants"]:
            a = sub(R.align(R.load_arrays(v), ba["item_id"].astype(str), f"{KEY}/{v['name']}"))
            assert np.array_equal(a["block"], bs["block"]) and np.allclose(a["y"], bs["y"])
            row = {m.name: {k: (round(x, 6) if isinstance(x, float) else x)
                            for k, x in M.paired_bootstrap(m, a, bs, bs["block"], C).items()} for m in ms}
            d = M.block_sums(np.stack([a["crps"] - bs["crps"], np.ones(len(inv))], 1), inv, len(ub))
            row["crps_diff_by_block"] = {b: round(float(x), 4) for b, x in zip(out["blocks"], d[:, 0] / d[:, 1])}
            out["variants"][v["name"]] = row
            print(f"  {v['name']:<10} " + "  ".join(f"{k} {row[k]['diff']:+.4f} [{row[k]['lo']:+.4f},{row[k]['hi']:+.4f}]"
                                                for k in ("crps", "cov_err", "pit_maxdev") if k in row)
                  + "  by block " + " ".join(f"{x:+.3f}" for x in row["crps_diff_by_block"].values()))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[clean_subset] wrote {harness.rel(path)}")


if __name__ == "__main__":
    main()
