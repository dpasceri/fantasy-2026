"""Leaderboard for one track: every result file vs the track baseline, paired block bootstrap.

    python -m bench.harness.compare <track> [--mode full|smoke] [--B 2000] [--seed 0]

For each (result, variant) on the baseline's items: metric value, baseline value, mean paired
difference (candidate - baseline) and its 95% CI from resampling whole blocks (test weeks or
replications) with replacement, plus p_worse = share of resamples where the candidate is worse.
Metric definitions come from bench.harness.tracks.<track>.METRICS when that module exists, else
every per-item array that shares a name with a baseline metric is compared as a plain mean (lower
is better). The reference is the baseline's variant named "baseline", else its first variant.
Writes bench/results/leaderboard_<track>.json (leaderboard_<track>.smoke.json in smoke mode).
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np

from . import RESULTS, rel
from . import metrics as M
from . import results as R


def outcome_arrays(track):
    """Per-item outcome arrays that must equal the baseline's (track module's OUTCOME_ARRAYS; y always)."""
    try:
        return tuple(getattr(importlib.import_module(f"bench.harness.tracks.{track}"), "OUTCOME_ARRAYS", ()))
    except ImportError:
        return ()


def _r(x):
    """9 significant digits (round(x, 6) zeroed tiny metrics such as the mc track's MSEs ~1e-6)."""
    return float(f"{x:.9g}")


def spec(track, base_arrays, base_metrics):
    try:
        mod = importlib.import_module(f"bench.harness.tracks.{track}")
        ms = getattr(mod, "METRICS", None)
        if ms:
            return list(ms), getattr(mod, "PRIMARY", ms[0].name)
    except ImportError:
        pass
    ms = [M.mean_of(k) for k in base_metrics if k in base_arrays and k not in ("item_id", "block")
          and np.issubdtype(np.asarray(base_arrays[k]).dtype, np.number)]
    return ms, ms[0].name if ms else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("track")
    ap.add_argument("--mode", default="full", choices=R.MODES)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    bpath = R.baseline_path(args.track, args.mode)
    if not os.path.exists(bpath):
        sys.exit(f"no baseline result {rel(bpath)} (run the _baseline_{args.track} job first)")
    base = R.read(bpath)
    ref = next((v for v in base["variants"] if v["name"] == "baseline"), base["variants"][0])
    ba = R.load_arrays(ref)
    ids = ba["item_id"].astype(str)
    blocks = ba["block"]
    ub, _ = M.block_index(blocks)
    C = M.bootstrap_counts(len(ub), args.B, args.seed)
    metrics, primary = spec(args.track, ba, ref["metrics"])
    metrics = [m for m in metrics if m.ok(ba)]
    outcomes = outcome_arrays(args.track)

    rows, skipped = [], []
    for res in R.find(args.track, args.mode):
        for v in res["variants"]:
            if res["key"] == base["key"] and v["name"] == ref["name"]:
                continue
            tag = f"{res['key']}/{v['name']}"
            try:
                a = R.align(R.load_arrays(v), ids, tag)
                if not np.array_equal(np.asarray(a["block"]), np.asarray(blocks)):
                    raise AssertionError(f"{tag}: block ids differ from the baseline's")
                for k in ("y",) + tuple(outcomes):
                    if k in a and k in ba and not np.allclose(np.asarray(a[k], float), np.asarray(ba[k], float),
                                                              rtol=0, atol=1e-6, equal_nan=True):
                        raise AssertionError(f"{tag}: outcomes {k} differ from the baseline's (data/ or the track's "
                                             "items changed between the runs; rerun the older one)")
            except (AssertionError, OSError, KeyError, ValueError) as e:
                skipped.append({"key": res["key"], "variant": v["name"], "reason": str(e)})
                continue
            out = {}
            for m in metrics:
                if m.ok(a):
                    out[m.name] = {k: (_r(x) if isinstance(x, float) else x)
                                   for k, x in M.paired_bootstrap(m, a, ba, blocks, C).items()}
            rows.append({"key": res["key"], "variant": v["name"], "n": int(len(ids)), "metrics": out,
                         "runtime_s": res.get("runtime_s"), "notes": res.get("notes", "")})
    if primary:  # best first: ascending diff, or descending when higher is better (lower=False)
        sgn = -1.0 if next((m.lower for m in metrics if m.name == primary), True) is False else 1.0
        rows.sort(key=lambda r: sgn * r["metrics"].get(primary, {}).get("diff", sgn * np.inf))
    base_vals = {m.name: _r(m.value(ba)) for m in metrics}
    board = {"track": args.track, "mode": args.mode, "baseline": rel(bpath), "baseline_variant": ref["name"],
             "primary": primary, "n_items": int(len(ids)), "n_blocks": int(len(ub)), "B": args.B,
             "baseline_metrics": base_vals, "rows": rows, "skipped": skipped,
             "lower_is_better": {m.name: m.lower for m in metrics}}
    sfx = ".smoke" if args.mode == "smoke" else ""
    out = os.path.join(RESULTS, f"leaderboard_{args.track}{sfx}.json")
    tmp = f"{out}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(board, f, indent=1)
    os.replace(tmp, out)

    show = ([primary] if primary in base_vals else []) + [m.name for m in metrics if m.lower is not None and m.name != primary]
    show = show[:4]
    print(f"{args.track} ({args.mode}): {len(ids)} items, {len(ub)} blocks, baseline {rel(bpath)}")
    print("  baseline " + "  ".join(f"{k}={base_vals[k]:.4f}" for k in show))
    for r in rows:
        cells = []
        for k in show:
            x = r["metrics"].get(k)
            if x:
                cells.append(f"{k} {x['diff']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]")
        print(f"  {r['key'] + '/' + r['variant']:<40} " + "  ".join(cells))
    for s in skipped:
        print(f"  SKIPPED {s['key']}/{s['variant']}: {s['reason']}")
    print(f"wrote {rel(out)}")
    return board


if __name__ == "__main__":
    main()
