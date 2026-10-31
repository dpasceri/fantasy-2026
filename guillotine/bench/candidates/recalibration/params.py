"""Collect the per-origin map parameters the Recal tasks dumped into one table per mode.

    python -m bench.candidates.recalibration.params smoke|full
Reads bench/results/recalibration/params_<mode>/<variant>__<t>.json, writes
bench/results/recalibration/params_<mode>.json and prints one line per (variant, origin).
"""
import glob
import json
import os
import sys

from bench.harness import RESULTS, rel


def main(mode):
    d = os.path.join(RESULTS, "recalibration")
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, f"params_{mode}", "*.json")))]
    rows.sort(key=lambda r: (r["variant"], r["origin_t"]))
    out = os.path.join(d, f"params_{mode}.json")
    with open(out, "w") as f:
        json.dump({"mode": mode, "rows": rows}, f, indent=1)
    for r in rows:
        c, t, ln = r["chop"], r["tot"], r["line"]
        q = "".join(f"  {k} l(.1/.5/.9) {m['l10']:.3f}/{m['l50']:.3f}/{m['l90']:.3f}" for k, m in (("tot", t), ("line", ln)) if m)
        print(f"[recal] {r['variant']:<10} {r['origin']} cal {r['cal_leagues']:>4} leagues: theta {c['theta']} eps {c['eps']}"
              f"  cal logloss {c['cal_logloss_identity']:.4f} -> {c['cal_logloss_fit']:.4f}{q}")
    print(f"[recal] wrote {rel(out)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
