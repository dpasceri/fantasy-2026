"""Post-step: split the paired comparison vs the track baseline into the 2025 blocks (also used for
design: the dev ablations ran on 2025 weeks 5-18) and the 2026 blocks (the cleaner holdout), and
store it as the extra field "holdout" in bench/results/hier_bayes[.smoke].json.

    python -m bench.candidates.hier_bayes.holdout smoke|full
"""
import json
import os
import sys

from bench import harness  # noqa: F401
import numpy as np
from bench.harness import metrics as M
from bench.harness import results as R

KEY, TRACK = "hier_bayes", "player"
T2026 = 18                                         # t of 2026 week 1
MET = [M.mean_of("crps"), M.mean_of("pinball")]


def main(mode):
    path = R.json_path(KEY, mode)
    res = R.read(path)
    base = R.read(R.baseline_path(TRACK, mode))
    ref = next((v for v in base["variants"] if v["name"] == "baseline"), base["variants"][0])
    ba = R.load_arrays(ref)
    ids = ba["item_id"].astype(str)
    out = {}
    for v in res["variants"]:
        a = R.align(R.load_arrays(v), ids, f"{KEY}/{v['name']}")
        assert np.array_equal(a["block"], ba["block"]) and np.allclose(a["y"], ba["y"], rtol=0, atol=1e-6), \
            "items, blocks or outcomes differ from the baseline's"
        rows = {}
        for split, m in (("2025_design", ba["block"] < T2026), ("2026_holdout", ba["block"] >= T2026)):
            if not m.any():
                continue
            sub = lambda x: {k: np.asarray(z)[m] for k, z in x.items()}
            rows[split] = {"n_items": int(m.sum()), "n_blocks": int(len(np.unique(ba["block"][m])))}
            for met in MET:
                r = M.paired_bootstrap(met, sub(a), sub(ba), ba["block"][m])
                rows[split][met.name] = {k: (float(f"{x:.6g}") if isinstance(x, float) else x) for k, x in r.items()}
        out[v["name"]] = rows
    res["holdout"] = out
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, path)
    assert not R.validate(R.read(path))
    for name, rows in out.items():
        print(f"[holdout] {KEY}/{name}: " + "  ".join(
            f"{s} crps {r['crps']['diff']:+.4f} [{r['crps']['lo']:+.4f},{r['crps']['hi']:+.4f}] ({r['n_blocks']} blk)"
            for s, r in rows.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
