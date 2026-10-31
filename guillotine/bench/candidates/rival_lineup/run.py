"""Runner for rival_lineup: lineup diagnostics, the team track for every variant, paired real-league tests.

    python -m bench.candidates.rival_lineup.run --mode smoke|full [--jobs N]

Writes bench/results/rival_lineup[.smoke].json via bench.harness.tracks.team.run, adding
"lineup_describe" and "lineup_eval" (lineup_eval.py) and "paired": week-block bootstrap differences
on the real-league metrics for roster_rum - roster_max (the question this candidate asks),
zero_start - baseline, and each roster variant - baseline (the as-played oracle), when the baseline
result of the same mode exists. Per-team-week lineup rows go to rival_lineup.lineup_rows[.smoke].npz.
"""
import argparse
import json
import os

from bench import harness
from bench.harness import RESULTS, rel
from bench.harness import metrics as M
from bench.harness import results as R
from bench.harness.tracks import team as TM
from bench.harness.tracks.player import factory

import numpy as np
from bench.candidates.rival_lineup import lineup_eval as LE
from bench.candidates.rival_lineup.candidate import VARIANTS, RivalLineup

KEY = "rival_lineup"
REAL = ("real_logloss", "real_brier", "real_rank", "real_crps_total", "real_crps_line")
VIRT = ("logloss", "crps_total", "crps_line")
NOTES = ("Opponent lineup behaviour on the real league only (virtual leagues have no managers: every variant "
         "= baseline there, bit for bit). zero_start: as-played starters, zero-proj starters get a Beta-shrunk "
         "P(play) + marginal draw. roster_max / roster_rum: lineups unknown, built from rosters; max-projection vs "
         "probit lineup model with status-quo inertia and partially pooled manager sloppiness. Compare roster_rum "
         "with roster_max ('paired'); the as-played baseline is their oracle.")


def paired(path, mode):
    res = R.read(path)
    arr = {v["name"]: R.load_arrays(v) for v in res["variants"]}
    bpath = R.baseline_path(TM.TRACK, mode)
    if os.path.exists(bpath):
        b = R.read(bpath)
        ref = next((v for v in b["variants"] if v["name"] == "baseline"), b["variants"][0])
        arr["baseline"] = R.align(R.load_arrays(ref), arr[VARIANTS[0]]["item_id"].astype(str), "baseline")
    ms = {m.name: m for m in TM.METRICS}
    blocks = arr[VARIANTS[0]]["block"]
    C = M.bootstrap_counts(len(M.block_index(blocks)[0]), 2000, 0)
    pairs = [("roster_rum", "roster_max"), ("zero_start", "baseline"), ("roster_max", "baseline"),
             ("roster_rum", "baseline")]
    out = {}
    for a, b in pairs:
        if a not in arr or b not in arr:
            continue
        out[f"{a}-{b}"] = {k: {x: (round(y, 5) if isinstance(y, float) else y)
                               for x, y in M.paired_bootstrap(ms[k], arr[a], arr[b], blocks, C).items()}
                           for k in REAL + VIRT}
    res["paired"] = out
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, path)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=R.MODES, default="smoke")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    TM.leagues()
    desc = LE.describe()
    lev = LE.evaluate(args.mode)
    rows = lev.pop("rows", [])
    sfx = ".smoke" if args.mode == "smoke" else ""
    if rows:
        np.savez_compressed(os.path.join(RESULTS, f"{KEY}.lineup_rows{sfx}.npz"),
                            **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"[rival_lineup] describe: {json.dumps({k: v for k, v in desc.items() if k != 'status_quo'})}", flush=True)
    for met in ("crps_proj", "nll", "zero_brier"):
        if f"{met}_diff" in lev:
            print(f"[rival_lineup] lineup {met}: " + "  ".join(
                f"{k} {v['diff']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]" for k, v in lev[f"{met}_diff"].items()), flush=True)
    path = TM.run(KEY, [(v, factory(RivalLineup, v)) for v in VARIANTS], args.mode, args.jobs, NOTES,
                  extra={"lineup_describe": desc, "lineup_eval": lev})
    pr = paired(path, args.mode)
    for k, v in pr.items():
        print(f"[rival_lineup] paired {k}: " + "  ".join(
            f"{m} {x['diff']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]" for m, x in v.items() if m in REAL), flush=True)
    bad = R.validate(R.read(path))
    assert not bad, bad
    print(f"[rival_lineup] wrote {rel(path)}", flush=True)


if __name__ == "__main__":
    main()
