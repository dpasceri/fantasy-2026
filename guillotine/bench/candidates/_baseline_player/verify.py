"""Check the baseline against the core model: fit on model.py's rows (everything before the current
leg) and compare centers/tables with data/model.npz and predictions with PlayerModel.quantiles.

    .venv/bin/python -m bench.candidates._baseline_player.verify
"""
from bench.harness import data as D

import numpy as np
from common import POS
from model import PlayerModel, Q, calibration_rows

from .ratio import EmpiricalRatio, grid_to_levels, ratio_grid


def main():
    lg = D.league()
    tr = D.train_before((lg.season, lg.leg))
    rows = calibration_rows(lg)
    n_items = int(tr["item"].sum())
    print(f"rows: harness {n_items}  model.calibration_rows {len(rows)}")
    # same rows, same values
    a = sorted((r[0], round(r[1], 9), round(r[2], 9)) for r in rows)
    d = tr[tr["item"]]
    b = sorted(zip(d["pos"], d["proj"].round(9), d["actual"].round(9)))
    print("identical (pos, proj, actual) multiset:", a == b)
    cand = EmpiricalRatio().fit(tr)
    z = np.load(f"{D.DATA}/model.npz")
    worst = 0.0
    for P in POS:
        c, t = cand.tables[P]
        dc = np.abs(c - z[f"{P}_centers"]).max() if c.shape == z[f"{P}_centers"].shape else np.inf
        dt = np.abs(t - z[f"{P}_tables"]).max() if t.shape == z[f"{P}_tables"].shape else np.inf
        worst = max(worst, dc, dt)
        print(f"{P}: buckets {len(c)} vs {len(z[f'{P}_centers'])}  max|d centers| {dc:.3g}  max|d tables| {dt:.3g}")
    # predictions vs PlayerModel.quantiles on next week's items
    pm = PlayerModel()
    w = D.week_frame(lg.season, lg.leg)
    R = w["proj"].to_numpy()[:, None] * ratio_grid(cand.tables, w["pos"].to_numpy(), w["proj"].to_numpy())
    ref = np.stack([pm.quantiles(p, x) for p, x in zip(w["pos"], w["proj"])])
    dq = np.abs(R - ref).max()
    lv = np.linspace(0.01, 0.99, 99)
    print(f"week {lg.leg} items {len(w)}: max|grid quantiles - PlayerModel.quantiles| {dq:.3g}; "
          f"harness levels shape {grid_to_levels(R, lv).shape}, Q={Q}")
    ok = worst < 1e-9 and dq < 1e-9
    print("REPRODUCES data/model.npz" if ok else "DIFFERS from data/model.npz")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
