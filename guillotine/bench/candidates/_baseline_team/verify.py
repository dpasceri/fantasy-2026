"""Check the team baseline against the core code.

1. fit_copula on model.calibration_rows (their order) reproduces data/model.npz's loadings and game share.
2. The same fit on the harness's training rows (table order, tie draws land on other rows) for reference.
3. WeekDraws' sampler equals sim.Universe.paths' week-0 scores bit for bit (same tables, seed, players).
4. With data/model.npz's parameters (fit on all 2025 + 2026 weeks 1-7, so in-sample), the track's real
   leagues reproduce data/backtest.json's P(chopped team) up to Monte Carlo noise (different draws).

    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates._baseline_team.verify
"""
from bench.harness import data as D
from bench.harness.tracks import team as T

import numpy as np
from common import POS, J
from model import calibration_rows
from sim import CHUNK, Universe

from bench.candidates._baseline_player.ratio import fit_tables, ratio_grid
from .copula import SEED, CopulaSim, WeekDraws, fit_copula, sample_chunk


class _PM:
    def __init__(self, tables, loading, g):
        self.tables, self.loading, self.game_share = tables, loading, g

    def quantiles(self, pos, proj):
        return proj * ratio_grid(self.tables, [pos], [proj])[0]


def main():
    ok = True
    lg = D.league()
    z = np.load(f"{D.DATA}/model.npz")
    ref_a, ref_g = z["loadings"], float(z["game_share"])
    rows = calibration_rows(lg)
    a, g = fit_copula([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                      [r[3] for r in rows], [r[4] for r in rows])
    da, dg = np.abs(a - ref_a).max(), abs(g - ref_g)
    print(f"1. copula on calibration_rows ({len(rows)} rows): loadings {np.round(a, 4)} vs {np.round(ref_a, 4)}, "
          f"g {g:.4f} vs {ref_g:.4f}; max|diff| {max(da, dg):.3g}")
    ok &= max(da, dg) < 1e-9

    c = CopulaSim().fit(D.train_before((lg.season, lg.leg)))
    ha = np.array([c.loading[p] for p in POS])
    print(f"2. harness-order refit: loadings {np.round(ha, 4)}, g {c.g:.4f} (max|diff| vs model.npz "
          f"{max(np.abs(ha - ref_a).max(), abs(c.g - ref_g)):.3g}; tie draws differ by row order)")

    w = 7
    wk = D.week_frame(lg.season, w, items_only=False, pre=True)
    c.S = CHUNK
    W = WeekDraws(wk, c.tables, c.loading, c.g, CHUNK, (SEED, lg.season, w))
    U = Universe(lg, _PM(c.tables, c.loading, c.g), W.pids, [w])
    _, s_core, _ = next(U.paths(CHUNK, np.random.default_rng([SEED, lg.season, w, 0])))
    d3 = np.abs(s_core - W.score).max()
    print(f"3. sampler vs sim.Universe.paths, 2026 week {w}, {W.n} players x {CHUNK} draws: max|diff| {d3:.3g}")
    ok &= d3 == 0

    tables = {P: (z[f"{P}_centers"], z[f"{P}_tables"]) for P in POS}
    m = CopulaSim().set_params(tables, {p: float(ref_a[i]) for i, p in enumerate(POS)}, ref_g)
    bt = {r["week"]: r for r in J("backtest.json")["weeks"]}
    worst, ll = 0.0, []
    for o in [(lg.season, x) for x in range(1, lg.leg)]:
        L = [x for x in T.test_leagues(o, "full") if x["kind"] == "real"][0]
        v = T.view(T.leagues()["by_id"][L["league_id"]], outcome=True)
        p = m.predict(L)["p_chop"]
        pc, ref = p[v["chopped"]], bt[o[1]]["p_model_actual_starters"]
        se = np.sqrt(2 * pc * (1 - pc) / m.S)        # both are 96000-draw estimates on different draws
        worst = max(worst, abs(pc - ref) / max(se, 1e-9))
        ll.append(-np.log(pc))
        print(f"4. week {o[1]}: {L['n']} teams, P(chopped {L['owners'][v['chopped']]}) {pc:.4f} vs backtest.json "
              f"{ref:.3f} (MC se of the diff {se:.4f})")
    print(f"   log-loss {np.mean(ll):.3f} vs backtest.json {J('backtest.json')['logloss_model']:.3f}; "
          f"worst |diff| {worst:.1f} MC se (+ 3-decimal rounding)")
    ok &= worst < 4.0
    print("VERIFIED" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
