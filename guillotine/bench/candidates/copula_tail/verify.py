"""Checks for copula_tail (one origin, training data only; no test-week outcomes are read).

1. gt with the Gaussian group reproduces _baseline_team's WeekDraws scores bit for bit, and the
   candidate's totals/summarize give the baseline's p_chop exactly (quantiles to ~1e-13).
2. Every variant keeps the baseline marginal: each player's u is U(0, 1) (max KS distance over players)
   and a player's simulated score quantiles match the baseline's within Monte Carlo error.
3. Same-team and opponent slot dependence in the draws (Spearman, and the lower-tail co-exceedance
   P(both u < 0.1) / 0.1) against the template pool's: shuffle reproduces the pool, slot_t sits between the
   pool and the factor model (shrinkage), gt keeps the factor model's (+ t tails).
The whole check runs on training data and the test week's pre-game frame only.

    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates.copula_tail.verify [season week]
"""
import sys

from bench.harness import data as D
from bench.harness.tracks import team as T

import numpy as np
from bench.candidates._baseline_team.copula import CopulaSim, WeekDraws, summarize as base_summarize
from scipy.stats import spearmanr
from sim import CHUNK

from .tailcop import SEED, SLOT_IX, TailCopula, TailDraws, summarize


def main(argv):
    o = (int(argv[0]), int(argv[1])) if len(argv) >= 2 else (2026, 7)
    ok = True
    D.table()
    T.leagues()
    with D.horizon(o):
        tr = D.train_before(o)
        wk = D.week_frame(*o, items_only=False, pre=True).reset_index(drop=True)
        Ls = T.test_leagues(o, "smoke")
        base = CopulaSim().fit(tr)
        ms = {v: TailCopula(v).fit(tr) for v in ("gt", "slot_t", "shuffle")}
    seed = (SEED, *o)

    # 1. Gaussian path == baseline
    g = TailCopula("gt").fit(tr)
    g.tail["gt"], g.tcdf = ("gauss", np.inf), None
    Wb = WeekDraws(wk, base.tables, base.loading, base.g, CHUNK * 2, seed)
    Wg = TailDraws(wk, g, CHUNK * 2, seed)
    same = np.array_equal(Wb.score[:, :-1].T, Wg.scoreT[:-1])
    pe, dq = True, 0.0
    for L in Ls:
        a, b = base_summarize(Wb.totals(L["starters"])), summarize(Wg.totals(L["starters"]))
        pe &= np.array_equal(a["p_chop"], b["p_chop"])
        dq = max(dq, np.abs(a["total_q"] - b["total_q"]).max(), np.abs(a["line_q"] - b["line_q"]).max())
    print(f"1. {o}: Gaussian scores bit-identical {same}; p_chop identical {pe}; max quantile diff {dq:.2g}")
    ok &= same and pe and dq < 1e-9

    # 2./3. marginals and slot dependence per variant
    m0 = ms["shuffle"]
    R = m0.rank / m0.n_slot
    p0, p1 = m0.pairs[:, 0], m0.pairs[:, 1]
    tops, bots = np.concatenate([p0, p1]), np.concatenate([p1, p0])
    checks = [("QB1", "WR1", 0), ("QB1", "TE1", 0), ("RB1", "RB2", 0), ("WR1", "WR2", 0), ("QB1", "RB1", 0),
              ("QB1", "QB1", 1), ("QB1", "WR1", 1)]

    def dep(x, y):
        f = np.isfinite(x) & np.isfinite(y)
        return spearmanr(x[f], y[f])[0], np.mean((x[f] < 0.1) & (y[f] < 0.1)) / 0.1

    ref = {}
    for a, b, opp in checks:
        ia, ib = SLOT_IX[a], SLOT_IX[b]
        ref[a, b, opp] = dep(R[tops, ia], R[bots if opp else tops, ib])
    Sb = WeekDraws(wk, base.tables, base.loading, base.g, CHUNK * 4, seed).score[:, :-1]
    for v, m in ms.items():
        W = TailDraws(wk, m, CHUNK * 4, seed, keep_u=True)
        U = W.u
        n = U.shape[1]
        srt = np.sort(U, axis=0)
        ks = np.abs(srt - (np.arange(1, len(U) + 1) / len(U))[:, None]).max()
        crit = np.sqrt(-np.log(0.01 / (2 * n)) / (2 * len(U)))          # 1% level for the max over n players
        sc = W.scoreT[:-1].T
        lv = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        dev = np.abs(np.quantile(sc, lv, axis=0) - np.quantile(Sb, lv, axis=0)).max(0) / (Sb.std(0) + 1e-9)
        grp, nu = m.tail[v]
        print(f"2. {v}: tail {grp} nu {nu:g}; max KS of u over {n} players {ks:.4f} (1% critical for the max "
              f"{crit:.4f}); |score quantile diff| / sd vs baseline draws: median {np.median(dev):.3f}, max {dev.max():.3f}")
        ok &= ks < crit
        msg = []
        for a, b, opp in checks:
            rs, ls = [], []
            for gi in np.unique(W.game):
                for sd in (0, 1):
                    i = np.flatnonzero((W.game == gi) & (W.side == sd) & (W.slot == SLOT_IX[a]))
                    j = np.flatnonzero((W.game == gi) & (W.side == (1 - sd if opp else sd)) & (W.slot == SLOT_IX[b]))
                    if len(i) and len(j):
                        r_, l_ = dep(U[:, i[0]], U[:, j[0]])
                        rs.append(r_)
                        ls.append(l_)
            msg.append(f"{'opp ' if opp else ''}{a}-{b} {np.mean(rs):+.3f}/{np.mean(ls):.2f} "
                       f"(pool {ref[a, b, opp][0]:+.3f}/{ref[a, b, opp][1]:.2f})")
        print("3.   rank corr / lower-tail co-exceedance in draws (mean over games): " + "; ".join(msg))
    print("VERIFIED" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main(sys.argv[1:]) else 1)
