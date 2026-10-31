"""Check the analytic chop model against the team baseline's Monte Carlo on the same fitted model.

Both are fit on the same origin's training rows (identical tables and copula), then for every smoke
test league of that week: per-team mean/sd/quantiles, P(chopped) and chop-line quantiles from
_baseline_team's 96000 common-random-number draws vs the cf and conv computations here (cross-team
factor on and off). Also: lattice moments vs exact moments, Mehler covariance vs simulated covariance,
and timings. Pre-game data and the fitted model only; test-week outcomes are never touched.

    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates.evt_order_stats.verify [season week]
"""
import sys
import time

from bench.harness import data as D
from bench.harness.tracks import team as T

import numpy as np
from bench.candidates._baseline_team.copula import CopulaSim
from bench.harness.metrics import LEVELS

from .analytic import CF_CAP, XLO, H, OrderStats, PlayerWeek, inv_cdf, one_factor, order_stats


def kl(p, q):
    p, q = np.maximum(p, 1e-12), np.maximum(q, 1e-12)
    return float(np.sum(p * np.log(p / q)))


def main(origin=(2026, 3)):
    train, tl = D.train_before(origin), T.train_leagues(origin)
    test = T.test_leagues(origin, "smoke")
    with D.horizon(origin):
        t0 = time.time()
        mc = CopulaSim().fit(train, tl)
        t1 = time.time()
        an = OrderStats("conv").fit(train, tl)
        t2 = time.time()
        print(f"fit: baseline {t1 - t0:.2f}s, analytic {t2 - t1:.2f}s; loadings {an.loading} g {an.g:.3f}")
        W = an.week(test[0])
        t3 = time.time()
        print(f"player week: {W.n} players in {t3 - t2:.2f}s")
        # lattice moments vs exact moments of the piecewise-linear quantile function
        pm = np.fft.irfft(W.spec0, W.NF)[:, :W.NP]
        x = XLO + H * np.arange(W.NP)
        lm = pm @ x
        lv = pm @ x ** 2 - lm ** 2
        print(f"player lattice (top {x[-1]:.0f}, max quantile {W.q.max():.0f}) vs exact moments: "
              f"max |mean diff| {np.abs(lm - W.mu).max():.4f}, max |var diff| {np.abs(lv - W.var).max():.3f} "
              f"(median var {np.median(W.var):.1f}; binning adds H^2/12 = {H * H / 12:.4f})")
        Wc = PlayerWeek(test[0]["week_df"], an.tables, an.loading, an.g, test[0]["slots"], CF_CAP)
        # MC once per week
        t4 = time.time()
        draws = mc.draws(test[0])
        t5 = time.time()
        print(f"baseline MC draws: {t5 - t4:.2f}s")
        rows = []
        tm = {"cf": 0.0, "conv": 0.0}
        for L in test:
            tot = draws.totals(L["starters"])
            pmc = np.bincount(tot.argmin(1), minlength=L["n"]) / len(tot)
            qmc = np.quantile(tot, LEVELS, axis=0).T
            lmc = np.quantile(tot.min(1), LEVELS)
            Cmc = np.cov(tot.T)
            out = {}
            for meth in ("cf", "conv"):
                s = time.time()
                mu, C, F, _ = (Wc if meth == "cf" else W).league(L["starters"], meth)
                lam = one_factor(C) if meth == "conv" else None
                p, Fm = order_stats(F, lam)
                tm[meth] += time.time() - s
                out[meth] = (mu, C, F, p, Fm, lam)
            mu, C, F, p, Fm, lam = out["conv"]
            p0, _ = order_stats(F, None)
            q = inv_cdf(F)
            mid = slice(4, 95)
            sd_mc = np.sqrt(np.diag(Cmc))
            off = ~np.eye(L["n"], dtype=bool)
            rows.append([np.abs(mu - tot.mean(0)).max(), np.abs(np.sqrt(np.diag(C)) - sd_mc).max(),
                         np.abs(q[:, mid] - qmc[:, mid]).max(), np.abs(inv_cdf(Fm)[0][mid] - lmc[mid]).max(),
                         np.abs(p - pmc).max(), kl(pmc, p), kl(pmc, p0),
                         np.abs(out["cf"][3] - pmc).max(), kl(pmc, out["cf"][3]),
                         np.abs(inv_cdf(out["cf"][4])[0][mid] - lmc[mid]).max(),
                         (C / np.outer(sd_mc, sd_mc))[off].mean(), (Cmc / np.outer(sd_mc, sd_mc))[off].mean(),
                         lam.mean(), (q[:, :5] - qmc[:, :5]).mean(), (inv_cdf(out["cf"][2])[:, :5] - qmc[:, :5]).mean()])
        r = np.array(rows)
        names = ["|mean| conv-MC", "|sd| conv-MC", "|q05-95| conv-MC", "|line q05-95| conv-MC",
                 "|p| conv-MC", "KL(MC||conv)", "KL(MC||conv, no factor)", "|p| cf-MC", "KL(MC||cf)",
                 "|line q05-95| cf-MC", "mean team corr (Mehler)", "mean team corr (MC)", "mean lambda",
                 "q01-05 conv-MC (signed)", "q01-05 cf-MC (signed)"]
        for k, nm in enumerate(names):
            print(f"  {nm:28s} max {r[:, k].max():.5f}  mean {r[:, k].mean():.5f}")
        print(f"MC p_chop standard error at p=0.1: {np.sqrt(0.09 / len(tot)):.5f}")
        # quadrature refinement: double every Gauss-Hermite node count
        from . import analytic as A
        base = [order_stats(W.league(L["starters"], "conv")[2], None)[0] for L in test]
        saved = (A.N_ONE, A.N_G2, A.N_SIDE)
        A.N_ONE, A.N_G2, A.N_SIDE = 2 * A.N_ONE, 2 * A.N_G2, 2 * A.N_SIDE
        W._gcache.clear()
        fine = [order_stats(W.league(L["starters"], "conv")[2], None)[0] for L in test]
        A.N_ONE, A.N_G2, A.N_SIDE = saved
        W._gcache.clear()
        print(f"GH nodes doubled: max |p diff| {max(np.abs(a - b).max() for a, b in zip(base, fine)):.2e}, "
              f"max KL {max(kl(b, a) for a, b in zip(base, fine)):.2e}")
        print(f"time per league: cf {1e3 * tm['cf'] / len(test):.1f} ms, conv {1e3 * tm['conv'] / len(test):.1f} ms; "
              f"MC draws {t5 - t4:.2f}s per week + totals")


if __name__ == "__main__":
    a = sys.argv[1:]
    main((int(a[0]), int(a[1])) if len(a) == 2 else (2026, 3))
