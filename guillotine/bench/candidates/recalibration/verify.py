"""Checks for the recalibration candidate.

1. Draws.totals equals the baseline's WeekDraws.totals bit for bit (transposed layout).
2. With identity maps, Recal.predict equals CopulaSim.predict: total_q and line_q bit for bit, p_chop
   equal wherever the baseline's p > 0 (zero-count teams get the FLOOR before renormalizing).
3. The maps recover known miscalibration on synthetic data: tempered chop outcomes (beta = 0.7, eps = 0.03)
   and PITs of a too-narrow forecast (true sd 1.3x the forecast's); n_eff collapses for clustered PITs.
4. The calibration set of an origin holds only weeks strictly before it.
5. The per-run calibration-set cache (RECAL_CACHE) returns exactly the uncached set.

    bench/cpuq -n 1 -- .venv/bin/python -m bench.candidates.recalibration.verify
"""
import os
import tempfile

from bench.harness import data as D
from bench.harness.tracks import team as T

import numpy as np
from bench.candidates._baseline_team.copula import S_DRAWS, SEED, CopulaSim
from scipy.special import ndtr

from .recal import ChopMap, Draws, LevelMap, Recal, cached_calibration_set, calibration_set, n_eff


def synthetic_chop(rng, n_leagues=1500, beta=0.7, eps=0.03):
    lp, cidx, Ns, off = [], [], [], 0
    for _ in range(n_leagues):
        N = int(rng.integers(10, 18))
        p = rng.dirichlet(np.full(N, 0.4))
        p = np.maximum(p, 1e-5)
        p /= p.sum()
        q = p ** beta
        q = (1 - eps) * q / q.sum() + eps / N
        lp.append(np.log(p))
        cidx.append(off + int(rng.choice(N, p=q)))
        Ns.append(N)
        off += N
    N = np.array(Ns)
    o = np.r_[0, np.cumsum(N)[:-1]]
    lpc = np.concatenate(lp)
    return {"lp": lpc, "zp": np.zeros_like(lpc), "dsd": np.zeros_like(lpc), "n_team": np.repeat(N, N).astype(float),
            "N": N.astype(float), "off": o, "grp": np.repeat(np.arange(len(N)), N), "cidx": np.array(cidx),
            "t_league": np.arange(len(N)) % 30}


def main():
    ok = True
    origin = (2026, 3)
    train = D.train_before(origin)
    tl = T.train_leagues(origin)
    test = T.test_leagues(origin, "smoke")
    with D.horizon(origin):
        base = CopulaSim().fit(train, tl)
        L = test[0]
        wd = base.draws(L)
        dr = Draws(L["week_df"], base.tables, base.loading, base.g, S_DRAWS, (SEED, L["season"], L["week"]))
        same = all(np.array_equal(dr.totals(x["starters"]), wd.totals(x["starters"]).T) for x in test)
        print(f"1. Draws.totals == WeekDraws.totals.T on {len(test)} leagues: {same}")
        ok &= same

        rc = Recal("temp")
        rc.base, rc.K, rc.R = base, 1, 2          # identity maps (no fit)
        d_q = d_l = d_p = 0.0
        for x in test:
            a, b = rc.predict(x), base.predict(x)
            d_q = max(d_q, np.abs(a["total_q"] - b["total_q"]).max())
            d_l = max(d_l, np.abs(a["line_q"] - b["line_q"]).max())
            pos = b["p_chop"] > 0
            pa = a["p_chop"] / a["p_chop"][pos].sum() * b["p_chop"][pos].sum()
            d_p = max(d_p, np.abs(pa[pos] - b["p_chop"][pos]).max())
        print(f"2. identity maps vs baseline: max|d total_q| {d_q:.3g}, max|d line_q| {d_l:.3g}, max|d p_chop| (p>0) {d_p:.3g}")
        ok &= d_q == 0 and d_l == 0 and d_p < 1e-12

        rng = np.random.default_rng(5)
        cal = synthetic_chop(rng)
        t1 = ChopMap("temp").fit(cal)
        mx = ChopMap("mix").fit(cal)
        cl = ChopMap("clogit").fit(cal)
        print(f"3a. synthetic chop (beta 0.7, eps 0.03): temp beta {t1.theta[0]:.3f}; mix beta {mx.theta[0]:.3f} "
              f"eps {mx.eps:.4f} (deff {mx.info['deff']}); clogit theta {np.round(cl.theta, 3)} eps {cl.eps:.4f}")
        ok &= abs(mx.theta[0] - 0.7) < 0.08 and abs(mx.eps - 0.03) < 0.03
        z = rng.standard_normal(20000) * 1.3 + 0.2
        u = ndtr(z)                                # PITs of a N(0, 1) forecast for N(0.2, 1.3^2) outcomes
        wk = np.arange(len(u)) % 40                 # 40 independent "weeks": ICC ~ 0, n_eff ~ n
        pr = LevelMap("probit").fit(u, wk)
        dc = LevelMap("dcp").fit(u, wk)
        want = ndtr(0.2 + 1.3 * np.array([-1.2816, 0.0, 1.2816]))
        print(f"3b. synthetic PITs (shift 0.2, sd 1.3): probit a {pr.info['a']} b {pr.info['b']}; "
              f"l(0.1/0.5/0.9) probit {pr.info['l10']}/{pr.info['l50']}/{pr.info['l90']}, "
              f"dcp {dc.info['l10']}/{dc.info['l50']}/{dc.info['l90']}, true {np.round(want, 4).tolist()}; "
              f"n_eff {pr.info['n_eff']} of {len(u)}")
        zc = np.repeat(rng.standard_normal(10), 100) + 0.3 * rng.standard_normal(1000)
        ne, icc = n_eff(zc, np.repeat(np.arange(10), 100))
        print(f"3c. clustered z (10 weeks x 100, ICC ~0.92): icc {icc:.3f}, n_eff {ne:.1f}")
        ok &= ne < 12
        ok &= abs(pr.info["b"] - 1.3) < 0.05 and np.abs(np.array([dc.info["l10"], dc.info["l50"], dc.info["l90"]]) - want).max() < 0.02

        cs = calibration_set(train, tl, K=2, R=4)
        t0 = D.t_of(origin)
        print(f"4. calibration weeks {cs['weeks']} (origin t={t0}), {len(cs['N'])} leagues: all < origin "
              f"{max(cs['weeks']) < t0}")
        ok &= max(cs["weeks"]) < t0

        with tempfile.TemporaryDirectory() as td:
            os.environ["RECAL_CACHE"] = td
            try:
                c1 = cached_calibration_set(train, tl, K=2, R=4)     # builds + writes
                c2 = cached_calibration_set(train, tl, K=2, R=4)     # loads
                c3 = cached_calibration_set(train, tl, K=1, R=4)     # other key: builds
            finally:
                os.environ.pop("RECAL_CACHE", None)
            nfiles = len([f for f in os.listdir(td) if f.endswith(".npz")])
        same = all(c1.keys() == cs.keys() == c2.keys() and all(np.array_equal(np.asarray(c[k]), np.asarray(cs[k]))
                                                              and np.asarray(c[k]).dtype == np.asarray(cs[k]).dtype
                                                              for k in cs) for c in (c1, c2))
        same &= c2["weeks"] == cs["weeks"] and c3["weeks"] == cs["weeks"][-1:] and nfiles == 2
        print(f"5. cached calibration set == uncached (built and loaded, same dtypes; a new key rebuilds): {same}")
        ok &= same
    print("OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
