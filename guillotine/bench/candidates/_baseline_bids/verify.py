"""Check the bids track and its baseline against the core model.

1. The track's unrepaired chances, with gain_core, are exactly bid_history's rows with budget > 0
   (BidModel's rows), in its order once sorted by raw_index.
2. Fit on those rows, BidModelRefit's parameters, p_bid and mu equal auction.BidModel(lg)'s (max |diff| = 0).
3. At every test leg L, the baseline's training rows equal BidModel's rows on a league cut off before L
   (lg.leg = L - 1), and fit in that order every parameter equals auction.BidModel on it (max |diff| = 0).
   In harness order (what the run uses) the logistic stops elsewhere on the a0 / a_log_budget ridge
   (parameters up to ~0.02 apart at leg 3, where nearly every budget is $200), exactly as BidModel itself
   does between processes (bid_history's row order follows str-hash set order); the test-leg p_bid agree
   within 1e-3 (seen: 3e-4) and mu exactly.
4. Point-in-time rosters: rosters_at(chop) equals the end-of-week matchup roster of every alive team in
   legs 3-4 (Sleeper's clean snapshots); elsewhere the differences are counted.
5. The exact independent-product top-bid pmf per group matches auction.top_rival_bids draws.
6. The guard: inside fit/predict no chance of the test leg or later is visible, rosters_at refuses later
   times, test frames carry no outcome column and data.table() stops at the horizon.

    .venv/bin/python -m bench.candidates._baseline_bids.verify
"""
import copy

from bench.harness import data as D
from bench.harness.tracks import bids as B

import numpy as np
from auction import BidModel, top_rival_bids

from .bidmodel import BidModelRefit

P_TOL = 1e-3   # harness vs bid_history row order: L-BFGS-B stops at a different point of the a0 / a_log_budget
               # ridge (budgets are nearly all $200 early), as BidModel itself does between processes


def params(m):
    return np.r_[m.a, [m.v[r] for r in sorted(m.v)], m.beta, [m.u[r] for r in sorted(m.u)], m.sigma, m.tau, m.r2]


def core_rows(c):
    """BidModel's rows from the track frame, in bid_history order, with its feature."""
    raw = c[~c["repaired"]].sort_values("raw_index")
    return raw.assign(gain=raw["gain_core"])


def fit_raw(raw, variant="logit"):
    return BidModelRefit(variant).fit_arrays(raw["rid"].to_numpy(), raw["gain"].to_numpy(float),
                                             raw["budget"].to_numpy(), raw["bid"].to_numpy(), raw["leg"].to_numpy())


def cut(lg, leg):
    """The league as bid_history would see it with leg - 1 the last leg."""
    lc = copy.copy(lg)
    lc.leg = leg - 1
    lc.tx = {w: x for w, x in lg.tx.items() if w < leg}
    lc.matchups = {w: x for w, x in lg.matchups.items() if w < leg}
    return lc


def main():
    lg = D.league()
    bm = BidModel(lg)
    c = B.chances()
    ok = True
    # 1-2. full league
    raw = core_rows(c)
    ref = [(r["run"], r["rid"], r["pid"], r["budget"], round(r["gain"], 12), r["bid"] or 0) for r in bm.rows]
    got = list(zip(raw["run"], raw["rid"], raw["pid"], raw["budget"], raw["gain"].round(12), raw["bid"]))
    same = ref == got
    print(f"rows: {len(c)} chances = {len(raw)} BidModel rows (identical in order: {same}) "
          f"+ {int(c['repaired'].sum())} it misses ({int((c['repaired'] & c['claim']).sum())} of them claims)")
    m = fit_raw(raw)
    dpar = np.abs(params(m) - params(bm)).max()
    dp = max(abs(float(x) - bm.p_bid(r, g, b)) for x, r, g, b in zip(m.p_logit(raw), raw["rid"], raw["gain"], raw["budget"]))
    dmu = max(abs(float(x) - bm.mu(r, g, b, l)) for x, r, g, b, l in
              zip(m.mu(raw), raw["rid"], raw["gain"], raw["budget"], raw["leg"]))
    print(f"full league: max|d params| {dpar:.3g}  max|d p_bid| {dp:.3g}  max|d mu| {dmu:.3g}")
    ok &= same and dpar == 0 and dp < 1e-12 and dmu < 1e-12
    # 3. per origin: the baseline's fit = BidModel on the league cut off before the test leg
    for leg in B.origins("full"):
        bl = BidModel(cut(lg, leg))
        tr = B.train_before(leg)
        rl = core_rows(c[c["leg"] < leg])
        same_l = [(r["run"], r["rid"], r["pid"], r["budget"], round(r["gain"], 12), r["bid"] or 0) for r in bl.rows] == \
            list(zip(rl["run"], rl["rid"], rl["pid"], rl["budget"], rl["gain"].round(12), rl["bid"]))
        d_exact = np.abs(params(fit_raw(rl)) - params(bl)).max()
        mh = BidModelRefit("baseline").fit(tr)
        d_harn = np.abs(params(mh) - params(bl)).max()
        te = B.test_frame(leg)
        dp_h = np.abs(mh.p_logit(te) - np.array([bl.p_bid(r, g, b) for r, g, b in zip(te["rid"], te["gain"], te["budget"])])).max()
        dmu_h = np.abs(mh.mu(te) - np.array([bl.mu(r, g, b, l) for r, g, b, l in
                                             zip(te["rid"], te["gain"], te["budget"], te["leg"])])).max()
        print(f"leg {leg}: {len(bl.rows)} BidModel rows, identical {same_l}; bid_history order max|d| {d_exact:.3g}; "
              f"harness order: max|d params| {d_harn:.2g}, on the test leg max|d p_bid| {dp_h:.2g}, max|d mu| {dmu_h:.2g}")
        ok &= same_l and d_exact == 0 and dp_h < P_TOL and dmu_h < 1e-9
    d = (c["gain"] - c["gain_core"]).abs()
    print("chances whose point-in-time gain differs from BidModel's, by leg:",
          c[~c["repaired"] & (d > 1e-9)].groupby("leg").size().to_dict())
    # 4. point-in-time rosters vs end-of-week matchup rosters
    for leg in range(1, lg.leg):
        chop = next(t for t in lg.tx[leg] if t["type"] == "chopped")
        ros = B.rosters_at(chop["status_updated"])
        alive = [mm for mm in lg.matchups[leg] if mm.get("players")]
        eq = sum(set(mm["players"]) == set(ros[mm["roster_id"]]) for mm in alive)
        print(f"leg {leg}: rosters_at(chop) == matchups players for {eq}/{len(alive)} teams")
        if leg in (3, 4):
            ok &= eq == len(alive)
    # 5. exact price pmf vs BidModel's sampler on the last leg's groups
    leg = int(c["leg"].max())
    t = c[c["leg"] == leg].reset_index(drop=True)
    b = fit_raw(raw, "baseline")
    out = b.predict(t)
    K = B.kmax()
    S = B.size_pmf(out, t["budget"].to_numpy(int), K)
    g = B.groups(t)
    gi = {k: i for i, k in enumerate(zip(g["run"], g["pid"]))}
    gidx = np.array([gi[k] for k in zip(t["run"], t["pid"])])
    P = B.price_pmf(B.joint_pmf(out["p_bid"], S), gidx, len(g))
    rng = np.random.default_rng(0)
    n = 200_000
    worst = 0.0
    for i, (run, pid) in enumerate(zip(g["run"], g["pid"])):
        rs = t[(t["run"] == run) & (t["pid"] == pid)]
        top = top_rival_bids(bm, dict(zip(rs["rid"], rs["budget"])), dict(zip(rs["rid"], rs["gain"])), leg, rng, n)
        emp = np.cumsum(np.bincount(top.astype(int), minlength=K + 1)[: K + 1]) / n
        worst = max(worst, np.abs(emp - np.cumsum(P[i])).max())
    tol = 4.5 * 0.5 / np.sqrt(n)
    print(f"leg {leg} groups {len(g)}: max|CDF exact - top_rival_bids MC| {worst:.4f} (MC tolerance {tol:.4f})")
    ok &= worst < tol
    # 6. guard
    leg = 5
    runs = sorted(B.test_frame(leg)["run"].unique())
    with B.guard(leg, int(runs[0]), leg):
        seen = int(B.chances()["leg"].max())
        tmax = int(D.table()["t"].max())
        try:
            B.rosters_at(int(runs[0]) + 1)
            refused = False
        except B.LeakError:
            refused = True
    tf = B.test_frame(leg)
    g_ok = seen < leg and refused and tmax == D.t_of(D.league().season, leg) and not set(B.TRAIN_ONLY) & set(tf)
    print(f"guard (leg {leg}): last visible leg {seen}, table stops at t={tmax}, later rosters refused {refused}, "
          f"test frame outcome-free {not set(B.TRAIN_ONLY) & set(tf)}")
    ok &= g_ok
    print("REPRODUCES auction.BidModel" if ok else "DIFFERS from auction.BidModel")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
