"""Rival FAAB bid model, fit from every bid this league has placed (winning and losing).

For each waiver run, each alive manager j and each player i that was on offer (chopped roster
or anything that drew a bid), compute
    gain_ji = average weekly lineup points j gains by adding i over the next 4 weeks
            = mean_w max(0, proj_iw - weakest starter i could replace in j's lineup)
and fit a hurdle model
    P(bid > 0)          = logistic(a0 + a1 * log(1 + gain) + a2 * log(budget) + v_j)
    log(bid) | bid > 0  = b0 + b1 * gain + b2 * log(budget) + b3 * leg + u_j + e,  e ~ N(0, s^2)
v_j (how often j bids) and u_j (how hard j bids) are manager effects shrunk toward 0:
v_j by a N(0, 1) prior, u_j by empirical Bayes. Bids are capped at the budget.
"""
from common import *  # noqa: F401,F403
from common import League, POS, best_lineup, eligible, np
from scipy.optimize import minimize

HORIZON = 4


def _minelig(lg, roster, w, val=None):
    val = val or (lambda p: lg.proj_pts(p, w))
    lu = best_lineup(lg.slots, [(val(p), p, lg.pos(p)) for p in roster])
    out = {}
    for P in POS:
        v = [val(p) if p else 0.0 for s, p in lu if eligible(s, P)]
        out[P] = min(v) if v else np.inf
    return out


def gain(lg, roster, pid, w0, w1, cache=None):
    """Avg weekly lineup gain from adding pid to roster over weeks w0..w1."""
    weeks = [w for w in range(w0, min(w1, lg.final_week) + 1)]
    if not weeks:
        return 0.0
    tot = 0.0
    for w in weeks:
        key = (id(roster), w)
        if cache is not None and key in cache:
            me = cache[key]
        else:
            me = _minelig(lg, roster, w)
            if cache is not None:
                cache[key] = me
        tot += max(0.0, lg.proj_pts(pid, w) - me[lg.pos(pid)])
    return tot / len(weeks)


def bid_history(lg):
    rows = []
    for leg in range(1, lg.leg + 1):
        tx = lg.tx[leg]
        chop = next((t for t in tx if t["type"] == "chopped"), None)
        chop_time = chop["status_updated"] if chop else None
        runs = sorted({t["status_updated"] for t in tx if t["type"] == "waiver"})
        ms = {m["roster_id"]: m for m in lg.matchups[leg]}
        for run in runs:
            claims = [t for t in tx if t["type"] == "waiver" and t["status_updated"] == run]
            post_chop = chop_time is not None and run > chop_time
            w0 = leg + 1 if post_chop else leg
            offered = {list(t["adds"])[0] for t in claims}
            if post_chop and chop.get("drops"):
                offered |= set(chop["drops"])
            offered = {p for p in offered if lg.pos(p) in POS}
            # budgets just before this run
            spent = {}
            for l2 in range(1, leg + 1):
                for t in lg.tx[l2]:
                    if t["status"] != "complete" or t["status_updated"] >= run:
                        continue
                    if t["type"] == "waiver":
                        r = t["roster_ids"][0]
                        spent[r] = spent.get(r, 0) + t["settings"]["waiver_bid"]
                    elif t["type"] == "trade":
                        for wb in t.get("waiver_budget") or []:
                            spent[wb["sender"]] = spent.get(wb["sender"], 0) + wb["amount"]
                            spent[wb["receiver"]] = spent.get(wb["receiver"], 0) - wb["amount"]
            bids = {}
            for t in claims:
                key = (t["roster_ids"][0], list(t["adds"])[0])
                bids[key] = max(bids.get(key, 0), t["settings"]["waiver_bid"])
            winners = {(t["roster_ids"][0], list(t["adds"])[0]) for t in claims if t["status"] == "complete"}
            chopped_rid = chop["roster_ids"][0] if (chop and post_chop) else None
            cache = {}
            for rid, m in ms.items():
                roster = m.get("players") or []
                if not roster or rid == chopped_rid:
                    continue
                budget = lg.budget - spent.get(rid, 0)
                for pid in offered:
                    if pid in roster:
                        continue
                    rows.append({
                        "leg": leg, "run": run, "rid": rid, "pid": pid, "budget": budget,
                        "gain": gain(lg, roster, pid, w0, w0 + HORIZON - 1, cache),
                        "bid": bids.get((rid, pid), 0) if (rid, pid) in bids else None,
                        "won": (rid, pid) in winners,
                    })
    return rows


class BidModel:
    def __init__(self, lg):
        self.lg = lg
        rows = [r for r in bid_history(lg) if r["budget"] > 0]
        self.rows = rows
        rids = sorted({r["rid"] for r in rows})
        ri = {rid: k for k, rid in enumerate(rids)}
        R = np.array([ri[r["rid"]] for r in rows])
        x1 = np.log1p(np.array([r["gain"] for r in rows]))
        x2 = np.log(np.array([r["budget"] for r in rows], float))
        y = np.array([1.0 if (r["bid"] or 0) > 0 else 0.0 for r in rows])

        def nll(th):
            v = th[3:]
            eta = th[0] + th[1] * x1 + th[2] * x2 + v[R]
            return np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * np.sum(v ** 2)

        th = minimize(nll, np.r_[-6.0, 1.5, 0.3, np.zeros(len(rids))], method="L-BFGS-B").x
        self.a = th[:3]
        self.v = {rid: float(th[3 + ri[rid]]) for rid in rids}

        pos = [r for r in rows if (r["bid"] or 0) > 0]
        X = np.column_stack([np.ones(len(pos)), [r["gain"] for r in pos], np.log([r["budget"] for r in pos]), [r["leg"] for r in pos]])
        Y = np.log([min(r["bid"], r["budget"]) for r in pos])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        prid = np.array([r["rid"] for r in pos])
        s2 = res.var()
        means = {rid: (res[prid == rid].mean(), int((prid == rid).sum())) for rid in set(prid)}
        tau2 = max(np.var([m for m, n in means.values()]) - s2 / np.mean([n for m, n in means.values()]), 0.0)
        self.u = {rid: (m * n / (n + s2 / tau2) if tau2 > 0 else 0.0) for rid, (m, n) in means.items()}
        self.n_bids = {rid: int(((prid == rid)).sum()) for rid in rids}
        self.n_offered = {rid: int((R == ri[rid]).sum()) for rid in rids}
        self.beta = beta
        self.sigma = float(np.sqrt(s2))
        self.tau = float(np.sqrt(tau2))
        self.r2 = float(1 - res.var() / Y.var())

    def p_bid(self, rid, g, budget):
        eta = self.a[0] + self.a[1] * np.log1p(g) + self.a[2] * np.log(max(budget, 1)) + self.v.get(rid, 0.0)
        return 1 / (1 + np.exp(-eta))

    def mu(self, rid, g, budget, leg):
        return self.beta[0] + self.beta[1] * g + self.beta[2] * np.log(max(budget, 1)) + self.beta[3] * leg + self.u.get(rid, 0.0)

    def median_bid(self, rid, g, budget, leg):
        return min(float(np.exp(self.mu(rid, g, budget, leg))), budget)

    def sample(self, rid, g, budget, leg, rng, size):
        """Rival bids in whole dollars (0 = no bid)."""
        if budget <= 0:
            return np.zeros(size)
        bid = rng.random(size) < self.p_bid(rid, g, budget)
        amt = np.minimum(np.round(np.exp(self.mu(rid, g, budget, leg) + self.sigma * rng.standard_normal(size))), budget)
        return np.where(bid, amt, 0.0)

    def summary(self):
        return {
            "participation": {k: round(float(x), 3) for k, x in zip(("a0", "a_log1p_gain", "a_log_budget"), self.a)},
            "size": {k: round(float(x), 3) for k, x in zip(("b0", "b_gain", "b_log_budget", "b_leg"), self.beta)}
            | {"sigma": round(self.sigma, 3), "tau": round(self.tau, 3), "r2": round(self.r2, 3)},
            "n_rows": len(self.rows), "n_bids": int(sum(1 for r in self.rows if (r["bid"] or 0) > 0)),
        }


def faab_history(lg):
    """League FAAB by completed week: spent on claims, burned (chopped team's unspent budget), left with alive teams.

    linear_left is the pace where every team spends its budget evenly over the season's auctions
    (one per chop, total_rosters - 2 of them): after k chops, (N - k) teams x B x (1 - k / (N - 2)).
    """
    N, B = lg.league["total_rosters"], lg.budget
    n_auctions = N - 2
    spent, dead, rows = {}, set(), []
    for leg in range(1, lg.leg + 1):
        tx = sorted(lg.tx[leg], key=lambda t: t["status_updated"])
        if not any(t["type"] == "chopped" for t in tx):
            continue  # week still in progress
        leg_spend, burned, top, buyers = 0, 0, 0, set()
        for t in tx:
            r = t["roster_ids"][0] if t.get("roster_ids") else None
            if t["type"] == "waiver" and t["status"] == "complete":
                b = t["settings"]["waiver_bid"]
                spent[r] = spent.get(r, 0) + b
                leg_spend += b
                top = max(top, b)
                if b > 0:
                    buyers.add(r)
            elif t["type"] == "trade" and t["status"] == "complete":
                for wb in t.get("waiver_budget") or []:  # FAAB can ride along in trades
                    spent[wb["sender"]] = spent.get(wb["sender"], 0) + wb["amount"]
                    spent[wb["receiver"]] = spent.get(wb["receiver"], 0) - wb["amount"]
            elif t["type"] == "chopped":
                burned += B - spent.get(r, 0)
                dead.add(r)
        k = len(dead)
        rows.append({"leg": leg, "spent": leg_spend, "burned": burned, "top_price": top, "buyers": len(buyers),
                     "alive": N - k, "left": sum(B - spent.get(r, 0) for r in lg.rosters if r not in dead),
                     "linear_left": round((N - k) * B * max(0.0, 1 - k / n_auctions), 1)})
    return rows


def top_rival_bids(bm, rivals, gains, leg, rng, n=200000):
    """Draws of the highest rival bid. rivals: {rid: budget}, gains: {rid: gain}."""
    top = np.zeros(n)
    for rid, budget in rivals.items():
        top = np.maximum(top, bm.sample(rid, gains[rid], budget, leg, rng, n))
    return top


def p_win_curve(top, max_bid):
    """P(my bid b beats the top rival bid) for b = 0..max_bid. Ties count half (waiver order)."""
    n = len(top)
    cnt = np.bincount(top.astype(int), minlength=max_bid + 2)
    lt = np.concatenate([[0], np.cumsum(cnt)])[: max_bid + 1] / n
    pw = lt + 0.5 * cnt[: max_bid + 1] / n
    pw[0] = 0.0  # a $0 claim never beats a real bid, and only wins uncontested players
    return pw


def price_backtest(lg, bm, seed=5):
    """Model's top-rival-bid distribution vs the price that actually cleared, per contested player."""
    rng = np.random.default_rng(seed)
    out = []
    by_run = {}
    for r in bm.rows:
        by_run.setdefault((r["run"], r["pid"]), []).append(r)
    for (run, pid), rs in by_run.items():
        bids = [r for r in rs if (r["bid"] or 0) > 0]
        if not bids or max(r["bid"] for r in bids) < 15:
            continue
        top = top_rival_bids(bm, {r["rid"]: r["budget"] for r in rs}, {r["rid"]: r["gain"] for r in rs}, rs[0]["leg"], rng, 40000)
        actual = max(r["bid"] for r in bids)
        out.append({"leg": rs[0]["leg"], "player": lg.label(pid), "actual": actual,
                    "p10": float(np.quantile(top, 0.1)), "median": float(np.median(top)), "p90": float(np.quantile(top, 0.9)),
                    "pct_rank": float((top < actual).mean() + 0.5 * (top == actual).mean())})
    return sorted(out, key=lambda r: (r["leg"], -r["actual"]))


if __name__ == "__main__":
    lg = League()
    bm = BidModel(lg)
    print(json.dumps(bm.summary(), indent=1))
    for rid in sorted(bm.v, key=lambda r: -bm.v[r]):
        print(f"  {lg.owner(rid):18s} bids-often {bm.v[rid]:+.2f}  bids-hard {bm.u.get(rid, 0):+.2f}  "
              f"({bm.n_bids[rid]} bids / {bm.n_offered[rid]} chances)  faab now {lg.faab(rid)}")
    bt = price_backtest(lg, bm)
    for r in bt:
        print(f"  leg {r['leg']} {r['player']:34s} cleared ${r['actual']:3d}  model p10/med/p90 {r['p10']:5.0f} {r['median']:5.0f} {r['p90']:5.0f}  pct {r['pct_rank']:.2f}")
    print("mean PIT", np.mean([r['pct_rank'] for r in bt]).round(3), " share inside p10-p90", np.mean([r['p10'] <= r['actual'] <= r['p90'] for r in bt]).round(3))
