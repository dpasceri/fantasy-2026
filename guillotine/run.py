"""Run every sim and write out/results.json, then render out/cheatsheet.html.

    python fetch.py && python run.py          # full run, ~5-6 min on 8 threads
    python run.py --quick                     # 1/8 of the sims, for iterating
    GSIM_THREADS=4 python run.py              # fewer cores

Before the chop it prices hypothetical auctions for the likeliest chop victims; once a chop has
happened and its roster is on waivers, it prices that actual auction instead.
"""
import argparse
import datetime as dt
import time

from common import *  # noqa: F401,F403
from common import OUT, POS, THREADS, J, League, best_lineup, np
import dynamic
from auction import BidModel, faab_history, gain, p_win_curve, price_backtest, top_rival_bids
from model import PlayerModel
from model import fit as fit_model
from sim import SeasonPlan, Universe, enumerate_lineups, run_season, run_week_lineups

BID_GRID = (0, 10, 20, 30, 45, 60, 80, 100, 125, 150, 165)  # in a $200 league; scaled to the league's FAAB
STRESS = (0.7, 0.3)  # rivals bid ~2x as often (logit +0.7) and ~35% higher (log +0.3)
# my bid multiplier, fixed: the sweep's cells tie within noise and its argmax chased that noise; 0.75x all
# season held up best on fresh seeds (bench/RESULTS.md). The sweep stays on the sheet for context.
POLICY = {"m_early": 0.75, "m_late": 0.75, "switch_week": 12}
# ...unless another style beats it by this many standard errors in this league (unpaired, so conservative; it also
# guards against picking the luckiest of 16 cells). Tuned leagues tie within noise and keep 0.75x.
SWITCH_Z = 3.0


def fa_candidates(lg, exclude, per_pos=None):
    per_pos = per_pos or {"QB": 3, "RB": 10, "WR": 10, "TE": 6}
    rostered = lg.rostered() | set(exclude)
    pool = {}
    for pid, inf in lg.info.items():
        if pid in rostered or inf["pos"] not in POS or not inf["team"]:
            continue
        ros = sum(lg.proj_pts(pid, w) for w in range(lg.week, lg.final_week + 1))
        pool.setdefault(inf["pos"], []).append((ros, pid))
    return [pid for P, k in per_pos.items() for _, pid in sorted(pool.get(P, []), reverse=True)[:k]]


def on_waivers(lg):
    """Dropped since the last waiver run (still on waivers), plus a pending chopped roster."""
    runs = [t["status_updated"] for w in lg.tx for t in lg.tx[w] if t["type"] == "waiver"]
    last = max(runs) if runs else 0
    out = {p for w in lg.tx for t in lg.tx[w] if t["status_updated"] > last for p in (t.get("drops") or {})}
    return out | set((lg.pending or {}).get("pool", []))


def game_days(lg):
    sched = {g["game_id"]: g for g in J("schedule.json")}
    out = {}
    for pid, x in lg.proj[lg.week].items():
        g = sched.get(x.get("game"))
        if g:
            out[pid] = dt.date.fromisoformat(g["date"]).strftime("%a")
    return out


def quantiles_from_hist(h, qs):
    c = np.cumsum(h) / max(h.sum(), 1)
    return {str(q): float(np.searchsorted(c, q) / 2) for q in qs}


def summarize_grid(pts):
    """pts: [(bid, p_win, se)] sorted by bid -> best bid, walk-away bid, gain vs not bidding."""
    bids = np.array([p[0] for p in pts], float)
    pw = np.array([p[1] for p in pts])
    sm = np.convolve(np.r_[pw[0], pw, pw[-1]], [0.25, 0.5, 0.25], mode="valid")
    j = int(np.argmax(sm))
    se = float(np.mean([p[2] for p in pts]))
    walk = max(int(b) for b, s in zip(bids, sm) if s >= sm[0])
    return {"grid": [{"bid": int(b), "p_win": float(p), "se": float(s)} for b, p, s in pts],
            "best_bid": int(bids[j]), "best_p_win": float(sm[j]), "p_win_if_zero": float(sm[0]),
            "walk_away": walk, "good_range": [int(min(b for b, s in zip(bids, sm) if s >= sm[j] - se)),
                                              int(max(b for b, s in zip(bids, sm) if s >= sm[j] - se))]}


def budget_steps(budget):
    """Budget cuts for the value-of-a-dollar curve: $20, 40, 80, 120 in a $200 league, scaled to the league's FAAB."""
    return [round(d * budget / 200) for d in (20, 40, 80, 120)]


def trade_faab(lg, rid):
    """Net FAAB this team has received in completed trades (negative = sent)."""
    net = 0
    for w in lg.tx:
        for t in lg.tx[w]:
            if t["type"] == "trade" and t["status"] == "complete":
                for wb in t.get("waiver_budget") or []:
                    net += wb["amount"] * ((wb["receiver"] == rid) - (wb["sender"] == rid))
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--me", default="dpasceri")
    args = ap.parse_args()
    k = 8 if args.quick else 1
    t_start = time.time()
    timings = {}
    t0 = [time.time()]

    def tick(name):
        timings[name] = round(time.time() - t0[0], 1)
        print(f"[{timings[name]:6.1f}s] {name}", flush=True)
        t0[0] = time.time()

    lg = League(args.me)
    grid = sorted({round(b * lg.budget / 200) for b in BID_GRID})
    _, model_report = fit_model(lg)
    pm = PlayerModel()
    bm = BidModel(lg)
    tick("fit models")

    teams, me = lg.alive(), lg.me
    me_t = teams.index(me)
    ros = {t: lg.players(t) for t in teams}
    pending = lg.pending
    waiv = on_waivers(lg)
    fas = fa_candidates(lg, waiv)
    weeks = list(range(lg.week, lg.final_week + 1))
    pend_pool = [p for p in (pending or {}).get("pool", []) if sum(lg.proj_pts(p, w) for w in weeks) > 0]
    U = Universe(lg, pm, [p for t in teams for p in ros[t]] + fas + pend_pool, weeks)
    days = game_days(lg)

    # 1. static season: base + every free agent added to my roster
    variants = [ros[me]] + [ros[me] + [f] for f in fas]
    S = run_season(U, SeasonPlan(U, teams, ros, me, variants), 640_000 // k)
    tick("season sim (static, base + FA adds)")
    surv = 1 - np.cumsum(S["chop"][0], axis=1)
    standings = []
    for j, t in enumerate(teams):
        lu = best_lineup(lg.slots, [(lg.proj_pts(p, lg.week), p, lg.pos(p)) for p in ros[t]])
        st = lg.rosters[t]["settings"]
        standings.append({
            "rid": t, "owner": lg.owner(t), "faab": lg.faab(t), "me": t == me,
            "p_chop_now": float(S["chop"][0, j, 0]), "p_win": float(S["win"][0, j]),
            "survival": [float(x) for x in surv[j]],
            "proj_now": round(sum(lg.proj_pts(p, lg.week) for _, p in lu if p), 1),
            "season_pts": round(st.get("fpts", 0) + st.get("fpts_decimal", 0) / 100, 1),
        })
    n_active = len([p for p in ros[me] if p not in (lg.rosters[me].get("reserve") or [])])
    open_spots = max(0, sum(1 for s in lg.league["roster_positions"] if s != "IR") - n_active)
    fa_rows = []
    for v, f in enumerate(fas, start=1):
        fa_rows.append({
            "pid": f, "player": lg.label(f), "pos": lg.pos(f), "day": days.get(f),
            "proj_now": round(lg.proj_pts(f, lg.week), 1),
            "ros_avg": round(float(np.mean([lg.proj_pts(f, w) for w in weeks])), 1),
            "d_win": float(S["win"][v, me_t] - S["win"][0, me_t]),
            "d_survive_now": float(S["chop"][0, me_t, 0] - S["chop"][v, me_t, 0]),
            "inj": lg.info[f]["inj"],
        })
    fa_rows.sort(key=lambda r: -r["d_win"])

    # 2. this week's lineup: every legal starting eight, scored on the same draws
    best_fa = [r["pid"] for r in sorted(fa_rows, key=lambda r: -r["d_survive_now"])[:1] if r["d_survive_now"] > 0]
    pool = ros[me] + best_fa
    lineups = enumerate_lineups(U, 0, pool, lg.slots, U.starters0.get(me))
    current = None
    if lg.week in lg.matchups:
        m = next(m for m in lg.matchups[lg.week] if m["roster_id"] == me)
        current = [p for p in m["starters"] if p in U.ix]
        if sorted(current) not in [sorted(x) for x in lineups]:
            lineups.append(current)
    opp_idx = np.stack([U.lineup_idx(0, ros[t], U.starters0.get(t)) for t in teams], 1)
    # the free agent is only on my roster (and so only a bench fallback) in the lineups that add him
    my_idx = np.stack([U.fixed_lineup_idx(0, lu, pool if set(lu) & set(best_fa) else ros[me]) for lu in lineups], 1)
    W8 = run_week_lineups(U, opp_idx, my_idx, me_t, 480_000 // k)
    tick(f"lineup sim ({len(lineups)} lineups)")
    proj_opt = sorted(p for _, p in best_lineup(lg.slots, [(U.proj[0, U.ix[p]], p, U.pos[U.ix[p]]) for p in ros[me]]) if p)
    lineup_rows = sorted(({
        "players": sorted(lu, key=lambda p: (POS.index(lg.pos(p)), -lg.proj_pts(p, lg.week))),
        "p_chop": float(W8["p_chop"][li]), "mean": float(W8["mean"][li]), "sd": float(W8["sd"][li]),
        "is_current": current is not None and sorted(lu) == sorted(current), "is_proj_opt": sorted(lu) == proj_opt,
        "uses_fa": [p for p in lu if p in best_fa]} for li, lu in enumerate(lineups)), key=lambda r: r["p_chop"])

    # 3. the next auction: static value of each player, rival bid curves
    if pending:
        cases = [(pending["rid"], pend_pool)]
        win0, w_lo, bid_leg = 0, lg.week, lg.week - 1
    else:
        cases = [(t, [i for i in ros[t] if sum(lg.proj_pts(i, w) for w in weeks[1:]) > 0])
                 for t in sorted((t for t in teams if t != me), key=lambda t: -S["chop"][0, teams.index(t), 0])]
        win0, w_lo, bid_leg = 1, lg.week + 1, lg.week
    var2, alive2, lab2 = [], [], []
    for kk, plist in cases:
        mask = [t != kk for t in teams]
        var2.append(ros[me]); alive2.append(mask); lab2.append((kk, None))
        for i in plist:
            var2.append(ros[me] + [i]); alive2.append(mask); lab2.append((kk, i))
    S2 = run_season(U, SeasonPlan(U, teams, ros, me, var2, alive0=alive2, wi0=win0), 320_000 // k, seed=3)
    base_win = {kk: float(S2["win"][v, me_t]) for v, (kk, i) in enumerate(lab2) if i is None}
    rng = np.random.default_rng(17)
    scen = []
    for kk, plist in cases:
        others = {t: lg.faab(t) for t in teams if t not in (me, kk)}
        rows = []
        for v, (k2, i) in enumerate(lab2):
            if k2 != kk or i is None:
                continue
            gains = {t: gain(lg, ros[t], i, w_lo, w_lo + 3) for t in others}
            top = top_rival_bids(bm, others, gains, bid_leg, rng, 60_000)
            curve = p_win_curve(top, lg.faab(me))
            rows.append({
                "pid": i, "player": lg.label(i), "pos": lg.pos(i), "inj": lg.info.get(i, {}).get("inj"),
                "gain_me": round(gain(lg, ros[me], i, w_lo, w_lo + 3), 2),
                "d_win_static": float(S2["win"][v, me_t]) - base_win[kk],
                "rival_top": {str(q): float(np.quantile(top, q)) for q in (0.5, 0.8, 0.9, 0.95)},
                "p_no_rival_bid": float((top == 0).mean()),
                "p_beat_rivals": {str(b): float(curve[b]) for b in grid if b <= lg.faab(me)},
                "rival_gains": [{"owner": lg.owner(t), "gain": round(g, 2), "faab": lg.faab(t)}
                                for t, g in sorted(gains.items(), key=lambda kv: -kv[1])[:4]],
            })
        rows.sort(key=lambda r: -r["d_win_static"])
        scen.append({"rid": kk, "owner": lg.owner(kk), "known": bool(pending),
                     "p_chop": 1.0 if pending else float(S["chop"][0, teams.index(kk), 0]),
                     "base_win": base_win[kk], "players": rows})
    tick(f"auction values, static ({len(var2)} variants)")

    # 4. season with auctions: aggression sweep, value of a dollar, stress test
    base = {"me": me, "teams": teams, "rosters": ros, "budget": {t: lg.faab(t) for t in teams},
            "pending_pool": pend_pool if pending else None}
    B = lg.faab(me)
    sweep = [{"m_early": a, "m_late": b, "switch_week": 12, "budget": B}
             for a in (0.5, 0.75, 1.0, 1.5) for b in (0.75, 1.0, 1.5, 2.0)]
    D1 = dynamic.run(U, bm, base, sweep, 10_000 // k)
    best = max(D1, key=lambda r: r["p_win"])
    pol = dict(POLICY)
    used = next(d for d in D1 if (d["m_early"], d["m_late"]) == (pol["m_early"], pol["m_late"]))
    if best["p_win"] - used["p_win"] > SWITCH_Z * np.hypot(best["se"], used["se"]):
        pol = {k: best[k] for k in POLICY}
        used = best
    tick("dynamic: aggression sweep")
    budgets = sorted({B, 0} | {max(B - d, 0) for d in budget_steps(lg.budget)}, reverse=True)
    bud_set = [pol | {"budget": b} for b in budgets] + [pol | {"budget": b, "rival_boost": STRESS} for b in budgets]
    D2 = dynamic.run(U, bm, base, bud_set, 10_000 // k, seed=12)
    tick("dynamic: budget curve (+ stress test)")

    # 4b. power rankings: play every path to the end and record who wins. "fitted" = everyone,
    # me included, bids the way they have all season; "policy" = I bid the recommended way.
    P_fit, P_pol = dynamic.run(U, bm, base, [pol | {"budget": B, "track_all": True, "fitted_me": True},
                                             pol | {"budget": B, "track_all": True}], 20_000 // k, seed=21)
    tick("dynamic: power rankings")
    power = []
    for j, t in enumerate(teams):
        lineup_pts = [sum(lg.proj_pts(p, w) for _, p in best_lineup(lg.slots, [(lg.proj_pts(p, w), p, lg.pos(p)) for p in ros[t]]) if p)
                      for w in weeks]
        s = next(x for x in standings if x["rid"] == t)
        power.append({"owner": lg.owner(t), "me": t == me, "faab": lg.faab(t), "p_win": P_fit["win_by"][j],
                      "p_win_policy": P_pol["win_by"][j], "e_weeks": P_fit["weeks_by"][j], "p_win_frozen": s["p_win"],
                      "p_chop_now": s["p_chop_now"], "ros_proj": round(float(np.mean(lineup_pts)), 1),
                      "worst_week": int(weeks[int(np.argmin(lineup_pts))]), "worst_proj": round(float(min(lineup_pts)), 1),
                      "survival": s["survival"], "season_pts": s["season_pts"]})
    power.sort(key=lambda x: -x["p_win"])
    pace = {"history": faab_history(lg), "start_total": lg.league["total_rosters"] * lg.budget,
            "n_auctions": lg.league["total_rosters"] - 2, "left_now": sum(lg.faab(t) for t in teams),
            "trade_net_me": trade_faab(lg, me),
            "forecast": {key: P_fit[key] for key in ("league_spend", "burned", "top_price")}}

    # 5. forced-bid grids: P(win | my bid) for the players that matter in the next auction
    jobs = []
    for sc in scen[: (1 if pending else 4)]:
        for r in [r for r in sc["players"] if r["gain_me"] >= dynamic.MIN_GAIN][: (6 if pending else 3)]:
            for b in grid:
                if b <= B:
                    extra = {} if pending else {"chopped0": sc["rid"]}
                    jobs.append(pol | {"budget": B, "force": {r["pid"]: b}, "_pid": r["pid"], "_bid": b, "_k": sc["rid"]} | extra)
    stress_jobs = [j | {"rival_boost": STRESS} for j in jobs if j["_pid"] == jobs[0]["_pid"] and j["_k"] == jobs[0]["_k"]] if jobs else []
    D3 = dynamic.run(U, bm, base, jobs + stress_jobs, (8_000 if pending else 4_000) // k, seed=13) if jobs else []
    tick(f"dynamic: forced-bid grid ({len(jobs) + len(stress_jobs)} settings)")
    for sc in scen:
        for r in sc["players"]:
            for tag, boosted in (("", False), ("stress_", True)):
                pts = sorted((d["_bid"], d["p_win"], d["se"]) for d in D3
                             if d["_k"] == sc["rid"] and d["_pid"] == r["pid"] and ("rival_boost" in d) == boosted)
                if pts:
                    r[tag + "bid"] = summarize_grid(pts)

    meta = J("meta.json")
    live = None
    if lg.live:  # the week is under way: where each team stands so far
        G = list(lg.live["games"].values())
        rows = []
        for t in teams:
            st = [lg.game_state(p) for p in (lg.starters(t, lg.week) or []) if p]
            rows.append({"owner": lg.owner(t), "me": t == me, "pts": round(sum(a for _, _, a in st), 1),
                         "left": sum(s != "post" for s, _, _ in st), "p_chop": float(S["chop"][0, teams.index(t), 0])})
        live = {"as_of": lg.live["fetched_at"], "n": {s: sum(g["state"] == s for g in G) for s in ("pre", "in", "post")},
                "in_progress": [f'{g["away"]} @ {g["home"]} {g["clock"] or ""}'.strip() for g in G if g["state"] == "in"],
                "teams": sorted(rows, key=lambda x: -x["pts"]), "locked": sorted(U.locked & set(ros[me]))}
    res = {
        "generated_at": int(time.time()), "fetched_at": meta["fetched_at"], "league": lg.league["name"],
        "league_id": lg.meta["league"], "season": lg.season, "week": lg.week, "final_week": lg.final_week,
        "me": lg.owner(me), "faab_me": B, "open_spots": open_spots, "weeks": weeks, "threads": THREADS,
        "pending": {"owner": lg.owner(pending["rid"]), "n": len(pend_pool)} if pending else None, "live": live,
        "timings": timings,
        "sims": {"season": S["N"], "lineups": W8["N"], "auction_static": S2["N"], "dynamic_paths": D1[0]["n"] if D1 else 0,
                 "grid_paths": D3[0]["n"] if D3 else 0},
        "standings": sorted(standings, key=lambda r: -r["p_win"]),
        "power": power, "faab_pace": pace,
        "chop_line": quantiles_from_hist(S["line_hist"][0], (0.1, 0.25, 0.5, 0.75, 0.9)),
        "my_score": quantiles_from_hist(S["my_hist"][0], (0.1, 0.25, 0.5, 0.75, 0.9)),
        "roster": [{"pid": p, "player": lg.label(p), "pos": lg.pos(p), "inj": lg.info[p]["inj"], "day": days.get(p),
                    "ir": p in (lg.rosters[me].get("reserve") or []),
                    "proj": [round(lg.proj_pts(p, w), 1) for w in weeks]} for p in ros[me]],
        "labels": {p: lg.label(p) for p in set(pool) | {p for lu in lineups for p in lu}},
        "lineups": lineup_rows[:10] + [r for r in lineup_rows[10:] if r["is_current"] or r["is_proj_opt"]],
        "n_lineups": len(lineup_rows),
        "free_agents": fa_rows,
        "scenarios": scen,
        "dynamic": {"sweep": D1, "budgets": [d for d in D2 if "rival_boost" not in d],
                    "budgets_stress": [d for d in D2 if "rival_boost" in d], "best": best, "used": used, "stress": STRESS,
                    "default": POLICY},
        "bid_model": bm.summary() | {"managers": [
            {"owner": lg.owner(t), "faab": lg.faab(t), "bids_often": round(bm.v.get(t, 0), 2), "bids_hard": round(bm.u.get(t, 0), 2),
             "n_bids": bm.n_bids.get(t, 0), "n_chances": bm.n_offered.get(t, 0)} for t in teams]},
        "price_backtest": price_backtest(lg, bm),
        "model_report": model_report,
        "backtest": J("backtest.json") if os.path.exists(os.path.join(DATA, "backtest.json")) else None,
    }
    timings["total"] = round(time.time() - t_start, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done in {timings['total']}s -> out/results.json")
    try:
        import report
    except ImportError:
        return
    report.render(res)
    print("-> out/cheatsheet.html")


if __name__ == "__main__":
    main()
