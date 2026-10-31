"""Season-with-auctions simulation: what is a FAAB dollar worth, and how hard should I bid?

Each path plays the rest of the season week by week: lineups from projections (known
injuries benched), lowest alive score chopped, then the chopped roster goes to a sealed-bid
FAAB auction. Rivals bid from the fitted BidModel given their own lineup gain and budget; I bid
m x the league's typical bid for my gain (m_early before switch_week, m_late after). Highest
bid wins and pays its bid, claims are processed from the highest bid down, budgets and the
roster cap are enforced (the weakest rest-of-season player is dropped).

Two ways to pin down the next auction, both with my bid on chosen players forced to fixed
amounts so that sweeping the amount on common random numbers traces P(win | my bid):
  pending   the chop already happened and its roster is on waivers (auction before week 0)
  chopped0  hypothetical: team k is the one chopped in the current week
rival_boost = (d_logit, d_log_size) makes every rival bid more often / higher (stress test).
"""
import math
import random

from common import *  # noqa: F401,F403
from common import POS, THREADS, best_lineup, eligible, np
from sim import _G, _pool_map, team_scores

HORIZON = 4
MAX_ROSTER = 15  # 13 active + 2 IR
MIN_GAIN = 2.0   # I don't bid on players worth less than this many lineup points a week


def _minelig(slots, roster, projw, pos_of, ix):
    lu = best_lineup(slots, [(projw[ix[p]], p, pos_of[p]) for p in roster])
    out = {}
    for P in POS:
        v = [projw[ix[p]] if p else 0.0 for s, p in lu if eligible(s, P)]
        out[P] = min(v) if v else math.inf
    return out


def _auction(ctx, pool, alive, rosters, budget, my_bid, fut, leg, pr, boost, fitted_me=False):
    """Run one waiver run in place. my_bid(p, g) -> my bid (ignored with fitted_me: I bid like my own history).

    Returns (my spend, my studs, league spend, top price).
    """
    me, proj, slots, pos_of, ix, bm = ctx["me"], ctx["proj"], ctx["slots"], ctx["pos_of"], ctx["ix"], ctx["bm"]
    ros_next = ctx["ros_val"][fut[0]]
    claims = []
    for t in alive:
        el = [_minelig(slots, rosters[t], proj[w2], pos_of, ix) for w2 in fut] if budget[t] > 0 else None
        for p in pool:
            u_part, z, u_tie = pr.random(), pr.gauss(0, 1), pr.random()  # always drawn: keeps paths aligned
            if el is None:
                continue
            P = pos_of[p]
            g = sum(max(0.0, proj[w2][ix[p]] - e[P]) for w2, e in zip(fut, el)) / len(fut)
            if t == me and not fitted_me:
                b = my_bid(p, g)
            else:
                q = bm.p_bid(t, g, budget[t])
                if boost[0]:
                    q = 1 / (1 + math.exp(-(math.log(q / (1 - q)) + boost[0])))
                b = round(math.exp(bm.mu(t, g, budget[t], leg) + boost[1] + bm.sigma * z)) if u_part < q else 0
            b = min(b, budget[t])
            if b >= 1:
                claims.append((b, u_tie, t, p, g))
    claims.sort(reverse=True)
    taken, spent, studs, total, top = set(), 0, 0, 0, 0
    for b, _, t, p, g in claims:
        if p in taken or budget[t] < b:
            continue
        taken.add(p)
        budget[t] -= b
        total += b
        top = max(top, b)
        rosters[t].append(p)
        if len(rosters[t]) > MAX_ROSTER:
            rosters[t].remove(min((q for q in rosters[t] if q != p), key=lambda q: ros_next[ix[q]]))
        if t == me:
            spent += b
            studs += g >= 8
    return spent, studs, total, top


def _paths_chunk(job):
    ci, C, seed, settings = job
    U, bm, base = _G["U"], _G["bm"], _G["base"]
    W = len(U.weeks)
    ix = U.ix
    ctx = {"me": base["me"], "bm": bm, "slots": U.lg.slots, "ix": ix, "pos_of": {p: U.pos[i] for p, i in ix.items()},
           "proj": [list(U.proj[wi]) for wi in range(W)],
           "ros_val": [list(U.proj[wi:].sum(0)) for wi in range(W)] + [[0.0] * U.n]}
    proj, slots, pos_of, me = ctx["proj"], ctx["slots"], ctx["pos_of"], ctx["me"]
    rng = np.random.default_rng([seed, ci])
    draws = [(s, a) for _, s, a in U.paths(C, rng, 0)]
    T = len(base["teams"])
    tix = {t: i for i, t in enumerate(base["teams"])}
    # week 0 under way: lineups are partly locked, so score every team's lock-aware lineup on all paths at once
    week0 = None
    if U.locked and not base.get("pending_pool"):
        idx0 = np.stack([U.lineup_idx(0, base["rosters"][t], U.starters0.get(t)) for t in base["teams"]], 1)
        week0 = team_scores(draws[0][0], draws[0][1], idx0)
    out = []
    for st in settings:
        track = st.get("track_all", False)  # keep playing after I'm chopped, to see who wins
        fitted = st.get("fitted_me", False)  # I bid like my own history (for a fair power ranking)
        tot = {"wins": 0, "weeks": 0, "spent": 0.0, "studs": 0, "n": C, "win_by": [0] * T, "weeks_by": [0] * T,
               "league_spend": [0.0] * (W + 1), "top_price": [0.0] * (W + 1), "burned": [0.0] * (W + 1)}
        boost = st.get("rival_boost", (0.0, 0.0))
        force = st.get("force", {})
        for c in range(C):
            pr = random.Random(seed * 1_000_003 + ci * 10_007 + c)
            rosters = {t: list(r) for t, r in base["rosters"].items()}
            budget = dict(base["budget"])
            budget[me] = st["budget"]
            alive = list(base["teams"])

            def policy(p, g, wk):
                if g < MIN_GAIN:
                    return 0
                m = st["m_early"] if wk < st["switch_week"] else st["m_late"]
                return round(m * math.exp(bm.mu(me, g, budget[me], wk) - bm.u.get(me, 0.0)))

            if base.get("pending_pool"):
                wk = U.weeks[0] - 1
                s_, k_, tl, tp = _auction(ctx, base["pending_pool"], alive, rosters, budget,
                                          lambda p, g: force[p] if p in force else policy(p, g, wk), [0, 1, 2, 3][:W], wk, pr, boost, fitted)
                tot["spent"] += s_
                tot["studs"] += k_
                tot["league_spend"][0] += tl
                tot["top_price"][0] += tp
            me_out = None
            for wi in range(W):
                wk = U.weeks[wi]
                if wi == 0 and st.get("chopped0") is not None:
                    low_t = st["chopped0"]       # condition on this team going out this week
                    my_bid = lambda p, g: force[p] if p in force else policy(p, g, wk)  # noqa: E731
                else:
                    sc, av = draws[wi][0][c], draws[wi][1][c]
                    pw = proj[wi]
                    low, low_t = math.inf, None
                    for t in alive:
                        if wi == 0 and week0 is not None:
                            tot_t = week0[c, tix[t]]
                        else:
                            lu = best_lineup(slots, [(pw[ix[p]], p, pos_of[p]) for p in rosters[t] if av[ix[p]]])
                            tot_t = sum(sc[ix[p]] for _, p in lu if p)
                        if tot_t < low:
                            low, low_t = tot_t, t
                    my_bid = lambda p, g: policy(p, g, wk)  # noqa: E731
                alive.remove(low_t)
                tot["weeks_by"][tix[low_t]] += wi
                tot["burned"][wi + 1] += budget[low_t]
                if low_t == me:
                    me_out = wi
                    if not track:
                        break
                if len(alive) == 1 or wi == W - 1:
                    break
                pool = [p for p in rosters[low_t] if ctx["ros_val"][wi + 1][ix[p]] > 0]
                fut = list(range(wi + 1, min(wi + 1 + HORIZON, W)))
                s_, k_, tl, tp = _auction(ctx, pool, alive, rosters, budget, my_bid, fut, wk, pr, boost, fitted)
                tot["spent"] += s_
                tot["studs"] += k_
                tot["league_spend"][wi + 1] += tl
                tot["top_price"][wi + 1] += tp
            if len(alive) == 1:
                tot["win_by"][tix[alive[0]]] += 1
                tot["weeks_by"][tix[alive[0]]] += W
            if me_out is None:
                tot["wins"] += alive == [me]
                tot["weeks"] += W
            else:
                tot["weeks"] += me_out
        out.append(tot)
    return out


def run(U, bm, base, settings, n_paths, seed=11, chunk=200):
    """P(win) etc. per setting. Jobs split over path chunks and setting groups; same chunk = same draws."""
    _G.update(U=U, bm=bm, base=base)
    n_chunks = max(1, n_paths // chunk)
    n_groups = max(1, min(len(settings), math.ceil(3 * THREADS / n_chunks)))
    groups = [list(range(g, len(settings), n_groups)) for g in range(n_groups)]
    jobs, owners = [], []
    for ci in range(n_chunks):
        for grp in groups:
            jobs.append((ci, chunk, seed, [settings[i] for i in grp]))
            owners.append(grp)
    res = _pool_map(_paths_chunk, jobs)
    acc = [None] * len(settings)
    for grp, r in zip(owners, res):
        for i, x in zip(grp, r):
            if acc[i] is None:
                acc[i] = {key: list(v) if isinstance(v, list) else v for key, v in x.items()}
                continue
            for key, v in x.items():
                acc[i][key] = [a + b for a, b in zip(acc[i][key], v)] if isinstance(v, list) else acc[i][key] + v
    out = []
    for st, tot in zip(settings, acc):
        n = tot["n"]
        p = tot["wins"] / n
        row = st | {"p_win": p, "se": math.sqrt(p * (1 - p) / n), "e_weeks": tot["weeks"] / n,
                    "avg_spent": tot["spent"] / n, "studs_bought": tot["studs"] / n, "n": n}
        if st.get("track_all"):
            row |= {key: [v / n for v in tot[key]] for key in ("win_by", "weeks_by", "league_spend", "top_price", "burned")}
        out.append(row)
    return out
