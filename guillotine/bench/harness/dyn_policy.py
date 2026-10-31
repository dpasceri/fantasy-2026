"""Season-with-auctions sim with a pluggable bid policy for me: a copy of dynamic.py for the bench.

Same model as dynamic.py (read its docstring): each path plays the rest of the season week by
week, the lowest alive score is chopped and its roster goes to a sealed first-price FAAB auction,
rivals bid from the fitted BidModel, the highest bid wins and pays its bid. Opt-in additions per
setting:
  policy     object with bid(player, gain, week, state) -> dollars; replaces my m x typical rule.
             Optional hooks: reset() at the start of every path, begin(state) once per auction.
  bm_rival   rivals bid from this BidModel; my policy still sees the point model (state.bm).
_paths_chunk also returns per-path outcomes (tot["path"]). A setting without policy / bm_rival
runs dynamic.py's code path and matches it bit for bit (_baseline_policy/verify.py checks it).
"""
import math
import random

from common import *  # noqa: F401,F403
from common import POS, best_lineup, eligible, np
from auction import BidModel
from scipy.optimize import minimize
from scipy.special import ndtr

HORIZON = 4
MAX_ROSTER = 15  # 13 active + 2 IR
MIN_GAIN = 2.0   # dynamic.py's rule: I don't bid on players worth less than this many lineup points a week
OUTCOMES = ("win", "weeks", "spend", "buys", "studs")

_G = {}  # U, bm, base; set by setup() before forking


def setup(U, bm, base):
    _G.update(U=U, bm=bm, base=base)


def _minelig(slots, roster, projw, pos_of, ix):
    lu = best_lineup(slots, [(projw[ix[p]], p, pos_of[p]) for p in roster])
    out = {}
    for P in POS:
        v = [projw[ix[p]] if p else 0.0 for s, p in lu if eligible(s, P)]
        out[P] = min(v) if v else math.inf
    return out


class State:
    """What my policy sees at one auction. Plain values, rebuilt per auction; treat as read-only.

    me, week (NFL week whose chop fed this auction = the bid-model leg; the next week played is
    week + 1), wi (its index in weeks, -1 for an auction pending before the first sim week),
    weeks (sim weeks), weeks_left (weeks still to play after this auction), auctions_left (after
    this one), alive (rids incl. me), n_alive, chopped (rid on offer, None for a pending pool),
    pool (pids on offer), gains {pid: my lineup gain}, budget (mine), budgets {rid: budget} (alive),
    rival_budgets, roster (mine), rosters {rid: tuple}, fut (week indices behind the gains),
    bm (point BidModel fit on all bid history; never the stressed or bootstrap rivals), rng (the
    policy's own random.Random, seeded per path).
    """

    def __init__(self, ctx, wi, wk, fut, pool, alive, rosters, budget, G, chopped, rng):
        U, me = ctx["U"], ctx["me"]
        self._ctx, self._G = ctx, G
        self.me, self.week, self.wi, self.fut = me, wk, wi, tuple(fut)
        self.weeks = tuple(U.weeks)
        self.weeks_left = len(U.weeks) - (wi + 1)
        self.auctions_left = max(self.weeks_left - 1, 0)
        self.alive, self.n_alive, self.chopped = tuple(alive), len(alive), chopped
        self.pool = tuple(pool)
        self.gains = dict(G.get(me, {}))
        self.budget = budget[me]
        self.budgets = {t: budget[t] for t in alive}
        self.rival_budgets = {t: budget[t] for t in alive if t != me}
        self.rosters = {t: tuple(rosters[t]) for t in alive}
        self.roster = self.rosters.get(me, ())
        self.bm, self.rng, self.min_gain = ctx["bm"], rng, MIN_GAIN

    # projections
    def proj(self, pid, wi):
        """Projected points of pid in sim week index wi (0 outside the sim or the universe)."""
        c = self._ctx
        return c["proj"][wi][c["ix"][pid]] if 0 <= wi < len(self.weeks) and pid in c["ix"] else 0.0

    def ros(self, pid, wi=None):
        """Rest-of-season projected points from week index wi (default: the next week played)."""
        c = self._ctx
        wi = self.wi + 1 if wi is None else wi
        return c["ros_val"][wi][c["ix"][pid]] if pid in c["ix"] else 0.0

    def pos(self, pid):
        return self._ctx["pos_of"].get(pid)

    # rivals
    def rival_gains(self, pid):
        """{rid: lineup gain} for every rival with money (same gain definition as mine)."""
        return {t: g[pid] for t, g in self._G.items() if t != self.me}

    def typical(self, g, budget=None, week=None):
        """League-typical bid for my gain g: exp(mu(me, g, budget, week) - my own bids-hard effect)."""
        bm = self.bm
        b = self.budget if budget is None else budget
        return math.exp(bm.mu(self.me, g, b, self.week if week is None else week) - bm.u.get(self.me, 0.0))

    def rival_cdf(self, pid, x):
        """P(highest rival claim on pid <= x) for integer x >= 0 under the point model (no stress),
        each rival on its own: ignores money spent earlier in the same run. Vectorized over x."""
        x = np.asarray(x, float)
        out = np.ones_like(x)
        bm, wk = self.bm, self.week
        for t, g in self.rival_gains(pid).items():
            B = self.budgets[t]
            q = bm.p_bid(t, g, B)
            f = 1 - q + q * ndtr((np.log(np.maximum(x, 0) + 0.5) - bm.mu(t, g, B, wk)) / bm.sigma)
            out *= np.where(x >= B, 1.0, f)
        return np.where(x < 0, 0.0, out)

    def p_win(self, pid, b):
        """Approximate P(a bid of b beats every rival claim on pid); ties split evenly. 0 for b < 1."""
        b = np.asarray(b, float)
        lo, hi = self.rival_cdf(pid, b - 1), self.rival_cdf(pid, b)
        return np.where(b < 1, 0.0, lo + 0.5 * (hi - lo))


def _policy_bids(pol, state):
    if hasattr(pol, "begin"):
        pol.begin(state)
    out = {}
    for p in state.pool:
        b = float(pol.bid(p, state.gains[p], state.week, state))
        if not math.isfinite(b):
            raise ValueError(f"policy bid {b} for {p} at week {state.week}")
        out[p] = int(round(b))  # whole dollars
    return out


def _auction(ctx, pool, alive, rosters, budget, my_bid, fut, leg, pr, boost, fitted_me=False, bm_r=None, state_bids=None):
    """Run one waiver run in place. my_bid(p, g) -> my bid (ignored with fitted_me: I bid like my own history);
    state_bids(G) -> {pid: my bid} replaces my_bid (policy hook). Rivals bid from bm_r (default the point model).

    Returns (my spend, my studs, league spend, top price, my buys).
    """
    me, proj, slots, pos_of, ix, bm = ctx["me"], ctx["proj"], ctx["slots"], ctx["pos_of"], ctx["ix"], ctx["bm"]
    bm_r = bm if bm_r is None else bm_r
    ros_next = ctx["ros_val"][fut[0]]
    G = {}  # gains first (deterministic), then the draws in dynamic.py's order: paths stay aligned
    for t in alive:
        if budget[t] > 0:
            el = [_minelig(slots, rosters[t], proj[w2], pos_of, ix) for w2 in fut]
            G[t] = {p: sum(max(0.0, proj[w2][ix[p]] - e[pos_of[p]]) for w2, e in zip(fut, el)) / len(fut) for p in pool}
    mine = {}
    if me in G and not fitted_me:
        mine = state_bids(G) if state_bids is not None else {p: my_bid(p, G[me][p]) for p in pool}
    claims = []
    for t in alive:
        for p in pool:
            u_part, z, u_tie = pr.random(), pr.gauss(0, 1), pr.random()  # always drawn: keeps paths aligned
            if t not in G:
                continue
            g = G[t][p]
            if t == me and not fitted_me:
                b = mine[p]
            else:
                q = bm_r.p_bid(t, g, budget[t])
                if boost[0]:
                    q = 1 / (1 + math.exp(-(math.log(q / (1 - q)) + boost[0])))
                b = round(math.exp(bm_r.mu(t, g, budget[t], leg) + boost[1] + bm_r.sigma * z)) if u_part < q else 0
            b = min(b, budget[t])
            if b >= 1:
                claims.append((b, u_tie, t, p, g))
    claims.sort(reverse=True)
    taken, spent, studs, total, top, buys = set(), 0, 0, 0, 0, 0
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
            buys += 1
    return spent, studs, total, top, buys


def _paths_chunk(job):
    ci, C, seed, settings = job
    U, bm, base = _G["U"], _G["bm"], _G["base"]
    W = len(U.weeks)
    ix = U.ix
    ctx = {"me": base["me"], "bm": bm, "U": U, "slots": U.lg.slots, "ix": ix, "pos_of": {p: U.pos[i] for p, i in ix.items()},
           "proj": [list(U.proj[wi]) for wi in range(W)],
           "ros_val": [list(U.proj[wi:].sum(0)) for wi in range(W)] + [[0.0] * U.n]}
    proj, slots, pos_of, me = ctx["proj"], ctx["slots"], ctx["pos_of"], ctx["me"]
    rng = np.random.default_rng([seed, ci])
    draws = [(s, a) for _, s, a in U.paths(C, rng, 0)]
    T = len(base["teams"])
    tix = {t: i for i, t in enumerate(base["teams"])}
    out = []
    for st in settings:
        track = st.get("track_all", False)  # keep playing after I'm chopped, to see who wins
        fitted = st.get("fitted_me", False)  # I bid like my own history (for a fair power ranking)
        pol = st.get("policy")               # bench hook: my bids come from pol.bid(...)
        bm_r = st.get("bm_rival")            # bench hook: rivals bid from this model
        tot = {"wins": 0, "weeks": 0, "spent": 0.0, "studs": 0, "n": C, "win_by": [0] * T, "weeks_by": [0] * T,
               "league_spend": [0.0] * (W + 1), "top_price": [0.0] * (W + 1), "burned": [0.0] * (W + 1),
               "path": {k: [] for k in OUTCOMES}}
        boost = st.get("rival_boost", (0.0, 0.0))
        force = st.get("force", {})
        for c in range(C):
            pr = random.Random(seed * 1_000_003 + ci * 10_007 + c)
            prng = random.Random(f"policy:{seed}:{ci}:{c}")  # the policy's own stream, never touches pr
            rosters = {t: list(r) for t, r in base["rosters"].items()}
            budget = dict(base["budget"])
            budget[me] = st["budget"]
            alive = list(base["teams"])
            my = [0, 0, 0]  # spend, studs, buys on this path
            if pol is not None and hasattr(pol, "reset"):
                pol.reset()

            def policy(p, g, wk):
                if g < MIN_GAIN:
                    return 0
                m = st["m_early"] if wk < st["switch_week"] else st["m_late"]
                return round(m * math.exp(bm.mu(me, g, budget[me], wk) - bm.u.get(me, 0.0)))

            def hook(wi, wk, fut, pool, chopped):
                if pol is None:
                    return None
                return lambda G: _policy_bids(pol, State(ctx, wi, wk, fut, pool, alive, rosters, budget, G, chopped, prng))

            def book(r, wi):
                s_, k_, tl, tp, nb = r
                tot["spent"] += s_
                tot["studs"] += k_
                tot["league_spend"][wi] += tl
                tot["top_price"][wi] += tp
                my[0] += s_
                my[1] += k_
                my[2] += nb

            if base.get("pending_pool"):
                wk = U.weeks[0] - 1
                fut = [0, 1, 2, 3][:W]
                book(_auction(ctx, base["pending_pool"], alive, rosters, budget,
                              lambda p, g: force[p] if p in force else policy(p, g, wk), fut, wk, pr, boost, fitted,
                              bm_r, hook(-1, wk, fut, base["pending_pool"], None)), 0)
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
                book(_auction(ctx, pool, alive, rosters, budget, my_bid, fut, wk, pr, boost, fitted,
                              bm_r, hook(wi, wk, fut, pool, low_t)), wi + 1)
            if len(alive) == 1:
                tot["win_by"][tix[alive[0]]] += 1
                tot["weeks_by"][tix[alive[0]]] += W
            won = me_out is None and alive == [me]
            if me_out is None:
                tot["wins"] += alive == [me]
                tot["weeks"] += W
            else:
                tot["weeks"] += me_out
            pth = tot["path"]
            pth["win"].append(float(won))
            pth["weeks"].append(float(W if me_out is None else me_out))
            pth["spend"].append(float(my[0]))
            pth["studs"].append(float(my[1]))
            pth["buys"].append(float(my[2]))
        out.append(tot)
    return out


# ---------------------------------------------------------------- bid model refits (robustness)
def refit_bm(bm, rows):
    """A BidModel fit on the given bid-history rows: the body of BidModel.__init__ after bid_history,
    copied verbatim (refit_bm(bm, bm.rows) reproduces bm exactly; verify.py checks it)."""
    self = BidModel.__new__(BidModel)
    self.lg = bm.lg
    rows = [r for r in rows if r["budget"] > 0]
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
    return self


def canonical_bm(bm):
    """bm refit on its rows in a fixed order. auction.bid_history builds rows by iterating a set of
    pid strings, so their order (and, through L-BFGS, the participation fit at ~1e-3) changes with
    PYTHONHASHSEED from one process to the next; sorting makes every run see the same model."""
    return refit_bm(bm, sorted(bm.rows, key=lambda r: (r["leg"], r["run"], r["rid"], r["pid"])))


def boot_bm(bm, seed, tag=20261029):
    """BidModel refit on a cluster bootstrap of the bid history: whole lots (waiver run, player:
    every manager's bid-or-not on that player in that run) resampled with replacement."""
    lots = {}
    for r in bm.rows:
        lots.setdefault((r["run"], r["pid"]), []).append(r)
    keys = sorted(lots)
    pick = np.random.default_rng([tag, int(seed)]).integers(0, len(keys), len(keys))
    return refit_bm(bm, [r for k in pick for r in lots[keys[k]]])


def run(U, bm, base, settings, n_paths, seed=11, chunk=200):
    """dynamic.run's interface (serial; the policy track parallelizes over seeds itself)."""
    setup(U, bm, base)
    n_chunks = max(1, n_paths // chunk)
    res = [_paths_chunk((ci, chunk, seed, settings)) for ci in range(n_chunks)]
    out = []
    for i, st in enumerate(settings):
        n = chunk * n_chunks
        w = sum(r[i]["wins"] for r in res)
        p = w / n
        out.append({k: v for k, v in st.items() if k not in ("policy", "bm_rival")}
                   | {"p_win": p, "se": math.sqrt(p * (1 - p) / n), "e_weeks": sum(r[i]["weeks"] for r in res) / n,
                      "avg_spent": sum(r[i]["spent"] for r in res) / n, "studs_bought": sum(r[i]["studs"] for r in res) / n, "n": n})
    return out
