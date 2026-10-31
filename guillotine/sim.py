"""Vectorized Monte Carlo engine.

One sim = draw every relevant player's week from the fitted distributions (copula-correlated by
NFL team and game), play the lineup each manager would set from projections (bench subs cover
inactive starters), chop the lowest alive score, repeat through the final week.

Every variant of my roster or lineup is scored on the same draws (common random numbers), so
differences between variants carry far less noise than either number alone. Chunks of sims run
in THREADS worker processes.
"""
import multiprocessing as mp
from itertools import combinations

from common import *  # noqa: F401,F403
from common import P_OUT, POS, THREADS, best_lineup, eligible, np
from model import Q
from scipy.special import ndtr

LEVEL_STEP = 0.10      # weekly random-walk sd (log scale) of a player's true level vs today's projection
# Absences after this week, fit to how often healthy starters (untagged, played last week) actually played
# 1-8 weeks later, in weeks 1-16 (nflverse snaps 2016-2026, bench/RESULTS.md): an available player starts an
# absence with weekly prob INJ_HAZARD; it is season-long with prob INJ_LONG, else Geometric with mean
# INJ_MEAN_WEEKS. That gives 94/87/84% played at 1/4/8 weeks ahead (QBs, who also get benched, 91/84/79%).
INJ_HAZARD = {"QB": 0.086, "other": 0.064}
INJ_LONG = {"QB": 0.20, "other": 0.126}
INJ_MEAN_WEEKS = {"QB": 1.45, "other": 2.07}
INJ_LONG_WEEKS = 99
CHUNK = 8000

_G = {}  # state shared with forked workers


class Universe:
    """Every player the sims touch, with per-week projections, quantile tables and NFL team/game ids."""

    def __init__(self, lg, pm, pids, weeks):
        self.lg, self.pm = lg, pm
        self.pids = list(dict.fromkeys(pids))
        self.ix = {p: i for i, p in enumerate(self.pids)}
        self.weeks = list(weeks)
        n, W = len(self.pids), len(self.weeks)
        self.n = n
        self.pos = [lg.pos(p) for p in self.pids]
        self.proj = np.zeros((W, n))
        self.qflat = np.zeros((W, n * Q), np.float32)
        self.team = np.zeros((W, n), np.int64)
        self.game = np.zeros((W, n), np.int64)
        self.n_team, self.n_game = [], []
        self.load = np.array([pm.loading.get(p, 0.0) for p in self.pos], np.float32)
        self.cload = np.sqrt(1 - self.load ** 2).astype(np.float32)
        tags = [lg.info.get(p, {}).get("inj") or "" for p in self.pids]
        self.p_out = np.array([P_OUT[t] if t in ("Questionable", "Doubtful") else 0.0 for t in tags], np.float32)
        # week 0 under way: players whose game has kicked off are locked (in or out of the lineup) and carry
        # their points so far; their score is that plus the unplayed share of a full-game draw
        self.locked, self.starters0 = set(), {}
        if lg.live and self.weeks and self.weeks[0] == lg.week:
            gs = [lg.game_state(p) for p in self.pids]
            self.started = np.array([s != "pre" for s, _, _ in gs])
            self.frac_left = np.array([f for _, f, _ in gs], np.float32)
            self.pts0 = np.array([a for _, _, a in gs], np.float32)
            self.locked = {p for p, s in zip(self.pids, self.started) if s}
            self.starters0 = {rid: lg.starters(rid, lg.week) for rid in lg.rosters}
        grp = ["QB" if p == "QB" else "other" for p in self.pos]
        self.haz = np.array([INJ_HAZARD[g] for g in grp], np.float32)
        self.p_long = np.array([INJ_LONG[g] for g in grp])
        self.p_geo = np.array([1.0 / INJ_MEAN_WEEKS[g] for g in grp])
        for wi, w in enumerate(self.weeks):
            teams, games = {}, {}
            for i, p in enumerate(self.pids):
                x = lg.proj[w].get(p) or {}
                pr = lg.proj_pts(p, w)
                self.proj[wi, i] = pr
                self.qflat[wi, i * Q:(i + 1) * Q] = pm.quantiles(self.pos[i], pr)
                self.team[wi, i] = teams.setdefault(x.get("team") or f"_{i}", len(teams))
                self.game[wi, i] = games.setdefault(x.get("game") or f"_{i}", len(games))
            self.n_team.append(len(teams))
            self.n_game.append(len(games))

    def paths(self, C, rng, wi0=0):
        """Yield (wi, score, avail) per week; arrays are (C, n+1), the last column is an empty slot."""
        n, g = self.n, self.pm.game_share
        loglev = np.zeros((C, n), np.float32)
        out_left = np.zeros((C, n), np.int16)
        for wi in range(wi0, len(self.weeks)):
            G = rng.standard_normal((C, self.n_game[wi]), dtype=np.float32)
            T = rng.standard_normal((C, self.n_team[wi]), dtype=np.float32)
            z = self.load * (np.float32(np.sqrt(g)) * G[:, self.game[wi]] + np.float32(np.sqrt(1 - g)) * T[:, self.team[wi]])
            z += self.cload * rng.standard_normal((C, n), dtype=np.float32)
            u = ndtr(z) * np.float32(Q - 1)
            i0 = np.minimum(u.astype(np.int64), Q - 2)
            fr = u - i0
            base = np.arange(n) * Q + i0
            q = self.qflat[wi]
            score = np.zeros((C, n + 1), np.float32)
            avail = np.zeros((C, n + 1), bool)
            s = q[base] * (1 - fr) + q[base + 1] * fr
            if wi == 0:
                avail[:, :n] = rng.random((C, n), dtype=np.float32) >= self.p_out
                if self.locked:
                    s = np.where(self.started, self.pts0 + self.frac_left * s, s)
                    avail[:, :n] |= self.started
            else:
                loglev += rng.standard_normal((C, n), dtype=np.float32) * np.float32(LEVEL_STEP)
                s *= np.exp(loglev - np.float32(0.5 * LEVEL_STEP ** 2 * wi))
                new = (out_left == 0) & (rng.random((C, n), dtype=np.float32) < self.haz)
                col = np.nonzero(new)[1]
                out_left[new] = np.where(rng.random(len(col)) < self.p_long[col], INJ_LONG_WEEKS,
                                         rng.geometric(self.p_geo[col]))
                avail[:, :n] = out_left == 0
                out_left[out_left > 0] -= 1
            score[:, :n] = s
            yield wi, score, avail

    def lineup_idx(self, wi, roster, starters=None):
        """(3, 8) indices: starters, first and second bench fallback (n = empty).

        With `starters` (the team's Sleeper lineup by slot) in a week under way, locked starters keep their slot,
        locked bench players stay out, and the open slots and their fallbacks go to the best unlocked players.
        """
        slots = self.lg.slots
        lock = self.locked if wi == 0 and starters is not None else set()
        fixed = [p if p in lock and p in self.ix else None for p in (starters or [None] * len(slots))[:len(slots)]]
        fixed += [None] * (len(slots) - len(fixed))
        free = [k for k, p in enumerate(fixed) if p is None]
        cands = [(self.proj[wi, self.ix[p]], p, self.pos[self.ix[p]]) for p in roster if p in self.ix and p not in lock]
        lu, subs = best_lineup([slots[k] for k in free], cands, with_subs=True)
        S = [self.ix[p] if p else self.n for p in fixed]
        B1, B2 = [self.n] * len(slots), [self.n] * len(slots)
        for (s, p), sb in zip(lu, subs):
            k = next(k for k in free if slots[k] == s)
            free.remove(k)
            S[k] = self.ix[p] if p else self.n
            B1[k] = self.ix[sb[0]] if len(sb) > 0 else self.n
            B2[k] = self.ix[sb[1]] if len(sb) > 1 else self.n
        return np.array([S, B1, B2])

    def fixed_lineup_idx(self, wi, starters, roster):
        """Indices for a hand-picked starting eight (subs = best eligible bench)."""
        slots = self.lg.slots
        val = lambda p: self.proj[wi, self.ix[p]]
        left = sorted(starters, key=lambda p: -val(p))
        assign = [None] * len(slots)
        for k, s in enumerate(slots):          # fixed positions take the best of their position
            if s in POS:
                p = next((p for p in left if self.pos[self.ix[p]] == s), None)
                if p:
                    assign[k] = p
                    left.remove(p)
        for k, s in enumerate(slots):
            if assign[k] is None:
                p = next((p for p in left if eligible(s, self.pos[self.ix[p]])), None)
                if p:
                    assign[k] = p
                    left.remove(p)
        lock = self.locked if wi == 0 else set()  # a bench player whose game has started can't come in
        bench = sorted((p for p in roster if p in self.ix and p not in starters and p not in lock and val(p) > 0), key=lambda p: -val(p))
        S = [self.ix[p] if p else self.n for p in assign]
        B = [[self.ix[b] for b in bench if eligible(s, self.pos[self.ix[b]])][:2] for s in slots]
        return np.array([S, [b[0] if b else self.n for b in B], [b[1] if len(b) > 1 else self.n for b in B]])


def team_scores(score, avail, idx):
    """Team totals (C, V) for lineup index arrays idx (3, V, 8)."""
    S, B1, B2 = idx
    x = np.where(avail[:, S], score[:, S],
                 np.where(avail[:, B1], score[:, B1], np.where(avail[:, B2], score[:, B2], 0.0)))
    return x.sum(-1)


def _pool_map(fn, jobs):
    if THREADS <= 1 or len(jobs) == 1:
        return [fn(j) for j in jobs]
    os.environ["GSIM_WORKER"] = "1"
    with mp.get_context("fork").Pool(THREADS) as pool:
        if not os.environ.get("GSIM_PROGRESS"):
            return pool.map(fn, jobs, chunksize=1)
        out, n = [], len(jobs)  # live.py's progress panel reads these lines (every whole percent)
        for i, r in enumerate(pool.imap(fn, jobs, chunksize=1), 1):
            out.append(r)
            if i == n or i * 100 // n != (i - 1) * 100 // n:
                print(f"@@progress {i} {n}", flush=True)
        return out


# ---------------------------------------------------------------- season / elimination sims
class SeasonPlan:
    """Opponents keep their rosters; V variants of my roster, each with its own starting alive set."""

    def __init__(self, U, teams, rosters, me, my_variants, alive0=None, wi0=0):
        self.teams = list(teams)                  # roster ids, column order
        self.me_t = self.teams.index(me)
        self.wi0 = wi0
        W = len(U.weeks)
        st = U.starters0.get
        self.opp_idx = {wi: np.stack([U.lineup_idx(wi, rosters[t], st(t)) for t in self.teams], 1) for wi in range(wi0, W)}
        self.my_idx = {wi: np.stack([U.lineup_idx(wi, r, st(me)) for r in my_variants], 1) for wi in range(wi0, W)}
        V, T = len(my_variants), len(self.teams)
        self.alive0 = np.ones((V, T), bool) if alive0 is None else np.asarray(alive0, bool)


def _season_chunk(job):
    ci, C, seed = job
    U, P = _G["U"], _G["plan"]
    rng = np.random.default_rng([seed, ci])
    V, T = P.alive0.shape
    W = len(U.weeks)
    alive = np.broadcast_to(P.alive0, (C, V, T)).copy()
    chops = np.zeros((V, T, W), np.int64)
    line_hist = np.zeros((W, 400), np.int64)   # chop line (lowest opponent score), 0.5-pt bins, variant 0
    my_hist = np.zeros((W, 400), np.int64)     # my score, variant 0
    offs = (np.arange(V) * T)[None, :]
    for wi, score, avail in U.paths(C, rng, P.wi0):
        if alive[0, 0].sum() <= 1:
            break
        opp = team_scores(score, avail, P.opp_idx[wi])
        mine = team_scores(score, avail, P.my_idx[wi])
        sc = np.repeat(opp[:, None, :], V, axis=1)
        sc[:, :, P.me_t] = mine
        sc[~alive] = np.inf
        k = sc.argmin(-1)
        np.put_along_axis(alive, k[..., None], False, axis=-1)
        chops[:, :, wi] += np.bincount((k + offs).ravel(), minlength=V * T).reshape(V, T)
        live0 = sc[:, 0, :].copy()
        live0[:, P.me_t] = np.inf
        ok = alive[:, 0, P.me_t] | (k[:, 0] == P.me_t)   # I was alive going into this week
        line_hist[wi] += np.bincount(np.clip(live0.min(-1)[ok] * 2, 0, 399).astype(int), minlength=400)
        my_hist[wi] += np.bincount(np.clip(mine[ok, 0] * 2, 0, 399).astype(int), minlength=400)
    wins = np.bincount((alive.argmax(-1) + offs).ravel(), minlength=V * T).reshape(V, T)
    return chops, wins, line_hist, my_hist


def run_season(U, plan, n_sims, seed=1):
    _G["U"], _G["plan"] = U, plan
    n_chunks = max(1, n_sims // CHUNK)
    res = _pool_map(_season_chunk, [(ci, CHUNK, seed) for ci in range(n_chunks)])
    chops = sum(r[0] for r in res)
    wins = sum(r[1] for r in res)
    N = n_chunks * CHUNK
    return {"N": N, "chop": chops / N, "win": wins / N,
            "line_hist": sum(r[2] for r in res), "my_hist": sum(r[3] for r in res)}


# ---------------------------------------------------------------- one-week lineup sims
def _week_chunk(job):
    ci, C, seed = job
    U, opp_idx, my_idx, me_t = _G["U"], _G["opp_idx"], _G["my_idx"], _G["me_t"]
    rng = np.random.default_rng([seed, ci])
    wi, score, avail = next(U.paths(C, rng, 0))
    opp = team_scores(score, avail, opp_idx)
    opp[:, me_t] = np.inf
    line = opp.min(-1)
    mine = team_scores(score, avail, my_idx)
    return (mine < line[:, None]).sum(0), mine.sum(0), (mine ** 2).sum(0)


def run_week_lineups(U, opp_idx, my_idx, me_t, n_sims, seed=2):
    _G.update(U=U, opp_idx=opp_idx, my_idx=my_idx, me_t=me_t)
    n_chunks = max(1, n_sims // CHUNK)
    res = _pool_map(_week_chunk, [(ci, CHUNK, seed) for ci in range(n_chunks)])
    N = n_chunks * CHUNK
    chopped = sum(r[0] for r in res) / N
    mean = sum(r[1] for r in res) / N
    sd = np.sqrt(np.maximum(sum(r[2] for r in res) / N - mean ** 2, 0))
    return {"N": N, "p_chop": chopped, "mean": mean, "sd": sd}


def enumerate_lineups(U, wi, roster, slots, starters=None):
    """Every legal starting eight from players projected > 0 (by position counts).

    In a week under way, `starters` (my Sleeper lineup) pins it: every locked starter is in, no locked bench player is.
    """
    lock = U.locked if wi == 0 and starters is not None else set()
    must = {p for p in starters or [] if p in lock}
    pool = [p for p in roster if p in U.ix and (p in must or (U.proj[wi, U.ix[p]] > 0 and p not in lock))]
    need = {p: sum(1 for s in slots if s == p) for p in POS}
    n_flex = sum(1 for s in slots if s not in POS)
    rest = [p for p in pool if p not in must]
    out = []
    for combo in combinations(rest, len(slots) - len(must)):
        combo = [p for p in pool if p in must] + list(combo)
        c = {p: 0 for p in POS}
        for p in combo:
            c[U.pos[U.ix[p]]] += 1
        if c["QB"] != need["QB"] or any(c[p] < need[p] for p in ("RB", "WR", "TE")):
            continue
        if sum(c[p] - need[p] for p in ("RB", "WR", "TE")) != n_flex:
            continue
        out.append(combo)
    return out
