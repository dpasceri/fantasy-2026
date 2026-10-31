"""Variance-reduced estimators for the mc track: condition on the copula factors, then combine teams.

Given one week's game and team factors F, every alive team's total is independent of every other's:
each player sits on exactly one roster, and everything else Universe.paths draws (idiosyncratic eps,
availability, level drift, injuries) is per player. So draw F once per group, K inner draws per team
on that F, and score many team combinations instead of only the K drawn paths:
  week    all K^T combinations, exactly (product of empirical survival functions, one sort per group)
  season  R rounds of per-team cyclic shifts within each group (random recombination of team paths)
This is the product-form / generalized U-statistic estimator of Kuntz, Crucinio & Johansen (2022)
applied conditionally on the factors (as in factor-copula credit models, Glasserman & Li 2005); it is
unbiased for any K. Layers on top, all keeping every single path's law exact:
  outer   scrambled Sobol' factors (importance-ordered dims) or antithetic pairs (F, -F)
  inner   Latin hypercube over the K draws of a group (random permutation per (group, player)) and/or
          antithetic pairs within the group, per noise kind (eps, availability, level steps, injuries)
  Rp      player-level recombination: Rp - 1 extra "virtual" draws per draw, every lineup component (week:
          slots linked by a shared sub; season: a player's whole path) read from its own cyclic shift of the
          group's K draws, all scored in the same combination statistic

The sampler re-implements sim.Universe.paths for players that sit in a lineup slot (starter or bench
sub) that week; others are skipped for scores but keep their level drift and injury state.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)

import warnings

import numpy as np
import sim
from model import Q
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

F32 = np.float32
POOL = 1024            # random permutations per call used for LHS (each composed with a random cyclic shift)
with warnings.catch_warnings():                 # scipy loads Sobol' direction numbers lazily on first use: do it at
    warnings.simplefilter("ignore")             # import (target-independent library init, like importing scipy)
    qmc.Sobol(2, scramble=False).random(2)


class Prep:
    """Per-target arrays: m = players in any lineup over the horizon; per week the players in that week's lineups."""

    def __init__(self, tg):
        U, W = tg.U, tg.horizon
        self.T, self.W = tg.T, W
        idx = {w: tg.idx[w].copy() for w in range(W)}
        # week 0 has no injury hazard: a sub behind a starter who always plays (p_out = 0) is never used,
        # nor a second sub behind a first sub who always plays, so point those at the empty column
        sure = np.append(U.p_out == 0, False)
        S0, B1, B2 = idx[0]
        idx[0][1] = np.where(sure[S0], U.n, B1)
        idx[0][2] = np.where(sure[S0] | sure[B1], U.n, B2)
        used = np.unique(np.concatenate([idx[w].ravel() for w in range(W)]))
        cols = used[used < U.n]
        self.m, self.cols = len(cols), cols
        pos = np.full(U.n + 1, -1, np.int64)
        pos[cols] = np.arange(self.m)
        g = U.pm.game_share
        self.rel, self.lidx, self.tab, self.base, self.ag, self.at, self.cl = [], [], [], [], [], [], []
        self.fcol, self.nf, self.forder, self.cmap = [], [], [], []
        for wi in range(W):
            r = np.unique(idx[wi])
            r = r[r < U.n]
            nr = len(r)
            loc = np.full(U.n + 1, nr, np.int64)
            loc[r] = np.arange(nr)
            self.rel.append(pos[r])
            self.cmap.append(np.append(pos[r], 0).astype(np.int64))
            self.lidx.append(loc[idx[wi]])                                  # (3, T, 8), nr = empty
            q = U.qflat[wi].reshape(U.n, Q)[r].astype(F32)
            tab = np.zeros((nr, Q), np.complex64)                              # (q, q[+1] - q): one gather
            tab.real, tab.imag[:, :-1] = q, np.diff(q, axis=1)
            self.tab.append(tab.ravel())
            self.base.append((np.arange(nr) * Q).astype(np.int32))
            ld = U.load[r]
            self.ag.append(F32(np.sqrt(g)) * ld)
            self.at.append(F32(np.sqrt(1 - g)) * ld)
            self.cl.append(U.cload[r])
            # compact factor columns: game factors then team factors that some lineup player loads on
            gid, tid = U.game[wi][r], U.team[wi][r]
            ug, gi = np.unique(gid, return_inverse=True)
            ut, ti = np.unique(tid, return_inverse=True)
            self.fcol.append((gi.astype(np.int64), (len(ug) + ti).astype(np.int64)))
            self.nf.append(len(ug) + len(ut))
            sd = q.std(1)
            wgt = np.bincount(gi, (self.ag[-1] * sd) ** 2, len(ug)).tolist() + \
                np.bincount(ti, (self.at[-1] * sd) ** 2, len(ut)).tolist()
            self.forder.append(np.argsort(-np.array(wgt), kind="stable"))       # Sobol dim order: heaviest first
        r0 = self.rel[0]
        po = U.p_out[cols[r0]]
        self.qi = np.flatnonzero(po > 0)                                       # week-0 columns that can sit
        self.p_out_q = po[self.qi]
        # week-0 totals from one score array with NaN = inactive: starters, then the sub cascade only on slots
        # whose starter can sit (or is empty)
        nr0 = len(r0)
        S0, B1, B2 = (a.ravel() for a in self.lidx[0])
        okc = np.append(po == 0, False)
        qs = np.flatnonzero(~okc[S0])
        # slot components: slots linked through a shared player (a sub behind two starters) move together in a
        # virtual draw; different components share no player, so they are independent given the factors
        par = list(range(len(S0)))

        def root(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        seen = {}
        for j in range(len(S0)):
            for c in (S0[j], B1[j], B2[j]):
                if c < nr0:
                    if c in seen:
                        par[root(j)] = root(seen[c])
                    else:
                        seen[c] = j
        comp = np.array([root(j) for j in range(len(S0))])
        team = np.arange(len(S0)) // (len(S0) // self.T)
        key = team * len(S0) + comp
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        cteam = team[order][starts]
        self.w0 = {"S": S0, "qs": qs, "ent": np.concatenate([S0, B1[qs], B2[qs]]).astype(np.int32),
                   "order": order, "starts": starts, "ncomp": len(starts),
                   "tstarts": np.flatnonzero(np.r_[True, cteam[1:] != cteam[:-1]])}
        assert len(self.w0["tstarts"]) == self.T

    def zf(self, wi, Fz):
        """Factor part of z, (NF, nr), from compact factor normals (NF, nf)."""
        gi, ti = self.fcol[wi]
        return self.ag[wi] * Fz[:, gi] + self.at[wi] * Fz[:, ti]

    def score(self, wi, z):
        """Quantile-table interpolation exactly as Universe.paths (one complex64 gather of (q, dq))."""
        u = ndtr(z) * F32(Q - 1)
        i0 = np.minimum(u.astype(np.int32), Q - 2)
        fr = u - i0.astype(F32)
        v = self.tab[wi][self.base[wi] + i0]
        return v.real + fr * v.imag


# ---------------------------------------------------------------- randomness
class Noise:
    """Inner draws for groups of K: iid; Latin hypercube over the K draws of each (group, column) for the kinds in
    lhs ('eps' score noise, 'av' week-0 availability, 'lvl' level steps, 'inj' injury uniforms); or antithetic
    normals (draw k + K/2 = -draw k) for the kinds in anti."""
    KINDS = ("eps", "av", "lvl", "inj")

    def __init__(self, rng, K, lhs, anti=()):
        self.rng, self.K = rng, K
        self.lhs = set() if not lhs or K == 1 else set(self.KINDS) if lhs is True else set(lhs)
        self.anti = set(anti) - self.lhs if K % 2 == 0 else set()
        if self.lhs:
            self.pool = np.argsort(rng.random((POOL, K)), axis=1).astype(np.int16)

    def _strata(self, NF, c):
        rng, K = self.rng, self.K
        p = self.pool.T[:, rng.integers(0, POOL, (NF, c))]                    # (K, NF, c)
        p = (p + rng.integers(0, K, (NF, c), dtype=np.int16)[None]) % np.int16(K)
        return np.ascontiguousarray(p.transpose(1, 0, 2))                     # (NF, K, c)

    def uniform(self, NF, c, kind):
        if kind not in self.lhs:
            return self.rng.random((NF, self.K, c), dtype=F32)
        return (self._strata(NF, c) + self.rng.random((NF, self.K, c), dtype=F32)) * F32(1.0 / self.K)

    def normal(self, NF, c, kind):
        if kind in self.anti:
            h = self.rng.standard_normal((NF, self.K // 2, c), dtype=F32)
            return np.concatenate([h, -h], 1)
        if kind not in self.lhs:
            return self.rng.standard_normal((NF, self.K, c), dtype=F32)
        u = self.uniform(NF, c, kind)                                         # float32 rounding can hit 0 or 1:
        return ndtri(np.clip(u, F32(2.0 ** -30), F32(1 - 2.0 ** -24), out=u))  # keep z finite (|z| <= 6.1)


def factors(rng, NF, nf, kind, order=None):
    """(NF, nf) factor normals: 'iid', 'anti' (rows in (x, -x) pairs) or 'sobol' (scrambled Sobol', LMS + digital
    shift; dim j of the net goes to factor order[j]); 'sobol_pad' also permutes the rows at random (one call per
    week: Owen's padding / Latin supercube, so different weeks' nets pair up at random)."""
    if kind == "iid":
        return rng.standard_normal((NF, nf), dtype=F32)
    if kind == "anti":
        h = rng.standard_normal(((NF + 1) // 2, nf), dtype=F32)
        return np.stack([h, -h], 1).reshape(-1, nf)[:NF]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                       # non power-of-2 n is fine (still unbiased)
        u = qmc.Sobol(nf, scramble=True, seed=rng).random(NF)
    if kind == "sobol_pad":                                                   # padding: rows randomly permuted, so
        u = u[rng.permutation(NF)]                                            # blocks (weeks) pair up at random
    out = np.empty((NF, nf), F32)
    out[:, order] = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    return out


# ---------------------------------------------------------------- week
def week_draws(P, Fz, noise):
    """(NF, K, nr + 1) week-0 scores of lineup players on factor rows Fz (NF, nf); NaN = inactive (last column
    = the empty slot)."""
    NF, K, nr = len(Fz), noise.K, len(P.rel[0])
    z = P.zf(0, Fz)[:, None, :] + P.cl[0] * noise.normal(NF, nr, "eps")
    sn = np.empty((NF, K, nr + 1), F32)
    sn[:, :, :nr] = P.score(0, z)
    sn[:, :, nr] = np.nan
    out = noise.uniform(NF, len(P.qi), "av") < P.p_out_q
    q = sn[:, :, P.qi]
    q[out] = np.nan
    sn[:, :, P.qi] = q
    return sn


def week_slots(P, sn):
    """(NF, K, ncomp) component sums of week-0 slot values (sim.team_scores' sub cascade) from NaN-coded scores."""
    w = P.w0
    V = sn[:, :, w["ent"]]
    nS, nq = len(w["S"]), len(w["qs"])
    x = V[:, :, :nS]
    xq = x[:, :, w["qs"]]
    for j in (1, 2):
        m = np.isnan(xq)
        xq[m] = V[:, :, nS + (j - 1) * nq:nS + j * nq][m]
    xq[np.isnan(xq)] = 0
    x[:, :, w["qs"]] = xq
    return np.add.reduceat(x[:, :, w["order"]], w["starts"], axis=2)


def team_sum(P, y):
    return np.add.reduceat(y, P.w0["tstarts"], axis=2)


def virtual(y2, K, shift):
    """Virtual draws (NF, K, ncomp): component c of draw k read from draw k + shift[g, c] of y2 = y doubled along K."""
    NF, _, nc = y2.shape
    base = (np.arange(NF, dtype=np.int32) * (2 * K * nc))[:, None] + shift * nc + np.arange(nc, dtype=np.int32)
    return y2.reshape(-1)[base[:, None, :] + (np.arange(K, dtype=np.int32) * nc)[None, :, None]]


def chop_vstat(X):
    """X (NF, M, T) team totals, independent across teams within a group. Returns (T,) sums over groups of
    mean_k prod_{s != t} Shat_s(X_t^k), Shat_s = share of team s's M draws above the value: the average of
    1{t lowest} over all M^T team combinations of each group. One merged sort per group; walking up the
    sorted values, each passed draw of team s multiplies Shat_s by (M - c - 1) / (M - c), so the log of the
    product over all teams is a running sum and the own-team factor is known from the draw's rank."""
    NF, M, T = X.shape
    assert M <= 1024 and T <= 16
    Xs = np.sort(np.ascontiguousarray(X.transpose(0, 2, 1), dtype=F32), axis=2)          # (NF, T, M) by team
    v = Xs.view(np.int32)
    key = np.where(v < 0, v ^ np.int32(0x7FFFFFFF), v).astype(np.int64) + (1 << 31)    # float order, >= 0
    key = (key << 14) | (np.arange(T, dtype=np.int64) << 10)[None, :, None] | np.arange(M, dtype=np.int64)
    key = key.reshape(NF, T * M)
    key.sort(axis=1)
    k = (key & 1023).astype(np.int32)                                                   # rank within its team
    with np.errstate(divide="ignore"):
        lf = np.log((M - np.arange(M + 1)) / M)                                         # log Shat after c passed
    inc = (lf[1:] - lf[:-1])[k]                                                         # -inf at a team's last
    L = np.zeros_like(inc)
    np.cumsum(inc[:, :-1], axis=1, out=L[:, 1:])                                        # sum_s log Shat_s, before
    w = np.exp(L - lf[k])
    return np.bincount(((key >> 10) & 15).ravel(), weights=w.ravel(), minlength=T)[:T] / M


def week_estimate(P, N, rng, outer="sobol", K=16, lhs=False, Rp=1, chunk=2048):
    """P(chopped this week) per team from N inner draws (N // K factor groups), each group scored on its K
    draws plus (Rp - 1) * K virtual draws, all M = Rp * K per team in the all-combination statistic."""
    NF = max(2, N // K)
    Fz = factors(rng, NF, P.nf[0], outer, P.forder[0])
    noise = Noise(rng, K, lhs)
    acc = np.zeros(P.T)
    gc = max(1, chunk // K)
    for a in range(0, NF, gc):
        y = week_slots(P, week_draws(P, Fz[a:a + gc], noise))
        nb, nc = len(y), y.shape[2]
        Xs = [team_sum(P, y)]
        if Rp > 1:
            y2 = np.concatenate([y, y], 1)
            for _ in range(Rp - 1):
                Xs.append(team_sum(P, virtual(y2, K, rng.integers(0, K, (nb, nc), dtype=np.int32))))
        acc += chop_vstat(np.concatenate(Xs, 1) if Rp > 1 else Xs[0])
    return acc / NF


# ---------------------------------------------------------------- season
def nan_totals(sn, lidx, shift=None, cmap=None, sn2=None):
    """(NF, K, T) team totals (sim.team_scores) from NaN-coded scores sn (NF, K, nr + 1) and lineup idx (3, T, 8).
    Virtual draws: shift (NF, m) per player (cmap: local col -> player), col of draw k read from draw
    (k + shift) % K; sn2 = sn doubled along K. Starters are gathered densely, subs only where a starter sits."""
    NF, K, n1 = sn.shape
    S, B1, B2 = (a.reshape(-1) for a in lidx)
    if shift is None:
        x = sn[:, :, S]
    else:
        base = (np.arange(NF, dtype=np.int32) * (2 * K * n1))[:, None] + shift[:, cmap[S]] * n1 + S.astype(np.int32)
        x = sn2.reshape(-1)[base[:, None, :] + (np.arange(K, dtype=np.int32) * n1)[None, :, None]]
    pos = np.flatnonzero(np.isnan(x))
    for B in (B1, B2):
        if not len(pos):
            break
        g, r = np.divmod(pos, K * len(S))
        k, j = np.divmod(r, len(S))
        col = B[j]
        if shift is not None:
            k = (k + shift[g, cmap[col]]) % K
        x.flat[pos] = sn[g, k, col]
        pos = pos[np.isnan(x.flat[pos])]
    x.flat[pos] = 0
    return x.reshape(NF, K, lidx.shape[1], -1).sum(-1)


def season_draws(P, NF, rng, noise, outer="anti"):
    """Per week, NaN-coded (inactive) scores (NF, K, nr_w + 1) of that week's lineup players: factors shared
    within each group of K, everything else (eps, availability, level drift, injuries) per path."""
    K, m, W = noise.K, P.m, P.W
    C = NF * K
    loglev = np.zeros((C, m), F32)
    out_left = np.zeros((C, m), np.int16)
    sns = []
    for wi in range(W):
        rel, nr = P.rel[wi], len(P.rel[wi])
        Fz = factors(rng, NF, P.nf[wi], outer, P.forder[wi])
        z = (P.zf(wi, Fz)[:, None, :] + P.cl[wi] * noise.normal(NF, nr, "eps")).reshape(C, nr)
        sn = np.empty((C, nr + 1), F32)
        sn[:, :nr] = P.score(wi, z)
        sn[:, nr] = np.nan
        if wi == 0:
            q = sn[:, P.qi]
            q[noise.uniform(NF, len(P.qi), "av").reshape(C, -1) < P.p_out_q] = np.nan
            sn[:, P.qi] = q
        else:
            loglev += noise.normal(NF, m, "lvl").reshape(C, m) * F32(sim.LEVEL_STEP)
            sn[:, :nr] *= np.exp(loglev[:, rel] - F32(0.5 * sim.LEVEL_STEP ** 2 * wi))
            new = (out_left == 0) & (noise.uniform(NF, m, "inj").reshape(C, m) < F32(sim.INJ_HAZARD))
            out_left[new] = rng.geometric(1.0 / sim.INJ_MEAN_WEEKS, int(new.sum()))
            sn[:, :nr][out_left[:, rel] != 0] = np.nan
            out_left[out_left > 0] -= 1
        sns.append(sn.reshape(NF, K, nr + 1))
    return sns


def season_totals(P, sns, shift=None, sns2=None):
    """(NF, K, T, W) weekly team totals of the drawn paths, or of virtual paths (shift (NF, m): each player's
    whole season path read from draw (k + shift) % K of its group)."""
    NF, K = sns[0].shape[:2]
    X = np.empty((NF, K, P.T, P.W), F32)
    for wi, sn in enumerate(sns):
        X[:, :, :, wi] = nan_totals(sn, P.lidx[wi], shift, P.cmap[wi], None if shift is None else sns2[wi])
    return X


def season_wins(Xc):
    """Xc (C, T, W) weekly totals of C combined paths -> (T,) win counts (chop the lowest alive each week)."""
    C, T, W = Xc.shape
    pen = np.zeros((C, T), F32)
    rows = np.arange(C)
    for wi in range(min(W, T - 1)):
        k = (Xc[:, :, wi] + pen).argmin(-1)
        pen[rows, k] = np.inf
    return np.bincount(pen.argmin(-1), minlength=T)


def recombine(X, R, rng):
    """Season wins over R rounds of team recombination within groups: round 0 plays the paths as drawn; round r
    gives team t's M draws a random cyclic shift c[g, r, t] (independent of the draws), so every combined path
    has the exact joint law given the group's factors. (T,) counts over NF*M*R paths."""
    NF, M, T, W = X.shape
    out = season_wins(X.reshape(NF * M, T, W))
    if R > 1:
        kk = np.arange(M)[None, :, None]
        gg = np.arange(NF)[:, None, None]
        tt = np.arange(T)[None, None, :]
        for _ in range(R - 1):
            c = rng.integers(0, M, (NF, 1, T))
            out = out + season_wins(X[gg, (kk + c) % M, tt].reshape(NF * M, T, W))
    return out


def season_estimate(P, N, rng, outer="anti", K=16, lhs=False, R=16, Rp=1, anti=(), chunk=2048):
    """P(win the league) per team from N drawn season paths (N // K groups), plus (Rp - 1) * K virtual paths per
    group (player-level recombination), then R rounds of team recombination over the M = Rp * K paths."""
    NF = max(2, N // K)
    NF += NF % 2                                                     # whole antithetic pairs
    noise = Noise(rng, K, lhs, anti)
    gc = max(2, (chunk // K) // 2 * 2)
    wins = np.zeros(P.T)
    for a in range(0, NF, gc):
        nb = min(gc, NF - a)
        sns = season_draws(P, nb, rng, noise, outer)
        X = [season_totals(P, sns)]
        if Rp > 1:
            sns2 = [np.concatenate([s, s], 1) for s in sns]
            for _ in range(Rp - 1):
                X.append(season_totals(P, sns, rng.integers(0, K, (nb, P.m), dtype=np.int32), sns2))
        wins += recombine(np.concatenate(X, 1) if Rp > 1 else X[0], R, rng)
    return wins / (NF * K * Rp * R)
