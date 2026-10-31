"""Team-track candidate: tail-dependent copulas on the baseline's marginals.

Only the dependence changes. Marginals are _baseline_player's ratio tables and the factor loadings a_pos and
game share g are _baseline_team.fit_copula's, all refit per origin on train items. Every variant draws a
uniform u per player (exactly U(0, 1) marginally) and reads the baseline's 256-level quantile grid, so each
player's score distribution is the baseline's.

Variants:
- gt: grouped t copula on the baseline's factor structure. The factor z is scaled by sqrt(nu / W),
  W ~ chi2(nu), and mapped with the t_nu CDF. W is drawn per game ("game": each game an independent t
  copula, i.e. joint extremes for one game's players: blowouts, shootouts, weather) or once per draw for the
  whole week ("slate": tail dependence across games too). Group and nu are picked per origin by pairwise
  composite likelihood (CL) over every same-week pair of players projected >= 3; the Gaussian baseline
  (bit-identical draws) is a candidate and wins ties.
- slot_t: depth-chart copula. Players map to slots (QB1, RB1-3, WR1-4, TE1-2 by projection rank on their
  NFL team that week); a game is a 20-dim Gaussian/t vector (10 slots per side) with a slot correlation
  matrix: the training games' normal-score correlations (same team and opponent blocks, home/away
  symmetrized), shrunk entrywise towards the baseline factor model by empirical Bayes
  (w = tau^2 / (tau^2 + 1/n), tau^2 = excess spread of empirical vs factor correlations), then made PSD.
  It can express what one positive factor cannot: QB1-WR1 stronger than QB-WR3, negative RB1-RB2. Group and
  nu are refit by CL with this matrix. Unslotted players keep the baseline factor draw.
- shuffle: empirical game-block copula (Schaake shuffle / ECC-style rank reordering). Each draw and game
  copies the slot pattern of a random training game (random orientation): a player's u is the randomized
  rank of that template team-week's residual among all templates' values for his slot (checkerboard
  empirical copula: u exactly uniform). Keeps any tail asymmetry, negative pairs and opponent effects.
  Slots a template lacks and unslotted players fall back to the gt draw.
Games are independent blocks except under a selected "slate" group (as in the baseline, which has none).
Draws: 96000 per week in chunks of sim.CHUNK with rng = default_rng([11, season, week, chunk]); each chunk
first draws the baseline's normals in the baseline's order (common random numbers), extras come after.
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import numpy as np
from bench.candidates._baseline_player.ratio import fit_tables, ratio_grid
from bench.candidates._baseline_team.copula import S_DRAWS, SEED, fit_copula
from bench.harness.metrics import LEVELS
from common import POS
from model import CORR_MIN_PROJ, Q
from scipy.special import gammaln, ndtr, ndtri, stdtr, stdtrit
from sim import CHUNK

NU_GRID = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 17.0, 25.0, 40.0, 70.0, 150.0)
CAP = {"QB": 1, "RB": 3, "WR": 4, "TE": 2}       # depth-chart slots per NFL team
SLOTS = [f"{p}{k}" for p in POS for k in range(1, CAP[p] + 1)]
SLOT_IX = {s: i for i, s in enumerate(SLOTS)}
K = len(SLOTS)
MIN_N = 20                                       # pairs seen fewer times keep the factor-model correlation


# ---------------------------------------------------------------- fitting
def pit_z(pos, proj, act, seed=7):
    """fit_copula's randomized-PIT normal scores: same rows, buckets and random numbers."""
    pos, proj, act = np.asarray(pos), np.asarray(proj, float), np.asarray(act, float)
    rng = np.random.default_rng(seed)
    ratio = act / proj
    z = np.zeros(len(proj))
    for P in POS:
        m = np.flatnonzero(pos == P)
        if not len(m):
            continue
        nb = int(np.clip(len(m) // 300, 5, 14))
        edges = np.quantile(proj[m], np.linspace(0, 1, nb + 1))
        b = np.clip(np.searchsorted(edges, proj[m], side="right") - 1, 0, nb - 1)
        for k in range(nb):
            idx = m[b == k]
            if not len(idx):
                continue
            r = np.sort(ratio[idx])
            lo = np.searchsorted(r, ratio[idx], side="left")
            hi = np.searchsorted(r, ratio[idx], side="right")
            u = (lo + rng.random(len(idx)) * (hi - lo)) / len(r)
            z[idx] = ndtri(np.clip(u, 1e-6, 1 - 1e-6))
    return z


def week_pairs(t, proj, game_key):
    """Every pair of players projected >= 3 in the same week: (I, J, same_game)."""
    rel = proj >= CORR_MIN_PROJ
    I, J = [], []
    for tt in np.unique(t[rel]):
        idx = np.flatnonzero(rel & (t == tt))
        iu, ju = np.triu_indices(len(idx), 1)
        I.append(idx[iu])
        J.append(idx[ju])
    I, J = np.concatenate(I), np.concatenate(J)
    return I, J, game_key[I] == game_key[J]


def fit_tail(z, I, J, rho, sg):
    """Grouped-t choice by pairwise composite likelihood (CL) of the week's pairs.

    gauss: the Gaussian copula (pairs across games independent). game: one chi2 mixing variable per game, so
    same-game pairs are bivariate t(rho, nu) and pairs across games stay independent. slate: one mixing
    variable per draw for the whole week, so pairs across games are t(0, nu) (tail dependent, uncorrelated).
    Returns (group, nu, {f"{group}{nu}": CL - CL_gauss}); the Gaussian wins ties."""
    zs, zr, rs = z[I[sg]], z[J[sg]], rho[sg]
    r2 = 1 - rs ** 2
    cl_g = float(np.sum(-0.5 * np.log(r2) - (rs ** 2 * (zs ** 2 + zr ** 2) - 2 * rs * zs * zr) / (2 * r2)))
    u = ndtr(z)
    r2 = 1 - rho ** 2
    out = {("gauss", np.inf): 0.0}
    for nu in NU_GRID:
        x = stdtrit(nu, u)
        l1 = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(x ** 2 / nu)
        xi, xj = x[I], x[J]
        lc = (-np.log(2 * np.pi) - 0.5 * np.log(r2) - (nu + 2) / 2 * np.log1p((xi ** 2 - 2 * rho * xi * xj + xj ** 2) / (nu * r2))
              - l1[I] - l1[J])
        out["game", nu] = float(lc[sg].sum()) - cl_g
        out["slate", nu] = float(lc.sum()) - cl_g
    (grp, nu), _ = max(out.items(), key=lambda kv: (round(kv[1], 6), kv[0][1]))
    return grp, nu, {f"{k[0]}{k[1]:g}": round(v, 1) for k, v in out.items()}


def slot_index(d):
    """(slot (n,) -1 = none, team-week id (n,)) for training rows: projection rank on the NFL team."""
    rk = d.groupby(["t", "team", "pos"])["proj"].rank(ascending=False, method="first").to_numpy()
    pos = d["pos"].to_numpy(str)
    slot = np.array([SLOT_IX.get(f"{p}{int(r)}", -1) for p, r in zip(pos, rk)], np.int64)
    key = np.char.add(np.char.add(d["t"].to_numpy().astype(str), "|"), d["team"].astype(str).to_numpy())
    _, tw = np.unique(key, return_inverse=True)
    return slot, tw


def build_templates(d, z, slot, tw):
    """Training games as slot templates.

    pairs (n_g, 2) team-week ids of every game with both teams; rank (n_tw, K) float (NaN = slot empty) =
    0-based rank of the slot's residual among the template team-weeks' values for that slot, so
    (rank + U(0,1)) / n_slot is exactly uniform over a random team-week; Zs = the normal scores
    ndtri((rank + 0.5) / n_slot)."""
    g_key = np.char.add(np.char.add(d["t"].to_numpy().astype(str), "|"), d["game"].astype(str).to_numpy())
    games = {}
    for k, gk in zip(tw, g_key):
        games.setdefault(gk, set()).add(int(k))
    pairs = np.array(sorted(tuple(sorted(s)) for s in games.values() if len(s) == 2), np.int64).reshape(-1, 2)
    n_tw = int(tw.max()) + 1
    Z = np.full((n_tw, K), np.nan)
    m = slot >= 0
    Z[tw[m], slot[m]] = z[m]
    used = np.zeros(n_tw, bool)
    used[pairs.ravel()] = True
    Z[~used] = np.nan                       # only team-weeks a template game can pick
    R = np.full_like(Z, np.nan)
    n_slot = np.ones(K)
    for s in range(K):
        f = np.flatnonzero(np.isfinite(Z[:, s]))
        n_slot[s] = max(len(f), 1)
        R[f[np.argsort(Z[f, s], kind="stable")], s] = np.arange(len(f))
    return pairs, R, n_slot, ndtri((R + 0.5) / n_slot)


def _pcorr(A, B):
    """Pairwise-complete Pearson correlations between the columns of A and B: (corr, n)."""
    C, N = np.zeros((A.shape[1], B.shape[1])), np.zeros((A.shape[1], B.shape[1]))
    for a in range(A.shape[1]):
        for b in range(B.shape[1]):
            f = np.isfinite(A[:, a]) & np.isfinite(B[:, b])
            N[a, b] = f.sum()
            if N[a, b] >= 3 and A[f, a].std() > 0 and B[f, b].std() > 0:
                C[a, b] = np.corrcoef(A[f, a], B[f, b])[0, 1]
    return C, N


def slot_corr(Zs, pairs, a_slot, g):
    """20 x 20 game correlation matrix [[S, X], [X, S]] (S same team, X opponents; slot order SLOTS),
    empirical normal-score correlations shrunk entrywise towards the factor model by empirical Bayes."""
    S_e, nS = _pcorr(Zs, Zs)
    top, bot = np.vstack([Zs[pairs[:, 0]], Zs[pairs[:, 1]]]), np.vstack([Zs[pairs[:, 1]], Zs[pairs[:, 0]]])
    X_e, nX = _pcorr(top, bot)              # both orientations: symmetric, each game counted twice
    nX = nX / 2
    F = np.outer(a_slot, a_slot)
    S_f, X_f = F.copy(), g * F
    off = ~np.eye(K, dtype=bool)
    e = np.concatenate([(S_e - S_f)[off & (nS >= MIN_N)], (X_e - X_f)[nX >= MIN_N]])
    n = np.concatenate([nS[off & (nS >= MIN_N)], nX[nX >= MIN_N]])
    tau2 = max(float(np.mean(e ** 2) - np.mean(1 / n)), 1e-4) if len(e) else 1e-4
    wS = np.where(nS >= MIN_N, tau2 / (tau2 + 1 / np.maximum(nS, 1)), 0.0)
    wX = np.where(nX >= MIN_N, tau2 / (tau2 + 1 / np.maximum(nX, 1)), 0.0)
    S = wS * S_e + (1 - wS) * S_f
    X = wX * X_e + (1 - wX) * X_f
    np.fill_diagonal(S, 1.0)
    Rm = np.block([[S, X], [X, S]])
    lam, V = np.linalg.eigh(Rm)
    Rm = (V * np.maximum(lam, 1e-3)) @ V.T
    s = np.sqrt(np.diag(Rm))
    Rm = Rm / np.outer(s, s)
    info = {"tau": round(float(np.sqrt(tau2)), 3), "w_med": round(float(np.median(np.concatenate([wS[off], wX.ravel()]))), 2),
            "min_eig": round(float(lam.min()), 3)}
    return Rm, info


# ---------------------------------------------------------------- sampling
_A = np.linspace(-np.arcsinh(1e6), np.arcsinh(1e6), 16001)     # uniform grid in asinh(x)


def _tcdf(nu):
    """Fast t_nu CDF table: F at x = sinh(_A) (piecewise linear in asinh(x), float32; |error| < 1e-6)."""
    return stdtr(nu, np.sinh(_A)).astype(np.float32)


def _tcdf_eval(F, x):
    a = (np.arcsinh(x) - np.float32(_A[0])) * np.float32((len(_A) - 1) / (_A[-1] - _A[0]))
    a = np.clip(a, 0, len(_A) - 1.001)
    i = a.astype(np.int64)
    fr = a - i
    return F[i] * (1 - fr) + F[i + 1] * fr


class TailDraws:
    """Every QB/RB/WR/TE projected > 0 in a week frame, with S score draws under the variant's copula."""

    def __init__(self, wk, m, S, seed, keep_u=False):
        sel = (wk["proj"].to_numpy(float) > 0) & wk["pos"].isin(POS).to_numpy()
        w = wk[sel]
        self.pids = w["pid"].astype(str).tolist()
        self.ix = {p: i for i, p in enumerate(self.pids)}
        n = self.n = len(self.pids)
        pos, proj = w["pos"].to_numpy(str), w["proj"].to_numpy(float)
        qflat = (proj[:, None] * ratio_grid(m.tables, pos, proj)).astype(np.float32).ravel()
        load = np.array([m.loading.get(p, 0.0) for p in pos], np.float32)
        cload = np.sqrt(1 - load ** 2).astype(np.float32)
        teams, games = {}, {}
        tnames = [t if isinstance(t, str) and t else f"_{i}" for i, t in enumerate(w["team"])]
        gnames = [x if isinstance(x, str) and x else f"_{i}" for i, x in enumerate(w["game"])]
        team = np.array([teams.setdefault(t, len(teams)) for t in tnames], np.int64)
        game = np.array([games.setdefault(x, len(games)) for x in gnames], np.int64)
        rk = w.assign(_t=tnames).groupby(["_t", "pos"])["proj"].rank(ascending=False, method="first").to_numpy()
        slot = np.array([SLOT_IX.get(f"{p}{int(r)}", -1) for p, r in zip(pos, rk)], np.int64)
        first = {}
        side = np.array([0 if first.setdefault(gnames[i], tnames[i]) == tnames[i] else 1 for i in range(n)], np.int64)
        slot[~np.array([isinstance(x, str) and bool(x) for x in w["game"]], bool)] = -1
        self.slot, self.team, self.side, self.game = slot, team, side, game
        nc = max(1, S // CHUNK)
        self.scoreT = np.zeros((n + 1, nc * CHUNK), np.float32)    # player-major: fast team sums
        self.u = np.empty((nc * CHUNK, n), np.float32) if keep_u else None     # verify.py only
        args = (load, cload, team, game, len(teams), len(games), slot, side)
        for ci in range(nc):
            u = self._chunk(m, np.random.default_rng([*seed, ci]), CHUNK, *args)
            if keep_u:
                self.u[ci * CHUNK:(ci + 1) * CHUNK] = u
            v = u * np.float32(Q - 1)
            i0 = np.minimum(v.astype(np.int64), Q - 2)
            fr = v - i0
            base = np.arange(n) * Q + i0
            self.scoreT[:n, ci * CHUNK:(ci + 1) * CHUNK] = (qflat[base] * (1 - fr) + qflat[base + 1] * fr).T

    @staticmethod
    def _chunk(m, rng, C, load, cload, team, game, n_team, n_game, slot, side):
        """(C, n) float32 uniforms."""
        n, g = len(load), m.g
        G = rng.standard_normal((C, n_game), dtype=np.float32)        # baseline's draws, baseline's order
        T = rng.standard_normal((C, n_team), dtype=np.float32)
        z = load * (np.float32(np.sqrt(g)) * G[:, game] + np.float32(np.sqrt(1 - g)) * T[:, team])
        z += cload * rng.standard_normal((C, n), dtype=np.float32)
        ok = slot >= 0
        if m.variant == "slot_t":
            E = rng.standard_normal((C, n_game, 2 * K), dtype=np.float32) @ m.chol.T
            z[:, ok] = E[:, game[ok], side[ok] * K + slot[ok]]
        group, nu = m.tail[m.variant]
        if np.isinf(nu):
            u = ndtr(z)
        else:
            grp = game if group == "game" else np.zeros(n, np.int64)          # slate: one W per draw
            W = rng.chisquare(nu, (C, int(grp.max()) + 1)).astype(np.float32)
            u = _tcdf_eval(m.tcdf, z * np.sqrt(np.float32(nu) / np.maximum(W, np.float32(1e-12)))[:, grp])
        if m.variant != "shuffle":
            return u
        pick = rng.integers(0, len(m.pairs), (C, n_game))                    # a training game per (draw, game)
        flip = rng.integers(0, 2, (C, n_game))                               # and its orientation
        V = rng.random((C, n), dtype=np.float32)
        tw = m.pairs[pick[:, game[ok]], flip[:, game[ok]] ^ side[ok]]       # (C, n_ok) template team-week
        ut = ((m.rank[tw, slot[ok]] + V[:, ok]) / m.n_slot[slot[ok]]).astype(np.float32)
        use = np.isfinite(ut)
        sub = u[:, ok]
        sub[use] = ut[use]
        u[:, ok] = sub
        return np.clip(u, 0.0, 1.0)

    def totals(self, starters):
        """(S, N) float64 team totals ('' or unknown = 0); equal to the baseline's score[:, idx].sum(-1)."""
        idx = np.array([[self.ix.get(p, self.n) for p in row] for row in starters], np.int64)
        tot = np.empty((self.scoreT.shape[1], len(idx)))
        for k, row in enumerate(idx):
            tot[:, k] = self.scoreT[row].sum(0, dtype=np.float64)
        return tot


def summarize(tot):
    """_baseline_team.copula.summarize via one sort (np.quantile's linear rule, equal to ~1e-13)."""
    S, N = tot.shape
    p = np.bincount(tot.argmin(1), minlength=N) / S
    h = LEVELS * (S - 1)
    lo = np.floor(h).astype(np.int64)
    hi = np.minimum(lo + 1, S - 1)
    f = h - lo
    s = np.sort(tot, axis=0)
    m = np.sort(tot.min(1))
    return {"p_chop": p, "total_q": (s[lo] * (1 - f)[:, None] + s[hi] * f[:, None]).T,
            "line_q": m[lo] * (1 - f) + m[hi] * f}


class TailCopula:
    def __init__(self, variant="gt"):
        assert variant in ("gt", "slot_t", "shuffle"), variant
        self.variant = variant
        self.S = S_DRAWS
        self._week = None

    def fit(self, train, train_leagues=None):
        d = train[train["item"].to_numpy(bool)].reset_index(drop=True)   # = model.calibration_rows
        pos, proj, act = d["pos"].to_numpy(str), d["proj"].to_numpy(float), d["actual"].to_numpy(float)
        self.tables = fit_tables(pos, proj, act)
        t = d["t"].to_numpy()
        tk = t.astype(str) + "-" + d["team"].astype(str).to_numpy()
        gk = t.astype(str) + "-" + d["game"].astype(str).to_numpy()
        a, self.g = fit_copula(pos, proj, act, tk, gk)
        self.loading = {p: float(a[i]) for i, p in enumerate(POS)}
        z = pit_z(pos, proj, act)
        slot, tw = slot_index(d)
        self.pairs, self.rank, self.n_slot, Zs = build_templates(d, z, slot, tw)
        I, J, sg = week_pairs(t, proj, gk)
        a_of = np.array([self.loading[p] for p in pos])
        same = tk[I] == tk[J]
        rho_f = np.where(sg, a_of[I] * a_of[J] * np.where(same, 1.0, self.g), 0.0)
        self.tail = {}
        g_, nu_, cl_ = fit_tail(z, I, J, rho_f, sg)
        self.tail["gt"] = self.tail["shuffle"] = (g_, nu_)
        msg = f"gt {g_} nu {nu_:g} (CL gain {cl_[f'{g_}{nu_:g}']})"
        if self.variant == "slot_t":
            a_slot = np.array([self.loading[s[:2]] for s in SLOTS])
            Rm, info = slot_corr(Zs, self.pairs, a_slot, self.g)
            self.chol = np.linalg.cholesky(Rm).astype(np.float32)
            si, sj = slot[I], slot[J]
            both = sg & (si >= 0) & (sj >= 0)
            # side of j relative to i: same team -> block S, opponents -> block X
            rho_s = np.where(both, Rm[si, np.where(same, 0, K) + sj], np.where(sg & (si < 0) & (sj < 0), rho_f, 0.0))
            g2, nu2, cl2 = fit_tail(z, I, J, np.clip(rho_s, -0.99, 0.99), sg)
            self.tail["slot_t"] = (g2, nu2)
            msg += f"; slot_t {g2} nu {nu2:g} (CL gain {cl2[f'{g2}{nu2:g}']}), shrink {info}"
        nu = self.tail[self.variant][1]
        self.tcdf = _tcdf(nu) if np.isfinite(nu) else None
        print(f"[copula_tail] {self.variant} t<{int(t.max()) + 1}: {msg}; {len(self.pairs)} template games; "
              f"loadings {np.round(a, 3).tolist()} g {self.g:.3f}", flush=True)
        return self

    def draws(self, league):
        key = (league["season"], league["week"])
        if self._week is None or self._week[0] != key:
            self._week = (key, TailDraws(league["week_df"], self, self.S, (SEED, *key)))
        return self._week[1]

    def predict(self, league):
        return summarize(self.draws(league).totals(league["starters"]))
