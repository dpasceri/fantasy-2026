"""Rival lineup behaviour: an inattention / random-utility mixture lineup model with partially pooled
manager attentiveness, plus a model for starters projected 0.

Lineup model. Each week manager m either
  - (prob lambda_m) leaves last week's lineup alone: every last-week starter still rostered stays in,
    even if now projected 0 (status quo / inattention); open slots are filled by projection; or
  - (prob 1 - lambda_m) sets the max-utility lineup over his rostered QB/RB/WR/TE with
        u_i = log(phat_i + 0.5) + sigma * (r * s_i + eps_i),   eps_i ~ N(0, 1) iid,
    s_i = 1 if i started last week (status-quo bonus), phat_i = proj_i, or gamma * his last projection
    for an eligible zero (see ZeroModel: recently projected >= 3, his team plays; game-time-decision
    optimism), else 0. sigma = 0, gamma = 0 is the max-projection lineup.
Slots QB, RB x2, WR x2, TE, then 2 FLEX from the rest (greedy is optimal for these nested slots).
r, sigma, gamma are shared; lambda_m ~ Beta(mu * kappa, (1 - mu) * kappa) with mu, kappa on grids
(kappa = 1e4 is complete pooling). Fit: tempered composite likelihood of the start indicators (each
roster player a Bernoulli with P(start) simulated by K_FIT draws on common random numbers, lapse PSI,
weight W_CL = 1/2 because one swap flips two indicators); the two components mix per team-week.
Everything sits on grids with fixed a-priori priors (skeptical: mass toward max projection), so the
posterior is exact on the grid. Prediction draws (r, sigma, gamma, lambda_m) from manager m's
posterior, then the component and eps, independently per score draw.

Zero-projection starters. Sleeper's final projection is 0 for players ruled out, but also for some
game-time decisions who then play (2026: Chase wk 4, Barkley wk 4, Adams wk 6 were started at proj 0
and scored). An eligible zero (last projection >= ZMIN_PROJ within ZWEEKS weeks, his last team plays
this week) plays with prob pi ~ Beta posterior: prior mean = the population rate among all such
player-weeks in training, prior strength Z_N0, updated by real-league starters at proj 0 before the
test week (managers' starts carry information). If he plays he scores a draw from the baseline
marginal at projection h * last projection (h = training mean of actual / last projection, clipped).
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import zlib

import numpy as np
from bench.candidates._baseline_player.ratio import ratio_grid
from common import POS
from model import Q
from scipy.stats import beta as _beta

POSI = {p: i for i, p in enumerate(POS)}
FIXED = ((0, 1), (1, 2), (2, 2), (3, 1))   # (position index, slots): QB, RB x2, WR x2, TE
FLEX_POS = (1, 2, 3)                       # FLEX = RB/WR/TE
N_FLEX = 2
N_SLOTS = sum(k for _, k in FIXED) + N_FLEX
P_OFF = 0.5                                # log(phat + P_OFF): zero-proj players stay finite
STALE = 100.0                              # utility bonus that keeps last week's starters (stale lineup)
PSI = 0.02                                 # lapse: P~ = PSI/2 + (1 - PSI) P
W_CL = 0.5                                 # composite-likelihood temper (a swap flips two indicators)
K_FIT = 400                                # CRN lineup draws per training team-week
# grids and priors, fixed a priori
R_GRID = np.array([0.0, 1.0, 2.0, 4.0, 8.0])           # status-quo bonus / sigma
LP_R = -R_GRID / 2.0
SIG = 0.02 * 2.0 ** (np.arange(11) / 2.0)              # 0.02 .. 0.64, uniform prior
GAM = np.array([0.0, 0.25, 0.5])                       # zero-proj optimism
LP_G = np.log([0.5, 0.25, 0.25])
LAM = np.array([0.005, 0.02, 0.04, 0.07, 0.1, 0.15, 0.22, 0.3, 0.4, 0.55, 0.75])   # inattention rate
MU = np.array([0.02, 0.04, 0.07, 0.1, 0.15, 0.22, 0.3])                              # subset of LAM
LP_MU = -MU / 0.1
KAPPA = np.array([2.0, 8.0, 32.0, 1e4])                # 1e4 = complete pooling (delta at mu)
SEED_FIT, SEED_PRED, SEED_Z = 23, 29, 31
ZMIN_PROJ, ZWEEKS, Z_N0 = 3.0, 3, 2.0
H_LO, H_HI, H_DEFAULT = 0.2, 1.0, 0.5


def _lam_prior():
    """(len(MU), len(KAPPA), len(LAM)) log pi(lambda | mu, kappa) on the LAM grid."""
    w = np.gradient(LAM)
    out = np.full((len(MU), len(KAPPA), len(LAM)), -np.inf)
    for i, mu in enumerate(MU):
        for j, k in enumerate(KAPPA):
            if k >= 1e4:
                out[i, j, int(np.argmin(np.abs(LAM - mu)))] = 0.0
                continue
            p = _beta.pdf(LAM, mu * k, (1 - mu) * k) * w
            out[i, j] = np.log(np.maximum(p, 1e-300)) - np.log(p.sum())
    return out


LP_LAM = _lam_prior()


# ---------------------------------------------------------------- lineups
def choose(U, pos, mask=False):
    """Max-utility lineups. U (M, n) utilities (-inf = not eligible), pos (n,) position index.
    Returns (M, N_SLOTS) column indices into the n players (-1 = empty slot), or with mask=True the
    (M, n) bool start mask. Works on each position's columns only (cheap for M ~ 1e5)."""
    M, n = U.shape
    out = np.full((M, N_SLOTS), -1, np.int64)
    if n == 0:
        return np.zeros((M, 0), bool) if mask else out
    c, fv, fi = 0, [], []
    for P, k in FIXED:                   # top k of each position; its next N_FLEX are flex candidates
        cols = np.flatnonzero(pos == P)
        if len(cols):
            X = U[:, cols]
            o = np.argsort(-X, axis=1, kind="stable")
            v = np.take_along_axis(X, o, 1)
            kk = min(k, len(cols))
            out[:, c:c + kk] = np.where(np.isfinite(v[:, :kk]), cols[o[:, :kk]], -1)
            if P in FLEX_POS and len(cols) > kk:
                fv.append(v[:, kk:kk + N_FLEX])
                fi.append(cols[o[:, kk:kk + N_FLEX]])
        c += k
    if fv:
        FV, FI = np.concatenate(fv, 1), np.concatenate(fi, 1)
        o = np.argsort(-FV, axis=1, kind="stable")[:, :N_FLEX]
        v = np.take_along_axis(FV, o, 1)
        out[:, c:c + v.shape[1]] = np.where(np.isfinite(v), np.take_along_axis(FI, o, 1), -1)
    if not mask:
        return out
    m = np.zeros(M * (n + 1), bool)
    m[(np.arange(M)[:, None] * (n + 1) + np.where(out >= 0, out, n)).ravel()] = True
    return m.reshape(M, n + 1)[:, :n]


def max_lineup(pos, proj):
    """Max-projection lineup (common.best_lineup: only players projected > 0 are eligible)."""
    U = np.where(proj > 0, proj, -np.inf)[None, :]
    return choose(U, pos)[0]




def team_feats(roster, proj_of, pos_of, prev, last_of=None):
    """(pids, pos idx, proj, last, s) for a roster's QB/RB/WR/TE. last = last projection of an eligible
    zero (last_of(pid) -> float or None), else 0; s = started last week."""
    pids = [p for p in roster if pos_of.get(p) in POSI]
    pos = np.array([POSI[pos_of[p]] for p in pids], np.int64)
    pr = np.array([max(float(proj_of.get(p, 0.0) or 0.0), 0.0) for p in pids])
    pl = np.array([(last_of(p) or 0.0) if (last_of is not None and pr[j] <= 0) else 0.0
                   for j, p in enumerate(pids)])
    s = np.array([p in prev for p in pids], float)
    return pids, pos, pr, pl, s


def _seed(*xs):
    return [int(x) if isinstance(x, (int, np.integer)) else zlib.crc32(str(x).encode()) for x in xs]


def base_util(pr, pl):
    """(len(GAM), n) log(phat + P_OFF) for every gamma."""
    ph = np.where(pr[None, :] > 0, pr[None, :], GAM[:, None] * pl[None, :])
    return np.log(ph + P_OFF)


def stale_mask(pos, pr, s):
    """(n,) bool: last week's starters kept, open slots by projection."""
    return choose((np.log(pr + P_OFF) + STALE * s)[None, :], pos, mask=True)[0]


def _cl(P, y):
    P = PSI / 2 + (1 - PSI) * P
    return W_CL * (y * np.log(P) + (1 - y) * np.log1p(-P)).sum(-1)


# ---------------------------------------------------------------- fit
def team_week_loglik(pos, pr, pl, s, y, seed):
    """Tempered composite log-lik of start indicators y (n,): ((R, SIG, GAM) for the utility component,
    scalar for the stale component)."""
    n = len(pr)
    eps = np.random.default_rng(seed).standard_normal((K_FIT, n)).astype(np.float32)
    b = base_util(pr, pl).astype(np.float32)                                    # (G, n)
    A, J, G = len(R_GRID), len(SIG), len(GAM)
    U = (b[None, None, :, None, :] + SIG.astype(np.float32)[None, :, None, None, None]
         * (R_GRID.astype(np.float32)[:, None, None, None, None] * s.astype(np.float32)
            + eps[None, None, None, :, :]))
    P = choose(U.reshape(-1, n), pos, mask=True).reshape(A, J, G, K_FIT, n).mean(3)
    return _cl(P, y), float(_cl(stale_mask(pos, pr, s).astype(float), y))


class LineupFit:
    """Grid posterior of the lineup model from training team-weeks.

    obs: list of dicts {mgr, week, pos, pr, pl, s, y}; kappa_idx restricts kappa (pooled = [3]);
    ll reuses another fit's per-team-week log-liks on the same obs."""

    def __init__(self, obs, kappa_idx=None, ll=None):
        self.n_obs = len(obs)
        self.kappa_idx = np.arange(len(KAPPA)) if kappa_idx is None else np.asarray(kappa_idx)
        self.ll = ll if ll is not None else [team_week_loglik(o["pos"], o["pr"], o["pl"], o["s"], o["y"],
                                                             _seed(SEED_FIT, o["week"], o["mgr"])) for o in obs]
        mgrs = sorted({o["mgr"] for o in obs})
        self.mgrs = {m: i for i, m in enumerate(mgrs)}
        lam = np.log(LAM)[None, None, None, :]
        lam1 = np.log1p(-LAM)[None, None, None, :]
        L = np.zeros((len(mgrs), len(R_GRID), len(SIG), len(GAM), len(LAM)))
        for o, (lr, ls) in zip(obs, self.ll):                                  # mixture per team-week
            L[self.mgrs[o["mgr"]]] += np.logaddexp(lam + ls, lam1 + lr[..., None])
        self.L = L
        lpl = LP_LAM[:, self.kappa_idx]                                          # (MU, Kp, LAM)
        self.lpl = lpl
        # Mm[m, a, j, g, mu, kp] = logsumexp_lam lpl + L
        self.Mm = np.logaddexp.reduce(L[:, :, :, :, None, None, :] + lpl[None, None, None, None], axis=-1)
        lp = (LP_R[:, None, None, None, None] + LP_G[None, None, :, None, None] + LP_MU[None, None, None, :, None]
              + np.zeros((1, len(SIG), 1, 1, len(self.kappa_idx))))
        self.logjoint = lp + self.Mm.sum(0)

    def _post(self):
        lj = self.logjoint
        return np.exp(lj - np.logaddexp.reduce(lj, axis=None))

    def summary(self):
        w = self._post()
        lam = {str(m): round(float((np.exp(self.cell_logp(m)).sum((0, 1, 2)) * LAM).sum()), 4) for m in self.mgrs}
        return {"n_team_weeks": self.n_obs, "n_managers": len(self.mgrs),
                "r_mean": round(float((w.sum((1, 2, 3, 4)) * R_GRID).sum()), 4),
                "sigma_mean": round(float((w.sum((0, 2, 3, 4)) * SIG).sum()), 4),
                "gamma_mean": round(float((w.sum((0, 1, 3, 4)) * GAM).sum()), 4),
                "mu_mean": round(float((w.sum((0, 1, 2, 4)) * MU).sum()), 4),
                "p_pooled": round(float(w.sum((0, 1, 2, 3))[list(self.kappa_idx).index(len(KAPPA) - 1)]), 4)
                if len(KAPPA) - 1 in self.kappa_idx else 0.0,
                "lambda_by_roster": lam}

    def cell_logp(self, mgr):
        """(R, SIG, GAM, LAM) log posterior predictive weights for manager mgr."""
        mi = self.mgrs.get(mgr)
        base = self.logjoint - (self.Mm[mi] if mi is not None else 0.0)           # (A, J, G, MU, Kp)
        Lm = self.L[mi] if mi is not None else np.zeros(self.L.shape[1:])        # (A, J, G, LAM)
        cell = base[..., None] + self.lpl[None, None, None] + Lm[:, :, :, None, None, :]
        out = np.logaddexp.reduce(cell.reshape(*cell.shape[:3], -1, len(LAM)), axis=3)
        return out - np.logaddexp.reduce(out, axis=None)


def sample_lineups(fit, mgr, pos, pr, pl, s, S, seed, mask=False):
    """(S, N_SLOTS) lineups (or (S, n) start masks) from manager mgr's posterior predictive
    (fit None or empty = max projection)."""
    if fit is None or fit.n_obs == 0:
        lu = max_lineup(pos, pr)
        if not mask:
            return np.repeat(lu[None, :], S, 0)
        m = np.zeros(len(pr), bool)
        m[lu[lu >= 0]] = True
        return np.repeat(m[None, :], S, 0)
    rng = np.random.default_rng(seed)
    lw = fit.cell_logp(mgr).ravel()
    c = rng.choice(len(lw), size=S, p=np.exp(lw - np.logaddexp.reduce(lw)))
    a, j, g, l = np.unravel_index(c, (len(R_GRID), len(SIG), len(GAM), len(LAM)))
    stale = rng.random(S) < LAM[l]
    eps = rng.standard_normal((S, len(pr)))
    U = base_util(pr, pl)[g] + SIG[j][:, None] * (R_GRID[a][:, None] * s[None, :] + eps)
    U[stale] = np.log(pr + P_OFF)[None, :] + STALE * s[None, :]
    return choose(U, pos, mask=mask)


# ---------------------------------------------------------------- zero-projection starters
class ZeroModel:
    """P(play) and score scale for starters projected 0 (see module docstring)."""

    def __init__(self, train):
        d = train[train["pos"].isin(POS).to_numpy()][["pid", "t", "pos", "proj", "team", "game", "actual"]]
        d = d.sort_values(["pid", "t"], kind="stable").reset_index(drop=True)
        pos_ = d["proj"].to_numpy(float) > 0
        d["lp"] = np.where(pos_, d["proj"], np.nan)
        d["lt"] = np.where(pos_, d["t"], np.nan)
        d["lteam"] = d["team"].where(pos_)
        g = d.groupby("pid", sort=False)
        for c in ("lp", "lt", "lteam"):                       # last positive projection strictly before t
            d["prev_" + c] = g[c].shift(1)
            d["prev_" + c] = d.groupby("pid", sort=False)["prev_" + c].ffill()
        games = d[d["game"].notna()].groupby("t")["team"].agg(lambda x: set(x.dropna()))
        self.games = {int(t): v for t, v in games.items()}
        el = self._eligible(d["proj"].to_numpy(float), d["t"].to_numpy(), d["prev_lp"].to_numpy(float),
                            d["prev_lt"].to_numpy(float), d["prev_lteam"].to_numpy(object))
        e = d[el]
        played = e["actual"].to_numpy(float) != 0
        self.n_pop, self.y_pop = int(len(e)), int(played.sum())
        self.rate0 = (self.y_pop + 0.5) / (self.n_pop + 1.0)
        ratio = e["actual"].to_numpy(float)[played] / e["prev_lp"].to_numpy(float)[played]
        self.h = float(np.clip(ratio.mean(), H_LO, H_HI)) if len(ratio) >= 5 else H_DEFAULT
        self.n_h = int(len(ratio))
        self.row = {(p, int(t)): (lp, lt, lteam) for p, t, lp, lt, lteam in
                    zip(d["pid"], d["t"], d["prev_lp"], d["prev_lt"], d["prev_lteam"])}
        last = d[pos_].groupby("pid", sort=False).tail(1)
        self.last = {p: (lp, lt, tm) for p, lp, lt, tm in zip(last["pid"], last["proj"], last["t"], last["team"])}
        self.n_start, self.y_start = 0, 0
        self.pi = self.rate0

    def _eligible(self, proj, t, lp, lt, lteam):
        ok = (proj <= 0) & np.isfinite(lp) & (lp >= ZMIN_PROJ) & np.isfinite(lt) & (t - np.nan_to_num(lt) <= ZWEEKS)
        return ok & np.array([o and (tm in self.games.get(int(tt), ())) for o, tm, tt in zip(ok, lteam, t)], bool)

    def train_last(self, pid, t):
        """Last projection of pid if he would be an eligible zero in training week t, else None."""
        r = self.row.get((pid, int(t)))
        if r is None:
            return None
        lp, lt, lteam = r
        ok = self._eligible(np.array([0.0]), np.array([t]), np.array([lp], float), np.array([lt], float),
                            np.array([lteam], object))[0]
        return float(lp) if ok else None

    def update_starters(self, events):
        """events: [(pid, t, played)] for real-league starters projected 0 in training weeks."""
        for p, t, played in events:
            if self.train_last(p, t) is not None:
                self.n_start += 1
                self.y_start += int(played)
        a0, b0 = Z_N0 * self.rate0, Z_N0 * (1 - self.rate0)
        self.pi = (a0 + self.y_start) / (a0 + b0 + self.n_start)
        return self

    def test_eligible(self, pid, t0, games_t0):
        """(last proj, pos-agnostic) if pid, projected 0 in test week t0, is an eligible zero; else None."""
        x = self.last.get(pid)
        if x is None:
            return None
        lp, lt, tm = x
        if lp >= ZMIN_PROJ and t0 - lt <= ZWEEKS and tm in games_t0:
            return float(lp)
        return None

    def draws(self, tables, pos, last_proj, S, seed):
        """(S,) float32 scores: Bernoulli(pi) x marginal draw at projection h * last_proj."""
        rng = np.random.default_rng(seed)
        pe = self.h * last_proj
        q = (pe * ratio_grid(tables, np.array([pos]), np.array([pe]))[0]).astype(np.float32)
        u = rng.random(S, dtype=np.float32) * np.float32(Q - 1)
        i0 = np.minimum(u.astype(np.int64), Q - 2)
        fr = u - i0
        x = q[i0] * (1 - fr) + q[i0 + 1] * fr
        return np.where(rng.random(S) < self.pi, x, 0.0).astype(np.float32)

    def summary(self):
        return {"pop_n": self.n_pop, "pop_played": self.y_pop, "rate0": round(self.rate0, 4),
                "starts_n": self.n_start, "starts_played": self.y_start, "pi": round(float(self.pi), 4),
                "h": round(self.h, 3), "h_n": self.n_h}
