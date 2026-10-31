"""Analytic / semi-analytic chop probabilities: order statistics of team totals, no Monte Carlo.

Same player model as the team baseline, refit per origin: _baseline_player's bucketed actual/proj
quantile tables (256 levels, read off linearly like sim.Universe.paths) and model.fit's one-factor
Gaussian copula z = a_pos (sqrt(g) G_game + sqrt(1-g) T_team) + sqrt(1-a^2) eps. Only the way team
totals and their minimum are computed changes.

Team totals.
  cf    first four cumulants: mean exact; variance with every same-team / same-game pair covariance
        of the copula by Mehler's Hermite expansion, Cov = sum_k rho^k c_k c'_k / k!; third and fourth
        cumulants additive over players; Cornish-Fisher quantile map (monotone-patched). Each player is
        winsorized at his 99% quantile first: a few table rows reach 10-20x projection, which would own
        the third/fourth moments. Johnson SU is infeasible: totals have skew ~0.3 with excess kurtosis
        ~0.06, below the lognormal line (Johnson's bounded SB region).
  conv  the exact lattice pmf (step H) under the copula. A team's players split into game groups,
        independent of each other; a group of several players is integrated over its game factor and
        NFL-team factors by Gauss-Hermite quadrature, its conditionally independent players convolve.
        All of it in the Fourier domain (a mixture over nodes is a mean of spectra).
Chop and line. P(i lowest) = int f_i prod_{j!=i} (1 - F_j) dx on the lattice (midpoint survival per
bin), line cdf = 1 - prod_j (1 - F_j): the exact finite-N order statistic (no asymptotic GEV: normal
minima reach the Gumbel limit only at rate 1/log N, useless at N = 10-17). conv adds cross-team
dependence: team-total covariances from shared NFL teams and games (Hermite), a one-factor Gaussian
copula on the totals fitted to their correlations, integrated by Gauss-Hermite (teams conditionally
independent given the factor).
cal = conv + T' = m + beta (mu - m) + kappa (T - mu) per league (m = league mean of mu), (beta, kappa)
by penalized maximum likelihood of earlier weeks' chops and team totals (rolling origin).
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import functools

import numpy as np
from numpy.fft import irfft, rfft
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize
from scipy.special import factorial, ndtr, ndtri

from bench.candidates._baseline_player.ratio import fit_tables, ratio_grid
from bench.candidates._baseline_team.copula import fit_copula
from bench.harness.metrics import LEVELS
from common import POS
from model import Q

H = 0.25                       # lattice step (points)
XLO = -10.0                    # first player-lattice point; 0 sits on the lattice
XHI = (90.0, 160.0)            # player lattice top: the week's largest quantile, clipped to this range
                               # (mass beyond folds into the top point; a 2026 QB bucket reaches 235)
CF_CAP = 0.99                  # cf: each player's score winsorized at this quantile level before moments
WLO, WHI = -20.0, 260.0        # team-total window (outside mass -> end points)
NWIN = int(round((WHI - WLO) / H))
EDGES = WLO - H / 2 + H * np.arange(NWIN + 1)     # team-window bin edges (bins centred on the lattice)
K_HERM = 10                    # Hermite terms (rho <= 0.36 here, rho^10 < 4e-5)
M_U = 8192                     # u-midpoints for Hermite coefficients
N_ONE, N_G2, N_SIDE = 24, 12, 20   # GH nodes: one-team group factor; game factor of a two-team group;
                                   # NFL-team factor of its multi-player side (a_QB = 0.95: sharp)
N_W = 9                        # GH nodes: cross-team common factor
VARIANTS = ("cf", "conv", "cal")
CAL = {"weeks": 12, "rep_step": 3, "n_eff_week": 8.0, "sd_beta": 0.15, "sd_kappa": 0.10, "stride": 4}


@functools.lru_cache(maxsize=None)
def gh(n):
    x, w = hermegauss(n)
    return x, w / w.sum()


def _ramp(n):
    return 1e-12 * np.arange(n)


def inv_cdf(F, levels=LEVELS, x=EDGES):
    """Quantiles at levels of a cdf F (..., len(x)) given at points x (piecewise linear)."""
    F = np.atleast_2d(F)
    return np.stack([np.interp(levels, f + _ramp(len(f)), x) for f in F])


class PlayerWeek:
    """Every QB/RB/WR/TE projected > 0 in a week frame (the baseline's WeekDraws pool): quantile
    rows, copula loadings, NFL team/game ids, exact moments, lattice cdfs and spectra, Hermite coefs."""

    def __init__(self, wk, tables, loading, g, slots, cap=None):
        m = (wk["proj"].to_numpy(float) > 0) & wk["pos"].isin(POS).to_numpy()
        w = wk[m]
        self.pids = w["pid"].astype(str).tolist()
        self.ix = {p: i for i, p in enumerate(self.pids)}
        n = self.n = len(self.pids)
        pos, proj = w["pos"].to_numpy(str), w["proj"].to_numpy(float)
        q = proj[:, None] * ratio_grid(tables, pos, proj)                  # (n, Q) score quantiles
        if cap is not None:                     # winsorize at level cap (u read off like the sim does)
            u = cap * (Q - 1)
            k = min(int(u), Q - 2)
            q = np.minimum(q, (q[:, k] * (1 - (u - k)) + q[:, k + 1] * (u - k))[:, None])
        self.q = q
        self.a = np.array([loading.get(p, 0.0) for p in pos], float)
        self.g = float(g)
        teams, games = {}, {}
        self.team = np.array([teams.setdefault(t if isinstance(t, str) and t else f"_{i}", len(teams))
                              for i, t in enumerate(w["team"])], np.int64)
        self.game = np.array([games.setdefault(x if isinstance(x, str) and x else f"_{i}", len(games))
                              for i, x in enumerate(w["game"])], np.int64)
        # exact raw moments of the piecewise-linear quantile function (a mixture of 255 uniforms)
        lo, hi = q[:, :-1], q[:, 1:]
        raw = [sum(lo ** j * hi ** (r - j) for j in range(r + 1)).mean(1) / (r + 1) for r in range(1, 5)]
        mu, m2, m3, m4 = raw
        self.mu = mu
        self.var = np.maximum(m2 - mu ** 2, 0.0)
        self.k3 = m3 - 3 * mu * m2 + 2 * mu ** 3
        self.k4 = (m4 - 4 * mu * m3 + 6 * mu ** 2 * m2 - 3 * mu ** 4) - 3 * self.var ** 2
        # cdf at the player-lattice edges: P(X < XLO + (k - 1/2) H), k <= NP; end mass folded into end points
        top = float(np.clip(np.ceil(q[:, -1].max() if n else 0.0) + 1.0, *XHI))
        NP = self.NP = int(round((top - XLO) / H)) + 1
        e = XLO + (np.arange(NP + 1) - 0.5) * H
        Fe = np.empty((n, NP + 1))
        for i in range(n):
            j = np.searchsorted(q[i], e, side="right")             # number of quantile nodes <= e
            s = np.clip(j - 1, 0, Q - 2)
            d = q[i, s + 1] - q[i, s]
            fr = np.divide(e - q[i, s], d, out=np.zeros_like(e), where=d > 0)
            Fe[i] = np.where(j == 0, 0.0, np.where(j >= Q, 1.0, (s + np.clip(fr, 0, 1)) / (Q - 1)))
        Fe[:, 0], Fe[:, -1] = 0.0, 1.0
        self.Ze = ndtri(np.clip(Fe, 0.0, 1.0))                     # latent z thresholds (+-inf at ends)
        self.S = len(slots)
        self.L = self.S * (NP - 1) + 1                             # team lattice length
        self.NF = 1 << int(np.ceil(np.log2(self.L)))
        self.spec0 = rfft(np.diff(Fe, axis=1), self.NF)            # unconditional spectra (n, NF/2+1)
        k0 = int(round(-XLO / H))
        self.zero = np.exp(-2j * np.pi * k0 * np.arange(self.NF // 2 + 1) / self.NF)   # point mass at 0
        self.k_lo = int(round((WLO - self.S * XLO) / H))
        # Hermite coefficients c_k = E[h(Z) He_k(Z)], h = q o Phi (u-midpoint rule)
        u = (np.arange(M_U) + 0.5) / M_U
        uu = u * (Q - 1)
        i0 = np.minimum(uu.astype(np.int64), Q - 2)
        fr = uu - i0
        z = ndtri(u)
        He = np.empty((M_U, K_HERM + 1))
        He[:, 0], He[:, 1] = 1.0, z
        for k in range(1, K_HERM):
            He[:, k + 1] = z * He[:, k] - k * He[:, k - 1]
        self.herm = np.empty((n, K_HERM + 1))
        for b in range(0, n, 256):
            qv = q[b:b + 256, i0] * (1 - fr) + q[b:b + 256, i0 + 1] * fr
            self.herm[b:b + 256] = qv @ He / M_U
        self.kfact = factorial(np.arange(K_HERM + 1))
        self._gcache = {}

    # ------------------------------------------------------------ dependence
    def rho(self, idx):
        """Latent copula correlations among players idx (same NFL team: a a'; same game: a a' g)."""
        a, t, gm = self.a[idx], self.team[idx], self.game[idx]
        aa = np.outer(a, a)
        return np.where(t[:, None] == t[None, :], aa, np.where(gm[:, None] == gm[None, :], aa * self.g, 0.0))

    def cov(self, idx):
        """Score covariance matrix among players idx (Mehler series; exact variances on the diagonal)."""
        R = self.rho(idx)
        c = self.herm[idx]
        C = np.zeros_like(R)
        Rk = np.ones_like(R)
        for k in range(1, K_HERM + 1):
            Rk = Rk * R
            C += Rk * np.outer(c[:, k], c[:, k]) / self.kfact[k]
        np.fill_diagonal(C, self.var[idx])
        return C

    # ------------------------------------------------------------ exact team pmf
    def _cspec(self, i, m, s, nf):
        """Spectra (size nf) of player i's score given latent z ~ N(m, s^2), one row per entry of m."""
        Fc = ndtr((self.Ze[i][None, :] - np.asarray(m, float)[:, None]) / s)
        return rfft(np.diff(Fc, axis=1), nf)

    def _group_spec(self, members):
        """Full-size spectrum of a game group's total (k players: lattice offset k XLO), integrated over
        its factors by Gauss-Hermite on a group-size FFT; cached per week (leagues share players)."""
        key = tuple(sorted(members))
        if key in self._gcache:
            return self._gcache[key]
        a, g = self.a, self.g
        n = (len(members) * (self.NP - 1)) + 1
        nf = 1 << int(np.ceil(np.log2(n)))
        sides = {}
        for i in members:
            sides.setdefault(self.team[i], []).append(i)
        if len(sides) == 1:                                       # one NFL team: its factor F ~ N(0, 1)
            x, w = gh(N_ONE)
            S = np.ones((len(x), nf // 2 + 1), complex)
            for i in members:
                S *= self._cspec(i, a[i] * x, np.sqrt(1 - a[i] ** 2), nf)
            spec = w @ S
        else:                                                     # both teams of a game: G outside
            xg, wg = gh(N_G2)
            out = np.ones((len(xg), nf // 2 + 1), complex)
            for side in sides.values():
                if len(side) == 1:
                    i = side[0]
                    out *= self._cspec(i, a[i] * np.sqrt(g) * xg, np.sqrt(1 - a[i] ** 2 * g), nf)
                    continue
                xt, wt = gh(N_SIDE)
                f = (np.sqrt(g) * xg[:, None] + np.sqrt(1 - g) * xt[None, :]).ravel()
                S = np.ones((len(f), nf // 2 + 1), complex)
                for i in side:
                    S *= self._cspec(i, a[i] * f, np.sqrt(1 - a[i] ** 2), nf)
                out *= np.einsum("gtf,t->gf", S.reshape(len(xg), len(xt), -1), wt)
            spec = wg @ out
        p = np.maximum(irfft(spec, nf)[:n], 0.0)
        self._gcache[key] = rfft(p / p.sum(), self.NF)
        return self._gcache[key]

    def team_pmf(self, row):
        """Exact pmf of a team total on the window lattice for player indices row (-1 = scores 0)."""
        spec = np.ones(self.NF // 2 + 1, complex)
        groups = {}
        for i in row:
            if i < 0:
                spec = spec * self.zero
            else:
                groups.setdefault(self.game[i], []).append(i)
        for mem in groups.values():
            spec = spec * (self.spec0[mem[0]] if len(mem) == 1 else self._group_spec(mem))
        p = np.maximum(irfft(spec, self.NF)[:self.L], 0.0)
        w = p[self.k_lo:self.k_lo + NWIN].copy()
        w[0] += p[:self.k_lo].sum()
        w[-1] += p[self.k_lo + NWIN:].sum()
        return w / w.sum()

    # ------------------------------------------------------------ a league
    def rows(self, starters):
        return [[self.ix.get(p, -1) if p else -1 for p in r] for r in starters]

    def league(self, starters, method):
        """mu (N,), team-total covariance (N, N), cdfs at EDGES (N, NWIN + 1), pmfs (N, NWIN) or None."""
        rows = self.rows(starters)
        N = len(rows)
        flat = np.array([i for r in rows for i in r if i >= 0], np.int64)
        lab = np.array([t for t, r in enumerate(rows) for i in r if i >= 0], np.int64)
        A = np.zeros((len(flat), N))
        A[np.arange(len(flat)), lab] = 1.0
        C = A.T @ self.cov(flat) @ A if len(flat) else np.zeros((N, N))
        mu = A.T @ self.mu[flat] if len(flat) else np.zeros(N)
        if method == "cf":
            k3 = A.T @ self.k3[flat] if len(flat) else np.zeros(N)
            k4 = A.T @ self.k4[flat] if len(flat) else np.zeros(N)
            F = np.stack([cf_cdf(mu[t], C[t, t], k3[t], k4[t]) for t in range(N)])
            return mu, C, F, None
        pmf = np.stack([self.team_pmf(r) for r in rows])
        F = np.concatenate([np.zeros((N, 1)), np.cumsum(pmf, 1)], 1)
        F[:, -1] = 1.0
        return mu, C, np.minimum(F, 1.0), pmf


def cf_cdf(k1, k2, k3, k4, x=EDGES):
    """Cornish-Fisher cdf at x from four cumulants; the map is kept increasing (slope >= 0.05 sd)."""
    sd = np.sqrt(max(k2, 1e-12))
    if sd < 1e-6:
        return (x >= k1).astype(float)
    g1, g2 = k3 / sd ** 3, k4 / sd ** 4
    z = np.linspace(-9.0, 9.0, 3601)
    w = z + (z ** 2 - 1) * g1 / 6 + (z ** 3 - 3 * z) * g2 / 24 - (2 * z ** 3 - 5 * z) * g1 ** 2 / 36
    dz = z[1] - z[0]
    dw = np.maximum(np.diff(w), 0.05 * dz)
    c = len(z) // 2
    w = np.concatenate([[0.0], np.cumsum(dw)])
    w -= w[c] - (-g1 / 6)                       # CF value at z = 0 is -g1/6
    return ndtr(np.interp(x, k1 + sd * w, z, left=-40.0, right=40.0))


def one_factor(C, iters=60, cap=0.7):
    """Loadings lam >= 0 with lam_i lam_j ~ corr_ij off the diagonal (least squares, Jacobi updates)."""
    sd = np.sqrt(np.maximum(np.diag(C), 1e-12))
    R = C / np.outer(sd, sd)
    N = len(R)
    off = ~np.eye(N, dtype=bool)
    if N < 3 or R[off].mean() <= 0:
        return np.zeros(N)
    lam = np.full(N, np.sqrt(R[off].mean()))
    Ro = np.where(off, R, 0.0)
    for _ in range(iters):
        den = (lam ** 2).sum() - lam ** 2
        lam = np.clip(np.divide(Ro @ lam, den, out=np.zeros(N), where=den > 0), 0.0, cap)
    return lam


def order_stats(F, lam=None, nw=N_W):
    """F (N, E) team cdfs at EDGES -> p_chop (N,), line cdf (E,); a common latent factor with
    loadings lam (Gaussian copula on the totals) is integrated by Gauss-Hermite."""
    N = len(F)
    if lam is None or not np.any(lam > 0):
        nodes, wts, lam = np.zeros(1), np.ones(1), np.zeros(N)
    else:
        nodes, wts = gh(nw)
    Z = ndtri(np.clip(F, 0.0, 1.0)) if len(nodes) > 1 else None
    s = np.sqrt(1 - lam ** 2)[:, None]
    p = np.zeros(N)
    Fmin = np.zeros(F.shape[1])
    for x, w in zip(nodes, wts):
        Fc = F if Z is None else ndtr((Z - lam[:, None] * x) / s)
        m = np.diff(Fc, axis=1)
        Sb = 1.0 - 0.5 * (Fc[:, :-1] + Fc[:, 1:])                     # survival at bin midpoints
        pre = np.vstack([np.ones((1, Sb.shape[1])), np.cumprod(Sb, 0)[:-1]])
        suf = np.vstack([np.cumprod(Sb[::-1], 0)[::-1][1:], np.ones((1, Sb.shape[1]))])
        p += w * (m * pre * suf).sum(1)
        Fmin += w * (1.0 - np.prod(1.0 - Fc, axis=0))
    return p / p.sum(), np.clip(Fmin, 0.0, 1.0)


def transform(F, mu, beta, kappa, stride=1):
    """cdfs of T' = m + beta (mu - m) + kappa (T - mu), m = mean(mu), at EDGES[::stride] (F piecewise
    linear between EDGES)."""
    m = mu.mean()
    x = mu[:, None] + (EDGES[None, ::stride] - (m + beta * (mu - m))[:, None]) / kappa
    pos = (x - EDGES[0]) / H
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, NWIN - 1)
    fr = np.clip(pos - i0, 0.0, 1.0)
    r = np.arange(len(F))[:, None]
    return F[r, i0] + fr * (F[r, i0 + 1] - F[r, i0])


def log_density(pmf, mu, y, beta, kappa):
    """log density of observed totals y under the transformed lattice pmfs (piecewise constant)."""
    m = mu.mean()
    x = mu + (y - (m + beta * (mu - m))) / kappa
    b = np.clip(np.floor((x - EDGES[0]) / H).astype(np.int64), 0, NWIN - 1)
    return np.log(np.maximum(pmf[np.arange(len(y)), b] / (H * kappa), 1e-12))


class OrderStats:
    """Team-track candidate: analytic chop probabilities (variants cf, conv, cal)."""

    def __init__(self, variant="conv"):
        assert variant in VARIANTS, f"unknown variant {variant}"
        self.variant = variant
        self.method = "cf" if variant == "cf" else "conv"
        self.beta, self.kappa = 1.0, 1.0
        self._week = None
        self.cal_info = {}

    def fit(self, train, train_leagues=None):
        d = train[train["item"].to_numpy(bool)]       # = model.calibration_rows: proj >= 0.5 and a game
        pos, proj, act = d["pos"].to_numpy(str), d["proj"].to_numpy(float), d["actual"].to_numpy(float)
        self.tables = fit_tables(pos, proj, act)
        t = d["t"].astype(str).to_numpy()
        a, self.g = fit_copula(pos, proj, act, t + "-" + d["team"].astype(str).to_numpy(),
                               t + "-" + d["game"].astype(str).to_numpy())
        self.loading = {p: float(a[i]) for i, p in enumerate(POS)}
        if self.variant == "cal" and train_leagues:
            self._calibrate(train, train_leagues)
        return self

    def player_week(self, wk, slots):
        return PlayerWeek(wk, self.tables, self.loading, self.g, slots, CF_CAP if self.method == "cf" else None)

    # ------------------------------------------------------------ rolling-origin recalibration
    def _calibrate(self, train, train_leagues):
        """(beta, kappa) maximizing sum_L w_L [log p_L(chopped) + mean_i log f_i(y_i)] + log prior over
        the last CAL weeks' leagues (virtual r % rep_step == 0 plus real); w_L = n_eff_week / n_week,
        because one week's replicates share every player outcome. Features use this origin's fit (in
        sample for those weeks, but only 5 copula numbers and ~50 bucket tables of ~500 rows each).
        p_chop here: independent teams on a stride-4 (1-point) lattice; log-normal priors on beta, kappa."""
        ts = sorted({L["t"] for L in train_leagues})[-CAL["weeks"]:]
        items, wts = [], []
        for t in ts:
            Ls = [L for L in train_leagues if L["t"] == t and
                  (L["kind"] == "real" or int(L["league_id"].rsplit("-", 1)[1]) % CAL["rep_step"] == 0)]
            if not Ls:
                continue
            W = self.player_week(train[train["t"].to_numpy() == t], Ls[0]["slots"])
            for L in Ls:
                mu, C, F, pmf = W.league(L["starters"], "conv")
                items.append((F, pmf, mu, np.asarray(L["points"], float), int(L["chopped"])))
                wts.append(CAL["n_eff_week"] / len(Ls))
        if not items:
            return

        def nll(th):
            b, k = np.exp(th)
            tot = 0.0
            for (F, pmf, mu, y, c), w in zip(items, wts):
                p, _ = order_stats(transform(F, mu, b, k, CAL["stride"]), None)
                tot += w * (np.log(np.clip(p[c], 1e-6, 1.0)) + log_density(pmf, mu, y, b, k).mean())
            return -tot + 0.5 * (th[0] / CAL["sd_beta"]) ** 2 + 0.5 * (th[1] / CAL["sd_kappa"]) ** 2

        r = minimize(nll, np.zeros(2), method="Nelder-Mead",
                     options={"xatol": 2e-3, "fatol": 1e-4, "maxiter": 200, "initial_simplex":
                              [[0.0, 0.0], [0.08, 0.0], [0.0, 0.05]]})
        th = np.clip(r.x, -0.5, 0.5)
        self.beta, self.kappa = float(np.exp(th[0])), float(np.exp(th[1]))
        self.cal_info = {"beta": round(self.beta, 4), "kappa": round(self.kappa, 4), "n_leagues": len(items),
                         "weeks": len(ts), "nll_gain": round(float(nll(np.zeros(2)) - r.fun), 4),
                         "evals": int(r.nfev)}
        print(f"[evt_order_stats] cal before t={max(ts) + 1}: {self.cal_info}", flush=True)

    # ------------------------------------------------------------ predict
    def week(self, league):
        key = (league["season"], league["week"])
        if self._week is None or self._week[0] != key:
            self._week = (key, self.player_week(league["week_df"], league["slots"]))
        return self._week[1]

    def predict(self, league):
        W = self.week(league)
        mu, C, F, _ = W.league(league["starters"], self.method)
        lam = None
        if self.method == "conv":
            lam = one_factor(C)
            if self.variant == "cal":
                F = transform(F, mu, self.beta, self.kappa)
        p, Fmin = order_stats(F, lam)
        return {"p_chop": p, "total_q": inv_cdf(F), "line_q": inv_cdf(Fmin)[0]}
