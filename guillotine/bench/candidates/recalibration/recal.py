"""Team track: post-hoc recalibration of the baseline's chop probabilities and team-total / chop-line quantiles.

Forecast: the baseline itself (_baseline_team.copula.CopulaSim: model.py marginal + team/game copula refit
on the training rows, 96000 CRN draws of the test week); with identity maps the output is the baseline's
(verify.py). The maps are fit on a calibration set of earlier virtual weeks s < origin built by
leave-one-week-out cross-fitting: for each s the baseline is refit on every training week except s (so on
data strictly before the origin, and out of sample for s) and simulated on week s's pre-game frame with
S_CAL draws; each of week s's virtual leagues gives p_chop, the PIT of every team's actual total and the
PIT of the actual chop line. Real leagues never enter the calibration set; the real test weeks get the
same maps.

Chop maps on p~ = max(p, FLOOR), penalized MLE of the chop log-loss (prior N(identity, PRIOR_SD^2), the
log-likelihood divided by its week-cluster design effect):
  temp    p' ~ p~^beta                                   temperature scaling (Guo et al. 2017)
  mix     p' = (1-eps) softmax(beta log p~) + eps/N      + a uniform floor against tail surprises
  clogit  p' = (1-eps) softmax(X theta) + eps/N,  X = [log p~, log p~ (N-13.5)/3.5, zproj, dlogsd]:
          conditional logit = Dirichlet calibration with exchangeable classes plus team covariates
Quantile maps: level tau -> l(tau), then the simulated quantile at l(tau) (read from the draws):
  probit  l = Phi(a + b Phi^-1(tau)), (a, b) the normal-QQ line of the calibration PITs' probits
  dcp     l = the empirical tau-quantile of the calibration PITs (Kuleshov et al. 2018; distributional
          conformal prediction, Chernozhukov et al. 2021): the pinball-optimal level for each tau
Both shrink toward the identity by n_eff / (n_eff + N0), n_eff = Kish effective size of the PITs under
their within-week correlation (leagues of a week share NFL outcomes, so the chop line is ~1 draw/week).
Environment (set by the run scripts): RECAL_K = most recent calibration weeks (0 = all), RECAL_R =
virtual replicates per calibration week (48 = all), RECAL_MODE = smoke|full (tags the parameter dump),
RECAL_CACHE = a per-run directory (cleared by the run scripts) where the variants of one origin share
its calibration set (it never depends on the variant; see cached_calibration_set).
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps)

import fcntl
import hashlib
import json
import os

import numpy as np
from bench.candidates._baseline_team.copula import S_DRAWS, SEED, CopulaSim, WeekDraws
from bench.harness import RESULTS
from bench.harness import data as D
from bench.harness.metrics import LEVELS, item_uniform
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

S_CAL = 48000                    # draws per calibration week (6 sim chunks)
FLOOR = 0.5 / S_CAL              # p~ floor, the same on calibration and test leagues
PRIOR_SD = 0.25                  # chop-map prior sd around the identity
N0 = 20                          # effective pseudo-PITs shrinking a quantile map toward the identity
EPS_MAX = 0.25
NMID, NSC = 13.5, 3.5            # league-size centering for clogit's log p~ x N term
VARIANTS = {"temp": ("temp", None), "tmix_qq": ("mix", "probit"), "clogit_dcp": ("clogit", "dcp")}
CHOP = {"temp": (1, False), "mix": (1, True), "clogit": (4, True)}   # (n theta, eps)


class Draws:
    """WeekDraws' scores transposed to (n + 1, S), the last row an empty slot, so a team total sums
    contiguous rows. Sums of 8 float32 scores are exact in float64, so totals() equals
    WeekDraws.totals(...).T bit for bit (verify.py)."""

    def __init__(self, wk, tables, loading, g, S, seed):
        w = WeekDraws(wk, tables, loading, g, S, seed)
        self.ix, self.n, self.S = w.ix, w.n, w.score.shape[0]
        self.scoreT = np.ascontiguousarray(w.score.T)

    def totals(self, starters):
        """(N, S) float64 team totals for an (N, slots) pid array ('' or unknown = 0)."""
        idx = np.array([[self.ix.get(p, self.n) for p in row] for row in starters], np.int64)
        tot = self.scoreT[idx[:, 0]].astype(np.float64)
        for j in range(1, idx.shape[1]):
            tot += self.scoreT[idx[:, j]]
        return tot


def features(tot, proj):
    """Per-team chop covariates: log p~ (raw p kept), zproj (projected total z-score in the league) and
    dlogsd (log simulated sd minus the league mean)."""
    N, S = tot.shape
    p = np.bincount(tot.argmin(0), minlength=N) / S
    P = np.asarray(proj, float).sum(1)
    zp = (P - P.mean()) / max(P.std(), 1e-9)
    lsd = np.log(np.maximum(tot.std(1), 1e-9))
    return p, np.log(np.maximum(p, FLOOR)), zp, lsd - lsd.mean()


def pit(x, y, u):
    """Randomized PIT of y under the draws x (..., S): (#{x < y} + u #{x == y}) / S."""
    lo = (x < y[..., None]).sum(-1)
    eq = (x == y[..., None]).sum(-1)
    return (lo + u * eq) / x.shape[-1]


def design(kind, lp, n_team, zp, dsd):
    if kind == "clogit":
        return np.column_stack([lp, lp * (n_team - NMID) / NSC, zp, dsd])
    return lp[:, None]


class ChopMap:
    """p' = (1 - eps) softmax(X theta) + eps / N per league; theta0 = identity (theta = e1)."""

    def __init__(self, kind):
        self.kind = kind
        self.k, self.use_eps = CHOP[kind]
        self.theta0 = np.eye(self.k)[0]
        self.theta = self.theta0.copy()
        self.eps = 0.0
        self.info = {}

    def _ll(self, x, X, off, grp, cidx, inv_n):
        """(n_leagues,) log-likelihood of each league's chopped team."""
        th, eps = x[:self.k], (x[self.k] if self.use_eps else 0.0)
        s = X @ th
        m = np.maximum.reduceat(s, off)
        lz = m + np.log(np.add.reduceat(np.exp(s - m[grp]), off))
        pc = (1 - eps) * np.exp(s[cidx] - lz) + eps * inv_n
        return np.log(np.maximum(pc, 1e-300))

    def _obj(self, x, scale, *args):
        return -self._ll(x, *args).sum() / scale + 0.5 * (((x[:self.k] - self.theta0) / PRIOR_SD) ** 2).sum()

    def _deff(self, x, args, week):
        """Design effect of the theta score under within-week clustering, tr(J_week) / tr(J_league) >= 1
        (per-league scores by central differences at the fit)."""
        h = 1e-5
        G = np.column_stack([(self._ll(x + h * e, *args) - self._ll(x - h * e, *args)) / (2 * h)
                             for e in np.eye(len(x))[:self.k]])
        _, gi = np.unique(week, return_inverse=True)
        Gw = np.column_stack([np.bincount(gi, G[:, j]) for j in range(self.k)])
        return max(1.0, float((Gw ** 2).sum() / max((G ** 2).sum(), 1e-12)))

    def fit(self, cal):
        """Penalized MLE; the log-likelihood is divided by its week-cluster design effect (a magnitude-
        adjusted composite likelihood), so correlated leagues of one week don't swamp the prior."""
        X = design(self.kind, cal["lp"], cal["n_team"], cal["zp"], cal["dsd"])
        args = (X, cal["off"], cal["grp"], cal["cidx"], 1.0 / cal["N"])
        x0 = np.r_[self.theta0, [0.0] if self.use_eps else []]
        bnd = [(0.05, 4.0)] + [(-3.0, 3.0)] * (self.k - 1) + ([(0.0, EPS_MAX)] if self.use_eps else [])
        r1 = minimize(self._obj, np.r_[self.theta0, [0.01] if self.use_eps else []], args=(1.0, *args),
                      method="L-BFGS-B", bounds=bnd)
        deff = self._deff(r1.x, args, cal["t_league"])
        r = minimize(self._obj, r1.x, args=(deff, *args), method="L-BFGS-B", bounds=bnd)
        self.theta, self.eps = r.x[:self.k].copy(), (float(r.x[self.k]) if self.use_eps else 0.0)
        n = len(cal["N"])
        self.info = {"theta": np.round(self.theta, 4).tolist(), "eps": round(self.eps, 5), "deff": round(deff, 3),
                     "theta_unadjusted": np.round(r1.x[:self.k], 4).tolist(),
                     "cal_logloss_identity": round(float(-self._ll(x0, *args).mean()), 5),
                     "cal_logloss_fit": round(float(-self._ll(r.x, *args).mean()), 5), "converged": bool(r.success)}
        return self

    def apply(self, lp, N, zp, dsd):
        s = design(self.kind, lp, np.full(N, float(N)), zp, dsd) @ self.theta
        q = np.exp(s - s.max())
        return (1 - self.eps) * q / q.sum() + self.eps / N


class LevelMap:
    """Quantile-level recalibration from calibration PITs u: tau -> l(tau) (identity when kind is None)."""

    def __init__(self, kind):
        self.kind = kind
        self.lev = LEVELS.copy()
        self.info = {}

    def fit(self, u, week):
        """u: calibration PITs, week: their week (cluster). Shrink toward the identity with weight
        n_eff / (n_eff + N0), n_eff the Kish effective size under the within-week correlation."""
        u = np.asarray(u, float)
        n = len(u)
        if self.kind is None or n == 0:
            return self
        z = ndtri(np.clip(u, 0.5 / S_CAL, 1 - 0.5 / S_CAL))
        ne, icc = n_eff(z, week)
        w = ne / (ne + N0)
        if self.kind == "probit":
            zq, x = np.quantile(z, LEVELS), ndtri(LEVELS)
            b = float(np.cov(x, zq, bias=True)[0, 1] / x.var())
            a = float(zq.mean() - b * x.mean())
            a, b = w * a, 1.0 + w * (max(b, 0.2) - 1.0)
            self.lev = ndtr(a + b * x)
            self.info = {"a": round(a, 4), "b": round(b, 4)}
        elif self.kind == "dcp":
            self.lev = w * np.quantile(u, LEVELS) + (1 - w) * LEVELS
        else:
            raise ValueError(self.kind)
        self.lev = np.maximum.accumulate(np.clip(self.lev, 0.0, 1.0))
        self.info.update(n=n, icc=round(icc, 4), n_eff=round(ne, 1), w=round(w, 3),
                         l10=round(float(np.interp(0.1, LEVELS, self.lev)), 4),
                         l50=round(float(np.interp(0.5, LEVELS, self.lev)), 4),
                         l90=round(float(np.interp(0.9, LEVELS, self.lev)), 4),
                         pit_cov10=round(float((u <= 0.1).mean()), 4), pit_cov50=round(float((u <= 0.5).mean()), 4),
                         pit_cov90=round(float((u <= 0.9).mean()), 4))
        return self


def n_eff(z, g):
    """(Kish effective sample size, ICC) of z clustered by g: one-way ANOVA ICC (>= 0), design effect
    1 + (mean cluster size - 1) ICC. A single cluster counts as one observation."""
    z = np.asarray(z, float)
    _, gi, m = np.unique(np.asarray(g), return_inverse=True, return_counts=True)
    n, k = len(z), len(m)
    if k < 2:
        return 1.0, 1.0
    zb = np.bincount(gi, z) / m
    msb = (m * (zb - z.mean()) ** 2).sum() / (k - 1)
    msw = ((z - zb[gi]) ** 2).sum() / max(n - k, 1)
    m0 = (n - (m ** 2).sum() / n) / (k - 1)
    sb = max(0.0, (msb - msw) / m0)
    icc = sb / max(sb + msw, 1e-12)
    return float(n / (1.0 + (n / k - 1.0) * icc)), float(icc)


def calibration_set(train, train_leagues, K=0, R=48, S=S_CAL):
    """Leave-one-week-out baseline forecasts of the virtual training leagues (replicates r < R of the
    last K weeks; K = 0: every week): flat per-team and per-league arrays for the maps."""
    virt = [L for L in train_leagues if L["kind"] == "virtual" and int(L["league_id"].rsplit("-", 1)[1]) < R]
    weeks = sorted({L["t"] for L in virt})
    weeks = weeks[-K:] if K else weeks
    tt = train["t"].to_numpy()
    rec = {k: [] for k in ("lp", "zp", "dsd", "n_team", "u_tot", "N", "chop", "u_line", "t", "t_team")}
    for s in weeks:
        Ls = [L for L in virt if L["t"] == s]
        m = CopulaSim().fit(train[tt != s])
        season, week = Ls[0]["season"], Ls[0]["week"]
        wk = D.week_frame(season, week, items_only=False, pre=True).reset_index(drop=True)
        dr = Draws(wk, m.tables, m.loading, m.g, S, (SEED, season, week))
        for L in Ls:
            tot = dr.totals(L["starters"])
            N = L["n"]
            _, lp, zp, dsd = features(tot, L["starter_proj"])
            u = item_uniform([f"recal-{L['league_id']}-{k}" for k in range(N + 1)])
            rec["lp"].append(lp)
            rec["zp"].append(zp)
            rec["dsd"].append(dsd)
            rec["n_team"].append(np.full(N, float(N)))
            rec["u_tot"].append(pit(tot, np.asarray(L["points"], float), u[:N]))
            rec["N"].append(N)
            rec["chop"].append(int(L["chopped"]))
            rec["u_line"].append(float(pit(tot.min(0), np.array(L["line"], float), u[N])))
            rec["t"].append(s)
            rec["t_team"].append(np.full(N, s))
        del dr
    N = np.array(rec["N"], np.int64)
    off = np.r_[0, np.cumsum(N)[:-1]].astype(np.int64)
    cal = {k: np.concatenate(rec[k]) for k in ("lp", "zp", "dsd", "n_team", "u_tot", "t_team")}
    cal.update(N=N.astype(float), off=off, grp=np.repeat(np.arange(len(N)), N), cidx=off + np.array(rec["chop"]),
               u_line=np.array(rec["u_line"]), t_league=np.array(rec["t"]), weeks=[int(w) for w in weeks])
    return cal


def _code_hash():
    h = hashlib.sha1()
    for f in (__file__, os.path.join(os.path.dirname(os.path.dirname(__file__)), "_baseline_team", "copula.py")):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def cached_calibration_set(train, train_leagues, K=0, R=48, S=S_CAL):
    """calibration_set shared by the variant tasks of one origin through $RECAL_CACHE (unset: no cache).

    The set is a deterministic function of (train, train_leagues, K, R, S) only, never of the variant, so
    the first task of an origin builds it under a file lock and the others load it (bit-identical; the
    3 variants then cost one cross-fit instead of three). The key hashes this module's and the baseline's
    code, K/R/S, the training rows' t/pid/actual and the training league ids, so another origin (or
    data / code) can never hit it; the run scripts clear the directory before and after each run."""
    d = os.environ.get("RECAL_CACHE")
    if not d:
        return calibration_set(train, train_leagues, K, R, S)
    h = hashlib.sha1(f"{_code_hash()}|{K}|{R}|{S}|{len(train)}".encode())
    h.update(np.ascontiguousarray(train["t"].to_numpy(np.int64)).tobytes())
    h.update("|".join(train["pid"].astype(str)).encode())
    h.update(np.ascontiguousarray(train["actual"].to_numpy(float)).tobytes())
    h.update("|".join(sorted(L["league_id"] for L in train_leagues)).encode())
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"cal_t{int(train['t'].max()) + 1:02d}_{h.hexdigest()[:20]}.npz")
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)              # released on close or process exit
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                cal = {k: z[k] for k in z.files}
            cal["weeks"] = [int(w) for w in cal["weeks"]]
            return cal
        cal = calibration_set(train, train_leagues, K, R, S)
        tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, **{k: np.asarray(v) for k, v in cal.items()})
        os.replace(tmp, path)
        return cal


class Recal:
    """Baseline forecast + recalibration maps fit on leave-one-week-out forecasts of earlier virtual weeks."""

    def __init__(self, variant="temp"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        self.variant = variant
        chop, qmap = VARIANTS[variant]
        self.chop = ChopMap(chop)
        self.tot = LevelMap(qmap)
        self.line = LevelMap(qmap)
        self.K = int(os.environ.get("RECAL_K", "0"))
        self.R = int(os.environ.get("RECAL_R", "48"))
        self._week = None

    def fit(self, train, train_leagues):
        self.base = CopulaSim().fit(train, train_leagues)
        cal = cached_calibration_set(train, train_leagues, self.K, self.R)
        self.chop.fit(cal)
        self.tot.fit(cal["u_tot"], cal["t_team"])
        self.line.fit(cal["u_line"], cal["t_league"])
        self.origin_t = int(train["t"].max()) + 1
        self._dump(cal)
        return self

    def _dump(self, cal):
        info = {"variant": self.variant, "origin_t": self.origin_t, "origin": "%d-%02d" % D.week_of(self.origin_t),
                "cal_weeks": cal["weeks"], "cal_leagues": int(len(cal["N"])), "cal_teams": int(len(cal["lp"])),
                "K": self.K, "R": self.R, "chop": self.chop.info, "tot": self.tot.info, "line": self.line.info}
        print(f"[recal] {json.dumps(info)}", flush=True)
        mode = os.environ.get("RECAL_MODE")
        if mode:
            d = os.path.join(RESULTS, "recalibration", f"params_{mode}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{self.variant}__{self.origin_t:02d}.json"), "w") as f:
                json.dump(info, f, indent=1)

    def draws(self, L):
        key = (L["season"], L["week"])
        if self._week is None or self._week[0] != key:
            self._week = None
            b = self.base
            self._week = (key, Draws(L["week_df"], b.tables, b.loading, b.g, S_DRAWS, (SEED, *key)))
        return self._week[1]

    def predict(self, L):
        tot = self.draws(L).totals(L["starters"])
        N = L["n"]
        _, lp, zp, dsd = features(tot, L["starter_proj"])
        return {"p_chop": self.chop.apply(lp, N, zp, dsd),
                "total_q": np.quantile(tot, self.tot.lev, axis=1).T,
                "line_q": np.quantile(tot.min(0), self.line.lev)}
