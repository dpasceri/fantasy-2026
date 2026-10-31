"""Fit weekly player score distributions from projected-vs-actual history.

Marginal: actual/projected ratio, empirical quantiles by position and projection bucket,
interpolated (in log projection) between bucket medians. Calibrated on last season plus this
season to date, using the week-of projection, so busts, in-game injuries and surprise
inactives are all inside the distribution.

Dependence: one-factor Gaussian copula. Player z = a_pos * F_team + sqrt(1 - a_pos^2) * eps,
F_team = sqrt(g) * G_game + sqrt(1 - g) * T_team. Loadings a_pos and game share g are fit to
same-team and same-game residual correlations.

    python model.py      # writes data/model.npz and prints a calibration report
"""
from common import *  # noqa: F401,F403  (thread caps before numpy)
from common import DATA, POS, J, League, np
from scipy.optimize import least_squares
from scipy.special import ndtri

Q = 256
LEVELS = np.linspace(0.0, 1.0, Q)
MIN_PROJ = 0.5
CORR_MIN_PROJ = 3.0  # only fantasy-relevant players enter the correlation fit


def calibration_rows(lg):
    rows = []
    for season, weeks in ((lg.meta["calib_season"], range(1, 19)), (lg.season, range(1, lg.leg))):
        for w in weeks:
            P, S = J(f"{season}/proj_{w}.json"), J(f"{season}/stats_{w}.json")
            for pid, x in P.items():
                p = lg.pts(x["stats"])
                if p < MIN_PROJ or not x.get("game"):
                    continue
                s = S.get(pid)
                a = lg.pts(s["stats"]) if s else 0.0
                rows.append((x["pos"], p, a, f"{season}-{w}-{x['team']}", f"{season}-{w}-{x['game']}"))
    return rows


def fit(lg, seed=7):
    rng = np.random.default_rng(seed)
    rows = calibration_rows(lg)
    pos = np.array([r[0] for r in rows])
    proj = np.array([r[1] for r in rows])
    act = np.array([r[2] for r in rows])
    ratio = act / proj
    z = np.zeros(len(rows))
    out, report = {}, {}
    for P in POS:
        m = np.flatnonzero(pos == P)
        nb = int(np.clip(len(m) // 300, 5, 14))
        edges = np.quantile(proj[m], np.linspace(0, 1, nb + 1))
        b = np.clip(np.searchsorted(edges, proj[m], side="right") - 1, 0, nb - 1)
        centers, tables, rep = [], [], []
        for k in range(nb):
            idx = m[b == k]
            r = np.sort(ratio[idx])
            centers.append(np.median(proj[idx]))
            tables.append(np.quantile(r, LEVELS))
            # randomized probability integral transform (handles the point mass at 0)
            lo = np.searchsorted(r, ratio[idx], side="left")
            hi = np.searchsorted(r, ratio[idx], side="right")
            u = (lo + rng.random(len(idx)) * (hi - lo)) / len(r)
            z[idx] = ndtri(np.clip(u, 1e-6, 1 - 1e-6))
            a = act[idx]
            rep.append({"proj": round(float(np.median(proj[idx])), 1), "n": int(len(idx)),
                        "mean_actual": round(float(a.mean()), 2), "sd": round(float(a.std()), 2),
                        "p_under_half": round(float((a < 0.5 * proj[idx]).mean()), 3),
                        "p10": round(float(np.quantile(a, 0.1)), 1), "p90": round(float(np.quantile(a, 0.9)), 1)})
        out[f"{P}_centers"] = np.array(centers)
        out[f"{P}_tables"] = np.array(tables)
        report[P] = rep

    # residual correlations: same NFL team, and opposing teams in the same game
    relevant = proj >= CORR_MIN_PROJ
    same, cross = {}, {}
    by_team, by_game = {}, {}
    for i in np.flatnonzero(relevant):
        by_team.setdefault(rows[i][3], []).append(i)
        by_game.setdefault(rows[i][4], []).append(i)
    for grp in by_team.values():
        for x in range(len(grp)):
            for y in range(x + 1, len(grp)):
                i, j = grp[x], grp[y]
                key = tuple(sorted((pos[i], pos[j])))
                s = same.setdefault(key, [0.0, 0])
                s[0] += z[i] * z[j]
                s[1] += 1
    for grp in by_game.values():
        for x in range(len(grp)):
            for y in range(x + 1, len(grp)):
                i, j = grp[x], grp[y]
                if rows[i][3] == rows[j][3]:
                    continue
                key = tuple(sorted((pos[i], pos[j])))
                s = cross.setdefault(key, [0.0, 0])
                s[0] += z[i] * z[j]
                s[1] += 1
    pairs = [(k, v[0] / v[1], v[1]) for k, v in same.items() if v[1] >= 50 and k != ("QB", "QB")]
    pi = {p: n for n, p in enumerate(POS)}

    def resid(a):
        return [np.sqrt(n) * (c - a[pi[k[0]]] * a[pi[k[1]]]) for k, c, n in pairs]

    a = least_squares(resid, x0=np.full(4, 0.3), bounds=(0.0, 0.95)).x
    num = sum(n * c * a[pi[k[0]]] * a[pi[k[1]]] for k, (sc, n) in cross.items() if n >= 50 for c in [sc / n])
    den = sum(n * (a[pi[k[0]]] * a[pi[k[1]]]) ** 2 for k, (sc, n) in cross.items() if n >= 50)
    g = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0
    out["loadings"] = a
    out["game_share"] = np.array(g)
    report["copula"] = {
        "loadings": {p: round(float(a[pi[p]]), 3) for p in POS},
        "game_share": round(g, 3),
        "same_team_corr": {f"{k[0]}-{k[1]}": round(c, 3) for k, c, n in sorted(pairs)},
        "opp_team_corr": {f"{k[0]}-{k[1]}": round(v[0] / v[1], 3) for k, v in sorted(cross.items()) if v[1] >= 50},
        "n_obs": len(rows),
    }
    np.savez(os.path.join(DATA, "model.npz"), **out)
    with open(os.path.join(DATA, "model_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return out, report


class PlayerModel:
    def __init__(self):
        d = np.load(os.path.join(DATA, "model.npz"))
        self.centers = {p: d[f"{p}_centers"] for p in POS}
        self.tables = {p: d[f"{p}_tables"] for p in POS}
        a = d["loadings"]
        self.loading = {p: float(a[i]) for i, p in enumerate(POS)}
        self.game_share = float(d["game_share"])

    def quantiles(self, pos, proj):
        """Score quantiles at LEVELS for a player with this projection."""
        if proj <= 0 or pos not in self.centers:
            return np.zeros(Q)
        c, t = self.centers[pos], self.tables[pos]
        x = np.log(proj)
        lc = np.log(c)
        if x <= lc[0]:
            r = t[0]
        elif x >= lc[-1]:
            r = t[-1]
        else:
            k = np.searchsorted(lc, x) - 1
            wgt = (x - lc[k]) / (lc[k + 1] - lc[k])
            r = (1 - wgt) * t[k] + wgt * t[k + 1]
        return proj * r


if __name__ == "__main__":
    import json as _json
    lg = League()
    _, rep = fit(lg)
    for P in POS:
        print(P)
        for r in rep[P]:
            print("  ", r)
    print(_json.dumps(rep["copula"], indent=1))
