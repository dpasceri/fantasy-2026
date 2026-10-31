"""Point-in-time signals to blend with Sleeper's projection, built causally week by week.

Every feature of a row at week s uses outcomes of weeks < s only (plus week s's pre-game columns),
so one pass over [train weeks .., test week] gives the training rows' features and the test rows'
features alike. Player and opponent signals live in PIT space: z = probit PIT of the outcome under
the base forecast fitted on earlier weeks only (inner rolling origin), so they are "how far above or
below Sleeper did he land", scale-free across projection levels.

Decayed sums use exponential weights in weeks, with SEASON_GAP extra weeks between seasons.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr

POS = ("QB", "RB", "WR", "TE")
SEASON_GAP = 8.0
HL_PLAYER, K_PLAYER = 6.0, 3.0     # player PIT / zero history: half-life (weeks), shrinkage (pseudo-games)
HL_SNAP, K_SNAP = 3.0, 1.0         # snap share
HL_OPP, K_OPP = 8.0, 10.0          # opponent x position points allowed (PIT units)
NAMES = ("ewz", "ewzero", "lneff", "dproj", "new", "missed", "snap", "opp_z", "team_proj", "spread",
         "rank2", "rank3", "td_share", "rec_share", "rush_share", "lp")


def zbar(a, b):
    """E[z | a < z < b] for z ~ N(0, 1) (the point itself when a == b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pt = a == b
    pa, pb = np.exp(-0.5 * np.where(np.isfinite(a), a, 0) ** 2), np.exp(-0.5 * np.where(np.isfinite(b), b, 0) ** 2)
    pa, pb = np.where(np.isfinite(a), pa, 0.0), np.where(np.isfinite(b), pb, 0.0)
    P = np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
    m = (pa - pb) / np.sqrt(2 * np.pi) / np.maximum(P, 1e-300)
    m = np.where(P > 1e-300, m, np.where(np.isfinite(a), a, b))
    return np.where(pt, a, m)


def build(items, rows):
    """Features (len(items), len(NAMES)) for item rows.

    items: item rows (train weeks and the test week) with t, season, pid, pos, team, opp, proj,
      team_proj, opp_proj, team_pos_rank, ps_*, and zb (PIT residual, NaN = none: test rows or no
      inner base yet) and ez (1{y == 0} - base atom at 0).
    rows: every final row of the train weeks (t, season, pid, pos, team, opp, proj, as_off_snp)."""
    items = items.reset_index(drop=True)
    pids = pd.Index(pd.unique(np.concatenate([items["pid"].to_numpy(str), rows["pid"].to_numpy(str)])))
    teams = pd.Index(pd.unique(np.concatenate([items["team"].dropna().to_numpy(str), items["opp"].dropna().to_numpy(str),
                                               rows["team"].dropna().to_numpy(str), rows["opp"].dropna().to_numpy(str)])))
    P, T = len(pids), len(teams)
    ip = pids.get_indexer(items["pid"].to_numpy(str))
    pos_i = np.array([POS.index(p) if p in POS else 0 for p in items["pos"]])
    opp_i = teams.get_indexer(items["opp"].fillna("").to_numpy(str))
    it_t = items["t"].to_numpy(int)
    zb = items["zb"].to_numpy(float)
    ez = items["ez"].to_numpy(float)
    proj = items["proj"].to_numpy(float)

    r = rows[rows["opp"].notna()]
    r_t = r["t"].to_numpy(int)
    r_p = pids.get_indexer(r["pid"].to_numpy(str))
    r_team = teams.get_indexer(r["team"].to_numpy(str))
    r_proj = r["proj"].to_numpy(float)
    r_pos = np.array([POS.index(p) if p in POS else 0 for p in r["pos"]])
    snp = r["as_off_snp"].fillna(0.0).to_numpy(float)
    tmax = r.assign(_s=snp).groupby(["t", "team"])["_s"].transform("max").to_numpy(float)
    share = np.divide(snp, tmax, out=np.zeros_like(snp), where=tmax > 0)
    season_of = dict(zip(np.concatenate([items["t"], rows["t"]]), np.concatenate([items["season"], rows["season"]])))
    games = {s: set(r_team[r_t == s]) for s in np.unique(r_t)}            # teams with a game per train week

    Wz, Sz, S0 = np.zeros(P), np.zeros(P), np.zeros(P)
    Ws, Ss = np.zeros(P), np.zeros(P)
    Wo, So = np.zeros((T + 1, 4)), np.zeros((T + 1, 4))
    last_proj, last_t, last_team = np.full(P, np.nan), np.full(P, -1), np.full(P, -1)
    snap_mean = np.full(4, 0.5)
    sn_sum, sn_n = np.zeros(4), np.zeros(4)
    out = np.zeros((len(items), len(NAMES)))
    prev = None
    for s in np.unique(np.concatenate([it_t, r_t])):
        if prev is not None:
            dt = (s - prev) + (SEASON_GAP if season_of[s] != season_of[prev] else 0.0)
            f = 0.5 ** (dt / HL_PLAYER)
            Wz *= f; Sz *= f; S0 *= f
            f = 0.5 ** (dt / HL_SNAP)
            Ws *= f; Ss *= f
            f = 0.5 ** (dt / HL_OPP)
            Wo *= f; So *= f
        prev = s
        m = np.flatnonzero(it_t == s)
        if len(m):
            p, k = ip[m], pos_i[m]
            out[m, 0] = Sz[p] / (Wz[p] + K_PLAYER)
            out[m, 1] = S0[p] / (Wz[p] + K_PLAYER)
            out[m, 2] = np.log1p(Wz[p])
            lp_ = last_proj[p]
            has = np.isfinite(lp_)
            out[m, 3] = np.where(has, (proj[m] - np.nan_to_num(lp_)) / (proj[m] + 2.0), 0.0)
            out[m, 4] = (~has).astype(float)
            miss = np.zeros(len(m))
            for j, (pi, lt) in enumerate(zip(p, last_t[p])):
                if lt >= 0:
                    tm = last_team[pi]
                    miss[j] = sum(1 for w in range(lt + 1, s) if w in games and tm in games[w])
            out[m, 5] = np.minimum(miss, 3.0)
            out[m, 6] = (Ss[p] + K_SNAP * snap_mean[k]) / (Ws[p] + K_SNAP)
            o = np.where(opp_i[m] >= 0, opp_i[m], T)
            out[m, 7] = So[o, k] / (Wo[o, k] + K_OPP)
            # updates from this week's outcomes (train rows only; test rows have zb NaN)
            u = m[np.isfinite(zb[m])]
            if len(u):
                np.add.at(Wz, ip[u], 1.0)
                np.add.at(Sz, ip[u], zb[u])
                np.add.at(S0, ip[u], ez[u])
                ou = np.where(opp_i[u] >= 0, opp_i[u], T)
                np.add.at(Wo, (ou, pos_i[u]), 1.0)
                np.add.at(So, (ou, pos_i[u]), zb[u])
        g = np.flatnonzero(r_t == s)
        if len(g):
            pg = r_p[g]
            last_proj[pg], last_t[pg], last_team[pg] = r_proj[g], s, r_team[g]
            play = snp[g] > 0
            np.add.at(Ws, pg[play], 1.0)
            np.add.at(Ss, pg[play], share[g][play])
            np.add.at(sn_n, r_pos[g][play], 1.0)
            np.add.at(sn_sum, r_pos[g][play], share[g][play])
            snap_mean = np.where(sn_n > 0, sn_sum / np.maximum(sn_n, 1), 0.5)
    # pre-game columns
    tp, op = items["team_proj"].to_numpy(float), items["opp_proj"].to_numpy(float)
    out[:, 8] = tp
    out[:, 9] = tp - op
    rk = items["team_pos_rank"].fillna(1).to_numpy(float)
    out[:, 10] = (rk == 2).astype(float)
    out[:, 11] = (rk >= 3).astype(float)
    col = lambda c: items[c].to_numpy(float) if c in items else np.zeros(len(items))  # noqa: E731
    pj = np.maximum(proj, 0.5)
    out[:, 12] = (6 * col("ps_rush_td") + 6 * col("ps_rec_td") + 4 * col("ps_pass_td")) / pj
    out[:, 13] = 0.5 * col("ps_rec") / pj
    out[:, 14] = 0.1 * col("ps_rush_yd") / pj
    out[:, 15] = np.log(pj)
    return out
