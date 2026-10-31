"""Pre-game features for player-week rows: projection mix, depth, game environment, own history.

build(train, test) -> (X_train_items, X_test, train_item_mask, names). History features use only
weeks strictly before each row's own week (pid x week matrices, prefix sums shifted one week), so a
training row never sees its own or later outcomes, and the test week sees only train (weeks < test).
"""
import numpy as np
import pandas as pd

from bench.harness import data as D

POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
K_SHRINK = 4.0       # pseudo-weeks shrinking a player's mean residual / zero rate
K_OPP = 30.0         # pseudo-items shrinking the opponent-by-position residual
P0_PRIOR = 0.18      # prior zero rate for the shrunk own zero rate
HALFLIFE = 3.0       # weeks, own residual EWM

NAMES = ["pos", "proj", "team_proj", "opp_proj", "game_tot", "spread", "team_pos_rank", "rel_top",
         "share_team", "td_share", "rec_share", "rush_share", "h_n", "h_e", "h_z", "h_ewm",
         "last_ratio", "trend", "since", "opp_res"]


def _scoring():
    try:
        return D.settings()["scoring"]
    except Exception:                       # league settings unavailable: half-PPR defaults
        return {"rec": 0.5, "rush_td": 6, "rec_td": 6, "pass_td": 4, "rush_yd": 0.1, "rec_yd": 0.1,
                "pass_yd": 0.04}


def _col(d, c):
    return d[c].to_numpy(float) if c in d else np.zeros(len(d))


def _prior(M):
    """(K, T) -> sum over strictly earlier columns."""
    c = np.cumsum(M, 1)
    return np.concatenate([np.zeros((len(M), 1)), c[:, :-1]], 1)


def build(train, test):
    sc = _scoring()
    te = test.copy()
    te["actual"] = np.nan
    keep = ["t", "pid", "pos", "team", "opp", "proj", "item", "team_proj", "opp_proj", "team_pos_rank",
            "actual"] + [c for c in train if c.startswith("ps_")]
    d = pd.concat([train[[c for c in keep if c in train]], te[[c for c in keep if c in te]]], ignore_index=True)
    is_test = np.r_[np.zeros(len(train), bool), np.ones(len(te), bool)]

    proj = d["proj"].to_numpy(float)
    pp = np.maximum(proj, 0.5)
    act = d["actual"].to_numpy(float)
    item = d["item"].to_numpy(bool)
    scored = item & np.isfinite(act) & ~is_test
    t = d["t"].to_numpy(np.int64)
    T = int(t.max()) + 1
    F = {}

    # stat mix of the projection (league scoring)
    s = lambda k: sc.get(k, 0.0) * _col(d, "ps_" + k)
    F["td_share"] = (s("rush_td") + s("rec_td") + s("pass_td")) / pp
    F["rec_share"] = s("rec") / pp
    F["rush_share"] = (s("rush_yd") + s("rush_td")) / pp

    # game environment and depth
    tp, op = d["team_proj"].to_numpy(float), d["opp_proj"].to_numpy(float)
    F["game_tot"], F["spread"], F["share_team"] = tp + op, tp - op, proj / tp
    top = d.groupby(["t", "team", "pos"])["proj"].transform("max").to_numpy(float)
    F["rel_top"] = proj / np.maximum(top, 0.5)

    # own history on a pid x week grid: residual e = (a - p) / (p + 3), zero indicator
    pi, pids = pd.factorize(d["pid"])
    P = len(pids)
    e = np.where(scored, (act - proj) / (proj + 3.0), 0.0)
    E, Z, N = np.zeros((P, T)), np.zeros((P, T)), np.zeros((P, T))
    E[pi, t], Z[pi, t], N[pi, t] = e, np.where(scored, act <= 0, 0.0), scored
    Pm = np.zeros((P, T))
    Pm[pi, t] = proj
    It = np.zeros((P, T), bool)
    It[pi, t] = item
    sE, sZ, sN = _prior(E), _prior(Z), _prior(N)
    F["h_n"] = sN[pi, t]
    F["h_e"] = sE[pi, t] / (sN[pi, t] + K_SHRINK)
    F["h_z"] = (sZ[pi, t] + K_SHRINK * P0_PRIOR) / (sN[pi, t] + K_SHRINK)
    dec = 0.5 ** (1.0 / HALFLIFE)
    num, den = np.zeros((P, T)), np.zeros((P, T))
    for k in range(1, T):                          # EWM of e over strictly earlier scored weeks
        num[:, k] = dec * num[:, k - 1] + E[:, k - 1]
        den[:, k] = dec * den[:, k - 1] + N[:, k - 1]
    F["h_ewm"] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)[pi, t]
    lagP = np.concatenate([np.zeros((P, 1)), Pm[:, :-1]], 1)
    F["last_ratio"] = lagP[pi, t] / pp
    Pi = np.where(It, Pm, 0.0)
    cP, cI = np.cumsum(Pi, 1), np.cumsum(It, 1)
    w4 = lambda C: np.concatenate([np.zeros((P, 1)), C[:, :-1]], 1) - np.concatenate([np.zeros((P, 5)), C[:, :-5]], 1)[:, :T]
    rP, rI = w4(cP), w4(cI)                        # last 4 weeks' item projections
    F["trend"] = np.where(rI[pi, t] > 0, proj / np.maximum(rP[pi, t] / np.maximum(rI[pi, t], 1), 0.5), 1.0)
    last = np.full((P, T), -100.0)
    for k in range(1, T):
        last[:, k] = np.where(It[:, k - 1], k - 1, last[:, k - 1])
    F["since"] = np.clip(t - last[pi, t], 0, 20)

    # opponent-by-position residual over earlier weeks (shrunk to 0)
    ok = d["opp"].notna().to_numpy()
    key = d["opp"].astype(str) + "|" + d["pos"].astype(str)
    ki, keys = pd.factorize(key)
    K = len(keys)
    Eo, No = np.zeros((K, T)), np.zeros((K, T))
    np.add.at(Eo, (ki, t), e)
    np.add.at(No, (ki, t), scored.astype(float))
    F["opp_res"] = np.where(ok, _prior(Eo)[ki, t] / (_prior(No)[ki, t] + K_OPP), 0.0)

    F["pos"] = d["pos"].map(POS_CODE).fillna(-1).to_numpy(float)
    for c in ("proj", "team_proj", "opp_proj", "team_pos_rank"):
        F[c] = d[c].to_numpy(float)
    X = np.stack([np.asarray(F[c], float) for c in NAMES], 1)
    X = np.where(np.isfinite(X), X, np.nan)
    tr_items = item & np.isfinite(act) & ~is_test
    return X[tr_items], X[is_test], tr_items[: len(train)], NAMES
