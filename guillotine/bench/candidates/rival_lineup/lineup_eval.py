"""Lineup-level checks on the real league (2026 weeks 1-7), outside the chop metrics.

describe(): what real managers did, measured against the max-projection lineup of the roster they had
  (descriptive, all weeks, never fed to a model): projected points left on the bench D, starters
  projected 0, actual points lost, status-quo share of swaps, and a manager-effect test on D
  (method-of-moments between-manager sd and a within-week label-permutation p-value). CIs resample weeks.
evaluate(mode): rolling-origin forecast of each test week's lineups from rosters (fit on earlier weeks
  only, inside data.horizon), scored against the lineups actually set. Models: max (max projection,
  what run.py assumes), pooled (lineup.LineupFit with kappa = 1e4: one inattention rate) and hier
  (partially pooled manager rates lambda_m). Per team-week:
  crps_proj (CRPS of the starters' projected total), nll (minus the composite log-lik of who started,
  lapse PSI, untempered), zero_brier (Brier of "starts someone projected 0"), d_pred / d_obs (projected points
  left on the bench). All lower is better except d_*; paired differences get a week-block bootstrap CI.
"""
from bench import harness  # noqa: F401

import numpy as np
from bench.candidates.rival_lineup import lineup as LU
from bench.candidates.rival_lineup.candidate import prev_starters, real_obs
from bench.harness import data as D
from bench.harness import metrics as M
from bench.harness.tracks import team as TM

K_EVAL = 4000
B = 2000


def _ci(x, blocks, seed=0):
    r = M.block_bootstrap(M.mean_of("x"), {"x": np.asarray(x, float)}, np.asarray(blocks), B, seed)
    return {k: round(v, 4) for k, v in r.items()}


def _pair(a, b, blocks):
    r = M.paired_bootstrap(M.mean_of("x"), {"x": np.asarray(a, float)}, {"x": np.asarray(b, float)},
                           np.asarray(blocks), B=B)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def _reals():
    return sorted((L for L in TM.leagues()["leagues"] if L["kind"] == "real"), key=lambda L: L["t"])


def describe():
    """Descriptive deviations of real lineups from max projection, 2026 weeks 1-7 (uses outcomes)."""
    rows, swaps = [], []
    views = {L["week"]: TM._view(L, outcome=True) for L in _reals()}
    for w, v in sorted(views.items()):
        proj, act, pos_of = TM._lookup(v["t"])
        prev = prev_starters(views.get(w - 1))
        for k in range(v["n"]):
            rid = int(v["team_ids"][k])
            pids, pos, pr, _, s = LU.team_feats(v["roster"][k], proj, pos_of, prev.get(rid, set()))
            opt = {pids[j] for j in LU.max_lineup(pos, pr) if j >= 0}
            st = {p for p in v["starters"][k] if p}
            po = sum(proj.get(p, 0.0) for p in opt)
            ps = sum(proj.get(p, 0.0) for p in st)
            ao = sum(act.get(p, 0.0) for p in opt)
            rows.append({"week": w, "rid": rid, "D": po - ps, "A": ao - float(v["points"][k]),
                         "zero": int(any(proj.get(p, 0.0) <= 0 for p in st)), "swaps": len(st - opt)})
            sp = prev.get(rid, set())
            if w > 1:
                swaps += [("in", p in sp) for p in st - opt] + [("out", p in sp) for p in opt - st]
    wk = np.array([r["week"] for r in rows])
    D_ = np.array([r["D"] for r in rows])
    rid = np.array([r["rid"] for r in rows])
    out = {"team_weeks": len(rows), "weeks": int(len(set(wk))),
           "bench_proj_pts": _ci(D_, wk), "share_D_pos": _ci(D_ > 1e-9, wk), "share_D_gt3": _ci(D_ > 3, wk),
           "zero_starter_rate": _ci([r["zero"] for r in rows], wk),
           "actual_pts_lost_vs_maxproj": _ci([r["A"] for r in rows], wk),
           "swaps_per_team_week": _ci([r["swaps"] for r in rows], wk)}
    si = [x for k, x in swaps if k == "in"]
    so = [x for k, x in swaps if k == "out"]
    out["status_quo"] = {"in_was_last_week_starter": round(float(np.mean(si)), 3), "n_in": len(si),
                         "out_was_last_week_starter": round(float(np.mean(so)), 3), "n_out": len(so)}
    # manager effect on D: one-way random effects (moments) and a within-week permutation test
    def between_sd(lab):
        ms = [D_[lab == m] for m in np.unique(lab)]
        means, ns = np.array([x.mean() for x in ms]), np.array([len(x) for x in ms])
        within = np.mean([x.var(ddof=1) for x in ms if len(x) > 1])
        return float(np.sqrt(max(0.0, means.var(ddof=1) - within * np.mean(1.0 / ns)))), float(means.var(ddof=1))
    tau, vm = between_sd(rid)
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(B):
        lab = rid.copy()
        for w in np.unique(wk):
            m = np.flatnonzero(wk == w)
            lab[m] = lab[rng.permutation(m)]
        perm.append(between_sd(lab)[1])
    out["manager_effect_D"] = {"between_sd": round(tau, 3), "var_of_manager_means": round(vm, 3),
                               "perm_p": round(float(np.mean(np.array(perm) >= vm)), 4),
                               "n_managers": int(len(np.unique(rid)))}
    return out


def _crps_sample(x, y):
    x = np.sort(np.asarray(x, float))
    K = len(x)
    return float(np.abs(x - y).mean() - (2 * np.arange(1, K + 1) - K - 1) @ x / K ** 2)


def evaluate(mode):
    """Rolling-origin lineup forecasts for the real test weeks of this mode's origins."""
    by = TM.leagues()["by_id"]
    rows = []
    fits = {}
    for s_, w in TM.origins(mode):
        lid = f"r-{s_}-{w:02d}"
        if lid not in by:
            continue
        o = (s_, w)
        train = D.train_before(o)
        with D.horizon(o):
            tl = TM.train_leagues(o)
            zero = LU.ZeroModel(train)
            obs, _ = real_obs(train, tl, zero)
            hier = LU.LineupFit(obs) if obs else None
            mods = {"pooled": LU.LineupFit(obs, kappa_idx=[len(LU.KAPPA) - 1], ll=hier.ll) if obs else None,
                    "hier": hier}
            v = TM.view(by[lid])
        own = {int(r): o for L in tl if L["kind"] == "real" for r, o in zip(L["team_ids"], L["owners"])}
        fits[lid] = {}
        for k, m in mods.items():
            sm = m.summary() if m is not None else None
            if sm:
                sm["lambda_by_owner"] = {own.get(int(r), r): x for r, x in sm.pop("lambda_by_roster").items()}
            fits[lid][k] = sm
        wk = D.week_frame(s_, w, items_only=False, pre=True)
        proj, pos_of = dict(zip(wk["pid"], wk["proj"].astype(float))), dict(zip(wk["pid"], wk["pos"]))
        last = [L for L in tl if L["kind"] == "real" and L["t"] == v["t"] - 1]
        prev = prev_starters(last[0]) if last else {}
        games = set(wk.loc[wk["game"].notna(), "team"].dropna())
        for k in range(v["n"]):
            rid = int(v["team_ids"][k])
            pids, pos, pr, pl, s = LU.team_feats(v["roster"][k], proj, pos_of, prev.get(rid, set()),
                                                 lambda p: zero.test_eligible(p, v["t"], games))
            st = {p for p in v["starters"][k] if p}                    # scoring only
            y = np.array([p in st for p in pids], float)
            p_obs = float(sum(pr[j] for j in range(len(pids)) if y[j]))
            mx = np.zeros(len(pids), bool)
            mx[[j for j in LU.max_lineup(pos, pr) if j >= 0]] = True
            p_max = float(pr[mx].sum())
            row = {"item": f"{lid}-{rid}", "week": w, "rid": rid, "d_obs": p_max - p_obs,
                   "zero_obs": float(any(pr[j] <= 0 for j in range(len(pids)) if y[j]))}
            for name in ("max", "pooled", "hier"):
                if name == "max" or mods[name] is None:
                    S = np.repeat(mx[None], K_EVAL, 0)
                else:
                    S = LU.sample_lineups(mods[name], rid, pos, pr, pl, s, K_EVAL,
                                          LU._seed(LU.SEED_PRED + 1, s_, w, rid), mask=True)
                tot = (S * pr[None]).sum(1)
                P = LU.PSI / 2 + (1 - LU.PSI) * S.mean(0)
                pz = float((S & (pr[None] <= 0)).any(1).mean())
                row[f"crps_proj_{name}"] = _crps_sample(tot, p_obs)
                row[f"nll_{name}"] = -float((y * np.log(P) + (1 - y) * np.log1p(-P)).sum())
                row[f"zero_brier_{name}"] = (pz - row["zero_obs"]) ** 2
                row[f"d_pred_{name}"] = p_max - float(tot.mean())
            rows.append(row)
    if not rows:
        return {"n": 0}
    wk_ = np.array([r["week"] for r in rows])
    col = lambda k: np.array([r[k] for r in rows])
    out = {"n_team_weeks": len(rows), "weeks": sorted(set(int(w) for w in wk_)), "fits": fits,
           "d_obs": _ci(col("d_obs"), wk_), "zero_obs": _ci(col("zero_obs"), wk_)}
    for met in ("crps_proj", "nll", "zero_brier", "d_pred"):
        out[met] = {name: _ci(col(f"{met}_{name}"), wk_) for name in ("max", "pooled", "hier")}
    for met in ("crps_proj", "nll", "zero_brier"):
        out[f"{met}_diff"] = {"hier-max": _pair(col(f"{met}_hier"), col(f"{met}_max"), wk_),
                              "pooled-max": _pair(col(f"{met}_pooled"), col(f"{met}_max"), wk_),
                              "hier-pooled": _pair(col(f"{met}_hier"), col(f"{met}_pooled"), wk_)}
    out["rows"] = rows
    return out
