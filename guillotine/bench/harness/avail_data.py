"""AVAILABILITY data: nflverse injury reports, weekly rosters and snap counts on a dense player x week grid.

Sources (regular season 2016-2026, cached under bench/cache/nflverse/, fetch() downloads what is missing):
  nflverse-data releases injuries_<s>.csv (final game status + practice status + body part per player-week),
  roster_weekly_<s>.csv (team, roster status, gsis/pfr/sleeper ids, birth date), snap_counts_<s>.csv
  (offensive snaps by pfr id), and nfldata games.csv (the schedule, for byes).
Players: every gsis id ever on a roster at QB/RB/WR/TE that has a pfr id (snap counts are keyed by pfr id, so
a missing snap row is a real 0). played = offense snaps >= 1 that week.

Everything is computed causally on a global REG-week axis WEEKS (2016-01 = 0 ...): the features of (player, g)
use outcomes of weeks < g plus the pre-game report / roster status of week g only.
  rep   game status at g: 0 none, 1 Questionable, 2 Doubtful, 3 Out      (final report, pre-game)
  prac  practice status at g: 0 none, 1 full, 2 limited, 3 did not participate
  injg  body-part group of the listed injury (INJ_GROUPS)
  rstat roster status at g: 1 active roster (ACT or INA merged, because INA is the game-day inactive list),
        2 reserve (IR/PUP/NFI), 3 practice squad, 4 other (cut, retired, ...), 0 no row
  d     team games missed in a row before g (0 = played his last team game); r = team games played in a row
  rel   mean offense snap share over his last 3 played games (his role when he plays; a one-off relief
        appearance by a backup does not make him relevant)
  gap   weeks since he last played; pct_last / pct_avg3: snap share of the last 1 / 3 team games (0 = missed)
  games_s / miss_s: team games / share missed this season; games_2y / miss_2y: same over 36 weeks (shrunk
  to 8%); n_rep4: weeks on the report (Q/D/Out) in the last 4 weeks of the season; rep_prev / prac_prev:
  report at his previous team-game week this season; age; wk = week of season; last_wk = 17 or 18.
  pos_rank: his rank by rel among his team's in-scope players at his position who are not ruled out (1 = top;
  a starter coming back from injury pushes his stand-in to 2; 0 = not in scope); game0: his team plays at g.
LEAKS avoided: roster ACT vs INA (game-day) is merged; nothing of week g's game is used for g's features.
A team-game week is the player's team (roster team, filled within the season) having a game; byes are skipped.
"""
import hashlib
import os
import pickle
import subprocess

from . import CACHE

import numpy as np
import pandas as pd

NFLV = os.path.join(CACHE, "nflverse")
SEASONS = tuple(range(2016, 2027))
POS = ("QB", "RB", "WR", "TE")
REL_MIN = 0.4        # relevant = mean offense share >= 40% over his last 3 played games ...
GAP_MAX = 18         # ... and played within the last 18 weeks
H_MAX = 8            # horizons 1..8 weeks ahead (the guillotine season is 16 weeks)
_REL = "https://github.com/nflverse/nflverse-data/releases/download"
FILES = {"games.csv": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"}
for _s in SEASONS:
    FILES[f"injuries_{_s}.csv"] = f"{_REL}/injuries/injuries_{_s}.csv"
    FILES[f"roster_weekly_{_s}.csv"] = f"{_REL}/weekly_rosters/roster_weekly_{_s}.csv"
    FILES[f"snap_counts_{_s}.csv"] = f"{_REL}/snap_counts/snap_counts_{_s}.csv"

INJ_GROUPS = ("none", "hamstring", "ankle", "knee", "concussion", "illness", "rest", "foot", "muscle", "upper",
              "other")
_INJ_KEYS = [("rest", ("not injury", "rest", "personal", "coach")), ("concussion", ("concuss", "head")),
             ("illness", ("illness", "ill", "flu", "covid")), ("hamstring", ("hamstring",)),
             ("ankle", ("ankle",)), ("knee", ("knee", "acl", "mcl", "patella", "meniscus")),
             ("foot", ("foot", "toe", "heel", "achilles", "plantar")),
             ("muscle", ("groin", "calf", "quad", "hip", "thigh", "oblique", "abdom", "back", "pector", "glute",
                         "core", "adductor", "pelvis", "shin", "tibia", "fibula", "leg", "spine")),
             ("upper", ("shoulder", "wrist", "hand", "elbow", "finger", "thumb", "forearm", "arm", "chest", "rib",
                        "neck", "bicep", "tricep", "collarbone", "clavicle", "jaw", "eye", "face"))]
FEATURES = ("rep", "prac", "injg", "ir_ret", "rep_prev", "prac_prev", "d", "r", "pct_last", "pct_avg3", "rel", "gap",
            "games_s", "miss_s", "games_2y", "miss_2y", "n_rep4", "age", "wk", "last_wk", "game0", "pos_rank")
DEPTH = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}     # pos_rank <= DEPTH = a fantasy-starter role on his team
_DATA = None


def fetch(force=False):
    """Download missing source files into bench/cache/nflverse/ (curl). Returns the names fetched."""
    os.makedirs(NFLV, exist_ok=True)
    got = []
    for name, url in FILES.items():
        p = os.path.join(NFLV, name)
        if os.path.exists(p) and not force:
            continue
        subprocess.run(["curl", "-sfL", "--max-time", "120", "-o", p + ".part", url], check=True)
        os.replace(p + ".part", p)
        got.append(name)
    return got


def _read(name, **kw):
    return pd.read_csv(os.path.join(NFLV, name), **kw)


def _signature():
    h = hashlib.sha1(open(__file__, "rb").read())
    for name in sorted(FILES):
        p = os.path.join(NFLV, name)
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{name}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()[:16]


def _injg(primary, secondary):
    s = (primary if isinstance(primary, str) and primary.strip() else secondary)
    if not isinstance(s, str) or not s.strip():
        return 0
    s = s.lower()
    for g, keys in _INJ_KEYS:
        if any(k in s for k in keys):
            return INJ_GROUPS.index(g)
    return INJ_GROUPS.index("other")


def _build():
    fetch()
    games = _read("games.csv", usecols=["season", "game_type", "week", "home_team", "away_team"])
    games = games[games.season.isin(SEASONS) & (games.game_type == "REG")]
    last = games.groupby("season").week.max().to_dict()
    weeks = [(s, w) for s in SEASONS for w in range(1, int(last[s]) + 1)]
    widx = {sw: i for i, sw in enumerate(weeks)}
    W = len(weeks)
    teams = sorted(set(games.home_team) | set(games.away_team))
    tix = {t: i for i, t in enumerate(teams)}
    sched = np.zeros((len(teams), W), bool)
    for s, w, a, b in games[["season", "week", "home_team", "away_team"]].itertuples(index=False):
        sched[tix[a], widx[(s, w)]] = sched[tix[b], widx[(s, w)]] = True

    rcols = ["season", "week", "game_type", "team", "position", "status", "gsis_id", "pfr_id", "sleeper_id",
             "birth_date", "full_name"]
    R = pd.concat([_read(f"roster_weekly_{s}.csv", usecols=rcols, dtype=str) for s in SEASONS], ignore_index=True)
    R = R[(R.game_type == "REG") & R.gsis_id.notna()].copy()
    R["season"], R["week"] = R.season.astype(int), R.week.astype(int)
    R = R[[(s, w) in widx for s, w in zip(R.season, R.week)]]
    sk = R[R.position.isin(POS)]
    pfr = sk.dropna(subset=["pfr_id"]).groupby("gsis_id").pfr_id.agg(lambda x: x.value_counts().index[0])
    pfr = pfr[~pfr.duplicated(keep=False)]                       # drop the few pfr ids shared by two gsis ids
    pos = sk.groupby("gsis_id").position.agg(lambda x: x.value_counts().index[0])
    gsis = sorted(set(pfr.index) & set(pos.index))
    pix = {g: i for i, g in enumerate(gsis)}
    P = len(gsis)
    meta = R[R.gsis_id.isin(pix)].sort_values(["season", "week"]).groupby("gsis_id").agg(
        name=("full_name", "last"), sleeper_id=("sleeper_id", "last"), birth=("birth_date", "first"))
    meta = meta.reindex(gsis)
    players = pd.DataFrame({"gsis": gsis, "pfr": pfr.reindex(gsis).to_numpy(), "pos": pos.reindex(gsis).to_numpy(),
                            "name": meta.name.to_numpy(), "sleeper_id": meta.sleeper_id.to_numpy(),
                            "birth": pd.to_datetime(meta.birth, errors="coerce").to_numpy()})

    # roster grid: one row per player-week, active > reserve > practice squad > other
    R = R[R.gsis_id.isin(pix)].copy()
    R["rs"] = R.status.map({"ACT": 1, "INA": 1, "RES": 2, "DEV": 3}).fillna(4).astype(int)
    R = R.sort_values("rs").drop_duplicates(["gsis_id", "season", "week"])
    ri = R.gsis_id.map(pix).to_numpy()
    rw = np.array([widx[(s, w)] for s, w in zip(R.season, R.week)])
    rstat = np.zeros((P, W), np.int8)
    team = np.full((P, W), -1, np.int16)
    rstat[ri, rw] = R.rs.to_numpy()
    team[ri, rw] = R.team.map(tix).fillna(-1).astype(int).to_numpy()
    season_w = np.array([s for s, _ in weeks])
    wk = np.array([w for _, w in weeks])
    for s in SEASONS:                                            # fill the team within each season
        cols = np.flatnonzero(season_w == s)
        for a, b in zip(cols[1:], cols[:-1]):
            team[:, a] = np.where(team[:, a] >= 0, team[:, a], team[:, b])
        for a, b in zip(cols[-2::-1], cols[:0:-1]):
            team[:, a] = np.where(team[:, a] >= 0, team[:, a], team[:, b])
    G0 = np.where(team >= 0, sched[np.maximum(team, 0), np.arange(W)[None, :]], False)    # schedule only

    # injury report grid (max severity if a player-week is listed twice)
    icols = ["season", "game_type", "week", "gsis_id", "report_primary_injury", "practice_primary_injury",
             "report_status", "practice_status"]
    I = pd.concat([_read(f"injuries_{s}.csv", usecols=icols, dtype=str) for s in SEASONS], ignore_index=True)
    I = I[(I.game_type == "REG") & I.gsis_id.isin(pix)].copy()
    I["season"], I["week"] = I.season.astype(int), I.week.astype(int)
    I = I[[(s, w) in widx for s, w in zip(I.season, I.week)]]
    I["rep"] = I.report_status.map({"Questionable": 1, "Doubtful": 2, "Out": 3}).fillna(0).astype(int)
    ps = I.practice_status.fillna("").str.lower()
    I["prac"] = np.select([ps.str.contains("did not"), ps.str.contains("limited"), ps.str.contains("full")], [3, 2, 1], 0)
    I["injg"] = [_injg(a, b) for a, b in zip(I.report_primary_injury, I.practice_primary_injury)]
    I = I.sort_values(["rep", "prac"], ascending=False).drop_duplicates(["gsis_id", "season", "week"])
    ii = I.gsis_id.map(pix).to_numpy()
    iw = np.array([widx[(s, w)] for s, w in zip(I.season, I.week)])
    rep = np.zeros((P, W), np.int8)
    prac = np.zeros((P, W), np.int8)
    injg = np.zeros((P, W), np.int8)
    rep[ii, iw], prac[ii, iw], injg[ii, iw] = I.rep.to_numpy(), I.prac.to_numpy(), I.injg.to_numpy()

    # snap grid (offense snaps by pfr id) and which team-weeks have snap data at all
    scols = ["season", "game_type", "week", "pfr_player_id", "team", "offense_snaps", "offense_pct"]
    S = pd.concat([_read(f"snap_counts_{s}.csv", usecols=scols, dtype={"pfr_player_id": str}) for s in SEASONS],
                  ignore_index=True)
    S = S[S.game_type == "REG"]
    S = S[[(int(s), int(w)) in widx for s, w in zip(S.season, S.week)]].copy()
    S["g"] = [widx[(int(s), int(w))] for s, w in zip(S.season, S.week)]
    tcov = np.zeros((len(teams), W), bool)
    tt = S.team.map(tix)
    tcov[tt[tt.notna()].astype(int).to_numpy(), S.g[tt.notna()].to_numpy()] = True
    final = np.array([tcov[sched[:, g], g].mean() >= 0.95 if sched[:, g].any() else False for g in range(W)])
    f2i = {p: i for i, p in enumerate(players.pfr)}
    S = S[S.pfr_player_id.isin(f2i)]
    S = S.groupby(["pfr_player_id", "g"], as_index=False).agg(snaps=("offense_snaps", "sum"), pct=("offense_pct", "max"))
    snaps = np.zeros((P, W), np.float32)
    pct = np.zeros((P, W), np.float32)
    si = S.pfr_player_id.map(f2i).to_numpy()
    snaps[si, S.g.to_numpy()] = S.snaps.to_numpy()
    pct[si, S.g.to_numpy()] = np.clip(S.pct.to_numpy(), 0, 1)
    played = snaps >= 1
    pct = np.where(played, pct, 0).astype(np.float32)
    G = G0 | played                         # past team-game weeks (a traded player can play on a bye of his old team)

    F = _features(dict(rep=rep, prac=prac, injg=injg, rstat=rstat, pct=pct, played=played, G=G, G0=G0, final=final,
                       season_w=season_w, wk=wk, last=last, birth=players.birth.to_numpy()))
    F["pos_rank"] = _pos_rank(F, rstat, team, players.pos.to_numpy())
    return {"weeks": weeks, "teams": teams, "sched": sched, "tcov": tcov, "final": final, "players": players,
            "team": team, "rstat": rstat, "G": G, "played": played, "pct": pct, "F": F}


def _features(A):
    rep, prac, pct, played, G, final = A["rep"], A["prac"], A["pct"], A["played"], A["G"], A["final"]
    P, W = pct.shape
    season_w, wk = A["season_w"], A["wk"]
    F = {k: np.zeros((P, W), np.float32) for k in FEATURES}
    d = np.zeros(P, np.int32)
    r = np.zeros(P, np.int32)
    lp = np.full(P, -10 ** 6, np.int64)
    h3 = np.zeros((P, 3), np.float32)
    n3 = np.zeros(P, np.int32)
    gs = np.zeros(P, np.int32)
    ms = np.zeros(P, np.int32)
    lrep = np.zeros(P, np.int8)
    lprac = np.zeros(P, np.int8)
    p3 = np.zeros((P, 3), np.float32)
    np3 = np.zeros(P, np.int32)
    LP = np.zeros((P, W), np.int64)
    gcount = np.zeros((P, W), np.int32)
    mcount = np.zeros((P, W), np.int32)
    for w in range(W):
        if wk[w] == 1:
            gs[:], ms[:], lrep[:], lprac[:] = 0, 0, 0, 0
        F["d"][:, w], F["r"][:, w] = d, r
        F["pct_last"][:, w] = h3[:, 0]
        F["pct_avg3"][:, w] = h3.sum(1) / np.maximum(n3, 1)
        F["games_s"][:, w] = gs
        F["miss_s"][:, w] = ms / np.maximum(gs, 1)
        F["rep_prev"][:, w], F["prac_prev"][:, w] = lrep, lprac
        F["rel"][:, w] = p3.sum(1) / np.maximum(np3, 1)
        LP[:, w] = lp
        g = G[:, w] & final[w]
        pl = played[:, w]
        d = np.where(g, np.where(pl, 0, np.minimum(d + 1, 30)), d)
        r = np.where(g, np.where(pl, np.minimum(r + 1, 30), 0), r)
        lp = np.where(pl, w, lp)
        new = np.concatenate([pct[:, w:w + 1], h3[:, :2]], 1)
        h3 = np.where(g[:, None], new, h3)
        n3 = np.where(g, np.minimum(n3 + 1, 3), n3)
        p3 = np.where(pl[:, None], np.concatenate([pct[:, w:w + 1], p3[:, :2]], 1), p3)
        np3 = np.where(pl, np.minimum(np3 + 1, 3), np3)
        gs += g
        ms += g & ~pl
        gcount[:, w], mcount[:, w] = g, g & ~pl
        lrep = np.where(G[:, w], rep[:, w], lrep)
        lprac = np.where(G[:, w], prac[:, w], lprac)
    cols = np.arange(W)
    ok = LP >= 0
    F["gap"] = np.where(ok, cols[None, :] - LP, 99).astype(np.float32)
    cg = np.concatenate([np.zeros((P, 1)), np.cumsum(gcount, 1)], 1)
    cm = np.concatenate([np.zeros((P, 1)), np.cumsum(mcount, 1)], 1)
    lo = np.maximum(cols - 36, 0)
    g2, m2 = cg[:, cols] - cg[:, lo], cm[:, cols] - cm[:, lo]
    F["games_2y"] = g2.astype(np.float32)
    F["miss_2y"] = ((m2 + 0.8) / (g2 + 10.0)).astype(np.float32)
    onrep = np.concatenate([np.zeros((P, 1)), np.cumsum((rep >= 1) & (rep <= 3), 1)], 1)
    first = np.array([np.flatnonzero(season_w == season_w[w])[0] for w in range(W)])
    lo4 = np.maximum(cols - 4, first)
    F["n_rep4"] = (onrep[:, cols] - onrep[:, lo4]).astype(np.float32)
    F["rep"], F["prac"], F["injg"] = rep.astype(np.float32), prac.astype(np.float32), A["injg"].astype(np.float32)
    F["ir_ret"] = ((A["rstat"] == 2) & (rep >= 1) & (rep <= 2)).astype(np.float32)
    start = np.array([np.datetime64(f"{s}-09-08") + np.timedelta64(7 * (w - 1), "D") for s, w in zip(season_w, wk)])
    birth = A["birth"].astype("datetime64[D]")
    age = (start[None, :] - birth[:, None]).astype("timedelta64[D]").astype(float) / 365.25
    F["age"] = np.where(np.isfinite(age), age, 26.0).astype(np.float32)
    F["wk"] = np.broadcast_to(wk.astype(np.float32), (P, W)).copy()
    F["last_wk"] = np.broadcast_to(np.array([A["last"][s] for s in season_w], np.float32), (P, W)).copy()
    F["game0"] = A["G0"].astype(np.float32)
    return F


def _scope(F, rstat, team, g):
    """In scope at g: relevant, on the active roster (or designated to return from IR), not ruled out."""
    return ((F["rel"][:, g] >= REL_MIN) & (F["gap"][:, g] <= GAP_MAX) & (team[:, g] >= 0)
            & ((rstat[:, g] == 1) | (F["ir_ret"][:, g] > 0)) & (F["rep"][:, g] != 3))


def _pos_rank(F, rstat, team, pos):
    P, W = team.shape
    out = np.zeros((P, W), np.float32)
    pc = np.array([POS.index(p) for p in pos])
    for g in range(W):
        ii = np.flatnonzero(_scope(F, rstat, team, g))
        key = team[ii, g].astype(np.int64) * 8 + pc[ii]
        o = np.lexsort((ii, -F["rel"][ii, g], key))
        ks = key[o]
        start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
        first = np.repeat(start, np.diff(np.r_[start, len(ks)]))
        out[ii[o], g] = np.arange(len(ks)) - first + 1
    return out


def grid():
    """The dense player x week arrays (cached; treat as read-only)."""
    global _DATA
    if _DATA is None:
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f"availability_{_signature()}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                _DATA = pickle.load(f)
        else:
            _DATA = _build()
            _DATA["items"], _DATA["periods"] = _items(_DATA), _periods(_DATA)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(_DATA, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            for f in os.listdir(CACHE):
                if f.startswith("availability_") and f.endswith(".pkl") and f != os.path.basename(path):
                    try:
                        os.remove(os.path.join(CACHE, f))
                    except OSError:
                        pass
    return _DATA


def relevant(A, g):
    """(P,) players in scope at week g: skill player with a pfr id, role share >= REL_MIN when last seen, seen
    within GAP_MAX weeks."""
    F = A["F"]
    return (F["rel"][:, g] >= REL_MIN) & (F["gap"][:, g] <= GAP_MAX)


def _feat_frame(A, ii, gg):
    return {k: A["F"][k][ii, gg] for k in FEATURES}


def _items(A):
    """Every item of every final week from week 2 on (all origins): kind 0 = gameday (Q/D at the origin, target =
    the origin's game), kind 1 = ahead (not ruled out at the origin, target = the week h = 1..H_MAX later in the
    same season on which the origin team plays)."""
    weeks, final, sched, tcov, team = A["weeks"], A["final"], A["sched"], A["tcov"], A["team"]
    pl, rstat, F = A["players"], A["rstat"], A["F"]
    W = len(weeks)
    out = []
    for g in range(W):
        s, w = weeks[g]
        if not final[g] or w == 1:                 # week 1: last season's roles are stale, no origins
            continue
        rep = F["rep"][:, g]
        pop = _scope(F, rstat, team, g)
        tm = np.maximum(team[:, g], 0)
        parts = []
        i0 = np.flatnonzero(pop & (rep >= 1) & (rep <= 2) & sched[tm, g] & tcov[tm, g])
        parts.append((i0, 0, g))
        for h in range(1, H_MAX + 1):
            gt = g + h
            if gt >= W or weeks[gt][0] != s or not final[gt]:
                break
            parts.append((np.flatnonzero(pop & sched[tm, gt] & tcov[tm, gt]), h, gt))
        for ii, h, gt in parts:
            if not len(ii):
                continue
            k = sched[tm[ii], g + 1:gt + 1].sum(1) if h else np.zeros(len(ii), int)
            rec = {"origin": np.full(len(ii), g), "season": s, "week": w, "pidx": ii, "kind": int(h > 0), "h": h,
                   "target": gt, "k": k, "y": A["played"][ii, gt].astype(np.int8)}
            rec.update(_feat_frame(A, ii, np.full(len(ii), g)))
            out.append(pd.DataFrame(rec))
    d = pd.concat(out, ignore_index=True)
    d = d.sort_values(["origin", "kind", "h", "pidx"], kind="stable").reset_index(drop=True)
    d.insert(0, "item_id", [f"{s}-{w:02d}-{p}-h{h}" for s, w, p, h in
                            zip(d.season, d.week, pl.gsis.to_numpy()[d.pidx], d.h)])
    d["pos"] = pl.pos.to_numpy()[d.pidx]
    d["team"] = np.array(A["teams"])[np.maximum(team[d.pidx, d.origin], 0)]
    for c in ("rep", "prac", "injg", "ir_ret", "rep_prev", "prac_prev", "d", "r", "games_s", "n_rep4", "wk",
              "last_wk", "game0", "pos_rank"):
        d[c] = d[c].astype(np.int16)
    d["starter"] = ((d.pos_rank >= 1) & (d.pos_rank <= d.pos.map(DEPTH))).astype(np.int8)
    return d


def _periods(A):
    """Person-periods: every (relevant player, team-game week) of every final week, with the week's pre-game
    features and the outcome played (y). The raw material for discrete-time hazard models."""
    weeks, final, tcov, team, G = A["weeks"], A["final"], A["tcov"], A["team"], A["G"]
    out = []
    for g in range(len(weeks)):
        if not final[g]:
            continue
        tm = np.maximum(team[:, g], 0)
        ii = np.flatnonzero(relevant(A, g) & G[:, g] & (team[:, g] >= 0) & tcov[tm, g])
        rec = {"g": np.full(len(ii), g), "season": weeks[g][0], "week": weeks[g][1], "pidx": ii,
               "rstat": A["rstat"][ii, g], "y": A["played"][ii, g].astype(np.int8)}
        rec.update(_feat_frame(A, ii, np.full(len(ii), g)))
        out.append(pd.DataFrame(rec))
    d = pd.concat(out, ignore_index=True)
    d["pos"] = A["players"].pos.to_numpy()[d.pidx]
    for c in ("rep", "prac", "injg", "ir_ret", "rep_prev", "prac_prev", "d", "r", "games_s", "n_rep4", "wk",
              "last_wk", "game0", "rstat", "pos_rank"):
        d[c] = d[c].astype(np.int16)
    return d


def items():
    return grid()["items"]


def periods():
    return grid()["periods"]


def g_of(season, week=None):
    if week is None:
        season, week = season
    return grid()["weeks"].index((int(season), int(week)))
