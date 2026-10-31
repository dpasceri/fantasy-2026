"""Loaders over data/: the player-week table for both seasons, league settings, 2026 matchups.

proj and actual are league-scoring points from common.League.pts applied to the cached stat
dicts, the exact call model.calibration_rows makes, so they match the core model bit for bit
(Sleeper's own half-PPR number is kept as ps_pts_half_ppr; it carries zero league weight).
A player with a projection row but no stats row scored 0, as in calibration_rows.

Columns of table():
  season, week, t (global chronological index over every cached (season, week)), pid, name, pos,
  team, opp, game, proj, ps_<stat> (stat-level projections; absent = 0, fetch.py drops zeros),
  item (the player track's inclusion rule = model.calibration_rows: proj >= 0.5 and a game id),
  team_proj / opp_proj (summed proj of the player's NFL team / opponent that week),
  team_pos_rank (1 = highest proj at his position on his team that week),
  final (the week's stats file exists), has_stats, actual, as_<stat> (actual stats; NaN when not final),
  inj_now, team_raw.
LEAKS: inj_now is today's injury tag, identical in all 35 proj files (checked), so it is never
point-in-time. pos and name are also current metadata (no position changes seen). The pid universe
is today's (3045 players, the same set every week), so 2025 rows miss players no longer in Sleeper's
player list. team/opp/game are point-in-time only on rows with a real projection (opp set; every
item has one): 2025 midseason trades show up in the week they happened and no game has more than two
teams. A row with no projection that week (bye, injured, inactive, not yet signed; opp missing, proj 0)
gets Sleeper's current team and that team's game instead (e.g. 2025 rows tagged with a team the
player joined in 2026), so team/game are blanked there and the raw value is kept as team_raw.
inj_now and team_raw are LEAKY: pregame()/train_before()/week_frame()/history() strip them.
horizon(origin) hides everything after origin's kickoff from table() while a candidate runs.
"""
import contextlib
import hashlib
import os
import pickle
import re

from . import CACHE, ROOT

import numpy as np
import pandas as pd
from common import DATA, J, League
from model import MIN_PROJ

ACTUAL = ("final", "has_stats", "actual")  # plus every as_* column
LEAKY = ("inj_now", "team_raw")
_TABLE = None
_LG = None
_HORIZON = None       # global week index t0 while horizon() is active
_MASKED = {}          # {"t": t0, "df": table as of t0's kickoff}


def league():
    """The core League object (2026 season, current leg)."""
    global _LG
    if _LG is None:
        _LG = League()
    return _LG


def _weeks():
    out = []
    for season in sorted(int(d) for d in os.listdir(DATA) if d.isdigit()):
        ws = [re.fullmatch(r"proj_(\d+)\.json", f) for f in os.listdir(os.path.join(DATA, str(season)))]
        out += [(season, w) for w in sorted(int(m.group(1)) for m in ws if m)]
    return out


WEEKS = _weeks()                              # every cached (season, week), chronological
T = {sw: i for i, sw in enumerate(WEEKS)}


def t_of(season, week=None):
    """Global index of (season, week); accepts a tuple too."""
    if week is None:
        season, week = season
    return T[(int(season), int(week))]


def week_of(t):
    return WEEKS[int(t)]


def _signature():
    h = hashlib.sha1(open(__file__, "rb").read())
    for core in ("common.py", "model.py"):        # League.pts scoring, MIN_PROJ
        h.update(open(os.path.join(ROOT, core), "rb").read())
    for s, w in WEEKS:
        for f in (f"{s}/proj_{w}.json", f"{s}/stats_{w}.json"):
            p = os.path.join(DATA, f)
            if os.path.exists(p):
                st = os.stat(p)
                h.update(f"{f}:{st.st_size}:{st.st_mtime_ns};".encode())
    st = os.stat(os.path.join(DATA, "league.json"))
    h.update(f"league:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _build():
    lg = league()
    rows = []
    for t, (s, w) in enumerate(WEEKS):
        P = J(f"{s}/proj_{w}.json")
        final = os.path.exists(os.path.join(DATA, f"{s}/stats_{w}.json"))
        S = J(f"{s}/stats_{w}.json") if final else None
        for pid, x in P.items():
            st = x["stats"]
            r = {"season": s, "week": w, "t": t, "pid": pid, "name": x["name"], "pos": x["pos"],
                 "team": x["team"], "opp": x["opp"], "game": x.get("game") or None,
                 "proj": lg.pts(st), "inj_now": x["inj"], "final": final}
            for k, v in st.items():
                r["ps_" + k] = v
            if final:
                a = S.get(pid)
                r["has_stats"] = a is not None
                r["actual"] = lg.pts(a["stats"]) if a else 0.0
                for k, v in (a["stats"] if a else {}).items():
                    r["as_" + k] = v
            rows.append(r)
    df = pd.DataFrame(rows)
    ps = sorted(c for c in df if c.startswith("ps_"))
    as_ = sorted(c for c in df if c.startswith("as_"))
    df[ps] = df[ps].fillna(0.0)
    fin = df["final"].to_numpy(bool)
    A = df[as_].fillna(0.0).to_numpy(float)
    A[~fin] = np.nan                     # unknown, not zero, for weeks without a stats file
    df[as_] = A
    df["has_stats"] = df["has_stats"].eq(True)
    # no projection that week (opp missing, proj 0, never an item): Sleeper fills today's team and its game
    df["team_raw"] = df["team"]
    nr = df["opp"].isna().to_numpy()
    df.loc[nr, "team"] = None
    df.loc[nr, "game"] = None
    df["item"] = (df["proj"] >= MIN_PROJ) & df["game"].notna()
    has_team = df["team"].notna()
    tp = df[has_team].groupby(["t", "team"])["proj"].sum()
    df["team_proj"] = np.where(has_team, tp.reindex(pd.MultiIndex.from_arrays([df["t"], df["team"]])).to_numpy(), np.nan)
    df["opp_proj"] = tp.reindex(pd.MultiIndex.from_arrays([df["t"], df["opp"]])).to_numpy()
    df["team_pos_rank"] = df[has_team].groupby(["t", "team", "pos"])["proj"].rank(ascending=False, method="first")
    lead = ["season", "week", "t", "pid", "name", "pos", "team", "opp", "game", "proj", "item",
            "team_proj", "opp_proj", "team_pos_rank"]
    tail = ["final", "has_stats", "actual"] + as_ + list(LEAKY)
    df = df[lead + ps + tail].sort_values(["t", "pid"], kind="stable").reset_index(drop=True)
    return df


def table():
    """The full player-week table (every cached week, future 2026 weeks included with actual NaN).

    Shared and cached: treat it as read-only (the frames handed to candidates are copies).
    Inside horizon(origin) it is the table as known at origin's kickoff instead (see horizon)."""
    _load()
    return _masked(_HORIZON) if _HORIZON is not None else _TABLE


@contextlib.contextmanager
def horizon(origin):
    """Within this block table() and everything built on it (train_before, week_frame, history,
    starters) shows only what was known at origin's kickoff: later weeks are dropped (their
    projections were made after origin's games), origin's actuals are blanked (final/has_stats False,
    actual and as_* NaN) and the LEAKY columns are removed. The result is a private copy, so a
    candidate that writes into it cannot touch the shared table. The player harness wraps every
    fit/predict in it. data/ files and common.League (League.info, League.proj) are not covered."""
    global _HORIZON
    prev = _HORIZON
    _HORIZON = t_of(origin)
    try:
        yield
    finally:
        _HORIZON = prev
        _MASKED.clear()


def _masked(t0):
    if _MASKED.get("t") != t0:
        d = _TABLE
        w = d[d["t"].to_numpy() <= t0].drop(columns=list(LEAKY))
        m = w["t"].to_numpy() == t0
        if m.any():
            w.loc[m, ["actual"] + [c for c in w if c.startswith("as_")]] = np.nan
            w.loc[m, ["final", "has_stats"]] = False
        _MASKED.clear()
        _MASKED.update(t=t0, df=w)
    return _MASKED["df"]


def _load():
    global _TABLE
    if _TABLE is None:
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f"player_weeks_{_signature()}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                _TABLE = pickle.load(f)
        else:
            _TABLE = _build()
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(_TABLE, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            for f in os.listdir(CACHE):          # drop caches of older data / code
                if f.startswith("player_weeks_") and f.endswith(".pkl") and f != os.path.basename(path):
                    try:
                        os.remove(os.path.join(CACHE, f))
                    except OSError:
                        pass


def proj_stat_cols(df=None):
    return [c for c in (table() if df is None else df) if c.startswith("ps_")]


def actual_cols(df=None):
    return [c for c in (table() if df is None else df) if c in ACTUAL or c.startswith("as_")]


def pregame(df):
    """Drop everything not known before kickoff (actuals) and the LEAKY columns."""
    return df.drop(columns=actual_cols(df) + [c for c in LEAKY if c in df])


def train_before(origin, items_only=False):
    """Every scored player-week strictly before origin (season, week), minus LEAKY columns (a copy)."""
    d = table()
    m = (d["t"].to_numpy() < t_of(origin)) & d["final"].to_numpy()
    if items_only:
        m &= d["item"].to_numpy()
    return d[m].drop(columns=list(LEAKY), errors="ignore")


def week_frame(season, week, items_only=True, pre=True):
    """One week's rows (a copy); pre=True strips actuals so it is safe to hand to a model."""
    d = table()
    m = d["t"].to_numpy() == t_of(season, week)
    if items_only:
        m &= d["item"].to_numpy()
    w = d[m]
    return pregame(w) if pre else w.drop(columns=list(LEAKY), errors="ignore")


def history(pid, before):
    """One player's scored rows strictly before (season, week), oldest first."""
    d = table()
    m = (d["pid"].to_numpy() == str(pid)) & (d["t"].to_numpy() < t_of(before)) & d["final"].to_numpy()
    return d[m].drop(columns=list(LEAKY), errors="ignore")


def item_ids(df):
    """Harness item ids for player-week rows: '<season>-<week:02d>-<pid>'."""
    s = df["season"].astype(str) + "-" + df["week"].map("{:02d}".format) + "-" + df["pid"].astype(str)
    return np.asarray(s.tolist(), dtype=str)


# ---------------------------------------------------------------- league + matchups (2026)
def settings():
    lg = league()
    return {"season": lg.season, "leg": lg.leg, "week": lg.week, "final_week": lg.final_week,
            "calib_season": lg.meta["calib_season"], "slots": list(lg.slots), "scoring": dict(lg.scoring),
            "budget": lg.budget, "n_teams": len(lg.rosters), "me": lg.me}


def matchups():
    """One row per (week, roster): points, starters, starters_points, players, players_points.

    final = the week is over (week < League.week); alive = the roster still had players."""
    lg = league()
    rows = []
    for w, ms in sorted(lg.matchups.items()):
        for m in ms:
            rows.append({"week": w, "roster_id": m["roster_id"], "owner": lg.owner(m["roster_id"]),
                         "points": m.get("points") or 0.0, "starters": list(m.get("starters") or []),
                         "starters_points": list(m.get("starters_points") or []),
                         "players": list(m.get("players") or []),
                         "players_points": dict(m.get("players_points") or {}),
                         "alive": bool(m.get("players")), "final": w < lg.week})
    return pd.DataFrame(rows)


def starters():
    """Long frame of 2026 starters: week, roster_id, slot_i, slot, pid (None for an empty slot),
    sleeper_pts (Sleeper's starter points) and actual / proj under league scoring from table()."""
    lg = league()
    d = table()
    key = d[d["season"] == lg.season].set_index(["week", "pid"])
    rows = []
    for _, m in matchups().iterrows():
        if not m["alive"]:
            continue
        for i, pid in enumerate(m["starters"]):
            pid = None if pid in ("0", 0, None) else str(pid)
            sp = m["starters_points"][i] if i < len(m["starters_points"]) else np.nan
            r = key.loc[(m["week"], pid)] if pid and (m["week"], pid) in key.index else None
            rows.append({"week": m["week"], "roster_id": m["roster_id"], "slot_i": i,
                         "slot": lg.slots[i] if i < len(lg.slots) else None, "pid": pid, "sleeper_pts": sp,
                         "proj": float(r["proj"]) if r is not None else 0.0,
                         "actual": float(r["actual"]) if r is not None else np.nan,
                         "final": m["final"]})
    return pd.DataFrame(rows)
