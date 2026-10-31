"""Pull everything the simulator needs from Sleeper's public API into data/.

    python fetch.py                      # current league, current week
    python fetch.py --calib-season 2025  # also which past season to calibrate on

Weekly projections/stats are slimmed to the fields the model uses, so the cache stays small. The week in
progress also goes to data/live.json (game states, clocks and box scores so far) for in-game runs.
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://api.sleeper.app/v1"
API2 = "https://api.sleeper.com"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_TEAM = {"WSH": "WAS"}  # ESPN abbreviations that differ from Sleeper's
POS = ("QB", "RB", "WR", "TE")
CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
POS_Q = "&".join(f"position[]={p}" for p in POS)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "guillotine-sim"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def save(name, obj):
    path = os.path.join(DATA, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


def slim_weekly(rows, keys):
    out = {}
    for r in rows or []:
        p = r.get("player") or {}
        if p.get("position") not in POS:
            continue
        st = r.get("stats") or {}
        out[r["player_id"]] = {
            "team": r.get("team"),
            "opp": r.get("opponent"),
            "game": r.get("game_id"),
            "pos": p.get("position"),
            "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "inj": p.get("injury_status"),
            "stats": {k: v for k, v in st.items() if (k in keys) and v},
        }
    return out


def live_games(season, leg, sched):
    """This week's games: state pre / in / post and the share of the game still to play.

    Sleeper's schedule gives the state; ESPN's scoreboard (faster, and the only one with a clock) overrides it
    when it answers. frac_left counts regulation time; overtime counts as 2% left.
    """
    games = {}
    for g in sched:
        if g["week"] == leg:
            st = {"in_game": "in", "complete": "post"}.get(g.get("status"), "pre")
            games[g["game_id"]] = {"away": g["away"], "home": g["home"], "state": st,
                                   "frac_left": {"pre": 1.0, "in": 0.5, "post": 0.0}[st], "clock": None, "kickoff": g.get("date")}
    try:
        events = get(f"{ESPN}?seasontype=2&week={leg}&dates={season}")["events"]
    except Exception as e:
        print(f"ESPN scoreboard unavailable ({e}); using Sleeper's game status only")
        events = []
    by_teams = {frozenset((g["away"], g["home"])): gid for gid, g in games.items()}
    for ev in events:
        c = ev["competitions"][0]
        gid = by_teams.get(frozenset(ESPN_TEAM.get(t["team"]["abbreviation"], t["team"]["abbreviation"]) for t in c["competitors"]))
        if gid is None:
            print(f"ESPN game {ev.get('shortName')} has no Sleeper match")
            continue
        s, g = c["status"], games[gid]
        g["state"], g["kickoff"] = s["type"]["state"], ev["date"]
        if g["state"] == "in":
            per, clk = int(s.get("period") or 1), float(s.get("clock") or 0)
            g["frac_left"] = max((4 - per) * 900 + clk, 0) / 3600 if per <= 4 else 0.02
            g["clock"] = s["type"].get("shortDetail") or f"Q{per} {s.get('displayClock', '')}"
        else:
            g["frac_left"] = 1.0 if g["state"] == "pre" else 0.0
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", help="Sleeper league id (default: guillotine_league_id in config.json)")
    ap.add_argument("--calib-season", type=int, default=None, help="past season used for calibration (default: season-1)")
    args = ap.parse_args()
    L = args.league
    if not L and os.path.exists(CONFIG):
        with open(CONFIG) as f:
            L = json.load(f).get("guillotine_league_id")
    if not L:
        ap.error("pass --league or set guillotine_league_id in config.json (see config.example.json)")

    state = get(f"{API}/state/nfl")
    league = get(f"{API}/league/{L}")
    season = int(league["season"])
    leg = int(state["leg"])
    s = league["settings"]
    final_week = s.get("start_week", 1) + league["total_rosters"] - 2  # one chop a week until one team is left
    calib = args.calib_season or season - 1
    keys = set(league["scoring_settings"]) | {"gp", "gms_active", "pts_half_ppr", "off_snp"}

    jobs = {
        "state.json": f"{API}/state/nfl",
        "league.json": f"{API}/league/{L}",
        "users.json": f"{API}/league/{L}/users",
        "rosters.json": f"{API}/league/{L}/rosters",
        "schedule.json": f"{API2}/schedule/nfl/regular/{season}",
    }
    for w in range(1, leg + 1):
        jobs[f"matchups_{w}.json"] = f"{API}/league/{L}/matchups/{w}"
        jobs[f"tx_{w}.json"] = f"{API}/league/{L}/transactions/{w}"
    weekly = {}
    for w in range(1, max(final_week, 17) + 1):
        weekly[f"{season}/proj_{w}.json"] = f"{API2}/projections/nfl/{season}/{w}?season_type=regular&{POS_Q}"
    for w in range(1, leg):
        weekly[f"{season}/stats_{w}.json"] = f"{API2}/stats/nfl/{season}/{w}?season_type=regular&{POS_Q}"
    for w in range(1, 19):
        weekly[f"{calib}/proj_{w}.json"] = f"{API2}/projections/nfl/{calib}/{w}?season_type=regular&{POS_Q}"
        weekly[f"{calib}/stats_{w}.json"] = f"{API2}/stats/nfl/{calib}/{w}?season_type=regular&{POS_Q}"
    # past calibration season never changes; skip it if cached
    weekly = {k: v for k, v in weekly.items() if not (k.startswith(f"{calib}/") and os.path.exists(os.path.join(DATA, k)))}

    def run(item):
        name, url = item
        obj = get(url)
        if name in weekly:
            obj = slim_weekly(obj, keys)
        save(name, obj)
        return name

    with ThreadPoolExecutor(max_workers=8) as ex:
        done = list(ex.map(run, list(jobs.items()) + list(weekly.items())))

    # the week in progress: game states and live box scores (Sleeper updates them during games)
    games = live_games(season, leg, json.load(open(os.path.join(DATA, "schedule.json"))))
    stats = slim_weekly(get(f"{API2}/stats/nfl/{season}/{leg}?season_type=regular&{POS_Q}"), keys)
    save("live.json", {"week": leg, "fetched_at": int(time.time()), "games": games, "stats": stats})
    n = {s: sum(g["state"] == s for g in games.values()) for s in ("pre", "in", "post")}
    print(f"week {leg} live: {n['post']} final, {n['in']} in progress, {n['pre']} not started; {len(stats)} box scores")

    save("meta.json", {"league": L, "season": season, "leg": leg, "final_week": final_week,
                       "calib_season": calib, "fetched_at": int(time.time())})
    print(f"fetched {len(done)} files  season={season} leg={leg} final_week={final_week} calib={calib}")


if __name__ == "__main__":
    main()
