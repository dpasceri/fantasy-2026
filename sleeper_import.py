#!/usr/bin/env python3
"""Import league data from the Sleeper API into ./data.

Writes untouched API responses to data/raw/ and a normalized
data/league_snapshot.json (teams, schedule/results, rostered players).
Also archives Sleeper's projections for every remaining week under
data/projections/sleeper/<date>/, since Sleeper overwrites old versions.
Safe to re-run any time; the large NFL players file is re-downloaded
at most once a day, per Sleeper's API guidelines.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "config.json"
API = "https://api.sleeper.app/v1"
PROJECTIONS_API = "https://api.sleeper.com/projections/nfl"  # undocumented; used by the Sleeper app
PROJECTION_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
PLAYERS_MAX_AGE_SECONDS = 24 * 60 * 60


def fetch(path):
    return fetch_url(f"{API}{path}")


def fetch_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-2026-import"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def load_players():
    """Sleeper asks that /players/nfl (~10 MB) be called at most once per day."""
    path = RAW_DIR / "players_nfl.json"
    if path.exists() and time.time() - path.stat().st_mtime < PLAYERS_MAX_AGE_SECONDS:
        print("players: using cached file (< 24h old)")
        return json.loads(path.read_text())
    print("players: downloading full NFL player list...")
    players = fetch("/players/nfl")
    save(path, players)
    return players


def import_projections(season, weeks):
    """Save each week's projected stats, keeping only players projected to do something."""
    out_dir = DATA_DIR / "projections" / "sleeper" / date.today().isoformat()
    query = urllib.parse.urlencode([("season_type", "regular")] + [("position[]", p) for p in PROJECTION_POSITIONS])
    for week in weeks:
        rows = fetch_url(f"{PROJECTIONS_API}/{season}/{week}?{query}")
        kept = [
            {k: r.get(k) for k in ("player_id", "team", "opponent", "game_id", "updated_at", "company", "stats")}
            for r in rows
            if any(v for k, v in (r.get("stats") or {}).items()
                   if isinstance(v, (int, float)) and not k.startswith(("adp", "pos_adp", "gp")))
        ]
        save(out_dir / f"week_{week:02d}.json", kept)
    return out_dir


def points(settings, key):
    # Sleeper splits totals into an int and a hundredths field, e.g. fpts=345, fpts_decimal=90.
    return round(settings.get(key, 0) + settings.get(f"{key}_decimal", 0) / 100, 2)


def player_info(pid, players):
    p = players.get(pid) or {}
    name = p.get("full_name") or " ".join(filter(None, [p.get("first_name"), p.get("last_name")])) or pid
    return {
        "name": name,
        "position": p.get("position"),
        "team": p.get("team"),
        "status": p.get("status"),
        "injury_status": p.get("injury_status"),
    }


def main(league_id):
    state = fetch("/state/nfl")
    league = fetch(f"/league/{league_id}")
    users = fetch(f"/league/{league_id}/users")
    rosters = fetch(f"/league/{league_id}/rosters")

    settings = league["settings"]
    first_week = settings.get("start_week", 1)
    last_regular_week = settings["playoff_week_start"] - 1
    last_final_week = settings.get("last_scored_leg", 0)

    playoff_rounds = (settings["playoff_teams"] - 1).bit_length()
    last_playoff_week = settings["playoff_week_start"] + playoff_rounds - 1

    matchups = {}
    for week in range(first_week, last_regular_week + 1):
        matchups[week] = fetch(f"/league/{league_id}/matchups/{week}")
    transactions = {}
    for week in range(first_week, max(state.get("week") or 0, last_final_week + 1) + 1):
        transactions[week] = fetch(f"/league/{league_id}/transactions/{week}")

    save(RAW_DIR / "state.json", state)
    save(RAW_DIR / "league.json", league)
    save(RAW_DIR / "users.json", users)
    save(RAW_DIR / "rosters.json", rosters)
    for week, entries in matchups.items():
        save(RAW_DIR / "matchups" / f"week_{week:02d}.json", entries)
    for week, entries in transactions.items():
        save(RAW_DIR / "transactions" / f"week_{week:02d}.json", entries)

    projection_weeks = range(max(state.get("week") or first_week, first_week), last_playoff_week + 1)
    projections_dir = import_projections(league["season"], projection_weeks) if projection_weeks else None

    players = load_players()
    users_by_id = {u["user_id"]: u for u in users}

    teams = []
    rostered = set()
    for r in sorted(rosters, key=lambda r: r["roster_id"]):
        owner = users_by_id.get(r["owner_id"], {})
        s = r["settings"]
        roster_players = r.get("players") or []
        rostered.update(roster_players)
        teams.append({
            "roster_id": r["roster_id"],
            "owner_id": r["owner_id"],
            "owner": owner.get("display_name"),
            "team_name": ((owner.get("metadata") or {}).get("team_name") or owner.get("display_name") or "").strip(),
            "wins": s.get("wins", 0),
            "losses": s.get("losses", 0),
            "ties": s.get("ties", 0),
            "points_for": points(s, "fpts"),
            "points_against": points(s, "fpts_against"),
            "faab_remaining": settings.get("waiver_budget", 0) - s.get("waiver_budget_used", 0),
            "players": roster_players,
            "starters": r.get("starters") or [],
            "reserve": r.get("reserve") or [],
        })

    schedule = []
    for week, entries in matchups.items():
        games = {}
        for e in entries:
            games.setdefault(e["matchup_id"], []).append({"roster_id": e["roster_id"], "points": e["points"]})
        for matchup_id, sides in sorted(games.items()):
            schedule.append({
                "week": week,
                "matchup_id": matchup_id,
                "final": week <= last_final_week,
                "teams": sides,
            })

    snapshot = {
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "league": {
            "league_id": league["league_id"],
            "name": league["name"],
            "season": league["season"],
            "status": league["status"],
            "nfl_week": state.get("week"),
            "last_final_week": last_final_week,
            "num_teams": settings["num_teams"],
            "regular_season_weeks": [first_week, last_regular_week],
            "playoff_week_start": settings["playoff_week_start"],
            "playoff_teams": settings["playoff_teams"],
            "last_playoff_week": last_playoff_week,
            "trade_deadline_week": settings.get("trade_deadline"),
            "faab_budget": settings.get("waiver_budget", 0),
            "reserve_slots": settings.get("reserve_slots", 0),
            "roster_positions": league["roster_positions"],
            "scoring_settings": league["scoring_settings"],
        },
        "teams": teams,
        "schedule": schedule,
        "players": {pid: player_info(pid, players) for pid in sorted(rostered)},
    }
    save(DATA_DIR / "league_snapshot.json", snapshot)
    print_summary(snapshot)
    if projections_dir:
        print(f"Saved projections for weeks {projection_weeks.start}-{projection_weeks.stop - 1} to {projections_dir}")


def print_summary(snap):
    lg = snap["league"]
    print(f"\n{lg['name']} ({lg['season']}) - NFL week {lg['nfl_week']}, "
          f"weeks 1-{lg['last_final_week']} final")
    print(f"\n{'Team':<30} {'Owner':<16} {'W-L-T':>7} {'PF':>8} {'PA':>8}")
    standings = sorted(snap["teams"], key=lambda t: (t["wins"] + 0.5 * t["ties"], t["points_for"]), reverse=True)
    for t in standings:
        print(f"{t['team_name'][:30]:<30} {t['owner'][:16]:<16} "
              f"{t['wins']}-{t['losses']}-{t['ties']:<3} {t['points_for']:>8.2f} {t['points_against']:>8.2f}")
    names = {t["roster_id"]: t["team_name"] for t in snap["teams"]}
    week = lg["last_final_week"] + 1
    games = [g for g in snap["schedule"] if g["week"] == week]
    if games:
        print(f"\nWeek {week} matchups:")
        for g in games:
            a, b = g["teams"]
            print(f"  {names[a['roster_id']]}  vs  {names[b['roster_id']]}")
    missing = [pid for pid, p in snap["players"].items() if p["position"] is None]
    print(f"\n{len(snap['players'])} rostered players, {len(missing)} not found in Sleeper's player list"
          + (f": {missing}" if missing else ""))
    print(f"Wrote {DATA_DIR / 'league_snapshot.json'}")


if __name__ == "__main__":
    league_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not league_id and CONFIG.exists():
        league_id = json.loads(CONFIG.read_text()).get("trade_league_id")
    if not league_id:
        sys.exit("usage: sleeper_import.py LEAGUE_ID (or set trade_league_id in config.json, see config.example.json)")
    main(league_id)
