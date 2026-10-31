#!/usr/bin/env python3
"""Deterministic trade calculator for CMC::MCMC.

Scores Sleeper's projected stats with the league's own scoring rules, picks the
best legal starting lineup for every remaining week (picking up the best free
agents to fill bye weeks and other holes), and compares both teams before and
after a trade. Market values come from FantasyCalc.
Refresh data first:  python3 sleeper_import.py && python3 fantasycalc_import.py

  python3 trade_calc.py league                 every team's projected lineup strength
  python3 trade_calc.py report <team>          one team's roster, lineup and weak spots
  python3 trade_calc.py eval <team> --give "Tyler Warren, Aaron Jones" --get "Trey McBride" [--faab-give 20] [--faab-get 0]
  python3 trade_calc.py faab                   FAAB left per team and every winning bid so far

<team> is a roster id or part of a team or owner name. --give and --get are from
CMC::MCMC's side: --give is what CMC::MCMC sends, --get is what it receives.
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

MY_ROSTER_ID = 12
DATA = Path(__file__).resolve().parent / "data"
SLOT_ELIGIBLE = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "K": {"K"}, "DEF": {"DEF"},
    "FLEX": {"RB", "WR", "TE"}, "WRRB_FLEX": {"RB", "WR"}, "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}
# How many of the best unrostered players per position a team can stream into holes each week.
FREE_AGENT_DEPTH = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}
# Defense tiers as (highest value in tier, scoring key). Projections give averages, so this is approximate.
PTS_ALLOW_TIERS = [(0, "pts_allow_0"), (6, "pts_allow_1_6"), (13, "pts_allow_7_13"), (20, "pts_allow_14_20"),
                   (27, "pts_allow_21_27"), (34, "pts_allow_28_34"), (math.inf, "pts_allow_35p")]
YDS_ALLOW_TIERS = [(99, "yds_allow_0_100"), (199, "yds_allow_100_199"), (299, "yds_allow_200_299"),
                   (349, "yds_allow_300_349"), (399, "yds_allow_350_399"), (449, "yds_allow_400_449"),
                   (499, "yds_allow_450_499"), (549, "yds_allow_500_549"), (math.inf, "yds_allow_550p")]
CAVEATS = ("Assumes no new injuries; bye weeks and other holes are filled with the best free agent. 100/200-yard bonuses aren't in the projections, which slightly undervalues "
           "high-yardage players. FAAB has no projected-points value; weigh it yourself (see `faab`).")


def league_points(stats, scoring, is_def):
    """Fantasy points for a stat line under this league's rules (reproduces Sleeper's own totals)."""
    pts = sum(v * scoring[k] for k, v in stats.items()
              if k in scoring and isinstance(v, (int, float)) and not k.startswith(("pts_allow_", "yds_allow_")))
    if is_def:
        for key, tiers in (("pts_allow", PTS_ALLOW_TIERS), ("yds_allow", YDS_ALLOW_TIERS)):
            if stats.get(key) is not None:
                value = round(stats[key])
                pts += scoring.get(next(name for top, name in tiers if value <= top), 0)
    return pts


def latest(folder, pattern):
    found = sorted(folder.glob(pattern))
    if not found:
        sys.exit(f"No data in {folder}. Run sleeper_import.py and fantasycalc_import.py first.")
    return found[-1]


class League:
    def __init__(self):
        snap = json.loads((DATA / "league_snapshot.json").read_text())
        lg = snap["league"]
        self.scoring = lg["scoring_settings"]
        self.slots = [s for s in lg["roster_positions"] if s != "BN"]
        self.active_limit = len(lg["roster_positions"])
        self.reserve_slots = lg.get("reserve_slots", 0)
        self.teams = {t["roster_id"]: t for t in snap["teams"]}
        self.players = snap["players"]
        self.on_ir = {p for t in snap["teams"] for p in t["reserve"]}

        proj_dir = latest(DATA / "projections" / "sleeper", "*")
        self.proj = {}
        for f in sorted(proj_dir.glob("week_*.json")):
            rows = json.loads(f.read_text())
            self.proj[int(f.stem.split("_")[1])] = {
                r["player_id"]: league_points(r["stats"], self.scoring, r["player_id"].isalpha()) for r in rows}
        rostered = {p for t in snap["teams"] for p in t["players"]}
        with open(DATA / "raw" / "players_nfl.json") as f:
            positions = {pid: info.get("position") for pid, info in json.load(f).items()}
        self.free_agents = {}
        for week, points in self.proj.items():
            by_pos = {}
            for pid, pts in points.items():
                pos = "DEF" if pid.isalpha() else positions.get(pid)
                if pid not in rostered and pos in FREE_AGENT_DEPTH:
                    by_pos.setdefault(pos, []).append(pts)
            self.free_agents[week] = {f"FA_{pos}_{i}": pts for pos, values in by_pos.items()
                                      for i, pts in enumerate(sorted(values, reverse=True)[:FREE_AGENT_DEPTH[pos]], 1)}
        weeks = sorted(self.proj)
        self.ros_weeks = [w for w in weeks if w <= lg["regular_season_weeks"][1]]
        self.po_weeks = [w for w in weeks if w > lg["regular_season_weeks"][1]]
        self.windows = [(f"wks {ws[0]}-{ws[-1]}", ws) for ws in (self.ros_weeks, self.po_weeks) if ws]

        market_file = latest(DATA / "market" / "fantasycalc", "*.json")
        self.market = {str(v["player"].get("sleeperId")): v["value"] for v in json.loads(market_file.read_text())}
        self.sources = f"projections {proj_dir.name} (Sleeper/RotoWire), market values {market_file.stem} (FantasyCalc)"

    # --- basics -------------------------------------------------------------------------------
    def pos(self, pid):
        if pid.startswith("FA_"):
            return pid.split("_")[1]
        return self.players.get(pid, {}).get("position")

    def name(self, pid):
        if pid.startswith("FA_"):
            return f"free-agent {pid.split('_')[1]}"
        return self.players.get(pid, {}).get("name", pid)

    def label(self, pid):
        return f"{self.name(pid)} ({self.pos(pid)})"

    def best_lineup(self, roster, value):
        """Highest-value legal lineup, filling single-position slots before flex slots."""
        pool = sorted(roster, key=lambda p: value.get(p, 0), reverse=True)
        used, lineup = set(), []
        for slot in sorted(self.slots, key=lambda s: len(SLOT_ELIGIBLE[s])):
            pick = next((p for p in pool if p not in used and self.pos(p) in SLOT_ELIGIBLE[slot]), None)
            if pick:
                used.add(pick)
                lineup.append((slot, pick))
        return sum(value.get(p, 0) for _, p in lineup), lineup

    def week_points(self, roster, week):
        """Points of the best lineup from the roster plus that week's best free agents."""
        values = {p: self.proj[week].get(p, 0) for p in roster}
        values.update(self.free_agents[week])
        return self.best_lineup(list(values), values)[0]

    def strength(self, roster, weeks):
        """Average weekly points of the best lineup the roster can field in each of those weeks."""
        return sum(self.week_points(roster, w) for w in weeks) / len(weeks)

    def rate(self, pid, weeks):
        """Projected points per game over the weeks the player is expected to play, and how many weeks that is."""
        played = [self.proj[w][pid] for w in weeks if self.proj[w].get(pid, 0) > 0]
        return (sum(played) / len(played) if played else 0.0), len(played)

    def rate_lineup(self, roster, weeks):
        rates = {p: self.rate(p, weeks)[0] for p in roster}
        return self.best_lineup(roster, rates)[1], rates

    def standings(self):
        return sorted(self.teams, key=lambda r: (self.teams[r]["wins"] + 0.5 * self.teams[r]["ties"],
                                                 self.teams[r]["points_for"]), reverse=True)

    def ranks(self, rosters, weeks):
        scores = {rid: self.strength(roster, weeks) for rid, roster in rosters.items()}
        order = sorted(scores, key=scores.get, reverse=True)
        return scores, {rid: order.index(rid) + 1 for rid in order}

    def find_team(self, text):
        if text.isdigit() and int(text) in self.teams:
            return int(text)
        hits = [rid for rid, t in self.teams.items()
                if text.lower() in f"{t['team_name']} {t['owner']}".lower()]
        if len(hits) != 1:
            sys.exit(f"Team '{text}' matched {len(hits)} teams. Use a roster id: "
                     + ", ".join(f"{rid}={t['team_name']}" for rid, t in self.teams.items()))
        return hits[0]

    def find_players(self, text, rid):
        roster = self.teams[rid]["players"]
        found = []
        for part in filter(None, (s.strip().replace("’", "'") for s in text.split(","))):
            exact = [p for p in roster if part.lower() in (p.lower(), self.name(p).lower())]
            hits = exact or [p for p in roster if part.lower() in self.name(p).lower()]
            if len(hits) != 1:
                sys.exit(f"'{part}' matched {len(hits)} players on {self.teams[rid]['team_name']}. Roster: "
                         + ", ".join(self.name(p) for p in roster))
            found.append(hits[0])
        return found

    # --- trades -------------------------------------------------------------------------------
    def after_trade(self, rid, out, incoming):
        """Roster after a trade, dropping the least useful players if it ends up over the active limit."""
        team = self.teams[rid]
        players = [p for p in team["players"] if p not in out] + incoming
        reserve = [p for p in team["reserve"] if p not in out]
        for p in incoming:
            if p in self.on_ir and len(reserve) < self.reserve_slots:
                reserve.append(p)
        weeks = self.ros_weeks + self.po_weeks
        drops = []
        while len([p for p in players if p not in reserve]) > self.active_limit:
            def points_without(p):
                rest = [q for q in players if q != p]
                return sum(self.week_points(rest, w) for w in weeks)
            active = [p for p in players if p not in reserve]
            drop = max(active, key=lambda p: (round(points_without(p), 6),
                                              -sum(self.proj[w].get(p, 0) for w in weeks)))
            players.remove(drop)
            drops.append(drop)
        return players, reserve, drops

    def evaluate(self, other, give, get, faab_give=0, faab_get=0):
        me = MY_ROSTER_ID
        errors = []
        if faab_give > self.teams[me]["faab_remaining"]:
            errors.append(f"CMC::MCMC only has ${self.teams[me]['faab_remaining']} FAAB")
        if faab_get > self.teams[other]["faab_remaining"]:
            errors.append(f"{self.teams[other]['team_name']} only has ${self.teams[other]['faab_remaining']} FAAB")
        if not (give or get or faab_give or faab_get):
            errors.append("empty trade")
        if errors:
            sys.exit("Invalid trade: " + "; ".join(errors))

        before = {rid: list(t["players"]) for rid, t in self.teams.items()}
        my_after, my_reserve, my_drops = self.after_trade(me, give, get)
        their_after, their_reserve, their_drops = self.after_trade(other, get, give)
        after = {**before, me: my_after, other: their_after}
        names = {me: self.teams[me]["team_name"], other: self.teams[other]["team_name"]}

        def side(players, faab):
            return " + ".join(filter(None, [", ".join(self.label(p) for p in players), f"${faab} FAAB" if faab else ""])) or "nothing"
        lines = [f"TRADE  {names[me]} sends: {side(give, faab_give)}",
                 f"       {names[other]} sends: {side(get, faab_get)}", ""]
        w = max(len(n) for n in names.values()) + 26
        lines.append(f"{'':34}{names[me]:<{w}}{names[other]}")
        for label, weeks in self.windows:
            s0, r0 = self.ranks(before, weeks)
            s1, r1 = self.ranks(after, weeks)
            cells = [f"{s0[rid]:.1f} -> {s1[rid]:.1f} ({s1[rid] - s0[rid]:+.1f}), rank {r0[rid]}->{r1[rid]}"
                     for rid in (me, other)]
            lines.append(f"{'Starting lineup pts/wk, ' + label:<34}{cells[0]:<{w}}{cells[1]}")
        mv_give = sum(self.market.get(p, 0) for p in give)
        mv_get = sum(self.market.get(p, 0) for p in get)
        cells = [f"sends {mv_give:,} / gets {mv_get:,} ({mv_get - mv_give:+,})",
                 f"sends {mv_get:,} / gets {mv_give:,} ({mv_give - mv_get:+,})"]
        lines.append(f"{'Market value (FantasyCalc)':<34}{cells[0]:<{w}}{cells[1]}")
        cells = [f"${self.teams[me]['faab_remaining'] - faab_give + faab_get}",
                 f"${self.teams[other]['faab_remaining'] - faab_get + faab_give}"]
        lines.append(f"{'FAAB after':<34}{cells[0]:<{w}}{cells[1]}")

        def roster_note(players, reserve, drops):
            if drops:
                return "must drop " + ", ".join(self.label(p) for p in drops)
            open_spots = self.active_limit - (len(players) - len(reserve))
            return f"{open_spots} open roster spot(s)" if open_spots > 0 else "roster full, no drops"
        cells = [roster_note(my_after, my_reserve, my_drops), roster_note(their_after, their_reserve, their_drops)]
        lines.append(f"{'Roster':<34}{cells[0]:<{w}}{cells[1]}")
        lines.append("")
        for rid in (me, other):
            for label, weeks in self.windows:
                old, _ = self.rate_lineup(before[rid], weeks)
                new, rates = self.rate_lineup(after[rid], weeks)
                old_ids, new_ids = {p for _, p in old}, {p for _, p in new}
                ins = [f"+{self.name(p)} {rates[p]:.1f}" for p in new_ids - old_ids]
                outs = [f"-{self.name(p)} {self.rate(p, weeks)[0]:.1f}" for p in old_ids - new_ids]
                if ins or outs:
                    lines.append(f"{names[rid]} starters, {label} (pts/game): " + ", ".join(ins + outs))
        lines += ["", "Notes: " + CAVEATS, f"Data: {self.sources}"]
        return "\n".join(lines)

    # --- reports ------------------------------------------------------------------------------
    def activity(self):
        moves = {rid: Counter() for rid in self.teams}
        bids = []
        for f in sorted((DATA / "raw" / "transactions").glob("week_*.json")):
            for tx in json.loads(f.read_text()):
                if tx["status"] != "complete":
                    continue
                for rid in tx.get("roster_ids") or []:
                    moves[rid][tx["type"]] += 1
                if tx["type"] == "waiver":
                    bids.append((tx.get("leg"), tx["roster_ids"][0], (tx.get("settings") or {}).get("waiver_bid", 0),
                                 list((tx.get("adds") or {}).keys())))
        return moves, bids

    def league_table(self):
        rosters = {rid: t["players"] for rid, t in self.teams.items()}
        ranked = [self.ranks(rosters, weeks) for _, weeks in self.windows]
        head = "".join(f"{'Lineup ' + label:>20}" for label, _ in self.windows)
        lines = [f"{'#':>2} {'Team':<28}{'Owner':<17}{'W-L-T':<8}{'PF':>8}{head}{'FAAB':>6}"]
        for i, rid in enumerate(self.standings(), 1):
            t = self.teams[rid]
            cells = "".join(f"{f'{s[rid]:.1f} (#{r[rid]})':>20}" for s, r in ranked)
            lines.append(f"{i:>2} {t['team_name'][:27]:<28}{(t['owner'] or '')[:16]:<17}"
                         f"{t['wins']}-{t['losses']}-{t['ties']:<4}{t['points_for']:>8.1f}{cells}{'$' + str(t['faab_remaining']):>6}")
        lines.append("\nLineup = average weekly points of the best legal lineup from current rosters, "
                     "using projected stats scored with league rules.")
        return "\n".join(lines)

    def slot_values(self, roster, weeks):
        lineup, rates = self.rate_lineup(roster, weeks)
        out, seen = {}, Counter(self.slots)
        grouped = {}
        for slot, p in lineup:
            grouped.setdefault(slot, []).append(p)
        for slot in dict.fromkeys(self.slots):
            for i, p in enumerate(sorted(grouped.get(slot, []), key=rates.get, reverse=True), 1):
                out[f"{slot}{i}" if seen[slot] > 1 else slot] = (p, rates[p])
        return out

    def report(self, rid):
        t = self.teams[rid]
        moves, bids = self.activity()
        standing = self.standings().index(rid) + 1
        pf_rank = sorted(self.teams, key=lambda r: self.teams[r]["points_for"], reverse=True).index(rid) + 1
        spent = sum(b for _, r, b, _ in bids if r == rid)
        active = len([p for p in t["players"] if p not in t["reserve"]])
        m = moves[rid]
        lines = [f"== {t['team_name']} (roster {rid}, owner {t['owner']}) ==",
                 f"Record {t['wins']}-{t['losses']}-{t['ties']} (standing {standing} of {len(self.teams)}), "
                 f"points for {t['points_for']:.1f} (rank {pf_rank}), points against {t['points_against']:.1f}",
                 f"FAAB ${t['faab_remaining']} left (spent ${spent}) | moves so far: {m['waiver']} waiver claims, "
                 f"{m['free_agent']} free-agent adds, {m['trade']} trades | roster {active}/{self.active_limit} active"
                 f" + {len(t['reserve'])} IR"]
        rosters = {r: tm["players"] for r, tm in self.teams.items()}
        strengths = []
        for label, weeks in self.windows:
            s, r = self.ranks(rosters, weeks)
            strengths.append(f"{label} avg {s[rid]:.1f} pts/wk (rank {r[rid]})")
        lines.append("Projected starting lineup strength: " + " | ".join(strengths))

        lines.append("\nStarting lineup by slot, pts/game vs the league's median starter in that slot:")
        profiles = []
        for label, weeks in self.windows:
            mine = self.slot_values(t["players"], weeks)
            league = [self.slot_values(tm["players"], weeks) for tm in self.teams.values()]
            median = {k: statistics.median(v[k][1] for v in league if k in v) for k in mine}
            profiles.append((label, mine, median))
        lines.append(f"  {'slot':<6}" + "".join(f"{label:<36}" for label, _, _ in profiles))
        for k in profiles[0][1]:
            cells = []
            for _, mine, median in profiles:
                p, v = mine.get(k, (None, 0))
                cells.append(f"{(self.name(p) if p else '-')[:20]:<20} {v:5.1f} ({v - median[k]:+5.1f})    ")
            lines.append(f"  {k:<6}" + "".join(cells))

        ros, po = self.windows[0][1], self.windows[-1][1]
        starters = {p for _, p in self.rate_lineup(t["players"], ros)[0]}
        lines.append(f"\nRoster (pts/game when projected to play; games = weeks projected to play):")
        lines.append(f"  {'pos':<4}{'name':<24}{'nfl':<5}{'injury':<13}{self.windows[0][0]:>9}{'games':>6}"
                     f"{self.windows[-1][0]:>11}{'market':>8}  role")
        order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4, "DEF": 5}
        for p in sorted(t["players"], key=lambda p: (order.get(self.pos(p), 9), -self.rate(p, ros)[0])):
            info = self.players.get(p, {})
            ros_rate, games = self.rate(p, ros)
            role = "IR" if p in t["reserve"] else ("starter" if p in starters else "bench")
            lines.append(f"  {self.pos(p) or '?':<4}{self.name(p)[:23]:<24}{(info.get('team') or '-'):<5}"
                         f"{(info.get('injury_status') or '')[:12]:<13}{ros_rate:>9.1f}{games:>6}"
                         f"{self.rate(p, po)[0]:>11.1f}{self.market.get(p, 0):>8,}  {role}")
        for p in t["reserve"]:
            back = next((w for w in sorted(self.proj) if self.proj[w].get(p, 0) > 0), None)
            lines.append(f"  IR: {self.name(p)} " + (f"projected back week {back}; will need an active roster spot"
                                                       if back else "not projected to return this season"))
        lines.append(f"\nMarket = FantasyCalc redraft value (0 = outside their top ~200). Data: {self.sources}")
        return "\n".join(lines)

    def faab_report(self):
        moves, bids = self.activity()
        with open(DATA / "raw" / "players_nfl.json") as f:
            everyone = json.load(f)
        lines = ["FAAB left: " + ", ".join(f"{t['team_name']} ${t['faab_remaining']}"
                                         for t in sorted(self.teams.values(), key=lambda t: -t["faab_remaining"])),
                 "", "Winning waiver bids so far (week, team, bid, player):"]
        for week, rid, bid, adds in sorted(bids, key=lambda b: -b[2]):
            who = ", ".join(everyone.get(p, {}).get("full_name") or p for p in adds)
            lines.append(f"  wk{week} {self.teams[rid]['team_name']:<28} ${bid:<4} {who}")
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("league")
    sub.add_parser("report").add_argument("team")
    ev = sub.add_parser("eval")
    ev.add_argument("team")
    ev.add_argument("--give", default="", help="players CMC::MCMC sends, comma-separated")
    ev.add_argument("--get", default="", help="players CMC::MCMC receives, comma-separated")
    ev.add_argument("--faab-give", type=int, default=0, help="FAAB dollars CMC::MCMC sends")
    ev.add_argument("--faab-get", type=int, default=0, help="FAAB dollars CMC::MCMC receives")
    sub.add_parser("faab")
    args = ap.parse_args()

    lg = League()
    if args.cmd == "league":
        print(lg.league_table())
    elif args.cmd == "report":
        print(lg.report(lg.find_team(args.team)))
    elif args.cmd == "faab":
        print(lg.faab_report())
    else:
        other = lg.find_team(args.team)
        if other == MY_ROSTER_ID:
            sys.exit("Pick the other team in the trade, not CMC::MCMC.")
        give = lg.find_players(args.give, MY_ROSTER_ID)
        get = lg.find_players(args.get, other)
        print(lg.evaluate(other, give, get, args.faab_give, args.faab_get))


if __name__ == "__main__":
    main()
