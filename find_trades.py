#!/usr/bin/env python3
"""Brute-force search for win-win trades between CMC::MCMC and every other team.

Scores every trade where CMC::MCMC sends 1-3 players and receives 1-2 players (wider with flags),
using trade_calc's model (league scoring, best lineup each week, free agents
streamed into holes), on all CPU cores. A trade is "golden" when both teams'
starting lineups improve in both the rest of the regular season and the playoffs
and the other team doesn't give up market value.

  python3 find_trades.py                              # search, save results, print the best finds
  python3 find_trades.py --max-send 4 --max-receive 3  # bigger packages (takes a few minutes)
  python3 find_trades.py --three-team                  # CMC::MCMC -> team A -> team B -> CMC::MCMC
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
from datetime import date
from pathlib import Path

from trade_calc import MY_ROSTER_ID, League

MIN_UPGRADE = 0.75   # a received player must add this many pts/wk to CMC::MCMC's lineup on his own
KEEP_IF = {"me_best": 1.0, "them_worst": -2.5}  # only keep trades this good for us and this tolerable for them
OUT_DIR = Path(__file__).resolve().parent / "trade_search"

LG = League()
WINDOWS = [LG.ros_weeks, LG.po_weeks]
TOTAL = {p: sum(LG.proj[w].get(p, 0) for w in LG.proj) for t in LG.teams.values() for p in t["players"]}
BASE = {rid: [LG.strength(t["players"], ws) for ws in WINDOWS] for rid, t in LG.teams.items()}


def roster_after(rid, out, incoming):
    """Roster after a trade; over the active limit, drop the lowest projected players (fast version of trade_calc's rule)."""
    team = LG.teams[rid]
    players = [p for p in team["players"] if p not in out] + list(incoming)
    reserve = [p for p in team["reserve"] if p not in out]
    for p in incoming:
        if p in LG.on_ir and len(reserve) < LG.reserve_slots:
            reserve.append(p)
    active = [p for p in players if p not in reserve]
    while len(active) > LG.active_limit:
        drop = min(active, key=TOTAL.get)
        active.remove(drop)
        players.remove(drop)
    return players


def gains(rid, roster):
    return [round(LG.strength(roster, ws) - base, 2) for ws, base in zip(WINDOWS, BASE[rid])]


def search_team(job):
    other, send_sets, max_receive = job
    mine, theirs = LG.teams[MY_ROSTER_ID]["players"], LG.teams[other]["players"]
    upgrades = [p for p in theirs if max(gains(MY_ROSTER_ID, mine + [p])) >= MIN_UPGRADE]
    receive_sets = {tuple(sorted(c)) for u in upgrades for n in range(max_receive)
                    for rest in itertools.combinations([p for p in theirs if p != u], n) for c in [(u, *rest)]}
    found = []
    for send in send_sets:
        for receive in receive_sets:
            me = gains(MY_ROSTER_ID, roster_after(MY_ROSTER_ID, send, receive))
            if max(me) < KEEP_IF["me_best"]:
                continue
            them = gains(other, roster_after(other, receive, send))
            if min(them) < KEEP_IF["them_worst"]:
                continue
            found.append({"team": other, "send": list(send), "receive": list(receive),
                          "me_ros": me[0], "me_po": me[1], "them_ros": them[0], "them_po": them[1],
                          "market_send": sum(LG.market.get(p, 0) for p in send),
                          "market_receive": sum(LG.market.get(p, 0) for p in receive)})
    return found


# --- three-team trades: CMC::MCMC sends to A, A sends to B, B sends to CMC::MCMC --------------------
ADD, REMOVE = {}, {}  # each team's lineup change from adding/removing one player; filled before forking
SCREEN = -2.0         # summed single-player effects a partner may show before the exact check (they aren't additive)


def single_player_effects():
    everyone = [p for t in LG.teams.values() for p in t["players"]]
    add = {rid: {p: gains(rid, t["players"] + [p]) for p in everyone if p not in t["players"]}
           for rid, t in LG.teams.items()}
    remove = {rid: {p: gains(rid, [q for q in t["players"] if q != p]) for p in t["players"]}
              for rid, t in LG.teams.items()}
    return add, remove


def subsets(players, sizes, must):
    return [c for n in sizes for c in itertools.combinations(players, n) if any(p in must for p in c)]


def summed(table, players):
    return [sum(table[p][i] for p in players) for i in (0, 1)]


def search_three(job):
    """Screen with summed single-player effects, then score survivors exactly for all three teams."""
    a, b = job
    me, team_a, team_b = (LG.teams[r]["players"] for r in (MY_ROSTER_ID, a, b))
    sweeteners = [()] + [(p,) for p in me]  # optionally CMC::MCMC also sends one player straight to B
    platinum = {p for p in team_b if ADD[MY_ROSTER_ID][p][1] >= 1.5}
    if not platinum:
        return []
    b_sends = subsets(team_b, (1, 2), platinum)
    a_sends = [(s, summed(ADD[b], s), summed(REMOVE[a], s))
               for s in subsets(team_a, (1, 2), {p for p in team_a if max(ADD[b][p]) >= 1.0})]
    me_sends = [(s, summed(ADD[a], s), summed(REMOVE[MY_ROSTER_ID], s))
                for s in subsets(me, (1, 2, 3), {p for p in me if max(ADD[a][p]) >= 0.5})]
    found = []
    value = lambda ps: sum(LG.market.get(p, 0) for p in ps)
    for sb in b_sends:
        me_in, b_out = summed(ADD[MY_ROSTER_ID], sb), summed(REMOVE[b], sb)
        for sa, b_in, a_out in a_sends:
            for sw in sweeteners:
                b_net = [b_in[i] + b_out[i] + sum(ADD[b][p][i] for p in sw) for i in (0, 1)]
                if min(b_net) < SCREEN:
                    continue
                me_in_sw = [me_in[i] + sum(REMOVE[MY_ROSTER_ID][p][i] for p in sw) for i in (0, 1)]
                for sc, a_in, me_out in me_sends:
                    if (set(sc) & set(sw) or min(a_in[i] + a_out[i] for i in (0, 1)) < SCREEN
                            or me_in_sw[0] + me_out[0] < 0 or me_in_sw[1] + me_out[1] < 1.0):
                        continue
                    me_fx = gains(MY_ROSTER_ID, roster_after(MY_ROSTER_ID, sc + sw, sb))
                    if me_fx[1] < 1.5:
                        continue
                    a_fx = gains(a, roster_after(a, sa, sc))
                    b_fx = gains(b, roster_after(b, sb, sa + sw)) if min(a_fx) >= -1.5 else None
                    if b_fx is None or min(b_fx) < -1.5:
                        continue
                    found.append({"a": a, "b": b, "me_sends": list(sc), "me_sends_b": list(sw), "a_sends": list(sa),
                                  "b_sends": list(sb), "me": me_fx, "a_fx": a_fx, "b_fx": b_fx,
                                  "a_market": value(sc) - value(sa), "b_market": value(sa) + value(sw) - value(sb)})
    return found


def main_three():
    global ADD, REMOVE
    ADD, REMOVE = single_player_effects()
    jobs = [(a, b) for a in LG.teams for b in LG.teams if MY_ROSTER_ID not in (a, b) and a != b]
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        results = [t for part in pool.imap_unordered(search_three, jobs) for t in part]
    for t in results:
        t["golden"] = t["me"][0] >= 1.0 and t["me"][1] >= 2.0 and min(t["a_fx"] + t["b_fx"]) >= 0
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{date.today().isoformat()}-three-team.json"
    out.write_text(json.dumps(results, indent=0, ensure_ascii=False))
    golden = sorted((t for t in results if t["golden"]), key=lambda t: -sum(t["me"]))
    print(f"Kept {len(results):,} three-team trades; {len(golden)} are golden (all three lineups better). Saved to {out}")
    name = lambda ps: ", ".join(LG.name(p) for p in ps)
    team = lambda rid: LG.teams[rid]["team_name"][:20]
    for t in golden[:25]:
        extra = f" (+ [{name(t['me_sends_b'])}] straight to {team(t['b'])})" if t["me_sends_b"] else ""
        print(f"  CMC sends [{name(t['me_sends'])}] to {team(t['a'])}{extra}; it sends [{name(t['a_sends'])}] to {team(t['b'])}; "
              f"it sends [{name(t['b_sends'])}] to CMC | CMC {t['me'][0]:+.1f}/{t['me'][1]:+.1f}  "
              f"A {t['a_fx'][0]:+.1f}/{t['a_fx'][1]:+.1f}  B {t['b_fx'][0]:+.1f}/{t['b_fx'][1]:+.1f}  "
              f"market A {t['a_market']:+,} B {t['b_market']:+,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-send", type=int, default=3)
    ap.add_argument("--max-receive", type=int, default=2)
    ap.add_argument("--three-team", action="store_true")
    args = ap.parse_args()
    if args.three_team:
        return main_three()
    mine = LG.teams[MY_ROSTER_ID]["players"]
    send_sets = [c for n in range(1, args.max_send + 1) for c in itertools.combinations(mine, n)]
    chunks = [send_sets[i::32] for i in range(32)]
    jobs = [(rid, chunk, args.max_receive) for rid in LG.teams if rid != MY_ROSTER_ID for chunk in chunks]
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        results = [t for part in pool.imap_unordered(search_team, jobs) for t in part]

    for t in results:
        t["market_gap_for_them"] = t["market_send"] - t["market_receive"]
        t["golden"] = (t["me_ros"] >= 1.0 and t["me_po"] >= 2.0 and t["them_ros"] >= 0 and t["them_po"] >= 0
                       and t["market_gap_for_them"] >= 0)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{date.today().isoformat()}.json"
    out.write_text(json.dumps(results, indent=0, ensure_ascii=False))
    golden = sorted((t for t in results if t["golden"]), key=lambda t: -(t["me_po"] + t["me_ros"]))
    print(f"Kept {len(results):,} trades that help CMC::MCMC by {KEEP_IF['me_best']}+ pts/wk without costing the other team more than "
          f"{-KEEP_IF['them_worst']}; {len(golden)} are golden. Saved to {out}")
    seen = set()
    for t in golden:
        key = (t["team"], tuple(t["receive"]))
        if key in seen:
            continue
        seen.add(key)
        name = lambda ps: ", ".join(LG.name(p) for p in ps)
        print(f"  {LG.teams[t['team']]['team_name'][:24]:<24} send {name(t['send'])} | get {name(t['receive'])} | "
              f"me {t['me_ros']:+.1f}/{t['me_po']:+.1f}  them {t['them_ros']:+.1f}/{t['them_po']:+.1f}  "
              f"market for them {t['market_gap_for_them']:+,}")


if __name__ == "__main__":
    main()
