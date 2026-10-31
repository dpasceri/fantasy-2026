"""Shared setup: CPU limits, cached Sleeper data, league scoring, lineup logic.

Import this before numpy anywhere else so the thread cap takes effect.
GSIM_THREADS (default 8) caps BLAS threads and worker processes; the process also renices itself.
"""
import os

THREADS = int(os.environ.get("GSIM_THREADS", "8"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1" if os.environ.get("GSIM_WORKER") else str(THREADS))
try:
    os.nice(10)  # yield to whatever else is running on the box
except OSError:
    pass

import json

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
POS = ("QB", "RB", "WR", "TE")
FLEX_OK = {
    "FLEX": ("RB", "WR", "TE"),
    "SUPER_FLEX": ("QB", "RB", "WR", "TE"),
    "REC_FLEX": ("WR", "TE"),
    "WRRB_FLEX": ("RB", "WR"),
}
# chance a player with this week's tag sits; Out/IR are already zeroed in projections. On final Friday
# reports Doubtful players played ~1% and Questionable ~70% (bench/RESULTS.md); Questionable stays at 0.25
# because the sheet runs midweek, when the tag is broader and only a Friday-report rate is measured.
P_OUT = {"Questionable": 0.25, "Doubtful": 0.99, "Out": 1.0, "IR": 1.0, "PUP": 1.0, "Sus": 1.0, "NA": 1.0}


def J(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


class League:
    def __init__(self, me="dpasceri"):
        self.meta = J("meta.json")
        self.league = J("league.json")
        self.season = self.meta["season"]
        self.leg = self.meta["leg"]
        self.final_week = self.meta["final_week"]
        self.scoring = self.league["scoring_settings"]
        self.slots = [s for s in self.league["roster_positions"] if s not in ("BN", "IR", "TAXI")]
        self.budget = self.league["settings"]["waiver_budget"]
        self.users = {u["user_id"]: u for u in J("users.json")}
        self.rosters = {r["roster_id"]: r for r in J("rosters.json")}
        self.proj = {w: J(f"{self.season}/proj_{w}.json") for w in range(1, self.final_week + 1)}
        self.matchups = {w: J(f"matchups_{w}.json") for w in range(1, self.leg + 1)}
        self.tx = {w: J(f"tx_{w}.json") for w in range(1, self.leg + 1)}
        self.info = {}
        for w in sorted(self.proj):
            for pid, x in self.proj[w].items():
                self.info[pid] = {"name": x["name"], "pos": x["pos"], "team": x["team"], "inj": x["inj"]}
        self._pp = {}
        self.me = self.rid_of(me)
        # the week to simulate from: the current leg, or the next one once all its games are final
        sched = [g for g in J("schedule.json") if g["week"] == self.leg]
        done = sched and all(g.get("status") == "complete" for g in sched)
        self.week = self.leg + 1 if done and self.leg < self.final_week else self.leg
        self.pending = self._pending_auction()
        # the simulated week is under way: some games have kicked off (data/live.json, from fetch.py)
        live = J("live.json") if os.path.exists(os.path.join(DATA, "live.json")) else None
        on = live and live["week"] == self.week and any(g["state"] != "pre" for g in live["games"].values())
        self.live = live if on else None

    def game_state(self, pid):
        """(state, share of the game left, points so far) for this player's game this week; state is pre / in / post."""
        if not self.live:
            return "pre", 1.0, 0.0
        s = self.live["stats"].get(pid)
        g = self.live["games"].get((s or {}).get("game") or (self.proj[self.week].get(pid) or {}).get("game"))
        if not g:
            return "pre", 1.0, 0.0
        return g["state"], g["frac_left"], self.pts(s["stats"]) if s else 0.0

    def starters(self, rid, w):
        """The lineup this team has set in Sleeper for week w, by slot (None = empty slot)."""
        m = next((m for m in self.matchups.get(w, []) if m["roster_id"] == rid), None)
        return [p if p and p != "0" else None for p in m["starters"]] if m else None

    def _pending_auction(self):
        """The latest chop whose roster hasn't been through a waiver run yet, if any."""
        chops = [t for w in self.tx for t in self.tx[w] if t["type"] == "chopped"]
        if not chops:
            return None
        t = max(chops, key=lambda x: x["status_updated"])
        if any(x["type"] == "waiver" and x["status_updated"] > t["status_updated"] for w in self.tx for x in self.tx[w]):
            return None
        return {"rid": t["roster_ids"][0], "pool": [p for p in (t.get("drops") or {}) if self.pos(p) in POS],
                "time": t["status_updated"]}

    # --- scoring / projections
    def pts(self, stats):
        return sum(self.scoring.get(k, 0) * v for k, v in stats.items())

    def proj_pts(self, pid, w):
        key = (pid, w)
        if key not in self._pp:
            x = self.proj.get(w, {}).get(pid)
            self._pp[key] = self.pts(x["stats"]) if x else 0.0
        return self._pp[key]

    def pos(self, pid):
        return self.info.get(pid, {}).get("pos")

    def label(self, pid):
        i = self.info.get(pid)
        return f"{i['name']} ({i['pos']}, {i['team'] or 'FA'})" if i else pid

    # --- league members
    def owner(self, rid):
        return self.users.get(self.rosters[rid]["owner_id"], {}).get("display_name", f"roster {rid}")

    def rid_of(self, display_name):
        for rid in self.rosters:
            if self.owner(rid).lower() == display_name.lower():
                return rid
        raise KeyError(display_name)

    def alive(self):
        """Teams still in: rosters with players that haven't been chopped (some leagues leave a chopped roster's players)."""
        chopped = {t["roster_ids"][0] for w in self.tx for t in self.tx[w] if t["type"] == "chopped"}
        return sorted(r for r, x in self.rosters.items() if x.get("players") and r not in chopped)

    def faab(self, rid):
        return self.budget - self.rosters[rid]["settings"].get("waiver_budget_used", 0)

    def players(self, rid):
        return list(self.rosters[rid].get("players") or [])

    def rostered(self):
        return {p for r in self.rosters.values() for p in (r.get("players") or [])}


def eligible(slot, pos):
    return pos == slot or pos in FLEX_OK.get(slot, ())


def best_lineup(slots, cands, with_subs=False):
    """Max-projection lineup. cands: iterable of (value, pid, pos).

    Fixed slots first, then flex slots from most to least restrictive; greedy is optimal for nested
    flex sets like QB/RB/RB/WR/WR/TE/FLEX/FLEX. With with_subs, each slot also gets up to two
    bench fallbacks (best eligible unused players), used when the starter turns out inactive.
    """
    order = sorted((c for c in cands if c[0] > 0), key=lambda c: -c[0])
    used, lineup = set(), []
    fixed = [s for s in slots if s in POS]
    flex = sorted((s for s in slots if s not in POS), key=lambda s: len(FLEX_OK.get(s, POS)))
    for s in fixed + flex:
        pick = next((c for c in order if c[1] not in used and eligible(s, c[2])), None)
        if pick:
            used.add(pick[1])
        lineup.append((s, pick[1] if pick else None))
    if not with_subs:
        return lineup
    bench = [c for c in order if c[1] not in used]
    subs = []
    for s, _ in lineup:
        subs.append([c[1] for c in bench if eligible(s, c[2])][:2])
    return lineup, subs


def min_eligible(slots, lineup, pos_of, val_of):
    """Weakest starter a player of each position would displace (0 if a slot is empty)."""
    out = {}
    for p in POS:
        vals = [val_of(pid) if pid else 0.0 for s, pid in lineup if eligible(s, p)]
        out[p] = min(vals) if vals else np.inf
    return out
