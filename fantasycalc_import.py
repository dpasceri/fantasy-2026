#!/usr/bin/env python3
"""Save FantasyCalc's redraft trade values to data/market/fantasycalc/<date>.json.

These are market prices derived from real fantasy trades, not projections:
they say what other managers tend to accept, which is the other half of a
trade decision. Run sleeper_import.py first so the league settings exist.
"""

import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

API = "https://api.fantasycalc.com/values/current"
DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    league = json.loads((DATA_DIR / "raw" / "league.json").read_text())
    slots = league["roster_positions"]
    params = {
        "isDynasty": "false",
        "numQbs": 2 if "SUPER_FLEX" in slots or slots.count("QB") > 1 else 1,
        "numTeams": league["settings"]["num_teams"],
        "ppr": f"{league['scoring_settings'].get('rec', 0):g}",
    }
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "fantasy-2026-import"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        values = json.load(resp)

    out = DATA_DIR / "market" / "fantasycalc" / f"{date.today().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(values, indent=1, ensure_ascii=False))
    top = ", ".join(f"{v['player']['name']} {v['value']}" for v in values[:5])
    print(f"{len(values)} player values ({params}); top: {top}\nWrote {out}")


if __name__ == "__main__":
    main()
