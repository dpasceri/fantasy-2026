// Loads guillotine/data/ (written by fetch.py) into the `raw` shape League expects. That folder holds a real
// league and is gitignored, so tests that use it skip themselves when it's missing (as on CI).
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

export const GDIR = fileURLToPath(new URL("../../guillotine/", import.meta.url));
const DATA = join(GDIR, "data");
export const HAVE_DATA = existsSync(join(DATA, "meta.json"));

const J = (name) => JSON.parse(readFileSync(join(DATA, name), "utf8"));
const maybe = (name) => (existsSync(join(DATA, name)) ? J(name) : null);

export function loadLocal() {
  const meta = J("meta.json");
  const weeks = (season, kind, last) => {
    const o = {};
    for (let w = 1; w <= last; w++) {
      const x = maybe(`${season}/${kind}_${w}.json`);
      if (x) o[w] = x;
    }
    return o;
  };
  const matchups = {}, tx = {};
  for (let w = 1; w <= meta.leg; w++) {
    matchups[w] = J(`matchups_${w}.json`);
    tx[w] = J(`tx_${w}.json`);
  }
  return {
    meta, league: J("league.json"), users: J("users.json"), rosters: J("rosters.json"), schedule: J("schedule.json"),
    proj: { season: weeks(meta.season, "proj", 18), calib: weeks(meta.calib_season, "proj", 18) },
    stats: { season: weeks(meta.season, "stats", 18), calib: weeks(meta.calib_season, "stats", 18) },
    matchups, tx, live: maybe("live.json"),
  };
}

export const PY = join(GDIR, ".venv/bin/python");
