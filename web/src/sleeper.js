// Sleeper's public API. Both hosts answer with `access-control-allow-origin: *`, so the browser calls them directly.
export const API = "https://api.sleeper.app/v1";
export const API2 = "https://api.sleeper.com"; // projections, stats, schedule (undocumented; used by the Sleeper app)

export async function getJSON(url, tries = 3) {
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status} ${r.statusText} from ${url}`);
      return await r.json();
    } catch (e) {
      if (i === tries - 1) throw e;
      await new Promise((ok) => setTimeout(ok, 1000 * (i + 1)));
    }
  }
}

/** League, users, rosters and the NFL state: enough to check the format and list the teams. */
export async function loadLeague(id) {
  const [league, users, rosters, state] = await Promise.all([
    getJSON(`${API}/league/${id}`),
    getJSON(`${API}/league/${id}/users`),
    getJSON(`${API}/league/${id}/rosters`),
    getJSON(`${API}/state/nfl`),
  ]);
  if (!league) throw new Error(`No Sleeper league with id ${id}`);
  return { league, users: users || [], rosters: rosters || [], state };
}

/** Alive teams (rosters that still have players), labeled by owner display name and team name. */
export function teams({ users, rosters }) {
  const byId = new Map(users.map((u) => [u.user_id, u]));
  return rosters
    .filter((r) => (r.players || []).length)
    .map((r) => {
      const u = byId.get(r.owner_id) || {};
      const name = u.display_name || `Roster ${r.roster_id}`;
      const team = u.metadata?.team_name;
      return { rid: r.roster_id, name, label: team && team !== name ? `${name} (${team})` : name };
    })
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
}
