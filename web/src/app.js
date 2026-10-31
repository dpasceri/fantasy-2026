import { parseLeagueId, problems } from "./checks.js";
import { loadLeague, teams } from "./sleeper.js";

const $ = (id) => document.getElementById(id);
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

function say(text, bad = false, list = []) {
  const el = $("status");
  el.className = bad ? "bad" : "";
  el.textContent = text;
  if (list.length) {
    const ul = document.createElement("ul");
    for (const x of list) ul.appendChild(Object.assign(document.createElement("li"), { textContent: x }));
    el.appendChild(ul);
  }
}

async function load(raw) {
  const id = parseLeagueId(raw);
  $("team-box").hidden = true;
  if (!id) return say("That doesn't look like a Sleeper league ID. It's the long number in the league's URL.", true);
  say("Loading league…");
  let data;
  try {
    data = await loadLeague(id);
  } catch (e) {
    return say(`Couldn't load that league from Sleeper (${e.message}).`, true);
  }
  store.set("league", id);
  const { league } = data;
  const why = problems(league);
  if (why.length) return say(`${league.name} can't be simulated:`, true, why);

  const list = teams(data);
  const sel = $("team");
  sel.replaceChildren(new Option("Pick your team", ""), ...list.map((t) => new Option(t.label, t.rid)));
  const saved = store.get(`team:${id}`);
  if (saved && list.some((t) => String(t.rid) === saved)) sel.value = saved;
  const s = league.settings;
  $("league-facts").textContent = `${league.season} · ${list.length} of ${league.total_rosters} teams alive · ` +
    `$${s.waiver_budget} FAAB · week ${data.state.leg}`;
  $("team-box").hidden = false;
  say(`${league.name} loaded.`);
  sel.onchange = () => store.set(`team:${id}`, sel.value);
}

$("league-form").addEventListener("submit", (e) => {
  e.preventDefault();
  load($("league-id").value);
});

const last = store.get("league");
if (last) $("league-id").value = last;
