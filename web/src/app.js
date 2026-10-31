import { parseLeagueId, problems } from "./checks.js";
import { loadAll } from "./data.js";
import { League } from "./league.js";
import { InlinePool, WorkerPool } from "./pool.js";
import { render } from "./report.js";
import { runAll } from "./run.js";
import { loadLeague, teams } from "./sleeper.js";

const $ = (id) => document.getElementById(id);
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
let leagueId = null, running = false;

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
  leagueId = null;
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
  $("league-facts").textContent = `${league.season} · ${league.total_rosters} teams · $${s.waiver_budget} FAAB · week ${data.state.leg}`;
  $("team-box").hidden = false;
  leagueId = id;
  $("run").disabled = !sel.value;
  say(`${league.name} loaded.`);
}

function makePool() {
  const n = Math.max(1, Math.min(16, (navigator.hardwareConcurrency || 4) - 1));
  try {
    return new WorkerPool(n, () => new Worker(new URL("./worker.js", import.meta.url), { type: "module" }));
  } catch {
    return new InlinePool();
  }
}

function progress(label, frac) {
  $("progress").hidden = false;
  $("progress-label").textContent = label;
  $("progress-bar").value = frac;
}

let cssText = null;
async function run() {
  if (running || !leagueId) return;
  const rid = Number($("team").value);
  if (!rid) return;
  running = true;
  $("run").disabled = true;
  $("sheet-box").hidden = true;
  const t0 = performance.now();
  let pool = null;
  try {
    progress("Downloading league data and projections…", 0);
    const raw = await loadAll(leagueId, (d, n) => progress(`Downloading league data and projections (${d} of ${n})…`, (0.15 * d) / n));
    const lg = new League(raw, rid);
    if (!lg.alive().includes(rid)) throw new Error("That team has already been chopped, so there's nothing left to simulate.");
    if (lg.week > lg.final_week) throw new Error("This league's season is over.");
    pool = makePool();
    let stageName = "", stageAt = 0, stageN = 8;
    const r = await runAll(lg, pool, {
      onStage: (name, i, n) => { stageName = name; stageAt = i; stageN = n; progress(`${name}…`, 0.15 + (0.85 * i) / n); },
      onProgress: (d, n) => progress(`${stageName} (${Math.round((100 * d) / n)}%)…`, 0.15 + (0.85 * (stageAt + d / n)) / stageN),
    });
    cssText ??= await (await fetch(new URL("../sheet.css", import.meta.url))).text();
    const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n${render(r, { css: cssText })}</html>`;
    showSheet(html, r);
    const secs = Math.round((performance.now() - t0) / 1000);
    progress(`Done in ${secs} s on ${pool.size} ${pool.size === 1 ? "core" : "cores"}.`, 1);
  } catch (e) {
    console.error(e);
    $("progress").hidden = true;
    say(`The sims failed: ${e.message}`, true);
  } finally {
    if (pool) pool.close();
    running = false;
    $("run").disabled = !$("team").value;
  }
}

function showSheet(html, r) {
  const frame = $("sheet");
  frame.onload = () => {
    const fit = () => { frame.style.height = `${frame.contentDocument.documentElement.scrollHeight}px`; };
    fit();
    new ResizeObserver(fit).observe(frame.contentDocument.body);
  };
  frame.srcdoc = html;
  const a = $("download");
  if (a.href) URL.revokeObjectURL(a.href);
  a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
  a.download = `chop-sheet-${r.season}-week-${r.week}.html`;
  $("sheet-box").hidden = false;
}

$("league-form").addEventListener("submit", (e) => {
  e.preventDefault();
  load($("league-id").value);
});
$("team").addEventListener("change", () => {
  if (leagueId) store.set(`team:${leagueId}`, $("team").value);
  $("run").disabled = running || !$("team").value;
});
$("run").addEventListener("click", run);

const last = store.get("league");
if (last) {
  $("league-id").value = last;
  load(last);
}
