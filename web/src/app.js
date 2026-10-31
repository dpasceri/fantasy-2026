import { parseLeagueId, problems } from "./checks.js";
import { loadAll } from "./data.js";
import { League } from "./league.js";
import { InlinePool, WorkerPool } from "./pool.js";
import { esc, fmtComma, fmtG } from "./pyfmt.js";
import { pct, render } from "./report.js";
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

/* sheet-doc:begin */
// The sheet is one self-contained file: render()'s markup (byte-identical to report.py's), sheet.css, and two additions
// made here so report.js stays untouched: the key figures under the header, and a script that moves the calls to the
// top, numbers the sections into parts, builds the section rail (a bar with a menu on narrow screens), links each call
// to the section behind it and steps through sections with j / k. report.py's markup asks for web fonts the web sheet
// doesn't use; they're dropped so the sheet makes no requests.
function keyFigures(r) {
  const mine = r.standings.find((s) => s.me);
  const safe = [...r.standings].sort((a, b) => a.p_chop_now - b.p_chop_now).indexOf(mine) + 1;
  const pw = r.power.find((x) => x.me);
  const p = mine.p_chop_now;
  const risk = p < 0.08 ? "good" : p < 0.15 ? "warn" : "crit";
  const left = r.faab_pace.left_now;
  const kf = (label, short, mini, bar, value, sub, cls = "") =>
    `<div class="kf" data-short="${short}" data-mini="${mini}"${bar ? " data-bar" : ""}><span class="kl">${label}</span>`
    + `<span class="kv${cls ? " " + cls : ""}">${value}</span><span class="ks">${sub}</span></div>`;
  const [med, line] = [fmtG(r.my_score["0.5"]), fmtG(r.chop_line["0.5"])];
  return `<div class="keys" role="group" aria-label="Key figures" data-who="${esc(r.me)}" data-when="week ${r.week} of ${r.final_week}">`
    + kf(`Chop risk · week ${r.week}`, "chop", pct(p), true, pct(p), `#${safe} safest of ${r.standings.length}`, risk)
    + kf("Win the league", "win", `${pct(pw.p_win, 0)} → ${pct(pw.p_win_policy, 0)}`, true, pct(pw.p_win, 0),
      `<b>${pct(pw.p_win_policy, 0)}</b> bidding as recommended · #${r.power.indexOf(pw) + 1} of ${r.power.length}`)
    + kf(`Your median · week ${r.week}`, "median", `${med} / ${line}`, false, med, `chop line <b>${line}</b> (lowest rival)`)
    + kf("FAAB", "faab", `$${r.faab_me}`, true, `$${r.faab_me}`,
      left > 0 ? `<b>${Math.round((100 * r.faab_me) / left)}%</b> of the $${fmtComma(left)} left in the league` : "none left in the league")
    + "</div>";
}

// Runs inside the sheet (serialized with Function.prototype.toString), so it can't use anything from this module.
function sheetScript() {
  const d = document, board = d.querySelector(".board"), wrap = d.querySelector(".wrap");
  if (!board || !wrap) return;
  const callsBox = board.querySelector(".calls")?.closest(".box");
  if (callsBox) board.prepend(callsBox); // summary first
  const secs = [...board.children].filter((e) => e.matches("section.box"));
  if (!secs.length) return;
  const RULES = [ // h2 text -> id, short label, part
    [/^The calls/, "calls", "The calls", "Summary"],
    [/so far$/, "live", "Live scores", "Summary"],
    [/^Power rankings/, "power", "Power rankings", "Summary"],
    [/chop line$/, "chop-line", "Chop line", "This week"],
    [/: lineup$/, "lineup", "Lineup", "This week"],
    [/^Free agents/, "free-agents", "Free agents", "This week"],
    [/^Your roster/, "roster", "Roster", "This week"],
    [/^Auction: (?:if )?(.+) (?:is|was) chopped$/, "auction", null, "Auctions"],
    [/FAAB dollar/, "faab-value", "FAAB value", "Money"],
    [/^How hard to bid/, "bid-styles", "Bid styles", "Money"],
    [/FAAB pace/, "faab-pace", "FAAB pace", "Money"],
    [/^Rival bidders/, "rivals", "Rival bidders", "Money"],
    [/^The math: (.+)$/, "math", null, "Back of the sheet"],
    [/^Re-run/, "re-run", "Re-run", "Back of the sheet"],
  ];
  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const el = (tag, cls, text) => {
    const e = d.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  let part = "";
  const items = secs.map((sec, i) => {
    const h = sec.querySelector("h2");
    const title = h ? [...h.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join("").trim() : "";
    const deck = h?.querySelector("small")?.textContent ?? "";
    let id = "s" + (i + 1), label = title || "Section " + (i + 1), extra = "", owner = null;
    for (const [re, key, name, grp] of RULES) {
      const m = title.match(re);
      if (!m) continue;
      part = grp;
      if (key === "auction") {
        owner = label = m[1];
        id = "auction-" + slug(m[1]);
        extra = /on waivers/.test(deck) ? "now" : (deck.match(/^[\d.]+%/) || [""])[0];
      } else if (key === "math") {
        label = "Math: " + m[1];
        id = "math-" + slug(m[1]);
      } else {
        label = name;
        id = key;
      }
      break;
    }
    if (d.getElementById(id)) id += "-" + (i + 1);
    sec.id = id;
    if (h && part) h.dataset.part = part;
    return { id, label, extra, owner, part, n: String(i + 1).padStart(2, "0") };
  });

  // the rail: who, key figures, sections by part
  const nav = el("nav", "toc");
  nav.setAttribute("aria-label", "Sections");
  const bar = el("button", "toc-bar"), cur = el("span", "toc-cur"), mini = el("span", "toc-mini");
  bar.type = "button";
  bar.setAttribute("aria-expanded", "false");
  bar.setAttribute("aria-controls", "toc-body");
  bar.append(cur, mini);
  const body = el("div", "toc-body");
  body.id = "toc-body";
  const keys = d.querySelector(".keys");
  if (keys) {
    body.append(el("div", "toc-who", keys.dataset.who), el("div", "toc-wk", keys.dataset.when));
    const dl = el("dl", "toc-keys");
    for (const kf of keys.querySelectorAll(".kf")) {
      const cls = kf.querySelector(".kv").className.replace("kv", "").trim();
      dl.append(el("dt", null, kf.dataset.short), el("dd", cls, kf.dataset.mini));
      if (kf.hasAttribute("data-bar")) {
        const b = el("b", cls);
        b.append(el("span", null, kf.dataset.short), kf.querySelector(".kv").textContent);
        mini.append(b);
      }
    }
    body.append(dl);
  }
  const list = el("div", "toc-list");
  let grp = null, lastPart = null;
  const links = items.map((it) => {
    if (!grp || it.part !== lastPart) {
      grp = el("div", "toc-group");
      if (it.part) grp.append(el("div", "toc-gh", it.part));
      list.append(grp);
      lastPart = it.part;
    }
    const a = el("a");
    a.href = "#" + it.id;
    a.append(el("span", "n", it.n), el("span", "t", it.label), el("span", "x", it.extra));
    grp.append(a);
    return a;
  });
  const hint = el("p", "toc-hint");
  hint.append(el("kbd", null, "j"), " ", el("kbd", null, "k"), " next / previous section");
  body.append(list, hint);
  nav.append(bar, body);
  wrap.prepend(nav);
  wrap.classList.add("has-toc");

  // wide tables whose first column is a name keep it pinned while they scroll sideways (narrow screens; see sheet.css)
  for (const t of d.querySelectorAll(".tbl > table:not(.heat)")) {
    const th = t.querySelector("th")?.textContent.trim();
    if (th && th !== "#" && th !== "Wk") t.classList.add("pin");
  }

  // each call links to the section behind it
  const CALLS = { "Best shape": "power", "This week": "chop-line", Lineup: "lineup", "Free pickup": "free-agents",
    Money: "bid-styles", "League pace": "faab-pace" };
  for (const k of d.querySelectorAll(".calls .k")) {
    const name = k.textContent, txt = k.nextElementSibling?.textContent ?? "";
    const it = name === "Auction"
      ? items.find((x) => x.owner && (txt.startsWith(`If ${x.owner} is chopped`) || txt.startsWith(`${x.owner} was chopped`)))
      : items.find((x) => x.id === CALLS[name]);
    if (!it) continue;
    const a = el("a", "k", name);
    a.href = "#" + it.id;
    a.append(el("span", "to", "→ " + it.n));
    k.replaceWith(a);
  }

  // where you are: the rail's current item, the bar's label; j / k step through sections
  let at = -1, want = null, lockUntil = 0, ticking = false;
  const setCur = (i) => {
    if (i === at) return;
    if (at >= 0) links[at].removeAttribute("aria-current");
    at = i;
    links[i].setAttribute("aria-current", "true");
    cur.replaceChildren(el("span", "n", items[i].n), items[i].label);
    if (nav.scrollHeight > nav.clientHeight + 1 && getComputedStyle(nav).overflowY === "auto") {
      const a = links[i], top = a.offsetTop, bot = top + a.offsetHeight;
      if (top < nav.scrollTop + 48) nav.scrollTop = top - 48;
      else if (bot > nav.scrollTop + nav.clientHeight - 48) nav.scrollTop = bot - nav.clientHeight + 48;
    }
  };
  const spy = () => {
    ticking = false;
    nav.classList.toggle("past", !keys || keys.getBoundingClientRect().bottom < 56);
    if (performance.now() < lockUntil) return;
    const line = innerHeight * 0.3;
    let n = 0;
    secs.forEach((s, i) => { if (s.getBoundingClientRect().top <= line) n = i; });
    if (innerHeight + scrollY >= d.documentElement.scrollHeight - 4) n = want !== null && want >= n ? want : secs.length - 1;
    setCur(n);
  };
  const go = (i) => {
    i = Math.max(0, Math.min(secs.length - 1, i));
    want = i;
    lockUntil = performance.now() + 900;
    setTimeout(spy, 950);
    secs[i].scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    setCur(i);
  };
  const open = (on) => {
    nav.classList.toggle("open", on);
    bar.setAttribute("aria-expanded", String(on));
  };
  bar.addEventListener("click", () => open(!nav.classList.contains("open")));
  // In-page links are handled here: in the app the sheet is an srcdoc iframe, where "#id" would load the app's own page.
  d.addEventListener("click", (e) => {
    const a = e.target.closest?.('a[href^="#"]');
    const i = a ? items.findIndex((x) => "#" + x.id === a.getAttribute("href")) : -1;
    if (i >= 0) {
      e.preventDefault();
      open(false);
      go(i);
    } else if (!nav.contains(e.target)) open(false);
  });
  d.addEventListener("keydown", (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    const t = e.target;
    if (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
    const base = performance.now() < lockUntil ? want : at;
    if (e.key === "j") go(base + 1);
    else if (e.key === "k") go(base === at && secs[at].getBoundingClientRect().top < -40 ? at : base - 1);
    else if (e.key === "Escape") open(false);
    else return;
    e.preventDefault();
  });
  addEventListener("scroll", () => {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(spy);
    }
  }, { passive: true });
  for (const ev of ["wheel", "touchstart"]) addEventListener(ev, () => { want = null; lockUntil = 0; }, { passive: true });
  addEventListener("resize", spy);
  spy();
}

function sheetDoc(r, css) {
  let keys = "";
  try {
    keys = keyFigures(r);
  } catch (e) {
    console.error(e); // the figures are a convenience; the sheet goes out without them
  }
  const html = render(r, { css })
    .replace(/<link [^>]*href="https:\/\/fonts\.[^>]*>\n?/g, "")
    .replace("</header>", `</header>\n${keys}`);
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n`
    + `${html}<script>(${sheetScript})();</script>\n</html>`;
}
/* sheet-doc:end */

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
    pool = makePool(); // the workers load while the data downloads
    const cssP = cssText === null ? fetch(new URL("../sheet.css", import.meta.url)).then((r) => r.text()) : null;
    const raw = await loadAll(leagueId, (d, n) => progress(`Downloading league data and projections (${d} of ${n})…`, (0.15 * d) / n));
    const lg = new League(raw, rid);
    if (!lg.alive().includes(rid)) throw new Error("That team has already been chopped, so there's nothing left to simulate.");
    if (lg.week > lg.final_week) throw new Error("This league's season is over.");
    let stageName = "", frac = 0; // stages overlap, so progress counts work done over the whole run
    const r = await runAll(lg, pool, {
      onStage: (name) => { stageName = name; progress(`${stageName} (${Math.round(100 * frac)}%)…`, 0.15 + 0.85 * frac); },
      onProgress: (d, n) => { frac = Math.min(d / n, 1); progress(`${stageName} (${Math.round(100 * frac)}%)…`, 0.15 + 0.85 * frac); },
    });
    if (cssP) cssText = await cssP;
    showSheet(sheetDoc(r, cssText), r);
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

// The sheet gets the whole viewport and scrolls on its own, so its section rail and bar can stay pinned.
function showSheet(html, r) {
  const frame = $("sheet");
  frame.onload = () => {
    $("sheet-box").scrollIntoView({ behavior: "smooth", block: "start" });
    frame.contentWindow.focus(); // j / k work straight away
  };
  frame.srcdoc = html;
  const a = $("download");
  if (a.href.startsWith("blob:")) URL.revokeObjectURL(a.href);
  a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
  a.download = `chop-sheet-${r.season}-week-${r.week}.html`;
  $("sheet-name").textContent = a.download;
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
