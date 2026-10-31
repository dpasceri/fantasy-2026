// The cheat sheet: a port of guillotine/report.py's render(). Given the same results object it returns the same
// HTML string (test/report.test.js diffs the two). Python formatting quirks live in pyfmt.js.
import {
  esc, fmtComma, fmtE, fmtFixed, fmtG, fmtPct, maxBy, minBy, pyFloat, pyFloatSigned, pyRound, pySum,
  sortedBy, strftime, truthy,
} from "./pyfmt.js";

const JS = `
(function(){
  var tip=document.getElementById('tip');
  function show(e){var t=e.target.closest('[data-tip]'); if(!t){tip.classList.remove('on');return;}
    tip.textContent=t.getAttribute('data-tip'); tip.classList.add('on');
    var x=e.clientX+14, y=e.clientY+14, w=tip.offsetWidth, h=tip.offsetHeight;
    if(x+w>innerWidth-8) x=e.clientX-w-14; if(y+h>innerHeight-8) y=e.clientY-h-14;
    tip.style.left=x+'px'; tip.style.top=y+'px';}
  document.addEventListener('pointermove',show);
  document.addEventListener('pointerleave',function(){tip.classList.remove('on')});
  document.addEventListener('focusin',function(e){var t=e.target.closest('[data-tip]'); if(!t)return; var r=t.getBoundingClientRect();
    tip.textContent=t.getAttribute('data-tip'); tip.classList.add('on'); tip.style.left=(r.left)+'px'; tip.style.top=(r.bottom+6)+'px';});
  document.addEventListener('focusout',function(){tip.classList.remove('on')});
})();
`;

const f1 = (v) => fmtFixed(v, 1);
const f0 = (v) => fmtFixed(v, 0);

export const pct = (x, d = 1) => fmtFixed(100 * x, d) + "%";
export const pp = (x, d = 1) => fmtFixed(100 * x, d, { sign: "+" }) + " pts";

/** 'Bijan Robinson (RB, ATL)' -> 'B. Robinson' */
export function short(label) {
  const name = label.split(" (")[0];
  const parts = name.split(/\s+/).filter(Boolean);
  return parts.length < 2 ? name : `${[...parts[0]][0]}. ${parts.slice(1).join(" ")}`;
}

function riskChip(p) {
  const [cls, word] = p < 0.08 ? ["good", "low"] : p < 0.15 ? ["warn", "watch"] : ["crit", "high"];
  return `<span class="chip ${cls}"><i></i>${pct(p)} ${word}</span>`;
}

export function niceTicks(lo, hi, n = 4) {
  const span = hi - lo;
  const raw = span / n;
  // 10 ** int(f"{raw:e}".split("e")[1]): the exponent after rounding to 7 significant digits
  const e = Math.trunc(Number(fmtE(raw).split("e")[1]));
  const mag = e >= 0 ? 10 ** e : 1 / 10 ** -e;
  const step = minBy([1, 2, 2.5, 5, 10].map((s) => s * mag), (s) => Math.abs(s - raw));
  let t = lo > 0 ? (Math.trunc(lo / step) + (lo % step ? 1 : 0)) * step : 0;
  const out = [];
  while (t <= hi + 1e-9) {
    out.push(pyRound(t, 6));
    t += step;
  }
  return out;
}

// ------------------------------------------------------------------ charts (inline SVG)
/** rows: [label, quantiles, color var, tip] -> horizontal p10..p90 ranges, p25..p75 thick, median tick. */
function rangeChart(rows, lo, hi) {
  const [W, H, L, R] = [620, 40 + 34 * rows.length, 128, 16];
  const x = (v) => L + (v - lo) / (hi - lo) * (W - L - R);
  const out = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Score ranges">`];
  for (const t of niceTicks(lo, hi, 6)) {
    out.push(`<line x1="${f1(x(t))}" x2="${f1(x(t))}" y1="8" y2="${H - 22}" stroke="var(--grid)" stroke-width="1"/>`
      + `<text x="${f1(x(t))}" y="${H - 8}" text-anchor="middle">${fmtG(t)}</text>`);
  }
  rows.forEach(([label, q, color, tip], i) => {
    const y = 24 + 34 * i;
    out.push(`<g data-tip="${esc(tip)}" tabindex="0">`
      + `<rect x="${L - 120}" y="${y - 14}" width="${W - L + 120}" height="28" fill="transparent"/>`
      + `<text x="${L - 10}" y="${y + 4}" text-anchor="end" class="lbl">${esc(label)}</text>`
      + `<line x1="${f1(x(q["0.1"]))}" x2="${f1(x(q["0.9"]))}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="2" stroke-linecap="round"/>`
      + `<rect x="${f1(x(q["0.25"]))}" y="${y - 6}" width="${f1(x(q["0.75"]) - x(q["0.25"]))}" height="12" rx="3" fill="${color}" opacity=".9"/>`
      + `<line x1="${f1(x(q["0.5"]))}" x2="${f1(x(q["0.5"]))}" y1="${y - 10}" y2="${y + 10}" stroke="var(--ink)" stroke-width="2"/>`
      + `<text x="${f1(x(q["0.5"]))}" y="${y - 13}" text-anchor="middle" class="lbl-strong">${fmtG(q["0.5"])}</text></g>`);
  });
  out.push("</svg>");
  return out.join("");
}

/** series: [name, color var, [[x, y, tip]]]. 2px lines, 8px dots with a surface ring, hover tips. */
function lineChart(series, xlab, ylab, xDom, yDom, {
  W = 560, H = 250, marks = null, fmtY = (v) => fmtFixed(100 * v, 0) + "%", fmtX = (v) => "$" + fmtG(v), xticks = 5,
} = {}) {
  const [L, R, T, B] = [46, 14, 14, 34];
  const x = (v) => L + (v - xDom[0]) / (xDom[1] - xDom[0]) * (W - L - R);
  const y = (v) => T + (1 - (v - yDom[0]) / (yDom[1] - yDom[0])) * (H - T - B);
  const o = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylab)} by ${esc(xlab)}">`];
  for (const t of niceTicks(yDom[0], yDom[1], 4)) {
    o.push(`<line x1="${L}" x2="${W - R}" y1="${f1(y(t))}" y2="${f1(y(t))}" stroke="var(--grid)"/>`
      + `<text x="${L - 6}" y="${f1(y(t) + 4)}" text-anchor="end">${fmtY(t)}</text>`);
  }
  for (const t of niceTicks(xDom[0], xDom[1], xticks)) {
    o.push(`<text x="${f1(x(t))}" y="${H - 16}" text-anchor="middle">${fmtX(t)}</text>`);
  }
  o.push(`<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="var(--rule)"/>`);
  o.push(`<text x="${f0((L + W - R) / 2)}" y="${H - 2}" text-anchor="middle" class="lbl">${esc(xlab)}</text>`);
  for (const m of marks || []) {
    if (m[0] === "h") {
      o.push(`<line x1="${L}" x2="${W - R}" y1="${f1(y(m[1]))}" y2="${f1(y(m[1]))}" stroke="var(--ink-2)" stroke-width="1"/>`
        + `<text x="${W - R}" y="${f1(y(m[1]) - 4)}" text-anchor="end" class="lbl">${esc(m[2])}</text>`);
    }
  }
  for (const [, color, pts] of series) {
    const d = pts.map(([a, b], i) => `${i === 0 ? "M" : "L"}${f1(x(a))},${f1(y(b))}`).join(" ");
    o.push(`<path d="${d}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`);
    for (const [a, b, tip] of pts) {
      o.push(`<g data-tip="${esc(tip)}" tabindex="0"><circle cx="${f1(x(a))}" cy="${f1(y(b))}" r="11" fill="transparent"/>`
        + `<circle cx="${f1(x(a))}" cy="${f1(y(b))}" r="4" fill="${color}" stroke="var(--sheet)" stroke-width="2"/></g>`);
    }
  }
  for (const m of marks || []) {
    if (m[0] === "pt") {
      o.push(`<circle cx="${f1(x(m[1]))}" cy="${f1(y(m[2]))}" r="6.5" fill="none" stroke="var(--ink)" stroke-width="2"/>`
        + `<text x="${f1(x(m[1]))}" y="${f1(y(m[2]) - 11)}" text-anchor="middle" class="lbl-strong">${esc(m[3])}</text>`);
    }
  }
  o.push("</svg>");
  return o.join("");
}

/** Survival sparkline. (report.py builds standings rows with it but never places them on the page.) */
export function spark(vals, me, W = 120, H = 26) {
  const x = (i) => 2 + i / Math.max(vals.length - 1, 1) * (W - 4);
  const y = (v) => 2 + (1 - v) * (H - 4);
  const d = vals.map((v, i) => `${i === 0 ? "M" : "L"}${f1(x(i))},${f1(y(v))}`).join(" ");
  const c = truthy(me) ? "var(--you)" : "var(--muted)";
  return `<svg viewBox="0 0 ${W} ${H}" style="width:${W}px" aria-hidden="true"><line x1="2" x2="${W - 2}" y1="${H - 2}" y2="${H - 2}" stroke="var(--grid)"/>`
    + `<path d="${d}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"/></svg>`;
}

function heatCell(v, lo, hi) {
  const steps = ["--h5", "--h4", "--h3", "--h2", "--h1"];
  const k = hi <= lo ? 0 : Math.min(4, Math.trunc((v - lo) / (hi - lo + 1e-12) * 5));
  const ink = k < 2 ? "var(--ink)" : "var(--sheet)";
  return `background:var(${steps[k]});color:${ink}`;
}

const s_ = (n) => (n === 1 ? "" : "s");
// The pieces of report.py that use `" class=\'me\'"` (single quotes) vs `' class="me"'` (double quotes).
const meSq = (on) => (truthy(on) ? " class='me'" : "");
const arrEq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// ------------------------------------------------------------------ page
/** The cheat sheet HTML. Pass `css` (report.py's CSS) to inline it in a <style> as report.py does; without it the
 *  page relies on sheet.css and the <style> line is left out. */
export function render(r, { css } = {}) {
  const me = r.me;
  const st = r.standings;
  const mine = st.find((s) => truthy(s.me));
  const byChop = sortedBy(st, (s) => s.p_chop_now);
  const safeRank = byChop.map((s) => s.owner).indexOf(me) + 1;
  const wk = r.week;
  const lab = r.labels;
  const D = r.dynamic;
  const bud = sortedBy(D.budgets, (d) => d.budget);
  const budS = sortedBy(D.budgets_stress, (d) => d.budget);
  const topB = maxBy(bud, (d) => d.budget);
  const best = D.best;
  const used = "used" in D ? D.used : best;
  const worstAggr = minBy(D.sweep.filter((d) => d.m_early >= 1.5), (d) => d.p_win) ?? null;
  const lus = r.lineups;
  const cur = lus.find((l) => truthy(l.is_current)) ?? null;
  const bestLu = lus[0];
  const bestNoFa = lus.find((l) => !truthy(l.uses_fa)) ?? null;
  const fa = r.free_agents.filter((f) => f.d_win > 0.002);
  const scen = r.scenarios;
  const sims = r.sims;

  // ---- power + league money
  const pw = r.power;
  const mePw = pw.find((x) => truthy(x.me));
  const meRank = pw.indexOf(mePw) + 1;
  const pace = r.faab_pace;
  const hist = pace.history;
  const fc = pace.forecast;
  const B0 = Math.floor(pace.start_total / (pace.n_auctions + 2)); // per-team starting budget
  const usd = (x) => pyRound((x * B0) / 200); // a $200-league dollar amount, in this league's FAAB
  const last = hist.length ? hist[hist.length - 1] : null;
  const burnedTot = pySum(hist.map((h) => h.burned));
  const diedFull = hist.filter((h) => h.burned >= B0).length;
  const spentHist = hist.map((h) => h.spent);
  const nLeftAuctions = pace.n_auctions - hist.length;
  const iNext = truthy(r.pending) ? 0 : 1; // forecast index of the next waiver run
  const nxt = fc.top_price[iNext];
  const later = fc.top_price[Math.min(iNext + 3, fc.top_price.length - 1)];

  // ---- the calls
  const calls = [];
  const others = pw.filter((x) => !truthy(x.me)).slice(0, 2);
  let lead;
  if (meRank === 1) {
    lead = `<b>You.</b> ${pct(mePw.p_win, 0)} to win the league if everyone, you included, keeps bidding the way they have; `
      + `next are ${esc(others[0].owner)} (${pct(others[0].p_win, 0)}) and ${esc(others[1].owner)} (${pct(others[1].p_win, 0)}).`;
  } else {
    lead = `<b>${esc(pw[0].owner)}</b> (${pct(pw[0].p_win, 0)} to win if everyone keeps bidding the way they have). `
      + `You're #${meRank} at ${pct(mePw.p_win, 0)}.`;
  }
  calls.push(["Best shape", lead + ` Bid the recommended way and you're at <b>${pct(mePw.p_win_policy, 0)}</b>.`]);
  calls.push(["This week", `Chop risk <b>${pct(mine.p_chop_now)}</b>, #${safeRank} safest of ${st.length}. `
    + `The score to beat (lowest rival) has a median of <b>${fmtG(r.chop_line["0.5"])}</b> `
    + `(9 weeks in 10 under ${fmtG(r.chop_line["0.9"])}). Your median is <b>${fmtG(r.my_score["0.5"])}</b>.`]);
  if (cur && bestNoFa) {
    let txt;
    if (arrEq(cur.players, bestNoFa.players) || Math.abs(cur.p_chop - bestNoFa.p_chop) < 0.002) {
      txt = `Your current lineup is already the safest of the ${r.n_lineups} legal lineups from your roster (${pct(cur.p_chop)} chop).`;
    } else {
      const ins = bestNoFa.players.filter((p) => !cur.players.includes(p)).map((p) => short(lab[p]));
      const outs = cur.players.filter((p) => !bestNoFa.players.includes(p)).map((p) => short(lab[p]));
      txt = `Start <b>${ins.join(", ")}</b> over ${outs.join(", ")}: chop risk ${pct(cur.p_chop)} → <b>${pct(bestNoFa.p_chop)}</b>.`;
    }
    if (truthy(bestLu.uses_fa) && bestLu.p_chop < bestNoFa.p_chop - 0.002) {
      const ins = bestLu.players.filter((p) => !bestNoFa.players.includes(p)).map((p) => short(lab[p]));
      const outs = bestNoFa.players.filter((p) => !bestLu.players.includes(p)).map((p) => short(lab[p]));
      txt += ` Adding <b>${ins.join(", ")}</b> and starting him over ${outs.join(", ")} gets it to <b>${pct(bestLu.p_chop)}</b>.`;
    }
    const day = new Map(r.roster.map((p) => [p.pid, p.day]));
    const lockedNow = (truthy(r.live) ? r.live : {}).locked ?? [];
    const early = bestLu.players.filter((p) => day.get(p) === "Thu" && !lockedNow.includes(p)).map((p) => short(lab[p]));
    if (early.length) {
      txt += ` ${early.join(", ")} play${early.length === 1 ? "s" : ""} Thursday and lock${early.length === 1 ? "s" : ""} at kickoff.`;
    }
    calls.push(["Lineup", txt]);
  }
  if (fa.length) {
    const fa0 = fa[0];
    let extra = truthy(r.open_spots)
      ? ` You have ${r.open_spots} open roster spot${s_(r.open_spots)}, so no drop needed.`
      : " You'll need to drop your weakest bench player.";
    const have = r.roster.filter((p) => p.pos === fa0.pos && !truthy(p.ir));
    if (fa0.pos === "QB" && have.length) {
      extra += ` He's injury cover for ${esc(short(have[0].player))}. The sim keeps rosters frozen and can't grab a replacement after an injury`
        + " the way you can, so read that number as an upper bound.";
    }
    calls.push(["Free pickup", `Add <b>${esc(fa0.player)}</b> now. He costs $0 (waivers have cleared) and adds <b>${pp(fa0.d_win)}</b> to your chance to win the league.${extra}`]);
  }
  const when = used.m_early === used.m_late ? "all season" : `before week 12, ${fmtG(used.m_late)}× after`;
  let money = `Bid about <b>${fmtG(used.m_early)}×</b> what a typical manager in this league would bid for the same lineup gain, ${when}.`;
  const dflt = D.default || {};
  const d0 = D.sweep.find((d) => d.m_early === dflt.m_early && d.m_late === dflt.m_late);
  if (d0 && (d0.m_early !== used.m_early || d0.m_late !== used.m_late))
    money += ` That's not the usual ${fmtG(d0.m_early)}×: in this league's sims it clearly wins more often `
      + `(${pct(used.p_win)} vs ${pct(d0.p_win)} to win the league).`;
  if (worstAggr && used.p_win - worstAggr.p_win >= 0.005) {
    money += ` Bidding 1.5× early drops your win chance by up to ${f0(100 * (used.p_win - worstAggr.p_win))} pts.`;
  }
  const tb = topB.budget;
  const marks = sortedBy(bud, (d) => -d.budget).filter((d) => [tb, tb - usd(40), tb - usd(80), tb - usd(120)].includes(d.budget));
  const segs = marks.slice(1).map((b, i) => {
    const a = marks[i];
    return `$${a.budget}→$${b.budget} left costs <b>${f1(100 * (a.p_win - b.p_win))} pts</b>`;
  });
  if (segs.length) money += " Spending gets more expensive the lower you go: " + segs.join("; ") + " of win chance.";
  calls.push(["Money", money]);
  if (last) {
    const gap = last.linear_left - last.left;
    calls.push(["League pace",
      `${gap > 0 ? "Ahead of" : "Behind"} the expected burn: the ${last.alive} alive teams hold <b>$${fmtComma(last.left)}</b>, `
      + `$${fmtFixed(Math.abs(gap), 0, { comma: true })} (${fmtPct(Math.abs(gap) / last.linear_left, 0)}) ${gap > 0 ? "below" : "above"} an even-spending pace. `
      + `Chopped teams took <b>$${fmtComma(burnedTot)}</b> out unspent (${diedFull} of ${hist.length} with their full budget). `
      + `$${fmtComma(pace.left_now)} over ${nLeftAuctions} auctions is $${f0(pace.left_now / Math.max(nLeftAuctions, 1))} each vs `
      + `$${f0(pySum(spentHist) / spentHist.length)} spent per auction so far, so prices fall: forecast top price `
      + `<b>$${f0(nxt)}</b> at the next run, about $${f0(later)} three runs later.`]);
  }
  for (const sc of scen.slice(0, 3)) {
    const rows = sc.players.filter((p) => truthy(p.bid));
    if (!rows.length) continue;
    const p0 = maxBy(rows, (p) => p.bid.best_p_win - p.bid.p_win_if_zero);
    const b = p0.bid;
    const lift = b.best_p_win - b.p_win_if_zero;
    const head = truthy(sc.known) ? `${sc.owner} was chopped` : `If ${sc.owner} is chopped (${pct(sc.p_chop, 0)})`;
    if (b.best_bid > 0 && lift >= 0.005) {
      calls.push(["Auction", `${head}: bid <b>$${b.best_bid}</b> on <b>${esc(short(p0.player))}</b> and walk away above $${b.walk_away} `
        + `(+${f1(100 * lift)} pts vs skipping). Rivals' top bid on him: median $${f0(p0.rival_top["0.5"])}, 90th pct $${f0(p0.rival_top["0.9"])}.`]);
    } else {
      const big = maxBy(sc.players, (p) => p.d_win_static);
      if (r.faab_me < usd(10))
        calls.push(["Auction", `${head}: with $${r.faab_me} FAAB you can't outbid anyone, so put in $0 claims and hope players clear. `
          + `${esc(short(big.player))} would add ${pp(big.d_win_static)} if he got through.`]);
      else if (big.d_win_static >= 0.01)
        calls.push(["Auction", `${head}: rivals price these players out. ${esc(short(big.player))} would add ${pp(big.d_win_static)} `
          + `if free, but no bid beats skipping by even half a point once you pay what it takes. $0–${usd(10)} flyers only.`]);
      else
        calls.push(["Auction", `${head}: nobody there is worth real money. $0–${usd(10)} flyers only: no bid on `
          + `${esc(short(big.player))} beats skipping by even half a point, because your lineup barely improves.`]);
    }
  }
  const callHtml = calls.map(([k, v]) => `<li><div><span class="k">${esc(k)}</span><span class="v">${v}</span></div></li>`).join("");

  // ---- week box
  const [qMe, qLine] = [r.my_score, r.chop_line];
  const lo = Math.max(0, Math.min(qLine["0.1"], qMe["0.1"]) - 10);
  const hi = Math.max(qLine["0.9"], qMe["0.9"]) + 10;
  const rc = rangeChart([
    [`${me} (you)`, qMe, "var(--you)", `Your week ${wk} score\n10th pct ${fmtG(qMe["0.1"])} · median ${fmtG(qMe["0.5"])} · 90th ${fmtG(qMe["0.9"])}`],
    ["Lowest rival", qLine, "var(--rival)", `Lowest rival score = the chop line\n10th pct ${fmtG(qLine["0.1"])} · median ${fmtG(qLine["0.5"])} · 90th ${fmtG(qLine["0.9"])}`],
  ], lo, hi);

  const lv = r.live;
  const locked = new Set(truthy(lv) ? lv.locked : []);
  const luRow = (l, i) => {
    const tags = [];
    if (truthy(l.is_current)) tags.push('<span class="tag you">current</span>');
    if (truthy(l.is_proj_opt)) tags.push('<span class="tag">max points</span>');
    if (truthy(l.uses_fa)) tags.push(`<span class="tag">adds ${esc(short(lab[l.uses_fa[0]]))}</span>`);
    const names = l.players.map((p) => (locked.has(p) ? `<b>${esc(short(lab[p]))}</b>` : esc(short(lab[p])))).join(", ");
    const cls = truthy(l.is_current) ? ' class="me"' : "";
    return `<tr${cls}><td class="num">${i}</td><td class="num r">${pct(l.p_chop, 2)}</td>`
      + `<td class="num r">${f1(l.mean)} ± ${f0(l.sd)}</td><td style="min-width:260px">${names}</td><td>${tags.join(" ")}</td></tr>`;
  };
  let liveBox = "";
  if (truthy(lv)) {
    const n = lv.n;
    const asOf = strftime(lv.as_of, "%a %-I:%M %p");
    const prog = truthy(lv.in_progress) ? ` (${esc(lv.in_progress.join(", "))})` : "";
    const liveRows = lv.teams.map((t) => `<tr${meSq(t.me)}><td>${esc(t.owner)}</td><td class="num r">${f1(t.pts)}</td>`
      + `<td class="num r">${t.left}</td><td class="num r">${pct(t.p_chop)}</td></tr>`).join("");
    liveBox = `<section class="box"><h2>Week ${wk} so far <small>scores as of ${asOf}</small></h2><div class="inner">`
      + `<p class="note">${n.post} of ${pySum(Object.values(n))} games final, ${n.in} in progress${prog}. Every sim below starts from these scores: `
      + "finished players keep their points, players in a live game add the unplayed share of a full-game draw, and lineups are locked "
      + "the way Sleeper locks them (bold in the lineup table).</p>"
      + '<div class="tbl"><table><thead><tr><th>Team</th><th class="r">Pts so far</th><th class="r">Starters left</th>'
      + `<th class="r">Chop risk</th></tr></thead><tbody>${liveRows}</tbody></table></div></div></section>`;
  }
  const luHtml = lus.map((l, i) => luRow(l, i + 1 <= 10 ? i + 1 : "—")).join("");

  const rosterRows = sortedBy(r.roster, (p) => -pySum(p.proj)).map((p) =>
    `<tr><td>${esc(p.player)}${truthy(p.ir) ? " <span class='tag'>IR</span>" : ""}</td>`
    + `<td>${esc(p.inj || "")}</td><td>${esc(p.day || "bye")}</td>`
    + p.proj.map((v) => `<td class="num r">${fmtG(v)}</td>`).join("") + "</tr>").join("");

  const faRows = r.free_agents.slice(0, 10).map((f) =>
    `<tr><td>${esc(f.player)}</td><td class="num r">${fmtG(f.proj_now)}</td><td class="num r">${fmtG(f.ros_avg)}</td>`
    + `<td class="num r">${pp(f.d_win)}</td><td class="num r">${pp(f.d_survive_now, 2)}</td><td>${esc(f.inj || "")}</td></tr>`).join("");

  // ---- auction board
  const auctionNote = '<p class="note"><b>Gain to you</b> is lineup points per week over the next 4 weeks. <b>Win chance if free</b> is the static boost '
    + "with no future auctions. <b>Bid</b> is the bid that maximized P(win league) in full season-with-auctions sims; rivals bid from "
    + "their fitted models and your later auctions are played out too. <b>Walk-away</b> is the highest bid that still leaves you better "
    + "off than skipping him. Hover a player for who else wants him.</p>";
  const scenHtml = (sc, first) => {
    const rows = sc.players.slice(0, 7);
    const trs = rows.map((p) => {
      const b = truthy(p.bid) ? p.bid : null;
      const rec = b ? `<b>$${b.best_bid}</b> <span class="num" style="color:var(--muted)">(≤ $${b.walk_away})</span>` : "—";
      const gainLine = b ? `+${f1(100 * (b.best_p_win - b.p_win_if_zero))} pts` : "";
      const rg = p.rival_gains.slice(0, 3).map((g) => `${g.owner} ${fmtG(g.gain)}/wk ($${g.faab})`).join("; ");
      return `<tr><td data-tip="${esc("Top rival gains: " + rg)}" tabindex="0">${esc(p.player)}${truthy(p.inj) ? " · " + esc(p.inj) : ""}</td>`
        + `<td class="num r">${fmtG(p.gain_me)}</td><td class="num r">${pp(p.d_win_static)}</td>`
        + `<td class="num r">$${f0(p.rival_top["0.5"])} / $${f0(p.rival_top["0.9"])}</td>`
        + `<td class="num r">${pct(p.p_no_rival_bid, 0)}</td><td class="num r">${rec}</td><td class="num r">${gainLine}</td></tr>`;
    });
    const minis = [];
    for (const p of rows) {
      if (!truthy(p.bid)) continue;
      const b = p.bid;
      const pts = b.grid.map((g) => [g.bid, g.p_win, `Bid $${g.bid} on ${short(p.player)}\nP(win league) ${pct(g.p_win)} ± ${pct(g.se)}`]);
      const ser = [["You", "var(--you)", pts]];
      const sb = truthy(p.stress_bid) ? p.stress_bid : null;
      if (sb) {
        ser.push(["Rivals bid harder", "var(--rival)", sb.grid.map((g) => [g.bid, g.p_win, `Rivals bid harder · bid $${g.bid}\nP(win league) ${pct(g.p_win)}`])]);
      }
      const ys = [...b.grid.map((g) => g.p_win), ...(sb ? sb.grid.map((g) => g.p_win) : [])];
      const [ylo, yhi] = [Math.max(0, Math.min(...ys) - 0.02), Math.min(1, Math.max(...ys) + 0.03)];
      const ch = lineChart(ser, "your bid", "P(win league)", [0, Math.max(...b.grid.map((g) => g.bid))], [ylo, yhi], {
        W: 240, H: 170, xticks: 3,
        marks: [["h", b.p_win_if_zero, "skip him"], ["pt", b.best_bid, b.best_p_win, `$${b.best_bid}`]],
      });
      const leg = sb ? '<div class="legend"><span><i style="background:var(--you)"></i>rivals as fitted</span>'
        + '<span><i style="background:var(--rival)"></i>rivals bid harder</span></div>' : "";
      minis.push(`<div class="mini"><h4>${esc(p.player)}</h4><p class="verdict">Best bid <b>$${b.best_bid}</b> `
        + `(anything $${b.good_range[0]}–$${b.good_range[1]} is within noise) · walk away above <b>$${b.walk_away}</b>`
        + `${sb ? " · if rivals bid harder: $" + sb.best_bid : ""}</p>${leg}${ch}</div>`);
    }
    const head = truthy(sc.known) ? `Auction: ${esc(sc.owner)} was chopped` : `Auction: if ${esc(sc.owner)} is chopped`;
    const sub = truthy(sc.known) ? `on waivers now · ${fmtComma(sims.grid_paths)} season paths per bid`
      : `${pct(sc.p_chop)} likely this week · ${fmtComma(sims.grid_paths)} paths per bid`;
    return `<section class="box"><h2>${head} <small>${sub}</small></h2><div class="inner">${first ? auctionNote : ""}`
      + '<div class="tbl"><table><thead><tr><th>Player</th><th class="r">Gain to you /wk</th>'
      + '<th class="r">Win chance if free</th><th class="r">Top rival bid med / p90</th><th class="r">No rival bids</th>'
      + `<th class="r">Bid (walk-away)</th><th class="r">vs skipping</th></tr></thead><tbody>${trs.join("")}</tbody></table></div>`
      + (minis.length ? `<div class="minis">${minis.join("")}</div>` : "") + "</div></section>";
  };
  const scenBlocks = scen.slice(0, 5).map((sc, i) => scenHtml(sc, i === 0)).join("\n");

  // ---- money charts
  const bmax = Math.max(...bud.map((d) => d.budget));
  const pts1 = bud.map((d) => [d.budget, d.p_win, `$${d.budget} left · rivals as fitted\nP(win league) ${pct(d.p_win)} ± ${pct(d.se)}`]);
  const pts2 = budS.map((d) => [d.budget, d.p_win, `$${d.budget} left · rivals bid harder\nP(win league) ${pct(d.p_win)}`]);
  const ysAll = [...bud, ...budS].map((d) => d.p_win);
  const moneyChart = lineChart([["rivals as fitted", "var(--you)", pts1], ["rivals bid harder", "var(--rival)", pts2]],
    "FAAB you start the auctions with", "P(win league)", [0, bmax], [0, Math.min(1, Math.max(...ysAll) + 0.05)], { W: 560, H: 260 });
  const ms = [...new Set(D.sweep.map((d) => d.m_early))].sort((a, b) => a - b);
  const ml = [...new Set(D.sweep.map((d) => d.m_late))].sort((a, b) => a - b);
  const vals = D.sweep.map((d) => d.p_win);
  const [loV, hiV] = [Math.min(...vals), Math.max(...vals)];
  const heat = ["<table class='heat'><thead><tr><th>Weeks before 12 ↓ / after →</th>" + ml.map((b) => `<th class='r'>${fmtG(b)}×</th>`).join("") + "</tr></thead><tbody>"];
  for (const a of ms) {
    heat.push(`<tr><th>${fmtG(a)}×</th>`);
    for (const b of ml) {
      const d = D.sweep.find((d) => d.m_early === a && d.m_late === b);
      const ring = a === used.m_early && b === used.m_late ? ";outline:2px solid var(--you);outline-offset:-2px" : "";
      const tip = `Bid ${fmtG(a)}× typical before week 12, ${fmtG(b)}× after\nP(win league) ${pct(d.p_win)} ± ${pct(d.se)}\nAvg spent $${f0(d.avg_spent)}`;
      heat.push(`<td style="${heatCell(d.p_win, loV, hiV)}${ring}" data-tip="${esc(tip)}" tabindex="0">${pct(d.p_win, 1)}</td>`);
    }
    heat.push("</tr>");
  }
  heat.push("</tbody></table>");
  const topTxt = best === used || (best.m_early === used.m_early && best.m_late === used.m_late)
    ? "It is also the top cell this run."
    : `The top cell this run (${fmtG(best.m_early)}× / ${fmtG(best.m_late)}×, ${pct(best.p_win)}) is within noise of it: `
      + "on fresh simulations these cells reshuffle, and 0.75× all season held up best.";

  // ---- rivals
  const bmr = r.bid_model;
  const rivRows = sortedBy(bmr.managers, (m) => -m.faab).map((m) =>
    `<tr${meSq(m.owner === me)}><td>${esc(m.owner)}</td><td class="num r">$${m.faab}</td>`
    + `<td class="num r">${m.n_bids} / ${m.n_chances}</td><td class="num r">${fmtFixed(m.bids_often, 2, { sign: "+" })}</td>`
    + `<td class="num r">${fmtFixed(m.bids_hard, 2, { sign: "+" })}</td></tr>`).join("");
  const pb = r.price_backtest;
  const pbIn = pb.filter((x) => x.p10 <= x.actual && x.actual <= x.p90).length;
  const pbRows = sortedBy(pb, (x) => -x.actual).slice(0, 8).map((x) =>
    `<tr><td class="num">${x.leg}</td><td>${esc(x.player)}</td><td class="num r">$${x.actual}</td>`
    + `<td class="num r">$${f0(x.median)}</td><td class="num r">$${f0(x.p10)}–$${f0(x.p90)}</td></tr>`).join("");

  // ---- math / model card
  const mr = r.model_report;
  const cop = mr.copula;
  const bt = r.backtest;
  let btTxt = "";
  if (truthy(bt)) {
    const W = bt.weeks;
    const hits = W.filter((w) => w.model_rank_of_chopped === 1).length;
    const avgP = pySum(W.map((w) => w.p_model_actual_starters)) / W.length;
    const avgU = pySum(W.map((w) => w.p_uniform)) / W.length;
    btTxt = `Replaying weeks 1–${W.length} with the rosters and starters each team actually used, the model gave the team that really got `
      + `chopped ${pct(avgP)} on average (a uniform guess gives ${pct(avgU)}) and ranked them most likely ${hits} of ${W.length} weeks. `
      + `Log-loss ${fmtFixed(bt.logloss_model, 3)} vs ${fmtFixed(bt.logloss_uniform, 3)} for uniform. Chops are noisy; this is an edge, not a crystal ball.`;
  }
  const [sz, pa] = [bmr.size, bmr.participation];
  let fitSrc, r2Txt;
  if (truthy(bmr.prior)) {
    fitSrc = `This league has only ${bmr.n_bids} bids so far, too few to fit, so these are a prior fit on another `
      + "league's bids, with budgets scaled to this league's FAAB and no per-manager effects.";
    r2Txt = "";
  } else {
    fitSrc = "Fit on every bid this league has placed, winners and losers, with each manager's budget at the time.";
    r2Txt = ` R² = ${pyFloat(sz.r2)}.`;
  }
  const gen = strftime(r.generated_at, "%a %b %-d, %-I:%M %p");
  const fet = strftime(r.fetched_at, "%a %b %-d, %-I:%M %p");
  const pend = truthy(r.pending) ? ` · auction pending for ${esc(r.pending.owner)}` : "";
  const weeks = r.weeks;
  const wRange = `${weeks[0]}–${weeks[weeks.length - 1]}`;

  // ---- power rankings box
  const pwRows = pw.map((x, i) =>
    `<tr${meSq(x.me)}><td class="num">${i + 1}</td><td>${esc(x.owner)}`
    + `${truthy(x.me) ? `<br><small class=sub>${pct(x.p_win_policy)} with the recommended bids</small>` : ""}</td>`
    + `<td class="num r"><b>${pct(x.p_win)}</b></td>`
    + `<td class="num r">${pct(x.p_win_frozen)}</td><td class="num r">$${x.faab}</td>`
    + `<td class="num r">${fmtG(x.ros_proj)}</td>`
    + `<td class="num r" data-tip="${esc(`Weakest projected lineup: week ${x.worst_week}, ${fmtG(x.worst_proj)} pts (byes/injuries)`)}" tabindex="0">wk ${x.worst_week} · ${f0(x.worst_proj)}</td>`
    + `<td>${riskChip(x.p_chop_now)}</td><td class="num r">${f1(x.e_weeks)}</td></tr>`).join("");
  const broke = pw.filter((x) => x.faab <= usd(15)).map((x) => x.owner);
  const brokeTxt = broke.length ? broke.map(esc).join(", ") + ` have $${usd(15)} or less` : "nobody is broke yet";
  const powerBox = `<section class="box"><h2>Power rankings <small>who wins with the auctions played out · ${fmtComma(r.sims.dynamic_paths)}+ season paths</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>#</th><th>Team</th><th class="r">Win league</th><th class="r">Frozen rosters</th><th class="r">FAAB</th><th class="r">Lineup proj /wk</th><th class="r">Weakest week</th><th>Chop wk ${wk}</th><th class="r">Weeks survived of ${weeks.length}</th></tr></thead><tbody>${pwRows}</tbody></table></div>
  <p class="note"><b>Win league</b> plays every remaining week and auction, with every manager, you included, bidding the way they have all season. That's the fair ranking; under your name is your chance if you bid the recommended way instead. <b>Frozen rosters</b> is the same race with no more auctions: it rewards teams that already spent (${brokeTxt}) and ignores what money will buy. Lineup proj is the average projected best lineup over weeks ${wRange}; the weakest week is the bye-week hole most likely to get them chopped.</p>
</div></section>`;

  // ---- FAAB pace box
  const nTeams = pace.n_auctions + 2;
  const act = [[0, pace.start_total, `Season start\n${nTeams} teams · $${fmtComma(pace.start_total)}`]];
  for (const h of hist) {
    act.push([h.leg, h.left, `After week ${h.leg}\n${h.alive} alive hold $${fmtComma(h.left)}\nspent $${h.spent} · burned $${h.burned} · top price $${h.top_price}`]);
  }
  const lin = [];
  for (let kk = 0; kk <= pace.n_auctions; kk++) {
    const v = (nTeams - kk) * B0 * Math.max(0.0, 1 - kk / pace.n_auctions);
    lin.push([kk, v, `Even pace after ${kk} auctions\n$${fmtFixed(v, 0, { comma: true })}`]);
  }
  const fpts = [];
  if (last) {
    let curLeft = pace.left_now;
    fpts.push([last.leg, curLeft, `Now\n$${fmtComma(curLeft)}`]);
    if (truthy(r.pending)) curLeft -= fc.league_spend[0];
    for (let i = 1; i < weeks.length; i++) {
      curLeft -= fc.burned[i] + fc.league_spend[i];
      const left = curLeft < 0 ? 0 : curLeft; // max(cur_left, 0) keeps cur_left on a tie, like Python
      fpts.push([weeks[i - 1], left,
        `Forecast after week ${weeks[i - 1]}\n$${fmtFixed(left, 0, { comma: true })} left\nspend $${f0(fc.league_spend[i])} · burned $${f0(fc.burned[i])} · top $${f0(fc.top_price[i])}`]);
    }
  }
  const paceChart = lineChart([["actual", "var(--s3)", act], ["even-spending pace", "var(--muted)", lin], ["forecast", "var(--s4)", fpts]],
    "week", "FAAB held by alive teams", [0, r.final_week], [0, pace.start_total], {
      W: 560, H: 260, fmtY: (v) => (v >= 1000 ? `$${fmtG(v / 1000)}K` : `$${fmtG(v)}`), fmtX: (v) => fmtG(v),
    });
  const paceRows = hist.map((h) =>
    `<tr><td class="num">${h.leg}</td><td class="num r">${h.alive}</td><td class="num r">$${h.spent}</td>`
    + `<td class="num r">$${h.burned}</td><td class="num r">$${h.top_price}</td><td class="num r">${h.buyers}</td>`
    + `<td class="num r">$${fmtComma(h.left)}</td><td class="num r">$${fmtFixed(h.linear_left, 0, { comma: true })}</td></tr>`).join("");
  const tn = pace.trade_net_me ?? 0;
  const traded = tn > 0 ? ` (you received $${tn})` : tn < 0 ? ` (you sent $${-tn})` : "";
  const paceBox = `<section class="box"><h2>League FAAB pace <small>$${fmtComma(pace.start_total)} at the start</small></h2><div class="inner">
  <div class="legend"><span><i style="background:var(--s3)"></i>actual</span><span><i style="background:var(--muted)"></i>even-spending pace</span><span><i style="background:var(--s4)"></i>model forecast</span></div>
  ${paceChart}
  <div class="tbl"><table><thead><tr><th>Wk</th><th class="r">Alive</th><th class="r">Spent</th><th class="r">Burned</th><th class="r">Top price</th><th class="r">Buyers</th><th class="r">Left</th><th class="r">Even pace</th></tr></thead><tbody>${paceRows}</tbody></table></div>
  <p class="note"><b>Burned</b> is the unspent FAAB a team takes with it when it's chopped. <b>Even pace</b> means every team spends its $${B0} evenly over the season's ${pace.n_auctions} auctions. FAAB sent in trades is counted${traded}. The forecast comes from the power-ranking sims.</p>
</div></section>`;

  const L = cop.loadings, sameC = cop.same_team_corr, oppC = cop.opp_team_corr;
  const seAt5 = fmtFixed(100 * (0.05 * 0.95 / Math.max(sims.season, 1)) ** 0.5, 2);
  // Timings are Python floats, so they print as "445.0", not "445".
  const timings = Object.entries(r.timings).map(([k, v]) => `${k} ${pyFloat(v)}s`).join(", ");
  const style = css === undefined ? "" : `<style>${css}</style>\n`;

  return `<title>${esc(me)} Chop Sheet</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
${style}<div class="wrap">
<header class="top">
  <div><div class="sub">${esc(r.league)} · guillotine · week ${wk} of ${r.final_week}${pend}</div>
  <h1>${esc(me)} chop sheet</h1></div>
  <div class="meta">FAAB $${r.faab_me} · data pulled ${fet}<br>sims run ${gen} on ${r.threads} threads</div>
</header>

<div class="board">
${liveBox}
<section class="box"><h2>The calls <small>start here</small></h2><div class="inner"><ol class="calls">${callHtml}</ol></div></section>

${powerBox}

<section class="box"><h2>Week ${wk}: the chop line <small>${fmtComma(sims.season)} sims</small></h2><div class="inner">
  <div class="legend"><span><i style="background:var(--you)"></i>you</span><span><i style="background:var(--rival)"></i>lowest of the other ${st.length - 1}</span><span>bar = middle half, line = 10th–90th pct, tick = median</span></div>
  ${rc}
  <p class="note">You're chopped when your score falls below the lowest rival's. Your chop risk this week: ${riskChip(mine.p_chop_now)}</p>
</div></section>
<section class="box"><h2>Week ${wk}: lineup <small>${r.n_lineups} legal lineups · ${fmtComma(sims.lineups)} sims each</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>#</th><th class="r">Chop risk</th><th class="r">Pts ± sd</th><th>Starters</th><th></th></tr></thead><tbody>${luHtml}</tbody></table></div>
  <p class="note">Every lineup is scored on the same simulated weeks, so even small gaps between rows are real differences, not noise. Inactive starters are covered by your best eligible bench player.</p>
</div></section>

<section class="box"><h2>Free agents for your open spot <small>$0, first come</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>Player</th><th class="r">Wk ${wk}</th><th class="r">ROS /wk</th><th class="r">Win league</th><th class="r">Survive wk ${wk}</th><th></th></tr></thead><tbody>${faRows}</tbody></table></div>
  <p class="note">Change in your chances from adding the player and changing nothing else (opponents' rosters frozen). Players dropped since the last waiver run are left out because they still need a claim.</p>
</div></section>
<section class="box"><h2>Your roster <small>projected points, weeks ${wRange}</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>Player</th><th>Status</th><th>Wk ${wk}</th>${weeks.map((w) => `<th class="r">${w}</th>`).join("")}</tr></thead><tbody>${rosterRows}</tbody></table></div>
</div></section>

${scenBlocks}

<section class="box"><h2>What a FAAB dollar is worth <small>${fmtComma(D.budgets[0].n)} paths per point</small></h2><div class="inner">
  <div class="legend"><span><i style="background:var(--you)"></i>rivals as fitted</span><span><i style="background:var(--rival)"></i>rivals bid ~2× as often, ~35% higher</span></div>
  ${moneyChart}
  <p class="note">Same rest of season, different starting budget. The curve is flat at the top, so a little spending is free. It falls fast near the bottom: money is what lets you buy the studs that come loose when strong teams get chopped late.</p>
</div></section>
<section class="box"><h2>How hard to bid <small>× the league-typical bid for the same gain</small></h2><div class="inner">
  <div class="tbl">${heat.join("")}</div>
  <p class="note">P(win league) for each bidding style, all on the same ${fmtComma(D.sweep[0].n)} season paths. Darker is better. The sheet bids <b>${fmtG(used.m_early)}×</b> early and <b>${fmtG(used.m_late)}×</b> after week 12 (outlined, ${pct(used.p_win)}). ${topTxt} Overbidding early burns the budget on players who only marginally upgrade you.</p>
</div></section>

${paceBox}
<section class="box"><h2>Rival bidders <small>${bmr.n_bids} bids over ${fmtComma(bmr.n_rows)} chances</small></h2><div class="inner">
  <div class="tbl"><table><thead><tr><th>Manager</th><th class="r">FAAB</th><th class="r">Bids / chances</th><th class="r">Bids often</th><th class="r">Bids hard</th></tr></thead><tbody>${rivRows}</tbody></table></div>
  <p class="note"><b>Bids often</b> and <b>bids hard</b> are each manager's fitted effects (log-odds and log-dollars vs league average, shrunk toward 0). The dangerous ones have money <i>and</i> a positive score. Price check: ${pbIn} of ${pb.length} contested players this season cleared inside the model's 10th–90th range.</p>
  <div class="tbl"><table><thead><tr><th>Wk</th><th>Player</th><th class="r">Cleared</th><th class="r">Model median</th><th class="r">Model 80% range</th></tr></thead><tbody>${pbRows}</tbody></table></div>
</div></section>

<section class="box"><h2>The math: scores <small>back of the sheet, 1 of 3</small></h2><div class="inner math">
  <h3>1 · The chop</h3>
  <p>Each week, the lowest score among the alive teams is chopped. Your weekly risk is the probability that your total is below the minimum of the others, which depends on the whole joint distribution, not just averages:</p>
  <div class="eq">P(chop) = P( S_you &lt; min_j S_j )
S_team  = Σ_slots score(starter, or first available bench sub)</div>
  <h3>2 · One player's week</h3>
  <p>Fitted on ${fmtComma(cop.n_obs)} projected-vs-actual player weeks (last season plus this one). For each position and projection level, the empirical distribution of actual ÷ projected gives the quantile function, so busts, in-game injuries and boom weeks all come straight from data. Projections were unbiased in every bucket.</p>
  <div class="eq">score_i = proj_i · R_pos(u_i ; proj_i)
          R = empirical quantiles of actual ÷ projected
u_i     = Φ( a_pos · F_team + √(1 − a_pos²) · ε_i )
F_team  = √g · G_game + √(1 − g) · T_team
a = QB ${pyFloat(L.QB)}, RB ${pyFloat(L.RB)}, WR ${pyFloat(L.WR)}, TE ${pyFloat(L.TE)}     g = ${pyFloat(cop.game_share)}</div>
  <p>Measured residual correlations: QB–WR ${pyFloat(sameC["QB-WR"])}, QB–TE ${pyFloat(sameC["QB-TE"])}, RB–RB ${pyFloat(sameC["RB-RB"])} (committees cannibalize), opposing QBs ${pyFloat(oppC["QB-QB"])} (shootouts). Future weeks add a random-walk drift in each player's level (sd 0.10 per week, log scale) and a 2% weekly chance of a multi-week injury (mean 3 weeks).</p>
</div></section>

<section class="box"><h2>The math: bids <small>back of the sheet, 2 of 3</small></h2><div class="inner math">
  <h3>3 · Rivals' bids</h3>
  <div class="eq">gain_ji = mean over the next 4 wks of
          max(0, proj_iw − weakest starter i could replace on team j)
P(j bids)      = logistic(${pyFloat(pa.a0)} + ${pyFloat(pa.a_log1p_gain)}·ln(1+gain) ${pyFloatSigned(pa.a_log_budget)}·ln(budget) + v_j)
ln(bid | bids) = ${pyFloat(sz.b0)} + ${pyFloat(sz.b_gain)}·gain + ${pyFloat(sz.b_log_budget)}·ln(budget) + ${pyFloat(sz.b_leg)}·week + u_j + e
                 sd(e) = ${pyFloat(sz.sigma)}</div>
  <p>${fitSrc} Each extra lineup point a week raises a typical bid by about ${f0(100 * (2.71828 ** sz.b_gain - 1))}%, and bids rise about ${f0(100 * (2.71828 ** sz.b_leg - 1))}% per week as the season goes on.${r2Txt}</p>
  <h3>4 · Your bid</h3>
  <p>Sleeper FAAB is a sealed first-price auction: highest bid wins and pays its own bid. If F(b) is the chance your bid beats every rival and V is what the player adds to your win chance, the textbook answer is the bid that maximizes F(b) · (V − λb), where λ is what a dollar is worth to you later. Here λ isn't constant (the money curve bends), and V depends on what you'd buy later instead. So the sheet doesn't use the formula directly. It plays the whole rest of the season with every candidate bid and picks the one with the highest P(win league):</p>
  <div class="eq">b* = argmax_b  P(win league | bid b on player i,
                then your best policy in every later auction)</div>
</div></section>

<section class="box"><h2>The math: checks <small>back of the sheet, 3 of 3</small></h2><div class="inner math">
  <h3>5 · Why Monte Carlo, and why not just more of it</h3>
  <p>The quantities here are minimums of correlated, skewed, zero-inflated sums run through lineup logic and elimination, so there's no clean closed form. Past about a million sims the sampling error (±${seAt5} pts on a 5% risk) is far below the model error in the projections. So the CPU goes to <b>common random numbers</b> instead: every lineup, pickup and bid is scored on the same simulated weeks, which makes differences between choices sharper than more draws would. The rest goes into the ${D.sweep.length}-policy and per-bid season sweeps.</p>
  <h3>6 · Does it work?</h3>
  <p>${btTxt}</p>
  <h3>Known gaps</h3>
  <p>Frozen-roster numbers ignore the auctions (the dynamic sims don't). Rivals are assumed to start their max-projection lineup. Bid behavior is fit from about 7 weeks of history, so the stress-test line shows how the answer moves if rivals bid harder. The roster cap drops the weakest rest-of-season player automatically.</p>
</div></section>

<section class="box"><h2>Re-run it <small>do this Tuesday after the chop, before bids lock</small></h2><div class="inner">
  <pre class="cmd">cd ~/Desktop/"Fantasy 2026"/guillotine
.venv/bin/python fetch.py && .venv/bin/python run.py
    # ~7 min on 8 threads, runs at low priority
GSIM_THREADS=4 .venv/bin/python run.py   # fewer cores
.venv/bin/python run.py --quick          # 1/8 of the sims, ~1 min</pre>
  <p class="note">After the chop, the auction boxes switch from "if X is chopped" to the real roster on waivers and spend their sims on that one auction. Timings for this run: ${timings}.</p>
</div></section>
</div>
</div>
<div id="tip" role="tooltip"></div>
<script>${JS}</script>
`;
}
