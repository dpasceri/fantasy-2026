// What a worker does: keep shared inputs (set once per run), then run sim chunks on request. The inline pool
// calls the same functions on the main thread, so tests and browsers without workers get identical results.
import { Bids } from "./auction.js";
import { pathsChunk } from "./dynamic.js";
import { seasonChunk, weekChunk } from "./sim.js";

export function makeEngine() {
  const state = {};
  return {
    set(key, value) {
      if (key === "dyn") value = { ...value, ctx: { ...value.ctx, bm: new Bids(value.ctx.bm) }, mcache: new Map() };
      state[key] = value;
    },
    run(kind, job) {
      if (kind === "season") return seasonChunk(state.U, state[job.plan], job.ci, job.C, job.seed);
      if (kind === "week") return weekChunk(state.U, state[job.plan], job.ci, job.C, job.seed);
      if (kind === "paths") return pathsChunk({ U: state.U, ...state.dyn }, job);
      throw new Error(`unknown job ${kind}`);
    },
  };
}
