// What a worker does: keep shared inputs (set once per run), then run sim chunks on request. The inline pool
// calls the same functions on the main thread, so tests and browsers without workers get identical results.
import { pathsChunk, prepDyn } from "./dynamic.js";
import { prepSeason, seasonGroups, week0Chunk, weekChunk } from "./sim.js";

export function makeEngine() {
  const state = {};
  return {
    set(key, value) {
      state[key] = key === "dyn" ? prepDyn(state.U, value) : value; // U is set first
      for (const k of Object.keys(state)) if (k.startsWith("season:")) delete state[k]; // prepared from the inputs: rebuilt on the next job
    },
    run(kind, job) {
      if (kind === "season") { // job.plans: the keys of the plans to run on the same draws
        const key = `season:${job.K}:${job.plans.join()}`;
        state[key] ??= prepSeason(state.U, job.plans.map((k) => state[k]), job.K);
        return seasonGroups(state[key], job);
      }
      if (kind === "week0") return week0Chunk(state.U, state[job.plan], job.ci, job.C, job.seed);
      if (kind === "week") return weekChunk(state.U, state[job.plan], job.ci, job.C, job.seed);
      if (kind === "paths") return pathsChunk(state.dyn, job);
      throw new Error(`unknown job ${kind}`);
    },
  };
}
