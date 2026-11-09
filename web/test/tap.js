// A pool that keeps what a run sends its workers (the Universe data, the season plans, the dynamic sims' inputs and
// every job list), so tests can call the sim engines on a real run's inputs.
import { InlinePool } from "../src/pool.js";
import { League } from "../src/league.js";
import { runAll } from "../src/run.js";

export class TapPool extends InlinePool {
  constructor() {
    super();
    this.sent = {};
    this.jobs = []; // [kind, jobs] per map() call, in order
  }

  async set(key, value) {
    this.sent[key] = structuredClone(value);
    return super.set(key, value);
  }

  async map(kind, jobs, onDone) {
    this.jobs.push([kind, structuredClone(jobs)]);
    return super.map(kind, jobs, onDone);
  }
}

/** Run a league at 1/k of the sims and return {lg, r, sent, jobs}. */
export async function tapRun(raw, me, { k = 160, ...opts } = {}) {
  const lg = new League(raw, me, opts), pool = new TapPool();
  const r = await runAll(lg, pool, { k });
  return { lg, r, sent: pool.sent, jobs: pool.jobs };
}
