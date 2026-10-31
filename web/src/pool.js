// Spreads sim chunks over CPU cores. WorkerPool uses Web Workers (one per core); InlinePool runs on the
// calling thread. Both: set(key, value) sends shared inputs to every worker, map(kind, jobs, onDone) runs jobs.
import { makeEngine } from "./engine.js";

export class InlinePool {
  constructor() {
    this.size = 1;
    this.engine = makeEngine();
  }

  async set(key, value) {
    this.engine.set(key, structuredClone(value));
  }

  async map(kind, jobs, onDone = () => {}) {
    const out = [];
    for (const job of jobs) {
      out.push(this.engine.run(kind, job));
      onDone(out.length, jobs.length);
      await new Promise((ok) => setTimeout(ok, 0)); // let the page repaint between chunks
    }
    return out;
  }

  close() {}
}

export class WorkerPool {
  constructor(size, makeWorker) {
    this.size = size;
    this.workers = Array.from({ length: size }, () => makeWorker());
    this.next = 0;
    this.waiting = new Map();
    for (const w of this.workers) {
      const onMsg = (m) => {
        const { id, result, error } = m.data ?? m;
        const p = this.waiting.get(id);
        this.waiting.delete(id);
        if (error) p.fail(new Error(error));
        else p.ok(result);
      };
      if (w.addEventListener) {
        w.addEventListener("message", onMsg);
        w.addEventListener("error", (e) => this._failAll(e.message || "worker error"));
      } else {
        w.on("message", (data) => onMsg({ data })); // node:worker_threads
        w.on("error", (e) => this._failAll(String(e)));
      }
    }
  }

  _failAll(msg) {
    for (const p of this.waiting.values()) p.fail(new Error(msg));
    this.waiting.clear();
  }

  _send(w, msg) {
    const id = this.next++;
    return new Promise((ok, fail) => {
      this.waiting.set(id, { ok, fail });
      w.postMessage({ ...msg, id });
    });
  }

  set(key, value) {
    return Promise.all(this.workers.map((w) => this._send(w, { type: "set", key, value })));
  }

  /** Runs jobs on whichever worker is free; results come back in job order. */
  map(kind, jobs, onDone = () => {}) {
    const out = new Array(jobs.length);
    let i = 0, done = 0;
    const feed = async (w) => {
      while (i < jobs.length) {
        const k = i++;
        out[k] = await this._send(w, { type: "run", kind, job: jobs[k] });
        onDone(++done, jobs.length);
      }
    };
    return Promise.all(this.workers.map(feed)).then(() => out);
  }

  close() {
    for (const w of this.workers) w.terminate();
  }
}
