// Spreads sim chunks over CPU cores. WorkerPool uses Web Workers (one per core); InlinePool runs on the
// calling thread. Both: set(key, value) sends shared inputs to every worker, map(kind, jobs, onDone) runs jobs.
// Jobs from several map() calls share one queue in the order they were added, so a stage's tail is filled by
// the next stage's jobs; every job's result depends only on its own inputs, never on which worker ran it or when.
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
    this.queue = []; // jobs not yet sent: {kind, job, ok, fail}
    this.idle = [...this.workers];
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
    for (const q of this.queue) q.fail(new Error(msg));
    this.queue = [];
  }

  _send(w, msg) {
    const id = this.next++;
    return new Promise((ok, fail) => {
      this.waiting.set(id, { ok, fail });
      w.postMessage({ ...msg, id });
    });
  }

  /** Workers see a set before any job queued after it (messages arrive in order), so use a fresh key per stage. */
  set(key, value) {
    return Promise.all(this.workers.map((w) => this._send(w, { type: "set", key, value })));
  }

  /** Queues jobs for whichever worker is free next; results come back in job order. */
  map(kind, jobs, onDone = () => {}) {
    let done = 0;
    const out = jobs.map((job) => new Promise((ok, fail) => this.queue.push({ kind, job, fail, ok: (r) => { onDone(++done, jobs.length); ok(r); } })));
    this._pump();
    return Promise.all(out);
  }

  _pump() {
    while (this.idle.length && this.queue.length) {
      const w = this.idle.pop(), q = this.queue.shift();
      this._send(w, { type: "run", kind: q.kind, job: q.job }).then(q.ok, q.fail).then(() => {
        this.idle.push(w);
        this._pump();
      });
    }
  }

  close() {
    for (const w of this.workers) w.terminate();
  }
}
