// node:worker_threads adapter for src/engine.js (the browser uses src/worker.js).
import { parentPort } from "node:worker_threads";
import { makeEngine } from "../src/engine.js";

const engine = makeEngine();
parentPort.on("message", (m) => {
  try {
    if (m.type === "set") {
      engine.set(m.key, m.value);
      parentPort.postMessage({ id: m.id });
    } else parentPort.postMessage({ id: m.id, result: engine.run(m.kind, m.job) });
  } catch (e) {
    parentPort.postMessage({ id: m.id, error: String((e && e.stack) || e) });
  }
});
