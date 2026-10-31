import { makeEngine } from "./engine.js";

const engine = makeEngine();
self.onmessage = ({ data: m }) => {
  try {
    if (m.type === "set") {
      engine.set(m.key, m.value);
      self.postMessage({ id: m.id });
    } else self.postMessage({ id: m.id, result: engine.run(m.kind, m.job) });
  } catch (e) {
    self.postMessage({ id: m.id, error: String((e && e.stack) || e) });
  }
};
