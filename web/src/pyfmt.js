// Python formatting semantics the cheat sheet relies on, so web/src/report.js matches guillotine/report.py byte for byte.
// All rounding works on the exact decimal value of the double and breaks exact ties to even, as CPython does
// (JS toFixed/toPrecision break ties upward: (0.125).toFixed(2) is "0.13", Python's f"{0.125:.2f}" is "0.12").

/** Exact decimal expansion of finite |x| as [intDigits, fracDigits]: |x| = m·2^-k = m·5^k / 10^k. */
function exactAbs(x) {
  const a = Math.abs(x);
  if (Number.isInteger(a)) return [BigInt(a).toString(), ""];
  let m = a, k = 0;
  while (!Number.isInteger(m)) { m *= 2; k++; } // exact: doubling only bumps the exponent
  const s = (BigInt(m) * 5n ** BigInt(k)).toString().padStart(k + 1, "0");
  return [s.slice(0, -k), s.slice(-k).replace(/0+$/, "")];
}

const neg = (x) => x < 0 || Object.is(x, -0);

function nonFinite(x) {
  if (Number.isNaN(x)) return "nan";
  return x < 0 ? "-inf" : "inf";
}

/** Add one to a string of decimal digits ("099" -> "100", "99" -> "100"). */
function increment(digits) {
  const s = (BigInt(digits) + 1n).toString();
  return s.padStart(digits.length, "0");
}

/** Whether `keep` rounds up given the dropped digits `rest`, round-half-even on the exact value. */
function roundsUp(keep, rest) {
  if (!rest || rest[0] < "5") return false;
  if (rest[0] > "5" || /[1-9]/.test(rest.slice(1))) return true;
  return (Number(keep[keep.length - 1] || "0") % 2) === 1;
}

/** |x| rounded to d decimals: [intDigits, fracDigits (length d)]. */
function fixedParts(x, d) {
  const [i, f] = exactAbs(x);
  let keep = i + f.slice(0, d).padEnd(d, "0");
  if (roundsUp(keep, f.slice(d))) keep = increment(keep);
  const int = keep.slice(0, keep.length - d).replace(/^0+(?=\d)/, "") || "0";
  return [int, keep.slice(keep.length - d)];
}

/** |x| rounded to p significant digits: {digits (length p), exp} with |x| ~ d.ddd × 10^exp. */
function sigDigits(x, p) {
  const [i, f] = exactAbs(x);
  const all = i + f;
  const first = all.search(/[1-9]/);
  if (first < 0) return { digits: "0".repeat(p), exp: 0 };
  const exp = i.length - 1 - first;
  let keep = all.slice(first, first + p).padEnd(p, "0");
  if (roundsUp(keep, all.slice(first + p))) keep = increment(keep);
  return keep.length > p ? { digits: keep.slice(0, p), exp: exp + 1 } : { digits: keep, exp };
}

const expSuffix = (e) => `e${e < 0 ? "-" : "+"}${String(Math.abs(e)).padStart(2, "0")}`;

function group(int) {
  return int.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/** f"{x:.{d}f}", with `sign: "+"` for f"{x:+.{d}f}" and `comma: true` for f"{x:,.{d}f}". */
export function fmtFixed(x, d, { sign = "", comma = false } = {}) {
  const s = neg(x) ? "-" : sign;
  if (!Number.isFinite(x)) return (x < 0 ? "" : sign) + nonFinite(x);
  const [int, frac] = fixedParts(x, d);
  return s + (comma ? group(int) : int) + (d > 0 ? "." + frac : "");
}

/** f"{x:.{p}e}" */
export function fmtE(x, p = 6) {
  if (!Number.isFinite(x)) return nonFinite(x);
  const { digits, exp } = sigDigits(x, p + 1);
  return (neg(x) ? "-" : "") + digits[0] + (p > 0 ? "." + digits.slice(1) : "") + expSuffix(exp);
}

/** f"{x:g}" (precision 6): fixed for 1e-4 <= |x| < 1e6, else exponent; trailing zeros dropped. */
export function fmtG(x, p = 6) {
  if (!Number.isFinite(x)) return nonFinite(x);
  const s = neg(x) ? "-" : "";
  if (x === 0) return s + "0";
  const { digits, exp } = sigDigits(x, p);
  if (exp >= -4 && exp < p) {
    const int = exp >= 0 ? digits.slice(0, exp + 1) : "0";
    const frac = (exp >= 0 ? digits.slice(exp + 1) : "0".repeat(-exp - 1) + digits).replace(/0+$/, "");
    return s + int + (frac ? "." + frac : "");
  }
  const rest = digits.slice(1).replace(/0+$/, "");
  return s + digits[0] + (rest ? "." + rest : "") + expSuffix(exp);
}

/** f"{x:.{d}%}" (CPython multiplies by 100 in floating point, then formats as 'f'). */
export const fmtPct = (x, d = 0) => fmtFixed(x * 100, d) + "%";

/** str(x) / repr(x) for a Python float: shortest round-trip digits like JS, but always with a decimal
 *  point ("100.0") and exponent form outside 1e-4 <= |x| < 1e16 ("1e-05", "1e+16"). null prints "None". */
export function pyFloat(x) {
  if (x == null) return "None";
  if (!Number.isFinite(x)) return nonFinite(x);
  const s = neg(x) ? "-" : "";
  if (x === 0) return s + "0.0";
  // Shortest digits from JS, e.g. "0.000123" or "1.5e-7", re-laid out Python's way.
  const [m, e = "0"] = String(Math.abs(x)).split("e");
  const [mi, mf = ""] = m.split(".");
  const lead = (mi + mf).match(/^0*/)[0].length;
  const digits = (mi + mf).slice(lead).replace(/0+$/, "");
  const exp = Number(e) + mi.length - 1 - lead;
  if (exp < -4 || exp >= 16) return s + digits[0] + (digits.length > 1 ? "." + digits.slice(1) : "") + expSuffix(exp);
  if (exp < 0) return s + "0." + "0".repeat(-exp - 1) + digits;
  return s + digits.slice(0, exp + 1).padEnd(exp + 1, "0") + "." + (digits.slice(exp + 1) || "0");
}

/** f"{x:+}" for a Python float. */
export function pyFloatSigned(x) {
  const s = pyFloat(x);
  return s.startsWith("-") ? s : "+" + s;
}

/** f"{n:,}" for a Python int. */
export function fmtComma(n) {
  return Number.isInteger(n) ? (n < 0 ? "-" : "") + group(String(Math.abs(n))) : group(pyFloat(n));
}

/** round(x, nd) for a Python float: correctly rounded, ties to even on the exact value. */
export function pyRound(x, nd = 0) {
  if (!Number.isFinite(x)) return x;
  const [int, frac] = fixedParts(x, nd);
  const v = Number(int + (nd > 0 ? "." + frac : ""));
  return neg(x) ? -v : v;
}

/** sum() of floats: CPython 3.12+ uses Neumaier compensated summation, which can differ from a plain loop. */
export function pySum(xs) {
  let s = 0, c = 0;
  for (const x of xs) {
    const t = s + x;
    c += Math.abs(s) >= Math.abs(x) ? (s - t) + x : (x - t) + s;
    s = t;
  }
  return c && Number.isFinite(c) ? s + c : s;
}

/** Python truthiness: empty lists, empty dicts and "" are false; NaN is true. */
export function truthy(v) {
  if (v == null) return false;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "object") return Object.keys(v).length > 0;
  return Boolean(v);
}

/** html.escape(s) with quote=True. */
export function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#x27;");
}

/** max/min(xs, key=f): the first element with the extreme key, like Python (later ties never win). */
export function maxBy(xs, f) {
  let best, bk;
  for (const x of xs) {
    const k = f(x);
    if (best === undefined || k > bk) [best, bk] = [x, k];
  }
  return best;
}

export function minBy(xs, f) {
  let best, bk;
  for (const x of xs) {
    const k = f(x);
    if (best === undefined || k < bk) [best, bk] = [x, k];
  }
  return best;
}

/** sorted(xs, key=f) for numeric keys (stable, like Python). */
export const sortedBy = (xs, f) => [...xs].sort((a, b) => f(a) - f(b));

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** datetime.fromtimestamp(ts).strftime(fmt) in local time, C locale, for %a %b %-d %-I %M %p. */
export function strftime(ts, fmt) {
  const t = new Date(ts * 1000);
  const h = t.getHours();
  const codes = {
    a: DAYS[t.getDay()], b: MONTHS[t.getMonth()], "-d": String(t.getDate()),
    "-I": String(h % 12 || 12), M: String(t.getMinutes()).padStart(2, "0"), p: h < 12 ? "AM" : "PM",
  };
  return fmt.replace(/%(-?[a-zA-Z])/g, (m, c) => codes[c] ?? m);
}
