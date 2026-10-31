// Numeric helpers that numpy/scipy gave the Python version for free.

/** erfc with fractional error < 1.2e-7 everywhere (Numerical Recipes' Chebyshev fit). */
export function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

/** Standard normal CDF (scipy.special.ndtr). */
export const ndtr = (x) => 0.5 * erfc(-x / Math.SQRT2);

/** Inverse standard normal CDF (scipy.special.ndtri): Acklam's approximation plus one Halley step. */
export function ndtri(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  let x;
  if (p < 0.02425) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p > 1 - 0.02425) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else {
    const q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  const e = ndtr(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}

/** np.quantile(sorted, q) with the default linear method; `s` must be sorted ascending. */
export function quantileSorted(s, q) {
  const n = s.length;
  if (!n) return NaN;
  const h = (n - 1) * q, lo = Math.floor(h), hi = Math.min(lo + 1, n - 1);
  return s[lo] + (h - lo) * (s[hi] - s[lo]);
}

export const sortNum = (a) => Float64Array.from(a).sort();
export const quantile = (a, q) => quantileSorted(sortNum(a), q);
export const median = (a) => quantile(a, 0.5);
export const sum = (a) => { let s = 0; for (const x of a) s += x; return s; };
export const mean = (a) => sum(a) / a.length;
/** Population variance (np.var). */
export function variance(a) {
  const m = mean(a);
  let s = 0;
  for (const x of a) s += (x - m) * (x - m);
  return s / a.length;
}

/** np.searchsorted on an ascending array. */
export function searchsorted(a, x, side = "left") {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (side === "left" ? a[mid] < x : a[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Solve A x = b for a small dense symmetric positive definite A (Cholesky). A is an array of rows. */
export function solveSPD(A, b) {
  const n = b.length, L = A.map(() => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j][j];
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/** Least squares X b = y (np.linalg.lstsq for a full-rank tall X), via the normal equations. */
export function lstsq(X, y) {
  const p = X[0].length;
  const A = Array.from({ length: p }, () => new Float64Array(p)), b = new Float64Array(p);
  for (let r = 0; r < X.length; r++) {
    const x = X[r];
    for (let i = 0; i < p; i++) {
      b[i] += x[i] * y[r];
      for (let j = 0; j < p; j++) A[i][j] += x[i] * x[j];
    }
  }
  return solveSPD(A, b);
}

// Python's round(): correctly rounded on the exact binary value, ties to even.
export { pyRound } from "./pyfmt.js";
