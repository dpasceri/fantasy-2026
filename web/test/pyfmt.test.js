// Expected values come from CPython 3.14 (f-strings, repr, round, sum, html.escape, strftime).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  esc, fmtComma, fmtE, fmtFixed, fmtG, fmtPct, maxBy, minBy, pyFloat, pyFloatSigned, pyRound, pySum, strftime, truthy,
} from "../src/pyfmt.js";

// x -> [.1f, .0f, +.2f, ,.0f, g, .0%, repr, +, e]
const CASES = [
  [0.125, ["0.1", "0", "+0.12", "0", "0.125", "12%", "0.125", "+0.125", "1.250000e-01"]],
  [2.5, ["2.5", "2", "+2.50", "2", "2.5", "250%", "2.5", "+2.5", "2.500000e+00"]],
  [202.25, ["202.2", "202", "+202.25", "202", "202.25", "20225%", "202.25", "+202.25", "2.022500e+02"]],
  [-0.001, ["-0.0", "-0", "-0.00", "-0", "-0.001", "-0%", "-0.001", "-0.001", "-1.000000e-03"]],
  [0, ["0.0", "0", "+0.00", "0", "0", "0%", "0.0", "+0.0", "0.000000e+00"]],
  [-0, ["-0.0", "-0", "-0.00", "-0", "-0", "-0%", "-0.0", "-0.0", "-0.000000e+00"]],
  [1e-5, ["0.0", "0", "+0.00", "0", "1e-05", "0%", "1e-05", "+1e-05", "1.000000e-05"]],
  [100, ["100.0", "100", "+100.00", "100", "100", "10000%", "100.0", "+100.0", "1.000000e+02"]],
  [1e16, ["10000000000000000.0", "10000000000000000", "+10000000000000000.00", "10,000,000,000,000,000", "1e+16",
    "1000000000000000000%", "1e+16", "+1e+16", "1.000000e+16"]],
  [123456.5, ["123456.5", "123456", "+123456.50", "123,456", "123456", "12345650%", "123456.5", "+123456.5", "1.234565e+05"]],
  [1234567, ["1234567.0", "1234567", "+1234567.00", "1,234,567", "1.23457e+06", "123456700%", "1234567.0", "+1234567.0", "1.234567e+06"]],
  [0.0001, ["0.0", "0", "+0.00", "0", "0.0001", "0%", "0.0001", "+0.0001", "1.000000e-04"]],
  [1.234e-5, ["0.0", "0", "+0.00", "0", "1.234e-05", "0%", "1.234e-05", "+1.234e-05", "1.234000e-05"]],
  [999999.5, ["999999.5", "1000000", "+999999.50", "1,000,000", "1e+06", "99999950%", "999999.5", "+999999.5", "9.999995e+05"]],
  [2.675, ["2.7", "3", "+2.67", "3", "2.675", "268%", "2.675", "+2.675", "2.675000e+00"]],
  [1e21, ["1000000000000000000000.0", "1000000000000000000000", "+1000000000000000000000.00", "1,000,000,000,000,000,000,000",
    "1e+21", "99999999999999991611392%", "1e+21", "+1e+21", "1.000000e+21"]],
  [NaN, ["nan", "nan", "+nan", "nan", "nan", "nan%", "nan", "+nan", "nan"]],
  [Infinity, ["inf", "inf", "+inf", "inf", "inf", "inf%", "inf", "+inf", "inf"]],
];

test("number formatting matches Python format specs", () => {
  for (const [x, want] of CASES) {
    const got = [fmtFixed(x, 1), fmtFixed(x, 0), fmtFixed(x, 2, { sign: "+" }), fmtFixed(x, 0, { comma: true }), fmtG(x),
      fmtPct(x, 0), pyFloat(x), pyFloatSigned(x), fmtE(x)];
    assert.deepEqual(got, want, `x = ${x}`);
  }
});

test("pyFloat prints None for null; fmtComma groups ints", () => {
  assert.equal(pyFloat(null), "None");
  assert.equal(fmtComma(3074), "3,074");
  assert.equal(fmtComma(-1234567), "-1,234,567");
  assert.equal(fmtComma(999), "999");
});

test("pyRound rounds the exact value, ties to even", () => {
  assert.deepEqual([0.1 + 0.2, 2.5e-7, 1.0000005, 7.5e-7].map((x) => pyRound(x, 6)), [0.3, 0, 1.000001, 1e-6]);
  assert.equal(pyRound(2.5), 2);
  assert.equal(pyRound(3.5), 4);
});

test("pySum is compensated like CPython 3.12+", () => {
  assert.equal(pySum(Array(10).fill(0.1)), 1);
  assert.equal(pySum([1e16, 1, -1e16]), 1);
  assert.equal(pySum([]), 0);
});

test("truthy follows Python", () => {
  assert.deepEqual([[], {}, "", 0, null, undefined, false].map(truthy), Array(7).fill(false));
  assert.deepEqual([[0], { a: 1 }, "x", NaN, -1, true].map(truthy), Array(6).fill(true));
});

test("esc matches html.escape", () => {
  assert.equal(esc(`<a href="x">Tom & Jerry's</a>`), "&lt;a href=&quot;x&quot;&gt;Tom &amp; Jerry&#x27;s&lt;/a&gt;");
});

test("maxBy/minBy keep the first of equal keys, like max/min(key=)", () => {
  const xs = [{ k: 1, i: 0 }, { k: 2, i: 1 }, { k: 2, i: 2 }, { k: 1, i: 3 }];
  assert.equal(maxBy(xs, (x) => x.k).i, 1);
  assert.equal(minBy(xs, (x) => x.k).i, 0);
  assert.equal(minBy([], (x) => x), undefined);
});

test("strftime matches datetime.fromtimestamp(...).strftime in the local zone", () => {
  process.env.TZ = "America/New_York"; // the zone the expected values were made in
  const fmt = "%a %b %-d, %-I:%M %p";
  assert.equal(strftime(1793414277, fmt), "Fri Oct 30, 10:37 PM");
  assert.equal(strftime(0, fmt), "Wed Dec 31, 7:00 PM");
  assert.equal(strftime(1767268800, fmt), "Thu Jan 1, 7:00 AM");
  assert.equal(strftime(1793413832, "%a %-I:%M %p"), "Fri 10:30 PM");
});
