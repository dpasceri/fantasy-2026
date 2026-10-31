// Seeded random numbers: xoshiro128** seeded by splitmix32, so a (seed, chunk) pair always gives the same
// stream no matter which worker runs it. Not numpy's stream, so results match Python only in distribution.
// Normals use Marsaglia & Tsang's ziggurat: one 32-bit draw and a table lookup ~99% of the time.

function splitmix32(a) {
  return () => {
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t ^= t >>> 15;
    t = Math.imul(t, 0x735a2d97);
    return (t ^ (t >>> 15)) >>> 0;
  };
}

// ziggurat tables (128 layers)
const KN = new Uint32Array(128), WN = new Float64Array(128), FN = new Float64Array(128);
{
  const m1 = 2147483648, vn = 9.91256303526217e-3;
  let dn = 3.442619855899, tn = dn;
  const q = vn / Math.exp(-0.5 * dn * dn);
  KN[0] = (dn / q) * m1;
  KN[1] = 0;
  WN[0] = q / m1;
  WN[127] = dn / m1;
  FN[0] = 1;
  FN[127] = Math.exp(-0.5 * dn * dn);
  for (let i = 126; i >= 1; i--) {
    dn = Math.sqrt(-2 * Math.log(vn / dn + Math.exp(-0.5 * dn * dn)));
    KN[i + 1] = (dn / tn) * m1;
    tn = dn;
    FN[i] = Math.exp(-0.5 * dn * dn);
    WN[i] = dn / m1;
  }
}
const ZIG_R = 3.442619855899;

export class Rng {
  /** seeds: integers, e.g. (seed, chunk), like np.random.default_rng([seed, ci]). */
  constructor(...seeds) {
    let h = 0x811c9dc5;
    for (const s of seeds.flat()) h = Math.imul(h ^ (s >>> 0), 0x01000193) ^ Math.imul(Math.floor(s / 2 ** 32), 0x5bd1e995);
    const sm = splitmix32(h);
    this.a = sm() | 0;
    this.b = sm() | 0;
    this.c = sm() | 0;
    this.d = sm() | 0;
  }

  /** A uniform 32-bit integer, as a signed int32. */
  nextI32() {
    const b = this.b, r = Math.imul(rotl(Math.imul(b, 5), 7), 9), t = b << 9;
    this.c ^= this.a;
    this.d ^= b;
    this.b ^= this.c;
    this.a ^= this.d;
    this.c ^= t;
    this.d = rotl(this.d, 11);
    return r;
  }

  /** Uniform on (0, 1). */
  random() {
    return ((this.nextI32() >>> 0) + 0.5) * 2.3283064365386963e-10;
  }

  /** Standard normal (ziggurat). */
  normal() {
    let hz = this.nextI32(), iz = hz & 127;
    if (Math.abs(hz) < KN[iz]) return hz * WN[iz];
    for (;;) {
      if (iz === 0) { // the tail beyond ZIG_R
        let x, y;
        do {
          x = -Math.log(this.random()) / ZIG_R;
          y = -Math.log(this.random());
        } while (y + y < x * x);
        return hz > 0 ? ZIG_R + x : -ZIG_R - x;
      }
      const x = hz * WN[iz];
      if (FN[iz] + this.random() * (FN[iz - 1] - FN[iz]) < Math.exp(-0.5 * x * x)) return x;
      hz = this.nextI32();
      iz = hz & 127;
      if (Math.abs(hz) < KN[iz]) return hz * WN[iz];
    }
  }

  /** Geometric on 1, 2, ... with success probability p (numpy's rng.geometric). */
  geometric(p) {
    return p >= 1 ? 1 : Math.max(1, Math.ceil(Math.log(this.random()) / Math.log1p(-p)));
  }
}

const rotl = (x, k) => (x << k) | (x >>> (32 - k));
