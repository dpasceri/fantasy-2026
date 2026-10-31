// Seeded random numbers: xoshiro128** seeded by splitmix32, so a (seed, chunk) pair always gives the same
// stream no matter which worker runs it. Not numpy's stream, so results match Python only in distribution.

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

export class Rng {
  /** seeds: integers, e.g. [seed, chunk], like np.random.default_rng([seed, ci]). */
  constructor(...seeds) {
    let h = 0x811c9dc5;
    for (const s of seeds.flat()) h = Math.imul(h ^ (s >>> 0), 0x01000193) ^ Math.imul(Math.floor(s / 2 ** 32), 0x5bd1e995);
    const sm = splitmix32(h);
    this.s = new Uint32Array([sm(), sm(), sm(), sm()]);
    this.spare = null;
  }

  nextU32() {
    const s = this.s;
    const r = Math.imul(rotl(Math.imul(s[1], 5), 7), 9) >>> 0;
    const t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11);
    return r;
  }

  /** Uniform on (0, 1). */
  random() {
    return (this.nextU32() + 0.5) * 2.3283064365386963e-10;
  }

  /** Standard normal (Marsaglia polar method). */
  normal() {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u, v, s;
    do {
      u = this.random() * 2 - 1;
      v = this.random() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    this.spare = v * m;
    return u * m;
  }

  /** Geometric on 1, 2, ... with success probability p (numpy's rng.geometric). */
  geometric(p) {
    return p >= 1 ? 1 : Math.max(1, Math.ceil(Math.log(this.random()) / Math.log1p(-p)));
  }
}

const rotl = (x, k) => (x << k) | (x >>> (32 - k));
