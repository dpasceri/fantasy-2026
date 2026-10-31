"""Sanity checks for SimOpt (tuning seed 1000 only, about 10 s on 1 core).

1. SimOpt on the step family reproduces the baseline Multiplier bit for bit (every path, every condition).
2. cap() agrees with State.rival_cdf: F(cap - 1) >= alpha and F(cap - 2) < alpha, on captured auction states.
3. Every frozen variant in tuned.json builds and bids finite amounts on those states.
"""
import json

from bench import harness  # noqa: F401
from bench.harness import dyn_policy as DP
from bench.harness.tracks import policy as T
from bench.candidates._baseline_policy.multiplier import Multiplier
from bench.candidates.sim_optimization.policy import SimOpt, TUNED, cap

import numpy as np


def main():
    E = T.env()
    a, b = T.tune([Multiplier("baseline"), SimOpt(theta={"m_early": 1.0, "m_late": 0.75})], seeds=[1000],
                  conditions=("", "_stress", "_bmboot"))
    for k, v in a["arrays"].items():
        assert np.array_equal(v, b["arrays"][k]), k
    print(f"1. SimOpt(1.0/0.75) == Multiplier('baseline') on seed 1000, all conditions (p_win {a['p_win']:.3f})")

    class Probe(SimOpt):
        seen = []

        def begin(self, state):
            super().begin(state)
            if len(self.seen) < 25:
                self.seen.append(state)

    DP._paths_chunk((0, 10, 1001, [{"policy": Probe(theta={"alpha": 0.9}), "budget": E["base"]["budget"][E["base"]["me"]]}]))
    n = 0
    for st in Probe.seen:
        for p in st.pool:
            for al in (0.3, 0.6, 0.9, 0.99):
                c = cap(p, st, al)
                F = st.rival_cdf(p, np.array([c - 1, c - 2]))
                assert F[0] >= al - 1e-12 and (c == 1 or F[1] < al + 1e-12), (p, al, c, F)
                n += 1
    print(f"2. cap() matches State.rival_cdf on {n} (state, player, alpha) cases")
    tv = json.load(open(TUNED))["variants"]
    for v in tv:
        pol = SimOpt(variant=v)
        bids = [pol.bid(p, st.gains[p], st.week, st) for st in Probe.seen for p in st.pool]
        assert all(np.isfinite(bids)), v
        print(f"3. {v}: {pol.th} -> {sum(x >= 1 for x in bids)} claims on {len(bids)} offers, max ${max(bids):.0f}")


if __name__ == "__main__":
    main()
