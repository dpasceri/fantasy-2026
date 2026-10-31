"""Baseline policy track candidate: dynamic.py's rule, bid m x the league-typical bid for my gain.

typical(g) = exp(mu(me, g, my budget, week) - u_me): the BidModel's median bid for my lineup gain at
my budget, without my own bids-hard effect. m = m_early before switch_week, m_late from it on; no
bid below MIN_GAIN lineup points a week. Bit-for-bit the policy dynamic.py simulates (verify.py).

Variants (m_early / m_late, switch week 12):
  baseline     1.0 / 0.75   the policy the pipeline runs: run.py's sweep argmax on this snapshot (out/results.json
                            dynamic.best, picked on seed-11 paths), used for its p_win_policy, budget curve and bid grid
  m0.75        0.75 / 0.75  the runner-up there (0.4273 vs 0.4277, se 0.005: a tie); the cheat sheet says "0.75-1x"
  x1.0         1.0 / 1.0    bid the league-typical amount all season
"""
from bench import harness  # noqa: F401  (core modules on sys.path, thread caps before numpy)
from bench.harness.dyn_policy import MIN_GAIN

VARIANTS = {"baseline": (1.0, 0.75, 12), "m0.75": (0.75, 0.75, 12), "x1.0": (1.0, 1.0, 12)}


class Multiplier:
    def __init__(self, variant="baseline", m_early=None, m_late=None, switch_week=None):
        a, b, s = VARIANTS[variant]
        self.variant = variant
        self.m_early = a if m_early is None else m_early
        self.m_late = b if m_late is None else m_late
        self.switch_week = s if switch_week is None else switch_week

    def bid(self, player, gain, week, state):
        if gain < MIN_GAIN:
            return 0
        m = self.m_early if week < self.switch_week else self.m_late
        return round(m * state.typical(gain))
