"""MC-track baseline: plain Monte Carlo with sim.py's own sampler and chop rule.

n iid paths in chunks of 8000, chunk ci drawn from rng([seed, ci]) exactly as sim.run_season does,
so with n a multiple of 8000 the season estimate equals sim.run_season(...)["win"][0] bit for bit
(verify.py). No variance reduction: this is what every mc candidate's efficiency is measured against.
"""
from bench.harness.tracks import mc


class PlainMC:
    def __init__(self, variant="baseline"):
        self.variant = variant

    def estimate(self, target, n, seed):
        return mc.plain(target, n, seed)
