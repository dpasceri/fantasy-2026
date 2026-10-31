"""Shared backtest harness. Import as bench.harness with cwd = the guillotine folder.

Puts the guillotine folder on sys.path so the core modules (common, model, sim, ...) import, and
imports common first so its BLAS thread caps land before numpy loads.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BENCH)
RESULTS = os.path.join(BENCH, "results")
CACHE = os.path.join(BENCH, "cache")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import common  # noqa: E402,F401  (thread caps before numpy)


def slots():
    """Cores this process may use (cpuq sets CPUQ_SLOTS; 1 outside cpuq)."""
    return max(1, int(os.environ.get("CPUQ_SLOTS", "1")))


def task_threads():
    """Threads one fit/predict task may use: the harness sets HARNESS_THREADS when it runs
    origins in parallel; otherwise the whole cpuq allocation."""
    return max(1, int(os.environ.get("HARNESS_THREADS") or slots()))


def task_seed():
    """Seed for one fit/predict task: the harness sets HARNESS_SEED per origin (the same for every
    variant, so variants share random numbers) and also seeds numpy's and random's global RNGs."""
    return int(os.environ.get("HARNESS_SEED", "0"))


def rel(path):
    """Path relative to the guillotine folder, as stored in result files."""
    return os.path.relpath(path, ROOT)


def absp(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)
