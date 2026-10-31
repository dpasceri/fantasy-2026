"""Result files: bench/results/<key>.json (full) or <key>.smoke.json (smoke), one npz per variant.

{"key", "track", "mode": "smoke"|"full",
 "variants": [{"name", "metrics": {...}, "per_item": "bench/results/<key>__<variant>.npz", ...}],
 "baseline": "bench/results/_baseline_<track>.json", "runtime_s", "notes", ...extra}
Smoke runs use <key>.smoke.json, <key>__<variant>.smoke.npz and point at the smoke baseline, so a
smoke run never overwrites a full one. Each npz holds item_id, block and the per-item arrays.
"""
import json
import os
import re
import time

import numpy as np

from . import RESULTS, absp, rel

MODES = ("smoke", "full")
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
REQUIRED = ("key", "track", "mode", "variants", "baseline", "runtime_s", "notes")


def _sfx(mode):
    assert mode in MODES, mode
    return ".smoke" if mode == "smoke" else ""


def json_path(key, mode):
    return os.path.join(RESULTS, f"{key}{_sfx(mode)}.json")


def npz_path(key, variant, mode):
    return os.path.join(RESULTS, f"{key}__{variant}{_sfx(mode)}.npz")


def baseline_path(track, mode):
    return json_path(f"_baseline_{track}", mode)


def write(key, track, mode, variants, runtime_s, notes="", **extra):
    """variants: [{"name", "metrics", "arrays": {item_id, block, ...}} + optional extra fields].
    Writes the npz files and the JSON (atomically); returns the JSON path."""
    assert NAME_RE.match(key), f"bad key {key!r}"
    os.makedirs(RESULTS, exist_ok=True)
    out = []
    for v in variants:
        name = v["name"]
        assert NAME_RE.match(name), f"bad variant name {name!r}"
        a = v["arrays"]
        assert "item_id" in a and "block" in a, "per-item arrays need item_id and block"
        n = len(a["item_id"])
        for k, x in a.items():
            assert len(x) == n, f"{name}: array {k} has {len(x)} rows, expected {n}"
        p = npz_path(key, name, mode)
        tmp = p[:-4] + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **{k: _plain(x) for k, x in a.items()})
        os.replace(tmp, p)
        rec = {k: x for k, x in v.items() if k != "arrays"}
        rec["per_item"] = rel(p)
        out.append(rec)
    res = {"key": key, "track": track, "mode": mode, "variants": out,
           "baseline": rel(baseline_path(track, mode)), "runtime_s": round(float(runtime_s), 1),
           "notes": notes, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    p = json_path(key, mode)
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1, default=_jsonable)
    os.replace(tmp, p)
    return p


def _plain(x):
    """numpy array without object dtype (strings become fixed-width unicode) so npz loads pickle-free."""
    x = np.asarray(x)
    return x.astype(str) if x.dtype == object else x


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def read(path):
    with open(absp(path)) as f:
        return json.load(f)


def load_arrays(v):
    """Per-item arrays of a variant record (or an npz path) as a dict of numpy arrays."""
    p = v["per_item"] if isinstance(v, dict) else v
    with np.load(absp(p), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def validate(res, check_files=True):
    """List of schema problems (empty = valid)."""
    bad = [f"missing field {k}" for k in REQUIRED if k not in res]
    if res.get("mode") not in MODES:
        bad.append(f"mode {res.get('mode')!r}")
    for v in res.get("variants") or []:
        for k in ("name", "metrics", "per_item"):
            if k not in v:
                bad.append(f"variant missing {k}")
        if check_files and "per_item" in v:
            if not os.path.exists(absp(v["per_item"])):
                bad.append(f"missing {v['per_item']}")
                continue
            a = load_arrays(v)
            for k in ("item_id", "block"):
                if k not in a:
                    bad.append(f"{v['per_item']} lacks {k}")
            n = {len(x) for x in a.values()}
            if len(n) != 1:
                bad.append(f"{v['per_item']} arrays differ in length {sorted(n)}")
            if len(a.get("item_id", [])) != len(set(a.get("item_id", []).tolist())):
                bad.append(f"{v['per_item']} has duplicate item ids")
    if not res.get("variants"):
        bad.append("no variants")
    return bad


def find(track, mode="full"):
    """Every result JSON of a track and mode (leaderboards and the template excluded)."""
    out = []
    sfx = ".smoke.json" if mode == "smoke" else ".json"
    for f in sorted(os.listdir(RESULTS)):
        if not f.endswith(sfx) or f.startswith(("leaderboard_", "_template")):
            continue
        if mode == "full" and f.endswith(".smoke.json"):
            continue
        try:
            r = read(os.path.join(RESULTS, f))
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and r.get("track") == track and r.get("mode") == mode:
            out.append(r)
    return out


# ---------------------------------------------------------------- alignment
def assert_aligned(ids, ref, what=""):
    ids, ref = np.asarray(ids).astype(str), np.asarray(ref).astype(str)
    assert len(ids) == len(ref), f"{what}: {len(ids)} items vs {len(ref)} harness items"
    if not np.array_equal(ids, ref):
        miss = np.setdiff1d(ref, ids)[:5]
        extra = np.setdiff1d(ids, ref)[:5]
        raise AssertionError(f"{what}: item ids differ from the harness (missing {miss.tolist()}, extra {extra.tolist()}, "
                             "or a different order)")


def align(a, ref_ids, what=""):
    """Reorder per-item arrays to ref_ids order; AssertionError unless the item sets are identical."""
    ids = np.asarray(a["item_id"]).astype(str)
    ref = np.asarray(ref_ids).astype(str)
    if np.array_equal(ids, ref):
        return a
    assert len(ids) == len(ref) and len(np.unique(ids)) == len(ids) and set(ids.tolist()) == set(ref.tolist()), \
        f"{what}: item set differs from the reference ({len(ids)} vs {len(ref)} items)"
    order = {k: i for i, k in enumerate(ids.tolist())}
    ix = np.array([order[k] for k in ref.tolist()])
    return {k: np.asarray(x)[ix] for k, x in a.items()}


def evaluate(metric_list, a, ref_ids, what=""):
    """Assert a's items are exactly the harness items (same order), then summarize the metrics."""
    from .metrics import summarize
    assert_aligned(a["item_id"], ref_ids, what)
    return summarize(metric_list, a)
