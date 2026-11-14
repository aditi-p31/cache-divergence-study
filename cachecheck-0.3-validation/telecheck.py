"""Read-only check of the cache telemetry inside llm-cachecheck reports (pre-run review 2026-11-11, F6).

Usage: python3 telecheck.py <dir with cachecheck *.json reports>
Per report: turn 1 must reuse nothing, the recompute path must reuse nothing, every count must be
reported, and in a both-off control the cache path must reuse nothing too. Any CHECK line must be
read before the arm is quoted.
"""
import glob, json, os, sys

for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    if os.path.basename(f) in ("model_provenance.json",):
        continue
    try:
        P = json.load(open(f))["probes"]
    except Exception as e:
        print(os.path.basename(f), "CHECK unreadable", repr(e)[:80]); continue
    col = lambda k: [t["cached_tokens"] for p in P for t in p[k]]
    t1, w, c = col("turn1"), col("cache_path"), col("recompute_path")
    mx = lambda v: max([x or 0 for x in v], default=None)
    ctrl = "control" in os.path.basename(f)
    bad = not P or mx(t1) or mx(c) or None in t1 + w + c or (ctrl and mx(w))
    print("%s: %s; turn-1 cached max %s; cache-path turn-2 cached min %s max %s; recompute cached max %s; not reported %d" % (
        os.path.basename(f), "CHECK" if bad else "ok", mx(t1), min([x or 0 for x in w], default=None), mx(w), mx(c), (t1 + w + c).count(None)))
