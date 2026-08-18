"""Analyse the repair experiments and emit repair_findings.json.

Three questions, each answered by a distinct comparison:

  ordering   Does the execution order of passes explain the divergence?
             Same config run with a cache-off pass between the two cache-on
             passes (A) and with them adjacent (B).
  cacheram   Does llama.cpp's server-level LRU prompt cache drive run-to-run
             divergence? Same fresh server, same episodes, only --cache-ram
             differs.
  gradient   Does the quantization gradient survive under a controlled cache
             configuration? Cross-arm divergence at three weight formats with
             --cache-ram 0.

Plus the reset control summary, which asks whether the cached path is
reproducible when cache state is restored to a known point.
"""

import json
import math
from pathlib import Path

ROOT = Path("results-repair")


def load(p: Path) -> dict:
    by = {}
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            by.setdefault(r["episode_id"], []).append(r)
    for v in by.values():
        v.sort(key=lambda x: x["request_idx"])
    return by


def toks(rec):
    ch = rec["response"]["choices"][0]
    lp = ch.get("logprobs") or {}
    if lp.get("content"):
        return [t["id"] for t in lp["content"]]
    return lp.get("tokens") or [ch.get("text", "")]


def diverged(a, b) -> bool:
    if len(a) != len(b):
        return True
    return any(x["prompt_sha256"] != y["prompt_sha256"] or toks(x) != toks(y)
               for x, y in zip(a, b))


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def compare(p1: Path, p2: Path) -> dict | None:
    if not p1.exists() or not p2.exists():
        return None
    d1, d2 = load(p1), load(p2)
    eps = sorted(set(d1) & set(d2))
    k = sum(diverged(d1[e], d2[e]) for e in eps)
    return {"n": len(eps), "n_diverged": k,
            "rate": round(k / len(eps), 4) if eps else None,
            "ci95": wilson(k, len(eps))}


out = {}

# --- ordering: within-arm cache-on repeat, both orderings, both engines ---
out["ordering"] = {}
for eng, cfgs in (("llamacpp", ("lcpp-orderA", "lcpp-orderB")),
                  ("vllm", ("vllm-orderA", "vllm-orderB"))):
    for cfg in cfgs:
        r = compare(ROOT / "repair" / cfg / "main" / "arm_on" / "raw_requests.jsonl",
                    ROOT / "repair" / cfg / "repeat" / "arm_on" / "raw_requests.jsonl")
        x = compare(ROOT / "repair" / cfg / "main" / "arm_on" / "raw_requests.jsonl",
                    ROOT / "repair" / cfg / "main" / "arm_off" / "raw_requests.jsonl")
        if r or x:
            out["ordering"][cfg] = {"engine": eng, "within_arm_on": r, "cross_arm": x}

# --- cacheram isolation ---
out["cacheram"] = {}
for lbl in ("cacheram-zero", "cacheram-default"):
    r = compare(ROOT / "cacheram" / lbl / "main" / "arm_on" / "raw_requests.jsonl",
                ROOT / "cacheram" / lbl / "repeat" / "arm_on" / "raw_requests.jsonl")
    if r:
        out["cacheram"][lbl] = r

# --- quantization gradient under controlled cache config ---
out["gradient_controlled"] = {}
for lbl, base in (("f16", ROOT / "quantgrad" / "f16" / "main"),
                  ("q4km", ROOT / "repair" / "lcpp-orderB" / "main"),
                  ("q3km", ROOT / "quantgrad" / "q3km" / "main")):
    x = compare(base / "arm_on" / "raw_requests.jsonl",
                base / "arm_off" / "raw_requests.jsonl")
    if x:
        out["gradient_controlled"][lbl] = x

# --- reset control ---
rc = ROOT / "repair" / "reset-lcpp" / "summary.json"
if rc.exists():
    rows = json.load(open(rc))
    n = len(rows)
    cold = sum(r["cold_reproducible"] for r in rows)
    warm = sum(r["warm_reproducible"] for r in rows)
    eff = sum(r["cache_effect"] for r in rows)
    cond = [r for r in rows if r["cold_reproducible"]]
    out["reset_control_llamacpp"] = {
        "n": n, "recompute_reproducible": cold, "cached_reproducible": warm,
        "cached_reproducible_given_cold": sum(r["warm_reproducible"] for r in cond),
        "n_cold_reproducible": len(cond),
        "cache_effect_within_generation": eff,
        "ci95_cached_reproducible": wilson(warm, n),
    }

json.dump(out, open("analysis/repair_findings.json", "w"), indent=1)

print("ORDERING (within-arm cache-on repeat)")
for cfg, v in out["ordering"].items():
    w = v["within_arm_on"]; x = v["cross_arm"]
    ws = f"{100*w['rate']:.1f}% ({w['n_diverged']}/{w['n']})" if w else "n/a"
    xs = f"{100*x['rate']:.1f}%" if x else "n/a"
    print(f"  {cfg:<16} within-arm {ws:<18} cross-arm {xs}")
print("\nCACHE-RAM ISOLATION (fresh server both arms, only the flag differs)")
for lbl, v in out["cacheram"].items():
    print(f"  {lbl:<18} {100*v['rate']:>6.1f}%  ({v['n_diverged']}/{v['n']})  CI {v['ci95']}")
print("\nQUANTIZATION GRADIENT, controlled (cross-arm)")
for lbl, v in out["gradient_controlled"].items():
    print(f"  {lbl:<18} {100*v['rate']:>6.1f}%  CI {v['ci95']}")
r = out.get("reset_control_llamacpp")
if r:
    print(f"\nRESET CONTROL (n={r['n']}): recompute reproducible "
          f"{r['recompute_reproducible']}/{r['n']}, cached reproducible "
          f"{r['cached_reproducible']}/{r['n']}, cache effect "
          f"{r['cache_effect_within_generation']}/{r['n']}")
print("\nwrote analysis/repair_findings.json")
