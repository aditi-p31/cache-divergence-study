"""Robustness checks the referee round asked for.

Two questions the main analysis does not answer:

  position   Episodes within a cell share one server and run in fixed order,
             and this study's own thesis is that state carries across
             requests. Episodes are therefore not exchangeable, and Wilson
             intervals assume they are. This measures whether divergence
             depends on execution position, and reports a moving-block
             bootstrap interval that does not assume independence.

  power      What paired accuracy difference could the bridge actually
             detect? Reported as a simulation against the observed
             discordant rate rather than an assertion.

Usage: uv run python analysis/robustness.py
"""

import gzip
import json
import math
import random
from pathlib import Path

random.seed(42)


def open_maybe_gz(p):
    if p.exists():
        return open(p)
    gz = p.with_suffix(p.suffix + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt")
    raise FileNotFoundError(p)


def load(p):
    by = {}
    for line in open_maybe_gz(p):
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


def diverged(a, b):
    if len(a) != len(b):
        return True
    return any(x["prompt_sha256"] != y["prompt_sha256"] or toks(x) != toks(y)
               for x, y in zip(a, b))


def series(p1, p2):
    """Divergence indicator per episode, in execution order."""
    d1, d2 = load(p1), load(p2)
    eps = sorted(set(d1) & set(d2), key=lambda e: int(e.rsplit("_", 1)[1]))
    return [1 if diverged(d1[e], d2[e]) else 0 for e in eps]


def perm_trend_p(x, n_perm=20000):
    """Permutation test for a monotone trend in divergence with position."""
    n = len(x)
    idx = list(range(n))
    obs = sum(i * v for i, v in enumerate(x))
    count = 0
    y = list(x)
    for _ in range(n_perm):
        random.shuffle(y)
        if sum(i * v for i, v in enumerate(y)) <= obs:
            count += 1
    lo = (count + 1) / (n_perm + 1)
    return 2 * min(lo, 1 - lo + 1 / (n_perm + 1))


def moving_block_ci(x, block=10, n_boot=10000):
    """Moving-block bootstrap CI for the divergence rate, which preserves
    local dependence between neighbouring episodes."""
    n = len(x)
    nblocks = math.ceil(n / block)
    starts = list(range(n - block + 1))
    rates = []
    for _ in range(n_boot):
        s = []
        for _ in range(nblocks):
            st = random.choice(starts)
            s.extend(x[st:st + block])
        rates.append(sum(s[:n]) / n)
    rates.sort()
    return rates[int(0.025 * n_boot)], rates[int(0.975 * n_boot)]


def binom_cdf(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * binom_cdf(min(b, c), n))


def power_sim(n_items, disc_rate, net_pp, n_sim=4000):
    """Power of the exact McNemar test against a given net shift, at the
    observed discordant rate."""
    hits = 0
    for _ in range(n_sim):
        b = c = 0
        # split the discordant mass to produce the requested net difference
        p_disc = disc_rate
        p_b = p_disc / 2 + (net_pp / 100) / 2
        p_c = p_disc / 2 - (net_pp / 100) / 2
        if p_c < 0:
            p_b, p_c = p_disc, 0.0
        for _ in range(n_items):
            u = random.random()
            if u < p_b:
                b += 1
            elif u < p_b + p_c:
                c += 1
        if mcnemar(b, c) < 0.05:
            hits += 1
    return hits / n_sim


out = {}
# Resolve the results root: the artifact repo ships results/ and
# results-repair/ at the top level; the development tree keeps the grid
# under results-pod/results/.
GRID = Path("results-pod/results") if Path("results-pod/results").exists() else Path("results")
R = Path("results-repair")

print("POSITION DEPENDENCE (divergence vs execution order within a cell)")
cells = {
    "cacheram-default": (R / "cacheram/cacheram-default/main/arm_on/raw_requests.jsonl",
                         R / "cacheram/cacheram-default/repeat/arm_on/raw_requests.jsonl"),
    "grid lcpp-qwen7b-q4km": (GRID / "lcpp-qwen7b-q4km/main/arm_on/raw_requests.jsonl",
                              GRID / "lcpp-qwen7b-q4km/repeat/arm_on/raw_requests.jsonl"),
}
for name, (a, b) in cells.items():
    if not (a.exists() or a.with_suffix(a.suffix + ".gz").exists()):
        continue
    x = series(a, b)
    q = [sum(x[i:i + 20]) for i in range(0, 80, 20)]
    p = perm_trend_p(x)
    lo, hi = moving_block_ci(x)
    naive = sum(x) / len(x)
    out[name] = {"rate": naive, "quartiles": q, "trend_p": round(p, 4),
                 "block_ci95": [round(lo, 4), round(hi, 4)]}
    print(f"  {name:<24} rate {naive:.3f}  quartiles {q}  trend p={p:.4f}")
    print(f"  {'':<24} moving-block 95% CI [{lo:.3f}, {hi:.3f}]")

print("\nBRIDGE POWER (exact McNemar, observed discordant rate)")
tot = disc = 0
for d in sorted(GRID.glob("bridge-*")):
    if d.name == "bridge-lcpp-qwen7b-q4km":
        continue  # nested inside the 500-item run
    rows = json.load(open(d / "cache_on" / "summary.json"))
    tot += len(rows)
    disc += sum(1 for r in rows
                if r["warm_correct"] != r["cold_correct"])
rate = disc / tot
print(f"  distinct items {tot}, discordant {disc} ({100*rate:.2f}%)")
for net in (0.5, 0.8, 1.0, 1.5, 2.0):
    pw = power_sim(tot, rate, net)
    print(f"  net shift {net:>4.1f} pp -> power {pw:.2f}")
    out.setdefault("power", {})[f"{net}pp"] = round(pw, 3)
out["bridge_discordant_rate"] = round(rate, 4)
out["bridge_n_distinct"] = tot

json.dump(out, open("analysis/robustness.json", "w"), indent=1)
print("\nwrote analysis/robustness.json")
