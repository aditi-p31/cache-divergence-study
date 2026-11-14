#!/usr/bin/env python3
"""Single-stream latency of vLLM 0.33.0 batch-invariant mode, from the November 2026 logs.

No GPU and no new run: reads raw_requests.jsonl of the five 7B passes in
results-currency-2026-11/results (from currency_results.tgz, sha256 64525a86...).
Every server ran with --max-num-seqs 1 and the harness sent one request at a
time, so this is batch size one. latency_s is client wall time around one
non-streaming /v1/completions call (harness/handler.py).

Fair comparison: only requests that are identical in both passes (same
episode and request index, same prompt sha256, identical completion tokens).
Passes ran one after another, about an hour apart; B_vllm/arm_off against
B_vllm_rep/arm_off (identical outputs) gives the run-to-run timing noise.

Usage: python3 latency_matched.py <results dir>
"""
import json, os, statistics as st, sys

R = sys.argv[1]
PASSES = ["B_vllm/arm_on", "B_vllm/arm_off", "B_vllm_rep/arm_off",
          "D_vllm_bi/arm_on", "D_vllm_bi/arm_off"]
NAMES = {"B_vllm/arm_on": "default, cache on", "B_vllm/arm_off": "default, cache off",
         "B_vllm_rep/arm_off": "default, cache off (repeat)",
         "D_vllm_bi/arm_on": "batch-invariant, cache on", "D_vllm_bi/arm_off": "batch-invariant, cache off"}
data = {p: [json.loads(l) for l in open(os.path.join(R, p, "raw_requests.jsonl"))] for p in PASSES}

print("Whole passes (different trajectories; totals are not directly comparable):")
for p in PASSES:
    lat = [r["latency_s"] for r in data[p]]
    print("  %-30s %4d requests, median %.3f s, p95 %.3f s, latency sum %.1f s"
          % (NAMES[p], len(lat), st.median(lat), st.quantiles(lat, n=20)[18], sum(lat)))

key = lambda r: (r["episode_id"], r["request_idx"])
toks = lambda r: r["response"]["choices"][0]["logprobs"]["tokens"]
print("Matched requests (identical prompt and completion in both passes):")
for a, b in [("B_vllm/arm_off", "B_vllm_rep/arm_off"),
             ("B_vllm/arm_on", "D_vllm_bi/arm_on"),
             ("B_vllm/arm_off", "D_vllm_bi/arm_off"),
             ("B_vllm/arm_off", "D_vllm_bi/arm_on"),
             ("B_vllm_rep/arm_off", "D_vllm_bi/arm_on"),
             ("B_vllm/arm_on", "B_vllm/arm_off")]:
    A = {key(r): r for r in data[a]}; B = {key(r): r for r in data[b]}
    m = [(A[k], B[k]) for k in A if k in B and A[k]["prompt_sha256"] == B[k]["prompt_sha256"]
         and toks(A[k]) == toks(B[k])]
    la = sum(x["latency_s"] for x, _ in m); lb = sum(y["latency_s"] for _, y in m)
    med = st.median(y["latency_s"] / x["latency_s"] for x, y in m if x["latency_s"] > 0)
    eps = len({x["episode_id"] for x, _ in m})
    print("  %-28s -> %-28s %4d requests in %2d episodes: time ratio %.2f (sums), %.2f (median per request)"
          % (NAMES[a], NAMES[b], len(m), eps, lb / la, med))
