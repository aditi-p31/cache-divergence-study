"""The control the review said was missing: is the cached path reproducible
when cache state is restored to the same point?

The study's within-arm comparison shows two cache-enabled passes differ. A
referee answers: of course, their cache states differed. Only a controlled
cache state answers that, which is what this does.

HOW THE COLD STATE IS ESTABLISHED, and why not the obvious way. llama.cpp
exposes POST /slots/0?action=erase. It returns HTTP 200 and does not clear
the live prompt cache when issued between requests: measured directly, a
request following an "erase" still reported 109 cached prompt tokens, the
same as the warm request before it. Trusting it would have produced
plausible-looking but meaningless data. Two mechanisms are used instead:

  toggle (default) llama.cpp per-request cache_prompt=false forces a full
                   recompute regardless of slot contents. Verified: such
                   requests report 0 cached tokens. vLLM has no per-request
                   control, so there the prefix cache is reset through
                   /reset_prefix_cache and the effect is VERIFIED from
                   telemetry rather than assumed.
  restart          the server process is restarted between generations, the
                   only unconditionally certain reset. Slower, used for a
                   smaller subset as a cross-check.

Per item, four phases:
  cold_1  full recompute            cold_2  full recompute
  warm_1  full prefix hit           warm_2  full prefix hit

  cold_1 vs cold_2 : recompute path stable across cache generations?
  warm_1 vs warm_2 : CACHED path stable when cache state is equivalent?
                     This is the control that was missing.
  cold_1 vs warm_1 : the cache effect within one generation.

The run ABORTS if the manipulation check fails on more than a small fraction
of requests, so a silently ineffective reset cannot masquerade as a result.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer

SEED = 42
SYSTEM = ("You are a helpful assistant. Solve the math problem step by step and "
          "end your response with the final answer on its own line in the form "
          "#### <number>")
MAX_VIOLATION_RATE = 0.10


def cached_tokens_of(resp) -> int | None:
    """prompt_tokens_details is a pydantic model in the openai client, but a
    plain dict when a server returns an unmodelled shape. Handle both, and
    return None only when genuinely absent."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    det = getattr(usage, "prompt_tokens_details", None)
    if det is None:
        return None
    if isinstance(det, dict):
        return det.get("cached_tokens")
    return getattr(det, "cached_tokens", None)


def vllm_reset(base: str) -> str:
    root = base.removesuffix("/v1")
    try:
        r = requests.post(f"{root}/reset_prefix_cache", timeout=120)
        return f"http {r.status_code}"
    except requests.RequestException as exc:
        return f"FAILED {exc!r}"


def restart_server(cmd: str, health_url: str) -> str:
    subprocess.run("pkill -9 -f llama-server; pkill -9 -f 'vllm serve'",
                   shell=True, capture_output=True)
    time.sleep(5)
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    for _ in range(300):
        try:
            if requests.get(health_url, timeout=5).status_code == 200:
                return "restarted"
        except requests.RequestException:
            pass
        time.sleep(2)
    return "RESTART TIMEOUT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-hf-id", required=True)
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--backend", choices=["llamacpp", "vllm"], required=True)
    ap.add_argument("--mode", choices=["toggle", "restart"], default="toggle")
    ap.add_argument("--restart-cmd", default=None,
                    help="shell command to relaunch the server (mode=restart)")
    ap.add_argument("--health-url", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.mode == "restart" and not args.restart_cmd:
        sys.exit("mode=restart requires --restart-cmd")
    health = args.health_url or args.base_url.removesuffix("/v1") + "/health"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_requests.jsonl"
    if raw_path.exists():
        raw_path.unlink()  # never append to a previous attempt

    tok = AutoTokenizer.from_pretrained(args.model_hf_id)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    ds = load_dataset("openai/gsm8k", "main", split="test")

    rows, violations, checked = [], 0, 0
    for i in range(args.n):
        item = ds[i]
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": item["question"]}],
            tokenize=False, add_generation_prompt=True)

        texts, cached, how = {}, {}, {}
        for phase in ("cold_1", "warm_1", "cold_2", "warm_2"):
            is_cold = phase.startswith("cold")
            extra = {"seed": SEED}

            if is_cold:
                if args.mode == "restart":
                    how[phase] = restart_server(args.restart_cmd, health)
                elif args.backend == "llamacpp":
                    extra["cache_prompt"] = False
                    how[phase] = "cache_prompt=false"
                else:
                    how[phase] = vllm_reset(args.base_url)
            else:
                if args.backend == "llamacpp":
                    extra["cache_prompt"] = True
                how[phase] = "warm"

            t0 = time.time()
            resp = client.completions.create(
                model=args.served_model, temperature=0.0, prompt=prompt,
                max_tokens=args.max_tokens, logprobs=1,
                extra_body=extra, timeout=900)
            ct = cached_tokens_of(resp)
            cached[phase] = ct
            texts[phase] = resp.choices[0].text

            # manipulation check, per request, hard
            if ct is not None:
                checked += 1
                if is_cold and ct > 8:
                    violations += 1
                if (not is_cold) and ct == 0:
                    violations += 1

            with open(raw_path, "a") as fh:
                fh.write(json.dumps({
                    "item_idx": i, "phase": phase, "backend": args.backend,
                    "mode": args.mode, "how": how[phase], "cached_tokens": ct,
                    "latency_s": time.time() - t0,
                    "response": resp.model_dump(),
                }) + "\n")

        rows.append({
            "item_idx": i,
            "cold_reproducible": texts["cold_1"] == texts["cold_2"],
            "warm_reproducible": texts["warm_1"] == texts["warm_2"],
            "cache_effect": texts["cold_1"] != texts["warm_1"],
            "cached_tokens": cached,
        })

        if checked and violations / checked > MAX_VIOLATION_RATE and i >= 2:
            json.dump(rows, open(out / "summary_ABORTED.json", "w"), indent=1)
            sys.exit(
                f"ABORT after {i+1} items: manipulation check failing "
                f"({violations}/{checked} requests). The cold/warm states are "
                f"not what the protocol assumes, so any result would be "
                f"meaningless. Inspect {raw_path} before rerunning.")

        if (i + 1) % 10 == 0:
            print(f"{i+1}/{args.n}: cold-repro "
                  f"{sum(r['cold_reproducible'] for r in rows)}, warm-repro "
                  f"{sum(r['warm_reproducible'] for r in rows)}, cache-effect "
                  f"{sum(r['cache_effect'] for r in rows)}, "
                  f"check-violations {violations}/{checked}")

    json.dump(rows, open(out / "summary.json", "w"), indent=1)
    n = len(rows)
    print(f"\nRESET CONTROL  backend={args.backend} mode={args.mode} n={n}")
    print(f"  manipulation-check violations            : {violations}/{checked}")
    print(f"  recompute path reproducible              : "
          f"{sum(r['cold_reproducible'] for r in rows)}/{n}")
    print(f"  CACHED path reproducible (same state)    : "
          f"{sum(r['warm_reproducible'] for r in rows)}/{n}")
    print(f"  cold vs warm differ within one generation: "
          f"{sum(r['cache_effect'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
