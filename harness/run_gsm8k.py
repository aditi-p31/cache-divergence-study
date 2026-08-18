"""GSM8K single-turn bridge: cold (cache miss) vs warm (cache hit) contrast.

Replication link to arXiv:2604.15409's single-turn setting, using the
protocol from vLLM issue #33123: send the identical prompt twice in a row.
Request A is the cold pass (nothing cached), request B is the warm pass
(full prefix hit). Under cache_prompt=false (llama.cpp control arm) B must
equal A; with caching enabled any A-vs-B difference is a pure cache-path
divergence on a byte-identical request.

Selection rule (frozen): first N items of the GSM8K test split in dataset
order. Scoring: exact match on the final "#### <number>" answer.

Usage:
  uv run python harness/run_gsm8k.py --model-hf-id Qwen/Qwen2.5-3B-Instruct \
      --served-model qwen3b-q4km --base-url http://127.0.0.1:8080/v1 \
      --backend llamacpp --cache on --n 200 --out-dir results/gsm8k/qwen3b
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer

SEED = 42
SYSTEM = (
    "You are a helpful assistant. Solve the math problem step by step and "
    "end your response with the final answer on its own line in the form "
    "#### <number>"
)


def extract_answer(text: str) -> str | None:
    m = re.findall(r"####\s*([\-\d,\.]+)", text)
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-hf-id", required=True)
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--backend", choices=["llamacpp", "vllm", "sglang"], required=True)
    ap.add_argument("--cache", choices=["on", "off"], required=True,
                    help="llamacpp: per-request cache_prompt; vllm/sglang: "
                         "must match the server's cache flag (recorded only)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir) / f"cache_{args.cache}"
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_requests.jsonl"

    tok = AutoTokenizer.from_pretrained(args.model_hf_id)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    ds = load_dataset("openai/gsm8k", "main", split="test")

    extra = {"seed": SEED}

    results = []
    for i in range(args.n):
        item = ds[i]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": item["question"]},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        gold = extract_answer(item["answer"])
        passes = {}
        # Cold/warm protocol on llama.cpp: the cold pass sends
        # cache_prompt=false (forced full recompute regardless of slot
        # contents; llama.cpp still writes the KV as it processes), the warm
        # pass sends cache_prompt=true and reuses the KV the cold pass just
        # wrote, giving a full-prefix hit on a byte-identical request. The
        # cached_tokens telemetry in the raw log verifies both (cold: 0,
        # warm: ~full prompt). On vLLM/SGLang there is no per-request
        # control; the cache setting is server-level and pass purity is
        # assessed post hoc from cached_tokens telemetry.
        # Four passes: cold/cold2 (recompute path, determinism control) then
        # warm/warm2 (cache-hit path, determinism control). Divergence is
        # attributed to the cache path only when cold==cold2, warm==warm2,
        # and cold!=warm.
        for pass_name in ("cold", "cold2", "warm", "warm2"):
            if args.backend == "llamacpp":
                extra["cache_prompt"] = pass_name.startswith("warm")
            t0 = time.time()
            resp = client.completions.create(
                model=args.served_model,
                temperature=0.0,
                prompt=prompt,
                max_tokens=args.max_tokens,
                logprobs=1,
                extra_body=extra,
                timeout=600,
            )
            rec = {
                "item_idx": i,
                "pass": pass_name,
                "cache_setting": args.cache,
                "backend": args.backend,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "latency_s": time.time() - t0,
                "response": resp.model_dump(),
            }
            with open(raw_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            text = resp.choices[0].text
            passes[pass_name] = {
                "text": text,
                "answer": extract_answer(text),
                "cached_tokens": ((resp.usage.prompt_tokens_details or {})
                                  if resp.usage else {}),
            }
        row = {
            "item_idx": i,
            "gold": gold,
            "cold_answer": passes["cold"]["answer"],
            "warm_answer": passes["warm"]["answer"],
            "cold_correct": passes["cold"]["answer"] == gold,
            "warm_correct": passes["warm"]["answer"] == gold,
            "texts_identical": passes["cold"]["text"] == passes["warm"]["text"],
            "cold_deterministic": passes["cold"]["text"] == passes["cold2"]["text"],
            "warm_deterministic": passes["warm"]["text"] == passes["warm2"]["text"],
        }
        results.append(row)
        if (i + 1) % 20 == 0:
            div = sum(not r["texts_identical"] for r in results)
            print(f"{i+1}/{args.n}: {div} divergent, "
                  f"cold acc {sum(r['cold_correct'] for r in results)}, "
                  f"warm acc {sum(r['warm_correct'] for r in results)}")

    with open(out / "summary.json", "w") as fh:
        json.dump(results, fh, indent=1)
    div = sum(not r["texts_identical"] for r in results)
    flip = sum(r["cold_correct"] != r["warm_correct"] for r in results)
    print(f"done: {div}/{len(results)} text-divergent, {flip} correctness flips")


if __name__ == "__main__":
    main()
