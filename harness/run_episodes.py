"""Run BFCL v4 multi-turn episodes through one (config, arm) cell.

Usage:
  uv run python harness/run_episodes.py \
      --model-hf-id Qwen/Qwen2.5-3B-Instruct \
      --served-model qwen2.5-3b-q4km \
      --base-url http://127.0.0.1:8080/v1 \
      --backend llamacpp --arm on --n-episodes 5 \
      --out-dir results/dryrun/qwen3b-q4km-llamacpp

Episode selection rule (frozen, no cherry-picking): the first N episodes of
BFCL_v4_multi_turn_base.json in ascending numeric id order.

Outputs per cell:
  raw_requests.jsonl   every request/response with tokens and logprobs
  model_results.json   BFCL-format per-episode responses (for the checker)
  episode_meta.json    per-episode wall time, steps, forced-quit flags
"""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import bfcl_eval
from bfcl_eval.utils import populate_test_cases_with_predefined_functions

from handler import FAMILY_HANDLERS

DATA = Path(bfcl_eval.__file__).parent / "data"


def load_episodes(category: str, n: int) -> list[dict]:
    entries = []
    with open(DATA / f"BFCL_v4_{category}.json") as fh:
        for line in fh:
            entries.append(json.loads(line))
    entries.sort(key=lambda e: int(e["id"].rsplit("_", 1)[1]))
    entries = entries[:n]
    return populate_test_cases_with_predefined_functions(entries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-hf-id", required=True)
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--backend", choices=["llamacpp", "vllm", "sglang"], required=True)
    ap.add_argument("--family", choices=["qwen", "llama"], default="qwen")
    ap.add_argument("--arm", choices=["on", "off"], required=True)
    ap.add_argument("--category", default="multi_turn_base")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ctx", type=int, default=16384)
    args = ap.parse_args()

    out = Path(args.out_dir) / f"arm_{args.arm}"
    out.mkdir(parents=True, exist_ok=True)

    handler = FAMILY_HANDLERS[args.family](
        model_hf_id=args.model_hf_id,
        base_url=args.base_url,
        served_model_name=args.served_model,
        backend=args.backend,
        cache_arm=(args.arm == "on"),
        raw_log_path=out / "raw_requests.jsonl",
        max_context_length=args.ctx,
    )

    episodes = load_episodes(args.category, args.n_episodes)
    all_results = {}
    all_meta = {}
    for entry in episodes:
        eid = entry["id"]
        handler.current_episode_id = eid
        handler.request_counter = 0
        t0 = time.time()
        try:
            model_responses, metadata = handler.inference_multi_turn_prompting(
                deepcopy(entry), include_input_log=False, exclude_state_log=True
            )
            err = None
        except Exception as exc:  # noqa: BLE001 - record and continue
            model_responses, metadata = None, None
            err = repr(exc)
        wall = time.time() - t0
        all_results[eid] = model_responses
        all_meta[eid] = {
            "wall_s": wall,
            "n_requests": handler.request_counter,
            "error": err,
        }
        print(f"[{args.arm}] {eid}: {wall:.1f}s, {handler.request_counter} requests"
              + (f" ERROR {err}" if err else ""))
        with open(out / "model_results.json", "w") as fh:
            json.dump(all_results, fh, indent=1)
        with open(out / "episode_meta.json", "w") as fh:
            json.dump(all_meta, fh, indent=1)

    print("done:", out)


if __name__ == "__main__":
    main()
