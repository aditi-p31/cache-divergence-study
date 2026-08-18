"""Score a cell's model_results.json with BFCL's own multi-turn checker.

Mirrors bfcl_eval.eval_checker.eval_runner's multi-turn path exactly
(decode per step via handler.decode_execute, drop empty/undecodable steps,
fail on force-termination turn-count mismatch, then multi_turn_checker).

Usage:
  uv run python harness/score_results.py results/smoke/arm_on \
      --model-hf-id Qwen/Qwen2.5-3B-Instruct --category multi_turn_base

Writes scores.json next to model_results.json: {episode_id: {valid, error_type}}
"""

import argparse
import json
from pathlib import Path

import bfcl_eval
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
)
from bfcl_eval.utils import populate_test_cases_with_predefined_functions

DATA = Path(bfcl_eval.__file__).parent / "data"


def load_jsonl_by_id(path: Path) -> dict:
    out = {}
    with open(path) as fh:
        for line in fh:
            e = json.loads(line)
            out[e["id"]] = e
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--model-hf-id", required=True)
    ap.add_argument("--category", default="multi_turn_base")
    ap.add_argument("--family", choices=["qwen", "llama"], default="qwen")
    args = ap.parse_args()

    arm_dir = Path(args.arm_dir)
    results = json.load(open(arm_dir / "model_results.json"))
    entries = load_jsonl_by_id(DATA / f"BFCL_v4_{args.category}.json")
    gts = load_jsonl_by_id(DATA / "possible_answer" / f"BFCL_v4_{args.category}.json")

    # decode_execute needs a handler instance; no server contact happens.
    from handler import FAMILY_HANDLERS

    handler = FAMILY_HANDLERS[args.family](
        model_hf_id=args.model_hf_id,
        base_url="http://127.0.0.1:1/v1",
        served_model_name="unused",
        backend="llamacpp",
        cache_arm=True,
        raw_log_path=arm_dir / "unused_raw.jsonl",
    )

    scores = {}
    for eid, model_result_list in results.items():
        gt_list = gts[eid]["ground_truth"]
        entry = populate_test_cases_with_predefined_functions([dict(entries[eid])])[0]
        if model_result_list is None:
            scores[eid] = {"valid": False, "error_type": "harness:inference_error"}
            continue
        if len(model_result_list) != len(gt_list):
            scores[eid] = {"valid": False, "error_type": "multi_turn:force_terminated"}
            continue
        decoded_all = []
        for turn_results in model_result_list:
            decoded_turn = []
            for step in turn_results:
                try:
                    d = handler.decode_execute(step, has_tool_call_tag=False)
                    if is_empty_execute_response(d):
                        continue
                    decoded_turn.append(d)
                except Exception:
                    continue
            decoded_all.append(decoded_turn)
        try:
            res = multi_turn_checker(
                decoded_all, gt_list, entry, args.category, args.model_hf_id
            )
            scores[eid] = {
                "valid": bool(res["valid"]),
                "error_type": res.get("error_type"),
            }
        except Exception as exc:  # noqa: BLE001
            scores[eid] = {"valid": False, "error_type": f"checker:{exc!r}"}

    with open(arm_dir / "scores.json", "w") as fh:
        json.dump(scores, fh, indent=1)
    n_ok = sum(s["valid"] for s in scores.values())
    print(f"{arm_dir}: {n_ok}/{len(scores)} episodes valid")
    for eid, s in scores.items():
        print(" ", eid, s)


if __name__ == "__main__":
    main()
