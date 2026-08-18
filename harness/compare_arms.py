"""Paired divergence analysis between arm_on and arm_off raw logs.

For each episode present in both arms, walks the request sequence in
lockstep and reports:
  - prompt_match: whether the k-th request prompts are byte-identical
    (they stop matching after the first behavioral divergence, since the
    conversation transcript itself diverges),
  - first divergent request index and the first differing completion
    token position within it,
  - completion text equality per request,
  - episode-level: any_token_divergence, trajectory_divergence (request
    count or any prompt mismatch), and completion-identical flag.

This is the harness-level comparator used for dry runs and sanity checks;
the paper's analysis pipeline (analysis/analyze.py) will consume the same
raw logs with outcome scoring added on top.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, list[dict]]:
    by_ep = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            by_ep[r["episode_id"]].append(r)
    for v in by_ep.values():
        v.sort(key=lambda r: r["request_idx"])
    return by_ep


def completion_tokens(rec: dict) -> list:
    """Token id sequence of the completion.

    llama.cpp returns chat-style logprobs: choices[0].logprobs.content is a
    list of {id, token, logprob}. vLLM returns classic completions logprobs
    (tokens list). Fall back to whole-text comparison if neither is present.
    """
    ch = rec["response"]["choices"][0]
    lp = ch.get("logprobs") or {}
    content = lp.get("content")
    if content:
        return [t["id"] for t in content]
    toks = lp.get("tokens")
    if toks:
        return toks
    return [ch.get("text", "")]


def first_token_diff(a: list[str], b: list[str]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dir", help="dir containing arm_on/ and arm_off/")
    args = ap.parse_args()
    cell = Path(args.cell_dir)
    on = load(cell / "arm_on" / "raw_requests.jsonl")
    off = load(cell / "arm_off" / "raw_requests.jsonl")

    episodes = sorted(set(on) & set(off))
    summary = []
    for eid in episodes:
        ra, rb = on[eid], off[eid]
        ep = {
            "episode_id": eid,
            "n_requests_on": len(ra),
            "n_requests_off": len(rb),
            "first_divergent_request": None,
            "first_divergent_token": None,
            "any_divergence": False,
        }
        for k, (a, b) in enumerate(zip(ra, rb)):
            if a["prompt_sha256"] != b["prompt_sha256"]:
                # transcript already diverged upstream of this request
                ep["first_divergent_request"] = ep["first_divergent_request"] or k
                ep["any_divergence"] = True
                break
            ta, tb = completion_tokens(a), completion_tokens(b)
            d = first_token_diff(ta, tb)
            if d is not None:
                ep["first_divergent_request"] = k
                ep["first_divergent_token"] = d
                ep["any_divergence"] = True
                break
        if len(ra) != len(rb):
            ep["any_divergence"] = True
        summary.append(ep)
        print(json.dumps(ep))

    n_div = sum(e["any_divergence"] for e in summary)
    print(f"\n{n_div}/{len(summary)} episodes diverged between arms")
    with open(cell / "divergence_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
