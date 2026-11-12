#!/usr/bin/env python3
"""cachecheck: how often does the prompt cache change your server's answers?

Sends short two-turn conversations to an OpenAI-compatible server with
greedy decoding and compares the second answer on two paths:

  cache path      turn 2 reuses the KV state the server wrote while it
                  generated turn 1 (a cache hit)
  recompute path  turn 2 is processed from scratch (no cache reuse)

Each path runs twice. A probe counts only when both paths repeat, the
server's own telemetry confirms that the cache hit covered the turn-1
answer, and the recompute path reused nothing. The method adapts the
paired design of Patodiya, "Same Request, Different Answer: Quantization
Amplifies Cache-Induced Divergence in LLM Serving" (IEEE Access, 2026,
doi:10.1109/ACCESS.2026.3749922; arXiv:2609.04748).

Two modes:

  --mode llamacpp     one llama.cpp server; the cache is switched per
                      request with "cache_prompt" (start it with -np 1)
  --mode two-servers  vLLM, SGLang or another OpenAI-compatible stack:
                      --base-url serves with prefix caching on, and
                      --nocache-url serves the same model with it off
                      (tested on real vLLM 0.33.0 and SGLang 0.5.23
                      servers; results in the README)

Standard library only. Python 3.8 or newer.

Usage (from PyPI: pip install llm-cachecheck, then run llm-cachecheck
in place of python cachecheck.py):
  python cachecheck.py --base-url http://127.0.0.1:8080
  python cachecheck.py --mode two-servers --base-url http://127.0.0.1:8000 \
      --nocache-url http://127.0.0.1:8100 --model Qwen/Qwen2.5-3B-Instruct

Exit status: 0 no difference and at least 80% of probes valid, 1 at
least one valid probe differs, 2 inconclusive.
"""

import argparse
import binascii
import json
import os
import sys
import time
import urllib.error
import urllib.request

__version__ = "0.2.3"

CITATION = (
    "Patodiya, A. Same Request, Different Answer: Quantization Amplifies "
    "Cache-Induced Divergence in LLM Serving. IEEE Access, 2026. "
    "doi:10.1109/ACCESS.2026.3749922"
)

MIN_VALID_SHARE = 0.8

SHARE_URL = ("https://github.com/aditi-p31/cache-divergence-study/issues/new"
             "?template=cachecheck-result.yml")

# Two-turn probes. Turn 1 asks for a worked answer of a few sentences, so
# the server writes a useful amount of KV state while generating it.
# Turn 2 builds on that answer, so it is processed on top of that state.
PROBES = [
    ("A train leaves at 9:40 and arrives at 13:15. How long is the trip? "
     "Show your steps.",
     "If the train had left 25 minutes later and travelled at the same "
     "speed, when would it arrive? Explain briefly."),
    ("List four prime numbers between 50 and 80 and say how you checked "
     "each one.",
     "Add the four numbers and say whether the sum is prime."),
    ("Explain in a short paragraph what a hash table is and why lookups "
     "are usually fast.",
     "Now describe one situation where a hash table lookup becomes slow."),
    ("Write a Python function that returns the second largest number in "
     "a list. Add a short comment on each step.",
     "Rewrite the function so it also handles lists with repeated values "
     "and lists shorter than two."),
    ("A shop sells pens at 3 for 2 dollars. How much do 15 pens cost? "
     "Show the steps.",
     "The shop now offers 10 percent off orders above 8 dollars. What do "
     "15 pens cost now?"),
    ("Summarize the water cycle in four or five sentences.",
     "Which step of the cycle you described is most affected by higher "
     "temperatures, and why?"),
    ("Give three tips for writing clear error messages in software, with "
     "one example each.",
     "Pick the most important tip and rewrite this message with it: "
     "'Error 17: operation failed.'"),
    ("Convert 72 degrees Fahrenheit to Celsius and show the formula.",
     "Now convert your answer to Kelvin and explain the step."),
    ("Describe how binary search works on a sorted list of 100 numbers. "
     "How many steps does it need at most?",
     "What changes if the list has 1000 numbers? Give the new maximum."),
    ("Write a SQL query that returns the five customers with the highest "
     "total order value from tables customers(id, name) and "
     "orders(id, customer_id, amount).",
     "Change the query so it only counts orders from the year 2025, given "
     "a column orders.created_at."),
    ("A rectangle has a perimeter of 36 cm and its length is twice its "
     "width. Find its area. Show the steps.",
     "If both sides grow by 2 cm, by how much does the area grow?"),
    ("Explain the difference between a process and a thread in a few "
     "sentences.",
     "Give one bug that can happen with threads but not with separate "
     "processes, and how to avoid it."),
    ("Plan a three-step morning routine for someone who wants to read "
     "more. Keep each step to one sentence.",
     "Adapt the routine for someone who has only 15 minutes."),
    ("What is 17 times 24? Work it out in two different ways.",
     "Use your result to compute 17 times 25 without multiplying again."),
    ("Write a regular expression that matches dates like 2026-11-01 and "
     "explain each part.",
     "Extend it so it rejects months above 12. Explain the change."),
    ("Explain what a p-value is to someone who has never studied "
     "statistics, in four sentences.",
     "Give one common misreading of a p-value and correct it."),
    ("A recipe for 4 people needs 300 g of rice and 2 onions. Scale it "
     "for 10 people and show the steps.",
     "Round the amounts to what you would actually buy, and say why."),
    ("Describe what a git rebase does, in a short paragraph.",
     "When would you choose merge instead of rebase? Give one reason."),
    ("A loan of 1000 dollars grows by 5 percent each year. What is it "
     "after 3 years? Show each year.",
     "How many whole years until it first exceeds 1500 dollars?"),
    ("List the steps to make a cup of tea, numbered.",
     "Rewrite the steps for someone using a microwave instead of a "
     "kettle."),
]


class RequestError(Exception):
    pass


def http_json(url, body=None, api_key=None, timeout=300):
    """POST body (or GET when body is None) and return the decoded JSON."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode(errors="replace")
        raise RequestError("HTTP %d: %s" % (exc.code, detail))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RequestError(str(exc))


def post_chat(url, messages, args, cache_prompt=None, cache_salt=None):
    """One greedy chat completion, returned as a small dict."""
    body = {"messages": messages, "temperature": 0,
            "max_tokens": args.max_tokens, "seed": args.seed}
    if args.model:
        body["model"] = args.model
    if cache_prompt is not None:
        body["cache_prompt"] = cache_prompt
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    data = http_json(url.rstrip("/") + "/v1/chat/completions", body,
                     args.api_key, args.timeout)
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise RequestError("unexpected response: " + json.dumps(data)[:300])
    usage = data.get("usage") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": (msg.get("reasoning_content") or msg.get("reasoning")
                      or ""),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": cached_tokens(data),
        "system_fingerprint": data.get("system_fingerprint"),
    }


def cached_tokens(data):
    """Prompt tokens served from the cache, if the server reports it."""
    details = (data.get("usage") or {}).get("prompt_tokens_details") or {}
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return details["cached_tokens"]
    timings = data.get("timings") or {}
    if timings.get("cache_n") is not None:
        return timings["cache_n"]
    return None


def first_difference(a, b):
    """Character index where two strings first differ, or None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def same_answer(x, y):
    return x["content"] == y["content"] and x["reasoning"] == y["reasoning"]


def hit_check(turn1, turn2, tolerance):
    """Did the cache-path turn 2 reuse the KV of turn 1 and its answer?

    Returns (ok, note). ok is None when the server gave no telemetry.
    """
    cached = turn2["cached_tokens"]
    p1, c1 = turn1["prompt_tokens"], turn1["completion_tokens"]
    if cached is None or p1 is None or c1 is None:
        return None, "server did not report cached tokens"
    # The last sampled token never had its KV computed, hence the - 1.
    needed = p1 + c1 - 1 - tolerance
    if cached >= needed:
        return True, "cached %d of %d" % (cached, p1 + c1 - 1)
    return False, ("cache hit covered %d tokens, turn 1 and its answer "
                   "need about %d" % (cached, p1 + c1 - 1))


def run_probe(probe, index, args, nonce):
    """Run one probe on both paths, each twice. Returns a result dict."""
    q1, q2 = probe
    turn1 = [{"role": "user", "content": q1}]
    llama = args.mode == "llamacpp"
    nocache_url = args.base_url if llama else args.nocache_url

    # Cache path, twice. Turn 1 is computed fresh each time, so the KV of
    # its answer is written during generation; turn 2 then reuses it. On
    # llama.cpp cache_prompt=false forces the fresh turn 1. On other
    # stacks each iteration gets its own cache_salt, shared by its two
    # turns, so iteration 2 cannot reuse what iteration 1 left behind.
    first1, cache_runs = None, []
    for it in range(2):
        salt = None if llama else "cachecheck-%s-%d-%d" % (nonce, index, it)
        t1 = post_chat(args.base_url, turn1, args,
                       False if llama else None, salt)
        if first1 is None:
            first1 = t1
        turn2 = turn1 + [{"role": "assistant", "content": first1["content"]},
                         {"role": "user", "content": q2}]
        t2 = post_chat(args.base_url, turn2, args,
                       True if llama else None, salt)
        cache_runs.append((t1, t2))

    turn2 = turn1 + [{"role": "assistant", "content": first1["content"]},
                     {"role": "user", "content": q2}]
    nocache_runs = [post_chat(nocache_url, turn2, args,
                              False if llama else None)
                    for _ in range(2)]

    (t1a, w1), (t1b, w2) = cache_runs
    c1, c2 = nocache_runs
    problems = []
    if not same_answer(t1a, t1b):
        problems.append("turn 1 did not repeat")
    if not same_answer(w1, w2):
        problems.append("cache path did not repeat")
    if not same_answer(c1, c2):
        problems.append("recompute path did not repeat")
    if not (t1a["content"] and w1["content"] and c1["content"]):
        problems.append("empty answer (a thinking model? turn thinking off)")

    hits = [hit_check(t1, t2, args.hit_tolerance) for t1, t2 in cache_runs]
    leaks = [c["cached_tokens"] for c in nocache_runs]
    if not args.skip_hit_check:
        for ok, note in hits:
            if ok is False:
                problems.append(note)
        if any(ok is None for ok, _ in hits):
            problems.append("cache hit not confirmed: server did not "
                            "report cached tokens")
        if any(n for n in leaks if n is not None):
            problems.append("recompute path reused %d cached tokens"
                            % max(n for n in leaks if n is not None))
        elif any(n is None for n in leaks):
            problems.append("recompute path not confirmed: server did not "
                            "report cached tokens")
    problems = list(dict.fromkeys(problems))

    differs = not same_answer(w1, c1)
    return {
        "probe": index,
        "valid": not problems,
        "problems": problems,
        "paths_differ": differs,
        "reasoning_differs": w1["reasoning"] != c1["reasoning"],
        "first_difference_char": first_difference(w1["content"],
                                                   c1["content"]),
        "cache_hit": [note for _, note in hits],
        "recompute_cached_tokens": leaks,
        "turn1": [t1a, t1b],
        "cache_path": [w1, w2],
        "recompute_path": [c1, c2],
    }


def server_props(args):
    """llama.cpp /props: slot count and build, or {} if unavailable."""
    try:
        props = http_json(args.base_url.rstrip("/") + "/props", None,
                          args.api_key, 30)
    except RequestError:
        return {}
    return {"total_slots": props.get("total_slots"),
            "build_info": props.get("build_info"),
            "model_file": os.path.basename(props.get("model_path") or "")}


def result_block(args, props, summary, when):
    """A short, copyable summary of the run. It leaves out the server URL
    and any key, so it is safe to paste in public."""
    build = props.get("build_info") or "(fill in engine and version)"
    model = args.model or props.get("model_file") or "(fill in model)"
    lines = [
        "---- llm-cachecheck result ----",
        "tool: llm-cachecheck %s" % __version__,
        "date (UTC): %s" % when,
        "mode: %s" % args.mode,
        "engine build: %s" % build,
        "model: %s" % model,
        "GPU or CPU: (fill in)",
        "probes: %d planned, %d valid, %d differed" % (
            summary["probes_planned"], summary["probes_valid"],
            summary["probes_diverged"]),
        "max tokens: %d, seed: %d, hit check: %s" % (
            args.max_tokens, args.seed, summary["hit_check"]),
        "-------------------------------",
    ]
    return "\n".join(lines)


def status_line(r):
    if not r["valid"]:
        return "inconclusive (%s)" % "; ".join(r["problems"])
    if not r["paths_differ"]:
        return "same"
    if r["first_difference_char"] is None:
        return "DIFFERENT (in the reasoning text)"
    return "DIFFERENT (from character %d)" % r["first_difference_char"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure how often prompt caching changes greedy answers.")
    ap.add_argument("--base-url", required=True,
                    help="server root, e.g. http://127.0.0.1:8080")
    ap.add_argument("--mode", choices=["llamacpp", "two-servers"],
                    default="llamacpp",
                    help="llamacpp (default): one llama.cpp server. "
                         "two-servers: vLLM, SGLang or another OpenAI-"
                         "compatible stack, with a second server that has "
                         "prefix caching off")
    ap.add_argument("--nocache-url",
                    help="two-servers mode: same model, prefix caching off")
    ap.add_argument("--model", default=None,
                    help="model name to send (needed by vLLM and SGLang)")
    ap.add_argument("--n", type=int, default=len(PROBES),
                    help="number of probes, 1 to %d" % len(PROBES))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--hit-tolerance", type=int, default=None,
                    help="tokens the cache hit may fall short by (default "
                         "1 for llamacpp, 32 for two-servers, which covers "
                         "block-sized caches)")
    ap.add_argument("--skip-hit-check", action="store_true",
                    help="for servers that report no cached tokens; the "
                         "cache hit is then assumed, not confirmed")
    ap.add_argument("--api-key", default=os.environ.get("CACHECHECK_API_KEY"))
    ap.add_argument("--out", default="cachecheck_report.json")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)
    if args.mode == "two-servers" and not args.nocache_url:
        ap.error("--mode two-servers needs --nocache-url")
    if not 1 <= args.n <= len(PROBES):
        ap.error("--n must be between 1 and %d" % len(PROBES))
    if args.hit_tolerance is None:
        args.hit_tolerance = 1 if args.mode == "llamacpp" else 32

    props = server_props(args) if args.mode == "llamacpp" else {}
    slots = props.get("total_slots")
    if slots not in (None, 1):
        print("Warning: the server has %d slots. Restart it with -np 1 so "
              "other requests cannot share or evict its cache." % slots)

    nonce = binascii.hexlify(os.urandom(4)).decode()
    probes = PROBES[:args.n]
    results, failures = [], 0
    for i, probe in enumerate(probes, 1):
        try:
            r = run_probe(probe, i, args, nonce)
        except RequestError as exc:
            failures += 1
            print("probe %2d/%d: request failed: %s" % (i, len(probes), exc))
            continue
        results.append(r)
        print("probe %2d/%d: %s" % (i, len(probes), status_line(r)))

    valid = [r for r in results if r["valid"]]
    diverged = [r for r in valid if r["paths_differ"]]
    reasons = {}
    for r in results:
        for p in r["problems"]:
            key = p.split(":")[0] if "cache hit covered" not in p \
                else "cache hit too short"
            reasons[key] = reasons.get(key, 0) + 1
    summary = {"probes_planned": len(probes), "probes_failed": failures,
               "probes_valid": len(valid), "probes_diverged": len(diverged),
               "inconclusive_reasons": reasons,
               "hit_check": "skipped" if args.skip_hit_check else "on"}
    report = {"tool": "llm-cachecheck " + __version__, "mode": args.mode,
              "base_url": args.base_url, "nocache_url": args.nocache_url,
              "model": args.model, "max_tokens": args.max_tokens,
              "seed": args.seed, "server_props": props,
              "summary": summary, "probes": results, "method": CITATION}
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)

    print()
    print("%d of %d valid probes gave a different turn-2 answer on the cache "
          "path." % (len(diverged), len(valid)))
    if len(valid) < len(probes):
        print("%d of %d probes were not valid:" % (len(probes) - len(valid),
                                                    len(probes)))
        if failures:
            print("  request failed: %d" % failures)
        for key, n in sorted(reasons.items()):
            print("  %s: %d" % (key, n))
        if args.mode == "llamacpp":
            print("For a clean check, run llama-server with -np 1 and "
                  "--cache-ram 0, and send it no other traffic.")
        else:
            print("Make sure both servers serve the same model with the same "
                  "settings, report cached tokens, and get no other traffic.")
    if args.skip_hit_check:
        print("Cache hits were assumed, not confirmed (--skip-hit-check).")
    print("Report written to %s" % args.out)
    print()
    print(result_block(args, props, summary,
                       time.strftime("%Y-%m-%d %H:%M", time.gmtime())))
    print("To share this result (optional), open: " + SHARE_URL)
    print("Method: " + CITATION)

    if diverged:
        return 1
    if failures or len(valid) < MIN_VALID_SHARE * len(probes):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
