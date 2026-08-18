"""Manipulation check: prove the cache arm was what we claim it was.

Different engines expose cache exposure differently, so we take each at its
word in its own idiom and record the result next to the data:

  llama.cpp  per-request usage.prompt_tokens_details.cached_tokens in the
             response body (already captured in raw_requests.jsonl)
  vLLM       "Prefix cache hit rate: N%" in the engine log; vLLM 0.11.0
             does not populate cached_tokens on the completions endpoint,
             so the response body cannot be used
  SGLang     "cache hit rate" in the engine log

Usage:
  uv run python harness/extract_cache_telemetry.py /workspace \
      -o cache_telemetry.json
"""

import argparse
import json
import re
from pathlib import Path

VLLM_RATE = re.compile(r"Prefix cache hit rate:\s*([\d.]+)%")
SGLANG_RATE = re.compile(r"cache hit rate:\s*([\d.]+)%", re.IGNORECASE)


def scan_log(path: Path) -> dict | None:
    rates = []
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    for pat in (VLLM_RATE, SGLANG_RATE):
        rates.extend(float(m) for m in pat.findall(text))
    if not rates:
        return None
    return {
        "n_observations": len(rates),
        "min_pct": min(rates),
        "max_pct": max(rates),
        "final_pct": rates[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory holding srv_*.log files")
    ap.add_argument("-o", "--out", default="cache_telemetry.json")
    args = ap.parse_args()

    out = {}
    for log in sorted(Path(args.root).glob("srv_*.log")):
        stem = log.stem[len("srv_"):]
        arm = "on"
        if stem.endswith("_off"):
            arm, stem = "off", stem[:-4]
        elif stem.endswith("_on"):
            arm, stem = "on", stem[:-3]
        stats = scan_log(log)
        if stats:
            out.setdefault(stem, {})[arm] = stats

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"{'config':<28} {'arm':>4} {'obs':>5} {'min%':>7} {'max%':>7} {'final%':>7}")
    print("-" * 64)
    for cfg, arms in sorted(out.items()):
        for arm, s in sorted(arms.items()):
            print(f"{cfg:<28} {arm:>4} {s['n_observations']:>5} "
                  f"{s['min_pct']:>7.1f} {s['max_pct']:>7.1f} {s['final_pct']:>7.1f}")
    print(f"\nwrote {args.out}")
    print("Expected pattern: cache-on arms show a nonzero hit rate, "
          "cache-off arms show 0 or no observations at all.")


if __name__ == "__main__":
    main()
