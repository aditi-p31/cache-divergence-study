"""Cochran-Armitage trend test for the quantization gradient (Section V-D).

Tests for a linear trend in cross-arm divergence across ordered weight
formats, on the four Qwen2.5-7B llama.cpp cells of Table 1. Scores are
equally spaced over the format ordering F16 < Q8_0 < Q4_K_M < Q3_K_M.

Reads findings.json; writes trend_test.json.

Usage:
  uv run python analysis/trend_test.py analysis/findings.json -o analysis/trend_test.json
"""

import argparse
import json
import math

ORDER = ["f16", "q80", "q4km", "q3km"]


def cochran_armitage(events, totals, scores):
    """Two-sided Cochran-Armitage test for trend in proportions."""
    N = sum(totals)
    R = sum(events)
    pbar = R / N
    sbar = sum(s * n for s, n in zip(scores, totals)) / N
    num = sum(s * e for s, e in zip(scores, events)) - pbar * sum(
        s * n for s, n in zip(scores, totals))
    var = pbar * (1 - pbar) * sum(
        n * (s - sbar) ** 2 for s, n in zip(scores, totals))
    z = num / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("findings")
    ap.add_argument("-o", "--out", default="analysis/trend_test.json")
    args = ap.parse_args()
    f = json.load(open(args.findings))

    cells = {}
    for c in f["configs"]:
        parts = c["config"].split("-")
        if parts[0] == "lcpp" and parts[1] == "qwen7b" and parts[2] in ORDER:
            cross = c["cross_arm_main"]
            n = cross["n_episodes"]
            cells[parts[2]] = [round(cross["rate"] * n), n]

    missing = [q for q in ORDER if q not in cells]
    if missing:
        raise SystemExit(f"missing cells: {missing}")

    events = [cells[q][0] for q in ORDER]
    totals = [cells[q][1] for q in ORDER]
    scores = list(range(len(ORDER)))
    z, p = cochran_armitage(events, totals, scores)

    out = {"z": round(z, 2), "p": p, "scores": scores,
           "cells": {q: cells[q] for q in ORDER}}
    json.dump(out, open(args.out, "w"), indent=1)
    for q in ORDER:
        print(f"  {q:<6} {cells[q][0]}/{cells[q][1]}")
    print(f"  Cochran-Armitage z = {z:.2f}, two-sided p = {p:.3g}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
