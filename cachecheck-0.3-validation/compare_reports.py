#!/usr/bin/env python3
"""Compare a one-server llm-cachecheck report with a two-servers report.

Both runs must use the same model, the same probes, the same options and
the same cache-on server. One-server mode is a valid substitute for the
two-server setup only if, probe by probe, its salted recompute answer is
identical to the answer of the server with prefix caching off.

Usage:
  python compare_reports.py ONE_SERVER.json TWO_SERVERS.json [more pairs]

Pairs are given as ONE TWO ONE TWO ... Exit status 0 when every pair
agrees on every probe, 1 otherwise.
"""

import json
import sys


def load(path):
    with open(path) as fh:
        return json.load(fh)


def text(answer):
    return (answer.get("content") or "", answer.get("reasoning") or "")


def compare(one_path, two_path):
    one, two = load(one_path), load(two_path)
    problems = []
    if one.get("mode") != "one-server":
        problems.append("%s is not a one-server report" % one_path)
    if two.get("mode") != "two-servers":
        problems.append("%s is not a two-servers report" % two_path)
    for key in ("model", "max_tokens", "seed", "probe_set",
                "probe_file_sha256"):
        if one.get(key) != two.get(key):
            problems.append("%s differs: %r vs %r"
                            % (key, one.get(key), two.get(key)))
    by_one = {p["probe"]: p for p in one["probes"]}
    by_two = {p["probe"]: p for p in two["probes"]}
    shared = sorted(set(by_one) & set(by_two))
    if len(shared) != one["summary"]["probes_planned"] or \
            len(shared) != two["summary"]["probes_planned"]:
        problems.append("probe sets differ or a request failed: %d shared, "
                        "%d and %d planned"
                        % (len(shared), one["summary"]["probes_planned"],
                           two["summary"]["probes_planned"]))
    rows, recompute_same, cache_same, turn1_same = [], 0, 0, 0
    for i in shared:
        a, b = by_one[i], by_two[i]
        r_same = text(a["recompute_path"][0]) == text(b["recompute_path"][0])
        c_same = text(a["cache_path"][0]) == text(b["cache_path"][0])
        t_same = text(a["turn1"][0]) == text(b["turn1"][0])
        recompute_same += r_same
        cache_same += c_same
        turn1_same += t_same
        rows.append((i, a["valid"], b["valid"], t_same, c_same, r_same,
                     a["paths_differ"], b["paths_differ"],
                     a["recompute_cached_tokens"]))
        if not (a["valid"] and b["valid"]):
            problems.append("probe %d not valid in both runs" % i)
    print("%s vs %s" % (one_path, two_path))
    print("probe valid(1,2) turn1= cache= recompute= differs(1,2) "
          "one-server recompute cached tokens")
    for row in rows:
        print("%5d   %d,%d      %-5s %-6s %-10s %d,%d          %s"
              % (row[0], row[1], row[2], row[3], row[4], row[5],
                 row[6], row[7], row[8]))
    n = len(shared)
    print("turn 1 identical: %d of %d" % (turn1_same, n))
    print("cache path identical: %d of %d" % (cache_same, n))
    print("salted recompute identical to the cache-off server: %d of %d"
          % (recompute_same, n))
    print("differ on the cache path: one-server %d, two-servers %d"
          % (one["summary"]["probes_diverged"],
             two["summary"]["probes_diverged"]))
    for p in problems:
        print("PROBLEM: " + p)
    ok = not problems and n > 0 and recompute_same == n \
        and cache_same == n and turn1_same == n
    print("AGREE" if ok else "DO NOT AGREE")
    print()
    return ok


def main(argv):
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    results = [compare(argv[k], argv[k + 1]) for k in range(0, len(argv), 2)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
