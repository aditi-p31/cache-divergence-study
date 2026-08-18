"""Re-derive GSM8K answers from stored raw text with a robust extractor.

The collection-time extractor accepted only the "#### <number>" form that the
system prompt requested. Models frequently comply in a different but
unambiguous way, most often \\boxed{...}, and those responses were scored
incorrect even when the value was right. The rate is high enough (11 to 38
percent depending on configuration) to distort every accuracy figure derived
from the bridge, so answers are re-derived here from the raw responses.

This changes no model output. It re-reads the text that was already logged
and applies a wider set of answer patterns, in priority order, so the
correction is auditable and requires no additional inference.

Usage:
  uv run python analysis/reextract_bridge.py results-pod/results -o rescored/
"""

import argparse
import gzip
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

# Priority order matters: the explicitly requested form wins, then common
# unambiguous alternatives. No bare "last number in the text" fallback, which
# would silently pick up intermediate arithmetic.
PATTERNS = [
    re.compile(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)"),
    re.compile(r"\\boxed\{\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\}"),
    re.compile(r"\\boxed\{\\text\{\s*\$?(-?[\d,]+(?:\.\d+)?)[^}]*\}\}"),
    re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*\$?(-?[\d,]+(?:\.\d+)?)", re.I),
    re.compile(r"\\boxed\{\s*\\?\$\s*(-?[\d,]+(?:\.\d+)?)"),
    re.compile(r"\\boxed\{\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\\text"),
    re.compile(r"\*\*\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\*\*\s*[\.\)]?\s*$", re.M),
]


def numeric(s):
    """Compare answers as numbers. String equality scores 57.00 as wrong
    against a gold of 57, which is the same formatting-not-correctness
    mistake this script exists to correct."""
    if s is None:
        return None
    try:
        return Decimal(s.replace(",", "").rstrip(".").strip())
    except (InvalidOperation, AttributeError):
        return None


def normalise(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").rstrip(".").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s or None


def extract(text: str) -> tuple[str | None, str | None]:
    """Return (answer, which_pattern) using the highest-priority match."""
    for i, pat in enumerate(PATTERNS):
        m = pat.findall(text)
        if m:
            return normalise(m[-1]), f"p{i}"
    return None, None


def open_maybe_gz(p: Path):
    if p.exists():
        return open(p)
    gz = p.with_suffix(p.suffix + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt")
    raise FileNotFoundError(p)


def rescore(bridge_dir: Path) -> dict | None:
    raw = bridge_dir / "cache_on" / "raw_requests.jsonl"
    summ = bridge_dir / "cache_on" / "summary.json"
    if not summ.exists():
        return None
    try:
        fh = open_maybe_gz(raw)
    except FileNotFoundError:
        return None

    # gold answers come from the original summary, which recorded them
    gold = {r["item_idx"]: normalise(r["gold"]) for r in json.load(open(summ))}

    texts: dict[tuple[int, str], str] = {}
    finish: dict[tuple[int, str], str] = {}
    with fh:
        for line in fh:
            r = json.loads(line)
            ch = r["response"]["choices"][0]
            texts[(r["item_idx"], r["pass"])] = ch["text"]
            finish[(r["item_idx"], r["pass"])] = ch.get("finish_reason")

    rows = []
    pattern_use: dict[str, int] = {}
    for idx in sorted({i for i, _ in texts}):
        got, finishes = {}, {}
        for p in ("cold", "cold2", "warm", "warm2"):
            finishes[p] = finish.get((idx, p))
            t = texts.get((idx, p))
            if t is None:
                continue
            a, which = extract(t)
            got[p] = a
            if which:
                pattern_use[which] = pattern_use.get(which, 0) + 1
        if "cold" not in got or "warm" not in got:
            continue
        g = gold.get(idx)
        rows.append({
            "item_idx": idx,
            "gold": g,
            "cold_answer": got["cold"],
            "warm_answer": got["warm"],
            "cold_correct": numeric(got["cold"]) is not None
                            and numeric(got["cold"]) == numeric(g),
            "warm_correct": numeric(got["warm"]) is not None
                            and numeric(got["warm"]) == numeric(g),
            "cold_truncated": finishes.get("cold") == "length",
            "warm_truncated": finishes.get("warm") == "length",
            "cold_unparsed": got["cold"] is None,
            "warm_unparsed": got["warm"] is None,
            "texts_identical": texts.get((idx, "cold")) == texts.get((idx, "warm")),
            "cold_deterministic": texts.get((idx, "cold")) == texts.get((idx, "cold2")),
            "warm_deterministic": texts.get((idx, "warm")) == texts.get((idx, "warm2")),
        })
    return {"rows": rows, "pattern_use": pattern_use}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("-o", "--out", default=None,
                    help="if given, write rescored summary.json into each bridge dir under this root")
    args = ap.parse_args()
    root = Path(args.results_root)

    print(f"{'bridge':<28} {'n':>5} {'unparsed old':>13} {'unparsed new':>13} "
          f"{'acc cold':>9} {'acc warm':>9}")
    print("-" * 82)
    for d in sorted(root.glob("bridge-*")):
        old = json.load(open(d / "cache_on" / "summary.json"))
        res = rescore(d)
        if not res:
            continue
        rows = res["rows"]
        n = len(rows)
        old_unp = sum(1 for r in old if r["cold_answer"] is None)
        new_unp = sum(1 for r in rows if r["cold_unparsed"])
        ac = sum(1 for r in rows if r["cold_correct"])
        aw = sum(1 for r in rows if r["warm_correct"])
        print(f"{d.name:<28} {n:>5} {old_unp:>13} {new_unp:>13} {ac:>9} {aw:>9}")
        if args.out:
            outdir = Path(args.out) / d.name / "cache_on"
            outdir.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(outdir / "summary.json", "w"), indent=1)
    if args.out:
        print(f"\nrescored summaries written under {args.out}")


if __name__ == "__main__":
    main()
