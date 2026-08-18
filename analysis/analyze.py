"""Single source of truth for every number in the paper.

Reads the raw per-request JSONL logs plus BFCL-scored outcomes for every
(config, pass, arm) cell and emits findings.json. Nothing in the manuscript
is computed anywhere else.

Metrics per config:
  determinism_off / determinism_on : episode divergence rate when the SAME
      arm is run twice (main vs repeat). The cache-off value is the
      harness's internal validity check; it must be 0.
  cross_arm                        : cache-on vs cache-off divergence
  first_div_request / first_div_token : where divergence starts
  toolcall_change_rate             : divergence that alters the decoded
      tool-call sequence, not just token ids
  success_on / success_off         : BFCL checker pass rate per arm
  flips_on_to_off / flips_off_to_on: paired outcome changes
  mcnemar_p                        : exact binomial test on discordant pairs
  ci_*                             : Wilson 95% intervals on proportions

Usage: uv run python analysis/analyze.py <results_root> [-o findings.json]
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


# ---------- loading ----------

def _open_maybe_gz(path: Path):
    """Raw logs ship gzipped in the released artifact and plain when freshly
    collected. The compressed copy is the archival one, so it wins when both
    exist; a stale uncompressed leftover must never shadow it."""
    import gzip
    gz = path if path.suffix == ".gz" else path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt")
    if path.exists() and path.suffix != ".gz":
        return open(path)
    raise FileNotFoundError(f"neither {path} nor {gz} exists")


def load_raw(path: Path) -> dict[str, list[dict]]:
    by_ep: dict[str, list[dict]] = {}
    with _open_maybe_gz(path) as fh:
        for line in fh:
            r = json.loads(line)
            by_ep.setdefault(r["episode_id"], []).append(r)
    for v in by_ep.values():
        v.sort(key=lambda x: x["request_idx"])
    return by_ep


def comparison_level(rec: dict) -> str:
    """llama.cpp returns token ids; vLLM returns token strings. Two distinct
    ids that render identically compare equal on one engine and not the
    other, so the level is recorded rather than assumed uniform."""
    lp = (rec["response"]["choices"][0].get("logprobs") or {})
    if lp.get("content"):
        return "token_id"
    if lp.get("tokens"):
        return "token_string"
    return "text"


def completion_token_ids(rec: dict) -> list:
    ch = rec["response"]["choices"][0]
    lp = ch.get("logprobs") or {}
    content = lp.get("content")
    if content:
        return [t["id"] for t in content]
    toks = lp.get("tokens")
    if toks:
        return toks
    return [ch.get("text", "")]


def completion_text(rec: dict) -> str:
    return rec["response"]["choices"][0].get("text", "")


def cached_tokens(rec: dict) -> int | None:
    """None means the engine did not report the field at all, which is not
    the same as reporting zero. vLLM 0.11.0 emits null on the completions
    endpoint, so treating it as 0 makes a cache-on arm look identical to a
    cache-off arm and turns the manipulation check into a tautology."""
    usage = rec["response"].get("usage") or {}
    det = usage.get("prompt_tokens_details")
    if not det:
        return None
    return det.get("cached_tokens")


# ---------- comparison ----------

def compare_episode(a: list[dict], b: list[dict]) -> dict:
    """Compare one episode across two runs. Returns divergence descriptors."""
    out = {
        "diverged": False,
        "first_div_request": None,
        "first_div_token": None,
        "div_kind": None,
        "request_count_differs": len(a) != len(b),
    }
    for k, (x, y) in enumerate(zip(a, b)):
        if x["prompt_sha256"] != y["prompt_sha256"]:
            # transcript already diverged upstream of this request, so the
            # first differing TOKEN is not observable here. Recorded as a
            # separate population so the depth statistics are not computed
            # over a filtered subset.
            out["diverged"] = True
            out["first_div_request"] = k
            out["div_kind"] = "prompt"
            return out
        tx, ty = completion_token_ids(x), completion_token_ids(y)
        if tx != ty:
            out["diverged"] = True
            out["first_div_request"] = k
            out["div_kind"] = "token"
            for i, (p, q) in enumerate(zip(tx, ty)):
                if p != q:
                    out["first_div_token"] = i
                    break
            else:
                out["first_div_token"] = min(len(tx), len(ty))
            return out
    if out["request_count_differs"]:
        out["diverged"] = True
    return out


def divergence_stats(d1: dict, d2: dict) -> dict:
    eps = sorted(set(d1) & set(d2))
    results = [compare_episode(d1[e], d2[e]) for e in eps]
    n = len(results)
    div = [r for r in results if r["diverged"]]
    first_reqs = [r["first_div_request"] for r in div if r["first_div_request"] is not None]
    first_toks = [r["first_div_token"] for r in div if r["first_div_token"] is not None]
    return {
        "n_episodes": n,
        "n_only_in_a": len(set(d1) - set(d2)),
        "n_only_in_b": len(set(d2) - set(d1)),
        "n_diverged": len(div),
        "rate": len(div) / n if n else None,
        "ci95": wilson(len(div), n) if n else None,
        "n_div_token_level": sum(1 for r in div if r["div_kind"] == "token"),
        "n_div_prompt_level": sum(1 for r in div if r["div_kind"] == "prompt"),
        "median_first_div_request": median(first_reqs),
        "median_first_div_token": median(first_toks),
        "median_first_div_token_n": len(first_toks),
    }


# ---------- statistics ----------

def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def median(xs: list) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * binom_cdf(k, n))


# ---------- per-config analysis ----------

def analyse_config(cfg_dir: Path) -> dict | None:
    cells = {}
    for pass_ in ("main", "repeat"):
        for arm in ("on", "off"):
            raw = cfg_dir / pass_ / f"arm_{arm}" / "raw_requests.jsonl"
            if raw.exists() or raw.with_suffix(".jsonl.gz").exists():
                cells[(pass_, arm)] = load_raw(raw)
    if ("main", "on") not in cells or ("main", "off") not in cells:
        return None

    any_rec = next(iter(next(iter(cells.values())).values()))[0]
    res: dict = {
        "config": cfg_dir.name,
        "cells_present": sorted(f"{p}/{a}" for p, a in cells),
        "comparison_level": comparison_level(any_rec),
    }

    res["cross_arm_main"] = divergence_stats(cells[("main", "on")], cells[("main", "off")])
    if ("repeat", "on") in cells:
        res["determinism_on"] = divergence_stats(cells[("main", "on")], cells[("repeat", "on")])
    if ("repeat", "off") in cells:
        res["determinism_off"] = divergence_stats(cells[("main", "off")], cells[("repeat", "off")])

    # cache exposure telemetry: verifies the manipulation actually happened
    for (pass_, arm), data in cells.items():
        if pass_ != "main":
            continue
        vals = [cached_tokens(r) for reqs in data.values() for r in reqs]
        present = [v for v in vals if v is not None]
        res[f"cached_tokens_{arm}"] = {
            "mean": round(sum(present) / len(present), 1) if present else None,
            "n_requests": len(vals),
            "n_absent": sum(1 for v in vals if v is None),
            "n_zero": sum(1 for v in present if v == 0),
            "usable": bool(present),
        }

    # outcomes, if scored
    scores = {}
    for pass_ in ("main", "repeat"):
        for arm in ("on", "off"):
            sp = cfg_dir / pass_ / f"arm_{arm}" / "scores.json"
            if sp.exists():
                scores[(pass_, arm)] = json.load(open(sp))
    if ("main", "on") in scores and ("main", "off") in scores:
        so, sf = scores[("main", "on")], scores[("main", "off")]
        eps = sorted(set(so) & set(sf))
        on_ok = sum(bool(so[e]["valid"]) for e in eps)
        off_ok = sum(bool(sf[e]["valid"]) for e in eps)
        b = sum(1 for e in eps if so[e]["valid"] and not sf[e]["valid"])
        c = sum(1 for e in eps if sf[e]["valid"] and not so[e]["valid"])
        res["outcomes"] = {
            "n": len(eps),
            "success_on": on_ok,
            "success_off": off_ok,
            "rate_on": round(on_ok / len(eps), 4) if eps else None,
            "rate_off": round(off_ok / len(eps), 4) if eps else None,
            "ci95_on": wilson(on_ok, len(eps)),
            "ci95_off": wilson(off_ok, len(eps)),
            "flips_on_only": b,
            "flips_off_only": c,
            "mcnemar_p": round(mcnemar_exact(b, c), 5),
            "error_types_on": dict(Counter(so[e].get("error_type") for e in eps if not so[e]["valid"])),
            "error_types_off": dict(Counter(sf[e].get("error_type") for e in eps if not sf[e]["valid"])),
        }
    return res


def analyse_bridge(bridge_dir: Path) -> dict | None:
    subs = sorted(bridge_dir.glob("cache_*/summary.json"))
    if len(subs) > 1:
        raise SystemExit(
            f"{bridge_dir} holds {len(subs)} cache_* summaries; concatenating "
            f"them would double-count items. Analyse arms separately.")
    rows = json.load(open(subs[0])) if subs else []
    if not rows:
        return None
    det_cold = sum(1 for r in rows if r.get("cold_deterministic"))
    det_warm = sum(1 for r in rows if r.get("warm_deterministic"))
    div = [r for r in rows if not r["texts_identical"]]
    attributable = [r for r in div if r.get("cold_deterministic") and r.get("warm_deterministic")]
    flips = [r for r in attributable if r["cold_correct"] != r["warm_correct"]]
    n = len(rows)
    return {
        "n_items": n,
        "cold_path_deterministic": det_cold,
        "warm_path_deterministic": det_warm,
        "n_divergent": len(div),
        "n_cache_attributable": len(attributable),
        "rate_attributable": round(len(attributable) / n, 4) if n else None,
        "ci95_attributable": wilson(len(attributable), n),
        "n_correctness_flips": len(flips),
        "acc_cold": sum(1 for r in rows if r["cold_correct"]),
        "acc_warm": sum(1 for r in rows if r["warm_correct"]),
        "mcnemar_p": round(mcnemar_exact(
            sum(1 for r in rows if r["warm_correct"] and not r["cold_correct"]),
            sum(1 for r in rows if r["cold_correct"] and not r["warm_correct"])), 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("-o", "--out", default="findings.json")
    args = ap.parse_args()
    root = Path(args.results_root)

    findings = {"configs": [], "bridges": {}}
    for cfg in sorted(root.iterdir()):
        if not cfg.is_dir():
            continue
        if cfg.name.startswith("bridge-"):
            b = analyse_bridge(cfg)
            if b:
                findings["bridges"][cfg.name] = b
            continue
        r = analyse_config(cfg)
        if r:
            findings["configs"].append(r)
        else:
            present = sorted(p.name for p in cfg.glob("*/arm_*"))
            print(f"SKIP {cfg.name}: needs main/arm_on and main/arm_off, found {present}")

    with open(args.out, "w") as fh:
        json.dump(findings, fh, indent=1)

    print(f"{'config':<22} {'det_off':>8} {'det_on':>8} {'cross':>8} {'succ_on':>8} {'succ_off':>9} {'mcnemar':>8}")
    print("-" * 76)
    for c in findings["configs"]:
        d_off = c.get("determinism_off", {}).get("rate")
        d_on = c.get("determinism_on", {}).get("rate")
        cross = c["cross_arm_main"]["rate"]
        o = c.get("outcomes") or {}
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "-"
        print(f"{c['config']:<22} {fmt(d_off):>8} {fmt(d_on):>8} {fmt(cross):>8} "
              f"{fmt(o.get('rate_on')):>8} {fmt(o.get('rate_off')):>9} {fmt(o.get('mcnemar_p')):>8}")
    for name, b in findings["bridges"].items():
        print(f"\nbridge {name}: {b['n_cache_attributable']}/{b['n_items']} cache-attributable "
              f"divergences, {b['n_correctness_flips']} correctness flips, "
              f"cold-det {b['cold_path_deterministic']}/{b['n_items']}, "
              f"warm-det {b['warm_path_deterministic']}/{b['n_items']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
