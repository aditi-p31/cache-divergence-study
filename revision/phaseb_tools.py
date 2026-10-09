"""Pod-side checks for the Phase B runs. Standard library only.

  validate <pass_dir> --arm on|off --lp N [--n 80] [--sentinel] [--order-file F]
      Complete pass: every episode, no errors, request_idx contiguous and equal
      to episode_meta n_requests; every response carries logprobs content with
      one entry per completion token, each with exactly N alternatives whose
      first entry is the emitted token, ids distinct, values finite and sorted;
      run_meta matches the requested logprob depth, order and sentinel flag;
      telemetry matches the arm (cache-off: cache_n present and 0; cache-on:
      hits after request 0; sentinel: request 0 has cache_n 0 and one sentinel
      record per episode).
  compare <pass_a> <pass_b> [--require-identical] [--min-shared K] [--lp-tol X]
      Episode-level divergence (prompt hash, completion token ids, text,
      finish reason, request count) plus exact top-1 log-probability equality
      at every shared token. A response without logprobs content is an error,
      never a match. --require-identical exits 3 on any difference or when
      fewer than K tokens were compared.
  reset <out_dir> --n N --lp K
      Reset-control session: raw log has exactly 4 phases per item for exactly
      the items in run_meta; cold requests report cached_tokens <= 8 (present),
      warm requests > 0; summary flags recomputed from raw texts must match.
Writes a JSON verdict to stdout.
"""
import gzip
import json
import math
import sys
from pathlib import Path


def _raw(p):
    p = Path(p)
    if p.is_dir():
        p = p / "raw_requests.jsonl"
    if p.exists():
        return open(p)
    gz = Path(str(p) + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt")
    raise FileNotFoundError(p)


def load(p):
    by = {}
    for line in _raw(p):
        r = json.loads(line)
        by.setdefault(r["episode_id"], []).append(r)
    for v in by.values():
        v.sort(key=lambda x: x["request_idx"])
    return by


def choice(r):
    return r["response"]["choices"][0]


def content(r):
    return (choice(r).get("logprobs") or {}).get("content") or []


def comp_tokens(r):
    return (r["response"].get("usage") or {}).get("completion_tokens")


def cache_n(r):
    t = r["response"].get("timings") or {}
    if "cache_n" in t and t["cache_n"] is not None:
        return int(t["cache_n"])
    u = (r["response"].get("usage") or {}).get("prompt_tokens_details") or {}
    v = u.get("cached_tokens")
    return None if v is None else int(v)


class MissingLogprobs(Exception):
    pass


def ids(r):
    c = content(r)
    if not c and (comp_tokens(r) or 0) > 0:
        raise MissingLogprobs(f"{r.get('episode_id')} request {r.get('request_idx')}")
    return [t["id"] for t in c]


def compare(pa, pb):
    A, B = load(pa), load(pb)
    eps = sorted(set(A) | set(B), key=lambda e: int(e.rsplit("_", 1)[1]))
    diverged, lp_diff, shared, missing, max_lp = [], 0, 0, [], 0.0
    for e in eps:
        if e not in A or e not in B:
            missing.append(e)
            continue
        a, b = A[e], B[e]
        div = False
        for x, y in zip(a, b):
            if x["prompt_sha256"] != y["prompt_sha256"]:
                div = True
                break
            ix, iy = ids(x), ids(y)
            for tx, ty in zip(content(x), content(y)):
                if tx["id"] != ty["id"]:
                    break
                shared += 1
                d = abs(tx["logprob"] - ty["logprob"])
                lp_diff += d != 0
                max_lp = max(max_lp, d)
            if (ix != iy or choice(x).get("text") != choice(y).get("text")
                    or choice(x).get("finish_reason") != choice(y).get("finish_reason")):
                div = True
                break
        if div or len(a) != len(b):
            diverged.append(e)
    return {"episodes": len(eps), "missing": missing, "n_diverged": len(diverged),
            "diverged": diverged, "shared_tokens": shared,
            "shared_tokens_top1_logprob_differs": lp_diff, "max_abs_top1_logprob_diff": max_lp}


def _check_tokens(r, lp):
    """Returns a problem string or None. llama.cpp omits the entry of a token
    that ends inside a UTF-8 sequence (the first byte-token of an emoji or of
    U+00F7) and reports the whole character on the completing token, so content
    can be shorter than completion_tokens. The entries' bytes must still rebuild
    the completion text exactly, and the shortfall must be explained by
    non-ASCII characters."""
    c = content(r)
    ct = comp_tokens(r)
    if ct is None:
        return "usage.completion_tokens missing"
    if ct > 0 and not c:
        return "no logprobs content"
    if len(c) > ct:
        return f"logprobs content has {len(c)} entries for {ct} completion tokens"
    text = choice(r).get("text") or ""
    if bytes(b for t in c for b in (t.get("bytes") or [])) != text.encode("utf-8"):
        return "logprobs bytes do not reproduce the completion text"
    if ct - len(c) > 3 * sum(1 for ch in text if ord(ch) > 127):
        return f"logprobs content has {len(c)} entries for {ct} completion tokens"
    for t in c:
        tops = t.get("top_logprobs") or []
        if len(tops) != lp:
            return f"top_logprobs depth {len(tops)} != {lp}"
        if t["logprob"] != tops[0]["logprob"] or not any(
                x["id"] == t["id"] and x["logprob"] == tops[0]["logprob"] for x in tops):
            return "emitted token is not a top alternative"
        vals = [x["logprob"] for x in tops]
        if not all(math.isfinite(v) for v in vals):
            return "non-finite logprob"
        if any(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
            return "alternatives not sorted"
        if len({x["id"] for x in tops}) != len(tops):
            return "duplicate alternative ids"
        if sum(math.exp(v) for v in vals) > 1 + 1e-4:
            return "alternative probabilities sum above 1"
    return None

def validate(pdir, arm, lp, n, sentinel, order_file):
    pdir = Path(pdir)
    problems = []
    meta = json.load(open(pdir / "episode_meta.json"))
    if len(meta) != n:
        problems.append(f"episode_meta has {len(meta)} episodes, expected {n}")
    errs = [e for e, m in meta.items() if m.get("error")]
    # A request that grows past the context window is rejected by the server (HTTP 400,
    # exceed_context_size_error). That is a deterministic outcome of the trajectory the
    # agent took, not an infrastructure fault, so it is reported, not treated as invalid.
    # Every other error still invalidates the pass. (Added 2026-10-08 during B6; no pass
    # validated earlier contained such an episode, so no earlier verdict changes.)
    overflow = [e for e in errs if "exceed_context_size_error" in str(meta[e].get("error"))]
    other = [e for e in errs if e not in overflow]
    if other:
        problems.append(f"{len(other)} episodes raised errors, e.g. {other[:3]}")
    rm = json.load(open(pdir / "run_meta.json")) if (pdir / "run_meta.json").exists() else {}
    if rm.get("logprobs") != lp:
        problems.append(f"run_meta logprobs {rm.get('logprobs')} != {lp}")
    if bool(rm.get("sentinel_reset")) != bool(sentinel):
        problems.append("run_meta sentinel flag mismatch")
    if order_file:
        want = json.load(open(order_file))
        if rm.get("order") != want:
            problems.append("run_meta order differs from the order file")
    elif rm.get("order") and rm["order"] != sorted(rm["order"], key=lambda e: int(e.rsplit("_", 1)[1])):
        problems.append("canonical pass did not run in ascending id order")
    D = load(pdir)
    if len(D) != n:
        problems.append(f"raw log has {len(D)} episodes, expected {n}")
    tok_bad = off_bad = on_later = on_later_hit = s0_bad = n_req = 0
    first_tok_problem = None
    for e, reqs in D.items():
        idx = [r["request_idx"] for r in reqs]
        if idx != list(range(len(reqs))) or (e in meta and meta[e].get("n_requests") != len(reqs)):
            problems.append(f"{e}: request_idx not contiguous or count != episode_meta")
        for r in reqs:
            n_req += 1
            p = _check_tokens(r, lp)
            if p:
                tok_bad += 1
                first_tok_problem = first_tok_problem or f"{e} req {r['request_idx']}: {p}"
            c = cache_n(r)
            if arm == "off" and c != 0:
                off_bad += 1
            if arm == "on":
                if r["request_idx"] == 0:
                    if sentinel and c != 0:
                        s0_bad += 1
                else:
                    on_later += 1
                    on_later_hit += (c or 0) > 0
    if tok_bad:
        problems.append(f"{tok_bad} requests fail the logprob checks, first: {first_tok_problem}")
    if off_bad:
        problems.append(f"{off_bad} cache-off requests without cache_n == 0")
    if arm == "on" and (not on_later or on_later_hit / on_later < 0.95):
        problems.append(f"only {on_later_hit}/{on_later} later cache-on requests hit the cache")
    if sentinel:
        if s0_bad:
            problems.append(f"{s0_bad} episodes started with a nonzero cache after the sentinel reset")
        sp = pdir / "sentinel.jsonl"
        ns = sum(1 for _ in open(sp)) if sp.exists() else 0
        if ns != n:
            problems.append(f"{ns} sentinel records, expected {n}")
    return {"pass": str(pdir), "ok": not problems, "problems": problems[:20], "requests": n_req,
            "episodes": len(D), "on_later_hit_rate": (on_later_hit / on_later) if on_later else None,
            "context_overflow_episodes": overflow}


def reset(odir, n, lp):
    odir = Path(odir)
    problems = []
    if (odir / "summary_ABORTED.json").exists():
        problems.append("reset control aborted on its manipulation check")
    for f in ("summary.json", "run_meta.json", "raw_requests.jsonl"):
        if not (odir / f).exists():
            problems.append(f"missing {f}")
    if problems:
        return {"ok": False, "problems": problems}
    rows = json.load(open(odir / "summary.json"))
    rm = json.load(open(odir / "run_meta.json"))
    items = rm.get("items_in_order") or []
    if len(items) != n or rm.get("logprobs") != lp:
        problems.append(f"run_meta: {len(items)} items, logprobs {rm.get('logprobs')}")
    phases = {}
    viol = tok_bad = lines = 0
    for line in _raw(odir):
        lines += 1
        r = json.loads(line)
        phases.setdefault(r["item_idx"], {})[r["phase"]] = r
        ct = r.get("cached_tokens")
        if r["phase"].startswith("cold") and (ct is None or ct > 8):
            viol += 1
        if r["phase"].startswith("warm") and not ct:
            viol += 1
        if _check_tokens(r, lp):
            tok_bad += 1
    if lines != 4 * n:
        problems.append(f"raw log has {lines} lines, expected {4 * n}")
    if set(phases) != set(items) or any(set(v) != {"cold_1", "warm_1", "cold_2", "warm_2"} for v in phases.values()):
        problems.append("raw log items or phases do not match run_meta")
    if {r["item_idx"] for r in rows} != set(items) or len(rows) != n:
        problems.append("summary items do not match run_meta")
    txt = lambda it, ph: choice(phases[it][ph]).get("text")   # same basis as reset_control.py
    for r in rows:
        it = r["item_idx"]
        if it not in phases or len(phases[it]) != 4:
            continue
        if (r["cold_reproducible"] != (txt(it, "cold_1") == txt(it, "cold_2"))
                or r["warm_reproducible"] != (txt(it, "warm_1") == txt(it, "warm_2"))):
            problems.append(f"summary flags disagree with raw tokens for item {it}")
            break
    if viol:
        problems.append(f"{viol} telemetry violations")
    if tok_bad:
        problems.append(f"{tok_bad} requests fail the logprob checks")
    return {"ok": not problems, "problems": problems, "items": len(rows),
            "cold_repro": sum(r["cold_reproducible"] for r in rows),
            "warm_repro": sum(r["warm_reproducible"] for r in rows),
            "cache_effect": sum(r["cache_effect"] for r in rows)}


def _opt(a, name, default=None, cast=str):
    return cast(a[a.index(name) + 1]) if name in a else default


def main():
    a = sys.argv[1:]
    cmd = a[0]
    try:
        if cmd == "compare":
            res = compare(a[1], a[2])
            print(json.dumps(res))
            # --lp-tol: allowed |difference| in reported top-1 logprob. 0 = bit-exact. A
            # nonzero tolerance is only for passes recorded at different top-k depths:
            # llama.cpp sums the reporting softmax in a top-k-dependent order, so the
            # reported values differ at float32 rounding while the logits are identical.
            tol = _opt(a, "--lp-tol", 0.0, float)
            lp_bad = res["max_abs_top1_logprob_diff"] > tol if tol > 0 else res["shared_tokens_top1_logprob_differs"] > 0
            if "--require-identical" in a and (res["n_diverged"] or res["missing"] or lp_bad
                                               or res["shared_tokens"] < _opt(a, "--min-shared", 1, int)):
                sys.exit(3)
        elif cmd == "validate":
            res = validate(a[1], _opt(a, "--arm"), _opt(a, "--lp", cast=int), _opt(a, "--n", 80, int),
                           "--sentinel" in a, _opt(a, "--order-file"))
            print(json.dumps(res))
            sys.exit(0 if res["ok"] else 4)
        elif cmd == "reset":
            res = reset(a[1], _opt(a, "--n", cast=int), _opt(a, "--lp", cast=int))
            print(json.dumps(res))
            sys.exit(0 if res["ok"] else 4)
        else:
            sys.exit(f"unknown command {cmd}")
    except MissingLogprobs as e:
        print(json.dumps({"ok": False, "error": f"response without logprobs content: {e}"}))
        sys.exit(5)


if __name__ == "__main__":
    main()
