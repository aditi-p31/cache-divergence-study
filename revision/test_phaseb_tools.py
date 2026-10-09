"""Known-answer and failure-mode tests for phaseb_tools.py, on real August
passes and synthetic records. Run from the repository root:
  project:  python revision/phaseB/test_phaseb_tools.py
  artifact: python revision/test_phaseb_tools.py
(the artifact keeps the original grid in results/ and ships raw logs gzipped; the
data() helper maps paths and hands the validator plain copies, so the validator
tested is byte-identical to the one that ran on the pod)."""
import gzip, json, shutil, subprocess, sys, tempfile
from pathlib import Path

T = str(Path(__file__).resolve().parent / "phaseb_tools.py"); PY = sys.executable
GRID = "results-pod/results" if Path("results-pod/results").exists() else "results"
_PLAIN = Path(tempfile.mkdtemp())


def data(p):
    """A pass directory holding a plain raw_requests.jsonl."""
    p = p.replace("results-pod/results", GRID)
    if (Path(p) / "raw_requests.jsonl").exists():
        return p
    out = _PLAIN / p.replace("/", "_")
    if not out.exists():
        shutil.copytree(p, out)
        for gz in out.glob("*.gz"):
            with gzip.open(gz, "rb") as fi, open(gz.with_suffix(""), "wb") as fo:
                fo.write(fi.read())
            gz.unlink()
    return str(out)
ok = True


def run(*a):
    p = subprocess.run([PY, T, *a], capture_output=True, text=True)
    try:
        return p.returncode, json.loads(p.stdout)
    except Exception:
        return p.returncode, p.stdout + p.stderr


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)
    ok &= bool(cond)


def rec(e, k, toks, text=None, logprobs=True, ct=None):
    c = [{"id": i, "token": s, "bytes": list(s.encode()), "logprob": lp,
          "top_logprobs": [{"id": i, "token": s, "bytes": list(s.encode()), "logprob": lp}]} for i, lp, s in toks]
    text = text if text is not None else "".join(s for _, _, s in toks)
    return {"episode_id": e, "request_idx": k, "prompt_sha256": f"{e}{k}", "latency_s": .1,
            "response": {"created": 1, "usage": {"completion_tokens": ct if ct is not None else len(toks)},
                         "timings": {"cache_n": 0},
                         "choices": [{"text": text, "finish_reason": "stop", "logprobs": ({"content": c} if logprobs else None)}]}}


def write(d, recs, meta=True):
    d.mkdir(parents=True)
    open(d / "raw_requests.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in recs))
    eps = sorted({x["episode_id"] for x in recs}, key=lambda v: int(v.rsplit("_", 1)[1]))
    json.dump({e: {"n_requests": sum(1 for x in recs if x["episode_id"] == e), "error": None} for e in eps},
              open(d / "episode_meta.json", "w"))
    if meta:
        json.dump({"order": eps, "logprobs": 1, "sentinel_reset": False}, open(d / "run_meta.json", "w"))


def august(src, tmp, arm_meta=True):
    d = tmp / Path(src).name; shutil.copytree(src, d)
    order = sorted({json.loads(l)["episode_id"] for l in open(d / "raw_requests.jsonl")}, key=lambda x: int(x.rsplit("_", 1)[1]))
    json.dump({"order": order, "logprobs": 1, "sentinel_reset": False}, open(d / "run_meta.json", "w"))
    return d


tmp = Path(tempfile.mkdtemp())
try:
    rc, r = run("compare", data("results-repair/repair/lcpp-orderA/main/arm_on"), data("results-repair/repair/lcpp-orderB/main/arm_on"), "--require-identical", "--min-shared", "20000")
    check("identical August passes pass an exact gate", rc == 0 and r["shared_tokens"] == 22355, (rc, r))
    rc, r = run("compare", data("results-pod/results/lcpp-qwen7b-q4km/main/arm_on"), data("results-pod/results/lcpp-qwen7b-q4km/main/arm_off"))
    check("grid Q4 on vs off = 60 (published)", r["n_diverged"] == 60, r)
    rc, r = run("compare", data("results-pod/results/lcpp-qwen7b-q4km/main/arm_on"), data("results-pod/results/lcpp-qwen7b-q4km/repeat/arm_on"))
    check("grid Q4 on rerun = 62 (published)", r["n_diverged"] == 62, r)
    rc, r = run("compare", data("results-repair/repair/lcpp-orderA/main/arm_on"), data("results-repair/repair/lcpp-orderB/main/arm_on"), "--require-identical", "--min-shared", "99999")
    check("min-shared floor enforced", rc == 3, rc)
    e = "multi_turn_base_0"
    write(tmp / "a", [rec(e, 0, [(1, -.1, "a"), (2, -.2, "b")]), rec(e, 1, [(3, -.3, "c"), (4, -.4, "d"), (5, -.5, "e")]), rec(e, 2, [(1, -.1, "a"), (2, -.2, "b"), (3, -.3, "c")])])
    write(tmp / "b", [rec(e, 0, [(1, -.1, "a"), (2, -.2, "b")]), rec(e, 1, [(3, -.3, "c"), (4, -.45, "d"), (5, -.5, "e")]), rec(e, 2, [(1, -.1, "a"), (2, -.2, "b"), (9, -.3, "z")]), rec(e, 3, [(1, -.1, "a")])])
    rc, r = run("compare", str(tmp / "a"), str(tmp / "b"))
    check("length mismatch keeps the full shared-token count", r["shared_tokens"] == 7 and r["shared_tokens_top1_logprob_differs"] == 1, r)
    write(tmp / "c", [rec(e, 0, [(1, -.1, "a")], logprobs=False)]); write(tmp / "d", [rec(e, 0, [(1, -.1, "a")], text="x", logprobs=False)])
    rc, r = run("compare", str(tmp / "c"), str(tmp / "d"), "--require-identical")
    check("responses without logprobs are an error", rc == 5, (rc, r))
    write(tmp / "e1", [rec(e, 0, [(1, -.1, "a")], text="abc")]); write(tmp / "f1", [rec(e, 0, [(1, -.1, "a")], text="xyz")])
    rc, r = run("compare", str(tmp / "e1"), str(tmp / "f1"), "--require-identical")
    check("differing text with equal ids is divergence", rc == 3, (rc, r))
    write(tmp / "g", [rec(e, 0, [(1, -.1000001, "a")])]); write(tmp / "h", [rec(e, 0, [(1, -.1, "a")])])
    rc, r = run("compare", str(tmp / "g"), str(tmp / "h"), "--require-identical", "--lp-tol", "1e-3")
    check("cross-depth tolerance accepts float-rounding differences", rc == 0, (rc, r))
    rc, r = run("compare", str(tmp / "g"), str(tmp / "h"), "--require-identical")
    check("exact mode still rejects them", rc == 3, (rc, r))
    # validate on genuine August passes, including F16/Q8 ones with merged UTF-8 entries
    for src, arm in [("results-pod/results/lcpp-qwen7b-f16/main/arm_off", "off"), ("results-pod/results/lcpp-qwen7b-q80/main/arm_on", "on"),
                     ("results-repair/quantgrad/f16/main/arm_on", "on"), ("results-repair/repair/lcpp-orderB/main/arm_off", "off")]:
        d = august(data(src), tmp / ("v_" + src.replace("/", "_")))
        rc, r = run("validate", str(d), "--arm", arm, "--lp", "1")
        check(f"genuine August pass accepted: {src}", rc == 0, r.get("problems") if isinstance(r, dict) else r)
    d = august(data("results-repair/repair/lcpp-orderB/main/arm_on"), tmp / "wrongdepth")
    rc, _ = run("validate", str(d), "--arm", "on", "--lp", "5")
    check("wrong logprob depth rejected", rc == 4, rc)
    lines = open(d / "raw_requests.jsonl").read().splitlines(); x = json.loads(lines[5]); x["response"]["choices"][0]["logprobs"] = None
    lines[5] = json.dumps(x); open(d / "raw_requests.jsonl", "w").write("\n".join(lines) + "\n")
    rc, _ = run("validate", str(d), "--arm", "on", "--lp", "1")
    check("stripped logprobs rejected", rc == 4, rc)
    # merged UTF-8 entry: 2 completion tokens, one content entry carrying the whole character
    write(tmp / "utf", [rec(e, 0, [(7, -.2, "÷")], ct=2)])
    rc, r = run("validate", str(tmp / "utf"), "--arm", "off", "--lp", "1", "--n", "1")
    check("merged multi-byte entry accepted", rc == 0, r)
    bad = rec(e, 0, [(7, -.2, "÷")], ct=2); bad["response"]["choices"][0]["text"] = "x"
    write(tmp / "utfbad", [bad])
    rc, r = run("validate", str(tmp / "utfbad"), "--arm", "off", "--lp", "1", "--n", "1")
    check("bytes that do not rebuild the text rejected", rc == 4, r)
    write(tmp / "short", [rec(e, 0, [(7, -.2, "a")], ct=3)])
    rc, r = run("validate", str(tmp / "short"), "--arm", "off", "--lp", "1", "--n", "1")
    check("unexplained token shortfall on ASCII text rejected", rc == 4, r)
    # context overflow is a trajectory outcome; any other error still invalidates the pass
    for name, err, want in (("ovf", "BadRequestError(... 'type': 'exceed_context_size_error' ...)", 0),
                            ("oth", "APIConnectionError('Connection error.')", 4)):
        write(tmp / name, [rec(e, 0, [(1, -.1, "a")])])
        m = json.load(open(tmp / name / "episode_meta.json")); m[e]["error"] = err
        json.dump(m, open(tmp / name / "episode_meta.json", "w"))
        rc, r = run("validate", str(tmp / name), "--arm", "off", "--lp", "1", "--n", "1")
        check(f"error classification: {name}", rc == want and (name != "ovf" or r.get("context_overflow_episodes") == [e]), (rc, r))
    # reset control
    rs = tmp / "reset"; shutil.copytree(data("results-repair/repair/reset-lcpp"), rs)
    json.dump({"items_in_order": list(range(40)), "item_seed": None, "logprobs": 1}, open(rs / "run_meta.json", "w"))
    rc, r = run("reset", str(rs), "--n", "40", "--lp", "1")
    check("August reset session accepted (40/40/40, 14)", rc == 0 and r["cache_effect"] == 14, r)
    lines = open(rs / "raw_requests.jsonl").read().splitlines(); open(rs / "raw_requests.jsonl", "w").write("\n".join(lines[:8]) + "\n")
    rc, r = run("reset", str(rs), "--n", "40", "--lp", "1")
    check("stale summary with partial raw log rejected", rc == 4, r)
finally:
    shutil.rmtree(tmp); shutil.rmtree(_PLAIN, ignore_errors=True)
print("\nALL PASS" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
