"""Known-answer tests for phaseb_stats.py on a synthetic results tree with a
planted divergence pattern. The pipeline reads files and compares passes; the
expected values are computed directly from the planted matrix.

Usage: uv run python analysis/test_phaseb_stats.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phaseb_stats as pb  # noqa: E402

N_EP = 24
RATES = [0.2, 0.45, 0.65, 0.7]


def tok(i, lp, margin):
    tops = [{"id": i, "token": "t", "bytes": [], "logprob": lp},
            {"id": i + 1000, "token": "u", "bytes": [], "logprob": lp - margin}] + \
           [{"id": i + 2000 + k, "token": "v", "bytes": [], "logprob": lp - margin - 1 - k} for k in range(3)]
    return {"id": i, "token": "t", "bytes": [], "logprob": lp, "top_logprobs": tops}


def write(path, records):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "raw_requests.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def episode(e, flip, margin_at_flip, cache_hits, order_pos, merged_req=None):
    recs = []
    for k in range(2):
        toks = [tok(10 + j, -0.05, 2.0) for j in range(4)]
        if k == 0:
            toks[1] = tok(11, -0.6, margin_at_flip)
        if flip and k == 0:
            toks[1] = tok(99, -0.6, margin_at_flip)
        recs.append({"episode_id": f"multi_turn_base_{e}", "request_idx": k, "prompt_sha256": f"p{e}_{k}",
                     "latency_s": 0.1,
                     "response": {"created": 1000 + order_pos * 10 + k,
                                  "usage": {"completion_tokens": len(toks) + (1 if merged_req == k else 0)},
                                  "timings": {"cache_n": (50 if cache_hits else 0), "prompt_ms": 5.0,
                                              "prompt_n": 10, "predicted_ms": 20.0},
                                  "choices": [{"text": "", "logprobs": {"content": toks}}]}})
    return recs


def build(root, D):
    rng = np.random.default_rng(5)
    for j, f in enumerate(pb.FORMATS):
        for o in (0, 1):
            order = list(range(N_EP)) if o == 0 else list(rng.permutation(N_EP))
            write(root / "B2" / f"{f}_order{o}" / "arm_off",
                  [r for pos, e in enumerate(order) for r in episode(e, False, 0.05, False, pos)])
        for h in pb.HISTORIES:
            order = list(range(N_EP)) if h == 0 else list(np.random.default_rng([h]).permutation(N_EP))
            write(root / "B1" / f"{f}_order{h}" / "arm_on",
                  [r for pos, e in enumerate(order) for r in episode(e, bool(D[e, h, j]), 0.05, True, pos)])


def test_sweep_and_controls():
    rng = np.random.default_rng(42)
    D = np.zeros((N_EP, len(pb.HISTORIES), 4), int)
    for j, p in enumerate(RATES):
        D[:, :, j] = rng.random((N_EP, len(pb.HISTORIES))) < p
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "results"
        build(root, D)
        eps, T = pb.sweep_tensor(root)
        assert len(eps) == N_EP and T.shape == (N_EP, 11, 4)
        assert np.array_equal(T.astype(int), D), "tensor must equal the planted pattern"
        res = pb.sweep_analysis(eps, T)
        truth = D[:, 1:, :].mean(axis=(0, 1))
        for j, f in enumerate(pb.FORMATS):
            got = res["history_level"][pb.LABEL[f]]["mean"]
            assert abs(got - truth[j]) < 1e-12, (f, got, truth[j])
            sd = D[:, 1:, j].mean(axis=0).std(ddof=1)
            assert abs(res["history_level"][pb.LABEL[f]]["sd_between_histories"] - sd) < 1e-12
            lo, hi = res["two_way_bootstrap"]["rate_ci95"][pb.LABEL[f]]
            assert lo <= truth[j] <= hi
        for j, f in enumerate(pb.FORMATS[1:], start=1):
            dd = res["two_way_bootstrap"]["diff_vs_F16"][pb.LABEL[f]]["diff"]
            assert abs(dd - (truth[j] - truth[0])) < 1e-12
        assert res["complete_random_histories"] == list(range(1, 11))
        # off checks: order1 vs order0 must be identical by construction
        oc = pb.off_checks(root, Path(d) / "nonexistent")
        assert all(v["order1_vs_order0_diverged"] == 0 for v in oc.values()), oc
        # mechanism: flips occur only at the planted position with margin 0.05
        mech = pb.mechanism_analysis(root)
        for j, f in enumerate(pb.FORMATS):
            m = mech[pb.LABEL[f]]
            # canonical history only: each position counted once
            assert m["flips"] == int(D[:, 0, j].sum()), (f, m["flips"], int(D[:, 0, j].sum()))
            assert m["pooled_random_histories"]["flips"] == int(D[:, 1:, j].sum())
            if m["flips"]:
                assert abs(m["margin_at_flips"]["median"] - 0.05) < 1e-9
            assert m["abs_margin_perturbation_nonflip"]["mean"] < 1e-12
            hz = m["flip_hazard_by_recompute_margin"]["[0.03,0.1)"]
            assert hz["flips"] == int(D[:, 0, j].sum()) and hz["positions"] == N_EP


def test_glmm_trend_direction():
    rng = np.random.default_rng(7)
    D = np.zeros((N_EP, len(pb.HISTORIES), 4), int)
    for j, p in enumerate(RATES):
        D[:, :, j] = rng.random((N_EP, len(pb.HISTORIES))) < p
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "results"
        build(root, D)
        eps, T = pb.sweep_tensor(root)
        res = pb.sweep_analysis(eps, T)
        assert res["glmm_b1_by_history"]["mean"] > 0
        assert res["exact_permutation_episode_counts"]["p_one_sided_increasing"] < 0.01
        assert res["slope_per_step"]["points"] > 0


def test_perm_counts_exact_against_enumeration():
    """The convolution must equal brute-force enumeration of every assignment
    of one format permutation per episode."""
    import itertools
    rng = np.random.default_rng(11)
    C = rng.integers(0, 6, size=(3, 4))
    q = [0, 1, 2, 3]
    r = pb.exact_perm_trend_counts(C, q)
    perms = list(itertools.permutations(q))
    t_obs = int(sum(np.dot(q, row) for row in C))
    ts = [sum(int(np.dot(p, row)) for p, row in zip(combo, C)) for combo in itertools.product(perms, repeat=3)]
    p_up = sum(t >= t_obs for t in ts) / len(ts)
    assert abs(r["p_one_sided_increasing"] - p_up) < 1e-12 and r["T"] == t_obs


def test_perm_counts_shares_permutation_across_histories():
    """A persistent episode-by-format effect repeated over 10 histories must not
    behave like 10 independent strata: duplicating the histories leaves the
    count-based p unchanged up to scale, whereas stratifying by history would not."""
    rng = np.random.default_rng(3)
    base = (rng.random((40, 4)) < 0.5).astype(int)
    one = pb.exact_perm_trend_counts(base, [0, 1, 2, 3])["p_two_sided"]
    ten = pb.exact_perm_trend_counts(base * 10, [0, 1, 2, 3])["p_two_sided"]
    assert abs(one - ten) < 1e-9, (one, ten)


def test_strict_episode_sets():
    rng = np.random.default_rng(42)
    D = np.zeros((N_EP, len(pb.HISTORIES), 4), int)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "results"
        build(root, D)
        f = root / "B1" / "q80_order3" / "arm_on" / "raw_requests.jsonl"
        lines = [l for l in open(f) if '"multi_turn_base_7"' not in l]
        open(f, "w").writelines(lines)
        try:
            pb.sweep_tensor(root)
        except ValueError:
            return
        raise AssertionError("a missing episode must raise, not shrink the denominator")


def test_margins_skip_merged_and_unexposed():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        on = [r for e in range(4) for r in episode(e, False, 0.05, True, e, merged_req=(0 if e == 0 else None))]
        off = [r for e in range(4) for r in episode(e, False, 0.05, False, e)]
        for r in on:
            if r["episode_id"].endswith("_1") and r["request_idx"] == 1:
                r["response"]["timings"]["cache_n"] = 0
        write(root / "on", on); write(root / "off", off)
        rows = pb.margin_records(root / "on" / "raw_requests.jsonl", root / "off" / "raw_requests.jsonl")
        eps = [r[0] for r in rows]
        assert "multi_turn_base_0" not in eps, "episode with merged entries in request 0 must contribute nothing"
        assert eps.count("multi_turn_base_1") == 4, "unexposed request 1 of episode 1 must be skipped"
        assert eps.count("multi_turn_base_2") == 8


def test_b6_uses_execution_order():
    """B6 quartiles must follow the order episodes actually ran in."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "results"
        order = list(np.random.default_rng(3).permutation(80))
        diverge_first20 = set(order[:20])        # the first 20 to RUN diverge
        base = root / "B6" / "q4km_default_order1"
        for name, flipset in (("main", set()), ("repeat", diverge_first20)):
            recs = [r for pos, e in enumerate(order) for r in episode(e, e in flipset, 0.05, True, pos)]
            write(base / name / "arm_on", recs)
        json.dump({"order": [f"multi_turn_base_{e}" for e in order]}, open(base / "main" / "arm_on" / "run_meta.json", "w"))
        r = pb.b6_analysis(root)["histories"]["1"]
        assert r["quartiles_in_execution_order"] == [20, 0, 0, 0], r
        assert r["k"] == 20 and r["trend_p_two_sided"] < 0.001


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
