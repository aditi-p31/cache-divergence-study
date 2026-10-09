"""Known-answer tests for revision_stats.py. Every statistic reported in the
revision is produced by a function exercised here against a known answer,
an independent implementation, or a simulation with a planted effect.

Usage: uv run python analysis/test_revision_stats.py
"""

import gzip
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_stats as rs  # noqa: E402

# Politis-White circular block lengths from arch 8.0.0
# (arch.bootstrap.optimal_block_length), computed on these exact series.
PW_REF = {
    "iid_bern_0.4": 0.613112,
    "markov_0.85": 3.865655,
    "ar1_0.6_n200": 8.249791,
    "trend_like": 7.597888,
}


def _pw_series():
    rng = np.random.default_rng(7)
    out = {"iid_bern_0.4": (rng.random(80) < 0.4).astype(int).tolist()}
    x = [0]
    for _ in range(79):
        x.append(x[-1] if rng.random() < 0.85 else 1 - x[-1])
    out["markov_0.85"] = x
    ar = [0.0]
    for _ in range(199):
        ar.append(0.6 * ar[-1] + rng.normal())
    out["ar1_0.6_n200"] = ar
    out["trend_like"] = [1] * 16 + [0] * 4 + [1] * 6 + [0] * 14 + [1] * 6 + [0] * 14 + [1] * 3 + [0] * 17
    return out


def test_pw_matches_arch():
    for k, v in _pw_series().items():
        got = rs.pw_block_length(v)
        assert abs(got - PW_REF[k]) < 1e-5, (k, got, PW_REF[k])


def test_pw_constant_series():
    assert rs.pw_block_length([0] * 80) is None
    assert rs.block_rule(80, None) == 5
    assert rs.block_rule(80, 0.6) == 5
    assert rs.block_rule(80, 7.6) == 8


def test_cbb_indices_wrap_and_length():
    rng = np.random.default_rng(0)
    idx = rs.cbb_indices(10, 4, 50, rng)
    assert idx.shape == (50, 10)
    for row in idx:
        for blk in range(0, 8, 4):
            seg = row[blk: blk + 4]
            assert all((seg[i + 1] - seg[i]) % 10 == 1 for i in range(len(seg) - 1))


def _markov(n, stay, rng):
    x = np.empty(n, int)
    x[0] = rng.random() < 0.5
    for i in range(1, n):
        x[i] = x[i - 1] if rng.random() < stay else 1 - x[i - 1]
    return x


def test_cbb_se_iid():
    rng = np.random.default_rng(11)
    ses = []
    for s in range(40):
        x = (rng.random(80) < 0.4).astype(int)
        ses.append(rs.cbb_ci(x, 1, n_boot=4000, seed=s)[1])
    target = math.sqrt(0.4 * 0.6 / 80)
    assert abs(np.mean(ses) / target - 1) < 0.08, (np.mean(ses), target)


def test_cbb_coverage_dependent():
    """On a persistent binary chain, the block rule must cover the true mean
    far better than the iid bootstrap, which is the reviewers' point."""
    rng = np.random.default_rng(12)
    cov_rule = cov_iid = 0
    reps = 300
    for s in range(reps):
        x = _markov(80, 0.85, rng)
        b = rs.block_rule(80, rs.pw_block_length(x))
        lo, hi = rs.cbb_ci(x, b, n_boot=2000, seed=s)[0]
        cov_rule += lo <= 0.5 <= hi
        lo, hi = rs.cbb_ci(x, 1, n_boot=2000, seed=s)[0]
        cov_iid += lo <= 0.5 <= hi
    cov_rule /= reps
    cov_iid /= reps
    assert cov_rule > cov_iid + 0.15, (cov_rule, cov_iid)
    assert cov_rule > 0.80, cov_rule


def test_wilson_and_zero_bound():
    lo, hi = rs.wilson(31, 80)
    assert abs(lo - 0.2882) < 1e-4 and abs(hi - 0.4971) < 1e-4
    assert abs(rs.zero_upper(80) - 0.036755) < 1e-5


def test_mcnemar_exact():
    assert abs(rs.mcnemar_exact(7, 13) - stats.binomtest(7, 20, 0.5).pvalue) < 1e-12
    assert abs(rs.mcnemar_exact(7, 13) - 0.263176) < 1e-5
    assert rs.mcnemar_exact(0, 0) == 1.0


def _sim_glmm(n, q, b0, b1, sigma, rng):
    u = rng.normal(0, sigma, n)
    eta = b0 + b1 * np.asarray(q)[None, :] + u[:, None]
    return (rng.random(eta.shape) < 1 / (1 + np.exp(-eta))).astype(int)


def test_agq_matches_quadrature():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        pat = (rng.random(4) < 0.5).astype(float)
        eta = rng.normal(0, 1.5, 4)
        sig = float(np.exp(rng.normal(0.3, 0.8)))
        a = rs._cluster_loglik(pat[None, :], eta, sig)[0]
        b = rs._cluster_loglik_quad(pat, eta, sig)
        worst = max(worst, abs(a - b))
    assert worst < 1e-7, worst


def test_glmm_recovers_planted_slope():
    rng = np.random.default_rng(21)
    q = [0, 1, 2, 3]
    est, cover = [], 0
    for _ in range(80):
        D = _sim_glmm(80, q, -0.5, 0.8, 1.4, rng)
        g = rs.glmm_fit(D, q)
        est.append(g["b1"])
        lo, hi = g["b1_ci95_wald"]
        cover += lo <= 0.8 <= hi
    assert abs(np.mean(est) - 0.8) < 0.08, np.mean(est)
    assert 0.86 <= cover / 80 <= 1.0, cover / 80


def test_glmm_boundary_no_heterogeneity():
    """No episode heterogeneity: sigma is estimated at zero, the fit must not
    crash, and b1 must match ordinary logistic regression."""
    rng = np.random.default_rng(24)
    q = np.array([0, 1, 2, 3])
    p = 1 / (1 + np.exp(-(-0.8 + 0.6 * q)))
    D = (rng.random((200, 4)) < p[None, :]).astype(int)
    g = rs.glmm_fit(D, q)
    from scipy import optimize as so
    nll = lambda t: -np.sum(D * (t[0] + t[1] * q[None, :]) - np.logaddexp(0, t[0] + t[1] * q[None, :]))
    ref = so.minimize(nll, [0, 0], method="BFGS").x
    assert g["sigma"] < 0.2, g["sigma"]
    assert abs(g["b1"] - ref[1]) < 0.05, (g["b1"], ref[1])
    assert g["se_b1"] > 0 and math.isfinite(g["se_b1"])


def test_glmm_null_is_calibrated():
    rng = np.random.default_rng(22)
    rej = 0
    for _ in range(80):
        D = _sim_glmm(80, [0, 1, 2, 3], 0.0, 0.0, 1.4, rng)
        rej += rs.glmm_fit(D, [0, 1, 2, 3])["lr_p"] < 0.05
    assert rej / 80 <= 0.12, rej / 80


def test_clogit_recovers_planted_slope():
    rng = np.random.default_rng(23)
    est = []
    for _ in range(150):
        D = _sim_glmm(80, [0, 1, 2, 3], -0.5, 0.8, 1.4, rng)
        est.append(rs.clogit_fit(D, [0, 1, 2, 3])["beta"])
    assert abs(np.mean(est) - 0.8) < 0.08, np.mean(est)


def test_clogit_likelihood_bruteforce():
    """The conditional likelihood equals the brute-force sum over every
    outcome vector with the same row total."""
    import itertools
    q = np.array([0, 1, 2, 3], float)
    D = np.array([[0, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    b = 0.37
    ll = 0.0
    for r in D:
        alts = [np.array(v) for v in itertools.product([0, 1], repeat=4) if sum(v) == r.sum()]
        ll += b * q @ r - math.log(sum(math.exp(b * q @ a) for a in alts))
    fit = rs.clogit_fit(D, q)
    # verify the optimum: derivative of the brute-force likelihood is ~0 at fit["beta"]
    def bf(bb):
        s = 0.0
        for r in D:
            alts = [np.array(v) for v in itertools.product([0, 1], repeat=4) if sum(v) == r.sum()]
            s += bb * q @ r - math.log(sum(math.exp(bb * q @ a) for a in alts))
        return s
    h = 1e-5
    assert abs((bf(fit["beta"] + h) - bf(fit["beta"] - h)) / (2 * h)) < 1e-4
    assert np.isfinite(ll)


def test_exact_perm_hand_case():
    r = rs.exact_perm_trend(np.array([[0, 0, 0, 1]]), [0, 1, 2, 3])
    assert r["T"] == 3 and abs(r["p_one_sided_increasing"] - 0.25) < 1e-12


def test_exact_perm_matches_monte_carlo():
    rng = np.random.default_rng(31)
    D = _sim_glmm(80, [0, 1, 2, 3], -0.5, 0.25, 1.2, rng)
    ex = rs.exact_perm_trend(D, [0, 1, 2, 3])
    q = np.array([0, 1, 2, 3])
    t_obs = int((D @ q).sum())
    n_mc = 100000
    T = np.zeros(n_mc, int)
    for r in D:
        s = int(r.sum())
        if s in (0, 4):
            T += s and int(q.sum())
            continue
        perm = np.argsort(rng.random((n_mc, 4)), axis=1)[:, :s]
        T += q[perm].sum(axis=1)
    p_mc = (T >= t_obs).mean()
    assert abs(p_mc - ex["p_one_sided_increasing"]) < 0.006, (p_mc, ex)


def _write_pass(path, episodes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path) + ".gz", "wt") as fh:
        t = 1000
        for e, reqs in episodes:
            for k, (sha, toks, cache_n) in enumerate(reqs):
                t += 1
                fh.write(json.dumps({
                    "episode_id": e, "request_idx": k, "prompt_sha256": sha, "latency_s": 0.1,
                    "response": {"created": t, "timings": {"cache_n": cache_n},
                                 "choices": [{"text": "", "logprobs": {"content": [
                                     {"id": i, "logprob": lp} for i, lp in toks]}}]}}) + "\n")


def test_series_and_perturbation_on_synthetic_logs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        on = d / "on" / "raw_requests.jsonl"
        off = d / "off" / "raw_requests.jsonl"
        _write_pass(on, [("ep_0", [("a", [(1, -0.1), (2, -0.5)], 0), ("b", [(3, -0.20)], 50)]),
                         ("ep_1", [("c", [(4, -0.3), (9, -0.7)], 40)])])
        _write_pass(off, [("ep_0", [("a", [(1, -0.1), (2, -0.5)], 0), ("b", [(3, -0.25)], 0)]),
                          ("ep_1", [("c", [(4, -0.3), (5, -0.8)], 0)])])
        eps, x = rs.series(on, off)
        assert eps == ["ep_0", "ep_1"] and list(x) == [0, 1]
        toks, divs, neg_n, neg_viol = rs.perturbation(on, off)
        assert neg_n == 2 and neg_viol == 0
        hit = sorted(round(t[2], 6) for t in toks if t[1] > 0)
        assert hit == [0.0, 0.05], hit
        assert len(divs) == 1 and divs[0]["episode"] == "ep_1" and divs[0]["pos"] == 1
        assert abs(divs[0]["p1_off"] - math.exp(-0.8)) < 1e-12


def test_gsm8k_problem_bootstrap_on_synthetic():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, n in (("bridge-a", 4), ("bridge-b", 6)):
            rows = []
            for i in range(n):
                rows.append({"item_idx": i, "cold_correct": True, "warm_correct": not (name == "bridge-b" and i == 5),
                             "cold_deterministic": True, "warm_deterministic": True})
            (d / name / "cache_on").mkdir(parents=True)
            json.dump(rows, open(d / name / "cache_on" / "summary.json", "w"))
        g = rs.gsm8k_analysis(d, ["bridge-a", "bridge-b"], ["bridge-a", "bridge-b"])["pooled"]
        assert g["n_observations"] == 10 and g["n_unique_problems"] == 6
        assert g["observations_per_problem"] == {"1": 2, "2": 4}
        assert g["favor_recompute"] == 1 and g["favor_cached"] == 0
        assert abs(g["net_accuracy_diff_warm_minus_cold"] + 0.1) < 1e-12


def test_float_identity_synthetic():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a = d / "a" / "raw_requests.jsonl"; b = d / "b" / "raw_requests.jsonl"
        _write_pass(a, [("ep_0", [("s", [(1, -0.1), (2, -0.5)], 0)]), ("ep_1", [("t", [(3, -0.2)], 0)])])
        _write_pass(b, [("ep_0", [("s", [(1, -0.1), (2, -0.50001)], 0)]), ("ep_1", [("t", [(3, -0.2)], 0)])])
        r = rs.float_identity(a, b)
        assert r == {"tokens": 3, "identical_tokens": 3, "identical_logprobs": 2}, r


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
