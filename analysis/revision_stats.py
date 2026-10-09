"""Statistical reanalysis for the IEEE Access resubmission (Access-2026-42095).

Answers the reviewers' requests that can be met from the existing logs:

  cells        every agentic divergence proportion with a dependence-aware
               interval: circular block bootstrap (Politis and Romano 1992),
               block length by the Politis-White (2004) rule with the
               Patton-Politis-White (2009) correction, floored at
               ceil(n^(1/3)), plus a block-length sensitivity sweep
  trend        the quantization gradient under the paired design: random-
               intercept logistic GLMM (ML, adaptive quadrature), exact
               conditional logistic regression, exact within-episode
               permutation trend test, and a block bootstrap of episode rows
               carried jointly across formats
  contrasts    paired contrasts with dependence-aware intervals
  zero         the cache-off replay result as a configuration-level count
  gsm8k        the outcome bridge clustered by GSM8K problem
  mechanism    cache-hit versus recompute log-probability perturbation at
               matched positions, and top-1 probability at divergence sites
  latency      single-stream cost of the serving modes, from logged timings

Divergence is defined by analyze.compare_episode, unchanged, so every count
here matches the published pipeline. Episodes are ordered by numeric id,
which equals execution order in every logged pass (checked at run time).

Usage: uv run python analysis/revision_stats.py -o analysis/revision_findings.json
"""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import compare_episode, load_raw  # noqa: E402

SEED = 20261007
B = 10000
LEVELS = 0.95


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------

def ep_num(e):
    return int(e.rsplit("_", 1)[1])


def resolve(p):
    p = Path(p)
    return p if p.exists() or Path(str(p) + ".gz").exists() else None


def pass_records(path):
    return load_raw(Path(path))


def execution_order_ok(d):
    by_time = sorted(d, key=lambda e: min(r["response"]["created"] for r in d[e]))
    by_id = sorted(d, key=ep_num)
    return by_time == by_id


def series(path_a, path_b):
    """Episode ids and 0/1 divergence indicators, in execution order."""
    a, b = pass_records(path_a), pass_records(path_b)
    if not (execution_order_ok(a) and execution_order_ok(b)):
        raise RuntimeError(f"execution order != id order: {path_a} / {path_b}")
    eps = sorted(set(a) & set(b), key=ep_num)
    x = np.array([1 if compare_episode(a[e], b[e])["diverged"] else 0 for e in eps])
    return eps, x


# --------------------------------------------------------------------------
# dependence-aware intervals
# --------------------------------------------------------------------------

def pw_block_length(x):
    """Politis-White (2004) optimal block length with the Patton-Politis-White
    (2009) correction, for the circular block bootstrap. Mirrors the reference
    implementation in the arch package. Returns None for a constant series."""
    x = np.asarray(x, float)
    n = len(x)
    eps = x - x.mean()
    if not np.any(eps):
        return None
    b_max = math.ceil(min(3 * math.sqrt(n), n / 3))
    kn = max(5, int(math.log10(n)))
    m_max = int(math.ceil(math.sqrt(n))) + kn
    cv = 2 * math.sqrt(math.log10(n) / n)
    acv = np.zeros(m_max + 1)
    abs_acorr = np.zeros(m_max + 1)
    opt_m = None
    for i in range(m_max + 1):
        v1 = eps[i + 1:] @ eps[i + 1:]
        v2 = eps[: -(i + 1)] @ eps[: -(i + 1)]
        cross = eps[i:] @ eps[: n - i]
        acv[i] = cross / n
        abs_acorr[i] = abs(cross) / math.sqrt(v1 * v2) if v1 > 0 and v2 > 0 else 0.0
        if i >= kn and opt_m is None and np.all(abs_acorr[i - kn: i] < cv):
            opt_m = i - kn
    m = 2 * max(opt_m, 1) if opt_m is not None else m_max
    m = min(m, m_max)
    g = 0.0
    lr = acv[0]
    for k in range(1, m + 1):
        lam = 1.0 if k / m <= 0.5 else 2 * (1 - k / m)
        g += 2 * lam * k * acv[k]
        lr += 2 * lam * acv[k]
    d_cb = 4 / 3 * lr ** 2
    if d_cb <= 0 or g == 0:
        return 1.0
    return float(min(((2 * g ** 2) / d_cb) ** (1 / 3) * n ** (1 / 3), b_max))


def block_rule(n, pw):
    floor = math.ceil(n ** (1 / 3))
    return max(floor, int(math.ceil(pw))) if pw is not None else floor


def cbb_indices(n, block, n_boot, rng):
    """Row indices for the circular block bootstrap: blocks wrap around the
    end of the series, and each replicate is truncated to length n."""
    k = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def cbb_ci(x, block, n_boot=B, seed=SEED, stat=np.mean):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = cbb_indices(len(x), block, n_boot, rng)
    reps = stat(x[idx], axis=1) if stat is np.mean else np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(reps, [(1 - LEVELS) / 2, (1 + LEVELS) / 2])
    return [float(lo), float(hi)], float(np.std(reps, ddof=1))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def zero_upper(n, level=LEVELS):
    """Exact one-sided upper bound for a binomial rate when 0 of n occur."""
    return 1 - (1 - level) ** (1 / n)


def summarize_cell(x):
    x = np.asarray(x)
    n, k = len(x), int(x.sum())
    out = {"n": n, "k": k, "rate": k / n, "wilson95": wilson(k, n)}
    pw = pw_block_length(x)
    if pw is None:
        out.update({"pw_block": None, "block_used": None, "cbb95": None,
                    "note": "constant series; block bootstrap degenerate",
                    "exact_upper95_if_independent": zero_upper(n) if k == 0 else None,
                    "reported95": None})
        return out
    b = block_rule(n, pw)
    ci, se = cbb_ci(x, b)
    sens = {}
    for L in (1, 5, 10, 20):
        sens[str(L)] = cbb_ci(x, L)[0]
    # Reported interval: the envelope of the block-bootstrap and Wilson
    # intervals, so allowing for dependence can widen an interval but never
    # narrow it below the independent-trials interval. Percentile block
    # bootstraps undercover for rare events and can come out narrower than
    # Wilson when the estimated autocorrelation is negative.
    w = out["wilson95"]
    out.update({"pw_block": round(pw, 3), "block_used": b, "cbb95": ci, "cbb_se": se,
                "sensitivity_cbb95_by_block": sens,
                "reported95": [min(ci[0], w[0]), max(ci[1], w[1])]})
    return out


# --------------------------------------------------------------------------
# paired trend across formats (D: episodes x formats, q: scores)
# --------------------------------------------------------------------------

def _pattern_counts(D):
    pats, counts = np.unique(D, axis=0, return_counts=True)
    return pats, counts


GH_X, GH_W = np.polynomial.hermite.hermgauss(100)


def _cluster_loglik_quad(pat, eta_fixed, sigma):
    """Reference evaluation of one cluster's marginal log-likelihood by
    adaptive numerical integration (scipy.integrate.quad). Slow; used to
    validate the quadrature below."""
    def h(z):
        eta = eta_fixed + sigma * z
        return np.sum(pat * eta - np.logaddexp(0, eta)) - 0.5 * z * z
    res = optimize.minimize_scalar(lambda z: -h(z), bounds=(-40, 40), method="bounded")
    zhat = res.x
    hmax = h(zhat)
    val, _ = integrate.quad(lambda z: math.exp(h(z) - hmax), zhat - 40, zhat + 40,
                            points=[zhat], limit=400, epsabs=1e-13, epsrel=1e-11)
    return hmax + math.log(val) - 0.5 * math.log(2 * math.pi)


def _cluster_loglik(pats, eta_fixed, sigma):
    """Marginal log-likelihood of each outcome pattern under a random
    intercept, by adaptive Gauss-Hermite quadrature (100 nodes) centred at
    each pattern's posterior mode. pats: patterns x formats."""
    pats = np.asarray(pats, float)
    z = np.zeros(len(pats))
    for _ in range(50):
        eta = eta_fixed[None, :] + sigma * z[:, None]
        mu = 1 / (1 + np.exp(-eta))
        g = sigma * (pats - mu).sum(axis=1) - z
        hh = -sigma ** 2 * (mu * (1 - mu)).sum(axis=1) - 1
        step = g / hh
        z = z - step
        if np.max(np.abs(step)) < 1e-12:
            break
    eta = eta_fixed[None, :] + sigma * z[:, None]
    mu = 1 / (1 + np.exp(-eta))
    s = 1 / np.sqrt(sigma ** 2 * (mu * (1 - mu)).sum(axis=1) + 1)
    nodes = z[:, None] + math.sqrt(2) * s[:, None] * GH_X[None, :]          # P x K
    etan = eta_fixed[None, None, :] + sigma * nodes[:, :, None]              # P x K x J
    h = (pats[:, None, :] * etan - np.logaddexp(0, etan)).sum(axis=2) - 0.5 * nodes ** 2
    log_terms = h + GH_X[None, :] ** 2 + np.log(GH_W)[None, :]
    m = log_terms.max(axis=1)
    return m + np.log(np.exp(log_terms - m[:, None]).sum(axis=1)) + np.log(math.sqrt(2) * s) - 0.5 * math.log(2 * math.pi)


def glmm_negloglik(theta, pats, counts, q, fix_b1=None):
    if fix_b1 is None:
        b0, b1, lsig = theta
    else:
        b0, lsig = theta
        b1 = fix_b1
    sigma = math.exp(lsig)
    eta = b0 + b1 * np.asarray(q, float)
    return -float(np.dot(counts, _cluster_loglik(pats, eta, sigma)))


def _num_hessian(f, x, h=1e-4):
    x = np.asarray(x, float)
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h
            ej = np.zeros(k); ej[j] = h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def glmm_fit(D, q):
    """Random-intercept logistic GLMM, logit P(D_ij=1) = b0 + b1 q_j + u_i,
    u_i ~ N(0, sigma^2), fitted by maximum likelihood with each cluster's
    integral evaluated by adaptive numerical quadrature."""
    D = np.asarray(D, int)
    pats, counts = _pattern_counts(D)
    q = np.asarray(q, float)
    f = lambda t: glmm_negloglik(t, pats, counts, q)
    best = None
    for start in ([0.0, 0.5, 0.0], [-1.0, 1.0, 0.7], [0.5, 0.2, 1.2]):
        r = optimize.minimize(f, start, method="Nelder-Mead",
                              options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
        r = optimize.minimize(f, r.x, method="BFGS", options={"gtol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    b0, b1, lsig = best.x
    boundary = math.exp(lsig) < 1e-3
    cov = None
    if not boundary:
        H = _num_hessian(f, best.x)
        try:
            cov = np.linalg.inv(H)
            boundary = not (np.all(np.isfinite(cov)) and cov[1, 1] > 0)
        except np.linalg.LinAlgError:
            boundary = True
    if boundary:
        # The random-intercept variance is estimated at zero, so the MLE lies on
        # the boundary and the model reduces to ordinary logistic regression;
        # the SE of b1 comes from that reduced model.
        def nll_fixed(t):
            eta = t[0] + t[1] * np.asarray(q, float)[None, :]
            return -float(np.sum(D * eta - np.logaddexp(0, eta)))
        rf = optimize.minimize(nll_fixed, [b0, b1], method="BFGS", options={"gtol": 1e-9})
        b0, b1 = rf.x
        lsig = math.log(1e-12)
        cov = np.linalg.inv(_num_hessian(nll_fixed, rf.x))
        se_b1 = math.sqrt(cov[1, 1])
        best_fun = rf.fun
    else:
        se_b1 = math.sqrt(cov[1, 1])
        best_fun = best.fun
    f0 = lambda t: glmm_negloglik(t, pats, counts, q, fix_b1=0.0)
    r0 = optimize.minimize(f0, [b0, lsig], method="Nelder-Mead",
                           options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
    lr = 2 * (r0.fun - best_fun)
    z = stats.norm.ppf((1 + LEVELS) / 2)
    return {"b0": b0, "b1": b1, "sigma": math.exp(lsig), "se_b1": se_b1,
            "b1_ci95_wald": [b1 - z * se_b1, b1 + z * se_b1],
            "or_per_step": math.exp(b1),
            "or_per_step_ci95": [math.exp(b1 - z * se_b1), math.exp(b1 + z * se_b1)],
            "lr_stat": lr, "lr_p": float(stats.chi2.sf(max(lr, 0), 1)),
            "loglik": -best_fun, "sigma_at_boundary": boundary,
            "converged": bool(boundary or best.success or abs(best.jac).max() < 1e-3)}


def _subset_sums(q, s):
    return [sum(c) for c in itertools.combinations(q, s)]


def clogit_fit(D, q):
    """Exact conditional logistic regression, stratified by episode: the
    episode intercept is conditioned out by the episode's divergence count."""
    D = np.asarray(D, int)
    q = np.asarray(q, float)
    rows = [r for r in D if 0 < r.sum() < len(q)]
    obs = [float(q @ r) for r in rows]
    alts = [np.array(_subset_sums(q, int(r.sum()))) for r in rows]

    def nll(b):
        return -sum(t * b - (np.log(np.sum(np.exp(b * a - (b * a).max()))) + (b * a).max())
                    for t, a in zip(obs, alts))

    r = optimize.minimize_scalar(nll, bounds=(-20, 20), method="bounded",
                                 options={"xatol": 1e-10})
    b = r.x
    info = 0.0
    for a in alts:
        w = np.exp(b * a - (b * a).max()); w /= w.sum()
        info += w @ a ** 2 - (w @ a) ** 2
    se = 1 / math.sqrt(info)
    z = stats.norm.ppf((1 + LEVELS) / 2)
    return {"beta": b, "se": se, "ci95": [b - z * se, b + z * se],
            "or_per_step": math.exp(b), "n_informative_episodes": len(rows),
            "n_concordant_dropped": int(len(D) - len(rows))}


def exact_perm_trend(D, q):
    """Exact within-episode permutation trend test. Under the null, each
    episode's divergences are equally likely to fall on any subset of formats
    of the same size. T = sum_i sum_j q_j D_ij (integer scores)."""
    D = np.asarray(D, int)
    qi = [int(v) for v in q]
    if any(v != w for v, w in zip(qi, q)):
        raise ValueError("integer scores required for the exact convolution")
    dist = {0: 1.0}
    for r in D:
        s = int(r.sum())
        sums = _subset_sums(qi, s)
        p = 1.0 / len(sums)
        nd = {}
        for t, pt in dist.items():
            for v in sums:
                nd[t + v] = nd.get(t + v, 0.0) + pt * p
        dist = nd
    t_obs = int(sum(np.asarray(qi) @ r for r in D))
    ts = np.array(sorted(dist)); ps = np.array([dist[t] for t in ts])
    mean = float(ts @ ps)
    p_upper = float(ps[ts >= t_obs].sum())
    p_two = float(min(1.0, ps[np.abs(ts - mean) >= abs(t_obs - mean) - 1e-9].sum()))
    return {"T": t_obs, "null_mean": mean, "p_one_sided_increasing": p_upper, "p_two_sided": p_two}


def row_block_bootstrap(D, q, block, n_boot=B, seed=SEED):
    """Resample blocks of consecutive episodes (rows, all formats carried
    together). block=1 is the plain episode cluster bootstrap."""
    D = np.asarray(D, float)
    n, J = D.shape
    rng = np.random.default_rng(seed)
    idx = cbb_indices(n, block, n_boot, rng)
    rates = D[idx].mean(axis=1)                      # n_boot x J
    qc = np.asarray(q, float) - np.mean(q)
    slope = (rates @ qc) / (qc @ qc)                 # OLS slope of rate on score
    lo, hi = (1 - LEVELS) / 2, (1 + LEVELS) / 2
    out = {"block": block,
           "rate_ci95": [list(map(float, np.quantile(rates[:, j], [lo, hi]))) for j in range(J)],
           "slope_per_step_ci95": list(map(float, np.quantile(slope, [lo, hi]))),
           "diff_vs_first_ci95": [list(map(float, np.quantile(rates[:, j] - rates[:, 0], [lo, hi])))
                                  for j in range(1, J)]}
    return out


def trend_analysis(D, q, labels):
    D = np.asarray(D, int)
    rates = D.mean(axis=0)
    qc = np.asarray(q, float) - np.mean(q)
    out = {"formats": labels, "scores": list(q), "n_episodes": int(D.shape[0]),
           "rates": [float(r) for r in rates],
           "diff_vs_first": [float(rates[j] - rates[0]) for j in range(1, len(q))],
           "slope_per_step": float((rates @ qc) / (qc @ qc)),
           "patterns": {"".join(map(str, p)): int(c) for p, c in zip(*_pattern_counts(D))}}
    out["glmm"] = glmm_fit(D, q)
    out["clogit"] = clogit_fit(D, q)
    out["exact_permutation"] = exact_perm_trend(D, q)
    blocks = [block_rule(D.shape[0], pw_block_length(D[:, j] - D[:, 0]))
              for j in range(1, D.shape[1])]
    b = max(blocks)
    out["bootstrap_episode"] = row_block_bootstrap(D, q, 1)
    out["bootstrap_block"] = row_block_bootstrap(D, q, b)
    return out


# --------------------------------------------------------------------------
# GSM8K, clustered by problem
# --------------------------------------------------------------------------

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def gsm8k_analysis(root, pooled, all_bridges):
    per = {}
    obs = []  # (problem, config, cold_correct, warm_correct, attributable_flip)
    for name in all_bridges:
        rows = json.load(open(Path(root) / name / "cache_on" / "summary.json"))
        b = c = 0
        n_det = 0
        for r in rows:
            det = bool(r.get("cold_deterministic")) and bool(r.get("warm_deterministic"))
            n_det += det
            cc, wc = bool(r["cold_correct"]), bool(r["warm_correct"])
            if det and cc and not wc:
                b += 1
            if det and wc and not cc:
                c += 1
            if name in pooled:
                obs.append((int(r["item_idx"]), name, cc, wc, det and cc != wc))
        n = len(rows)
        per[name] = {"n": n, "n_both_paths_deterministic": n_det,
                     "acc_cold": sum(bool(r["cold_correct"]) for r in rows) / n,
                     "acc_warm": sum(bool(r["warm_correct"]) for r in rows) / n,
                     "flips_favor_recompute": b, "flips_favor_cached": c,
                     "mcnemar_exact_p": mcnemar_exact(b, c)}
    probs = sorted({o[0] for o in obs})
    by_p = {p: [o for o in obs if o[0] == p] for p in probs}
    n_obs = len(obs)
    net = (sum(o[3] for o in obs) - sum(o[2] for o in obs)) / n_obs
    flips_per_problem = [sum(o[4] for o in by_p[p]) for p in probs]
    fav_c = sum(1 for o in obs if o[4] and o[3])
    fav_r = sum(1 for o in obs if o[4] and o[2])
    net_by_problem = [sum((o[3] - o[2]) for o in by_p[p] if o[4]) for p in probs]
    pos = sum(1 for v in net_by_problem if v > 0)
    neg = sum(1 for v in net_by_problem if v < 0)
    # stratified problem-level bootstrap: strata = the set of configurations a problem appears in
    strata = {}
    for p in probs:
        strata.setdefault(tuple(sorted(o[1] for o in by_p[p])), []).append(p)
    d_by_p = {p: sum(o[3] - o[2] for o in by_p[p]) for p in probs}
    rng = np.random.default_rng(SEED)
    reps = np.zeros(B)
    for s, plist in strata.items():
        arr = np.array([d_by_p[p] for p in plist], float)
        draw = rng.integers(0, len(arr), size=(B, len(arr)))
        reps += arr[draw].sum(axis=1)
    reps /= n_obs
    ci = list(map(float, np.quantile(reps, [(1 - LEVELS) / 2, (1 + LEVELS) / 2])))
    # fsum: exactly rounded, so the value does not depend on the order the observations were read in
    naive_se = math.sqrt(math.fsum(((o[3] - o[2]) - net) ** 2 for o in obs) / (n_obs - 1) / n_obs)
    return {"per_bridge": per,
            "pooled": {"configurations": pooled, "n_observations": n_obs,
                       "n_unique_problems": len(probs),
                       "observations_per_problem": {str(k): v for k, v in
                                                    sorted(_count([len(by_p[p]) for p in probs]).items())},
                       "attributable_flips": fav_c + fav_r, "favor_cached": fav_c, "favor_recompute": fav_r,
                       "max_flips_on_one_problem": int(max(flips_per_problem)),
                       "problems_with_any_flip": int(sum(1 for v in flips_per_problem if v)),
                       "pooled_mcnemar_p": mcnemar_exact(fav_r, fav_c),
                       "problem_clustered_sign_test_p": mcnemar_exact(neg, pos),
                       "net_accuracy_diff_warm_minus_cold": net,
                       "net_diff_ci95_problem_bootstrap": ci,
                       "net_diff_se_problem_bootstrap": float(np.std(reps, ddof=1)),
                       "net_diff_se_naive": naive_se}}


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# --------------------------------------------------------------------------
# mechanism: log-probability perturbation at matched positions
# --------------------------------------------------------------------------

def _content(rec):
    return (rec["response"]["choices"][0].get("logprobs") or {}).get("content") or []


def _cache_n(rec):
    t = rec["response"].get("timings") or {}
    if "cache_n" in t:
        return int(t["cache_n"])
    u = (rec["response"].get("usage") or {}).get("prompt_tokens_details") or {}
    return int(u.get("cached_tokens") or 0)


def perturbation(path_on, path_off, first_request_only=False, max_tokens=None):
    """Walk cache-on and cache-off passes in lockstep up to the first
    difference. Returns per-token records at shared positions and the top-1
    probabilities at each episode's first divergent token."""
    A, Bp = pass_records(path_on), pass_records(path_off)
    toks, divs, neg_viol, neg_n = [], [], 0, 0
    for e in sorted(set(A) & set(Bp), key=ep_num):
        reqs = list(zip(A[e], Bp[e]))
        if first_request_only:
            reqs = reqs[:1]
        for k, (ra, rb) in enumerate(reqs):
            if ra["prompt_sha256"] != rb["prompt_sha256"]:
                break
            ca, cb = _content(ra), _content(rb)
            cn = _cache_n(ra)
            stop = False
            for i, (ta, tb) in enumerate(zip(ca, cb)):
                if max_tokens is not None and i >= max_tokens:
                    break
                if ta["id"] != tb["id"]:
                    divs.append({"episode": e, "request": k, "pos": i,
                                 "p1_on": math.exp(ta["logprob"]), "p1_off": math.exp(tb["logprob"])})
                    stop = True
                    break
                d = abs(ta["logprob"] - tb["logprob"])
                if cn == 0:
                    neg_n += 1
                    neg_viol += d != 0.0
                toks.append((e, cn, d, math.exp(tb["logprob"])))
            if stop:
                break
            if len(ca) != len(cb):
                break
    return toks, divs, neg_n, neg_viol


def mechanism_analysis(cells):
    """cells: list of (label, path_on, path_off) for formats in severity order."""
    out = {"formats": [c[0] for c in cells], "per_format": {}, "matched_request0_first20": {}}
    ep_means = {}
    ep_lowconf = {}
    for label, pon, poff in cells:
        toks, divs, neg_n, neg_viol = perturbation(pon, poff)
        d = np.array([t[2] for t in toks if t[1] > 0])
        p = np.array([t[3] for t in toks])
        g = np.array([t[2] / (1 - t[3]) for t in toks if t[1] > 0 and (1 - t[3]) >= 1e-3])
        p1d = np.array([v["p1_off"] for v in divs])
        out["per_format"][label] = {
            "shared_tokens": len(toks), "shared_tokens_cache_hit": int(len(d)),
            "negative_control_tokens": neg_n, "negative_control_nonzero": int(neg_viol),
            "abs_dlogprob": {"mean": float(d.mean()), "median": float(np.median(d)),
                             "p90": float(np.quantile(d, .9)), "p99": float(np.quantile(d, .99)),
                             "share_nonzero": float((d > 0).mean())},
            "normalized_g": {"n": int(len(g)), "median": float(np.median(g)) if len(g) else None,
                             "mean": float(g.mean()) if len(g) else None},
            "p1_shared": {"median": float(np.median(p)), "share_below_0.6": float((p < 0.6).mean())},
            "first_divergent_token": {"n": int(len(p1d)),
                                      "p1_recompute_median": float(np.median(p1d)) if len(p1d) else None,
                                      "p1_recompute_iqr": list(map(float, np.quantile(p1d, [.25, .75]))) if len(p1d) else None,
                                      "share_p1_below_0.6": float((p1d < 0.6).mean()) if len(p1d) else None}}
        em = {}
        for e, cn, dd, pp in toks:
            if cn > 0:
                em.setdefault(e, []).append(dd)
        ep_means[label] = {e: float(np.mean(v)) for e, v in em.items()}
        el = {}
        for e, cn, dd, pp in toks:
            el.setdefault(e, []).append(pp < 0.6)
        ep_lowconf[label] = {e: float(np.mean(v)) for e, v in el.items()}
        t0, _, _, _ = perturbation(pon, poff, first_request_only=True, max_tokens=20)
        d0 = np.array([t[2] for t in t0])
        out["matched_request0_first20"][label] = {"n": int(len(d0)), "mean": float(d0.mean()),
                                                  "p90": float(np.quantile(d0, .9)),
                                                  "p99": float(np.quantile(d0, .99)),
                                                  "share_nonzero": float((d0 > 0).mean())}
    labels = out["formats"]
    for name, src in (("episode_mean_abs_dlogprob", ep_means), ("episode_share_p1_below_0.6", ep_lowconf)):
        common = sorted(set.intersection(*[set(src[l]) for l in labels]), key=ep_num)
        M = np.array([[src[l][e] for l in labels] for e in common])
        res = {"n_episodes": len(common), "format_means": list(map(float, M.mean(axis=0)))}
        if len(labels) >= 3 and len(common) >= 3:
            pt = stats.page_trend_test(M)
            fr = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
            res.update({"page_L": float(pt.statistic), "page_p_increasing": float(pt.pvalue),
                        "friedman_p": float(fr.pvalue)})
            pt_dec = stats.page_trend_test(M[:, ::-1])
            res["page_p_decreasing"] = float(pt_dec.pvalue)
        rng = np.random.default_rng(SEED)
        draw = rng.integers(0, len(common), size=(B, len(common)))
        reps = M[draw].mean(axis=1)
        res["format_means_ci95_episode_bootstrap"] = [list(map(float, np.quantile(reps[:, j], [.025, .975])))
                                                      for j in range(M.shape[1])]
        ratio = reps[:, -1] / reps[:, 0] if np.all(reps[:, 0] > 0) else None
        if ratio is not None:
            res["last_over_first_ratio"] = float(M[:, -1].mean() / M[:, 0].mean())
            res["last_over_first_ratio_ci95"] = list(map(float, np.quantile(ratio, [.025, .975])))
        out[name] = res
    return out


# --------------------------------------------------------------------------
# latency of serving modes (single stream)
# --------------------------------------------------------------------------

def pass_timing(path):
    d = pass_records(path)
    recs = [r for v in d.values() for r in v]
    t = [r["response"].get("timings") or {} for r in recs]
    lat = np.array([r["latency_s"] for r in recs])
    pms = np.array([x.get("prompt_ms", np.nan) for x in t], float)
    pn = np.array([x.get("prompt_n", np.nan) for x in t], float)
    cn = np.array([x.get("cache_n", np.nan) for x in t], float)
    gen = np.array([x.get("predicted_ms", np.nan) for x in t], float)
    return {"requests": len(recs), "pass_minutes": float(lat.sum() / 60),
            "latency_s_mean": float(lat.mean()), "latency_s_median": float(np.median(lat)),
            "prefill_ms_mean": float(np.nanmean(pms)), "prefill_ms_median": float(np.nanmedian(pms)),
            "prompt_tokens_computed_mean": float(np.nanmean(pn)),
            "prompt_tokens_from_cache_mean": float(np.nanmean(cn)),
            "generation_ms_mean": float(np.nanmean(gen))}


# --------------------------------------------------------------------------
# cache-off replay: identity below the token level
# --------------------------------------------------------------------------

def _token_logprobs(rec):
    lp = rec["response"]["choices"][0].get("logprobs") or {}
    c = lp.get("content")
    if c:
        return [(t["id"], t["logprob"]) for t in c]
    toks = lp.get("tokens") or []
    lps = lp.get("token_logprobs") or []
    return list(zip(toks, lps))


def float_identity(path_a, path_b):
    """Compare two passes token by token (ids for llama.cpp, strings for vLLM)
    and count tokens whose reported log-probability is bit-identical."""
    A, Bp = pass_records(path_a), pass_records(path_b)
    n = same_tok = same_lp = 0
    for e in sorted(set(A) & set(Bp), key=ep_num):
        for ra, rb in zip(A[e], Bp[e]):
            for (ta, la), (tb, lb) in zip(_token_logprobs(ra), _token_logprobs(rb)):
                n += 1
                same_tok += ta == tb
                same_lp += (ta == tb) and (la == lb)
    return {"tokens": n, "identical_tokens": same_tok, "identical_logprobs": same_lp}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

GRID = Path("results-pod/results")
REP = Path("results-repair")
CONFIGS = ["lcpp-qwen7b-f16", "lcpp-qwen7b-q80", "lcpp-qwen7b-q4km", "lcpp-qwen7b-q3km",
           "lcpp-llama8b-q80", "lcpp-llama8b-q4km", "lcpp-llama8b-q3km", "lcpp-qwen14b-q4km",
           "vllm-qwen7b-fp16", "vllm-llama8b-fp16"]
RAW = "raw_requests.jsonl"


def P(*parts):
    return str(Path(*parts) / RAW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="analysis/revision_findings.json")
    ap.add_argument("--quick", action="store_true", help="skip GLMM fits (for smoke tests)")
    ap.add_argument("--grid", default=None, help="original grid root (project: results-pod/results; artifact: results)")
    ap.add_argument("--repair", default=None, help="repair runs root (default results-repair)")
    args = ap.parse_args()
    global GRID, REP
    if args.grid:
        GRID = Path(args.grid)
    if args.repair:
        REP = Path(args.repair)
    out = {"meta": {"seed": SEED, "bootstrap_replicates": B, "level": LEVELS,
                    "block_rule": "max(ceil(n^(1/3)), ceil(Politis-White CB length)), circular blocks, percentile CI",
                    "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                    "glmm_se": "Wald, from the full observed information over (b0, b1, log sigma)",
                    "pw_reference_implementation": "arch 8.0.0 optimal_block_length (circular), matched in tests"}}

    # ---- every agentic proportion ----
    cells = {}
    for c in CONFIGS:
        g = GRID / c
        for key, a, b in (("cacheoff_rerun", P(g, "main", "arm_off"), P(g, "repeat", "arm_off")),
                          ("cacheon_rerun", P(g, "main", "arm_on"), P(g, "repeat", "arm_on")),
                          ("cross_arm", P(g, "main", "arm_on"), P(g, "main", "arm_off"))):
            _, x = series(a, b)
            cells[f"table1/{c}/{key}"] = dict(summarize_cell(x), source=[a, b])
    for key, a, b in (
            ("fig1/cacheoff_rerun", P(REP, "repair", "lcpp-orderB", "main", "arm_off"), P(REP, "repair", "lcpp-orderB", "repeat", "arm_off")),
            ("fig1/cacheon_rerun_promptcache_off", P(REP, "cacheram", "cacheram-zero", "main", "arm_on"), P(REP, "cacheram", "cacheram-zero", "repeat", "arm_on")),
            ("fig1/cacheon_rerun_promptcache_default", P(REP, "cacheram", "cacheram-default", "main", "arm_on"), P(REP, "cacheram", "cacheram-default", "repeat", "arm_on")),
            ("fig1/cross_arm", P(REP, "repair", "lcpp-orderB", "main", "arm_on"), P(REP, "repair", "lcpp-orderB", "main", "arm_off")),
            ("table2/default_adjacent", P(REP, "cacheram", "cacheram-default", "main", "arm_on"), P(REP, "cacheram", "cacheram-default", "repeat", "arm_on")),
            ("table2/default_cacheoff_between", P(GRID, "lcpp-qwen7b-q4km", "main", "arm_on"), P(GRID, "lcpp-qwen7b-q4km", "repeat", "arm_on")),
            ("table2/disabled_adjacent", P(REP, "repair", "lcpp-orderB", "main", "arm_on"), P(REP, "repair", "lcpp-orderB", "repeat", "arm_on")),
            ("table2/disabled_cacheoff_between", P(REP, "repair", "lcpp-orderA", "main", "arm_on"), P(REP, "repair", "lcpp-orderA", "repeat", "arm_on")),
            ("controlled/f16_cross", P(REP, "quantgrad", "f16", "main", "arm_on"), P(REP, "quantgrad", "f16", "main", "arm_off")),
            ("controlled/q4km_cross", P(REP, "repair", "lcpp-orderB", "main", "arm_on"), P(REP, "repair", "lcpp-orderB", "main", "arm_off")),
            ("controlled/q3km_cross", P(REP, "quantgrad", "q3km", "main", "arm_on"), P(REP, "quantgrad", "q3km", "main", "arm_off")),
            ("vllm/qwen7b_fresh_server_rerun", P(REP, "repair", "vllm-orderA", "main", "arm_on"), P(REP, "repair", "vllm-orderA", "repeat", "arm_on")),
            ("vllm/qwen7b_shared_server_rerun", P(REP, "repair", "vllm-orderB", "main", "arm_on"), P(REP, "repair", "vllm-orderB", "repeat", "arm_on"))):
        _, x = series(a, b)
        cells[key] = dict(summarize_cell(x), source=[a, b])
    out["cells"] = cells

    # ---- zero-event statement ----
    zero = [k for k in cells if k.endswith("/cacheoff_rerun") and k.startswith("table1/")]
    out["cacheoff_replay"] = {
        "configurations": len(zero), "episodes_each": [cells[k]["n"] for k in zero],
        "divergent_episodes": [cells[k]["k"] for k in zero],
        "statement": f"no cache-disabled divergence in any of the {len(zero)} configurations",
        "per_configuration_exact_upper95_if_episodes_independent": zero_upper(80)}

    out["cacheoff_replay"]["float_identity"] = {
        c: float_identity(P(GRID, c, "main", "arm_off"), P(GRID, c, "repeat", "arm_off")) for c in CONFIGS}

    # ---- paired contrasts ----
    def paired_diff(a1, b1, a2, b2):
        e1, x1 = series(a1, b1)
        e2, x2 = series(a2, b2)
        assert e1 == e2
        d = x1 - x2
        pw = pw_block_length(d)
        blk = block_rule(len(d), pw)
        ci, se = cbb_ci(d, blk)
        return {"diff": float(d.mean()), "block_used": blk, "cbb95": ci,
                "episode_bootstrap95": cbb_ci(d, 1)[0]}
    out["contrasts"] = {
        "promptcache_default_minus_disabled_rerun": paired_diff(
            P(REP, "cacheram", "cacheram-default", "main", "arm_on"), P(REP, "cacheram", "cacheram-default", "repeat", "arm_on"),
            P(REP, "cacheram", "cacheram-zero", "main", "arm_on"), P(REP, "cacheram", "cacheram-zero", "repeat", "arm_on")),
        "order_between_minus_adjacent_promptcache_default": paired_diff(
            P(GRID, "lcpp-qwen7b-q4km", "main", "arm_on"), P(GRID, "lcpp-qwen7b-q4km", "repeat", "arm_on"),
            P(REP, "cacheram", "cacheram-default", "main", "arm_on"), P(REP, "cacheram", "cacheram-default", "repeat", "arm_on")),
    }

    # ---- paired quantization trend ----
    def matrix(paths):
        cols, eps0 = [], None
        for a, b in paths:
            e, x = series(a, b)
            if eps0 is None:
                eps0 = e
            assert e == eps0
            cols.append(x)
        return np.column_stack(cols)
    gridD = matrix([(P(GRID, f"lcpp-qwen7b-{f}", "main", "arm_on"), P(GRID, f"lcpp-qwen7b-{f}", "main", "arm_off"))
                    for f in ("f16", "q80", "q4km", "q3km")])
    ctrlD = matrix([(P(REP, "quantgrad", "f16", "main", "arm_on"), P(REP, "quantgrad", "f16", "main", "arm_off")),
                    (P(REP, "repair", "lcpp-orderB", "main", "arm_on"), P(REP, "repair", "lcpp-orderB", "main", "arm_off")),
                    (P(REP, "quantgrad", "q3km", "main", "arm_on"), P(REP, "quantgrad", "q3km", "main", "arm_off"))])
    if not args.quick:
        out["trend_grid_production_default"] = trend_analysis(gridD, [0, 1, 2, 3], ["F16", "Q8_0", "Q4_K_M", "Q3_K_M"])
        out["trend_controlled_existing"] = trend_analysis(ctrlD, [0, 2, 3], ["F16", "Q4_K_M", "Q3_K_M"])

    # ---- GSM8K ----
    pooled = ["bridge-qwen7b-f16", "bridge-qwen7b-q80", "bridge-qwen7b-q3km",
              "bridge-qwen14b-q4km", "bridge-qwen7b-q4km-n500"]
    allb = sorted(p.name for p in GRID.glob("bridge-*") if (p / "cache_on" / "summary.json").exists())
    out["gsm8k"] = gsm8k_analysis(GRID, pooled, allb)

    # ---- mechanism ----
    out["mechanism_grid"] = mechanism_analysis(
        [(lbl, P(GRID, f"lcpp-qwen7b-{f}", "main", "arm_on"), P(GRID, f"lcpp-qwen7b-{f}", "main", "arm_off"))
         for lbl, f in (("F16", "f16"), ("Q8_0", "q80"), ("Q4_K_M", "q4km"), ("Q3_K_M", "q3km"))])
    out["mechanism_controlled"] = mechanism_analysis(
        [("F16", P(REP, "quantgrad", "f16", "main", "arm_on"), P(REP, "quantgrad", "f16", "main", "arm_off")),
         ("Q4_K_M", P(REP, "repair", "lcpp-orderB", "main", "arm_on"), P(REP, "repair", "lcpp-orderB", "main", "arm_off")),
         ("Q3_K_M", P(REP, "quantgrad", "q3km", "main", "arm_on"), P(REP, "quantgrad", "q3km", "main", "arm_off"))])

    # ---- latency (Qwen2.5-7B Q4_K_M, second machine, same day) ----
    out["latency_q4km_machine2"] = {
        "recompute_cache_off": pass_timing(P(REP, "repair", "lcpp-orderB", "main", "arm_off")),
        "prefix_reuse_promptcache_default": pass_timing(P(REP, "cacheram", "cacheram-default", "main", "arm_on")),
        "prefix_reuse_promptcache_disabled": pass_timing(P(REP, "cacheram", "cacheram-zero", "main", "arm_on")),
    }

    # ---- construct check: orderA really interposes the cache-off pass ----
    def span(p):
        d = pass_records(p)
        ts = [r["response"]["created"] for v in d.values() for r in v]
        return min(ts), max(ts)
    sA = {k: span(P(REP, "repair", "lcpp-orderA", *k.split("/"))) for k in ("main/arm_on", "main/arm_off", "repeat/arm_on")}
    sB = {k: span(P(REP, "repair", "lcpp-orderB", *k.split("/"))) for k in ("main/arm_on", "main/arm_off", "repeat/arm_on")}
    out["construct_checks"] = {
        "orderA_off_pass_between_on_passes": sA["main/arm_on"][1] <= sA["main/arm_off"][0] and sA["main/arm_off"][1] <= sA["repeat/arm_on"][0],
        "orderB_on_passes_adjacent": sB["main/arm_on"][1] <= sB["repeat/arm_on"][0] and not (sB["main/arm_on"][1] <= sB["main/arm_off"][0] <= sB["repeat/arm_on"][0]),
    }

    # requests that ended at the generation cap (4096 tokens BFCL, 1024 GSM8K), by workload
    tot, cap = {"bfcl": 0, "gsm8k": 0}, {"bfcl": 0, "gsm8k": 0}
    for root in (GRID, REP):
        for f in Path(root).rglob("raw_requests.jsonl*"):
            if any(part.startswith("gate") for part in f.relative_to(root).parts):
                continue  # the pod's pre-collection gate runs are not study data (and are not in the artifact)
            kind = "gsm8k" if any(x.startswith("bridge") for x in f.parts) or "reset-lcpp" in f.parts else "bfcl"
            from analyze import _open_maybe_gz
            with _open_maybe_gz(Path(str(f).removesuffix(".gz"))) as fh:
                for line in fh:
                    r = json.loads(line)
                    tot[kind] += 1
                    cap[kind] += r["response"]["choices"][0].get("finish_reason") == "length"
    out["token_cap"] = {"requests": tot, "finish_reason_length": cap}
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
