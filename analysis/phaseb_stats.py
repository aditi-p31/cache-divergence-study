"""Phase B analysis: the controlled, replicated quantization sweep and the
controls the reviewers requested (Access-2026-42095 resubmission).

Inputs (pulled from the pod): revision/phaseB/pulled/results/
  B2/<fmt>_order{0,1}/arm_off     cache-off references (top-5 logprobs)
  B1/<fmt>_order{0..10}/arm_on    cache-on passes, fresh server each,
                                  order 0 canonical, 1..10 randomized
  B3/<fmt>_order{0,1}/arm_on      sentinel-isolated episodes
  B4/<fmt>_session{1,2,3}/        reset control, 100 GSM8K items
  B5/s<k>_<mode>/arm_*            single-stream latency modes

Outputs analysis/phaseb_findings.json. Divergence uses the paper's
definition (analyze.compare_episode). Episodes are matched by id; the
cache-off path is history-independent (verified here, not assumed).

Usage: uv run python analysis/phaseb_stats.py [--root revision/phaseB/pulled/results]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import _open_maybe_gz, compare_episode, load_raw  # noqa: E402
import revision_stats as rs  # noqa: E402

FORMATS = ["f16", "q80", "q4km", "q3km"]
LABEL = {"f16": "F16", "q80": "Q8_0", "q4km": "Q4_K_M", "q3km": "Q3_K_M", "q4deq": "q4deq"}
SCORE = {"f16": 0, "q80": 1, "q4km": 2, "q3km": 3}
HISTORIES = list(range(0, 11))
RANDOM_H = list(range(1, 11))
SEED = rs.SEED
B = rs.B


def ep_num(e):
    return int(e.rsplit("_", 1)[1])


def P(root, *parts):
    return Path(root, *parts, "raw_requests.jsonl")


def exists(root, *parts):
    p = P(root, *parts)
    return p.exists() or Path(str(p) + ".gz").exists()


def div_by_id(pa, pb, strict=False):
    """Divergence per episode, matched by episode id. strict=True requires both
    passes to hold the same episode set, so a missing episode can never shrink a
    denominator silently."""
    a, b = load_raw(Path(pa)), load_raw(Path(pb))
    if strict and set(a) != set(b):
        raise ValueError(f"episode sets differ: {pa} has {len(a)}, {pb} has {len(b)}; "
                         f"only in first {sorted(set(a) - set(b))[:5]}, only in second {sorted(set(b) - set(a))[:5]}")
    eps = sorted(set(a) & set(b), key=ep_num)
    return {e: compare_episode(a[e], b[e])["diverged"] for e in eps}


def off_ref(root, f, h):
    """Cache-off reference for history h. B2 checks that the recompute path does
    not depend on history; only if the pod flagged that it does (marker file) are
    the per-history references used (B2 order1 for h=1, B1off for h>=2)."""
    if not Path(root, "B2", f".off_per_history_{f}").exists() or h == 0:
        return P(root, "B2", f"{f}_order0", "arm_off")
    if h == 1:
        return P(root, "B2", f"{f}_order1", "arm_off")
    if not exists(root, "B1off", f"{f}_order{h}", "arm_off"):
        raise RuntimeError(f"cache-off output depends on history for {f}, but B1off/{f}_order{h} is missing")
    return P(root, "B1off", f"{f}_order{h}", "arm_off")


# --------------------------------------------------------------------------
# 1. off-path history independence (B2) and cross-machine reproduction
# --------------------------------------------------------------------------

def off_checks(root, refs):
    out = {}
    for f in FORMATS:
        r = {}
        if exists(root, "B2", f"{f}_order0", "arm_off") and exists(root, "B2", f"{f}_order1", "arm_off"):
            d = div_by_id(P(root, "B2", f"{f}_order1", "arm_off"), P(root, "B2", f"{f}_order0", "arm_off"), strict=True)
            r["order1_vs_order0_diverged"] = int(sum(d.values()))
            r["n"] = len(d)
            r["order1_vs_order0_identity"] = rs.float_identity(str(P(root, "B2", f"{f}_order1", "arm_off")),
                                                               str(P(root, "B2", f"{f}_order0", "arm_off")))
        ref = Path(refs, f"aug_grid_{f}_off", "raw_requests.jsonl")
        if (ref.exists() or Path(str(ref) + ".gz").exists()) and exists(root, "B2", f"{f}_order0", "arm_off"):
            d = div_by_id(P(root, "B2", f"{f}_order0", "arm_off"), ref)
            r["vs_august_grid_diverged"] = int(sum(d.values()))
        out[LABEL[f]] = r
    return out


# --------------------------------------------------------------------------
# 2. the replicated controlled sweep (B1 vs B2 order0)
# --------------------------------------------------------------------------

def sweep_tensor(root):
    """D[e, h, j]: episode e diverged between the cache-on pass of history h
    and the cache-off reference, for format j. Missing cells are NaN."""
    eps = None
    cols = {}
    for f in FORMATS:
        for h in HISTORIES:
            if not exists(root, "B1", f"{f}_order{h}", "arm_on"):
                continue
            d = div_by_id(P(root, "B1", f"{f}_order{h}", "arm_on"), off_ref(root, f, h), strict=True)
            if eps is None:
                eps = sorted(d, key=ep_num)
            elif set(d) != set(eps):
                raise ValueError(f"B1/{f}_order{h} has a different episode set from the first cell")
            cols[(f, h)] = d
    if eps is None:
        return None, None
    D = np.full((len(eps), len(HISTORIES), len(FORMATS)), np.nan)
    for (f, h), d in cols.items():
        D[:, h, FORMATS.index(f)] = [float(d[e]) for e in eps]
    return eps, D


def two_way_bootstrap(D, hs, n_boot=B, seed=SEED):
    """Resample episodes and histories independently (pigeonhole bootstrap,
    conservative); returns replicate format rates (n_boot x formats)."""
    rng = np.random.default_rng(seed)
    n_e = D.shape[0]
    hs = np.asarray(hs)
    reps = np.empty((n_boot, D.shape[2]))
    for b in range(n_boot):
        ei = rng.integers(0, n_e, n_e)
        hi = hs[rng.integers(0, len(hs), len(hs))]
        reps[b] = np.nanmean(D[ei][:, hi, :], axis=(0, 1))
    return reps


def sweep_analysis(eps, D):
    out = {"n_episodes": len(eps), "formats": [LABEL[f] for f in FORMATS]}
    have = ~np.isnan(D).all(axis=0)                       # histories x formats present
    out["histories_present"] = {LABEL[f]: [h for h in HISTORIES if have[h, j]] for j, f in enumerate(FORMATS)}
    per_hist = np.nanmean(D, axis=0)                      # histories x formats
    out["rate_by_history"] = {LABEL[f]: {str(h): (None if np.isnan(per_hist[h, j]) else float(per_hist[h, j]))
                                          for h in HISTORIES} for j, f in enumerate(FORMATS)}
    rh = [h for h in RANDOM_H if have[h].all()]
    out["complete_random_histories"] = rh
    if not rh:
        return out
    R = per_hist[rh]                                      # random histories x formats
    k = len(rh)
    tq = stats.t.ppf(0.975, k - 1) if k > 1 else float("nan")
    out["history_level"] = {
        LABEL[f]: {"mean": float(R[:, j].mean()), "sd_between_histories": float(R[:, j].std(ddof=1)) if k > 1 else None,
                   "t95": [float(R[:, j].mean() - tq * R[:, j].std(ddof=1) / math.sqrt(k)),
                           float(R[:, j].mean() + tq * R[:, j].std(ddof=1) / math.sqrt(k))] if k > 1 else None,
                   "min": float(R[:, j].min()), "max": float(R[:, j].max())}
        for j, f in enumerate(FORMATS)}
    reps = two_way_bootstrap(D, rh)
    lo, hi = 0.025, 0.975
    out["two_way_bootstrap"] = {
        "rate_ci95": {LABEL[f]: list(map(float, np.quantile(reps[:, j], [lo, hi]))) for j, f in enumerate(FORMATS)},
        "diff_vs_F16": {LABEL[f]: {"diff": float(R[:, j].mean() - R[:, 0].mean()),
                                   "ci95": list(map(float, np.quantile(reps[:, j] - reps[:, 0], [lo, hi])))}
                        for j, f in enumerate(FORMATS) if j > 0},
        "replicates": B, "seed": SEED, "unit": "episodes and histories resampled independently"}
    # each history separately: the paired per-history difference F16 -> format
    out["per_history_diffs_vs_F16"] = {LABEL[f]: [float(R[i, j] - R[i, 0]) for i in range(k)]
                                       for j, f in enumerate(FORMATS) if j > 0}
    out["histories_where_every_quantized_format_exceeds_F16"] = int(sum(all(R[i, j] > R[i, 0] for j in range(1, 4)) for i in range(k)))
    # each step to the next coarser format, paired by episode and history (the shape of the gradient)
    out["adjacent_steps"] = {f"{LABEL[FORMATS[j]]}-{LABEL[FORMATS[j - 1]]}": {
        "diff": float(R[:, j].mean() - R[:, j - 1].mean()),
        "ci95": list(map(float, np.quantile(reps[:, j] - reps[:, j - 1], [lo, hi]))),
        "histories_up": int(sum(R[i, j] > R[i, j - 1] for i in range(k))),
        "histories_down": int(sum(R[i, j] < R[i, j - 1] for i in range(k)))} for j in range(1, len(FORMATS))}
    # history sensitivity of individual episodes
    Dr = D[:, rh, :]
    vary = (Dr.max(axis=1) != Dr.min(axis=1))             # episodes x formats
    out["episodes_whose_status_varies_across_histories"] = {LABEL[f]: int(vary[:, j].sum()) for j, f in enumerate(FORMATS)}
    # paired trend: GLMM with episode random intercept on per-(episode, format)
    # pooled histories, and an exact permutation test stratified by (episode, history)
    q = [SCORE[f] for f in FORMATS]
    # The episode is the unit that recurs across histories, so the permutation acts
    # on each episode's counts summed over histories, with ONE format permutation
    # per episode shared by all its histories. Stratifying by (episode, history)
    # would treat an episode-by-format effect that persists across histories as
    # ten independent observations and make p too small.
    C = Dr.sum(axis=1).astype(int)                        # episodes x formats, counts over histories
    out["exact_permutation_episode_counts"] = exact_perm_trend_counts(C, q)
    qc = np.asarray(q, float) - np.mean(q)
    slope_reps = (reps @ qc) / (qc @ qc)
    out["slope_per_step"] = {"points": float((R.mean(axis=0) @ qc) / (qc @ qc)),
                             "ci95_two_way_bootstrap": list(map(float, np.quantile(slope_reps, [lo, hi])))}
    canon = D[:, 0, :] if have[0].all() else None
    if canon is not None:
        out["glmm_canonical_history"] = rs.glmm_fit(canon.astype(int), q)
    # GLMM per random history, summarised across histories (history as replication unit)
    b1s = [rs.glmm_fit(D[:, h, :].astype(int), q)["b1"] for h in rh]
    out["glmm_b1_by_history"] = {"values": [float(x) for x in b1s], "mean": float(np.mean(b1s)),
                                 "sd": float(np.std(b1s, ddof=1)) if k > 1 else None,
                                 "t95": [float(np.mean(b1s) - tq * np.std(b1s, ddof=1) / math.sqrt(k)),
                                         float(np.mean(b1s) + tq * np.std(b1s, ddof=1) / math.sqrt(k))] if k > 1 else None,
                                 "or_per_step_mean": float(math.exp(np.mean(b1s)))}
    return out


def exact_perm_trend_counts(C, q):
    """Exact within-episode permutation test on per-episode counts. Under the
    null of no format effect, the formats' count columns of an episode are
    exchangeable: each of the J! permutations is equally likely. T = sum_e sum_j
    q_j C_ej; its exact null distribution is the convolution over episodes."""
    import itertools
    C = np.asarray(C, int)
    qi = [int(v) for v in q]
    perms = list(itertools.permutations(qi))
    dist = {0: 1.0}
    for row in C:
        vals = {}
        for pq in perms:
            v = int(np.dot(pq, row))
            vals[v] = vals.get(v, 0) + 1
        nd = {}
        for t, pt in dist.items():
            for v, c in vals.items():
                nd[t + v] = nd.get(t + v, 0.0) + pt * c / len(perms)
        dist = nd
    t_obs = int(sum(np.dot(qi, row) for row in C))
    ts = np.array(sorted(dist)); ps = np.array([dist[t] for t in ts])
    mean = float(ts @ ps)
    return {"T": t_obs, "null_mean": mean, "n_episodes": int(len(C)),
            "p_one_sided_increasing": float(ps[ts >= t_obs].sum()),
            "p_two_sided": float(min(1.0, ps[np.abs(ts - mean) >= abs(t_obs - mean) - 1e-9].sum()))}


# --------------------------------------------------------------------------
# 3. mechanism with top-5 logprobs: true top-1/top-2 margins
# --------------------------------------------------------------------------

def _content(rec):
    return (rec["response"]["choices"][0].get("logprobs") or {}).get("content") or []


def _margin(tok):
    tops = sorted((t["logprob"] for t in (tok.get("top_logprobs") or [])), reverse=True)
    return (tops[0] - tops[1]) if len(tops) >= 2 else None


def _merged(rec):
    """llama.cpp folds a token that ends inside a UTF-8 character into the next
    logprobs entry, so content can be shorter than completion_tokens; positions
    are then no longer one per token and margins of merged entries are not the
    margin of the decision that was taken."""
    ct = int(((rec["response"].get("usage") or {}).get("completion_tokens")) or 0)
    return len(_content(rec)) < ct


def margin_records(path_on, path_off):
    """At every shared position (identical prompt, identical preceding tokens)
    of a request served on the cache-hit path (cache_n > 0 on the cache-enabled
    pass): recompute-path margin m_off, cache-path margin m_on, and whether the
    emitted token flipped there. The walk of an episode ends at the first flip,
    at a prompt mismatch, or at a request with merged UTF-8 entries."""
    A, Bp = load_raw(Path(path_on)), load_raw(Path(path_off))
    rows = []
    for e in sorted(set(A) & set(Bp), key=ep_num):
        stop = False
        for ra, rb in zip(A[e], Bp[e]):
            if ra["prompt_sha256"] != rb["prompt_sha256"] or _merged(ra) or _merged(rb):
                break
            exposed = rs._cache_n(ra) > 0
            for ta, tb in zip(_content(ra), _content(rb)):
                ma, mb = _margin(ta), _margin(tb)
                flip = ta["id"] != tb["id"]
                if exposed and ma is not None and mb is not None:
                    rows.append((e, mb, ma, flip))
                if flip:
                    stop = True
                    break
            if stop or len(_content(ra)) != len(_content(rb)):
                break
    return rows


MARGIN_BINS = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 1e9]
COARSE_BINS = [0, 0.1, 0.3, 1, 3, 1e9]          # for the figure: every bin holds enough positions


def _hazard(rows, n_boot=2000, seed=SEED, bins=MARGIN_BINS):
    """Flip hazard by recompute-path margin bin, with an episode-cluster
    bootstrap interval per bin (positions within an episode are dependent)."""
    eps = sorted({r[0] for r in rows}, key=ep_num)
    by_e = {e: [r for r in rows if r[0] == e] for e in eps}
    out = {}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(eps), size=(n_boot, len(eps)))
    for lo, hi in zip(bins[:-1], bins[1:]):
        pe = np.array([sum(1 for r in by_e[e] if lo <= r[1] < hi) for e in eps], float)
        fe = np.array([sum(1 for r in by_e[e] if lo <= r[1] < hi and r[3]) for e in eps], float)
        n, k = int(pe.sum()), int(fe.sum())
        ci = None
        if n >= 20:
            num, den = fe[draws].sum(axis=1), pe[draws].sum(axis=1)
            ok = den > 0
            ci = list(map(float, np.quantile(num[ok] / den[ok], [.025, .975])))
        out[f"[{lo},{hi})"] = {"positions": n, "flips": k, "flip_rate": (k / n) if n else None, "ci95_episode_bootstrap": ci}
    return out


def _recompute_margins(path_off):
    """Top-two logit margin at every emitted token of a cache-off pass,
    grouped by episode (the recompute path, history-independent)."""
    A = load_raw(Path(path_off))
    return {e: [m for r in reqs for t in _content(r) if (m := _margin(t)) is not None] for e, reqs in A.items()}


def mechanism_analysis(root, fmts=None, onkey="B1", offkey="B2", canon_only=True):
    fmts = fmts or FORMATS
    out = {"note": "hazard and perturbation use the canonical history only, so each position counts once; "
                   "pooled-over-histories counts are descriptive"}
    per_ep_dm, per_ep_near = {}, {}
    for f in fmts:
        off = P(root, offkey, f"{f}_order0", "arm_off")
        if not exists(root, offkey, f"{f}_order0", "arm_off") or not exists(root, onkey, f"{f}_order0", "arm_on"):
            continue
        rows = margin_records(P(root, onkey, f"{f}_order0", "arm_on"), off)
        m_off = np.array([r[1] for r in rows]); m_on = np.array([r[2] for r in rows]); flip = np.array([r[3] for r in rows])
        dv = div_by_id(P(root, onkey, f"{f}_order0", "arm_on"), off, strict=True)
        flipped_eps = {r[0] for r in rows if r[3]}
        nf = ~flip
        dm = np.abs(m_on[nf] - m_off[nf])
        q = lambda a, x: float(np.quantile(a, x)) if len(a) else None
        res = {"positions": int(len(rows)), "flips": int(flip.sum()),
               "divergent_episodes": int(sum(dv.values())),
               "divergent_episodes_with_located_flip": int(sum(1 for e, d in dv.items() if d and e in flipped_eps)),
               "margin_at_flips": {"median": q(m_off[flip], .5), "max": float(m_off[flip].max()) if flip.any() else None},
               "abs_margin_perturbation_nonflip": {"mean": float(dm.mean()), "median": q(dm, .5), "p10": q(dm, .1),
                                                   "p25": q(dm, .25), "p75": q(dm, .75), "p90": q(dm, .9), "p99": q(dm, .99)},
               "flip_hazard_by_recompute_margin": _hazard(rows),
               "flip_hazard_coarse": _hazard(rows, bins=COARSE_BINS)}
        pooled = []
        for h in HISTORIES[1:]:
            if exists(root, onkey, f"{f}_order{h}", "arm_on"):
                ref = off_ref(root, f, h) if onkey == "B1" else off
                pooled += margin_records(P(root, onkey, f"{f}_order{h}", "arm_on"), ref)
        if pooled:
            pf = np.array([r[3] for r in pooled]); pm = np.array([r[1] for r in pooled])
            res["pooled_random_histories"] = {"positions": int(len(pooled)), "flips": int(pf.sum()),
                                              "flips_by_bin": {f"[{lo},{hi})": int(pf[(pm >= lo) & (pm < hi)].sum())
                                                               for lo, hi in zip(MARGIN_BINS[:-1], MARGIN_BINS[1:])}}
        em = {}
        for e, a, b, fl in rows:
            if not fl:
                em.setdefault(e, []).append(abs(b - a))
        per_ep_dm[f] = {e: float(np.median(v)) for e, v in em.items()}
        rm = _recompute_margins(off)
        per_ep_near[f] = {e: float(np.mean(np.array(v) < 0.1)) for e, v in rm.items() if v}
        allm = np.array([m for v in rm.values() for m in v])
        res["recompute_margins_all_tokens"] = {"tokens": int(len(allm)), "median": float(np.median(allm)),
                                               "share_below_0.1": float((allm < 0.1).mean()), "share_below_0.5": float((allm < 0.5).mean())}
        out[LABEL[f]] = res
    labels = [LABEL[f] for f in fmts if LABEL[f] in out]
    for name, src in (("episode_median_abs_margin_perturbation", per_ep_dm), ("episode_share_recompute_margin_below_0.1", per_ep_near)):
        keys = [f for f in fmts if LABEL[f] in labels]
        if len(keys) < 2:
            continue
        common = sorted(set.intersection(*[set(src[f]) for f in keys]), key=ep_num)
        if not common:
            continue
        M = np.array([[src[f][e] for f in keys] for e in common])
        r = {"formats": labels, "n_episodes": len(common), "means": list(map(float, M.mean(axis=0)))}
        if len(keys) >= 3 and len(common) >= 3:
            r["page_p_increasing"] = float(stats.page_trend_test(M).pvalue)
            r["page_p_decreasing"] = float(stats.page_trend_test(M[:, ::-1]).pvalue)
        rng = np.random.default_rng(SEED)
        reps = M[rng.integers(0, len(common), size=(B, len(common)))].mean(axis=1)
        r["means_ci95_episode_bootstrap"] = [list(map(float, np.quantile(reps[:, j], [.025, .975]))) for j in range(M.shape[1])]
        if np.all(reps[:, 0] > 0):
            r["last_over_first"] = float(M[:, -1].mean() / M[:, 0].mean())
            r["last_over_first_ci95"] = list(map(float, np.quantile(reps[:, -1] / reps[:, 0], [.025, .975])))
        out[name] = r
    return out


# --------------------------------------------------------------------------
# 4. sentinel isolation (B3), reset control (B4), latency (B5)
# --------------------------------------------------------------------------

def sentinel_analysis(root):
    out = {}
    for f in FORMATS:
        r = {}
        if exists(root, "B3", f"{f}_order0", "arm_on"):
            for o in (1, 2):
                if exists(root, "B3", f"{f}_order{o}", "arm_on"):
                    d = div_by_id(P(root, "B3", f"{f}_order{o}", "arm_on"), P(root, "B3", f"{f}_order0", "arm_on"), strict=True)
                    r[f"order{o}_vs_order0_diverged"] = int(sum(d.values())); r["n"] = len(d)
                    r[f"order{o}_vs_order0_identity"] = rs.float_identity(str(P(root, "B3", f"{f}_order{o}", "arm_on")),
                                                                          str(P(root, "B3", f"{f}_order0", "arm_on")))
            off = P(root, "B2", f"{f}_order0", "arm_off")
            if exists(root, "B2", f"{f}_order0", "arm_off"):
                c = div_by_id(P(root, "B3", f"{f}_order0", "arm_on"), off)
                r["cross_arm_under_isolation"] = {"k": int(sum(c.values())), "n": len(c)}
            for o in (0, 1, 2):
                v = Path(root, "B3", f"{f}_order{o}", "arm_on", "validation.json")
                if v.exists():
                    r[f"validation_order{o}"] = json.load(open(v))["ok"]
        out[LABEL[f]] = r
    return out


def reset_analysis(root):
    out = {}
    cold_c, warm_c = [], []
    for f in FORMATS:
        sess = {}
        texts = {}
        for s in (1, 2, 3):
            d = Path(root, "B4", f"{f}_session{s}")
            if not (d / "summary.json").exists():
                continue
            rows = json.load(open(d / "summary.json"))
            sess[str(s)] = {"items": len(rows), "recompute_reproducible": sum(r["cold_reproducible"] for r in rows),
                            "cached_reproducible": sum(r["warm_reproducible"] for r in rows),
                            "paths_differ": sum(r["cache_effect"] for r in rows)}
            for line in _open_maybe_gz(d / "raw_requests.jsonl"):
                r = json.loads(line)
                (cold_c if r["phase"].startswith("cold") else warm_c).append(rs._cache_n(r))
                texts.setdefault((r["item_idx"], r["phase"]), {})[s] = tuple(t["id"] for t in _content(r))
        cross = {}
        for ph in ("cold_1", "warm_1"):
            keys = [k for k in texts if k[1] == ph and len(texts[k]) >= 2]
            cross[ph] = {"items": len(keys), "identical_across_sessions": sum(len(set(texts[k].values())) == 1 for k in keys)}
        out[LABEL[f]] = {"sessions": sess, "cross_session_token_identity": cross}
    if cold_c:
        out["telemetry"] = {"cold_requests": len(cold_c), "cold_max_cached_tokens": int(max(cold_c)),
                            "warm_requests": len(warm_c), "warm_min_cached_tokens": int(min(warm_c)),
                            "warm_median_cached_tokens": float(np.median(warm_c))}
    return out


def latency_analysis(root):
    out = {}
    for mode, arm in (("promptcache_default", "on"), ("promptcache_disabled", "on"), ("recompute", "off")):
        ss = []
        for s in (1, 2, 3):
            d = Path(root, "B5", f"s{s}_{mode}", f"arm_{arm}")
            if exists(root, "B5", f"s{s}_{mode}", f"arm_{arm}"):
                t = rs.pass_timing(d / "raw_requests.jsonl")
                su = next(d.glob("*.startup_s"), None)
                t["server_startup_s"] = float(su.read_text()) if su else None
                ss.append(t)
        if ss:
            out[mode] = {"sessions": len(ss),
                         "latency_s_mean": [x["latency_s_mean"] for x in ss],
                         "prefill_ms_mean": [x["prefill_ms_mean"] for x in ss],
                         "pass_minutes": [x["pass_minutes"] for x in ss],
                         "requests": [x["requests"] for x in ss],
                         "server_startup_s": [x["server_startup_s"] for x in ss]}
    return out


def _series_in_execution_order(pass_a, pass_b, run_meta_dir):
    """0/1 divergence per episode in the order the episodes actually ran
    (taken from run_meta.json, never from episode ids)."""
    d = div_by_id(pass_a, pass_b)
    order = json.load(open(Path(run_meta_dir, "run_meta.json")))["order"]
    return [int(d[e]) for e in order if e in d]


def _perm_trend(x, n_perm=20000, seed=SEED):
    x = np.asarray(x, float); pos = np.arange(len(x)) - (len(x) - 1) / 2
    t = float(pos @ x)
    rng = np.random.default_rng(seed)
    null = np.array([pos @ rng.permutation(x) for _ in range(n_perm)])
    return float((np.sum(np.abs(null) >= abs(t)) + 1) / (n_perm + 1))


def b6_analysis(root):
    out = {"histories": {}}
    rates = []
    for k in range(1, 6):
        d = Path(root, "B6", f"q4km_default_order{k}")
        if not (d / "main" / "arm_on").exists():
            continue
        x = _series_in_execution_order(d / "main" / "arm_on" / "raw_requests.jsonl",
                                       d / "repeat" / "arm_on" / "raw_requests.jsonl", d / "main" / "arm_on")
        cell = rs.summarize_cell(np.array(x))
        qn = len(x) // 4
        q = [int(sum(x[i * qn:(i + 1) * qn if i < 3 else len(x)])) for i in range(4)]
        ovf = sorted({e for v in ("validation_main.json", "validation_repeat.json") if (d / v).exists()
                      for e in json.load(open(d / v)).get("context_overflow_episodes", [])})
        out["histories"][str(k)] = {"context_overflow_episodes": ovf,
                                    "rate": cell["rate"], "k": cell["k"], "n": cell["n"],
                                    "reported95": cell.get("reported95"), "block_used": cell.get("block_used"),
                                    "quartiles_in_execution_order": q, "trend_p_two_sided": _perm_trend(x)}
        rates.append(cell["rate"])
    # an order that was run twice on fresh servers (a kept earlier attempt): same history, independent processes
    reps = {}
    for k in range(1, 6):
        d = Path(root, "B6", f"q4km_default_order{k}")
        for prev in sorted(Path(root, "_failed").glob(f"B6_q4km_default_order{k}.a*/q4km_default_order{k}")):
            row = {}
            for p in ("main", "repeat"):
                a_, b_ = prev / p / "arm_on" / "raw_requests.jsonl", d / p / "arm_on" / "raw_requests.jsonl"
                if (a_.exists() or Path(str(a_) + ".gz").exists()) and (b_.exists() or Path(str(b_) + ".gz").exists()):
                    dv = div_by_id(a_, b_, strict=True)
                    row[p] = dict(rs.float_identity(str(a_), str(b_)), episodes=len(dv), diverged=int(sum(dv.values())))
            if row:
                reps[str(k)] = row
    out["fresh_server_replicates"] = reps
    if rates:
        r = np.array(rates); k = len(r)
        tq = stats.t.ppf(0.975, k - 1) if k > 1 else float("nan")
        out["summary"] = {"n_histories": k, "mean": float(r.mean()), "sd": float(r.std(ddof=1)) if k > 1 else None,
                          "min": float(r.min()), "max": float(r.max()),
                          "t95": [float(r.mean() - tq * r.std(ddof=1) / math.sqrt(k)), float(r.mean() + tq * r.std(ddof=1) / math.sqrt(k))] if k > 1 else None,
                          "canonical_order_reference": "results-repair/cacheram/cacheram-default (38.8%)"}
    return out


def gate_records(root):
    """Verdicts the pod wrote while the run was in progress (gates must pass; checks are
    informative): the comparisons against the August passes and between replicates."""
    out = {}
    for kind in ("gates", "checks"):
        for f in sorted(Path(root, kind).glob("*.json")):
            try:
                r = json.load(open(f))
            except (OSError, ValueError):
                continue
            keep = {k: r[k] for k in ("n_diverged", "episodes", "shared_tokens", "shared_tokens_top1_logprob_differs",
                                      "max_abs_logprob_diff", "ok") if k in r}
            out[f"{kind}/{f.stem}"] = keep
    return out


def token_cap(root):
    """Requests that ended because they reached the generation cap, by workload."""
    tot, cap = {"bfcl": 0, "gsm8k": 0}, {"bfcl": 0, "gsm8k": 0}
    for f in Path(root).rglob("raw_requests.jsonl*"):
        if "_staging" in f.parts or "_failed" in f.parts or "references" in f.parts:
            continue
        kind = "gsm8k" if "B4" in f.parts else "bfcl"
        with _open_maybe_gz(Path(str(f).removesuffix(".gz"))) as fh:
            for line in fh:
                r = json.loads(line)
                tot[kind] += 1
                cap[kind] += r["response"]["choices"][0].get("finish_reason") == "length"
    return {"requests": tot, "finish_reason_length": cap}


def b1r_analysis(root):
    """Process-level reproducibility: an independently started server at the canonical
    history against the B1 canonical pass (Q4_K_M: the B0 top-5 cache-on pass, run on its
    own server before B1)."""
    out = {}
    pairs = {f: (P(root, "B1R", f"{f}_order0_rep", "arm_on"), P(root, "B1", f"{f}_order0", "arm_on")) for f in ("f16", "q80", "q3km")}
    pairs["q4km"] = (P(root, "B0", "q4km_on_lp5", "arm_on"), P(root, "B1", "q4km_order0", "arm_on"))
    for f, (a, b) in pairs.items():
        if not (Path(str(a)).exists() or Path(str(a) + ".gz").exists()):
            continue
        d = div_by_id(a, b, strict=True)
        out[LABEL[f]] = dict(rs.float_identity(str(a), str(b)), episodes=len(d), diverged=int(sum(d.values())))
    return out


def b7_analysis(root):
    """Dequantized Q4_K_M weights served on the F16 kernels (q4deq), against F16 and
    Q4_K_M on the same histories: path divergence per history, the paired differences
    (bootstrap over episodes and histories), and the margin mechanism at the canonical
    history (perturbation size, near-tie share, flip margins)."""
    out = {"histories": {}}
    off = P(root, "B7", "q4deq_order0", "arm_off")
    if not exists(root, "B7", "q4deq_order0", "arm_off"):
        return out
    cols, eps = {}, None
    for h in range(0, 4):
        row = {}
        if exists(root, "B7", f"q4deq_order{h}", "arm_on"):
            d = div_by_id(P(root, "B7", f"q4deq_order{h}", "arm_on"), off, strict=True)
            row["q4deq"] = float(np.mean(list(d.values()))); cols[("q4deq", h)] = d
        for f in ("f16", "q4km"):
            if exists(root, "B1", f"{f}_order{h}", "arm_on") and exists(root, "B2", f"{f}_order0", "arm_off"):
                d = div_by_id(P(root, "B1", f"{f}_order{h}", "arm_on"), off_ref(root, f, h), strict=True)
                row[f] = float(np.mean(list(d.values()))); cols[(f, h)] = d
        out["histories"][str(h)] = row
    paired = [h for h, r in out["histories"].items() if all(k in r for k in ("f16", "q4deq", "q4km"))]
    out["histories_paired"] = paired
    out["mean_over_histories"] = {k: (float(np.mean([out["histories"][h][k] for h in paired])) if paired else None)
                                  for k in ("f16", "q4deq", "q4km")}
    if paired:
        hs = [int(h) for h in paired]
        eps = sorted(cols[("q4deq", hs[0])], key=ep_num)
        D = np.array([[[float(cols[(f, h)][e]) for f in ("f16", "q4deq", "q4km")] for h in hs] for e in eps])
        reps = two_way_bootstrap(D, list(range(len(hs))))
        lo, hi = 0.025, 0.975
        R = D.mean(axis=0)
        out["paired_diffs"] = {
            "q4deq_minus_f16": {"diff": float(R[:, 1].mean() - R[:, 0].mean()),
                                "ci95": list(map(float, np.quantile(reps[:, 1] - reps[:, 0], [lo, hi])))},
            "q4km_minus_q4deq": {"diff": float(R[:, 2].mean() - R[:, 1].mean()),
                                 "ci95": list(map(float, np.quantile(reps[:, 2] - reps[:, 1], [lo, hi])))},
            "share_of_f16_to_q4km_gap": float((R[:, 1].mean() - R[:, 0].mean()) / (R[:, 2].mean() - R[:, 0].mean()))
            if R[:, 2].mean() != R[:, 0].mean() else None,
            "unit": "episodes and histories resampled independently"}
    # the dequantized model is its own model: its recompute path against the F16 and Q4_K_M recompute paths
    out["recompute_path_vs"] = {}
    for f in ("f16", "q4km"):
        if exists(root, "B2", f"{f}_order0", "arm_off"):
            d = div_by_id(off, P(root, "B2", f"{f}_order0", "arm_off"), strict=True)
            out["recompute_path_vs"][LABEL[f]] = {"diverged": int(sum(d.values())), "episodes": len(d)}
    mech = mechanism_analysis(root, fmts=["q4deq"], onkey="B7", offkey="B7")
    if "q4deq" in mech:
        m = mech["q4deq"]
        out["mechanism_canonical"] = {k: m[k] for k in ("positions", "flips", "divergent_episodes",
                                                        "divergent_episodes_with_located_flip", "margin_at_flips",
                                                        "abs_margin_perturbation_nonflip", "recompute_margins_all_tokens")}
        out["perturbation_q4deq"] = m["abs_margin_perturbation_nonflip"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="revision/phaseB/pulled/results")
    ap.add_argument("--refs", default="revision/phaseB/pulled/references")
    ap.add_argument("-o", "--out", default="analysis/phaseb_findings.json")
    a = ap.parse_args()
    out = {"meta": {"seed": SEED, "bootstrap_replicates": B, "root": a.root}}
    out["off_path_checks"] = off_checks(a.root, a.refs)
    eps, D = sweep_tensor(a.root)
    if D is not None:
        out["sweep"] = sweep_analysis(eps, D)
        np.save(Path(a.out).with_suffix(".tensor.npy"), D)
        # two-way bootstrap of the history-averaged GLMM slope: slow, computed by or_bootstrap.py
        # and used only when it was computed from exactly this tensor
        ob = Path(a.out).with_name("phaseb_or_bootstrap.json")
        if ob.exists():
            import hashlib
            r = json.load(open(ob))
            if r.get("tensor_fingerprint") == hashlib.sha256(np.ascontiguousarray(D).tobytes()).hexdigest()[:16]:
                out["sweep"]["glmm_b1_two_way_bootstrap"] = r
    out["mechanism_top5"] = mechanism_analysis(a.root)
    out["sentinel_isolation"] = sentinel_analysis(a.root)
    out["reset_control"] = reset_analysis(a.root)
    out["latency"] = latency_analysis(a.root)
    out["process_replicate_canonical"] = b1r_analysis(a.root)
    out["gate_records"] = gate_records(a.root)
    out["token_cap"] = token_cap(a.root)
    out["production_default_randomized_histories"] = b6_analysis(a.root)
    out["dequantized_weights_control"] = b7_analysis(a.root)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
