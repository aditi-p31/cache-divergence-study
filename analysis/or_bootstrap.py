"""Two-way (episode x history) bootstrap of the history-averaged GLMM trend slope b1 on
the replicated sweep. Each replicate resamples the 80 episodes and the 10 random histories
with replacement, refits the random-intercept model in every resampled history, and
averages b1 over histories, so the interval carries episode sampling as well as history
variation (the per-history t-interval does not). Slow (one fit is ~0.3 s), so the result
is cached in analysis/phaseb_or_bootstrap.json together with a fingerprint of the input
tensor; phaseb_stats.py reads it when the fingerprint matches.

Usage: uv run python analysis/or_bootstrap.py [--reps 2000] [--workers 6]
"""
import argparse, hashlib, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_stats as rs  # noqa: E402

Q = [0, 1, 2, 3]
SEED = 20261007


def fingerprint(D):
    return hashlib.sha256(np.ascontiguousarray(D).tobytes()).hexdigest()[:16]


def one_rep(args):
    D, rh, seed = args
    rng = np.random.default_rng(seed)
    ei = rng.integers(0, D.shape[0], D.shape[0])
    hi = [rh[i] for i in rng.integers(0, len(rh), len(rh))]
    cache, vals = {}, []
    for h in hi:
        if h not in cache:
            cache[h] = rs.glmm_fit(D[ei, h, :].astype(int), Q)["b1"]
        vals.append(cache[h])
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="analysis/phaseb_findings.tensor.npy")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("-o", "--out", default="analysis/phaseb_or_bootstrap.json")
    a = ap.parse_args()
    D = np.load(a.tensor)
    rh = [h for h in range(1, D.shape[1]) if not np.isnan(D[:, h, :]).any()]
    point = float(np.mean([rs.glmm_fit(D[:, h, :].astype(int), Q)["b1"] for h in rh]))
    seeds = np.random.SeedSequence(SEED).generate_state(a.reps)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        reps = list(ex.map(one_rep, [(D, rh, int(s)) for s in seeds], chunksize=20))
    reps = np.array(reps)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    out = {"tensor_fingerprint": fingerprint(D), "histories": rh, "replicates": a.reps, "seed": SEED,
           "b1_mean_over_histories": point, "b1_ci95_two_way": [float(lo), float(hi)],
           "or_per_step": math.exp(point), "or_ci95_two_way": [math.exp(lo), math.exp(hi)],
           "unit": "episodes and histories resampled independently; GLMM refitted in every resampled history"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
