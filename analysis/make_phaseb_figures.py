"""Figures for the replicated controlled sweep (Phase B). Reads
analysis/phaseb_findings.json and the production-default grid in
analysis/findings.json.

  fig_gradient_replicated  divergence by weight format: every history as a dot,
                           history-level mean with the two-way bootstrap
                           interval, and the production-default grid for
                           reference
  fig_margins              (a) flip hazard against the recompute-path top-two
                           logit margin, one line per format; (b) size of the
                           cache-hit change in that margin, by format

Usage: uv run python analysis/make_phaseb_figures.py -o paper/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED, GREY = "#2c7fb8", "#c0392b", "#7f8c8d"
RAMP = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]   # lighter = higher precision
FMTS = ["F16", "Q8_0", "Q4_K_M", "Q3_K_M"]
GRID_KEY = {"F16": "f16", "Q8_0": "q80", "Q4_K_M": "q4km", "Q3_K_M": "q3km"}


def fig_gradient(pb, f, out):
    sw = pb["sweep"]
    hl = sw["history_level"]; ci = sw["two_way_bootstrap"]["rate_ci95"]
    rh = sw["complete_random_histories"]
    grid = {}
    for c in f["configs"]:
        p = c["config"].split("-")
        if p[0] == "lcpp" and p[1] == "qwen7b":
            grid[p[2]] = 100 * c["cross_arm_main"]["rate"]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(FMTS))
    rng = np.random.default_rng(1)
    for i, fm in enumerate(FMTS):
        pts = [100 * sw["rate_by_history"][fm][str(h)] for h in rh]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(pts)), pts, s=9, color=GREY, alpha=0.6, linewidths=0, zorder=2)
    m = [100 * hl[fm]["mean"] for fm in FMTS]
    lo = [m[i] - 100 * ci[fm][0] for i, fm in enumerate(FMTS)]
    hi = [100 * ci[fm][1] - m[i] for i, fm in enumerate(FMTS)]
    ax.errorbar(x, m, yerr=[lo, hi], fmt="o-", color=BLUE, lw=2, ms=6, capsize=3, zorder=3,
                label=f"controlled, {len(rh)} histories")
    g = [grid.get(GRID_KEY[fm]) for fm in FMTS]
    ax.plot(x, g, "s--", color=RED, lw=1.5, ms=5, mfc="white", zorder=3, label="production default")
    ax.set_xticks(x); ax.set_xticklabels(FMTS, fontsize=8)
    ax.set_ylabel("path divergence\n(% of episodes)", fontsize=8)
    ax.set_ylim(0, 100); ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_gradient_replicated.pdf"); plt.close(fig)


def fig_margins(pb, out):
    mech = pb["mechanism_top5"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 1.85))
    for i, fm in enumerate(FMTS):
        if fm not in mech:
            continue
        hz = mech[fm]["flip_hazard_coarse"]
        bins = list(hz)
        ok = [hz[k]["positions"] >= 20 for k in bins]
        rates = np.array([hz[k]["flip_rate"] if o else np.nan for k, o in zip(bins, ok)], float)
        lo = np.array([rates[j] - hz[k]["ci95_episode_bootstrap"][0] if o else 0 for j, (k, o) in enumerate(zip(bins, ok))])
        hi = np.array([hz[k]["ci95_episode_bootstrap"][1] - rates[j] if o else 0 for j, (k, o) in enumerate(zip(bins, ok))])
        x = np.arange(len(bins)) + (i - 1.5) * 0.06          # small dodge so error bars do not overlap
        a.errorbar(x, rates, yerr=[lo, hi], fmt="o-", color=RAMP[i], lw=2, ms=4, capsize=2, elinewidth=1, label=fm)
    labels = ["<0.1", "0.1-0.3", "0.3-1", "1-3", ">3"]
    a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=8)
    a.set_xlabel("recompute-path top-two logit margin", fontsize=8)
    a.set_ylabel("share of positions that flip", fontsize=8)
    a.set_ylim(0, 0.7); a.tick_params(labelsize=8); a.grid(alpha=0.3); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=7, title="weights", title_fontsize=7)
    a.set_title("(a) flip hazard", fontsize=8, loc="left")
    xs = [i for i, fm in enumerate(FMTS) if fm in mech]
    for i in xs:
        q = mech[FMTS[i]]["abs_margin_perturbation_nonflip"]
        b.vlines(i, q["p10"], q["p90"], color=RAMP[i], lw=1.5)
        b.vlines(i, q["p25"], q["p75"], color=RAMP[i], lw=6)
        b.plot(i, q["median"], "o", color="white", mec=RAMP[i], ms=5, mew=1.5)
    b.set_yscale("log"); b.set_xticks(xs); b.set_xticklabels([FMTS[i] for i in xs], fontsize=8)
    b.set_ylabel("|change in margin| on the\ncache-hit path (logits)", fontsize=8)
    b.tick_params(labelsize=8); b.grid(axis="y", alpha=0.3, which="both"); b.set_axisbelow(True)
    b.set_title("(b) size of the cache-path perturbation", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(out / "fig_margins.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="paper/figures")
    ap.add_argument("--phaseb", default="analysis/phaseb_findings.json")
    ap.add_argument("--findings", default="findings.json")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pb = json.load(open(a.phaseb)); f = json.load(open(a.findings))
    if "sweep" in pb and pb["sweep"].get("complete_random_histories"):
        fig_gradient(pb, f, out)
    if pb.get("mechanism_top5"):
        fig_margins(pb, out)
    print("written:", sorted(p.name for p in out.glob("fig_*replicated*.pdf")) + sorted(p.name for p in out.glob("fig_margins.pdf")))


if __name__ == "__main__":
    main()
