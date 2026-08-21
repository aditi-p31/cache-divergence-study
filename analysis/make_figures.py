"""Figures for the manuscript. Reads findings.json and repair_findings.json only.

Two figures, each earning its place:
  fig_mechanism  the paper's central argument in one panel: cache-off is
                 reproducible, cache-on is not, and the difference is a
                 configuration setting rather than a property of caching.
  fig_gradient   quantization amplifies the cache-versus-recompute difference,
                 shown for both the original and the controlled measurement so
                 the reader can see the pattern is not an artifact.

Usage: uv run python analysis/make_figures.py -o paper/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, GREY = "#2c7fb8", "#c0392b", "#7f8c8d"


def fig_mechanism(rep, out: Path):
    cr = rep["cacheram"]
    reset = rep["reset_control_llamacpp"]
    order = rep["ordering"]

    off = order["lcpp-orderB"]["cross_arm"]  # placeholder, replaced below
    off_rate = rep["cacheoff_rerun"]["rate"]
    off_ci = rep["cacheoff_rerun"]["ci95"]
    labels = ["cache off\n(re-run)", "cache on\n(re-run,\ncache-ram 0)",
              "cache on\n(re-run,\ncache-ram default)", "cache on vs off\n(same run)"]
    vals = [100 * off_rate,
            100 * cr["cacheram-zero"]["rate"],
            100 * cr["cacheram-default"]["rate"],
            100 * order["lcpp-orderB"]["cross_arm"]["rate"]]
    errs = [[100 * (off_rate - off_ci[0]), 100 * (off_ci[1] - off_rate)],
            [100 * (cr["cacheram-zero"]["rate"] - cr["cacheram-zero"]["ci95"][0]),
             100 * (cr["cacheram-zero"]["ci95"][1] - cr["cacheram-zero"]["rate"])],
            [100 * (cr["cacheram-default"]["rate"] - cr["cacheram-default"]["ci95"][0]),
             100 * (cr["cacheram-default"]["ci95"][1] - cr["cacheram-default"]["rate"])],
            [100 * (order["lcpp-orderB"]["cross_arm"]["rate"] - order["lcpp-orderB"]["cross_arm"]["ci95"][0]),
             100 * (order["lcpp-orderB"]["cross_arm"]["ci95"][1] - order["lcpp-orderB"]["cross_arm"]["rate"])]]
    colors = [BLUE, BLUE, RED, GREY]

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    x = range(len(vals))
    ax.bar(x, vals, color=colors, width=0.6,
           yerr=[[e[0] for e in errs], [e[1] for e in errs]],
           capsize=4, error_kw={"lw": 1, "ecolor": "#333"})
    for i, v in enumerate(vals):
        ax.text(i, v + errs[i][1] + 2.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("episodes differing (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig_mechanism.pdf")
    plt.close(fig)


def fig_gradient(f, rep, out: Path):
    order = ["f16", "q80", "q4km", "q3km"]
    lab = {"f16": "F16", "q80": "Q8_0", "q4km": "Q4_K_M", "q3km": "Q3_K_M"}
    orig = {}
    for c in f["configs"]:
        p = c["config"].split("-")
        if p[0] == "lcpp" and p[1] == "qwen7b":
            orig[p[2]] = 100 * c["cross_arm_main"]["rate"]
    ctrl = {k: 100 * v["rate"] for k, v in rep["gradient_controlled"].items()}

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs = [i for i, k in enumerate(order) if k in orig]
    ax.plot(xs, [orig[order[i]] for i in xs], "o-", color=RED,
            label="as collected")
    xs2 = [i for i, k in enumerate(order) if k in ctrl]
    ax.plot(xs2, [ctrl[order[i]] for i in xs2], "s--", color=BLUE,
            label="controlled cache configuration")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab[k] for k in order])
    ax.set_xlabel("weight format (coarser to the right)")
    ax.set_ylabel("cache on vs off, episodes differing (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_gradient.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="paper/figures")
    ap.add_argument("--findings", default="analysis/findings.json")
    ap.add_argument("--repair", default="analysis/repair_findings.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = json.load(open(a.findings))
    rep = json.load(open(a.repair))
    fig_mechanism(rep, out)
    fig_gradient(f, rep, out)
    print("figures written:")
    for p in sorted(out.glob("*.pdf")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
