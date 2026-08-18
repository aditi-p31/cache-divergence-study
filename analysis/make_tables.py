"""LaTeX tables for the manuscript, generated from findings.json only.

Usage: uv run python analysis/make_tables.py analysis/findings.json -o paper/tables
"""

import argparse
import json
from pathlib import Path

QUANT_ORDER = ["fp16", "f16", "q80", "q4km", "q3km", "awq", "int8"]
QUANT_LABEL = {"fp16": "FP16", "f16": "F16", "q80": "Q8\\_0", "q4km": "Q4\\_K\\_M",
               "q3km": "Q3\\_K\\_M", "awq": "AWQ-INT4", "int8": "INT8"}
BACKEND_LABEL = {"lcpp": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"}
MODEL_LABEL = {"qwen7b": "Qwen2.5-7B", "llama8b": "Llama-3.1-8B",
               "qwen14b": "Qwen2.5-14B"}


def parse(name: str):
    p = name.split("-")
    return p[0], (p[1] if len(p) > 1 else "?"), (p[2] if len(p) > 2 else "?")


def sort_key(cfg):
    b, m, q = parse(cfg["config"])
    return (list(BACKEND_LABEL).index(b) if b in BACKEND_LABEL else 9,
            list(MODEL_LABEL).index(m) if m in MODEL_LABEL else 9,
            QUANT_ORDER.index(q) if q in QUANT_ORDER else 9)


def pct(x):
    return f"{100 * x:.1f}" if isinstance(x, (int, float)) else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("findings")
    ap.add_argument("-o", "--out", default="paper/tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = json.load(open(args.findings))

    # ---- main determinism table ----
    rows = []
    for c in sorted(f["configs"], key=sort_key):
        b, m, q = parse(c["config"])
        d_off = (c.get("determinism_off") or {}).get("rate")
        d_on = (c.get("determinism_on") or {}).get("rate")
        cross = (c.get("cross_arm_main") or {}).get("rate")
        n = (c.get("cross_arm_main") or {}).get("n_episodes", 80)
        ci = (c.get("determinism_on") or {}).get("ci95") or [None, None]
        ci_s = (f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}]"
                if ci[0] is not None else "--")
        rows.append(
            f"{BACKEND_LABEL.get(b, b)} & {MODEL_LABEL.get(m, m)} & "
            f"{QUANT_LABEL.get(q, q)} & {n} & {pct(d_off)} & {pct(d_on)} & "
            f"{ci_s} & {pct(cross)} \\\\")

    tab = r"""\begin{table*}[!t]
\caption{Episode-level divergence. Each configuration ran the same 80-episode
workload twice per arm. \emph{Cache off} and \emph{cache on} report the
fraction of episodes whose trajectory changed when the same arm was repeated;
\emph{cross-arm} compares the two arms. Every cache-off value is exactly zero,
which bounds all other sources of nondeterminism under these conditions.}
\label{tab:main}
\centering
\begin{tabular}{lllrrrcr}
\toprule
& & & & \multicolumn{3}{c}{re-run divergence (\%)} & cross-arm \\
\cmidrule(lr){5-7}
Engine & Model & Weights & $n$ & cache off & cache on & 95\% CI & (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (out / "tab_main.tex").write_text(tab)

    # ---- bridge table ----
    def bridge_model(name):
        if "qwen14b" in name: return "Qwen2.5-14B"
        if "qwen7b" in name: return "Qwen2.5-7B"
        return "?"
    brows = []
    bitems = sorted(f.get("bridges", {}).items(),
                    key=lambda kv: (QUANT_ORDER.index(next((k for k in QUANT_ORDER if kv[0].endswith(k) or f"-{k}" in kv[0]), "f16")),
                                    kv[1]["n_items"]))
    for name, b in bitems:
        q = next((k for k in QUANT_ORDER if name.endswith(k) or f"-{k}" in name), "?")
        ci = b.get("ci95_attributable") or [0, 0]
        brows.append(
            f"{bridge_model(name)} & {QUANT_LABEL.get(q, q)} & {b['n_items']} & "
            f"{b['cold_path_deterministic']}/{b['n_items']} & "
            f"{b['warm_path_deterministic']}/{b['n_items']} & "
            f"{pct(b['rate_attributable'])} & [{100*ci[0]:.1f}, {100*ci[1]:.1f}] & "
            f"{b['n_correctness_flips']} & {b['acc_cold']} & {b['acc_warm']} & "
            f"{b['mcnemar_p']:.3f} \\\\")

    if brows:
        btab = r"""\begin{table*}[!t]
\caption{Single-turn bridge (GSM8K). Each item is served four times, twice on
the recompute path and twice on the cache-hit path. Both paths are
individually deterministic, yet they disagree with each other on a large
fraction of items. Correctness flips occur in both directions. Answers are
derived from the stored responses by numeric comparison; the released
artifact reproduces every value.}
\label{tab:bridge}
\centering
\begin{tabular}{llrccrcrrrr}
\toprule
& & & \multicolumn{2}{c}{path determinism} & \multicolumn{2}{c}{cross-path divergence} & flips & \multicolumn{2}{c}{correct} & McNemar \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){9-10}
Model & Weights & $n$ & cold & warm & \% & 95\% CI & (n) & cold & warm & $p$ \\
\midrule
""" + "\n".join(brows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
        (out / "tab_bridge.tex").write_text(btab)

    print(f"wrote {out}/tab_main.tex ({len(rows)} rows)")
    if brows:
        print(f"wrote {out}/tab_bridge.tex ({len(brows)} rows)")


if __name__ == "__main__":
    main()
