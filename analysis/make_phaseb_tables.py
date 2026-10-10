"""LaTeX table for the replicated controlled sweep, from phaseb_findings.json.
Usage: uv run python analysis/make_phaseb_tables.py -o paper/tables
"""
import argparse
import json
from pathlib import Path

FMTS = ["F16", "Q8_0", "Q4_K_M", "Q3_K_M"]
TEX = {"F16": "F16", "Q8_0": "Q8\\_0", "Q4_K_M": "Q4\\_K\\_M", "Q3_K_M": "Q3\\_K\\_M"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="paper/tables")
    ap.add_argument("--phaseb", default="analysis/phaseb_findings.json")
    a = ap.parse_args()
    sw = json.load(open(a.phaseb))["sweep"]
    hl, ci = sw["history_level"], sw["two_way_bootstrap"]["rate_ci95"]
    dif = sw["two_way_bootstrap"]["diff_vs_F16"]
    k = len(sw["complete_random_histories"])
    canon = {fm: sw["rate_by_history"][fm]["0"] for fm in FMTS}
    rows = []
    for fm in FMTS:
        h = hl[fm]
        canon_s = "--" if canon[fm] is None else f"{100*canon[fm]:.1f}"
        d = dif.get(fm)
        d_s = "--" if d is None else f"{100*d['diff']:+.1f} [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}]"
        exceed = "--" if fm == "F16" else f"{sum(1 for x in sw['per_history_diffs_vs_F16'][fm] if x > 0)}/{k}"
        rows.append(f"{TEX[fm]} & {canon_s} & {100*h['mean']:.1f} & {100*h['sd_between_histories']:.1f} & "
                    f"{100*h['min']:.1f}--{100*h['max']:.1f} & [{100*ci[fm][0]:.1f}, {100*ci[fm][1]:.1f}] & {d_s} & {exceed} \\\\")
    tab = r"""\begin{table*}[!t]
\caption{Replicated controlled sweep: cache-enabled against cache-disabled path
divergence for Qwen2.5-7B under llama.cpp, server-level prompt cache disabled
from launch, a fresh server for every pass, 80 episodes per pass. Each random
history is an independent episode order shared by all four formats. The
interval resamples episodes and histories; the difference from F16 is paired
by episode and history.}
\label{tab:sweep}
\centering
\begin{tabular}{lrrrcccc}
\toprule
& canonical & \multicolumn{4}{c}{""" + str(k) + r""" random histories (\%)} & difference & histories \\
\cmidrule(lr){3-6}
Weights & history (\%) & mean & SD & range & 95\% CI & from F16 (points) & above F16 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "tab_sweep.tex").write_text(tab)
    print(f"wrote {out}/tab_sweep.tex")
    reset = reset_table(json.load(open(a.phaseb)).get("reset_control") or {})
    if reset:
        (out / "tab_reset.tex").write_text(reset)
        print(f"wrote {out}/tab_reset.tex")


def reset_table(rc):
    """Restored-state control: per format, three sessions; returns '' without data."""
    rows = []
    for fm in FMTS:
        r = rc.get(fm) or {}
        ss = r.get("sessions") or {}
        if len(ss) != 3:
            continue
        get = lambda key: "/".join(str(ss[s][key]) for s in ("1", "2", "3"))
        n = ss["1"]["items"]
        x = r["cross_session_token_identity"]
        ident = f"{x['cold_1']['identical_across_sessions']}/{x['cold_1']['items']} & {x['warm_1']['identical_across_sessions']}/{x['warm_1']['items']}"
        rows.append(f"{TEX[fm]} & {n} & {get('recompute_reproducible')} & {get('cached_reproducible')} & {get('paths_differ')} & {ident} \\\\")
    if not rows:
        return ""
    return r"""\begin{table*}[!t]
\caption{Restored-state control in every weight format, Qwen2.5-7B under
llama.cpp, GSM8K test items 0 to 99: three independent server sessions per format, each serving the same items four times (twice on
the recompute path, twice on the cache-hit path) with the cold state verified
from server telemetry on every request. The sessions serve the same items in
different orders. Counts are given per session; the last two columns count
items whose tokens are identical in all three sessions, out of 100.}
\label{tab:reset}
\centering
\small
\begin{tabular}{@{}lrccccc@{}}
\toprule
& & \multicolumn{2}{c}{path reproduces} & paths & \multicolumn{2}{c}{same in all sessions} \\
\cmidrule(lr){3-4}\cmidrule(lr){6-7}
Weights & items & recompute & cache hit & differ & recompute & cache hit \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


if __name__ == "__main__":
    main()
