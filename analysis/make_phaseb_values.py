"""Every Phase B number quoted in the manuscript text, as LaTeX macros generated
from analysis/phaseb_findings.json, so that no value is copied by hand.

The text cites a value as \\pbv{sweep.Q4.mean}. A key that does not exist stops
the compile with an error rather than printing nothing.

Usage: uv run python analysis/make_phaseb_values.py [-o paper/tables/phaseb_values.tex] [--list]
"""
import argparse
import json
import math
from pathlib import Path

FMT_KEY = {"F16": "F16", "Q8_0": "Q8", "Q4_K_M": "Q4", "Q3_K_M": "Q3"}
BIN_KEYS = ["[0,0.01)", "[0.01,0.03)", "[0.03,0.1)", "[0.1,0.3)", "[0.3,1)", "[1,3)", "[3,1000000000.0)"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def num(x, d=2):
    return f"{x:.{d}f}"


def sig(x, n=2):
    """n significant figures, plain decimal (for small logit perturbations)."""
    if x == 0:
        return "0"
    d = max(0, n - 1 - int(math.floor(math.log10(abs(x)))))
    return f"{x:.{d}f}"


def pval(p):
    """p-values: three decimals down to 0.001, otherwise a power of ten in math mode."""
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"\\ensuremath{{{m}\\times 10^{{{int(e)}}}}}"


def signed_pp(x):
    """percentage points; a minus sign only when negative (typeset as a math minus)."""
    return f"{100 * x:.1f}".replace("-", "\\ensuremath{-}")


def collect(pb):
    v = {}
    sw = pb.get("sweep") or {}
    if sw.get("history_level"):
        hl, ci = sw["history_level"], sw["two_way_bootstrap"]["rate_ci95"]
        dif = sw["two_way_bootstrap"]["diff_vs_F16"]
        v["sweep.k"] = str(len(sw["complete_random_histories"]))
        v["sweep.allabove"] = str(sw["histories_where_every_quantized_format_exceeds_F16"])
        for lab, k in FMT_KEY.items():
            h = hl[lab]
            v[f"sweep.{k}.mean"] = pct(h["mean"])
            if h["sd_between_histories"] is not None:
                v[f"sweep.{k}.sd"] = pct(h["sd_between_histories"])
            v[f"sweep.{k}.min"] = pct(h["min"])
            v[f"sweep.{k}.max"] = pct(h["max"])
            v[f"sweep.{k}.cilo"] = pct(ci[lab][0])
            v[f"sweep.{k}.cihi"] = pct(ci[lab][1])
            c = sw["rate_by_history"][lab].get("0")
            if c is not None:
                v[f"sweep.{k}.canon"] = pct(c)
            v[f"sweep.{k}.vary"] = str(sw["episodes_whose_status_varies_across_histories"][lab])
            if lab in dif:
                v[f"sweep.{k}.diff"] = pct(dif[lab]["diff"])
                v[f"sweep.{k}.difflo"] = signed_pp(dif[lab]["ci95"][0])
                v[f"sweep.{k}.diffhi"] = signed_pp(dif[lab]["ci95"][1])
                v[f"sweep.{k}.above"] = str(sum(1 for x in sw["per_history_diffs_vs_F16"][lab] if x > 0))
        vary = list(sw["episodes_whose_status_varies_across_histories"].values())
        v["sweep.vary.min"], v["sweep.vary.max"] = str(min(vary)), str(max(vary))
        sds = [hl[lab]["sd_between_histories"] for lab in FMT_KEY]
        if None not in sds:
            v["sweep.sdmin"], v["sweep.sdmax"] = pct(min(sds)), pct(max(sds))
        g = sw["glmm_b1_by_history"]
        v["sweep.or.hist"] = num(g["or_per_step_mean"])
        if g.get("t95"):
            v["sweep.or.histlo"], v["sweep.or.histhi"] = num(math.exp(g["t95"][0])), num(math.exp(g["t95"][1]))
        for step, d in sw.get("adjacent_steps", {}).items():
            key = step.replace("Q8_0", "Q8").replace("Q4_K_M", "Q4").replace("Q3_K_M", "Q3").replace("-", "minus")
            v[f"sweep.step.{key}"] = signed_pp(d["diff"])
            v[f"sweep.step.{key}.abs"] = pct(abs(d["diff"]))
            v[f"sweep.step.{key}.lo"], v[f"sweep.step.{key}.hi"] = (signed_pp(x) for x in d["ci95"])
            v[f"sweep.step.{key}.up"], v[f"sweep.step.{key}.down"] = str(d["histories_up"]), str(d["histories_down"])
        tw = sw.get("glmm_b1_two_way_bootstrap")
        if tw:
            v["sweep.or.cilo"], v["sweep.or.cihi"] = num(tw["or_ci95_two_way"][0]), num(tw["or_ci95_two_way"][1])
            v["sweep.or.reps"] = f"{tw['replicates']:,}"
        sl = sw["slope_per_step"]
        v["sweep.slope"] = pct(sl["points"])
        v["sweep.slopelo"], v["sweep.slopehi"] = (signed_pp(x) for x in sl["ci95_two_way_bootstrap"])
        if "glmm_canonical_history" in sw:
            gc = sw["glmm_canonical_history"]
            v["sweep.or.canon"] = num(gc["or_per_step"])
            v["sweep.or.canonlo"], v["sweep.or.canonhi"] = num(gc["or_per_step_ci95"][0]), num(gc["or_per_step_ci95"][1])
        pe = sw["exact_permutation_episode_counts"]
        v["sweep.perm.pone"], v["sweep.perm.ptwo"] = pval(pe["p_one_sided_increasing"]), pval(pe["p_two_sided"])

    mech = pb.get("mechanism_top5") or {}
    for lab, k in FMT_KEY.items():
        m = mech.get(lab)
        if not m:
            continue
        v[f"mech.{k}.positions"] = f"{m['positions']:,}"
        v[f"mech.{k}.flips"] = str(m["flips"])
        v[f"mech.{k}.diveps"] = str(m["divergent_episodes"])
        if m["margin_at_flips"]["median"] is not None:
            v[f"mech.{k}.flipmargin.median"] = sig(m["margin_at_flips"]["median"])
            v[f"mech.{k}.flipmargin.max"] = sig(m["margin_at_flips"]["max"])
        v[f"mech.{k}.located"] = str(m["divergent_episodes_with_located_flip"])
        q = m["abs_margin_perturbation_nonflip"]
        for name in ("median", "p25", "p75", "p10", "p90"):
            v[f"mech.{k}.pert.{name}"] = sig(q[name])
        r = m["recompute_margins_all_tokens"]
        v[f"mech.{k}.near"] = pct(r["share_below_0.1"])
        v[f"mech.{k}.tokens"] = f"{r['tokens']:,}"
        hz = m["flip_hazard_by_recompute_margin"]
        for i, b in enumerate(BIN_KEYS):
            if b in hz:
                v[f"mech.{k}.bin{i}.flips"] = str(hz[b]["flips"])
                v[f"mech.{k}.bin{i}.pos"] = f"{hz[b]['positions']:,}"
    for key, name in (("episode_median_abs_margin_perturbation", "pert"), ("episode_share_recompute_margin_below_0.1", "near")):
        e = mech.get(key)
        if not e:
            continue
        if "last_over_first" in e:
            v[f"mech.{name}.ratio"] = num(e["last_over_first"], 1)
            v[f"mech.{name}.ratiolo"], v[f"mech.{name}.ratiohi"] = (num(x, 1) for x in e["last_over_first_ci95"])
        if "page_p_increasing" in e:
            v[f"mech.{name}.pageinc"] = pval(e["page_p_increasing"])
            v[f"mech.{name}.pagedec"] = pval(e["page_p_decreasing"])

    for lab, k in FMT_KEY.items():
        s = (pb.get("sentinel_isolation") or {}).get(lab) or {}
        for o in (1, 2):
            if f"order{o}_vs_order0_diverged" in s:
                v[f"sent.{k}.o{o}"] = str(s[f"order{o}_vs_order0_diverged"])
                v[f"sent.{k}.n"] = str(s["n"])
        if "cross_arm_under_isolation" in s:
            c = s["cross_arm_under_isolation"]
            v[f"sent.{k}.cross"] = str(c["k"])
            v[f"sent.{k}.crosspct"] = pct(c["k"] / c["n"])
        r = (pb.get("reset_control") or {}).get(lab) or {}
        for sid, ss in (r.get("sessions") or {}).items():
            v[f"reset.{k}.s{sid}.rec"] = str(ss["recompute_reproducible"])
            v[f"reset.{k}.s{sid}.cached"] = str(ss["cached_reproducible"])
            v[f"reset.{k}.s{sid}.differ"] = str(ss["paths_differ"])
            v[f"reset.{k}.items"] = str(ss["items"])
        for ph, c in (r.get("cross_session_token_identity") or {}).items():
            if c["items"]:
                v[f"reset.{k}.{ph.split('_')[0]}.ident"] = f"{c['identical_across_sessions']}/{c['items']}"

    pr = pb.get("process_replicate_canonical") or {}
    if pr:
        v["bone.formats"] = str(len(pr))
        v["bone.tokens"] = f"{sum(x['tokens'] for x in pr.values()):,}"
        v["bone.diverged"] = str(sum(x["diverged"] for x in pr.values()))
        v["bone.allidentical"] = "yes" if all(x["identical_logprobs"] == x["tokens"] for x in pr.values()) else "NO"
    si = pb.get("sentinel_isolation") or {}
    ids = [x for f in si.values() for key, x in f.items() if key.endswith("_identity")]
    if ids:
        v["sent.tokens"] = f"{sum(x['tokens'] for x in ids):,}"
        v["sent.allidentical"] = "yes" if all(x["identical_logprobs"] == x["tokens"] for x in ids) else "NO"
        v["sent.diverged"] = str(sum(f.get(f"order{o}_vs_order0_diverged", 0) for f in si.values() for o in (1, 2)))
    tel = (pb.get("reset_control") or {}).get("telemetry")
    if tel:
        v["reset.ncold"], v["reset.nwarm"] = f"{tel['cold_requests']:,}", f"{tel['warm_requests']:,}"
        v["reset.coldmax"], v["reset.warmmin"] = str(tel["cold_max_cached_tokens"]), str(tel["warm_min_cached_tokens"])
    for mode, key in (("promptcache_default", "default"), ("promptcache_disabled", "disabled"), ("recompute", "recompute")):
        L = (pb.get("latency") or {}).get(mode)
        if not L:
            continue
        lat = L["latency_s_mean"]
        v[f"lat.{key}.mean"] = num(sum(lat) / len(lat), 3)
        v[f"lat.{key}.min"], v[f"lat.{key}.max"] = num(min(lat), 3), num(max(lat), 3)
        pf = [x for x in L["prefill_ms_mean"] if x is not None]
        if pf:
            v[f"lat.{key}.prefill"] = num(sum(pf) / len(pf), 1)
        su = [x for x in L["server_startup_s"] if x is not None]
        if su:
            v[f"lat.{key}.startup"] = num(sum(su) / len(su), 1)
            v[f"lat.{key}.startupmax"] = num(max(su), 1)
        v[f"lat.{key}.sessions"] = str(L["sessions"])

    b6 = (pb.get("production_default_randomized_histories") or {}).get("summary")
    if b6:
        v["bsix.n"] = str(b6["n_histories"])
        v["bsix.mean"], v["bsix.min"], v["bsix.max"] = pct(b6["mean"]), pct(b6["min"]), pct(b6["max"])
        if b6.get("sd") is not None:
            v["bsix.sd"] = pct(b6["sd"])
    L = pb.get("latency") or {}
    if all(m in L for m in ("promptcache_default", "promptcache_disabled", "recompute")):
        mean = lambda m: sum(L[m]["latency_s_mean"]) / len(L[m]["latency_s_mean"])
        d, c, r = mean("promptcache_default"), mean("promptcache_disabled"), mean("recompute")
        v["lat.overhead.pct"] = f"{100 * (c / d - 1):.0f}"
        v["lat.retained.pct"] = f"{100 * (r - c) / (r - d):.0f}"
        v["lat.speedup"] = f"{r / c:.1f}"
        req = [n for m in ("promptcache_default", "promptcache_disabled") for n in L[m].get("requests", [])]
        if req:
            v["lat.req.min"], v["lat.req.max"] = str(min(req)), str(max(req))
    b6all = pb.get("production_default_randomized_histories") or {}
    reps = b6all.get("fresh_server_replicates") or {}
    if reps:
        rows = [x for r in reps.values() for x in r.values()]
        v["bsix.rep.orders"] = str(len(reps))
        v["bsix.rep.tokens"] = f"{sum(x['tokens'] for x in rows):,}"
        v["bsix.rep.diverged"] = str(sum(x["diverged"] for x in rows))
        v["bsix.rep.allidentical"] = "yes" if all(x["identical_logprobs"] == x["tokens"] for x in rows) else "NO"
    hist = b6all.get("histories") or {}
    if hist:
        ovf = {k: h["context_overflow_episodes"] for k, h in hist.items() if h.get("context_overflow_episodes")}
        v["bsix.ovf.orders"] = str(len(ovf))
        v["bsix.ovf.episodes"] = str(sum(len(x) for x in ovf.values()))
        q = [h["quartiles_in_execution_order"] for h in hist.values()]
        v["bsix.firstq.top"] = str(sum(1 for x in q if x[0] == max(x)))
        v["bsix.trendp.max"] = pval(max(h["trend_p_two_sided"] for h in hist.values()))
        bl = [h["block_used"] for h in hist.values() if h.get("block_used")]
        if bl:
            v["bsix.block.min"], v["bsix.block.max"] = str(min(bl)), str(max(bl))
        for k, h in hist.items():
            v[f"bsix.h{k}.k"] = str(h["k"])
            v[f"bsix.h{k}.quartiles"] = ", ".join(str(x) for x in h["quartiles_in_execution_order"])
            v[f"bsix.h{k}.trendp"] = pval(h["trend_p_two_sided"])
            v[f"bsix.h{k}.rate"] = pct(h["rate"])
            v[f"bsix.h{k}.lo"], v[f"bsix.h{k}.hi"] = pct(h["reported95"][0]), pct(h["reported95"][1])
        # dependence-aware intervals of the lowest and highest order, printed with the range
        for end, pick in (("min", min), ("max", max)):
            r = pick(h["rate"] for h in hist.values())
            hs = [h for h in hist.values() if h["rate"] == r]
            if len({tuple(h["reported95"]) for h in hs}) != 1 or pct(r) != v.get(f"bsix.{end}"):
                raise SystemExit(f"bsix.{end}: tied orders with different intervals, or mismatch with the summary")
            v[f"bsix.{end}.lo"], v[f"bsix.{end}.hi"] = pct(hs[0]["reported95"][0]), pct(hs[0]["reported95"][1])
    b7 = pb.get("dequantized_weights_control") or {}
    for fk, val in (b7.get("mean_over_histories") or {}).items():
        if val is not None:
            v[f"bseven.{fk}"] = pct(val)
    if b7.get("perturbation_q4deq"):
        v["bseven.pert.median"] = sig(b7["perturbation_q4deq"]["median"])
    if b7.get("histories_paired"):
        v["bseven.k"] = str(len(b7["histories_paired"]))
    pdiff = b7.get("paired_diffs") or {}
    for key, name in (("q4deq_minus_f16", "deqminusf"), ("q4km_minus_q4deq", "qminusdeq")):
        if key in pdiff:
            v[f"bseven.{name}"] = signed_pp(pdiff[key]["diff"])
            v[f"bseven.{name}.lo"], v[f"bseven.{name}.hi"] = (signed_pp(x) for x in pdiff[key]["ci95"])
    if pdiff.get("share_of_f16_to_q4km_gap") is not None:
        v["bseven.share"] = f"{100 * pdiff['share_of_f16_to_q4km_gap']:.0f}"
    for lab, key in (("F16", "f"), ("Q4_K_M", "q")):
        rv = (b7.get("recompute_path_vs") or {}).get(lab)
        if rv:
            v[f"bseven.offvs{key}"] = str(rv["diverged"])
    mc = b7.get("mechanism_canonical")
    if mc:
        v["bseven.near"] = pct(mc["recompute_margins_all_tokens"]["share_below_0.1"])
        v["bseven.diveps"], v["bseven.located"] = str(mc["divergent_episodes"]), str(mc["divergent_episodes_with_located_flip"])
        if mc["margin_at_flips"]["median"] is not None:
            v["bseven.flipmargin.median"] = sig(mc["margin_at_flips"]["median"])
    return v


def collect_original(rf):
    """Values from the reanalysis of the original data (analysis/revision_findings.json)."""
    v = {}
    t = rf.get("trend_grid_production_default") or {}
    if t:
        g, c, pe = t["glmm"], t["clogit"], t["exact_permutation"]
        v["grid.or"], v["grid.orlo"], v["grid.orhi"] = num(g["or_per_step"]), num(g["or_per_step_ci95"][0]), num(g["or_per_step_ci95"][1])
        v["grid.sigma"] = num(g["sigma"])
        v["grid.clogit.or"] = num(c["or_per_step"])
        v["grid.clogit.lo"], v["grid.clogit.hi"] = num(math.exp(c["ci95"][0])), num(math.exp(c["ci95"][1]))
        v["grid.clogit.informative"] = str(c["n_informative_episodes"])
        v["grid.perm.pone"], v["grid.perm.ptwo"] = pval(pe["p_one_sided_increasing"]), pval(pe["p_two_sided"])
        be = t["bootstrap_episode"]
        v["grid.slope"] = pct(t["slope_per_step"])
        v["grid.slopelo"], v["grid.slopehi"] = (pct(x) for x in be["slope_per_step_ci95"])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phaseb", default="analysis/phaseb_findings.json")
    ap.add_argument("-o", "--out", default="paper/tables/phaseb_values.tex")
    ap.add_argument("--list", action="store_true", help="print every key and value")
    a = ap.parse_args()
    pb = json.load(open(a.phaseb))
    v = collect(pb)
    rfp = Path(a.phaseb).with_name("revision_findings.json")
    rf = json.load(open(rfp)) if rfp.exists() else {}
    v.update(collect_original(rf))
    # reproduction verdicts the text states in words (values are yes/NO, checked at the final gates)
    gr = pb.get("gate_records") or {}
    def all0(prefix, lp=False):
        xs = [r for k, r in gr.items() if k.startswith(prefix)]
        return "yes" if xs and all(r.get("n_diverged") == 0 and (not lp or r.get("shared_tokens_top1_logprob_differs") == 0) for r in xs) else "NO"
    v["repro.b0exact"] = all0("gates/G0a", lp=True) if all0("gates/G0b", lp=True) == "yes" else "NO"
    v["repro.b2tokens"] = all0("gates/B2_") 
    v["repro.ctrltokens"] = all0("checks/B1_") if any(k.startswith("checks/B1_") and "august_controlled" in k for k in gr) else "NO"
    ids = [x["order1_vs_order0_identity"] for x in (pb.get("off_path_checks") or {}).values() if "order1_vs_order0_identity" in x]
    if ids:
        v["btwo.tokens"] = f"{sum(x['tokens'] for x in ids):,}"
        v["btwo.allidentical"] = "yes" if all(x["identical_logprobs"] == x["tokens"] for x in ids) else "NO"
    tcb, tco = pb.get("token_cap"), rf.get("token_cap")
    if tcb and tco:
        for w in ("bfcl", "gsm8k"):
            v[f"cap.{w}.n"] = f"{tcb['finish_reason_length'][w] + tco['finish_reason_length'][w]:,}"
            v[f"cap.{w}.total"] = f"{tcb['requests'][w] + tco['requests'][w]:,}"
    lines = ["% generated by analysis/make_phaseb_values.py from " + a.phaseb + "; do not edit",
             "\\makeatletter",
             "\\newcommand{\\pbv}[1]{\\ifcsname pbv@#1\\endcsname\\csname pbv@#1\\endcsname"
             "\\else\\errmessage{missing Phase B value: #1}\\fi}"]
    for k in sorted(v):
        lines.append(f"\\expandafter\\def\\csname pbv@{k}\\endcsname{{{v[k]}}}")
    lines.append("\\makeatother")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    if a.list:
        for k in sorted(v):
            print(f"{k:32s} {v[k]}")
    print(f"wrote {a.out} ({len(v)} values)")


if __name__ == "__main__":
    main()
