"""Figures for the Hugging Face article. Data read from the public repo checkout."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.dirname(os.path.abspath(__file__))

SURF = "#ffffff"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8984"; GRID = "#e8e7e3"
BLUE = "#2a78d6"; ORANGE = "#eb6834"
SANS = "Helvetica Neue"; TITLE = "Avenir Next"; MONO = "DejaVu Sans Mono"
plt.rcParams.update({"font.family": SANS, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "savefig.facecolor": SURF, "figure.facecolor": SURF, "axes.facecolor": SURF})

def box(ax, x, y, w, h, fc, ec, lw=1.6, r=0.025):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%s" % r,
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes, zorder=2))

def arrow(ax, p, q, color):
    ax.add_patch(FancyArrowPatch(p, q, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=22,
                                 lw=3, color=color, connectionstyle="arc3,rad=0", zorder=1))

# ---------- cover, 1300x650 at 2x ----------
fig = plt.figure(figsize=(13, 6.5), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="#f7f6f3", zorder=0))
ax.text(0.055, 0.70, "Same request,\ndifferent answer", fontsize=44, fontweight="bold", family=TITLE,
        va="center", ha="left", linespacing=1.05)
ax.text(0.057, 0.47, "Greedy decoding, a prompt cache,\nand two answers to one question.",
        fontsize=18, color=INK2, va="center", ha="left", linespacing=1.35)
ax.text(0.057, 0.215, "pip install llm-cachecheck", fontsize=16, family=MONO, color=INK, ha="left", va="center")
ax.text(0.057, 0.158, "Check your own llama.cpp, vLLM or SGLang server", fontsize=13, color=MUTED, ha="left", va="center")
L, R, GAP = 0.505, 0.965, 0.02
W = (R - L - GAP) / 2
box(ax, L, 0.715, R - L, 0.20, "#ffffff", "#cfcdc7")
ax.text(L + 0.018, 0.868, "SAME PROMPT, TEMPERATURE 0", fontsize=10.5, color=MUTED, fontweight="bold", va="center")
ax.text(L + 0.018, 0.778, "If the train had left 25 minutes later,\nwhen would it arrive?", fontsize=14.5, va="center", linespacing=1.3)
for x, col, lab, lines in [(L, BLUE, "cache hit", ["10:05 + 3 h = 1:05 PM", "1:05 PM + 35 min", "      = 1:40 PM", "arrives 13:40 (2:40 PM)"]),
                           (L + W + GAP, ORANGE, "recompute", ["10:05 + 3 h = 13:05", "13:05 + 35 min", "      = 13:40", "arrives 13:40"])]:
    cx = x + W / 2
    arrow(ax, (cx, 0.705), (cx, 0.56), col)
    ax.text(cx + 0.012, 0.632, lab, fontsize=14, color=col, fontweight="bold", ha="left", va="center")
    box(ax, x, 0.13, W, 0.40, "#ffffff", col, lw=2.2)
    hl = set(lines)
    tint = "#e3eefb" if col == BLUE else "#fde6dc"
    for i, t in enumerate(lines):
        ax.text(x + 0.016, 0.465 - i * 0.082, t, fontsize=12.5, family=MONO, va="center",
                fontweight="bold" if t.startswith("arrives") else "normal",
                bbox=dict(fc=tint, ec="none", pad=2.5) if t in hl else None)
    if col == BLUE:
        ax.text(x + W / 2, 0.095, "13:40 is 1:40 PM, not 2:40 PM", fontsize=11.5, color=INK2, ha="center", va="center", style="italic")
fig.savefig(os.path.join(OUT, "cover.png")); plt.close(fig)

# ---------- quantization ----------
s = json.load(open(os.path.join(REPO, "analysis/phaseb_findings.json")))["sweep"]
fmts = ["F16", "Q8_0", "Q4_K_M", "Q3_K_M"]
hist = [str(h) for h in s["complete_random_histories"]]
fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200)
fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.15)
means = []
for i, f in enumerate(fmts):
    vals = [100 * s["rate_by_history"][f][h] for h in hist]
    m = 100 * s["history_level"][f]["mean"]; means.append(m)
    xs = [i + (k - 4.5) * 0.028 for k in range(len(vals))]
    ax.scatter(xs, vals, s=34, color=BLUE, alpha=0.35, lw=0, zorder=2)
    ax.scatter([i], [m], s=150, color=BLUE, ec=SURF, lw=2, zorder=3)
    ax.text(i, m + 7.5, "%.1f%%" % m, fontsize=15, fontweight="bold", va="center", ha="center", color=INK)
ax.plot(range(4), means, color=BLUE, lw=2, zorder=1)
ax.set_xticks(range(4)); ax.set_xticklabels(["F16\n16-bit", "Q8_0\n8-bit", "Q4_K_M\n4-bit", "Q3_K_M\n3-bit"], fontsize=13.5, color=INK, linespacing=1.5)
dq = 100 * json.load(open(os.path.join(REPO, "analysis/phaseb_findings.json")))["dequantized_weights_control"]["mean_over_histories"]["q4deq"]
ax.scatter([2], [dq], s=150, fc=SURF, ec=INK2, lw=2.2, zorder=3)
ax.text(2.13, dq, "%.1f%%: the same 4-bit weights\nrun through the 16-bit kernels" % dq, fontsize=11.5, color=INK2, va="center", ha="left", linespacing=1.35)
ax.set_xlim(-0.45, 3.65); ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=12)
ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#cfcdc7"); ax.tick_params(length=0)
ax.set_ylabel("episodes whose trajectory changed", fontsize=12.5)
fig.text(0.09, 0.93, "Quantized weights made the cache-hit vs recompute split more frequent", fontsize=17,
         fontweight="bold", family=TITLE, ha="left")
fig.text(0.09, 0.875, "Qwen2.5-7B-Instruct on llama.cpp b10434, 80 multi-turn tool-use episodes, greedy decoding.\n"
         "Large dot: mean of 10 shuffled episode orders, each on a fresh server. Small dots: each order.", fontsize=11.5, color=INK2, ha="left", va="center", linespacing=1.4)
fig.savefig(os.path.join(OUT, "fig_quantization.png")); plt.close(fig)

# ---------- batch-invariant: result and cost ----------
cmp = lambda p: json.load(open(os.path.join(REPO, p)))["n_diverged"]
d1, b1 = cmp("currency-2026-11/cmp_B_vllm_path.json"), cmp("currency-2026-11/cmp_D_vllm_bi_path.json")
d2, b2 = cmp("cachecheck-0.3-validation/cmp_B2_vllm_path.json"), cmp("cachecheck-0.3-validation/cmp_D2_vllm_bi_path.json")
assert (d1, d2, b1, b2) == (48, 48, 0, 0)
# Time ratios on matched requests, from ../../batch-invariant-cost-2026-11/RESULT_2026-11-13.txt:
# default cache on -> default cache off 2.62 (596 requests); -> batch-invariant cache on 2.30 (543 requests)
TEAL = "#1baf7a"; GRAYBAR = "#a9a8a2"
rows = ["default vLLM,\ncache on", "default vLLM,\ncache off", "batch-invariant,\ncache on"]
cols = [GRAYBAR, GRAYBAR, TEAL]
fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4.9), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(left=0.17, right=0.97, top=0.74, bottom=0.17, wspace=0.12)
ys = [2, 1, 0]
for y, v, c in [(2, 48, GRAYBAR), (0, 0.6, TEAL)]:
    a1.barh(y, v, height=0.56, color=c, zorder=2)
a1.text(49.5, 2, "48 of 80", va="center", fontsize=13, fontweight="bold")
a1.text(2, 1, "no cache hits, nothing to compare", va="center", fontsize=11.5, color=MUTED, style="italic")
a1.text(2.2, 0, "0 of 80", va="center", fontsize=13, fontweight="bold")
a1.set_xlim(0, 80); a1.set_xticks([0, 20, 40, 60, 80])
a1.set_title("episodes that split\n(same result in both episode orders)", fontsize=12, color=INK2, loc="left", pad=10)
for y, v, c in zip(ys, [1.0, 2.62, 2.30], cols):
    a2.barh(y, v, height=0.56, color=c, zorder=2)
    a2.text(v + 0.06, y, "%.1fx" % v, va="center", fontsize=13, fontweight="bold")
a2.set_xlim(0, 3.2); a2.set_xticks([0, 1, 2, 3]); a2.set_xticklabels(["0", "1x", "2x", "3x"])
a2.set_title("time per request, against\ndefault vLLM with cache on", fontsize=12, color=INK2, loc="left", pad=10)
a1.set_yticks(ys); a1.set_yticklabels(rows, fontsize=12.5, color=INK, linespacing=1.3)
for a in (a1, a2):
    a.grid(axis="x", color=GRID, lw=1); a.set_axisbelow(True); a.tick_params(length=0, labelsize=11.5)
    for sp in ("top", "right", "bottom"): a.spines[sp].set_visible(False)
    a.spines["left"].set_color("#cfcdc7")
fig.text(0.03, 0.935, "Batch-invariant mode: no split, cache still on, no slower than cache off", fontsize=16.5,
         fontweight="bold", family=TITLE, ha="left")
fig.text(0.03, 0.875, "vLLM 0.33.0, Qwen2.5-7B-Instruct BF16, one RTX 4090, one request at a time, 80 BFCL multi-turn episodes.",
         fontsize=11, color=INK2, ha="left", va="center")
fig.text(0.03, 0.04, "Times compare the same requests in two passes (543 and 596 requests), from the November passes.",
         fontsize=10, color=MUTED, ha="left")
fig.savefig(os.path.join(OUT, "fig_batch_invariant.png")); plt.close(fig)

# ---------- mechanism ----------
fig = plt.figure(figsize=(10, 4.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
fig.text(0.04, 0.92, "Same tokens, two computations", fontsize=17, fontweight="bold", family=TITLE, ha="left")
fig.text(0.04, 0.855, "How turn 2 reaches the model on each path", fontsize=11.5, color=INK2, ha="left", va="center")
X0, segs = 0.20, [("turn 1 question", 0.19), ("turn 1 answer", 0.24), ("turn 2 question", 0.15)]
xs = [X0]
for _, w in segs: xs.append(xs[-1] + w)
for (lab, w), x in zip(segs, xs):
    ax.text(x + w / 2, 0.755, lab, ha="center", va="center", fontsize=11, color=INK2)
def seg(x, w, y, fc, ec, text, tc=INK):
    box(ax, x + 0.003, y, w - 0.006, 0.15, fc, ec, lw=1.6, r=0.012)
    ax.text(x + w / 2, y + 0.075, text, ha="center", va="center", fontsize=10.5, color=tc, linespacing=1.3)
LB = "#e3eefb"; LO = "#fde6dc"
ax.text(0.185, 0.555, "cache hit", ha="right", va="center", fontsize=13, fontweight="bold", color=BLUE)
seg(xs[0], segs[0][1], 0.48, LB, BLUE, "reused: prefilled\nduring turn 1")
seg(xs[1], segs[1][1], 0.48, LB, BLUE, "reused: written one token\nat a time while generating")
seg(xs[2], segs[2][1], 0.48, "#ffffff", BLUE, "new prefill")
ax.text(0.185, 0.275, "recompute", ha="right", va="center", fontsize=13, fontweight="bold", color=ORANGE)
seg(xs[0], xs[3] - xs[0], 0.20, LO, ORANGE, "one prefill pass over the whole conversation")
for y, c in [(0.555, BLUE), (0.275, ORANGE)]:
    arrow(ax, (xs[3] + 0.01, y), (0.83, 0.415 + (0.03 if c == BLUE else -0.03)), c)
ax.text(0.845, 0.415, "next-token\nlogits", ha="left", va="center", fontsize=12.5, fontweight="bold", linespacing=1.3)
ax.text(0.20, 0.09, "The two paths run different kernel shapes and summation orders, so the logits match only up to rounding.\n"
        "When the top two tokens are nearly tied, that rounding can pick a different token.", fontsize=10.8, color=INK2,
        ha="left", va="center", linespacing=1.45)
fig.savefig(os.path.join(OUT, "fig_mechanism.png")); plt.close(fig)
print("ok")
