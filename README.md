# Cache divergence study: harness, data, and analysis

Artifact for the paper *Same Request, Different Answer: Quantization Amplifies
Cache-Induced Divergence in LLM Serving*
([arXiv:2609.04748](https://arxiv.org/abs/2609.04748), under review at IEEE Access).

Author: Aditi Patodiya. Everything here is released so that each number in
the paper can be recomputed from the raw logs, and so that the measurement
can be repeated on other stacks.

## Cite this work

If you use this harness or the divergence dataset, please cite the paper:

```bibtex
@article{patodiya2026samerequest,
  title   = {Same Request, Different Answer: Quantization Amplifies
             Cache-Induced Divergence in LLM Serving},
  author  = {Patodiya, Aditi},
  journal = {arXiv preprint arXiv:2609.04748},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.04748},
  url     = {https://arxiv.org/abs/2609.04748}
}
```

GitHub's "Cite this repository" button (from `CITATION.cff`) emits the same
citation in BibTeX and APA.

## What the study measures

Serving stacks reuse the key-value tensors of a shared prompt prefix across
requests. That reuse changes floating point accumulation order, which can
change a sampled token. We measure how often this changes an agent's
trajectory and its task outcome, holding everything except the cache
setting fixed.

The design is paired and each configuration runs twice, so every arm can be
compared against itself. The cache-disabled self-comparison is the internal
validity check: in every llama.cpp configuration the repeated cache-disabled
run reproduces every emitted token id and its log-probability (for vLLM,
every token string and log-probability).

## Layout

```
harness/
  handler.py          BFCL handler subclass; the only override is the query
                      hook, so orchestration and scoring stay upstream code
  run_episodes.py     runs N agent episodes through one (config, arm) cell
  run_gsm8k.py        single-turn bridge, 4-pass cold/cold/warm/warm protocol
  compare_arms.py     token-level divergence between two runs
  score_results.py    scores a cell with the benchmark's own checker
  server_*.sh         pinned launch scripts for llama.cpp, vLLM, SGLang
  pod_*.sh            orchestration used for the reported runs
analysis/
  analyze.py          raw logs -> findings.json (every reported number)
  make_figures.py     findings.json -> paper figures
results/
  <config>/<pass>/arm_<on|off>/
      raw_requests.jsonl.gz   one record per request: prompt hash, tokens with
                           logprobs (token ids on llama.cpp, strings on vLLM), latency,
                           server-reported cached_tokens (llama.cpp; null on vLLM 0.11.0)
      model_results.json   benchmark-format responses
      scores.json          checker verdicts
      episode_meta.json    per-episode wall time, request count, errors
  bridge-<config>/cache_on/summary.json   per-item 4-pass bridge results
  MODEL_SHA256SUMS     SHA-256 of every model file used, GGUF and vLLM shards (provenance in VERSIONS.md)
VERSIONS.md           pinned versions, plan addenda, incident log
```

## Reproducing the numbers

```
uv sync        # installs the pinned environment from uv.lock (src/ holds only the stub that makes the project installable)
uv run python analysis/analyze.py results -o findings.json
uv run python analysis/analyze_repair.py --root results-repair -o repair_findings.json
uv run python analysis/make_figures.py --findings findings.json --repair repair_findings.json -o figures
uv run python analysis/reextract_bridge.py results   # re-derives the bridge answers; with -o DIR it writes summaries identical to results/bridge-*/cache_on/summary.json
```

`findings.json` and `repair_findings.json` at the repository root are the files the manuscript's
original-grid numbers come from (the revision adds `analysis/revision_findings.json` and
`analysis/phaseb_findings.json`). No number
in the paper is computed anywhere else.

## Reproducing the data collection

Hardware used: one NVIDIA RTX 4090 (24 GB), driver 570.211.01.

```
bash harness/pod_setup.sh          # CUDA toolkit, llama.cpp build, venvs
# pod_setup.sh installs vLLM unpinned; the measured runs used 0.11.0, pinned as in
# harness/pod_setup_repair.sh:
#   uv pip install "vllm==0.11.0" --torch-backend=cu128 && uv pip install "transformers<5"
```

The original grid was collected by resumable scripts that skip cells already
marked done: `harness/pod_run_grid.sh` and `pod_run_grid_v2.sh` (llama.cpp
lane, the first vLLM attempts and the Q4_K_M bridge), then `pod_vllm_lane.sh`, `pod_final_lane.sh`, `pod_phase2.sh` and
`pod_phase3.sh` (vLLM, bridges, Qwen2.5-14B). The ordering experiments and
controlled re-measurements (`results-repair/`) came from `pod_setup_repair.sh`,
`pod_repair.sh`, `pod_cacheram_run.sh` and `pod_quantgrad.sh`, and the revision
runs (`results-revision/`) from `revision/pod_setup_b.sh` and
`revision/pod_phaseB.sh`. The scripts also contain steps for SGLang and for
quantized vLLM configurations that were not run; no such results exist.

Pinned versions are in `VERSIONS.md` and are required to reproduce the
results, because the measured effects depend on the engine version and the
cache configuration. In particular llama.cpp is built from source at a fixed
commit, and the vLLM
environment pins `vllm==0.11.0` with `torch 2.8.0+cu128` and
`transformers<5`, because newer vLLM wheels require a CUDA 13 runtime that
the driver on our hardware does not provide.

## Determinism-relevant settings (held constant in every run)

- greedy decoding, temperature 0, seed 42 on every request
- batch size 1, serial requests, so batch composition cannot vary
- key-value cache in 16-bit floating point in all arms, never quantized
- context length 16384 in all engines
- cache arm set per request in llama.cpp, per server process in vLLM and
  SGLang, following each engine's own mechanism
- `cached_tokens` recorded per request, so cache exposure is measured
  rather than assumed

## Honest scope

- Models are 7B to 14B open-weight. No frontier or hosted models were
  tested.
- The agentic benchmark is hard for models this size and their success
  rates sit near the floor, so this artifact does not support conclusions
  about cache effects on agent task success. Trajectory-level measurements
  from that workload are unaffected. Outcome conclusions come from the
  mathematics bridge.
- One GPU model (RTX 4090) and single-tenant serving; three machines of the
  same type across the original grid (driver 570.211.01), the ordering
  experiments (a 580-series driver, see VERSIONS.md) and the revision runs
  (driver 595.91.07), with the revision machine checked against the earlier
  passes token for token before use.

## License

Code is released under the MIT License (see `LICENSE`). The measurement
dataset (`results/`, `results-repair/`, `results-revision/`, `findings.json`,
`repair_findings.json` and the revision findings files) is released under
CC BY 4.0. Benchmark data remains under its original license.

## Corrections applied after internal review (2026-08-16)

Two measurement defects were found by an internal review round before
submission. Both are disclosed here and in the paper, and both are auditable
from the released files.

**1. Answer extraction measured formatting, not correctness.** The
collection-time scorer in `harness/run_gsm8k.py` accepted only the
`#### <number>` form requested by the prompt. Models frequently answered
correctly in another unambiguous form, most often `\boxed{...}`, and those
responses were scored wrong. Between 11 and 38 percent of responses could
not be parsed by the original extractor, and after both corrections below,
between 13 and 37 percent of recompute-path responses changed from wrong to
correct (the figure the paper reports). The defect inverted the apparent
accuracy ordering across quantization levels. A second pass found that answers were also compared as strings, so
`57.00` did not match a gold of `57`.

Answers are re-derived offline from the stored response text by
`analysis/reextract_bridge.py`, with a wider pattern set and numeric
comparison. No new inference was run. Both versions ship: the corrected
scores are in each bridge's `cache_on/summary.json`, and the original
collection-time scores are preserved alongside as
`cache_on/summary_collection_time.json`.

Effect of the correction: unparseable answers per bridge fell from as many
as 158 to 0 or 1, accuracy moved from an implausible 53-76 percent to 89-93
percent, and cache-attributable correctness flips fell from 99 to 24 summed
over the six bridge runs (1,500 item-runs; the 200-item Qwen2.5-7B Q4_K_M run
repeats the first 200 problems of the 500-item run), and from 88 to 20 over the five
distinct configurations the paper pools (1,300 observations, 500 unique
problems). An apparent directional effect favouring the cached path (exact
binomial p = 0.009 over the six runs, 0.025 over the five) disappeared once
scoring was correct (p = 0.31 and p = 0.26); it had been an
artifact of the two paths differing in how often they emitted a parseable
format.

**2. Execution ordering was not uniform across configurations.** The
original grid ran the llama.cpp configurations as
`on/main, off/main, on/repeat, off/repeat`, placing a full cache-off pass
between the two cache-on passes, while the Qwen-14B and vLLM configurations
ran the cache-on passes adjacently. Ordering predicts the primary outcome
(Mann-Whitney p = 0.033), so the engine comparison in the original grid is
confounded with it. A repair run measures one llama.cpp and one vLLM
configuration under both orderings; see `results-repair/repair/` and the
paper's methods section.

## Verification notes

- The llama.cpp slot-erase endpoint (`POST /slots/0?action=erase`) returns
  HTTP 200 without clearing the live prompt cache when called between
  requests: a request issued immediately after an "erase" still reported 109
  cached prompt tokens. `harness/reset_control.py` therefore establishes the
  cold state with the per-request `cache_prompt=false` flag, which telemetry
  confirms yields 0 cached tokens, and aborts the run if more than 10
  percent of requests fail that check.
- vLLM 0.11.0 does not populate `prompt_tokens_details.cached_tokens` on the
  completions endpoint. `analysis/analyze.py` reports this as `n_absent`
  rather than as zero, and the vLLM manipulation check comes from the engine
  log via `harness/extract_cache_telemetry.py`.
- Divergence is compared on token ids for llama.cpp and on token strings for
  vLLM, because the two engines return different logprob shapes. Each
  configuration records which level was used in `findings.json` under
  `comparison_level`.


## Statistical checks in the first submission (superseded)

These scripts produced the statistics of the first submitted version. They are
kept so that the revision's changes can be audited.

- `analysis/robustness.py`: moving-block bootstrap, position-dependence
  permutation test, and a power simulation for the single-turn bridge.
  Output: `analysis/robustness.json`. The position-dependence test (quartile
  counts and permutation p-values in Section IV-C) is still reported. The
  moving-block bootstrap is replaced by the circular block bootstrap and the
  problem-level bridge analysis (`analysis/revision_stats.py`); the power
  simulation is no longer reported.
- `analysis/trend_test.py`: Cochran-Armitage trend test across weight formats.
  Output: `analysis/trend_test.json`. Replaced by paired trend models that
  keep episode identity (`analysis/revision_stats.py`,
  `analysis/phaseb_stats.py`).

## Revision (2026-10): new runs and analyses

The revised manuscript makes a replicated controlled sweep its primary
evidence for the quantization result and replaces the interval and trend
statistics. Everything in this section is new in the revision.

### Reanalysis of the original data (no new inference)

- `analysis/revision_stats.py` writes `analysis/revision_findings.json`.
  Intervals for agentic proportions use a circular block bootstrap
  (Politis and Romano), with the block length set per series by the
  Politis-White rule with the Patton-Politis-White correction, floored at
  five episodes; 10,000 replicates, seed 20261007. The reported interval is
  the envelope of that interval and the Wilson interval. Trend across weight
  formats uses models that keep episode identity: a random-intercept
  logistic model fitted by adaptive Gauss-Hermite quadrature (100 nodes), a
  conditional logistic regression fitted by conditional maximum likelihood
  and an exact within-episode permutation test. In the replicated sweep the permutation test acts on each
  episode's counts summed over histories (one format permutation per episode),
  and the mixed model is fitted per history. The single-turn bridge is analysed at the problem level
  (500 unique problems under 1,300 observations).
- `analysis/test_revision_stats.py`: 18 tests, including block lengths
  checked against the `arch` package (8.0.0).
- `analysis/make_tables.py` and `analysis/make_figures.py` read their
  intervals from `revision_findings.json`.

### New runs

Machine: one NVIDIA RTX 4090 (24 GB), driver 595.91.07, llama.cpp b10434
(commit 7e4c0a9), the same model files as the original grid (SHA-256 in
`SHA256SUMS`). Every pass runs on a freshly started server process with the
server-level prompt cache disabled from launch (`--cache-ram 0`), except the
production-default runs (B6, and one of the three latency modes in B5),
which keep the default.

| Block | What it measures |
|---|---|
| B0 | Gate: this machine reproduces the original Q4_K_M cache-off and cache-on passes in every token and log-probability, and requesting five log-probabilities instead of one leaves every token unchanged |
| B1 | Controlled sweep: F16, Q8_0, Q4_K_M, Q3_K_M under the canonical episode order and ten randomized orders, formats in a Williams order within each history |
| B1R | Fresh-process replicate of the canonical history |
| B2 | Cache-off references for each format under two orders (history independence of the recompute path) |
| B3 | Episode isolation: a sentinel prompt overwrites the cache before each episode, three orders per format |
| B4 | Restored-state control: four formats, three sessions each, 100 items per session |
| B5 | Single-stream latency and server restart time |
| B6 | Production-default prompt cache under five randomized orders |
| B7 | Q4_K_M weights dequantized to a 16-bit file and served on the F16 kernels |

Randomized orders: `revision/orders/order_k.json`, generated by
`numpy.random.default_rng([20261007, k]).permutation(80)`.

Harness changes are opt-in, and the defaults reproduce the original runs
byte for byte: `handler.py` reads the number of returned log-probabilities
from `CDS_LOGPROBS` (default 1); `run_episodes.py` adds `--order-file` and
`--sentinel-reset` and writes `run_meta.json`; `reset_control.py` adds
`--logprobs`, `--start` and `--item-seed`.

With five log-probabilities requested, llama.cpp computes the reported
log-probabilities over the requested top entries in 32-bit floating point,
so their last bits can differ from a one-log-probability run (largest
difference observed 2.3e-4). Token ids and the top-1/top-2 logit margin are
unaffected. Comparisons between runs with different log-probability depths
therefore match tokens exactly and log-probabilities within 1e-3; runs at
the same depth match exactly.

Orchestration and validation: `revision/pod_phaseB.sh` (each pass is staged,
validated and only then promoted), `revision/phaseb_tools.py` (record
validation and run comparison; tests in `revision/test_phaseb_tools.py`).

One validation rule changed during the run. Until 2026-10-08 04:15 UTC the
validator rejected any pass in which an episode ended with an error. In the
first random order of B6 (production-default prompt cache), episode 34 grew past
the 16,384-token context and the server rejected its next request, identically
in both passes. That is an outcome of the trajectory, not a fault, so the
validator now lists such episodes (`context_overflow_episodes`) and still
rejects every other error. No pass validated before the change contained such
an episode, so no earlier verdict changes. The rejected first attempt is kept in
`results-revision/_failed/`; it is identical to the accepted second attempt in
every token and log-probability, and the analysis reports that comparison as a
fresh-server replicate. The validator before the change is kept as
`results-revision/diag/phaseb_tools_before_20261008T0415Z.py.txt`.
Logs: `results-revision/<block>/<cell>/arm_<on|off>/raw_requests.jsonl.gz`,
with gate results in `results-revision/gates/`.

### Fixes to the released scripts (9 October 2026)

An independent check of the public repository found three defects in the August release, all fixed
in this version: `analyze_repair.py` opened only uncompressed logs, so on the released (gzipped)
data it exited successfully with empty comparison groups; it now reads either form, takes
`--root` and `-o`, and writes `repair_findings.json` at the repository root, including the
cache-off rerun entry (0 of 80 episodes) that the August export omitted. The README's figure
command used a positional argument `make_figures.py` does not accept, and its table command read a
stale copy at `analysis/findings.json` (removed); both now point at the root `findings.json` and
`repair_findings.json`. Every command in this README was rerun from a fresh clone after the fixes.

The revised paper labels the two vLLM configurations BF16 rather than FP16: no `vllm serve` command in
`harness/` (`pod_run_grid.sh`, `pod_run_grid_v2.sh`, `pod_vllm_lane.sh`, `pod_final_lane.sh`, `pod_phase2.sh`,
`pod_phase3.sh`, `server_vllm.sh`) passes a dtype override, so the engine served the checkpoints in their native
bfloat16.
The result directories keep their original `vllm-*-fp16` names. The script's comment "KV cache dtype:
default fp16" is kept as run; with no override the KV cache took the model dtype, bfloat16.

### Reproducing the revision numbers

```
uv run python analysis/revision_stats.py --grid results --repair results-repair
uv run python analysis/phaseb_stats.py --root results-revision --refs results-revision/references
uv run python analysis/or_bootstrap.py --reps 2000    # optional, slow; see below
uv run python analysis/make_phaseb_values.py
uv run python analysis/make_tables.py findings.json -o tables
uv run python analysis/make_phaseb_tables.py
uv run python analysis/make_phaseb_figures.py --findings findings.json
uv run python analysis/test_revision_stats.py
uv run python analysis/test_phaseb_stats.py
uv run python revision/test_phaseb_tools.py
```

The `make_*` scripts write the paper's tables and figures to `paper/tables/`,
`tables/` and `figures/` (created if missing). Rerunning the two `*_stats.py`
scripts from a fresh clone reproduces the committed `analysis/*_findings.json`
byte for byte (verified 2026-10-09 with the locked environment).

`analysis/or_bootstrap.py` computes the interval for the mixed-model odds
ratio per format step by resampling episodes and histories together and
refitting the model in every resampled history (2,000 replicates, seed
20261007; about 0.3 s per fit, so it uses all cores and takes a while). Its
result is committed as `analysis/phaseb_or_bootstrap.json` together with a
fingerprint of the divergence tensor it was computed from. `phaseb_stats.py`
includes that interval (`sweep.glmm_b1_two_way_bootstrap`) only when the
fingerprint matches the tensor it has just built from the logs, so the
committed file reproduces `phaseb_findings.json` exactly without rerunning
the bootstrap; rerunning it regenerates the same numbers from the same seed.

The first two commands rewrite `analysis/revision_findings.json` and
`analysis/phaseb_findings.json`; both reproduce the shipped files exactly
(apart from the input paths recorded in them). `make_phaseb_values.py` writes
every number the manuscript quotes from the new runs as LaTeX macros, so no
value in the text is typed by hand.
