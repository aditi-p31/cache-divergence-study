# Cache divergence study: harness, data, and analysis

Artifact for the paper *Same Request, Different Answer: Prompt Caching Makes
LLM Serving History-Dependent* (IEEE Access submission, 2026).

Author: Aditi Patodiya. Everything here is released so that each number in
the paper can be recomputed from the raw logs, and so that the measurement
can be repeated on other stacks.

## What the study measures

Serving stacks reuse the key-value tensors of a shared prompt prefix across
requests. That reuse changes floating point accumulation order, which can
change a sampled token. We measure how often this changes an agent's
trajectory and its task outcome, holding everything except the cache
setting fixed.

The design is paired and each configuration runs twice, so every arm can be
compared against itself. The cache-disabled self-comparison is the internal
validity check: it should be, and is, bit identical.

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
      raw_requests.jsonl   one record per request: prompt hash, token ids,
                           logprobs, latency, server-reported cached_tokens
      model_results.json   benchmark-format responses
      scores.json          checker verdicts
      episode_meta.json    per-episode wall time, request count, errors
  bridge-<config>/cache_on/summary.json   per-item 4-pass bridge results
VERSIONS.md           pinned versions, plan addenda, incident log
```

## Reproducing the numbers

```
uv sync
uv run python analysis/analyze.py results -o findings.json
uv run python analysis/make_figures.py findings.json -o figures
```

`findings.json` is the single source of truth for the manuscript. No number
in the paper is computed anywhere else.

## Reproducing the data collection

Hardware used: one NVIDIA RTX 4090 (24 GB), driver 570.211.01.

```
bash harness/pod_setup.sh          # CUDA toolkit, llama.cpp build, venvs
bash harness/pod_phase2.sh         # full grid, resumable, budget-capped
```

Pinned versions are in `VERSIONS.md` and are load-bearing for this study.
In particular llama.cpp is built from source at a fixed commit, and the vLLM
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
- One GPU model, one driver version, single-tenant serving.

## License

Code released under MIT. Raw logs released under CC BY 4.0. Benchmark data
remains under its original license.

## Corrections applied after internal review (2026-08-16)

Two measurement defects were found by an internal review round before
submission. Both are disclosed here and in the paper, and both are auditable
from the released files.

**1. Answer extraction measured formatting, not correctness.** The
collection-time scorer in `harness/run_gsm8k.py` accepted only the
`#### <number>` form requested by the prompt. Models frequently answered
correctly in another unambiguous form, most often `\boxed{...}`, and those
responses were scored wrong. Between 11 and 38 percent of responses were
affected, which inverted the apparent accuracy ordering across quantization
levels. A second pass found that answers were also compared as strings, so
`57.00` did not match a gold of `57`.

Answers are re-derived offline from the stored response text by
`analysis/reextract_bridge.py`, with a wider pattern set and numeric
comparison. No new inference was run. Both versions ship: the corrected
scores are in each bridge's `cache_on/summary.json`, and the original
collection-time scores are preserved alongside as
`cache_on/summary_collection_time.json`.

Effect of the correction: unparseable answers per bridge fell from as many
as 158 to 0 or 1, accuracy moved from an implausible 53-76 percent to 89-93
percent, and cache-attributable correctness flips fell from 99 to 24 across
1500 items. An apparent directional effect favouring the cached path
(p = 0.009) disappeared once scoring was correct (p = 0.31); it had been an
artifact of the two paths differing in how often they emitted a parseable
format.

**2. Execution ordering was not uniform across configurations.** The
original grid ran the llama.cpp configurations as
`on/main, off/main, on/repeat, off/repeat`, placing a full cache-off pass
between the two cache-on passes, while the Qwen-14B and vLLM configurations
ran the cache-on passes adjacently. Ordering predicts the primary outcome
(Mann-Whitney p = 0.033), so the engine comparison in the original grid is
confounded with it. A repair run measures one llama.cpp and one vLLM
configuration under both orderings; see `results/repair/` and the paper's
methods section.

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


## Statistical checks added at submission

- `analysis/robustness.py` computes the moving-block bootstrap and the
  position-dependence permutation test reported in the paper, and simulates
  the power of the single-turn bridge. Output: `analysis/robustness.json`.
- `analysis/trend_test.py` computes the Cochran-Armitage trend test across
  weight formats from `findings.json`. Output: `analysis/trend_test.json`
  (z = 5.68, p = 1.3e-8).
- `paper_preprint.pdf` is the submitted manuscript.

## License

Code is released under the MIT License (see `LICENSE`). The measurement
dataset (`results/`, `results-repair/`, `findings.json`,
`repair_findings.json`) is released under CC BY 4.0. `paper_preprint.pdf`
is covered by neither; it is the manuscript submitted to the IEEE, included
under the IEEE preprint policy.
