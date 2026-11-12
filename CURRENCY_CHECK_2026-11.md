# Currency check, November 2026

Does the paper's effect still hold on current releases? This note reports one
rerun on 2026-11-11 and 11-12 (UTC), on one rented GPU. It is a check of the published result on
newer software, not a new study. The data behind every number is in
[`currency-2026-11/`](currency-2026-11/).

## Setup

- One RTX 4090, driver 595.91.07. llama.cpp was built with CUDA 12.6; vLLM ran
  on CUDA 13.2 and SGLang on CUDA 13.0 (their PyTorch builds).
- The paper's model files: Qwen2.5-7B-Instruct GGUF Q4_K_M and F16, checked by
  sha256 against the paper's files ([`SHA256SUMS`](currency-2026-11/SHA256SUMS)),
  and the BF16 safetensors at revision `a09a354` for vLLM.
- 80 BFCL multi-turn episodes in one fixed order (one history), temperature 0,
  one request at a time, a fresh server per pass. llama.cpp passes ran with
  `--cache-ram 0`.
- "Changed" means the episode's trajectory differs between the cache-hit pass
  and the recompute pass. On the current builds (llama.cpp v0.9.0 at both
  weights, vLLM 0.33.0) the cache-off pass was run twice, and the repeat must be
  identical before a difference is read as a cache effect. The paper's build was
  checked against the paper's logs instead. The batch-invariant pass was not
  repeated.
- The decision rules and wording limits were written down before the measured
  passes started.

## Results

| Engine and build | Weights | Cache hit vs recompute | Cache-off pass run twice |
|---|---|---|---|
| llama.cpp b10434 (`7e4c0a9`, the paper's build) | Q4_K_M | 65 of 80 changed | both passes match the paper's logs exactly (same tokens and log-probs) |
| llama.cpp v0.9.0 (`cd959ba`) | Q4_K_M | 58 of 80 changed | identical, 0 log-prob differences |
| llama.cpp v0.9.0 (`cd959ba`) | F16 | 36 of 80 changed | identical, 0 log-prob differences |
| vLLM 0.33.0 | BF16 | 48 of 80 changed | identical, 0 log-prob differences |
| vLLM 0.33.0, `VLLM_BATCH_INVARIANT=1` | BF16 | 0 of 80 changed (0 log-prob differences) | |

What this supports, for one history:

- The effect is still present on llama.cpp v0.9.0 (58 of 80 episodes). The
  paper's build reproduced the paper's logs exactly on this machine, so the
  change from 65 to 58 comes from the build, not the hardware. v0.9.0 includes
  llama.cpp PR #31247 (Q8_1 correction sum); b10434 does not. About 1,700
  commits separate the two builds, and this run does not isolate which change
  is responsible.
- Quantization amplification is still seen on llama.cpp v0.9.0: Q4_K_M changed
  22 more episodes than F16 (58 vs 36), and both cache-off repeats were
  identical.
- The effect is still present on vLLM 0.33.0 (48 of 80; the paper's vLLM 0.11.0
  run had 50 of 80). With vLLM's
  batch-invariant mode on, no cache-hit vs recompute differences were seen.
  That mode also changes outputs compared with default vLLM (52 of 80 episodes
  differ between the two cache-off passes), so that comparison is not a cache
  result.
- One history is not the paper's ten-history average, so the paper's
  percentages are not quoted for the new builds.

## llm-cachecheck on real servers

Qwen2.5-3B-Instruct, BF16 (Hugging Face revision `aa8e725`), the tool's 20
conversations, two servers on one GPU:

| Server | Turn-2 answer differs on the cache path | Both caches off |
|---|---|---|
| vLLM 0.33.0 | 14 of 20, in each of 5 runs | 0 of 20 |
| SGLang 0.5.23 (CUDA graphs disabled) | 12 of 20, in each of 2 runs | 0 of 20 |
| llama.cpp v0.9.0 (Qwen2.5-7B Q4_K_M, one server) | 11 of 20, in each of 2 runs | |

Every probe was valid. In the vLLM and SGLang reports the first turn and the
recompute path reported 0 cached tokens (`telemetry_check.txt`).

## Settings to know

- vLLM, 7B passes: temperature 0 with the model's default repetition penalty
  (1.05), the same flags as the paper's vLLM runs. The tool runs used
  `--generation-config vllm` (no model defaults).
- All vLLM servers ran with `VLLM_USE_FLASHINFER_SAMPLER=0`, because the
  FlashInfer sampler needed a kernel build the machine could not do. That
  sampler only serves non-greedy requests.
- SGLang ran with `--dtype bfloat16 --context-length 4096
  --max-running-requests 1 --mem-fraction-static 0.60 --max-total-tokens 65536
  --random-seed 42 --sampling-defaults openai --disable-cuda-graph
  --enable-cache-report`. CUDA graphs were off, so this is not SGLang's default
  configuration.
- Two first attempts were redone and their data kept aside. The first vLLM tool
  run used port 8001, where RunPod's standard images run an nginx proxy that
  forwards to port 8000, so both URLs reached the same server; the tool flagged every probe invalid. The
  first SGLang start failed because SGLang 0.5.23 builds a kernel at start-up
  and the machine lacked `ninja`.

## Files

- `cmp_*.json`: every comparison (episodes, differing episodes, shared tokens,
  log-prob differences).
- `C_llamacpp/`, `C2_vllm/`, `S_sglang/`: llm-cachecheck reports, a telemetry
  check, and the model revision used.
- `env_manifest.txt`, `env_manifest_end.txt`: GPU, driver, engine versions and
  commits, package versions, file hashes.
- `SUMMARY.txt`, `SUMMARY_S.txt`: the run's own summaries. `SUMMARY.txt` lists the
  first, invalid vLLM tool attempt as `C_vllm` (its "control" there compared one
  server with itself); the valid redo is `C2_vllm/`, which it does not list.
- The two manifests had entries for unrelated work on the same machine (another
  project's scripts and models); those lines were removed from these copies.

The full run archive (26 MB compressed: raw request logs, server logs and
scripts) is kept by the author and is available on request.
