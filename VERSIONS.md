# Pinned versions (determinism-critical, never upgrade mid-collection)

- llama.cpp: release b10434, commit 7e4c0a968 (macOS arm64 binary in
  bin/llama-b10434; pod uses the same tag, CUDA build)
- bfcl-eval: 2026.3.23 (PyPI, pinned in uv.lock); ships BFCL v4 datasets
- Python: 3.11 (uv-managed), numpy 1.26.4 (bfcl-eval requirement)
- transformers: 5.15.0 (tokenizer only)
- Dry-run model: Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf
  (official Qwen repo, sha recorded in models/.cache/huggingface)
- Server settings (all runs): --parallel 1, -c 16384, -ngl 99, --seed 42,
  KV dtype f16 both arms; cache arm via per-request cache_prompt
- Sampling (all runs): temperature 0.0, seed 42, logprobs 1, max_tokens
  min(4096, ctx - prompt - 2)
- vLLM (pod lane): version pinned at pod setup time, recorded here before
  any measured run

## Plan addenda

- 2026-08-14 A1: The frozen plan named "BFCL v3 multi-turn". The pinned
  bfcl-eval package ships BFCL v4 (successor with identical multi-turn
  structure: 200 episodes/category, same checker). The study uses v4;
  category BFCL_v4_multi_turn_base. Reason: v4 is the maintained current
  version; v3 data is not shipped in the pinned package.
- 2026-08-14 A2: Episode selection rule (frozen before any results seen):
  the first N episodes of multi_turn_base in ascending numeric id order;
  N=80 for pod runs, N=5 for dry runs, N=2 for smoke tests.

## Dry-run findings (2026-08-14, Mac Metal, Qwen2.5-3B q4_k_m)

- Within-arm determinism: 5/5 episodes bit-identical across independent
  repeat runs. Cross-arm: 0/7 episodes diverged (token-id level).
  llama.cpp Metal batch-1 is cache-stable on this hardware; the divergence
  question now rests on the CUDA pod lanes (where the public bug reports
  live).
- Manipulation check: usage.prompt_tokens_details.cached_tokens confirms
  arms server-side (on-arm ~7.4k-9.1k cached tokens/request after cold
  start; off-arm uniformly 0). Cache exposure is a MEASURED per-request
  variable, not an assumed arm property.
- Cost model from measured token counts (mean 108,643 prefill tokens per
  episode, 14.8 requests/episode): full frozen grid (8 llama.cpp + 6 vLLM
  config-pairs x 80 episodes) ~= 12.5 GPU-h; with GSM8K bridge + setup
  ~= 16 GPU-h ~= $6-11 at community 4090 rates. 100% within-arm repeats
  (upgrade from 20%) adds ~12 h; SGLang third lane adds ~3 h; ALL of it
  fits inside the $30 balance. No top-up required.
- 2026-08-14 A3 (Aditi approved): budget-conditional upgrades activated.
  Within-arm repeats at 100% of episodes (was 20%); SGLang added as third
  backend lane (2 models x FP16/AWQ-INT4 x 2 cache states, RadixAttention
  on/off via --disable-radix-cache). Grid: 8 llama.cpp + 6 vLLM + 4-6
  SGLang config-pairs. Everything stays inside the $30 pod balance.
- GSM8K bridge validation (2026-08-14, 20 items, 4-pass protocol, Mac
  Metal, Qwen2.5-3B q4_k_m): cold and warm paths each 20/20 internally
  deterministic; 1/20 items diverges cold-vs-warm and is therefore cleanly
  cache-attributable; the divergence is a correctness flip (cold: no
  parseable answer, warm: correct). Telemetry pure (cold cached_tokens=0,
  warm=full). First confirmed instance of the paper's phenomenon, and on
  Metal, so it is not CUDA-specific.

## Pod stack (2026-08-15, recorded before any measured vLLM/SGLang run)

- Pod: RunPod on-demand RTX 4090 24GB, Ubuntu 24.04 UV template, CUDA
  driver 570.211.01, nvcc 12.6 (minimal component install)
- llama.cpp: built from source at b10434 / 7e4c0a9, GGML_CUDA, sm_89
- vLLM / SGLang: versions in pod setup.log (pip show output), mirrored in
  the results manifest synced to /mnt/persistent
- INCIDENT 2026-08-15: pod idled ~13h overnight (gate script bug: grep
  exit code under set -e suppressed the completion marker; watcher never
  fired). ~$10 of $30 burned idle. Fixes: grid script guards cells without
  set -e, syncs to /mnt/persistent after every config, self-stops the pod
  via runpodctl on exit/budget-cap (21h hard cap), and the local watcher
  treats pod-unreachable as terminal. Revised projection: grid $14-18 of
  ~$19 remaining.
